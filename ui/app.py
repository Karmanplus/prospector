"""Prospector: the NiceGUI app and its entry point. Run with: pixi run app

The window fills the screen and everything happens inside it. All the physics is in the
``prospector`` package; this layer only connects the layout to it. The four workspaces -- Project,
Find targets, Compare vehicles and Plan trajectory, live under ``ui/workspaces/``.

On Windows and macOS the app opens as a native desktop window (pywebview over the OS webview, Edge
WebView2 / Cocoa WebKit). On Linux it serves a browser tab instead: the only conda-forge web-view
engine is Qt/QtWebEngine, whose build crashes on launch, so no Linux backend is installed.
``PROSPECTOR_UI=web`` forces a browser tab anywhere; ``PROSPECTOR_UI=native`` forces the window
(unsupported on Linux, but honored for setups with a working backend installed).
"""
import os
import sys
from pathlib import Path

# Running `python ui/app.py` puts only ui/ on sys.path; add the repo root so the `ui` package and
# the pure `prospector` library import the same way as under pytest.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _preflight_config_library() -> None:
    """Check the config library is usable before anything imports state, and say so plainly if not.

    The working models are built from the library at import time, so an empty or missing library
    otherwise surfaces as a traceback out of a dataclass default, which tells a fresh installation
    nothing about what to fix. This turns it into the one command that fixes it.

    A fresh clone has no ``configs/`` at all, since the library is operator data and untracked, and
    is created by copying the example one. That is the overwhelmingly likely reason to be here, so
    it is the message. The environment variable is mentioned second, for an installation that keeps
    its library somewhere else.
    """
    from prospector import paths  # noqa: PLC0415  (has to run before the state import below)
    from prospector.spacecraft.propulsion import default_engine_dir

    engines = default_engine_dir()
    if engines.is_dir() and any(engines.glob("*.yaml")):
        return

    bundled = paths.config_dir() == paths.BUNDLED_CONFIG_DIR
    if bundled and not paths.BUNDLED_CONFIG_DIR.exists():
        detail = (f"There is no config library yet. Create one from the example set:\n\n"
                  f"    cp -r {paths.EXAMPLE_CONFIG_DIR.relative_to(paths.REPO_ROOT)} "
                  f"{paths.BUNDLED_CONFIG_DIR.relative_to(paths.REPO_ROOT)}\n\n"
                  f"  Then edit it. The copy is yours, and git does not track it.")
    else:
        where = "the config library" if bundled else f"${paths.CONFIG_DIR_ENV}"
        detail = (f"No engines found in {engines} ({where}).\n"
                  f"  The tool needs at least one engine definition to open: add one there, or "
                  f"point ${paths.CONFIG_DIR_ENV} at a library that has engines/*.yaml.")
    print(f"prospector: {detail}", file=sys.stderr)
    raise SystemExit(2)


_preflight_config_library()

# The solver stack (PyKEP, pygmo, NiceGUI) takes a while to import and the first-ever launch also
# downloads the small-body catalogue, so say something before the silence, or a first run looks
# like a hang.
print("prospector: starting - loading the solver stack, this takes a moment...",
      file=sys.stderr, flush=True)

from nicegui import ui  # noqa: E402

from ui import session, theme, topbar  # noqa: E402
from ui.theme import BORDER, PANEL, PICKAXE  # noqa: E402
from ui.workspaces import flight, target, vehicle  # noqa: E402


@ui.page("/")
def index() -> None:
    theme.page_styling()
    with ui.column().style("width:100vw;height:100vh;gap:0;overflow:hidden"):
        with ui.row().style(
                f"flex:0 0 52px;height:52px;width:100%;align-items:center;padding:0 14px;"
                f"background:{PANEL};border-bottom:1px solid {BORDER};box-sizing:border-box"):
            topbar.header()
        with ui.element("div").style(
                "flex:1 1 0;min-height:0;width:100%;overflow:hidden;position:relative"):
            topbar.main_body()
    # Stream background-characterization progress (no-op unless a job is in flight).
    ui.timer(2.0, target.poll_enrich)
    # Stream Plan-trajectory job progress (escape spiral / cruise); no-op off that workspace or
    # with no job in flight.
    ui.timer(1.0, flight.poll)
    # Stream the Compare-vehicles sweep progress; no-op off that workspace or with no running
    # session selected.
    ui.timer(5.0, vehicle.poll)
    # Animate the mission timeline while it is playing (no-op otherwise).
    ui.timer(0.1, flight.tick)
    # Record where the open project is, so reopening it comes back to the same runs instead of
    # re-flying work already on disk. Writes only when the state has moved.
    ui.timer(3.0, session.remember)


# Choose native window vs browser tab. Windows/macOS default to the native window (reliable OS
# webview); Linux defaults to a browser tab, having no working web-view backend (see module
# docstring). PROSPECTOR_UI overrides either way. The launch guard admits both `__main__` (direct
# launch) and `__mp_main__` (the child process NiceGUI spawns for the native window) so native mode
# is multiprocessing-safe on Windows/macOS spawn.
def _native_available() -> bool:
    pref = os.environ.get("PROSPECTOR_UI", "").lower()
    if pref == "web":
        return False
    try:
        import webview  # noqa: F401
    except ModuleNotFoundError:
        return False
    if pref == "native":
        return True
    return sys.platform in ("win32", "darwin")


# Listen on loopback only. NiceGUI's own default is 0.0.0.0 in browser mode, which would put an
# unauthenticated app that writes the config library and spawns solver processes on the LAN.
# PROSPECTOR_HOST=0.0.0.0 opts in for someone who wants that deliberately.
def _host() -> str:
    return os.environ.get("PROSPECTOR_HOST", "").strip() or "127.0.0.1"


if __name__ in {"__main__", "__mp_main__"}:
    _native = _native_available()
    # Announce the mode once (only in the real launch process, not NiceGUI's spawned child) so a
    # native window that fails to appear never looks like a silent no-op, since the hint is there.
    if __name__ == "__main__":
        if _native:
            print("Prospector: opening a native desktop window "
                  "(if nothing appears, run `PROSPECTOR_UI=web pixi run app` for a browser tab).",
                  flush=True)
        else:
            print(f"Prospector: serving at http://{_host()}:8089 (browser mode).", flush=True)
    ui.run(
        title="Prospector",
        favicon=PICKAXE,           # the app's pickaxe mark as the browser-tab icon
        host=_host(),
        port=8089,
        dark=True,
        reload=False,
        native=_native,
        show=not _native,          # auto-open a browser tab only in browser mode
        window_size=(1600, 1000) if _native else None,
    )
