"""The top bar, and choosing which workspace fills the rest of the window.

Kept out of the entry point (``app.py``) so that ``header`` and ``main_body`` are single shared
objects. A workspace such as Find targets can then call ``topbar.header.refresh()`` to update the
focus chip without importing the entry module, and so without re-running it.
"""
from __future__ import annotations

from nicegui import ui
from pydantic import ValidationError

from ui import report_export, settings, state
from ui.components import vehicle_build_mass_rows, vehicle_drive_rows
from ui.state import S
from ui.theme import ACCENT, AMBER, BORDER, MUTED, PANEL, PANEL2, PICKAXE, RED, TEXT
from ui.workspaces import flight, project, target, vehicle

_WORKSPACES = [("project", "Project", "tune"), ("target", "Find targets", "public"),
               ("vehicle", "Compare vehicles", "balance"), ("flight", "Plan trajectory", "rocket")]


@ui.refreshable
def header() -> None:
    with ui.row().classes("w-full items-center no-wrap").style("gap:0"):
        # LEFT: brand + project dropdown
        with ui.row().classes("items-center no-wrap").style("flex:1 1 0;min-width:0;gap:.6rem"):
            ui.label(PICKAXE).style(f"color:{ACCENT};font-size:1.25rem")
            ui.label("PROSPECTOR").style(
                f"color:{TEXT};font-weight:800;letter-spacing:.12em;font-size:.95rem")
            if S.project is not None:
                with ui.button(on_click=None).props("flat dense no-caps").style(f"color:{TEXT}"):
                    ui.icon("folder").style(f"color:{MUTED}").classes("mr-1")
                    ui.label(S.study_name)
                    ui.icon("expand_more").style(f"color:{MUTED}")
                    with ui.menu().style(f"background:{PANEL2};border:1px solid {BORDER}"):
                        for name in state.available_projects():
                            ui.menu_item(name, on_click=lambda name=name: _open(name))
                        ui.separator().style(f"background:{BORDER}")
                        with ui.menu_item(on_click=_new_project_dialog):
                            ui.icon("add").classes("mr-2").style(f"color:{MUTED}")
                            ui.label("New project…")
                        with ui.menu_item(on_click=_close):
                            ui.icon("logout").classes("mr-2").style(f"color:{MUTED}")
                            ui.label("Close project")
            # Application-level model settings (build/cost model + propellant library); always
            # available since they are not tied to a project.
            ui.button("Global settings", icon="calculate",
                      on_click=settings.open_settings).props(
                "flat dense no-caps").style(f"color:{MUTED}").tooltip(
                "Sizing and cost coefficients, propellants, launch types and returns")

        if S.project is None:
            return

        # CENTER: the four workspace tabs
        with ui.row().classes("items-center no-wrap justify-center").style("flex:0 0 auto;gap:.2rem"):
            for key, lbl, ic in _WORKSPACES:
                on = S.workspace == key
                ui.button(lbl, icon=ic, on_click=lambda k=key: _switch(k)).props(
                    "flat dense no-caps").style(
                    f"color:{ACCENT if on else MUTED};"
                    + (f"border-bottom:2px solid {ACCENT};border-radius:0" if on else ""))

        # RIGHT: focus target + active vehicle (each flags a difference from the project
        # default with a "modified" chip), then save / export.
        with ui.row().classes("items-center no-wrap justify-end").style(
                "flex:1 1 0;min-width:0;gap:.6rem"):
            _diff_chip("my_location", _focus_label(), state.target_modified(),
                       _reset_target, _overwrite_target)
            _vehicle_chip()
            _propellant_pill()
            ui.button("Save", icon="save", on_click=_save).props(
                "unelevated dense no-caps").set_enabled(S.dirty)
            ui.button(icon="picture_as_pdf", on_click=report_export.export_pdf).props(
                "flat dense").style(f"color:{MUTED}").tooltip(
                "Export the trajectory dossier as a PDF")


def _diff_chip(icon: str, value: str, modified: bool, reset_fn, overwrite_fn) -> None:
    """A top-bar chip for the focus target / active vehicle. When it differs from the project
    default it turns amber with a warning glyph and offers reset-to-default + make-default."""
    color = AMBER if modified else ACCENT
    with ui.row().classes("items-center no-wrap").style("gap:.2rem;min-width:0"):
        ui.icon(icon).style(f"color:{color}").classes("text-sm")
        ui.label(value).classes("truncate").style(
            f"color:{TEXT if not modified else AMBER};font-size:.82rem;max-width:150px")
        if modified:
            ui.icon("warning").style(f"color:{AMBER}").classes("text-sm")
            ui.button(icon="undo", on_click=reset_fn).props("flat dense round").style(
                f"color:{MUTED}").tooltip("Reset to project default")
            ui.button(icon="save_as", on_click=overwrite_fn).props("flat dense round").style(
                f"color:{AMBER}").tooltip("Make this the project default")


def _vehicle_chip() -> None:
    """The active-vehicle chip: its name, an at-a-glance inline readout (wet mass + engine
    count), the modified-from-default affordances, and a rich hover card of the derived
    performance so the full vehicle stays legible from any workspace (e.g. Plan trajectory).

    A vehicle whose mass budget leaves nothing for payload is a dead design, so it reads red
    here rather than only inside the hover card: the chip is the one piece of vehicle chrome
    visible from every workspace, and a shortfall found after a trajectory is solved has
    already wasted the solve."""
    modified = state.vehicle_modified()
    rc, build = _resolved_build()
    shortfall = _payload_shortfall(build)
    color = RED if shortfall is not None else (AMBER if modified else ACCENT)
    v = S.vehicle
    # The inline readout uses only plain-model fields (no catalog, never raises) so the chip is
    # always populated even when the vehicle doesn't resolve.
    inline = f"{v.wet_mass:.0f} kg · ×{v.engine_count}"
    with ui.row().classes("items-center no-wrap cursor-default").style("gap:.3rem;min-width:0"):
        ui.icon("satellite_alt").style(f"color:{color}").classes("text-sm")
        ui.label(v.name).classes("truncate").style(
            f"color:{color if (shortfall is not None or modified) else TEXT};"
            "font-size:.82rem;max-width:150px")
        ui.label(inline).classes("truncate").style(
            f"color:{MUTED};font-size:.72rem;font-family:monospace")
        if shortfall is not None:
            # The deficit rides inline, not just in the tooltip: "does not build" without a number
            # leaves the user hovering to find out how far off the design is.
            ui.label(f"payload {_signed_kg(-shortfall)}").classes("truncate").style(
                f"color:{RED};font-size:.72rem;font-family:monospace;font-weight:600")
        _vehicle_hover_card(rc, build)
        if shortfall is not None:
            ui.icon("error").style(f"color:{RED}").classes("text-sm").tooltip("Does not build")
        if modified:
            ui.icon("warning").style(f"color:{AMBER}").classes("text-sm")
            ui.button(icon="undo", on_click=_reset_vehicle).props("flat dense round").style(
                f"color:{MUTED}").tooltip("Reset to project default")
            ui.button(icon="save_as", on_click=_overwrite_vehicle).props("flat dense round").style(
                f"color:{AMBER}").tooltip("Make this the project default")


def _resolved_build() -> tuple[object | None, dict | None]:
    """The working config resolved plus its build assessment, or ``(None, None)`` when the config
    does not resolve. Both the chip and its hover card need these, and resolving once keeps a
    single render from pricing the escape twice."""
    try:
        rc = state.resolved()
    except (ValidationError, ValueError):
        return None, None
    return rc, project.assess_build(rc)


def _payload_shortfall(build: dict | None) -> float | None:
    """How many kilograms the bus overruns the dry-mass budget by, or None when the design fits
    (or when the build model could not size it, since a missing assessment is not a failed one)."""
    if not build or build.get("buildable") is not False:
        return None
    capacity = build.get("payload_capacity_kg")
    return None if capacity is None else -float(capacity)


def _signed_kg(val: float) -> str:
    """A mass carrying its sign, sub-kilogram values kept to a decimal so a shortfall of a few
    hundred grams cannot round away to a break-even "0 kg"."""
    return f"{val:+.1f} kg" if 0 < abs(val) < 1 else f"{val:+.0f} kg"


def _vehicle_hover_card(rc, build: dict | None) -> None:
    """A hover tooltip carrying the vehicle's spec, grouped for scanning: a Masses block (the load
    equation and the burnout/usable masses), a Build grid (bus component masses), and a Drive grid
    (thrust, array power, Isp, and the total ΔV the vehicle can deliver). Vehicle information only
    with no trajectory or mission terms such as escape, cruise or spiral. Falls back to the masses
    with a note when the vehicle doesn't resolve, so the chrome never breaks on a mid-edit config.

    ``rc`` is a resolved config or None; ``build`` its :func:`project.assess_build` result."""
    with ui.tooltip().style(
            f"background:{PANEL2};border:1px solid {ACCENT};border-radius:8px;padding:.6rem .75rem;"
            "box-shadow:0 6px 24px rgba(0,0,0,.45);min-width:300px"):
        v = S.vehicle
        ui.label(v.name).style(f"color:{TEXT};font-weight:700;font-size:.82rem")
        engines = " + ".join(f"{count}× {key}" for key, count in v.mounts) or "no engines"
        gas, estimated = state.active_propellant()
        ui.label(f"{engines}  ·  {gas}{' (est)' if estimated else ''}").style(
            f"color:{AMBER if estimated else MUTED};font-size:.72rem")
        if rc is None:
            ui.label(f"Dry {v.dry_mass:.0f} + prop {v.fuel_mass:.0f} = {v.wet_mass:.0f} kg wet").style(
                f"color:{TEXT};font-size:.74rem;margin-top:.3rem")
            ui.label("Config does not fully resolve").style(
                f"color:{AMBER};font-size:.68rem;margin-top:.2rem")
            return

        # Flight masses as dense flow lines: the load equation and the burnout/usable pair -- more
        # legible than stacked cells, and vehicle-only (no escape/cruise mass terms).
        _hover_head("Masses")
        _hover_flow(f"Dry {v.dry_mass:.0f} + prop {v.fuel_mass:.0f} = {v.wet_mass:.0f} kg wet")
        usable = f" · usable {rc.usable_propellant_kg:.0f}" if v.unusable_prop > 0 else ""
        _hover_flow(f"Burnout {v.burnout_mass:.0f} kg{usable}")

        rows = vehicle_build_mass_rows(build)
        if rows:
            _hover_grid("Build", rows)
        if _payload_shortfall(build) is not None:
            # Spell the failure out under the numbers that caused it. ``build_why`` names the bill,
            # meaning the sized items that have to fit before any payload does, and the dry mass
            # the design would need, so the card diagnoses rather than just condemns.
            with ui.row().classes("items-start no-wrap").style("gap:.25rem;margin-top:.3rem"):
                ui.icon("error").style(f"color:{RED}").classes("text-sm")
                ui.label(str(build.get("build_why") or "no payload capacity left")).style(
                    f"color:{RED};font-size:.68rem;max-width:340px;white-space:normal")
        _hover_grid("Drive", vehicle_drive_rows(rc))


def _hover_head(title: str) -> None:
    """A small accented section heading inside the vehicle hover card."""
    ui.label(title).style(
        f"color:{ACCENT};font-size:.62rem;font-weight:700;letter-spacing:.09em;"
        "text-transform:uppercase;margin-top:.5rem;margin-bottom:.1rem")


def _hover_flow(text: str) -> None:
    """A dense single-line mass fact (several values per line) in the hover card."""
    ui.label(text).style(f"color:{TEXT};font-size:.74rem;font-family:monospace")


def _hover_grid(title: str, rows: list) -> None:
    """A section heading over a two-column grid of ``(label, value)`` pairs, two stats per line,
    so a long list reads as a compact block rather than a tall stack."""
    _hover_head(title)
    with ui.element("div").style(
            "display:grid;grid-template-columns:1fr 1fr;column-gap:1.3rem;row-gap:.03rem;width:100%"):
        for label, val in rows:
            with ui.row().classes("items-center no-wrap justify-between w-full").style(
                    "gap:.5rem;min-width:0"):
                ui.label(label).style(f"color:{MUTED};font-size:.7rem;white-space:nowrap")
                # Every quantity in these grids is a physical magnitude, whether a mass, a power or
                # a speed, so a negative one is always an overrun and always reads red.
                color = RED if val.lstrip().startswith("-") else TEXT
                ui.label(val).style(
                    f"color:{color};font-size:.72rem;font-family:monospace;white-space:nowrap")


def _propellant_pill() -> None:
    """A small chip showing the active vehicle's working gas. Amber + 'est' when the gas is
    estimated (scaled from the engine's measured gas), muted when it's the native gas."""
    gas, estimated = state.active_propellant()
    color = AMBER if estimated else MUTED
    label = f"{gas} · est" if estimated else gas
    with ui.row().classes("items-center no-wrap").style("gap:.15rem"):
        ui.icon("local_gas_station").style(f"color:{color};font-size:1rem")
        pill = ui.label(label).style(f"color:{color};font-size:.74rem")
        if estimated:
            pill.tooltip("Thrust and Isp scaled from the engine's measured gas")


def _focus_label() -> str:
    if S.focus is None:
        return "no target"
    return str(S.focus.get("full_name") or S.focus.get("pdes") or "target")


@ui.refreshable
def main_body() -> None:
    if S.project is None:
        _welcome()
        return
    {"project": project.render, "target": target.render,
     "vehicle": vehicle.render, "flight": flight.render}[S.workspace]()


def _welcome() -> None:
    with ui.column().classes("w-full h-full items-center justify-center").style("gap:1.5rem"):
        with ui.row().classes("items-center gap-3"):
            ui.label(PICKAXE).style(f"color:{ACCENT};font-size:2.2rem")
            ui.label("Prospector").style(f"color:{TEXT};font-weight:800;font-size:1.8rem")
        ui.label("asteroid mission design").style(
            f"color:{MUTED};letter-spacing:.18em;text-transform:uppercase;font-size:.72rem")
        projects = state.available_projects()
        # The card is capped to the viewport and only the project list inside it scrolls, so the
        # title and the New-project row stay put however many studies the library holds.
        with ui.card().classes("p-0").style(
                f"background:{PANEL};border:1px solid {BORDER};width:420px;max-height:60vh;"
                f"display:flex;flex-direction:column;overflow:hidden"):
            with ui.row().classes("items-center gap-2 px-4 py-3 w-full").style(
                    f"border-bottom:1px solid {BORDER};flex:0 0 auto"):
                ui.icon("folder_open").style(f"color:{MUTED}")
                ui.label("Open a project").style(f"color:{TEXT};font-weight:600")
            with ui.element("div").classes("w-full").style("flex:1 1 auto;min-height:0;overflow-y:auto"):
                if not projects:
                    ui.label("No studies found in configs/studies/").classes("px-4 py-3").style(
                        f"color:{MUTED};font-size:.82rem")
                for name in projects:
                    with ui.row().classes(
                            "items-center no-wrap w-full px-4 py-3 cursor-pointer hover-row").on(
                            "click", lambda name=name: _open(name)):
                        ui.icon("folder").style(f"color:{ACCENT}")
                        ui.label(name).style(
                            f"color:{TEXT};font-weight:600;font-size:.9rem;margin-left:.4rem")
            with ui.row().classes("items-center no-wrap w-full px-4 py-3 cursor-pointer hover-row").style(
                    f"border-top:1px solid {BORDER};flex:0 0 auto").on("click", _new_project_dialog):
                ui.icon("add").style(f"color:{MUTED}")
                ui.label("New project…").style(
                    f"color:{MUTED};font-weight:600;font-size:.9rem;margin-left:.4rem")


# ---- behaviour ----

def _open(name: str) -> None:
    """Open a project, after settling any unsaved edits in the one that is open."""
    _guard_unsaved(lambda: _open_now(name))


def _open_now(name: str) -> None:
    restored = state.open_project(name)
    vehicle.reset()         # drop the previous project's session selection + sweep form state
    header.refresh()
    main_body.refresh()
    # Announced, not silent: a window that fills itself in should say where its contents came from,
    # or a result carried over from a previous session reads as one computed just now.
    if restored:
        ui.notify(f"Restored from your last session: {', '.join(restored)}", type="info")


def _close() -> None:
    _guard_unsaved(_close_now)


def _close_now() -> None:
    state.close_project()
    header.refresh()
    main_body.refresh()


def _guard_unsaved(then) -> None:
    """Run ``then`` at once when nothing is unsaved; otherwise ask first.

    Leaving a project used to drop its unsaved edits without a word, which read as the launch and
    arrival dates "resetting" when switching between projects. Save writes the open study (and
    stays put if that fails), Discard proceeds without writing, Cancel keeps the project open.
    """
    if S.project is None or not S.dirty:
        then()
        return
    with ui.dialog() as dlg, ui.card().style(
            f"background:{PANEL};border:1px solid {BORDER};min-width:360px"):
        ui.label(f"Save changes to '{S.study_name}'?").style(f"color:{TEXT};font-weight:600")
        ui.label("The project has edits that have not been saved.").style(
            f"color:{MUTED};font-size:.8rem")

        def _save_then() -> None:
            if _save():
                dlg.close()
                then()

        def _discard() -> None:
            dlg.close()
            then()

        with ui.row().classes("justify-end w-full gap-2 mt-2"):
            ui.button("Cancel", on_click=dlg.close).props("flat no-caps").style(f"color:{MUTED}")
            ui.button("Discard", on_click=_discard).props("flat no-caps").style(f"color:{AMBER}")
            ui.button("Save", icon="save", on_click=_save_then).props("unelevated no-caps")
    dlg.open()


def _new_project_dialog() -> None:
    """Name a new project and create it on the spot: the study is written to disk at once and
    opened in the Project workspace, so the top-bar Save simply overwrites it from then on. It can
    start blank or as a copy of the open project's mission, vehicle, screens and default target."""
    def _show() -> None:
        with ui.dialog() as dlg, ui.card().style(
                f"background:{PANEL};border:1px solid {BORDER};min-width:380px"):
            ui.label("New project").style(f"color:{TEXT};font-weight:600")
            name = ui.input("Project name").props("dense outlined autofocus").classes("w-full")
            copy = None
            if S.project is not None:
                copy = ui.switch(f"Start from a copy of '{S.study_name}'", value=True).props("dense")
                ui.label("Copies its mission, vehicle, screens and default target; the copy owns "
                         "its mission, so its dates are its own.").style(
                    f"color:{MUTED};font-size:.74rem")
            err = ui.label("").style(f"color:{ACCENT};font-size:.75rem")
            err.set_visibility(False)

            def _create() -> None:
                try:
                    state.new_project(name.value or "", bool(copy and copy.value))
                except Exception as exc:  # noqa: BLE001  (an unusable name or a bad config)
                    err.text = str(exc).splitlines()[0]
                    err.set_visibility(True)
                    return
                vehicle.reset()
                dlg.close()
                ui.notify(f"Created project '{S.study_name}'", type="positive")
                header.refresh()
                main_body.refresh()

            name.on("keydown.enter", _create)
            with ui.row().classes("justify-end w-full gap-2 mt-2"):
                ui.button("Cancel", on_click=dlg.close).props("flat no-caps").style(f"color:{MUTED}")
                ui.button("Create", icon="add", on_click=_create).props("unelevated no-caps")
        dlg.open()

    _guard_unsaved(_show)


def _switch(key: str) -> None:
    S.workspace = key
    S.selected = None
    header.refresh()
    main_body.refresh()


def _save() -> bool:
    """Overwrite the open study (and its mission + vehicle) in place. Save-as a new study
    lives in the Project workspace header. Returns whether the save went through."""
    project.flush()                       # capture any field edited but not yet committed
    if S.project is None:
        project._save_as_dialog()
        return False
    try:
        notes = state.persist_study(S.study_name, S.project)
    except Exception as exc:  # noqa: BLE001  (report a bad config rather than crashing)
        ui.notify(f"Save failed: {str(exc).splitlines()[0]}", type="negative")
        return False
    ui.notify(f"Saved study '{S.study_name}'", type="positive")
    report_forks(notes)
    header.refresh()
    return True


def report_forks(notes: list[str]) -> None:
    """Say when a save split a shared mission or vehicle off into this study's own copy. It is a
    change to what the project points at, so it is announced rather than left to be found in the
    library later."""
    for note in notes:
        ui.notify(note[0].upper() + note[1:], type="info", timeout=8000)


def _reset_target() -> None:
    state.reset_target()
    ui.notify("Target reset to the project default")
    header.refresh()
    main_body.refresh()


def _overwrite_target() -> None:
    state.overwrite_target()
    ui.notify("Project default target updated")
    header.refresh()


def _reset_vehicle() -> None:
    state.reset_vehicle()
    ui.notify("Vehicle reset to the project default")
    header.refresh()
    main_body.refresh()


def _overwrite_vehicle() -> None:
    """Confirm via the vehicle-summary dialog (computed masses, power, array sizing) before
    adopting the working vehicle as the project default."""
    def _commit() -> None:
        state.overwrite_vehicle()
        ui.notify("Project default vehicle updated")
        header.refresh()
    project.vehicle_summary_dialog(_commit)
