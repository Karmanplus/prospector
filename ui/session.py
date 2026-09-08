"""Where each project was left: the runs it had in force and the knobs they were flown at.

A ``Study`` is what reproduces an analysis: one mission, one vehicle, the screening terms. Which
solve happened to be on screen is no part of that, so none of this belongs in ``configs/studies/``.
It sits next to the results it points at, one file per project under ``runs/sessions/``, and it is
disposable: deleting it costs a re-run, never a result.

The file only ever points at results; it never holds one. Everything it names is checked against
the live config on the way back in. A cruise run comes back only if it still matches, a spiral only
if its fingerprint does, a flight-time trade only if it does; anything else is dropped. So a
restored window can only ever show a run the current configuration would produce again: edit the
vehicle between sessions and the windows come back empty, as they would clear mid-session.

The settings are restored before any of that, because several of them feed those checks. The
departure speed is part of what identifies a cruise run, and the spiral's duty cycle, time limit,
steering and radiation scenario are all part of what identifies an escape. Coming back with a
default departure speed would make every run on disk look out of date for the wrong reason.
"""
from __future__ import annotations

import json

from prospector import jobs, paths
from prospector.figures import products
from ui.state import S

# What "where I was" consists of, declared rather than open-coded so a snapshot and a restore
# cannot drift apart. The knobs are plain user settings, safe to restore unconditionally. The run
# ids and views are claims about results and are each validated before they take effect.
_KNOBS = (
    "departure_vinf",
    "launch_duty", "launch_years", "launch_steer", "launch_target_inc",
    "launch_radiation_model", "launch_coverglass_um", "launch_coverglass_density",
    "solve_duty_pct", "solve_slack_days", "solve_max_tof_days", "solve_nseg",
    "grid_colour", "grid_n_dep", "grid_n_tof", "grid_fast",
    "solve_starts",
    "return_duty_pct", "return_slack_days", "return_nseg",
    "flight_tab",
)
_RUNS = ("launch_run_id", "solve_run_id", "grid_run_id")
_VIEWS = ("focus", "grid_sel")

_last_written: dict | None = None


def _sessions_dir():
    """The session-pointer directory. Derived from ``paths.RUNS_DIR`` per call, never captured:
    the test suite redirects the run root, and a path captured at import would outlive that."""
    return paths.RUNS_DIR / "sessions"


def _path(project: str):
    from ui.state import slug
    return _sessions_dir() / f"{slug(project)}.json"


def snapshot() -> dict | None:
    """The restorable part of the live state, or None with no project open."""
    if S.project is None:
        return None
    snap = {k: getattr(S, k) for k in (*_KNOBS, *_RUNS, *_VIEWS)}
    snap["grid_sel"] = list(snap["grid_sel"]) if snap["grid_sel"] else None
    # The focus target is a population row, so its values may still be numpy scalars.
    snap["focus"] = jobs.jsonable_row(snap["focus"]) if snap["focus"] else None
    return snap


def _plain(value):
    """Unwrap anything left that ``json`` cannot take, such as a numpy value from a row or a
    pandas timestamp. Raises on a genuinely foreign type rather than coercing it to a string: a
    pointer that stores ``"<Figure ...>"`` restores something that was never state."""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"cannot persist {type(value).__name__} in a session pointer")


def remember() -> None:
    """Persist where this project is, if it has moved since the last write.

    Called from a page timer, so the change check is what keeps it from rewriting the file every
    tick. Failure is silent by design: this is a convenience pointer, and a read-only or full disk
    must cost a re-run next time rather than interrupt the session.
    """
    global _last_written
    snap = snapshot()
    if snap is None or snap == _last_written:
        return
    try:
        text = json.dumps(snap, indent=2, default=_plain)
    except (TypeError, ValueError):
        # Something the snapshot has no business holding landed in state. Remember the attempt so
        # the timer does not retry it every tick; the cost is one project's pointer.
        _last_written = snap
        return
    try:
        path = _path(S.project)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    except OSError:
        return
    _last_written = snap


def forget() -> None:
    """Drop the in-memory record of what was last written, so the next :func:`remember` writes.

    Called when a project is opened or closed: the previous project's snapshot must not be compared
    against the new one's state, or the first write for the new project is skipped.
    """
    global _last_written
    _last_written = None


def restore() -> list[str]:
    """Adopt the runs and knobs this project was last left with. Returns what came back, named
    for the user, so restoring is announced rather than silent.

    Call after the project's parts are loaded and the previous results cleared: this reads
    ``S.mission`` / ``S.vehicle`` through :func:`ui.state.resolved` to decide what still holds.
    """
    forget()
    if S.project is None:
        return []
    try:
        stored = json.loads(_path(S.project).read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(stored, dict):
        return []

    # The knobs first: they feed the checks below. Anything absent keeps its default, and a value
    # of the wrong type is skipped rather than trusted. This file is disposable, so a hand-edited
    # or truncated one must degrade to "nothing restored".
    for key in _KNOBS:
        if key in stored and _type_ok(key, stored[key]):
            setattr(S, key, stored[key])

    restored: list[str] = []
    focus = stored.get("focus")
    if isinstance(focus, dict) and focus.get("pdes"):
        S.focus = focus

    if _restore_spiral(stored.get("launch_run_id")):
        restored.append("Earth escape")
    # Resolved AFTER the spiral is settled: an adopted spiral refines the escape charge, which
    # moves the cruise-start mass and the departure window, which are both in the cruise signature.
    # Checking the cruise against an unrefined config would reject the very run that spiral was
    # flown for.
    if _restore_cruise(stored.get("solve_run_id")):
        restored.append("Trajectory")
    if _restore_grid(stored.get("grid_run_id")):
        restored.append("transfer grid")
    if _restore_sel(stored.get("grid_sel")):
        restored.append("picked cell")
    return restored


def _type_ok(key: str, value) -> bool:
    """Whether a stored knob value is the shape the field expects. ``None`` is allowed only for
    the two knobs that are genuinely optional."""
    if value is None:
        return key in ("launch_target_inc", "solve_max_tof_days")
    if key in ("flight_tab", "grid_colour"):
        return isinstance(value, str)
    if key == "grid_fast":
        return isinstance(value, bool)
    if key in ("launch_steer",):
        return isinstance(value, bool)
    if key == "launch_radiation_model":
        return isinstance(value, str)
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _resolved():
    from ui import state
    try:
        return state.resolved()
    except Exception:  # noqa: BLE001  (a config that will not resolve restores nothing)
        return None


def _restore_spiral(run_id) -> bool:
    """Adopt a recorded escape spiral if it still applies to this config.

    The validation is :func:`ui.state.session_spiral_run`'s, unchanged: finished, escaped, and
    matching the live escape fingerprint, so a restored spiral has cleared the same bar one flown a
    minute ago clears. The only difference is which session pointed at it.
    """
    from ui import state
    if not isinstance(run_id, str) or not run_id:
        return False
    rc = _resolved()
    if rc is None:
        return False
    S.launch_run_id = run_id
    if state.session_spiral_run(rc) is None:
        S.launch_run_id = None
        return False
    return True


def _restore_cruise(run_id) -> bool:
    """Adopt a recorded cruise solve if it converged, still matches the config, and was flown
    at the target now in focus.

    Two guards beyond the mid-session one. ``unknown_ok=False``: a run whose stored config cannot
    be read is no evidence of a match, and adopting on silence would put an unrelated trajectory on
    screen. And the target check, because the trajectory signature covers the config only and does
    not name a body, so a run flown to one asteroid matches a config now aimed at another.
    """
    from ui.workspaces import flight
    if not isinstance(run_id, str) or not run_id:
        return False
    rc = _resolved()
    if rc is None or not jobs.is_terminal(jobs.read_status(run_id)):
        return False
    if not products.cruise_run_matches(run_id, rc, flight._available_cruise_power_W(rc),
                                       unknown_ok=False):
        return False
    if not _same_target(run_id):
        return False
    res = jobs.read_result(run_id)
    if not res or not products.converged(res["sf"]["feasible"], res["sf"]["mismatch"]):
        return False
    S.solve_run_id = run_id
    return True


def _same_target(run_id) -> bool:
    """Whether a run was flown to the body now in focus. No focus, or a run that never recorded
    its target, means the question cannot be answered, so the run is not adopted."""
    focus = S.focus or {}
    try:
        stored = (jobs.read_job(run_id).get("target") or {}).get("pdes")
    except (OSError, ValueError, KeyError):
        return False
    return bool(stored) and bool(focus.get("pdes")) and str(stored) == str(focus["pdes"])


def _restore_grid(run_id) -> bool:
    """Adopt a recorded transfer grid if it still describes this config and this target.

    Same discipline as a cruise run, and for the same reason: the grid's cells are trajectories for
    one vehicle and one body, so a grid from a different config is a picture of a mission that no
    longer exists. The check is strictly the trajectory signature: an unreadable job spec is no
    evidence of a match, and a grid costs enough that adopting one on silence would be worse than
    re-running it.
    """
    from ui.workspaces import flight
    if not isinstance(run_id, str) or not run_id:
        return False
    rc = _resolved()
    if rc is None or not jobs.is_terminal(jobs.read_grid_status(run_id)):
        return False
    try:
        job = jobs.read_grid_job(run_id)
        cfg = job.get("config")
        if not cfg:
            return False
        from prospector.config import ResolvedConfig
        stored_rc = ResolvedConfig.model_validate(cfg)
    except (OSError, ValueError, KeyError):
        return False
    power = flight._available_cruise_power_W(rc)
    if products.trajectory_signature(stored_rc, power) != products.trajectory_signature(rc, power):
        return False
    focus = S.focus or {}
    target = (job.get("target") or {}).get("pdes")
    if not target or str(target) != str(focus.get("pdes") or ""):
        return False
    if jobs.read_grid_result(run_id) is None:
        return False
    S.grid_run_id = run_id
    return True


def _restore_sel(sel) -> bool:
    """Bring back the picked grid cell. This only means anything once the grid came back, since
    the cell
    is a coordinate into that surface, so without it there is nothing for it to point at."""
    if S.grid_run_id is None or not isinstance(sel, (list, tuple)) or len(sel) != 2:
        return False
    try:
        S.grid_sel = tuple(float(v) for v in sel)
    except (TypeError, ValueError):
        return False
    return True
