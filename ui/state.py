"""Application state for the NiceGUI app: the open study and the Find-targets working set.

Holds one :class:`AppState` for the life of the page. The app is a window that stays open rather
than a script that re-runs, so the models being worked on simply live in memory and the views read
them.

A project in the UI is a :class:`~prospector.config.Study`. Opening one loads its mission, its
vehicle and both screens: ``Screening`` for what is reachable, ``Desirability`` for what is worth
reaching. The delta-v budget the screen uses is always worked out from those parts by
:meth:`resolved`, never stored.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from prospector import jobs, population
from prospector import launch as launch_lib
from prospector.config import (
    Desirability,
    Mission,
    ResolvedConfig,
    Screening,
    Study,
    Vehicle,
    default_mission,
    default_vehicle,
    list_studies,
    load_mission,
    load_study,
    load_vehicle,
    save_mission,
    save_study,
    save_vehicle,
    studies_referencing,
)
from prospector.spacecraft import buildability
from prospector.spacecraft.propulsion import load_engines
from prospector.trades import sweep as sweep_lib


@dataclass
class AppState:
    """The single in-memory app state (one open project at a time)."""

    # ---- the open project (study) + its working parts ----
    project: str | None = None          # the open study's library key (None => welcome screen)
    study_name: str = "Untitled study"
    mission: Mission = field(default_factory=default_mission)
    vehicle: Vehicle = field(default_factory=default_vehicle)
    screening: Screening = field(default_factory=Screening)
    desirability: Desirability = field(default_factory=Desirability)
    # Library keys the mission/vehicle came from, so a study save can overwrite the parts it
    # references (None => never saved => save under a name slugged from the part's own name).
    mission_key: str | None = None
    vehicle_key: str | None = None
    dirty: bool = False

    # The project's saved defaults (baselines), captured on open and on save. The working
    # ``vehicle`` and ``focus`` may differ from these, as when a vehicle is traded in Compare or a
    # target is temporarily set in Find targets, and the top bar flags the difference with a
    # "modified" chip (with reset-to-default + overwrite-default actions).
    project_vehicle: Vehicle | None = None
    project_target: dict | None = None

    # ---- navigation ----
    workspace: str = "target"              # project | target | vehicle | flight

    # ---- the focus target (set in Find targets, read by the other workspaces) ----
    focus: dict | None = None

    # ---- Plan-trajectory workspace state -------------------------------------------------
    # The launch phase is controlled by one departure-speed knob; the rest are spiral knobs.
    departure_vinf: float = 0.0            # planned departure hyperbolic excess (km/s)
    launch_duty: float = 90.0             # spiral thruster duty cycle (%)
    launch_years: float = 8.0             # spiral run-time cap (years)
    launch_steer: bool = False            # steer the orbit plane during the spiral?
    launch_target_inc: float | None = None   # steer target inclination (deg) when steering
    # Solar-array radiation scenario for the escape spiral (configs/radiation/ key) + a live
    # coverglass-thickness override (um). These select the belt-degradation model and its dominant
    # shielding knob; both join the escape fingerprint so a change re-runs the spiral.
    launch_radiation_model: str = "ap8min-worstcase"
    launch_coverglass_um: float = 212.5
    launch_coverglass_density: float = 1.640

    flight_tab: str = "Earth escape"      # the active Plan-trajectory sub-tab
    porkchop_min: bool = False            # is the porkchop inset minimized?
    timeline_t: float = 0.0               # mission-timeline scrubber position [0, 1]
    timeline_playing: bool = False        # is the mission-timeline animating?

    # Detached-job handles for the three launch/cruise channels (run dirs under runs/).
    launch_run_id: str | None = None   # escape-spiral run
    solve_run_id: str | None = None    # Sims-Flanagan cruise run
    # The converged transfer grid: a detached job whose every cell is a real trajectory. This is
    # the surface the user reads and picks from. The impulsive porkchop it replaced was instant but
    # anti-correlated with converged cost, and its rejection verdict fired on cells that fly.
    grid_run_id: str | None = None      # the transfer-grid run in force
    grid_sel: tuple | None = None       # the picked (dep_mjd2000, tof_days) cell
    # ΔV by default: it is the quantity the budget is in, so it is the one that can be read
    # against capability without arithmetic. Delivered mass is the same information as a payoff.
    # Propellant margin by default: kilograms left in the tank after the cruise is the quantity
    # that decides whether a cell closes, now that the legs fly at different Isps. ΔV and delivered
    # mass remain on the menu.
    grid_colour: str = "margin"         # "margin" (kg left) | "dv" (km/s) | "mass" (delivered kg)
    grid_n_dep: int = 16                # grid density; the solve is serial in flight time, so the
    grid_n_tof: int = 16                # departure axis is the cheap one to widen
    grid_fast: bool = False             # one restart: which cells fly, in seconds, with no ΔV
    # Cruise solver knobs (live values the rail edits; forwarded to the worker on Run).
    # Per-segment on-time cap for the cruise solve. 100 = unrestricted, which is the
    # default because restricting it rejects trajectories that close while enforcing a
    # limit the converged answer is nowhere near (real studies run 13-27% duty).
    solve_duty_pct: float = 100.0
    solve_slack_days: float = 0.0
    solve_max_tof_days: float | None = None
    solve_nseg: int = 12
    # Seed-diverse parallel multi-start: how many trajectory-family seeds (departure epoch x
    # flight time) to optimize in parallel and keep the best of. 1 solves the clicked cell alone;
    # > 1 lets the solve escape a short-transfer local optimum and find the true optimum the
    # arrive_by deadline permits. Fanned across cores by the detached worker.
    solve_starts: int = 9
    # Return-leg solver knobs (independent of the outbound's), used only when the mission flies a
    # round trip. The destination + deadline are mission definition (on Mission); these are tuning,
    # forwarded under options["return_options"] on Run.
    return_duty_pct: float = 100.0
    return_slack_days: float = 0.0
    return_nseg: int = 20

    # ---- Find-targets data (the live reachability screen + its characterization) ----
    population: pd.DataFrame | None = None      # raw SBDB slice, cached by h_max
    population_h: float | None = None
    # The reachable, described frame before any filter is applied. This is the expensive part
    # (reachability scoring + enrichment merge). Recomputed only when the screening terms or the
    # population change; desirability filters re-select from it in-memory (no disk).
    screen_base: pd.DataFrame | None = None
    screen_df: pd.DataFrame | None = None       # screen_base + the `selected` column (rendered)
    enrich_job: str | None = None               # the dispatched characterization job id
    enrich_reachable: list | None = None        # the pdes set that job was dispatched for
    enrich_loaded: bool = False                    # its finished results merged into screen_df yet?
    target_view: str = "Reachability"              # canvas view selector
    # Show the bodies over the ΔV budget in the targets table too (with their shortfall), so an
    # out-of-reach body such as Venus or a main-belt asteroid can still be set as the focus.
    target_show_all: bool = False
    sel_target: str | None = None               # highlighted table row (a pdes value)
    selected: dict | None = None                # the floating HUD payload


S = AppState()


# ---------------------------------------------------------------------------
# project (study) lifecycle
# ---------------------------------------------------------------------------

def available_projects() -> list[str]:
    """Study keys in the library, newest-typical order (alphabetical, as the config layer lists)."""
    return list_studies()


def open_project(name: str) -> list[str]:
    """Load a study and its parts as the working set, clearing the previous screen results.

    Returns the names of the Plan-trajectory windows restored from the project's last session (see
    :mod:`ui.session`), for the caller to report. Empty when there was nothing to bring back or
    nothing still matched.
    """
    study = load_study(name)
    S.project = name
    S.study_name = study.name
    S.mission = load_mission(study.mission)
    # Size the array on the way in, the same as the vehicle editor does on every keystroke, so the
    # top-bar chip and the mass cards read the array this vehicle flies before anything is edited.
    S.vehicle = buildability.with_sized_array(load_vehicle(study.vehicle), load_engines())
    S.mission_key = study.mission
    S.vehicle_key = study.vehicle
    S.screening = study.screening
    S.desirability = study.desirability
    S.dirty = False
    S.workspace = "target"
    _clear_results()
    # Capture the project's saved defaults as baselines, then resolve the default target (best
    # effort, since a missing cache or unknown designation just leaves no focus).
    S.project_vehicle = S.vehicle.model_copy(deep=True)
    S.project_target = resolve_target(study.target) if study.target else None
    S.focus = (dict(S.project_target) if S.project_target else None)
    # Re-adopt the runs and knobs this project was last left with. Imported here rather than at
    # module scope: the session module reads state through this one.
    from ui import session
    return session.restore()


def close_project() -> None:
    from ui import session
    S.project = None
    S.project_vehicle = None
    S.project_target = None
    _clear_results()
    session.forget()      # the next project's first write must not be compared against this one


def resolve_target(query: str | None) -> dict | None:
    """Resolve a designation/name to a full target dict (best effort; None on miss or error)."""
    if not query:
        return None
    try:
        return population.get_target(str(query))
    except Exception:  # noqa: BLE001  (offline or no cache: the small-body catalog is unavailable)
        pass
    # The planets are authored data and need no fetch, so "Venus" resolves even when the asteroid
    # catalog does not; an asteroid designation stays unresolved rather than being guessed at.
    try:
        return population.get_target(str(query), population=population.planet_population())
    except Exception:  # noqa: BLE001
        return None


def _clear_results() -> None:
    """Drop the Find-targets screen/characterization so a new project starts clean."""
    S.population = None
    S.population_h = None
    S.screen_base = None
    S.screen_df = None
    S.enrich_job = None
    S.enrich_reachable = None
    S.enrich_loaded = False
    S.focus = None
    S.selected = None
    S.sel_target = None
    # Plan-trajectory runs and porkchops belong to one study, so drop them and stop an escape or
    # porkchop / cruise from the old study can't leak through into the new one.
    S.departure_vinf = 0.0
    S.launch_run_id = None
    S.solve_run_id = None
    S.grid_run_id = None
    S.grid_sel = None
    S.flight_tab = "Earth escape"
    S.timeline_t = 0.0
    S.timeline_playing = False


def mark_dirty() -> None:
    S.dirty = True


def slug(name: str) -> str:
    """A filename-safe library key from a display name ('High Frontier 2' -> 'high-frontier-2')."""
    s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return s or "untitled"


def persist_study(name: str, key: str) -> list[str]:
    """Save the working study and its parts so the artifact fully resolves and reloads clean.

    The mission and vehicle are written back to their own library keys and the study is saved
    referencing them. The two screening axes ride along inline. Afterwards the working set is clean
    (``dirty`` cleared) and points at the keys it was just saved under.

    A part is written back to the key it was loaded from only when no OTHER study references that
    key, or when it is unchanged from what is on disk. A shared part that was edited is saved under
    this study's own key instead (:func:`owned_part_key`), so editing the launch window of one study
    never moves the window of every study that happened to share its mission file. A part that was
    never loaded from a file (a new project) is owned by the study from the start.

    Returns human-readable notes about any part that was split off, for the caller to report.
    """
    name = name.strip() or "Untitled study"
    notes: list[str] = []
    vkey, forked = owned_part_key("vehicle", S.vehicle_key, key, _vehicle_unchanged)
    if forked:
        notes.append(f"vehicle saved as its own copy '{vkey}' (the original '{S.vehicle_key}' is "
                     f"shared with other studies)")
    mkey, forked = owned_part_key("mission", S.mission_key, key, _mission_unchanged)
    if forked:
        notes.append(f"mission saved as its own copy '{mkey}' (the original '{S.mission_key}' is "
                     f"shared with other studies)")
    save_vehicle(S.vehicle, vkey)
    save_mission(S.mission, mkey)
    save_study(Study(name=name, mission=mkey, vehicle=vkey, target=target_designation(S.focus),
                     screening=S.screening, desirability=S.desirability), key)
    S.vehicle_key, S.mission_key = vkey, mkey
    S.project, S.study_name, S.dirty = key, name, False
    # The just-saved working set is the project default now, so clear the "modified" flags.
    S.project_vehicle = S.vehicle.model_copy(deep=True)
    S.project_target = dict(S.focus) if S.focus else None
    return notes


def owned_part_key(part: str, current_key: str | None, study_key: str,
                   unchanged) -> tuple[str, bool]:
    """The library key a study's mission or vehicle is saved under, and whether that is a fork.

    ``current_key`` is where the part was loaded from (None: never saved). ``unchanged(key)`` says
    whether the working part still matches the file under ``key``. The part keeps its key when it
    is the study's own, when no other study references it, or when it was not edited; otherwise it
    is saved under a key derived from the study's (:func:`free_part_key`), leaving the shared file
    as the other studies expect it.
    """
    if current_key is None:
        return free_part_key(part, study_key), False
    others = [s for s in studies_referencing(part, current_key) if s != study_key]
    if not others or unchanged(current_key):
        return current_key, False
    return free_part_key(part, study_key), True


def free_part_key(part: str, study_key: str) -> str:
    """A key for a study-owned part: the study's own key, or the first ``-2``, ``-3``... variant
    of it that no OTHER study references. The study's own name is what makes the file findable
    next to the study; the suffix only ever appears when a stranger already sits on that name."""
    candidate, n = study_key, 1
    while any(s != study_key for s in studies_referencing(part, candidate)):
        n += 1
        candidate = f"{study_key}-{n}"
    return candidate


def _mission_unchanged(key: str) -> bool:
    try:
        return load_mission(key) == S.mission
    except (OSError, ValueError):
        return False


def _vehicle_unchanged(key: str) -> bool:
    """Compared after the same array sizing the open applies, so an untouched vehicle reads as
    unchanged rather than differing only by the array the app filled in on load."""
    try:
        return buildability.with_sized_array(load_vehicle(key), load_engines()) == S.vehicle
    except (OSError, ValueError):
        return False


def new_project(name: str, copy_from_open: bool = False) -> None:
    """Create a study named ``name``, save it at once, and make it the open project.

    Blank by default: the library's default mission and a default vehicle on the first engine the
    library offers, no target, permissive screens. ``copy_from_open`` starts instead from the open
    project's working mission, vehicle, screens and default target. Either way the new study owns
    its mission from the start (saved under the study's own key), and a copied vehicle keeps its
    library reference, forking on save only if it is edited (:func:`persist_study`).

    Raises :class:`ValueError` on an empty name or one whose key already exists, so the caller can
    show the message rather than overwrite a study.
    """
    name = name.strip()
    if not name:
        raise ValueError("give the project a name")
    key = slug(name)
    if key in list_studies():
        raise ValueError(f"a project saved as '{key}' already exists; pick another name")
    from ui import session
    if copy_from_open and S.project is not None:
        mission = S.mission.model_copy(deep=True)
        vehicle, vehicle_key = S.vehicle.model_copy(deep=True), S.vehicle_key
        screening, desirability = S.screening.model_copy(), S.desirability.model_copy()
        target = dict(S.project_target) if S.project_target else None
    else:
        mission = default_mission()
        vehicle, vehicle_key = buildability.with_sized_array(default_vehicle(), load_engines()), None
        screening, desirability, target = Screening(), Desirability(), None
    S.project, S.study_name = key, name
    S.mission, S.vehicle = mission, vehicle
    S.mission_key, S.vehicle_key = None, vehicle_key
    S.screening, S.desirability = screening, desirability
    _clear_results()
    session.forget()
    S.project_vehicle = None
    S.project_target = target
    S.focus = dict(target) if target else None
    S.workspace = "project"
    persist_study(name, key)


# ---------------------------------------------------------------------------
# working-vs-project "modified" state (the top-bar chips)
# ---------------------------------------------------------------------------

def target_designation(target: dict | None) -> str | None:
    """A stable reference string for a target dict (its designation, falling back to name)."""
    if not target:
        return None
    for key in ("pdes", "full_name", "name"):
        value = target.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def vehicle_modified() -> bool:
    """True when the working vehicle differs from the project's saved default."""
    return S.project_vehicle is not None and S.vehicle != S.project_vehicle


def active_propellant(vehicle=None) -> tuple[str, bool]:
    """The working gas the vehicle flies, and whether its thrust/Isp are estimated.

    The gas in force is the vehicle's loaded gas, or when none is loaded, the head engine's
    authored (native) gas. It is estimated whenever the loaded gas differs from that native gas
    (the performance is then scaled from the native numbers, not measured).
    """
    v = vehicle if vehicle is not None else S.vehicle
    catalog = load_engines()
    head = v.engines[0].type if v.engines else None
    native = getattr(catalog.get(head), "propellant", None) or "xenon"
    gas = v.propellant or native
    return gas, (gas != native)


def target_modified() -> bool:
    """True when the working focus target differs from the project's saved default."""
    return target_designation(S.focus) != target_designation(S.project_target)


def reset_vehicle() -> None:
    """Revert the working vehicle to the project default."""
    if S.project_vehicle is not None:
        S.vehicle = S.project_vehicle.model_copy(deep=True)
        S.vehicle_key = None if S.vehicle_key is None else S.vehicle_key


def overwrite_vehicle() -> None:
    """Make the working vehicle the project default (cleared 'modified'; written on next Save)."""
    S.project_vehicle = S.vehicle.model_copy(deep=True)
    S.dirty = True


def reset_target() -> None:
    """Revert the working focus to the project's default target."""
    S.focus = dict(S.project_target) if S.project_target else None


def overwrite_target() -> None:
    """Make the working focus target the project default (written on next Save)."""
    S.project_target = dict(S.focus) if S.focus else None
    S.dirty = True


def set_project_target(target: dict | None) -> None:
    """Set the project's default target and point the working focus at it (the Project
    target picker: editing the default adopts it as the active focus too)."""
    S.project_target = dict(target) if target else None
    S.focus = dict(target) if target else None
    S.dirty = True


# ---------------------------------------------------------------------------
# the focus target
# ---------------------------------------------------------------------------

def set_focus(target: dict | None) -> None:
    S.focus = target


# ---------------------------------------------------------------------------
# the launch phase: the departure-speed knob + the spiral options it drives
# ---------------------------------------------------------------------------

def departure_vinf() -> float:
    """The single departure-speed knob (km/s).

    One number drives the whole escape/cruise trade: the spiral's target v∞, the free departure
    allowance the cruise solvers (porkchop credit, Sims-Flanagan cap) get, and the departure speed
    every derived escape figure is priced at: the charge, the duration, the propellant, the window
    shift and the budget credit.
    """
    return float(S.departure_vinf)


def spiral_options() -> dict:
    """The current spiral-run options as ``solvers.spiral.solve`` kwargs.

    The injection orbit itself is FIXED by the launch type; these are the terms the operator
    controls between drop-off and escape: how much hyperbolic excess to buy, the duty cycle,
    optional plane steering, and the run-time cap.
    """
    opts = {
        "target_vinf_kms": float(S.departure_vinf),
        "duty_cycle": float(S.launch_duty) / 100.0,
        "max_years": float(S.launch_years),
        "radiation_model": str(S.launch_radiation_model),
        "coverglass_um": float(S.launch_coverglass_um),
        "coverglass_density": float(S.launch_coverglass_density),
    }
    if S.launch_steer and S.launch_target_inc is not None:
        opts["target_inc_deg"] = float(S.launch_target_inc)
    return opts


def spiral_fingerprint(rc) -> dict:
    """The escape fingerprint of the live config under the current spiral options."""
    return launch_lib.escape_fingerprint(rc, spiral_options())


def return_destinations() -> dict:
    """The return-destination catalog (EML2/EML1/SEL2/LDRO/LEO), keyed by library key."""
    return launch_lib.load_return_destinations()


def return_options() -> dict:
    """The return-leg solver knobs as a sub-dict for the solve ``options`` payload.

    Forwarded under ``options["return_options"]`` so they reach the chained return solve
    without touching the outbound's knobs. The return destination, deadline, and stay live on
    the Mission, set in Project or the return rail, not here; these are tuning only."""
    return {"nseg": int(S.return_nseg),
            "max_duty_cycle": float(S.return_duty_pct) / 100.0,
            "window_slack_days": float(S.return_slack_days)}


def session_spiral_run(rc) -> str | None:
    """The escape spiral in force: the run launched this session (``S.launch_run_id``), but
    only while it still applies, meaning finished, escaped, and matching the live config's escape
    fingerprint (the fingerprint excludes the v∞ target, so one run prices every setting of
    the knob; a vehicle/launch-type edit invalidates it). Returns its run id, or None.

    Deliberately not a library scan: a spiral is "in force" only because it is the one this project
    points at, whether run in this session or restored from the project's own record of the last
    one (:mod:`ui.session`), and never because a matching run happens to sit on disk. A scan would
    adopt a stranger's run; a restore adopts the project's own, and says so.
    """
    run_id = S.launch_run_id
    if not run_id or rc.launch.escape_provided:
        return None
    if not jobs.is_terminal(jobs.read_spiral_status(run_id)):
        return None
    summary = jobs.read_spiral_summary(run_id)
    if summary.get("status") != "escaped" or summary.get("dv_at_escape_kms") is None:
        return None
    try:
        if jobs.read_spiral_job(run_id).get("fingerprint") != spiral_fingerprint(rc):
            return None
    except (OSError, KeyError, ValueError):
        return None
    return run_id


def departure_declination_cap(rc) -> float:
    """The cone launch dictates to the cruise solvers: the departure asymptote's max
    |equatorial declination| (degrees), from the drop-off plane, or the steer target when
    spiral plane steering is on."""
    steer = float(S.launch_target_inc) if (S.launch_steer and S.launch_target_inc is not None) else None
    return launch_lib.max_departure_declination_deg(rc.launch, steer)


# ---------------------------------------------------------------------------
# resolution: derive the delta-v budget the screen compares against
# ---------------------------------------------------------------------------

def resolved() -> ResolvedConfig:
    """Resolve the working parts and price the escape at the planned departure v∞.

    The escape charge starts on the analytic estimate and is refined only by a spiral run launched
    in this session (:func:`session_spiral_run`): its recorded dV-vs-v∞ curve prices the escape
    charge / duration / propellant at the live :func:`departure_vinf`. With no session spiral the
    analytic estimate stands (priced at the same v∞ via :attr:`ResolvedConfig.departure_vinf_kms`),
    so nothing reads as "already solved" until the user flies the spiral.
    """
    rc = ResolvedConfig.build(S.mission, S.vehicle, S.screening, load_engines(), S.desirability)
    rc.departure_vinf_kms = S.departure_vinf
    run_id = session_spiral_run(rc)
    if run_id is None:
        return rc
    summary = jobs.read_spiral_summary(run_id)
    curve = ({k: summary.get(k) for k in ("curve_vinf_kms", "curve_dv_kms", "curve_tof_days")}
             if summary.get("curve_vinf_kms") else None)
    if curve is not None:
        # Priced the one way every cruise prices its escape (the design sweep, the worker's
        # rebuild of a sweep winner): the curve at the departure speed, the duration off the curve
        # or the constant-flow estimate, the propellant by the rocket equation at the wet mass.
        rc_priced, _terms = sweep_lib.price_escape(rc, curve, rc.departure_vinf_kms)
        return rc_priced
    # A run without a stored curve: its bare-escape numbers as recorded. The budget term is the
    # propellant-equivalent figure; an older summary has only one number, and that is what it was.
    equiv = summary.get("dv_at_escape_equiv_kms")
    rc.escape_dv_refined = float(equiv if equiv is not None else summary["dv_at_escape_kms"])
    if summary.get("tof_days") is not None:
        rc.escape_tof_refined = float(summary["tof_days"])
    if summary.get("propellant_kg") is not None:
        rc.escape_prop_refined = float(summary["propellant_kg"])
    return rc
