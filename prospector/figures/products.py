"""Solved results turned into the figures and derived numbers that go with them.

The app and the exported PDF have to show the same trajectory, the same escape and the same power
story. Building those in two places lets them drift: the report will happily embed a cruise the app
has already discarded as out of date. So everything between a stored result and a figure lives
here, and the interactive and printed versions differ only in their ``animate`` and ``controls``
arguments, never in the path they take through the code.

Nothing here touches the UI's own state. Deciding which run is the current one genuinely is session
state and stays in the UI, which passes the run id in. The figure builders return ``(caption,
figure)`` pairs, since the report needs a caption and the app can ignore it.
"""
from __future__ import annotations

import json
import math

import numpy as np

from prospector import jobs
from prospector import launch as launch_lib
from prospector.config import ResolvedConfig
from prospector.constants import G0_M_S2
from prospector.figures import escape as escape_figs
from prospector.figures import mass as mass_figs
from prospector.figures import power as power_figs
from prospector.figures import trajectory as traj_figs
from prospector.figures.porkchop import converged_grid
from prospector.solvers.simsflanagan import MISMATCH_TOL
from prospector.solvers.sunpower import sun_power_model


def converged(feasible, mismatch) -> bool:
    """Whether a solved leg closed: the optimizer's own flag, or a mismatch inside tolerance."""
    return bool(feasible) or (mismatch is not None and float(mismatch) < MISMATCH_TOL)


# ---------------------------------------------------------------------------
# Is a stored run still the current one?
# ---------------------------------------------------------------------------

def trajectory_signature(rc: ResolvedConfig, available_power_W: float | None) -> tuple:
    """The terms of a resolved config that determine the cruise trajectory.

What the vehicle can do, the mass and departure window the escape leaves the cruise with, the
    departure speed, the array power after the escape, and the mission's deadlines and return
    terms. Two configs matching on all of these produce the same trajectory, so a run on screen
    that no longer matches the live config is out of date. It has to disappear rather than sit
    there behind a warning, and it must not end up in a document either.
    """
    m = rc.mission
    dep0, dep1 = rc.departure_window
    sig = (
        round(float(rc.total_thrust_mN), 3), round(float(rc.effective_isp), 2),
        round(float(rc.total_power_W), 1), round(float(rc.cruise_start_mass_kg), 2),
        round(float(rc.departure_vinf_kms), 4),
        (round(float(available_power_W), 1) if available_power_W is not None else None),
        dep0.isoformat(), dep1.isoformat(), m.arrive_by.isoformat(),
        bool(m.return_trip), str(m.return_destination or ""),
        (m.return_by.isoformat() if getattr(m, "return_by", None) else ""),
        round(float(m.time_at_asteroid or 0.0), 3),
        round(float(getattr(m, "asteroid_payload_mass", 0.0) or 0.0), 3),
    )
    # A stack that switches between fixed modes flies a different power-limited cruise from one
    # that throttles smoothly on the same rated numbers. Appended only when it applies, so the
    # signature of a continuous stack, and of every run already on disk, is unchanged.
    modes = launch_lib.discrete_modes(rc)
    if modes:
        sig += (json.dumps(modes, sort_keys=True),)
    return sig


def cruise_run_matches(run_id: str, rc: ResolvedConfig | None,
                       available_power_W: float | None, *, unknown_ok: bool = True) -> bool:
    """Whether a finished cruise run was solved for the config in force now.

    Compares what the run was solved with, and the power it flew at, against the live config.
    Editing the vehicle, choosing a different one, or re-flying the escape spiral all move the
    cruise's starting mass, its departure window or its available power, and any of those makes the
    run out of date.

    ``unknown_ok`` decides what to do when the comparison cannot be made, and the two callers want
    opposite answers. For keeping a run the user just started on screen: a run whose job spec
    cannot be read, or that predates configs being stored, has nothing to compare against and is
    kept, rather than vanishing out from under them. For picking up a run found on disk: that same
    silence is no evidence at all, and acting on it would put an unrelated trajectory in front of
    someone who never asked for it, so pass ``unknown_ok=False``.
    """
    if rc is None:
        return unknown_ok
    try:
        job = jobs.read_job(run_id)
        cfg = job.get("config")
        if not cfg:
            return unknown_ok
        stored_rc = ResolvedConfig.model_validate(cfg)
        stored_power = (job.get("options") or {}).get("available_power_W")
    except (OSError, ValueError, KeyError):
        return unknown_ok
    return (trajectory_signature(stored_rc, stored_power)
            == trajectory_signature(rc, available_power_W))


def run_summary(result: dict) -> dict:
    """A compact, JSON-safe digest of a finished solve, for the run-history table.

    It lives here rather than in the worker because two things need it: the detached solve writes
    one when a job finishes, and the app writes one when a grid cell is rebuilt from its stored
    solution. One place to build it, so a run recorded either way reads the same in the history,
    which is the same reason the figures are here.
    """
    sf, tgt, lam = result["sf"], result["target"], result.get("lambert", {})
    summary = {
        "target": tgt["name"],
        "dep_mjd2000": float(sf["dep_mjd2000"]),
        "arr_mjd2000": float(sf["dep_mjd2000"]) + float(sf["tof_days"]),
        "tof_days": float(sf["tof_days"]),
        "dv_kms": float(sf["dv_kms"]),
        "propellant_kg": float(sf["propellant_kg"]),
        "feasible": bool(sf["feasible"]),
        "mismatch": float(sf["mismatch"]),
        "lambert_dv_kms": float(lam["best_dv_kms"]) if lam.get("best_dv_kms") is not None else None,
    }
    ret = result.get("return")
    if ret and "error" not in ret:
        rsf = ret["sf"]
        summary.update({
            "return_destination": ret.get("destination"),
            "return_dv_kms": float(rsf["dv_kms"]),
            "return_tof_days": float(rsf["tof_days"]),
            "return_propellant_kg": float(ret["total_return_propellant_kg"]),
            "insertion_propellant_kg": float(ret["insertion_propellant_kg"]),
            "delivered_mass_kg": float(ret["delivered_mass_kg"]),
            "return_feasible": bool(rsf["feasible"] and ret["feasible_propellant"]),
        })
    elif ret and "error" in ret:
        summary["return_error"] = ret["error"]
    return summary


# ---------------------------------------------------------------------------
# Reading the stored blocks, guarded
# ---------------------------------------------------------------------------

def spiral_block(spiral_run_id: str | None) -> dict | None:
    """The ``spiral`` block of an escape run, or None if there is no readable result.

    The caller decides which run counts. The app wants the one its project points at, whose
    fingerprint still matches, so whatever comes back here is already current.
    """
    if not spiral_run_id:
        return None
    res = jobs.read_spiral_result(spiral_run_id)
    return res.get("spiral") if res else None


def cruise_block(run_id: str | None, rc: ResolvedConfig | None = None,
                 available_power_W: float | None = None) -> dict | None:
    """The full result dict of a cruise run that is finished, converged, and still current.

    Returns None otherwise. Passing ``rc`` turns on the out-of-date check, and both the app and the
    report have to pass it, or a document can capture a trajectory the config no longer produces.
    """
    if not run_id or not jobs.is_terminal(jobs.read_status(run_id)):
        return None
    if rc is not None and not cruise_run_matches(run_id, rc, available_power_W):
        return None
    res = jobs.read_result(run_id)
    if res and converged(res["sf"]["feasible"], res["sf"]["mismatch"]):
        return res
    return None


def grid_block(run_id: str | None, rc: ResolvedConfig | None = None,
               available_power_W: float | None = None, *,
               target_pdes: str | None = None) -> dict | None:
    """The transfer grid of a run that is finished, readable, and still current.

    A grid is a picture of one vehicle flying to one body, so one whose config or target has moved
    on describes a mission that is no longer on screen. Both the app and the report gate on this,
    for the same reason they share :func:`cruise_block`: a document has nothing on it to mark a
    surface out of date.
    """
    if not run_id or not jobs.is_terminal(jobs.read_grid_status(run_id)):
        return None
    try:
        job = jobs.read_grid_job(run_id)
        cfg = job.get("config")
        if not cfg:
            return None
        stored = ResolvedConfig.model_validate(cfg)
    except (OSError, ValueError, KeyError):
        return None
    if rc is not None and (trajectory_signature(stored, available_power_W)
                           != trajectory_signature(rc, available_power_W)):
        return None
    if target_pdes is not None and (str((job.get("target") or {}).get("pdes") or "")
                                    != str(target_pdes or "")):
        return None
    return jobs.read_grid_result(run_id)


def grid_figure(grid: dict | None, rc: ResolvedConfig, *, colour_by: str = "margin",
                sel_dep: float | None = None, sel_tof: float | None = None):
    """The transfer grid and its best-per-flight-time curve as a ``(caption, figure)`` pair.

    None when there is no grid to draw. The curve across the grid is the flight-time trade, so this
    is what carries that trade in both the app and the document; there is no separate sweep.
    """
    if not grid or not grid.get("dep_mjd2000") or not grid.get("tof_days"):
        return None
    solved = int(grid.get("n_feasible") or 0)
    total = int(grid.get("n_cells") or 0)
    caption = (
        f"Transfer options: ({solved} of {total} solved)")
    fig = converged_grid(
        grid["dep_mjd2000"], grid["tof_days"], dv_kms=grid["dv_kms"],
        final_mass_kg=grid["final_mass_kg"], feasible=grid["feasible"],
        frontier=grid.get("frontier"), target_name=grid.get("target_name", ""),
        sel_dep=sel_dep, sel_tof=sel_tof,
        liftoff_offset_days=rc.escape_tof_days, colour_by=colour_by,
        dv_budget=rc.cruise_dv_limit, burnout_mass_kg=rc.vehicle.burnout_mass,
        past_deadline=grid.get("past_deadline"))
    return caption, fig


# ---------------------------------------------------------------------------
# Derived power and thrust numbers
# ---------------------------------------------------------------------------

def spiral_power_available_W(spiral_run_id: str | None) -> float | None:
    """The array power the cruise runs on: frozen at the post-escape (degraded) value, since the
    spacecraft is clear of the belts by then. None when no escape has flown or its power loop was
    not closed, in which case the cruise runs at rated power."""
    sp = spiral_block(spiral_run_id)
    if not sp:
        return None
    power = float(sp.get("power_available_end_W") or 0.0)
    return power if power > 0.0 else None


def available_thrust_fraction(rc: ResolvedConfig, available_power_W: float | None) -> float:
    """Fraction of rated thrust the cruise can produce at best: the operating point the
    post-escape array power buys at the closest approach a cruise is assumed to make (just
    inside 1 AU; see :mod:`prospector.solvers.sunpower`). 1.0 when power-rich or when no escape
    has flown. Further out the segments are capped below this by the solve itself."""
    if available_power_W is None:
        return 1.0
    model = sun_power_model(rc, available_power_W)
    if model is None:
        return 1.0
    rated = rc.total_thrust_mN * 1e-3
    return max(0.0, min(1.0, model.leg_thrust_N(None) / rated)) if rated > 0.0 else 1.0


def power_floor_pct(spiral: dict | None) -> float | None:
    """The array-power loss (% of beginning-of-life) past which the stack can no longer run at
    full thrust: ``(1 - power_required / bol) x 100``. The escape flies throttled beyond it.
    None when the power loop is not closed, so no zone is shaded."""
    sp = spiral or {}
    req = float(sp.get("power_required_W") or 0.0)
    avail_end = float(sp.get("power_available_end_W") or 0.0)
    frac_end = float(sp.get("power_fraction_end") or 0.0)
    bol = (avail_end / frac_end) if frac_end > 0.0 else 0.0
    if req <= 0.0 or bol <= 0.0:
        return None
    return max(0.0, (1.0 - req / bol) * 100.0)


def cruise_duty_cycle(sf: dict | None) -> dict | None:
    """The cruise's achieved duty cycle: the peak over any one segment, and the whole-phase mean.

    Both come off the same throttle history the propellant calculation uses. The throttle is the
    share of a segment's possible velocity change the solve used, and Isp is fixed here, so it is
    equally the share of that segment the engine spends firing.

    The two answer different questions and are usually far apart. The peak is what the per-segment
    limit acts on, so it says whether that limit was doing anything. The average is what the
    mission asks of the engine, and on real trips it is small, measured at 13-27%, because an
    efficient low-thrust trajectory is a few hard burns separated by long coasts. Quoting only the
    peak makes a mission look far busier than it is, which is how a duty-cycle limit ends up sized
    against the wrong number.

    None when the result carries no usable throttle history.
    """
    if not sf:
        return None
    days = np.asarray(sf.get("fine_times_days", []), float)
    thr = np.asarray(sf.get("fine_throttle", []), float)
    if len(days) < 2 or len(thr) != len(days):
        return None
    t_s = days * 86400.0
    span = float(t_s[-1] - t_s[0])
    if span <= 0.0:
        return None
    mean = float(np.sum(0.5 * (thr[1:] + thr[:-1]) * np.diff(t_s)) / span)
    cap = sf.get("max_duty_cycle")
    return {"peak": float(np.max(thr)), "mean": mean,
            "cap": None if cap is None else float(cap)}


def cruise_propellant_kg(rc: ResolvedConfig, sf: dict | None) -> float | None:
    """The propellant a cruise leg spent, in kilograms.

    The solved figure (``propellant_kg``) is what the leg burned at the operating points it flew,
    so it is used whenever the result carries it. The rocket equation at the vehicle's RATED Isp
    is the fallback for an older result without it, and it understates a power-limited cruise:
    the leg's ``dv_kms`` is defined with the leg's own, lower Isp, so inverting it with the rated
    Isp gives less mass than was actually lost. None with no result.
    """
    if not sf:
        return None
    if sf.get("propellant_kg") is not None:
        return float(sf["propellant_kg"])
    dv = sf.get("dv_kms")
    veff = float(rc.effective_isp) * G0_M_S2 * 1e-3
    if dv is None or veff <= 0.0:
        return None
    return float(rc.cruise_start_mass_kg) * (1.0 - math.exp(-float(dv) / veff))


def closure_ledger(rc: ResolvedConfig, sf: dict | None, ret: dict | None = None, *,
                   escape_propellant_kg: float | None = None) -> dict:
    """The mission's propellant account, in kilograms: what was aboard, what each leg spent at the
    operating points it flew, what is left, and whether it closes.

    The one ledger. The legs run at different Isps (and the escape at a changing one), so a
    delta-v subtraction against a single-Isp capability is not a closure test; the kilograms are.
    ``escape_propellant_kg`` overrides the config's escape term with a flown spiral's own mass
    history when the caller has it. ``ret`` is the return block; None means no return is planned,
    a block without its total (an error) leaves the account open.
    """
    usable = float(rc.usable_propellant_kg)
    escape = (float(rc.escape_propellant_kg) if escape_propellant_kg is None
              else float(escape_propellant_kg))
    cruise = cruise_propellant_kg(rc, sf)
    if ret is None:
        return_kg: float | None = 0.0
    else:
        total = ret.get("total_return_propellant_kg")
        return_kg = None if total is None else float(total)
    used = None if (cruise is None or return_kg is None) else escape + cruise + return_kg
    margin = None if used is None else usable - used
    return {"usable_kg": usable, "escape_kg": escape, "cruise_kg": cruise,
            "return_kg": return_kg, "used_kg": used, "margin_kg": margin,
            "closes": bool(margin is not None and margin >= 0.0)}


def cruise_thrust_ceiling_pct(rc: ResolvedConfig, sf: dict | None) -> float | None:
    """The cruise's thrust ceiling as a percentage of rated: the operating thrust the solve
    flew, over the vehicle's rated thrust. Passed to the trajectory figures so a
    power-limited cruise reads as its reduced throttle under a ceiling line, rather than looking
    like full thrust. None when unavailable."""
    if not sf:
        return None
    rated = rc.total_thrust_mN * 1e-3
    op = float(sf.get("thrust_N", 0.0))
    if rated <= 0.0 or op <= 0.0:
        return None
    return max(0.0, min(100.0, op / rated * 100.0))


# ---------------------------------------------------------------------------
# The propellant integral
# ---------------------------------------------------------------------------

def cruise_propellant_profile(res: dict | None):
    """Cumulative cruise propellant over time, and a cross-check of that integral.

    Returns ``(days, prop_cum_kg, verification)``: how much propellant the engine has used by each
    point, worked out as the area under the thrust history divided by the exhaust speed, and a dict
    comparing that against the propellant the solver itself reported. Empty arrays and None when
    the result has no usable thrust history.

    The throttle history is a fraction of the thrust the solve flew at, which is lower when the
    cruise ran short of power. Working against the vehicle's rated numbers instead would overstate
    the propellant and push the curve above the usable load, so the flown numbers are used and the
    rated ones are only a fallback for results that never recorded them.
    """
    empty = (np.empty(0), np.empty(0), None)
    if not res:
        return empty
    sf = res["sf"]
    days = np.asarray(sf.get("fine_times_days", []), float)
    thr = np.asarray(sf.get("fine_throttle", []), float)
    veh = res.get("vehicle", {})
    thrust_N = float(sf.get("thrust_N", float(veh.get("thrust_mN", 0.0)) * 1e-3))
    veff_ms = float(sf.get("isp_s", veh.get("isp_s", 0.0))) * G0_M_S2
    if len(days) < 2 or veff_ms <= 0.0 or thrust_N <= 0.0:
        return empty
    t_s = days * 86400.0
    rate = thrust_N * thr / veff_ms                 # propellant mass flow (kg/s) per sample
    seg = 0.5 * (rate[1:] + rate[:-1]) * np.diff(t_s)
    prop_cum = np.concatenate(([0.0], np.cumsum(seg)))
    tof_s = float(t_s[-1])
    avg_throttle = (float(np.sum(0.5 * (thr[1:] + thr[:-1]) * np.diff(t_s)) / tof_s)
                    if tof_s else 0.0)
    verification = {
        "thrust_N": thrust_N,
        "veff_kms": veff_ms / 1000.0,
        "avg_throttle": avg_throttle,
        "integral_prop_kg": float(prop_cum[-1]),
        "reported_prop_kg": float(sf.get("propellant_kg", 0.0)),
    }
    return days, prop_cum, verification


# ---------------------------------------------------------------------------
# Figures
#
# Each returns (caption, figure) pairs. ``animate`` and ``controls`` are the only difference
# between what the app shows and what goes on paper.
#
# Absent data returns nothing: no spiral run means no escape figures, which is a state and not a
# failure. A result block that exists and cannot be plotted raises, since swallowing it would drop
# a chart from a document with nothing to say it went missing. Rasterization is the one place a
# failure degrades, and it degrades visibly: the report draws a placeholder naming the error.
# ---------------------------------------------------------------------------

def escape_figures(spiral: dict | None, *, animate: bool = True) -> list:
    """The escape spiral's 3D climb-out and its energy / tilt / mass diagnostics."""
    if not spiral or not len(spiral.get("positions_km", [])):
        return []
    figs = []
    figs.append(("The escape spiral, seen from Earth (color shows elapsed time)",
                     escape_figs.spiral_3d(
                         spiral["positions_km"], spiral["times_days"], title="Escape spiral",
                         dv_kms=spiral.get("dv_kms"), tof_days=spiral.get("tof_days"),
                         animate=animate)))
    if all(k in spiral for k in ("energy_km2_s2", "inc_deg", "mass_kg")):
        figs.append(("Orbital energy, tilt, and spacecraft mass over the climb",
                     escape_figs.spiral_diagnostics(
                         spiral["times_days"], spiral["energy_km2_s2"],
                         spiral["inc_deg"], spiral["mass_kg"])))
    return figs


def array_power_figure(spiral: dict | None):
    """Array power lost over the escape, with the power-limited zone shaded.

    Uses the per-point figure the run flew, which follows whichever radiation model was chosen, and
    falls back to recomputing it with the default model for runs that predate storing it.
    """
    if not spiral or not len(spiral.get("positions_km", [])):
        return None
    times = np.asarray(spiral["times_days"], float)
    fraction = np.asarray(spiral.get("power_fraction", []), float)
    if fraction.size != len(times) or not fraction.size:
        from prospector.solvers.spiral import belt_power_profile
        pos = np.asarray(spiral["positions_km"], float)
        _in_belt, _belt_cum, fraction = belt_power_profile(pos, times)
    if not len(fraction):
        return None
    return ("Solar-array degradation over the escape (shaded band = flying power-limited)",
            power_figs.degradation_profile(times, fraction,
                                           power_floor_pct=power_floor_pct(spiral)))


def cruise_figures(res: dict | None, *, max_throttle_pct: float | None = None,
                   depart_label: str = "depart", arrive_label: str = "arrive",
                   animate: bool = True, controls: bool = True) -> list:
    """The converged cruise arc (colored by throttle) and its per-node diagnostics."""
    if not res:
        return []
    sf, orb, tgt = res["sf"], res.get("orbits", {}), res.get("target", {})
    figs = [("The cruise trajectory (color shows engine throttle)",
             traj_figs.trajectory_3d(
                         sf["fine_positions_au"], sf["fine_throttle"], sf["fine_times_days"],
                         sf["node_times_days"], sf["throttle"], orb["earth_au"],
                         orb["target_au"], orb["earth_track_au"], orb["target_track_au"],
                         depart_label, arrive_label, tgt.get("name", ""), sf.get("dv_kms"),
                         sf.get("tof_days"), target_i_deg=tgt.get("i"), animate=animate,
                     controls=controls, max_throttle_pct=max_throttle_pct))]
    if "node_thrust_normal" in sf:
        figs.append(("Thrust direction and orbit shape over the cruise",
                     traj_figs.trajectory_diagnostics(
                             sf["node_times_days"], sf["throttle"], sf["node_thrust_radial"],
                             sf["node_thrust_transverse"], sf["node_thrust_normal"],
                             sf["node_i_deg"], sf["node_a_au"], sf["node_e"],
                             speed_kms=sf.get("node_speed_kms"),
                             earth_speed_dep=sf.get("earth_speed_dep_kms"),
                             target_speed_arr=sf.get("target_speed_arr_kms"),
                             target_a=tgt.get("a"), target_e=tgt.get("e"),
                         target_i=tgt.get("i"), target_name=tgt.get("name", ""),
                         max_throttle_pct=max_throttle_pct)))
    return figs


def propellant_timeline_figure(rc: ResolvedConfig, spiral: dict | None,
                               cruise_days, cruise_prop_cum):
    """Cumulative propellant across escape (from the spiral's own mass history) and cruise,
    against the usable load."""
    if spiral and len(spiral.get("mass_kg", [])):
        ed = np.asarray(spiral["times_days"], float)
        mass = np.asarray(spiral["mass_kg"], float)
        ep = mass[0] - mass
    else:
        ed = ep = np.empty(0)
    cd = np.asarray(cruise_days, float) if cruise_days is not None else np.empty(0)
    cp = np.asarray(cruise_prop_cum, float) if cruise_prop_cum is not None else np.empty(0)
    if not len(ed) and not len(cd):
        return None
    return ("Propellant consumed over the mission",
            mass_figs.propellant_timeline(ed, ep, cd, cp, usable_kg=rc.usable_propellant_kg))


def mass_allocation_figure(components, *, dry_mass_kg: float):
    """Where the dry mass goes, from an already-computed component breakdown."""
    if not components:
        return None
    return ("Where the dry mass goes",
            mass_figs.mass_allocation(components, dry_mass_kg=float(dry_mass_kg)))
