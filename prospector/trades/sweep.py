"""
Finding the split between escape and cruise that costs the least delta-v overall.

The launch and the cruise trade against each other through one number: how fast the spacecraft is
going when it finally leaves Earth. Leaving faster costs the escape spiral more, priced off the
spiral's own curve, but hands the cruise free energy and tilt so the cruise costs less -- and the
other way round. Neither side sees the whole picture: the spiral does not know what the cruise
saves, and the cruise treats its departure speed as free up to a limit it is simply handed.

This module reads that trade end to end for one target. It runs the cruise solve at several
departure speeds, prices each one's escape off the spiral curve, and reports the total for each so
the cheapest is visible. Each point rebuilds the physics its speed implies instead of reusing one
set of numbers everywhere: leaving faster means a longer, hungrier spiral, so the cruise starts
later and lighter.

One thing to expect when reading a sweep: because each point's escape takes a different length of
time, each point's departure window sits somewhere different, so the departure dates cannot be
compared at face value. Every point therefore records its schedule too, in ``liftoff_date`` and in
the ``coast_days`` between the spiral handing over and the cruise departing, so its dates read as a
schedule instead of an unexplained offset.

Leaving at no extra speed at all, with the cruise buying every bit of energy and tilt itself, is
always a meaningful point, and the default set of speeds includes it. A point whose solve fails or
does not converge is recorded as not working, with whatever numbers exist. The sweep never gives up
partway, because a missing point is an option that quietly disappeared.

Importable with no UI dependency and no subprocess; ``worker.py sweep`` runs it detached, via
``jobs.submit_sweep``. The heavy solver import happens inside the per-point call, so this module
loads without the solver extras installed.

Units: km/s, kg, days; dates as MJD2000 / ISO strings in the returned points.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

import numpy as np

from prospector.config import ResolvedConfig
from prospector.constants import G0_KM_S2
from prospector.launch import LaunchOrbit, escape_dv_estimate, spiral_time_estimate_days

# The Sims-Flanagan transcription scales its v-infinity inequality by the cap, so a cap of a hard
# zero divides by zero. A 1 m/s allowance is physically indistinguishable from a zero-excess
# departure; the v_inf = 0 sweep point runs the solver with this floor while the escape side is
# priced at the true zero.
VINF_CAP_FLOOR_KMS = 1e-3

_MJD2000_EPOCH = datetime(2000, 1, 1)

ProgressFn = Callable[[int, int, dict], None]


# ---------------------------------------------------------------------------
# escape pricing: the spiral curve, extended analytically past its end
# ---------------------------------------------------------------------------

def escape_cost(curve: dict, vinf_kms, orbit: LaunchOrbit):
    """Price the launch phase at departure v-infinity ``vinf_kms``: ``(dv_kms, tof_days)``.

    ``curve`` is what a spiral run recorded: the ``curve_vinf_kms``, ``curve_dv_kms`` and
    ``curve_tof_days`` arrays from a spiral summary or solution. Inside the curve's range the cost
    is read straight off it, which means real flown numbers. Past the curve's last speed it is
    extended by adding on how much more the analytic estimate says the extra speed costs,

        dv(v) = curve_dv[-1] + (estimate(v) - estimate(v_max)),

    so the extension joins the flown data smoothly instead of running the curve's local slope off
    into nonsense. The time is extended by applying that extra delta-v at the rate of time per
    delta-v the curve ends at, where the mass is barely changing. With no curve at all the estimate
    prices the delta-v and the time comes back as NaN, leaving the caller to fall back to its own
    guess. A launch vehicle that escapes by itself costs the spacecraft nothing at any speed, which
    matches how a finished solve prices the mission total.

    Vectorized: ``vinf_kms`` may be a scalar or an array; scalars return float pairs.
    """
    v = np.asarray(vinf_kms, float)
    scalar = (v.ndim == 0)
    v = np.atleast_1d(v)

    if orbit.escape_provided:
        dv, tof = np.zeros_like(v), np.zeros_like(v)
        return (float(dv[0]), float(tof[0])) if scalar else (dv, tof)

    cv = _curve_array(curve, "curve_vinf_kms")
    cd = _curve_array(curve, "curve_dv_kms")
    ct = _curve_array(curve, "curve_tof_days")
    if cv.size == 0 or cd.size != cv.size:
        dv = np.hypot(orbit.v_circ_kms, v)             # the analytic sqrt(vc^2 + vinf^2)
        tof = np.full_like(v, np.nan)
        return (float(dv[0]), float(tof[0])) if scalar else (dv, tof)

    dv = np.interp(v, cv, cd)
    tof = np.interp(v, cv, ct) if ct.size == cv.size else np.full_like(v, np.nan)

    beyond = v > cv[-1]
    if np.any(beyond):
        marginal = np.hypot(orbit.v_circ_kms, v[beyond]) - escape_dv_estimate(orbit, cv[-1])
        dv[beyond] = cd[-1] + marginal
        if ct.size == cv.size:
            # Marginal time per marginal dv at the curve's end: the last segment's slope, or the
            # whole-curve average when the curve is a single point / flat.
            if cv.size >= 2 and cd[-1] > cd[-2]:
                rate = (ct[-1] - ct[-2]) / (cd[-1] - cd[-2])
            elif cd[-1] > 0:
                rate = ct[-1] / cd[-1]
            else:
                rate = 0.0
            tof[beyond] = ct[-1] + marginal * rate
    return (float(dv[0]), float(tof[0])) if scalar else (dv, tof)


def _curve_array(curve: dict | None, key: str) -> np.ndarray:
    """One curve array as float ndarray; missing/None becomes empty (lists and numpy
    arrays both pass through, so a spiral summary or a SpiralSolution dict works)."""
    value = (curve or {}).get(key)
    return np.asarray(value if value is not None else [], float)


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------

def default_vinf_grid(cap_kms: float, n: int = 7) -> list[float]:
    """An evenly spaced departure-v-infinity grid from 0 to ``cap_kms`` inclusive.

    Always includes 0.0, where the cruise buys all the tilt itself, and the cap. A cap of zero or
    less collapses to that single point.
    """
    cap = float(cap_kms)
    if cap <= 0.0:
        return [0.0]
    values = [float(x) for x in np.linspace(0.0, cap, max(2, int(n)))]
    values[0], values[-1] = 0.0, cap          # endpoints exact despite float spacing
    return values


def run_sweep(config: dict, target: dict, *, vinf_values, curve: dict | None,
              options: dict | None = None,
              progress: ProgressFn | None = None,
              workers: int = 1) -> list[dict]:
    """Sweep the Sims-Flanagan cruise solve over departure-v-infinity caps for one target.

    ``config`` is a serialized :class:`ResolvedConfig` (``rc.model_dump(mode="json")``); ``target``
    is a population row with the full orbit and identifiers; ``vinf_values`` are the departure
    speeds to try in km/s; ``curve`` is what the spiral run recorded (see :func:`escape_cost`; None
    or empty falls back to the estimate). ``options`` are solver settings such as ``nseg``,
    ``maxeval`` and ``restarts``, applied to every point. Any ``vinf_dep_kms`` in them is ignored,
    since the sweep owns the departure speed.

    Each point rebuilds the physics its speed implies before solving:

      * the escape delta-v and duration come from :func:`escape_cost` at that speed,
      * the mass the cruise starts with is the rocket equation applied to the full wet mass at
        that escape delta-v,
      * the departure window is the mission's liftoff window shifted by that point's own escape
        duration, since leaving faster means a longer spiral and a later cruise start,

    then it runs the cruise solve with that speed, that mass and that window. Returns one dict per
    speed with the escape, cruise and total figures (see ``_solve_point``). A point whose solve
    raises or fails to converge is recorded with ``feasible=False`` and whatever numbers exist; the
    sweep never gives up partway. ``progress(i, n, point)`` is called as each point finishes, in
    whatever order they finish, though the returned list always keeps the original order.

    ``workers`` above 1 spreads the points across that many processes. They are completely
    independent solves, and pygmo and pykep hold the GIL, so threads would not help. The default of
    1 keeps everything in this process.
    """
    rc = ResolvedConfig.model_validate(config)
    # Re-osculate the target at the mission's arrival era ONCE, up front - the refreshed row is
    # plain data, so the parallel children inherit it through pickling and every point solves
    # against the same (correct) orbit. See population.refresh_target_elements for why (planetary
    # close approaches).
    from prospector import population
    target = population.refresh_target_elements(dict(target), rc.mission.arrive_by)
    opts = dict(options or {})
    opts.pop("vinf_dep_kms", None)            # the sweep sets the cap per point
    # The return-segment solver knobs travel separately so they never reach the outbound
    # cruise solve; each point solves the return (when planned) with these.
    return_opts = opts.pop("return_options", None)
    values = [float(x) for x in vinf_values]
    workers = max(1, min(int(workers), len(values)))

    if workers == 1:
        points = []
        for k, vinf in enumerate(values):
            point = _solve_point(rc, target, vinf, curve, opts, return_opts)
            points.append(point)
            if progress is not None:
                progress(k + 1, len(values), point)
        return points

    # Every point is an independent solve (its own escape price, window, mass, and seed), so they
    # fan out across processes. pygmo and pykep hold the GIL, so threads would serialize. Each
    # child gets picklable plain data and rebuilds the config itself; progress streams in
    # completion order, the returned list keeps cap order.
    import concurrent.futures as cf

    points: list = [None] * len(values)
    done = 0
    # Forked workers are correct here: this runs inside the detached ``worker.py`` process, which
    # holds no listening socket for a child to inherit. A pool started inside the app itself would
    # have to spawn instead. ``with`` shuts the pool down waiting, so a failed point cannot leave
    # workers behind.
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_solve_point_job, config, target, vinf, curve, opts, return_opts): k
                   for k, vinf in enumerate(values)}
        for fut in cf.as_completed(futures):
            k = futures[fut]
            try:
                point = fut.result()
            except Exception as exc:          # a dead child must not abort the sweep
                point = _failed_point(rc, values[k], curve,
                                      f"{type(exc).__name__}: {exc}")
            points[k] = point
            done += 1
            if progress is not None:
                progress(done, len(values), point)
    return points


def _solve_point_job(config: dict, target: dict, vinf_kms: float,
                     curve: dict | None, opts: dict, return_opts: dict | None = None) -> dict:
    """Child-process entry for one sweep point: plain-data args in, point dict out."""
    rc = ResolvedConfig.model_validate(config)
    return _solve_point(rc, target, vinf_kms, curve, opts, return_opts)


def _failed_point(rc: ResolvedConfig, vinf_kms: float, curve: dict | None,
                  error: str) -> dict:
    """A point whose child process died outright: escape terms priced, solve fields empty."""
    try:
        point, _ = _point_base(rc, vinf_kms, curve)
    except Exception:
        point = {"vinf_kms": float(vinf_kms), "feasible": False}
    point["error"] = error
    return point


def _solve_point(rc: ResolvedConfig, target_row: dict, vinf_kms: float,
                 curve: dict | None, opts: dict, return_opts: dict | None = None) -> dict:
    """Evaluate one cap: price the escape, rebuild the post-spiral state, then run the same
    evaluation the app runs for a vehicle (:func:`pipeline.solve.evaluate_candidate`): the cruise
    from a walk of flight times, and the laden return leg when the mission returns."""
    point, rc_point = _point_base(rc, vinf_kms, curve)
    cap = max(float(vinf_kms), VINF_CAP_FLOOR_KMS)
    try:
        cells = _walk_cells(rc_point, target_row, cap)
        result = _evaluate_candidate(rc_point, target_row, cells=cells,
                                     sf_options={**opts, "vinf_dep_kms": cap},
                                     return_options=return_opts, workers=1, light=True)
    except Exception as exc:                       # a failed point must not abort the sweep
        point["error"] = f"{type(exc).__name__}: {exc}"
        return point
    return _point_from_result(point, rc_point, result)


def _point_from_result(point: dict, rc_point: ResolvedConfig, result: dict) -> dict:
    """Fill a priced point from the evaluation's result block (the app's own shape)."""
    sf = result["sf"]
    escape_dv, escape_prop = point["escape_dv_kms"], point["escape_propellant_kg"]
    cruise_dv = float(sf["dv_kms"])
    dep, arr = float(sf["dep_mjd2000"]), float(sf["dep_mjd2000"]) + float(sf["tof_days"])
    # Every point has its OWN escape duration, so every point's departure window sits somewhere
    # different, which is why the departure dates across a sweep look unrelated. Recording the seam
    # per point makes that legible: how much slack this cap leaves between the spiral handing over
    # and the cruise departing (see ResolvedConfig.departure_schedule).
    sched = rc_point.departure_schedule(_date_obj_from_mjd2000(dep))
    point.update({
        "coast_days": sched["coast_days"],
        "liftoff_date": (sched["liftoff_if_no_coast"].isoformat()
                         if sched["liftoff_if_no_coast"] else None),
        "liftoff_in_window": sched["liftoff_in_window"],
        "feasible": bool(sf["feasible"]),
        "mismatch": float(sf["mismatch"]),
        "settled": bool(sf.get("refresh_settled", True)),
        "cruise_dv_kms": cruise_dv,
        "total_dv_kms": escape_dv + cruise_dv,
        "tof_days": float(sf["tof_days"]),
        "cruise_propellant_kg": float(sf["propellant_kg"]),
        "propellant_kg": float(escape_prop) + float(sf["propellant_kg"]),
        "dep_mjd2000": dep,
        "arr_mjd2000": arr,
        "dep_date": _date_from_mjd2000(dep),
        "arr_date": _date_from_mjd2000(arr),
        "vinf_dep_used_kms": _vinf_dep_used(sf),
        # The optimizer's full answer at this cap. The trajectory is deterministic given the same
        # problem terms, so the winning point rebuilds into a complete run (cards, plots) without
        # ever re-optimizing. The converged per-segment thrust ceilings ride along so the rebuild
        # carries the exact capped problem.
        "decision_vector": _as_list(sf.get("decision_vector")),
        "seg_caps": _as_list(sf.get("seg_caps")),
        # The leg's thrust and Isp the vector was optimized with. The Isp a solve settles on
        # depends on its trajectory (the mean of its segments' operating points), so a rebuild
        # cannot derive it and has to be handed it.
        "cruise_thrust_N": _maybe_float(sf.get("thrust_N")),
        "cruise_isp_s": _maybe_float(sf.get("isp_s")),
        "cruise_seg_isp_s": _as_list(sf.get("seg_isp_s")),
    })

    # The laden return leg, when the mission flies one. It does not change total_dv_kms (the
    # escape/cruise split the sweep optimizes), and the return dV is reported separately, but its
    # propellant joins the mission total, so a design that can't fly home reads as such.
    ret = result.get("return")
    if rc_point.mission.return_trip and ret is not None:
        if "error" in ret:
            point["return_error"] = str(ret["error"])
            return point
        rsf = ret["sf"]
        insertion = float(ret["insertion_propellant_kg"])
        return_total = float(ret["total_return_propellant_kg"])
        point.update({
            "return_dv_kms": float(rsf["dv_kms"]),
            "return_tof_days": float(rsf["tof_days"]),
            "return_cruise_propellant_kg": float(rsf["propellant_kg"]),
            "insertion_prop_kg": insertion,
            "return_propellant_kg": return_total,
            "return_available_prop_kg": float(ret["available_propellant_kg"]),
            "return_feasible": bool(rsf["feasible"] and ret["feasible_propellant"]),
            "delivered_mass_kg": float(ret["delivered_mass_kg"]),
            "round_trip_days": (float(sf["tof_days"]) + float(rc_point.mission.time_at_asteroid or 0.0)
                                + float(rsf["tof_days"])),
            "propellant_kg": float(escape_prop) + float(sf["propellant_kg"]) + return_total,
            "return_decision_vector": _as_list(rsf.get("decision_vector")),
            "return_seg_caps": _as_list(rsf.get("seg_caps")),
            "return_thrust_N": _maybe_float(rsf.get("thrust_N")),
            "return_isp_s": _maybe_float(rsf.get("isp_s")),
            "return_seg_isp_s": _as_list(rsf.get("seg_isp_s")),
        })
    return point


def _as_list(value):
    """A plain list for JSON, or None."""
    return None if value is None else np.asarray(value, float).tolist()


def _evaluate_candidate(*args, **kwargs):
    """:func:`pipeline.solve.evaluate_candidate`, imported at the call so this module stays
    importable without the solver extras and the tests have one seam to stand in for it."""
    from prospector.trades.pipeline.solve import evaluate_candidate
    return evaluate_candidate(*args, **kwargs)


def with_escape_terms(rc: ResolvedConfig, *, dv_kms: float, tof_days: float,
                      propellant_kg: float) -> ResolvedConfig:
    """The config with a priced escape injected, the way a converged spiral refines it: the
    derived cruise-start mass and departure window then follow from these terms."""
    return rc.model_copy(update={
        "escape_dv_refined": float(dv_kms),
        "escape_tof_refined": float(tof_days),
        "escape_prop_refined": float(propellant_kg),
    })


def price_escape(rc: ResolvedConfig, curve: dict | None, vinf_kms: float
                 ) -> tuple[ResolvedConfig, dict]:
    """Price the escape at one departure speed and inject it into the config.

    The one place the escape is priced for a cruise, whether the app refines its budget from a
    flown spiral, the design sweep evaluates a cap, or the worker rebuilds a sweep's winner. The
    delta-v comes from :func:`escape_cost` (the spiral's propellant-equivalent curve, or the
    analytic estimate without one), the duration from the curve or the constant-mass-flow
    estimate, and the propellant from the rocket equation at the full wet mass. Returns the
    refined config and ``{escape_dv_kms, escape_tof_days, escape_propellant_kg}``.
    """
    escape_dv, escape_tof = escape_cost(curve, vinf_kms, rc.launch)
    escape_dv = float(escape_dv)
    if not np.isfinite(escape_tof):
        # No propagated duration at this v-infinity (no curve): the constant-mass-flow estimate
        # stands in, as the budget's own fallback does.
        escape_tof = spiral_time_estimate_days(
            rc.launch, wet_mass_kg=rc.vehicle.wet_mass, thrust_N=rc.total_thrust_mN * 1e-3,
            isp_s=rc.effective_isp, vinf_kms=vinf_kms)
    escape_tof = float(escape_tof)
    # Tsiolkovsky at the full wet mass: the propellant this cap's spiral burns.
    veff_kms = rc.effective_isp * G0_KM_S2
    escape_prop = (rc.vehicle.wet_mass * (1.0 - float(np.exp(-escape_dv / veff_kms)))
                   if veff_kms > 0 else 0.0)
    rc_point = with_escape_terms(rc, dv_kms=escape_dv, tof_days=escape_tof,
                                 propellant_kg=escape_prop)
    return rc_point, {"escape_dv_kms": escape_dv, "escape_tof_days": escape_tof,
                      "escape_propellant_kg": escape_prop}


def _point_base(rc: ResolvedConfig, vinf_kms: float,
                curve: dict | None) -> tuple[dict, ResolvedConfig]:
    """Price one cap's escape and rebuild the post-spiral state (no cruise solve yet).

    Returns the skeleton of the point, with the escape terms filled in and the solve fields still
    empty, plus the config for this point, whose mass and window follow from this speed.
    """
    rc_point, terms = price_escape(rc, curve, vinf_kms)
    escape_dv, escape_tof, escape_prop = (terms["escape_dv_kms"], terms["escape_tof_days"],
                                          terms["escape_propellant_kg"])
    window = rc_point.departure_window

    point = {
        "vinf_kms": float(vinf_kms),
        "escape_dv_kms": escape_dv,
        "escape_tof_days": escape_tof,
        "escape_propellant_kg": float(escape_prop),
        "cruise_start_mass_kg": float(rc_point.cruise_start_mass_kg),
        "dep_window": [window[0].isoformat(), window[1].isoformat()],
        "coast_days": None,
        "liftoff_date": None,
        "liftoff_in_window": None,
        "feasible": False,
        "mismatch": None,
        "cruise_dv_kms": None,
        "total_dv_kms": None,
        "tof_days": None,
        "cruise_propellant_kg": None,
        "propellant_kg": None,
        "dep_mjd2000": None,
        "arr_mjd2000": None,
        "dep_date": None,
        "arr_date": None,
        "vinf_dep_used_kms": None,
        # Return-leg fields, filled only when the mission returns (None otherwise so a one-way
        # point carries the same schema).
        "return_dv_kms": None,
        "return_tof_days": None,
        "return_cruise_propellant_kg": None,
        "insertion_prop_kg": None,
        "return_propellant_kg": None,
        "return_available_prop_kg": None,
        "return_feasible": None,
        "delivered_mass_kg": None,
        "round_trip_days": None,
        "return_decision_vector": None,
        "return_error": "",
        "error": "",
    }
    return point, rc_point


# Flight times tried per sweep point. A point stands in for choosing this vehicle and running the
# grid on it, so it has to look for the cheapest trip rather than trust one guess. Six covers the
# basin on the targets this was measured against, and a solve now costs a second or two, which it
# did not when the sweep was written.
SWEEP_TOF_STEPS = 6


def _seed_at(rc: ResolvedConfig, target, target_row: dict, tof_days: float, vinf_cap_kms: float):
    """A two-burn transfer at the cheapest departure for one flight time, or None.

    Departures are stepped across the point's own window and ranked on the two-burn cost with the
    departure speed this point buys already credited: the escape delivers that, and charging the
    cruise for it again favours cells a low-thrust vehicle does not want.
    """
    import numpy as np

    from prospector.solvers import lambert as lb

    dep0 = lb.mjd2000_from_date(rc.departure_window[0]) + 1.0
    dep1 = lb.mjd2000_from_date(rc.departure_window[1]) - 1.0
    deps = np.arange(dep0, max(dep1, dep0) + 1.0, 10.0)
    arrs = deps + float(tof_days)
    pc = lb.porkchop(target, deps, arrs, dep_body=lb.earth_planet(), max_revs=2)
    dv = lb.credited_dv(np.diag(pc.vinf_dep_kms), np.diag(pc.vinf_arr_kms),
                        max(float(vinf_cap_kms), 0.0))
    dv = np.where(np.isfinite(np.diag(pc.dv_kms)), dv, np.nan)
    if not np.isfinite(dv).any():
        return None
    i = int(np.nanargmin(dv))
    return lb.lambert_transfer(lb.earth_planet(), target, float(deps[i]), float(arrs[i]),
                               target_name=str(target_row.get("full_name") or "target"),
                               max_revs=2)


def _walk_cells(rc: ResolvedConfig, target_row: dict, vinf_cap_kms: float) -> list:
    """The starting cells for one sweep point: a walk of flight times from the longest the
    deadline allows down to the shortest, each at its cheapest two-burn departure.

    A point stands in for choosing this vehicle and running the grid on it, so it starts the
    shared multi-start (:func:`pipeline.solve.evaluate_candidate`) from a spread of flight times
    rather than one guess. One guess is not enough: a two-burn surface is cheapest on short
    transfers, and seeding there settles into a worse basin than a longer, more patient trip
    reaches. Measured on 2008 EV5 at a 2 km/s departure: 3.10 km/s from the walk against 3.56 from
    the cheapest single seed. A flight time with no two-burn transfer contributes no cell; with
    none at all, the longest trip from the window's last day is the one start.
    """
    from prospector.solvers import lambert as lb

    target = lb.planet_from_row(target_row)
    dep1 = lb.mjd2000_from_date(rc.departure_window[1]) - 1.0
    longest = lb.mjd2000_from_date(rc.mission.arrive_by) - dep1
    shortest = min(150.0, longest)
    cells = []
    for tof in np.linspace(longest, shortest, SWEEP_TOF_STEPS):
        try:
            seed = _seed_at(rc, target, target_row, float(tof), vinf_cap_kms)
        except Exception:                    # noqa: BLE001 -- one flight time, never the point
            seed = None
        if seed is not None:
            cells.append((float(seed.dep_mjd2000), float(seed.arr_mjd2000)))
    return cells or [(float(dep1), float(dep1 + longest))]


def _maybe_float(value) -> float | None:
    """A float, or None for a solution that does not carry the term (a stub, or an older shape)."""
    return None if value is None else float(value)


def _vinf_dep_used(sol) -> float | None:
    """How much of its allowed departure speed the optimizer spent (km/s).

    Read straight off the solution vector's departure velocity components, in m/s, with no need to
    look up where anything is. Which part of the vector to read is taken from the solver module
    rather than written out here: it sits between the final mass and the arrival velocity, so an
    offset that drifted would keep returning a plausible speed built from the wrong numbers.
    Returns None if the solution carries no vector it recognises.
    """
    from prospector.solvers.simsflanagan import _I_VINF_DEP

    try:
        vec = sol["decision_vector"] if isinstance(sol, dict) else sol.decision_vector
        z = np.asarray(vec, float)
        return float(np.linalg.norm(z[_I_VINF_DEP]) / 1000.0)
    except (TypeError, ValueError, AttributeError, IndexError, KeyError):
        return None


def _date_obj_from_mjd2000(mjd2000: float):
    """The ``date`` at an MJD2000 epoch (pure datetime; no ephemeris library)."""
    return (_MJD2000_EPOCH + timedelta(days=float(mjd2000))).date()


def _date_from_mjd2000(mjd2000: float) -> str:
    """ISO calendar date for an MJD2000 epoch."""
    return _date_obj_from_mjd2000(mjd2000).isoformat()
