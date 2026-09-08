"""Solving one target, and building the result the app reads.

    evaluate_candidate    one vehicle against one target from a spread of starts: the cruise, the
                          return when planned, and the result block; the interactive refine and the
                          design sweep both come through here, so a swept point IS the trajectory
                          the app would fly for that vehicle
    solve_from_cell       start from a chosen (departure, arrival) cell; what the UI uses
    result_from_decision  rebuild a solution that was already found, without solving again

Both return the same shape, so a swept point reopened from its stored solution looks no
different from one solved interactively. A planned return leg is solved in the same job, and a
return that cannot be solved records why instead of throwing away the outbound.
"""
from __future__ import annotations

from prospector.solvers import lambert as lb
from prospector.solvers import simsflanagan as sf
from prospector.trades.pipeline.arcs import ProgressFn, _assemble_leg, _mission_elements, _noop
from prospector.trades.pipeline.multistart import (
    _diverse_seed_cells,
    _multistart_outbound,
    solve_cells,
)


def _finish(rc, target, earth, target_row, seed, on_progress: ProgressFn, sf_kwargs,
            sol=None, progress_scale: tuple[float, float] = (0.0, 1.0),
            max_revs: int = 2, light: bool = False) -> dict:
    """Run the Sims-Flanagan solve from ``seed`` and build the result dict. Pass ``sol`` to build around a solution that has already been found, such as a sweep
    point rebuilt from its stored vector, without solving again. ``progress_scale`` maps this
    leg's 0..1 progress into part of the bar, so the outbound can report on (0, 0.5) while the
    return that follows it reports on (0.5, 1.0).

    ``sf_kwargs`` may carry ``n_starts``, where a value above 1 runs several starting guesses
    in parallel through :func:`_multistart_outbound`, and ``starts_workers``, the number of
    processes to use, where 0 or None sizes it to the available cores. Both are consumed here and
    not passed on to the solver."""
    lo, hi = progress_scale

    def scaled(stage: str, frac: float, message: str) -> None:
        on_progress(stage, lo + (hi - lo) * frac, message)

    def _sf_progress(done: int, total: int) -> None:
        scaled("simsflanagan", 0.15 + 0.78 * done / total,
               f"optimizing the trajectory ({done}/{total})")

    if sol is None:
        n_starts = int(sf_kwargs.pop("n_starts", 1) or 1)
        starts_workers = sf_kwargs.pop("starts_workers", None)
        if n_starts > 1 and seed is not None:
            sol = _multistart_outbound(rc, target_row, seed, sf_kwargs, n_starts,
                                       starts_workers, max_revs, scaled)
        else:
            sol = sf.solve_for_config(rc, target, seed=seed, progress=_sf_progress, **sf_kwargs)

    sf_block, orbits = _assemble_leg(sol, earth, target, 1.0, float(target_row["a"]), scaled,
                                     light=light)
    scaled("done", 1.0, "complete")
    name = str(target_row.get("full_name") or target_row.get("pdes") or "target")
    return {
        "target": {
            "pdes": str(target_row.get("pdes", "")), "name": name,
            "a": float(target_row["a"]), "e": float(target_row["e"]),
            "i": float(target_row["i"]),
        },
        # The mass the cruise starts with is what is left after the escape: liftoff mass minus the
        # propellant the spiral burned. Everything downstream reads this block, so it has to match
        # the mass the Sims-Flanagan solve started from.
        "vehicle": {"thrust_mN": rc.total_thrust_mN, "isp_s": rc.effective_isp,
                    "wet_mass_kg": rc.cruise_start_mass_kg},
        "sf": sf_block,
        "orbits": orbits,
    }


def _solve_return(rc, target_row, asteroid, earth, sf_block, on_progress: ProgressFn,
                  return_options: dict, return_x=None, light: bool = False) -> dict:
    """Solve, or rebuild, the loaded return leg and build its result block.

    The return leaves the asteroid once the stay is over, carrying what was collected, and meets
    Earth to deliver it to the chosen destination. ``return_options`` are the solver settings for
    the return leg, separate from the outbound's. Pass ``return_x``, a stored solution vector, to
    rebuild a swept return point in milliseconds instead of optimizing again. The block has the
    same ``sf`` and ``orbits`` as the outbound, plus the destination and the propellant account:
    what the return had left after the escape and the outbound cruise, and what it spent.
    """
    from prospector.launch import load_return_destinations
    dest = load_return_destinations()[rc.mission.return_destination]

    outbound_final = float(sf_block["final_mass_kg"])
    outbound_cruise_prop = float(sf_block["propellant_kg"])
    start_mass = rc.return_start_mass_kg(outbound_final)
    arr_outbound_mjd = float(sf_block["dep_mjd2000"]) + float(sf_block["tof_days"])
    stay_days = float(rc.mission.time_at_asteroid or 0.0)
    depart_mjd = arr_outbound_mjd + stay_days
    depart_date = lb.date_from_mjd2000(depart_mjd).date()
    depart_window = (depart_date, depart_date)

    lo, hi = (0.5, 1.0)

    def scaled(stage: str, frac: float, message: str) -> None:
        on_progress(stage, lo + (hi - lo) * frac, message)

    def _sf_progress(done: int, total: int) -> None:
        scaled("simsflanagan-return", 0.15 + 0.78 * done / total,
               f"optimizing the return trajectory ({done}/{total})")

    opts = dict(return_options or {})
    opts.setdefault("max_tof_days", rc.mission.return_max_tof_days)
    if return_x is not None:
        sol = sf.rebuild_return_for_config(
            rc, asteroid, return_x, start_mass_kg=start_mass, depart_window=depart_window,
            arrive_by=rc.mission.return_by, destination=dest, **opts)
    else:
        sol = sf.solve_return_for_config(
            rc, asteroid, start_mass_kg=start_mass, depart_window=depart_window,
            arrive_by=rc.mission.return_by, destination=dest, progress=_sf_progress, **opts)

    # The speeds are measured leaving the asteroid and arriving at Earth, but the colours stay tied
    # to the bodies as on the outbound, Earth blue and asteroid orange, so neither changes colour
    # between legs. The spacecraft therefore starts out on the orange orbit.
    sf_block_ret, orbits = _assemble_leg(
        sol, asteroid, earth, float(target_row["a"]), 1.0, scaled,
        blue_body=earth, blue_a_au=1.0, orange_body=asteroid, orange_a_au=float(target_row["a"]),
        light=light)
    scaled("done", 1.0, "return complete")

    available = rc.return_available_propellant_kg(outbound_cruise_prop)
    insertion_p = rc.insertion_propellant_kg(sol.final_mass_kg, dest)
    total_return_prop = float(sol.propellant_kg) + insertion_p
    return {
        "target": {"name": dest.name, "destination": rc.mission.return_destination},
        "sf": sf_block_ret,
        "orbits": orbits,
        "destination": rc.mission.return_destination,
        "destination_name": dest.name,
        "payload_kg": float(rc.mission.asteroid_payload_mass),
        "stay_days": stay_days,
        "insertion_dv_kms": rc.insertion_dv_kms(dest),
        "insertion_propellant_kg": insertion_p,
        "arrival_vinf_allow_kms": float(dest.arrival_vinf_kms),
        "available_propellant_kg": available,
        "cruise_propellant_kg": float(sol.propellant_kg),
        "total_return_propellant_kg": total_return_prop,
        # What gets delivered: arrival mass less the propellant the insertion burn uses.
        "delivered_mass_kg": float(sol.final_mass_kg) - insertion_p,
        "feasible_propellant": bool(total_return_prop <= available),
        "decision_vector": sol.decision_vector,
    }


def evaluate_candidate(rc, target_row, *, cells, sf_options: dict, return_options=None,
                       x0=None, workers=None, on_progress: ProgressFn | None = None,
                       light: bool = False, max_revs: int = 2) -> dict:
    """One vehicle against one target: the cruise from a spread of starts, the return when the
    mission plans one, and the result block the app reads.

    ``cells`` are the ``(dep_mjd2000, arr_mjd2000)`` starting guesses, each solved as its own
    start and the best kept (:func:`multistart.solve_cells`); ``x0`` is a warm-start vector for
    the first of them. ``sf_options`` are the outbound solver settings (``available_power_W`` is
    handed to the return too, which flies on the same array); ``return_options`` the return
    leg's. ``light`` skips the plot-only parts of the block (the sweep's mode). A return that
    cannot be solved is recorded under ``result["return"]["error"]`` rather than losing the
    outbound. Raises when no start produced a solution at all.
    """
    progress = on_progress or _noop
    sf_kwargs = dict(sf_options)
    for consumed in ("n_starts", "starts_workers", "return_options", "x0"):
        sf_kwargs.pop(consumed, None)
    return_options = dict(return_options or {})
    if "available_power_W" in sf_kwargs:
        return_options.setdefault("available_power_W", sf_kwargs["available_power_W"])
    row = _mission_elements(rc, target_row)
    target = lb.planet_from_row(row)
    earth = lb.earth_planet()
    scale = (0.0, 0.5) if rc.mission.return_trip else (0.0, 1.0)

    def scaled(stage: str, frac: float, message: str) -> None:
        progress(stage, scale[0] + (scale[1] - scale[0]) * frac, message)

    cells = [(float(d), float(a)) for d, a in cells]
    sol = solve_cells(rc, row, cells, sf_kwargs, workers, max_revs, scaled, x0=x0,
                      x0_cell=cells[0] if (x0 is not None and cells) else None)
    if sol is None:
        raise ValueError("no starting cell produced a solution (window too short to fly?)")
    result = _finish(rc, target, earth, row, None, progress, sf_kwargs, sol=sol,
                     progress_scale=scale, max_revs=max_revs, light=light)
    if rc.mission.return_trip:
        try:
            result["return"] = _solve_return(rc, row, target, earth, result["sf"], progress,
                                              return_options, light=light)
        except Exception as exc:
            result["return"] = {"error": str(exc)}
    return result


def solve_from_cell(rc, target_row, dep_mjd2000: float, arr_mjd2000: float, *,
                    on_progress: ProgressFn | None = None, max_revs: int = 2,
                    **sf_kwargs) -> dict:
    """Solve, starting from a chosen (departure, arrival) cell.

    The clicked cell is the first start and, with ``n_starts`` above 1, a spread of further cells
    over departure date and flight time joins it (:func:`multistart._diverse_seed_cells`), so a
    short clicked cell cannot trap the solve. ``x0``, a converged vector from the transfer grid,
    warm-starts the clicked cell. The result's ``lambert`` block summarises the cell that was
    picked.
    """
    progress = on_progress or _noop
    sf_kwargs = dict(sf_kwargs)
    return_options = dict(sf_kwargs.pop("return_options", None) or {})
    n_starts = int(sf_kwargs.pop("n_starts", 1) or 1)
    workers = sf_kwargs.pop("starts_workers", None)
    x0 = sf_kwargs.pop("x0", None)
    row = _mission_elements(rc, target_row)
    target = lb.planet_from_row(row)
    earth = lb.earth_planet()
    name = str(row.get("full_name") or row.get("pdes") or "target")

    progress("lambert", 0.05, "starting from the selected cell")
    seed = lb.lambert_transfer(earth, target, dep_mjd2000, arr_mjd2000,
                               target_name=name, max_revs=max_revs)
    if seed is None:
        raise ValueError("selected cell has no valid Lambert transfer (arrival before departure?)")
    if n_starts > 1:
        ceiling = (lb.mjd2000_from_date(rc.mission.arrive_by)
                   - lb.mjd2000_from_date(rc.departure_window[0]))
        max_tof = float(sf_kwargs["max_tof_days"]) if sf_kwargs.get("max_tof_days") else ceiling
        cells = _diverse_seed_cells(rc, seed.dep_mjd2000, seed.arr_mjd2000, n_starts,
                                    max_tof=max_tof, min_tof=float(sf_kwargs.get("min_tof_days", 150.0)))
    else:
        cells = [(float(seed.dep_mjd2000), float(seed.arr_mjd2000))]
    result = evaluate_candidate(rc, row, cells=cells, sf_options=sf_kwargs,
                                return_options=return_options, x0=x0, workers=workers,
                                on_progress=progress, max_revs=max_revs)
    result["lambert"] = {
        "best_dep_mjd2000": seed.dep_mjd2000, "best_arr_mjd2000": seed.arr_mjd2000,
        "best_dv_kms": seed.dv_kms, "best_tof_days": seed.tof_days,
        "vinf_dep_kms": seed.vinf_dep_kms, "vinf_arr_kms": seed.vinf_arr_kms,
        "c3_km2s2": seed.c3_km2s2,
    }
    return result


def result_from_decision(rc, target_row, x, *, on_progress: ProgressFn | None = None,
                         max_revs: int = 2, **sf_kwargs) -> dict:
    """Build the full result for a solution that was already found, without solving again.

    The v-infinity sweep runs a complete solve at every point but keeps only each point's solution
    vector. The winning point comes back through here and rebuilds, in milliseconds, into the same
    result an interactive run produces, including the cards and the trajectory. ``sf_kwargs`` has
    to be the settings the point was solved with, since the problem is rebuilt from them. A Lambert
    transfer at the solved cell fills in the comparison block, falling back to the solution's own
    departure and arrival v-infinity if that cell has no Lambert solution.
    """
    progress = on_progress or _noop
    # The return is rebuilt from its own stored solution vector and its own solver settings, so
    # take both out before the outbound rebuild claims the rest of sf_kwargs. Each leg's
    # per-segment thrust limits travel the same way, stored next to its vector, so the rebuilt
    # problems carry the limits they were solved under.
    return_options = sf_kwargs.pop("return_options", None)
    return_x = sf_kwargs.pop("return_decision_vector", None)
    return_terms = {k: sf_kwargs.pop(f"return_{k}")
                    for k in ("seg_caps", "thrust_N", "isp_s", "seg_isp_s")
                    if sf_kwargs.get(f"return_{k}") is not None}
    for k in ("return_seg_caps", "return_thrust_N", "return_isp_s", "return_seg_isp_s"):
        sf_kwargs.pop(k, None)
    if return_x is not None and return_terms:
        return_options = dict(return_options or {})
        return_options.update(return_terms)
    target_row = _mission_elements(rc, target_row)
    target = lb.planet_from_row(target_row)
    earth = lb.earth_planet()
    name = str(target_row.get("full_name") or target_row.get("pdes") or "target")

    progress("rebuild", 0.1, "rebuilding the stored solution")
    sol = sf.rebuild_for_config(rc, target, x, **sf_kwargs)
    scale = (0.0, 0.5) if (rc.mission.return_trip and return_x is not None) else (0.0, 1.0)
    result = _finish(rc, target, earth, target_row, seed=None, on_progress=progress,
                     sf_kwargs=sf_kwargs, sol=sol, progress_scale=scale)
    if rc.mission.return_trip and return_x is not None:
        try:
            result["return"] = _solve_return(rc, target_row, target, earth, result["sf"],
                                              progress, return_options or {}, return_x=return_x)
        except Exception as exc:
            result["return"] = {"error": str(exc)}

    arr_mjd2000 = sol.dep_mjd2000 + sol.tof_days
    seed = lb.lambert_transfer(earth, target, sol.dep_mjd2000, arr_mjd2000,
                               target_name=name, max_revs=max_revs)
    if seed is not None:
        result["lambert"] = {
            "best_dep_mjd2000": seed.dep_mjd2000, "best_arr_mjd2000": seed.arr_mjd2000,
            "best_dv_kms": seed.dv_kms, "best_tof_days": seed.tof_days,
            "vinf_dep_kms": seed.vinf_dep_kms, "vinf_arr_kms": seed.vinf_arr_kms,
            "c3_km2s2": seed.c3_km2s2,
        }
    else:
        sfb = result["sf"]
        result["lambert"] = {
            "best_dep_mjd2000": sol.dep_mjd2000, "best_arr_mjd2000": arr_mjd2000,
            "best_dv_kms": sfb["vinf_dep_kms"] + sfb["vinf_arr_kms"],
            "best_tof_days": sol.tof_days,
            "vinf_dep_kms": sfb["vinf_dep_kms"], "vinf_arr_kms": sfb["vinf_arr_kms"],
            "c3_km2s2": sfb["vinf_dep_kms"] ** 2,
        }
    return result

