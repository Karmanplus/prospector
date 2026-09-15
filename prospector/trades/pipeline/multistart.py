"""Trying several different starting guesses at once.

The optimizer only searches near wherever it starts, so a single clicked porkchop cell gets you one
answer -- usually the short trip that looks cheap when burns are treated as instant. The best
low-thrust trip is often a longer, more patient one that a short guess never finds. So these
helpers spread the starting guesses over departure dates and flight times, which produces genuinely
different kinds of trajectory, and keep whichever comes out best.

Every start runs the same physics under the same limits: launch window, arrival deadline, maximum
flight time, thrust cap and departure cone are all unchanged. Only the starting guess differs, and
the clicked cell is always one of them, so this can only match or beat the single-guess answer.

Starts run in separate processes rather than threads, because PyKEP and pygmo hold the GIL.
"""
from __future__ import annotations

import numpy as np

from prospector.solvers import lambert as lb
from prospector.solvers import transfer


def _better_sol(cand, best) -> bool:
    """Compare two solutions: one that converged always beats one that did not; between two
    that converged, more mass left over wins; between two that did not, the smaller miss wins.
    Same ordering as ``simsflanagan._better``, for whole solution objects."""
    if cand is None:
        return False
    if best is None:
        return True
    if bool(cand.feasible) != bool(best.feasible):
        return bool(cand.feasible)
    if cand.feasible:
        # A settled answer respects the ceilings its own path implies; an unsettled one still
        # fires somewhere the array cannot power it and reads cheaper than it is. Settled first.
        cs = bool(getattr(cand, "refresh_settled", True))
        bs = bool(getattr(best, "refresh_settled", True))
        if cs != bs:
            return cs
        return cand.final_mass_kg > best.final_mass_kg
    return cand.mismatch < best.mismatch


def _diverse_seed_cells(rc, clicked_dep: float, clicked_arr: float, n: int,
                        max_tof: float | None = None, min_tof: float = 150.0) -> list:
    """``n`` different ``(dep_mjd2000, arr_mjd2000)`` starting cells spread over departure date
    and flight time, which gives genuinely different kinds of trajectory.

    A clicked porkchop cell is usually the short trip that looks cheap when burns are treated as
    instant, and the optimizer only searches near where it starts. The best low-thrust trip is
    often a longer, more patient one that a short guess never reaches, so spreading the starts lets
    the search cover both. The clicked cell always comes first.

    Flight times stay within ``[min_tof, max_tof]`` and never run past the arrival deadline, so the
    starts respect the user's maximum flight time; ``max_tof=None`` spreads to the deadline.
    """
    start = lb.mjd2000_from_date(rc.departure_window[0])
    end = lb.mjd2000_from_date(rc.departure_window[1])
    arrive = lb.mjd2000_from_date(rc.mission.arrive_by)
    lo = max(130.0, float(min_tof))

    def _ceil(dep):
        room = arrive - dep
        return room if max_tof is None else min(float(max_tof), room)

    def _clamp(dep, arr):
        tof = min(max(arr - dep, lo), max(lo, _ceil(dep)))
        return (float(dep), float(dep + tof))

    cells = [_clamp(clicked_dep, clicked_arr)]              # the clicked cell, clamped to the cap
    if n > 1:
        n_dep = max(1, min(3, int(round(n ** 0.5))))
        per = max(2, -(-n // n_dep))                # ceil(n / n_dep), >= 2 flight times per date
        for dep in np.linspace(start, end, n_dep):
            hi = _ceil(dep)
            if hi <= lo:                                   # no room for a trip within the cap
                continue
            for tof in np.linspace(lo, hi, per):
                cells.append((float(dep), float(dep + tof)))
    seen, out = set(), []
    for dep, arr in cells:
        key = (round(dep), round(arr))
        if key in seen:
            continue
        seen.add(key)
        out.append((dep, arr))
        if len(out) >= n:
            break
    return out


def solve_cells(rc, target_row, cells, sf_kwargs, workers, max_revs, scaled, *, x0=None,
                x0_cell=None):
    """Solve the outbound leg from every ``(dep_mjd2000, arr_mjd2000)`` cell in ``cells`` and
    keep the best (:func:`_better_sol`): converged first, then settled, then heaviest arrival.

    The one search both the interactive refine and the design sweep run. Each start is a fresh
    :func:`simsflanagan.solve_for_config` from a two-burn guess at its cell; ``x0`` is a warm-start
    vector that belongs to ``x0_cell`` alone. Handed to every start it used to override their own
    seeds, so nine starts were the one start nine times over. ``workers`` above 1 spreads the
    starts across processes (fork; see :func:`_multistart_outbound`). None when every start raised.
    """
    import concurrent.futures as cf
    import os

    ms_kwargs = dict(sf_kwargs)
    ms_kwargs.pop("progress", None)                        # a callable can't cross to a child
    ms_kwargs.pop("x0", None)
    config_json = rc.model_dump(mode="json")
    row = dict(target_row)

    def _kwargs_for(dep, arr):
        mine = (x0 is not None and x0_cell is not None
                and abs(dep - float(x0_cell[0])) < 1e-6 and abs(arr - float(x0_cell[1])) < 1e-6)
        return {**ms_kwargs, "x0": x0} if mine else ms_kwargs

    jobs_args = [(config_json, row, float(dep), float(arr), int(max_revs), _kwargs_for(dep, arr))
                 for dep, arr in cells]
    n = len(jobs_args)
    if n == 0:
        return None
    if not workers or int(workers) <= 0:
        workers = max(1, min(n, (os.cpu_count() or 2) - 1))
    workers = max(1, int(workers))

    best, done = None, 0

    def _tick(sol) -> None:
        nonlocal best, done
        done += 1
        if _better_sol(sol, best):
            best = sol
        scaled("simsflanagan", 0.15 + 0.78 * done / n, f"{done}/{n} starts")

    if workers <= 1:
        for a in jobs_args:
            _tick(_outbound_start(a))
    else:
        # Forking is right here: this runs inside the detached ``worker.py`` process, which holds
        # no listening socket for a child to inherit, and forking skips re-importing the solver
        # stack in every worker. The flight-time study spawns instead because it runs inside the
        # app. The deciding factor is whether the parent owns a socket. A forked
        # child that inherits the app's socket keeps the port bound after the app exits. ``with``
        # waits for the pool to shut down, so a solve that raises leaves no workers.
        with cf.ProcessPoolExecutor(max_workers=workers) as pool:
            for sol in pool.map(_outbound_start, jobs_args):
                _tick(sol)
    return best


def _outbound_start(args):
    """One start, in a worker process. Rebuilds the config and target from plain data (the pool
    passes pickled arguments), starts from its own ``(dep, arr)`` cell, and returns the solution,
    or None if that start raised."""
    config_json, target_row, dep_mjd, arr_mjd, max_revs, sf_kwargs = args
    from prospector.config import ResolvedConfig
    try:
        rc = ResolvedConfig.model_validate(config_json)
        target = lb.planet_from_row(target_row)
        earth = lb.earth_planet()
        name = str(target_row.get("full_name") or target_row.get("pdes") or "target")
        seed = lb.lambert_transfer(earth, target, dep_mjd, arr_mjd,
                                   target_name=name, max_revs=max_revs)
        return transfer.for_config(rc).solve_for_config(rc, target, seed=seed, **sf_kwargs)
    except Exception:
        return None


def _multistart_outbound(rc, target_row, seed, sf_kwargs, n_starts, workers, max_revs, scaled):
    """Solve the outbound trip from a spread of starting guesses at once, and keep the best.

    The guesses are spread over departure dates and flight times by :func:`_diverse_seed_cells` and
    run in separate processes, since pykep and pygmo hold the GIL. The physics and the limits are
    those of a normal solve: launch window, ``arrive_by`` deadline, ``max_tof_days`` cap, thrust
    cap and departure cone are all unchanged. Only the starting guess varies, and the guesses stay
    inside the flight-time cap. The clicked cell is one of them, so this can only match or beat the
    single-guess answer.
    """

    ms_kwargs = dict(sf_kwargs)
    ms_kwargs.pop("progress", None)                        # a callable can't cross to a child
    # Respect the user's maximum flight time; only spread to the deadline if none was given.
    ceiling = lb.mjd2000_from_date(rc.mission.arrive_by) - lb.mjd2000_from_date(rc.departure_window[0])
    max_tof = float(ms_kwargs["max_tof_days"]) if ms_kwargs.get("max_tof_days") else float(ceiling)
    min_tof = float(ms_kwargs.get("min_tof_days", 150.0))
    cells = _diverse_seed_cells(rc, seed.dep_mjd2000, seed.arr_mjd2000, max(1, int(n_starts)),
                                max_tof=max_tof, min_tof=min_tof)
    x0 = ms_kwargs.pop("x0", None)
    best = solve_cells(rc, target_row, cells, ms_kwargs, workers, max_revs, scaled, x0=x0,
                       x0_cell=(float(seed.dep_mjd2000), float(seed.arr_mjd2000)))
    if best is None:
        # Every start raised, which happens with a window too short to fly. Fall back to a single
        # solve from the clicked cell, so the failure looks the same as it would if this code were
        # not here.
        best = transfer.for_config(rc).solve_for_config(rc, lb.planet_from_row(target_row),
                                                        seed=seed, **sf_kwargs)
    return best

