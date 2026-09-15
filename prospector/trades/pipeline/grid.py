"""A porkchop grid over departure date and flight time, where every cell is a real solve.

Each cell is a full Sims-Flanagan run. That would normally be too slow to do across a grid, and two
things make it practical: PyKEP computes its own gradients, and each cell is warm-started from
whatever its neighbour converged to.

Cells are solved from the longest flight time down to the shortest. Long trips have propellant to
spare and converge easily, while short ones sit close to the limit of what the vehicle can do, so
working downward gives the difficult cells a solved neighbour to start from. Flight times therefore
have to be done in order, but the departure dates within a column are independent and run in
parallel. Widening the departure axis costs almost nothing; lengthening the flight-time axis costs
wall clock.

Delta-v per cell lands within about 1% of a converged reference, or 15% in the worst case, which is
close enough to compare cells against each other. Anything picked for real gets solved again at
full effort.

A cell that fails to converge means the search found no trajectory at this effort level, which is a
weaker statement than there being no trajectory at all. Callers should report it as "not found";
treating the two alike loses reachable targets.

Units: MJD2000 days, km/s, kg.
"""
from __future__ import annotations

import concurrent.futures as cf
import math
import os

import numpy as np

from prospector.solvers import lambert as lb
from prospector.solvers import transfer
from prospector.trades.pipeline.arcs import _mission_elements


def jsonable(terms: dict) -> dict:
    """Solver settings as plain JSON numbers, so they survive a write and read back."""
    out = {}
    for k, v in terms.items():
        if v is None or isinstance(v, (bool, int, str)):
            out[k] = v
        elif isinstance(v, float):
            out[k] = float(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [float(x) for x in v]
        else:
            out[k] = float(v)
    return out

# Restarts per cell. A single restart tends to leave a cell sitting at its neighbour's answer.
# Measured on Apophis at a 400-day flight time, warm-started from a solution ten days away: 3.2445
# km/s at one restart, 3.0194 from scratch, 3.0187 at four, 2.9925 at eight. Gains past four are
# small, so four is the default when the grid quotes delta-v. The feasibility-only pass uses one.
GRID_RESTARTS = 4
# A two-leg restart is a whole new impulsive geometry descended twice, so a flyby cell at the
# direct grid's four restarts costs about twice a live direct cell and dead cells cost the same.
# Two keeps a 16x16 flyby grid within about twice the direct grid's time.
FLYBY_GRID_RESTARTS = 2
FAST_RESTARTS = 1
# Segments per cell, held the same across the grid. A stored solution cannot be re-sampled to a
# different segment count (see ``simsflanagan._adapt_decision_vector``), so the grid and anything
# warm-started from it have to agree. Accuracy levels off past 12.
GRID_NSEG = 12
# How far a cell's solved date may sit from the date it was pinned to. A rebuild has to use
# the same window or a perfectly good vector reads as outside the problem's bounds.
CELL_TOL_DAYS = 0.75


def grid_axes(rc, *, n_dep: int = 8, n_tof: int = 8, min_tof_days: float = 150.0):
    """The two axes: departure dates across the launch window, and flight times.

    Flight times come back longest first, which is the order they are solved in. The longest is
    measured from the FIRST departure date: the earliest liftoff can fly the longest trip the
    deadline allows, and those long, patient transfers are the cheap ones. Measuring from the last
    departure, as this once did, kept every cell inside the deadline but cut the axis short by the
    width of the launch window, and on Apophis that hid the transfers the design sweep found at
    0.3 km/s less. Cells a later departure cannot fly to the deadline are not solved
    (:func:`lowthrust_grid` marks them ``past_deadline``).
    """
    dep0 = lb.mjd2000_from_date(rc.departure_window[0])
    dep1 = lb.mjd2000_from_date(rc.departure_window[1])
    arrive = lb.mjd2000_from_date(rc.mission.arrive_by)
    # Whole days, set one day inside each end of the window. The solver takes a departure bound as
    # a date plus a tolerance, so a fractional date has to be rounded, and the rounding widens the
    # tolerance enough to push the bound back outside the launch window. A day of slack at each end
    # absorbs that. Duplicate dates collapse if the window is shorter than the requested number of
    # columns.
    lo_day, hi_day = math.ceil(dep0) + 1.0, math.floor(dep1) - 1.0
    if hi_day < lo_day:
        lo_day = hi_day = round(0.5 * (dep0 + dep1))
    dep = np.unique(np.round(np.linspace(lo_day, hi_day, max(1, int(n_dep)))))
    ceiling = arrive - float(dep[0])
    lo = max(1.0, float(min_tof_days))
    if ceiling <= lo:
        # Even the shortest trip from the first departure date misses the deadline. Returning a
        # single flight time at the limit lets the caller draw an empty grid instead of raising.
        return dep, np.array([max(lo, ceiling)])
    tof = np.linspace(ceiling, lo, max(1, int(n_tof)))
    return dep, tof


def leg_terms(sol) -> dict:
    """The thrust terms a solution was found under (``seg_caps``, ``thrust_N``, ``isp_s``,
    ``seg_isp_s``), JSON-safe, for ``rebuild_for_config``. A vector alone does not fix the
    trajectory: rebuilt without them an Apophis cell's mismatch went 1.5e-5 to 3.6e-3."""
    caps = getattr(sol, "seg_caps", None)
    seg_isp = getattr(sol, "seg_isp_s", None)
    isp = sol.isp_s
    # Two-leg: per-leg arrays stored flat (split at leg one's nseg on rebuild), Isps as the pair.

    def flat(parts):
        return np.concatenate([np.asarray(a, float).ravel() for a in parts]).tolist()

    return {"seg_caps": None if caps is None else flat(caps if isinstance(caps, tuple) else [caps]),
            "thrust_N": float(sol.thrust_N),
            "isp_s": [float(v) for v in isp] if isinstance(isp, (tuple, list)) else float(isp),
            "seg_isp_s": (None if seg_isp is None
                          else flat(seg_isp if isinstance(seg_isp, tuple) else [seg_isp]))}


def _cell(args):
    """Solve one cell of the grid, in a worker process.

    The pool pickles its arguments, so the config and target are rebuilt here from plain data and
    the result goes back as a plain dict. A cell that raises comes back as None, meaning the solve
    failed; it says nothing about whether a trajectory exists.
    """
    config_json, target_row, dep, tof, x0, kwargs = args
    from prospector.config import ResolvedConfig
    try:
        rc = ResolvedConfig.model_validate(config_json)
        target = lb.planet_from_row(target_row)
        leg = transfer.for_config(rc)
        sol = leg.solve_cell_for_config(rc, target, dep_mjd2000=float(dep), tof_days=float(tof),
                                       x0=None if x0 is None else np.asarray(x0, float), **kwargs)
        return {"dep_mjd2000": float(sol.dep_mjd2000), "tof_days": float(sol.tof_days),
                "dv_kms": float(sol.dv_kms), "final_mass_kg": float(sol.final_mass_kg),
                "feasible": bool(sol.feasible), "mismatch": float(sol.mismatch),
                "settled": bool(getattr(sol, "refresh_settled", True)),
                "decision_vector": np.asarray(sol.decision_vector, float).tolist(),
                "leg_terms": leg_terms(sol)}
    except (TypeError, AttributeError):
        # These come from a bad call -- a wrong argument or a misspelled name -- rather than from a
        # difficult cell. Every cell would fail the same way, so ignoring them empties the grid and
        # makes a reachable target look unreachable.
        raise
    except Exception:  # noqa: BLE001 -- one hard cell costs a cell, never the grid
        return None


def lowthrust_grid(rc, target_row, *, n_dep: int = 8, n_tof: int = 8,
                   nseg: int = GRID_NSEG, restarts: int = GRID_RESTARTS,
                   min_tof_days: float = 150.0, workers: int | None = None,
                   available_power_W: float | None = None, max_revs: int = 2,
                   progress=None, **sf_kwargs) -> dict:
    """Solve every cell of a (departure date x flight time) grid.

    Returns the two axes, the numbers for each cell, and each cell's solution. Keeping the
    solutions is what makes clicking a cell instant later: the full trajectory rebuilds from one in
    milliseconds, and a proper re-solve can start from it rather than from scratch.

    ``restarts`` is how many times each cell is re-tried from a different starting point. At
    ``FAST_RESTARTS`` the grid only says which cells work, in a few seconds; at ``GRID_RESTARTS``
    or above its delta-v is worth quoting too.

    The longest flight time is solved first, from a Lambert guess. Every shorter one starts from
    the answer the same departure date got at the flight time before it, so each departure date
    forms its own chain and the chains do not interfere. Only a converged answer is passed along,
    so a cell that fails costs one cell.

    ``sf_kwargs`` are the solver settings every cell runs under: departure and arrival v-infinity,
    duty-cycle cap, how far the departure date may move. Pass the same ones the main solve uses.
    Falling back to :func:`simsflanagan.solve`'s defaults makes the grid solve at a 1 km/s
    departure excess whatever the config says, so the grid describes a different mission than the
    trajectory flown from it, and its stored solutions cannot be rebuilt under the config's own
    settings. They are echoed back in the result as ``sf_kwargs`` so a rebuild uses what the cell
    was actually solved with rather than whatever the controls read now.
    """
    dep, tof = grid_axes(rc, n_dep=n_dep, n_tof=n_tof, min_tof_days=min_tof_days)
    n_d, n_t = dep.size, tof.size
    shape = (n_d, n_t)
    dv = np.full(shape, np.nan)
    mass = np.full(shape, np.nan)
    mismatch = np.full(shape, np.nan)
    feasible = np.zeros(shape, bool)
    # Whether each cell's answer respects the thrust ceilings its own path implies. A cell that
    # did not settle still fires where the array cannot power it and reads cheaper than it is.
    settled = np.ones(shape, bool)
    # Cells whose departure cannot reach the deadline at that flight time. Not solved: there is no
    # trajectory to look for, and the porkchop shows them as blank rather than "not found".
    arrive_mjd = lb.mjd2000_from_date(rc.mission.arrive_by)
    past_deadline = np.zeros(shape, bool)
    solved_dep = np.full(shape, np.nan)
    solved_tof = np.full(shape, np.nan)
    vectors: list[list] = [[None] * n_t for _ in range(n_d)]
    # The thrust terms each vector was solved under, kept beside it (see leg_terms).
    terms: list[list] = [[None] * n_t for _ in range(n_d)]

    if rc.mission.gravity_assist:
        restarts = min(int(restarts), FLYBY_GRID_RESTARTS)
    kwargs = {**sf_kwargs, "nseg": int(nseg), "restarts": int(restarts)}
    if available_power_W is not None:
        kwargs["available_power_W"] = float(available_power_W)
    # The settings a rebuild has to reproduce, kept separate from the grid's own shape so a caller
    # can pass them straight to ``result_from_decision`` without filtering.
    cell_terms = {k: v for k, v in kwargs.items() if k != "restarts"}
    config_json = rc.model_dump(mode="json")
    # Refresh the orbit once, up front, for the arrival date, and keep the row that was used. Cells
    # have to be solved against the orbit the body actually flies, and propagating the catalogue
    # orbit forward breaks across a close approach: Apophis's 2029 flyby moves it from 0.92 to 1.10
    # AU. Rebuilding a cell refreshes too, so a grid that skipped this would hand its solutions to
    # a problem built on a different orbit. The lookup is cached per body and date, so doing it
    # here also makes every later cell click a cache hit instead of a network round trip.
    row = dict(_mission_elements(rc, dict(target_row)))
    name = str(row.get("full_name") or row.get("pdes") or "target")

    if not workers or int(workers) <= 0:
        workers = max(1, min(n_d, (os.cpu_count() or 2) - 1))
    workers = max(1, int(workers))

    # The last answer that converged, per departure date. None until that date works once, after
    # which the next flight time along stops paying for a start from scratch.
    carried: list = [None] * n_d
    done = 0
    total = n_d * n_t

    for j in range(n_t):
        t = float(tof[j])
        args = []
        live: list[int] = []
        for i in range(n_d):
            if float(dep[i]) + t > arrive_mjd + CELL_TOL_DAYS:
                past_deadline[i, j] = True
                continue
            live.append(i)
            x0 = carried[i]
            if x0 is None:
                # Nothing to start from yet, so use a Lambert transfer at this cell. Its dates and
                # in-plane energy are a usable starting point even though its delta-v is not, which
                # is all a starting point has to be.
                try:
                    earth = lb.earth_planet()
                    seed = lb.lambert_transfer(earth, lb.planet_from_row(row), float(dep[i]),
                                               float(dep[i]) + t, target_name=name,
                                               max_revs=int(max_revs))
                except Exception:  # noqa: BLE001 -- no guess here; the cell starts from its bounds
                    seed = None
                args.append((config_json, row, float(dep[i]), t, None,
                             {**kwargs, "seed": seed} if seed is not None else dict(kwargs)))
            else:
                args.append((config_json, row, float(dep[i]), t,
                             np.asarray(x0, float).tolist(), dict(kwargs)))

        done += n_d - len(live)
        if not args:
            if progress is not None:
                progress(done, total)
            continue
        if workers <= 1:
            results = [_cell(a) for a in args]
        else:
            # Forked workers are safe here because this runs inside the detached grid worker, which
            # holds no listening socket for a child to inherit. Forking also avoids re-importing
            # the solver stack for every cell.
            with cf.ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_cell, args))

        for i, res in zip(live, results):
            done += 1
            if res is None:
                continue
            dv[i, j] = res["dv_kms"]
            mass[i, j] = res["final_mass_kg"]
            mismatch[i, j] = res["mismatch"]
            feasible[i, j] = res["feasible"]
            settled[i, j] = bool(res.get("settled", True))
            solved_dep[i, j] = res["dep_mjd2000"]
            solved_tof[i, j] = res["tof_days"]
            vectors[i][j] = res["decision_vector"]
            terms[i][j] = res.get("leg_terms")
            if res["feasible"]:
                carried[i] = res["decision_vector"]      # only a converged cell starts the next
        if progress is not None:
            progress(done, total)

    return {
        "dep_mjd2000": dep.tolist(),
        "tof_days": tof.tolist(),
        "dv_kms": dv.tolist(),
        "final_mass_kg": mass.tolist(),
        "mismatch": mismatch.tolist(),
        "feasible": feasible.tolist(),
        "settled": settled.tolist(),
        "past_deadline": past_deadline.tolist(),
        "solved_dep_mjd2000": solved_dep.tolist(),
        "solved_tof_days": solved_tof.tolist(),
        "decision_vectors": vectors,
        "leg_terms": terms,
        "nseg": int(nseg), "restarts": int(restarts),
        "sf_kwargs": jsonable(cell_terms),
        # The orbit every cell was solved against, so a rebuild uses the same one rather than
        # working out a new one.
        "target_row": {k: (v.isoformat() if hasattr(v, "isoformat") else v)
                       for k, v in row.items()},
        # Rebuilding a cell needs the narrow date range the cell was solved in, not the whole
        # mission window the config carries: against the wider range, a cell whose departure landed
        # just outside a rounded window edge gets rejected.
        "cell_tol_days": CELL_TOL_DAYS,
        "target_name": name, "target_pdes": str(row.get("pdes") or ""),
        "n_feasible": int(feasible.sum()), "n_cells": int(total),
    }


def cheapest_per_flight_time(grid: dict) -> list[dict]:
    """The cheapest cell at each flight time: best delta-v against how long the trip takes.

    This is why there is no separate flight-time study -- the curve is just the smallest value in
    each column of a grid that is already solved. A flight time where nothing converged reports
    ``None`` instead of being left out, so a gap in the curve stays visible as a gap.
    """
    dv = np.asarray(grid["dv_kms"], float)
    ok = np.asarray(grid["feasible"], bool)
    # Prefer answers that settled on their own ceilings; an unsettled cell reads cheaper than it
    # is, so it stands in only where nothing in its column settled.
    settled = np.asarray(grid.get("settled", np.ones_like(ok)), bool)
    dep = np.asarray(grid["dep_mjd2000"], float)
    tof = np.asarray(grid["tof_days"], float)
    mass = np.asarray(grid["final_mass_kg"], float)
    out = []
    for j in range(tof.size):
        pick = (ok[:, j] & settled[:, j]) if np.any(ok[:, j] & settled[:, j]) else ok[:, j]
        col = np.where(pick, dv[:, j], np.nan)
        if not np.isfinite(col).any():
            out.append({"tof_days": float(tof[j]), "dv_kms": None, "dep_mjd2000": None,
                        "final_mass_kg": None, "i_dep": None})
            continue
        i = int(np.nanargmin(col))
        out.append({"tof_days": float(tof[j]), "dv_kms": float(dv[i, j]),
                    "dep_mjd2000": float(dep[i]), "final_mass_kg": float(mass[i, j]),
                    "i_dep": i})
    return sorted(out, key=lambda p: p["tof_days"])


def _polish(args):
    """Solve one flight time with the departure date left free, in a worker process. Returns a
    plain dict, or None if it raised."""
    config_json, target_row, tof, x0, kwargs = args
    from prospector.config import ResolvedConfig
    try:
        rc = ResolvedConfig.model_validate(config_json)
        target = lb.planet_from_row(target_row)
        sol = transfer.for_config(rc).solve_for_config(rc, target, x0=np.asarray(x0, float),
                                  min_tof_days=max(1.0, float(tof) - CELL_TOL_DAYS),
                                  max_tof_days=float(tof) + CELL_TOL_DAYS, **kwargs)
        # Key on the flight time that was asked for, not the one that came back: the solved value
        # drifts within the tolerance the cell was given, and the curve is indexed by the request.
        # Keying on the solved value matches nothing and drops every point silently.
        return {"tof_days": float(tof), "solved_tof_days": float(sol.tof_days),
                "dep_mjd2000": float(sol.dep_mjd2000),
                "dv_kms": float(sol.dv_kms), "final_mass_kg": float(sol.final_mass_kg),
                "feasible": bool(sol.feasible), "mismatch": float(sol.mismatch),
                "decision_vector": np.asarray(sol.decision_vector, float).tolist(),
                "leg_terms": leg_terms(sol)}
    except Exception:  # noqa: BLE001 -- a polish that fails leaves the grid's own cell standing
        return None


def polish_best_per_flight_time(rc, target_row, grid: dict, *, restarts: int = GRID_RESTARTS,
                    workers: int | None = None,
                    available_power_W: float | None = None) -> list[dict]:
    """Re-solve each flight time with the departure date left free, starting from that column's
    best cell.

    The grid only tries a handful of departure dates; this searches between them. Neither one
    replaces the other. Trying every date cannot miss the best of the dates it tried, and where
    cost changes sharply from one departure date to the next, a search does miss it -- on Apophis a
    free-departure sweep came out up to 0.33 km/s worse than the grid's own best cells, and still
    0.14 km/s worse with eight times as many restarts. Within a single column, though, the search
    wins: it beat the grid on every 2008 EV5 column that converged (by 0.009-0.043 km/s) and on
    four of five Mars columns (up to 0.053 km/s). So the grid picks roughly when to leave and this
    sharpens the number.

    It costs one solve per flight time, around 8 against the grid's 64, and each one starts from an
    answer that already converged. A column where nothing converged is skipped: there is nothing to
    start from, and solving it from scratch would put a differently-obtained number on the same
    curve.

    Every solve runs under the terms the grid stored, so the only thing freed is the departure
    date. Taking the solver's own defaults instead hands the polish a 1 km/s departure excess the
    cells never had, and the free speed then reads as an improvement: measured on Apophis, columns
    came back up to 1.19 km/s under their own cells, against the 0.05 a real search wins.
    """
    kwargs = {**(grid.get("sf_kwargs") or {}), "restarts": int(restarts)}
    kwargs["nseg"] = int(grid.get("nseg") or kwargs.get("nseg") or GRID_NSEG)
    if available_power_W is not None:
        kwargs["available_power_W"] = float(available_power_W)
    config_json = rc.model_dump(mode="json")
    # The orbit the cells were solved against, re-osculated at the arrival era; a polish against
    # the catalogue row lands on a vector that misses by 4e-3 when a click rebuilds it against the
    # stored row (Didymos, three years of drift).
    row = dict(_mission_elements(rc, dict(target_row)))
    vectors = grid["decision_vectors"]

    args = []
    for p in cheapest_per_flight_time(grid):
        i = p["i_dep"]
        if i is None:
            continue
        j = grid["tof_days"].index(p["tof_days"])
        x0 = vectors[i][j]
        if x0 is None:
            continue
        args.append((config_json, row, float(p["tof_days"]), x0, dict(kwargs)))
    if not args:
        return []

    if not workers or int(workers) <= 0:
        workers = max(1, min(len(args), (os.cpu_count() or 2) - 1))
    workers = max(1, int(workers))
    if workers <= 1:
        out = [_polish(a) for a in args]
    else:
        with cf.ProcessPoolExecutor(max_workers=workers) as pool:
            out = list(pool.map(_polish, args))
    return [r for r in out if r is not None]


def best_per_flight_time(grid: dict, polished: list[dict]) -> list[dict]:
    """The best delta-v at each flight time, taking the grid cell or its re-solve, whichever won.

    Better means converged first, then cheaper. A re-solve counts only where it converged and
    improved, so running it can only move the curve down.
    """
    by_tof = {round(float(p["tof_days"]), 3): p for p in polished if p.get("feasible")}
    out = []
    for p in cheapest_per_flight_time(grid):
        best = dict(p)
        best["source"] = "grid"
        cand = by_tof.get(round(float(p["tof_days"]), 3))
        if cand is not None and (best["dv_kms"] is None or cand["dv_kms"] < best["dv_kms"]):
            best = {"tof_days": float(p["tof_days"]), "dv_kms": float(cand["dv_kms"]),
                    "dep_mjd2000": float(cand["dep_mjd2000"]),
                    "final_mass_kg": float(cand["final_mass_kg"]),
                    "i_dep": p["i_dep"], "source": "polish"}
        out.append(best)
    return out


def departure_spread(grid: dict) -> list[dict]:
    """How much the cost swings across departure dates at each flight time.

    The best-per-flight-time curve only shows the cheapest cell in each column, which hides how
    much that number depends on leaving on the right day. On some targets that is the biggest
    effect in the whole trade: the spread within one column runs 3.7-13.4% on 2008 EV5, up to 104%
    on Apophis and 140-688% on Mars.
    """
    dv = np.asarray(grid["dv_kms"], float)
    ok = np.asarray(grid["feasible"], bool)
    tof = np.asarray(grid["tof_days"], float)
    out = []
    for j in range(tof.size):
        col = dv[:, j][ok[:, j]]
        col = col[np.isfinite(col)]
        if col.size < 2:
            out.append({"tof_days": float(tof[j]), "lo_kms": None, "hi_kms": None,
                        "spread_pct": None, "n": int(col.size)})
            continue
        lo, hi = float(col.min()), float(col.max())
        out.append({"tof_days": float(tof[j]), "lo_kms": lo, "hi_kms": hi,
                    "spread_pct": 100.0 * (hi / lo - 1.0) if lo > 0 else None,
                    "n": int(col.size)})
    return sorted(out, key=lambda p: p["tof_days"])
