"""Solving one target, start to finish.

    arcs        turn a solved trajectory into something drawable and checkable
    grid        the grid of real trajectories over departure date and flight time
    multistart  several different starting guesses at once, so one bad guess cannot decide
                the answer
    solve       the entry points: solve one cell, a whole grid, or rebuild a stored one

Modules are split by what they are for, not by the order they run in. ``arcs`` and ``multistart``
are used by every entry point in ``solve``.

Every public name is re-exported here, so callers write ``pipeline.solve_from_cell(...)``.

Pure and importable: no UI framework, no subprocess. ``worker.py`` is the detached job wrapper.
"""
from __future__ import annotations

from prospector.trades.pipeline.arcs import ProgressFn  # noqa: F401
from prospector.trades.pipeline.grid import (  # noqa: F401
    FAST_RESTARTS,
    GRID_NSEG,
    GRID_RESTARTS,
    best_per_flight_time,
    cheapest_per_flight_time,
    departure_spread,
    grid_axes,
    lowthrust_grid,
    polish_best_per_flight_time,
)
from prospector.trades.pipeline.multistart import (  # noqa: F401
    _better_sol,
    _diverse_seed_cells,
    _multistart_outbound,
    _outbound_start,
)
from prospector.trades.pipeline.solve import (  # noqa: F401
    result_from_decision,
    solve_from_cell,
)

__all__ = [
    "FAST_RESTARTS",
    "GRID_NSEG",
    "GRID_RESTARTS",
    "ProgressFn",
    "cheapest_per_flight_time",
    "departure_spread",
    "grid_axes",
    "lowthrust_grid",
    "best_per_flight_time",
    "polish_best_per_flight_time",
    "result_from_decision",
    "solve_from_cell",
]
