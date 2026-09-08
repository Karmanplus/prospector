"""The vehicle design-space search: which vehicle can still reach the target, and how heavy.

    session    the status file and stop signal the app watches
    grid       working out what to search: engine pairings, and which settings to sweep
    evaluate   one candidate vehicle end to end: assemble it, price its escape, fly its cruise
    search     the session state and the two-stage driver
    report     what a finished session leaves behind
    args       the arguments the app starts this module with

Sweeps dry mass, propellant load, engine type and count, and working gas over one fixed mission,
and finds the heaviest dry mass that still works for each combination. Working means the whole
launch-and-cruise pipeline succeeds: a flown escape spiral prices the departure, a sweep solves the
split between escape and cruise instead of someone picking it, and the answer needs both a
converged trajectory and enough propellant in the tank.

Two things shape the design. The cruise solve ignores the tank, so a vehicle that cannot quite
manage it still converges and reports how far short it fell. That shortfall is a result in its own
right rather than a failure. And the instant analytic pass that runs first is built from two
optimistic bounds, so it can prove a combination will not work but never that it will. It orders
the search and narrows it; the solver alone decides.

Importable with no UI dependency. The app starts it as ``python -m
prospector.trades.design_search``, the same arrangement ``worker.py`` gives the solver jobs.

Units: km/s, kg, days.
"""
from __future__ import annotations

from prospector.trades.design_search.args import main, parse_args  # noqa: F401
from prospector.trades.design_search.evaluate import (  # noqa: F401
    MISMATCH_TOL,
    analytic_row,
    build_config,
    evaluate_combo,
    evaluate_one,
    spiral_curve,
)
from prospector.trades.design_search.grid import (  # noqa: F401
    model_combinations,
    parse_model_axes,
    parse_pairs,
)
from prospector.trades.design_search.report import persist_winner, write_outputs  # noqa: F401
from prospector.trades.design_search.search import (  # noqa: F401
    DEFAULT_OUT_ROOT,
    SPIRAL_CACHE_DIR,
    Search,
    run_search,
)
from prospector.trades.design_search.session import (  # noqa: F401
    Cancelled,
    write_session_status,
)
