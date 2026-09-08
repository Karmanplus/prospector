"""Drivers that run the solvers. No physics of their own.

    pipeline        solve one target end to end: the grid, a starting guess, the low-thrust solve
    sweep           the trade between escape and cruise for one target
    design_search   search across many candidate vehicles

Each covers more ground than the last. ``pipeline`` solves one target once, ``sweep`` runs the
cruise at a range of departure speeds to find where the split between escape and cruise is
cheapest, and ``design_search`` runs that whole stack for every candidate vehicle to find the
heaviest one that still works. These decide what to solve; the modules under ``solvers/`` decide
how.

``sweep.escape_cost`` is the one function that prices an escape. The UI and the search both call
it, so a departure speed can never be priced two different ways.

All three are importable with no UI dependency. Long runs go out as detached subprocesses:
``worker.py`` for the per-target jobs, and the design search as its own module.
"""
