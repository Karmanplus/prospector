"""Which leg solver a mission flies: direct (``simsflanagan``) or via a gravity assist (``flyby``).
Both expose ``solve_for_config``, ``solve_cell_for_config`` and ``rebuild_for_config`` with
solutions the pipeline reads the same way."""
from __future__ import annotations

from prospector.solvers import flyby, simsflanagan


def for_config(rc):
    """The leg solver module for a config: ``flyby`` when the mission names a planet."""
    return flyby if getattr(rc.mission, "gravity_assist", None) else simsflanagan
