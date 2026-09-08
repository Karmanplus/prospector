"""
Low-thrust cost over a date grid, anchored on the element-based estimate.

The impulse margin is ``a*T / dv_element``: how much velocity change the engine can build up over
the flight, against the least the orbit change can cost. A plain ratio.

The surface is the element estimate, scaled up where the vehicle is short of margin, with a small
term to break ties between departure dates. Each piece is used only where measurement supports
it: element physics sets the level, the margin supplies the trend with flight time, and the
Lambert cost only breaks ties.

Not a budget-grade number. The level carries about +/-15%, and the scaling near the edge is
calibrated on three targets, so this ranks and supplies starting guesses. ``docs/physics.md``
covers why a Lambert-based surface cannot do this job and what the measurements were.

Units: km/s, days; a in AU, i in degrees.
"""
from __future__ import annotations

# The margin x = a*T / dv_element, and what it means: below 1 the engine cannot produce the
# minimum however it is steered; 1-2 is tight (measured 1.24-1.29x the element estimate, and the
# optimizer converges less reliably); 2 and above is roomy (0.87-1.00x).
#
# CERTIFIED_MARGIN sits below 1 because the element estimate is not itself a floor: on Mars it
# runs 13% over the converged cost, so rejecting at x < 1 threw away a cell that flew comfortably.
CERTIFIED_MARGIN = 0.8


