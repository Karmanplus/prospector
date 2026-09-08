"""Plotly figures, grouped by what they show.

One module per area, all styling through :mod:`prospector.figures.theme` so the set reads as one
system. Every public name is re-exported here, so callers use ``figures.converged_grid(...)`` without
caring which module it lives in.

    theme        the dark palette, and the one place layout gets applied
    screen       what is reachable, the population's orbits, and what is worth reaching
    porkchop     the date grids: the solved low-thrust one, and the quick two-burn one
    trajectory   the flown cruise, its diagnostics, and the flight-time trade
    escape       the spiral around Earth and what it charges for departure speed
    power        array output, radiation dose, and the whole-mission timelines
    mass         propellant over the mission, and where the dry mass goes
    trade        the vehicle search scatter and its analysis

:mod:`prospector.figures.products` sits above these. It turns a solved result into the figures and
the numbers that go with them, so the app and the report build the same things from the same code.

Figures are always dark, matching the app's theme. The report lightens them for paper as it
renders, so there is no second palette here.
"""


from prospector.figures.escape import (  # noqa: F401
    BELT_EXTENT_KM,
    MOON_DIST_KM,
    OUTER_BELT_KM,
    belt_volume,
    spiral_3d,
    spiral_diagnostics,
)
from prospector.figures.mass import (  # noqa: F401
    mass_allocation,
    propellant_timeline,
)
from prospector.figures.porkchop import (  # noqa: F401
    converged_grid,
)
from prospector.figures.power import (  # noqa: F401
    PHASE_COLORS,
    altitude_radiation_profile,
    degradation_profile,
    engine_performance_timeline,
    mission_power_timeline,
)
from prospector.figures.screen import (  # noqa: F401
    SHAPE_LABEL,
    SIZE_LABEL,
    TIER_COLORS,
    TILT_LABEL,
    UNKNOWN_TIER,
    desirability_scatter,
    element_distributions,
    reachability_dome_3d,
    tier_breakdown,
)
from prospector.figures.theme import (  # noqa: F401
    ASTEROID_ORANGE,
    BG,
    DANGER,
    DEPARTURE,
    DOME,
    DV_SCALE,
    EARTH,
    EARTH_BLUE,
    ECCEN,
    GRID,
    MUTED,
    PAUSE_AMBER,
    PLAY_GREEN,
    PORKCHOP_SCALE,
    REFERENCE,
    SPACECRAFT,
    SURFACE,
    TEXT,
    THRUST_SCALE,
    mjd2000_to_datetime,
)
from prospector.figures.trade import (  # noqa: F401
    ENGINE_CYCLE,
    trade_scatter,
    vehicle_search_analysis,
)
from prospector.figures.trajectory import (  # noqa: F401
    distance_profile,
    trajectory_3d,
    trajectory_diagnostics,
)

__all__ = [
    "ASTEROID_ORANGE",
    "BELT_EXTENT_KM",
    "BG",
    "DANGER",
    "DEPARTURE",
    "DOME",
    "DV_SCALE",
    "EARTH",
    "EARTH_BLUE",
    "ECCEN",
    "ENGINE_CYCLE",
    "GRID",
    "MOON_DIST_KM",
    "MUTED",
    "OUTER_BELT_KM",
    "PAUSE_AMBER",
    "PHASE_COLORS",
    "PLAY_GREEN",
    "PORKCHOP_SCALE",
    "REFERENCE",
    "SHAPE_LABEL",
    "SIZE_LABEL",
    "SPACECRAFT",
    "SURFACE",
    "TEXT",
    "THRUST_SCALE",
    "TIER_COLORS",
    "TILT_LABEL",
    "UNKNOWN_TIER",
    "altitude_radiation_profile",
    "belt_volume",
    "degradation_profile",
    "desirability_scatter",
    "distance_profile",
    "element_distributions",
    "engine_performance_timeline",
    "mass_allocation",
    "mission_power_timeline",
    "mjd2000_to_datetime",
    "converged_grid",
    "propellant_timeline",
    "reachability_dome_3d",
    "spiral_3d",
    "spiral_diagnostics",
    "tier_breakdown",
    "trade_scatter",
    "trajectory_3d",
    "trajectory_diagnostics",
    "vehicle_search_analysis",
]
