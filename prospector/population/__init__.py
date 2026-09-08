"""Where target bodies come from: the candidate bodies, and lookups into them.

    sbdb       the SBDB element query and its cache, covering the near-Earth asteroids
    planets    the major planets, written out here: same columns, a different kind of body
    catalog    the two sources put together into the one table the screen runs over
    horizons   orbital elements for a particular date
    targets    look up one body by designation or name

The split follows where the data comes from rather than who asks for it. ``sbdb`` is a single
bulk query the screen runs over in numpy, while ``horizons`` is fetched one body and one date at
a time, only for a target already chosen. Keeping them apart is what lets the fast screen stay
light: it needs requests and pandas, nothing more.

Asteroids and planets are separate sources because they are fetched differently and screened
differently, since the brightness cutoff stands in for size on one and means nothing on the other.
They share one set of columns, so everything above treats them alike. ``catalog.load_population``
is the front door; ``sbdb.fetch_population`` is still the asteroid query on its own.

The public names are re-exported here, so callers write ``population.load_population(...)`` without
tracking which module holds what.
"""
from __future__ import annotations

from prospector.population.catalog import (  # noqa: F401
    MAIN_BELT_H_MAX,
    load_population,
    with_body_class,
)
from prospector.population.horizons import (  # noqa: F401
    HORIZONS_API,
    horizons_elements,
    parse_horizons_elements,
    refresh_target_elements,
)
from prospector.population.planets import (  # noqa: F401
    ASTEROID,
    BODY_CLASS_COL,
    PLANET,
    is_planet,
    jpl_lp_key,
    list_planets,
    planet_population,
)
from prospector.population.sbdb import (  # noqa: F401
    DEFAULT_CACHE_DIR,
    DEFAULT_TIMEOUT,
    ELEMENT_COLS,
    FIELDS,
    MAIN_BELT,
    NEAR_EARTH,
    ORBIT_COLS,
    SBDB_QUERY_API,
    SBDB_QUERY_API_VERSION,
    fetch_population,
)
from prospector.population.targets import (  # noqa: F401
    DEFAULT_H_MAX,
    display_name,
    get_target,
    search_targets,
)

__all__ = [
    "ASTEROID",
    "BODY_CLASS_COL",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_H_MAX",
    "DEFAULT_TIMEOUT",
    "ELEMENT_COLS",
    "FIELDS",
    "MAIN_BELT",
    "MAIN_BELT_H_MAX",
    "NEAR_EARTH",
    "HORIZONS_API",
    "ORBIT_COLS",
    "PLANET",
    "SBDB_QUERY_API",
    "SBDB_QUERY_API_VERSION",
    "display_name",
    "fetch_population",
    "get_target",
    "horizons_elements",
    "is_planet",
    "jpl_lp_key",
    "list_planets",
    "load_population",
    "parse_horizons_elements",
    "planet_population",
    "refresh_target_elements",
    "search_targets",
    "with_body_class",
]
