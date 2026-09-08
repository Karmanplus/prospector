"""The screenable population, assembled from its sources.

Three sources answer "what could this mission go to": a bulk SBDB query for near-Earth asteroids,
a second for the bright end of the main belt (both :mod:`prospector.population.sbdb`), and a list
of the major planets written out in (:mod:`prospector.population.planets`). They are fetched
differently: the queries are cached network calls keyed by a brightness cutoff, the planets a table
written out in source. The screen wants a single table, so this puts them together and owns the
rules that requires: the screening brightness cutoff applies to near-Earth bodies only, and the
main belt has its own, fixed one.

The main belt is far too populous to fetch at a near-Earth cutoff (over a million rows at H 25),
and nearly all of it is far too small to matter here. So it enters at :data:`MAIN_BELT_H_MAX`,
bright enough to hold the large named bodies (Ceres, Vesta, Psyche...) and a couple of thousand
others, independent of the slider. Most of the belt sits over any plausible low-thrust budget, so
these rows are usually seen with the screen's "show all" view rather than among the reachable set.

That rule matters. ``h_max`` stands in for size, and it does that well for asteroids, since a
dimmer body is a smaller body and below some size a mining target is not worth going to. Brightness
means nothing of the sort for a planet: every planet is far brighter than any cutoff anyone would
set, so leaving them inside the filter would admit them for a reason unrelated to why they belong
there. Planets skip the cutoff instead, so moving it only moves the small-body tail, which is the
only thing it describes.

Every row carries a ``body_class`` (see :mod:`prospector.population.planets`), so code that has to
treat the two differently, whether working out where a body is on a given date or judging its
mining value, can tell them apart instead of guessing from a designation.
"""
from __future__ import annotations

import warnings

import pandas as pd
import requests

from prospector.population.planets import ASTEROID, BODY_CLASS_COL, planet_population
from prospector.population.sbdb import MAIN_BELT, NEAR_EARTH, fetch_population

# The fixed brightness cutoff for the main belt (absolute magnitude). H 12 is roughly a 15 km body
# at a typical albedo; about 2,300 asteroids clear it, every large named one among them.
MAIN_BELT_H_MAX = 12.0


def load_population(h_max: float = 25.0, *, include_planets: bool = True,
                    main_belt_h_max: float | None = MAIN_BELT_H_MAX,
                    **fetch_kwargs) -> pd.DataFrame:
    """The whole screenable population: near-Earth asteroids brighter than ``h_max``, main-belt
    asteroids brighter than ``main_belt_h_max`` (None: no main belt), plus the major planets.

    ``include_planets=False`` returns the small bodies alone, still tagged with ``body_class`` so
    the column is present either way and no consumer has to handle its absence. Remaining keyword
    arguments pass through to :func:`prospector.population.sbdb.fetch_population` (session, cache
    directory, refresh, timeout, condition-code limit).

    The near-Earth slice comes first, then the main belt, then the planets, so the frame's leading
    order stays what the SBDB query returned and an existing caller sees no reordering. The main
    belt is best-effort once the near-Earth slice has loaded: a fetch that fails (offline, with the
    near-Earth slice served from cache) is reported with a warning and the belt left out, rather
    than taking the screen down for the part of the catalog that is almost never reachable.
    """
    small = with_body_class(fetch_population(h_max=h_max, group=NEAR_EARTH, **fetch_kwargs))
    if main_belt_h_max is not None:
        try:
            belt = fetch_population(h_max=main_belt_h_max, group=MAIN_BELT, **fetch_kwargs)
        except (requests.RequestException, OSError, ValueError) as exc:
            warnings.warn(f"main-belt catalog unavailable ({exc}); screening near-Earth bodies "
                          f"and planets only", stacklevel=2)
        else:
            belt = with_body_class(belt)
            if not belt.empty:
                small = (belt if small.empty
                         else pd.concat([small, belt], ignore_index=True, sort=False))
                # A body cannot be in both slices; the guard is for a stubbed fetch that returns
                # the same frame for each query.
                if "pdes" in small.columns:
                    small = small.drop_duplicates(subset="pdes", keep="first")
                small = small.reset_index(drop=True)
    if not include_planets:
        return small
    planets = planet_population()
    # A concat of an empty frame would drop dtypes, so an empty small-body result (a filtered-out
    # fetch, or a stubbed one in a test) yields the planets alone rather than a schema-less join.
    if small.empty:
        return planets.reset_index(drop=True)
    return pd.concat([small, planets], ignore_index=True, sort=False)


def with_body_class(df: pd.DataFrame) -> pd.DataFrame:
    """Tag a small-body frame with ``body_class = "asteroid"``, leaving any existing value alone.

    Idempotent, so it is safe on a frame that already carries the column (a cached frame, or one
    already composed here).
    """
    out = df.copy()
    if BODY_CLASS_COL in out.columns:
        out[BODY_CLASS_COL] = out[BODY_CLASS_COL].fillna(ASTEROID)
    else:
        out[BODY_CLASS_COL] = ASTEROID
    return out
