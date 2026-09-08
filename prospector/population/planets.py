"""The major planets as targets, using the same columns as the small-body population.

Why they are a separate source: the population comes from one SBDB query for asteroids
(``sb-kind=a``), so a planet cannot come through it at all, and there is nothing to widen. This
module supplies the planets as a small table with the same columns, so
:func:`prospector.population.load_population` can hand the screen one table holding both and
everything downstream keeps working unchanged.

What going to a planet means here: the solvers work out a rendezvous with the body's orbit. For an
asteroid that is the whole job. For a planet it means arriving at the edge of its gravitational
reach with whatever speed the transfer has left over. Getting into orbit around it, or entering its
atmosphere, is extra, and none of that is in the delta-v quoted. A planet that wants its capture
cost counted declares it the same way a return destination does (``arrival_vinf_kms`` and
``insertion_dv_kms``, see :mod:`prospector.launch`); until it does, the quoted delta-v is the cost
of getting there, and the UI says so.

The elements and SPK ids here are for the planets themselves: Mars is 499, not the barycentre 4. At
the scale of a transfer between planets the two are indistinguishable for any planet without a
massive moon, but the difference matters to anything reading SPK kernels rather than these
elements.

The elements are mean Keplerian elements at J2000, from JPL's "Keplerian Elements for Approximate
Positions of the Major Planets" (Standish, the 1800-2050 fit), and are stored as published:
semimajor axis, eccentricity, inclination, mean longitude, longitude of perihelion and longitude of
the ascending node. The population's ``w`` and ``ma`` are worked out from them here, so the table
can be checked against the source line by line rather than against arithmetic somebody already did
by hand.

These elements are here for the date-free screen and for display. They are not what the
date-resolved solvers use: propagating a planet forward from fixed elements over a multi-year trip
is the wrong tool, so :func:`prospector.solvers.lambert.planet_from_row` sends a planet row to
PyKEP's ``jpl_lp`` series instead, the same one Earth already uses. This table agrees with that
series at J2000 by construction, since both come from Standish, and a test pins that.

Units: a in AU, angles in degrees, matching :mod:`prospector.population.sbdb`.
"""
from __future__ import annotations

import pandas as pd

# The value of the ``body_class`` column, which travels with every population row so downstream
# code can tell what kind of body it is holding. Asteroid rows carry ASTEROID; the enrichment axis
# (mining value: size, composition, structural stability) is meaningful only for those.
ASTEROID = "asteroid"
PLANET = "planet"

# One row per planet, as published in the Standish J2000 table, plus the identifiers the
# population needs. ``h`` is absolute brightness from JPL; every planet is far brighter than any
# screening cutoff, though planets skip the cutoff anyway (see :func:`planet_population`).
#
# Earth is missing because it is where every transfer starts. Adding Pluto, or anything else
# PyKEP's jpl_lp series covers, is one row plus its ``jpl_lp`` key.
_PLANETS: tuple[dict, ...] = (
    {"name": "Mercury", "spkid": 199, "jpl_lp": "mercury", "h": -0.60,
     "a": 0.38709927, "e": 0.20563593, "i": 7.00497902,
     "mean_longitude_deg": 252.25032350, "perihelion_longitude_deg": 77.45779628,
     "node_longitude_deg": 48.33076593},
    {"name": "Venus", "spkid": 299, "jpl_lp": "venus", "h": -4.47,
     "a": 0.72333566, "e": 0.00677672, "i": 3.39467605,
     "mean_longitude_deg": 181.97909950, "perihelion_longitude_deg": 131.60246718,
     "node_longitude_deg": 76.67984255},
    {"name": "Mars", "spkid": 499, "jpl_lp": "mars", "h": -1.60,
     "a": 1.52371034, "e": 0.09339410, "i": 1.84969142,
     "mean_longitude_deg": -4.55343205, "perihelion_longitude_deg": -23.94362959,
     "node_longitude_deg": 49.55953891},
    {"name": "Jupiter", "spkid": 599, "jpl_lp": "jupiter", "h": -9.40,
     "a": 5.20288700, "e": 0.04838624, "i": 1.30439695,
     "mean_longitude_deg": 34.39644051, "perihelion_longitude_deg": 14.72847983,
     "node_longitude_deg": 100.47390909},
    {"name": "Saturn", "spkid": 699, "jpl_lp": "saturn", "h": -8.88,
     "a": 9.53667594, "e": 0.05386179, "i": 2.48599187,
     "mean_longitude_deg": 49.95424423, "perihelion_longitude_deg": 92.59887831,
     "node_longitude_deg": 113.66242448},
    {"name": "Uranus", "spkid": 799, "jpl_lp": "uranus", "h": -7.19,
     "a": 19.18916464, "e": 0.04725744, "i": 0.77263783,
     "mean_longitude_deg": 313.23810451, "perihelion_longitude_deg": 170.95427630,
     "node_longitude_deg": 74.01692503},
    {"name": "Neptune", "spkid": 899, "jpl_lp": "neptune", "h": -6.87,
     "a": 30.06992276, "e": 0.00859048, "i": 1.77004347,
     "mean_longitude_deg": 304.87997031, "perihelion_longitude_deg": 44.96476227,
     "node_longitude_deg": 131.78422574},
)

# Julian date of J2000.0 (2000-01-01 12:00 TT), the date these elements are quoted for. The
# population carries that date as a Julian date, the way SBDB does.
J2000_JD = 2451545.0

# The planets' orbits are the best-determined in the solar system, so they take the best
# orbit-quality code, on the same scale SBDB grades small bodies (0 = best, 9 = worst).
_CONDITION_CODE = 0

# The extra column planet rows add to the population schema, so a caller can find them by name.
BODY_CLASS_COL = "body_class"
JPL_LP_COL = "jpl_lp_key"


def planet_population() -> pd.DataFrame:
    """The major planets as a population frame, in the SBDB schema.

    Columns match :data:`prospector.population.sbdb.FIELDS`, being ``spkid, pdes, full_name, H,
    condition_code, a, e, i, om, w, ma, epoch``, plus ``body_class`` (:data:`PLANET`) and
    ``jpl_lp_key``, the name PyKEP's fitted series knows the planet by.

    ``w`` and ``ma`` are derived from the published longitudes: the argument of perihelion is
    ``perihelion_longitude - node_longitude`` and the mean anomaly is ``mean_longitude -
    perihelion_longitude``, both wrapped into ``[0, 360)``.

    No network access and no cache: this is authored data, so it is built fresh and is always
    available, so a planet target never depends on a fetch having succeeded.
    """
    rows = []
    for p in _PLANETS:
        w = (p["perihelion_longitude_deg"] - p["node_longitude_deg"]) % 360.0
        ma = (p["mean_longitude_deg"] - p["perihelion_longitude_deg"]) % 360.0
        rows.append({
            "spkid": p["spkid"],
            # Planets have no provisional designation, so the name serves as the primary key the
            # rest of the tool addresses a target by (``targets.get_target("Mars")``).
            "pdes": p["name"],
            "full_name": p["name"],
            "H": p["h"],
            "condition_code": _CONDITION_CODE,
            "a": p["a"], "e": p["e"], "i": p["i"],
            "om": p["node_longitude_deg"] % 360.0,
            "w": w, "ma": ma,
            "epoch": J2000_JD,
            BODY_CLASS_COL: PLANET,
            JPL_LP_COL: p["jpl_lp"],
        })
    return pd.DataFrame(rows)


def list_planets() -> list[str]:
    """The planet keys, in orbital order outward from the Sun."""
    return [p["name"] for p in _PLANETS]


def is_planet(target) -> bool:
    """Whether a target row or dict is a major planet rather than a small body.

    Reads the ``body_class`` column. A row without one is an asteroid: every planet row this module
    builds carries the marker, so its absence means the row came from the small-body query.
    """
    try:
        value = target.get(BODY_CLASS_COL)
    except AttributeError:
        return False
    return str(value) == PLANET


def jpl_lp_key(target) -> str | None:
    """The PyKEP ``jpl_lp`` ephemeris name for a planet row, or None for a small body.

    This is what lets the date-resolved solvers place a planet properly, instead of propagating it
    forward from fixed elements (see the module docstring).
    """
    if not is_planet(target):
        return None
    key = target.get(JPL_LP_COL)
    return str(key) if key else None
