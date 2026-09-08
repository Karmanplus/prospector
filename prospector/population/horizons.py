"""JPL Horizons: a body's orbit as it stands on a particular date.

The population carries one orbit per body, quoted for one date. Propagating forward from there
breaks across a close approach to a planet: Apophis's April 2029 Earth flyby moves it from a 0.92
AU orbit to 1.10 AU, so a rendezvous solved after that date using the cached orbit is chasing one
that no longer exists. Horizons integrates the real dynamics and will report the orbit at any date,
so asking it for the arrival date keeps the simple propagation accurate where the rendezvous
happens. Any failure returns the row unchanged with a warning: a stale orbit that a real solver
will check later beats a pipeline that stops dead.
"""
from __future__ import annotations

import json
import warnings
from datetime import date
from pathlib import Path

import requests

from prospector.population.sbdb import DEFAULT_CACHE_DIR, DEFAULT_TIMEOUT

HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"


# Horizons output keys mapped to the population's element columns. Angles in degrees, a in AU
# (OUT_UNITS=AU-D), everything relative to the Sun and Earth's orbit, which is the same frame and
# the same units the SBDB rows use.
_HORIZONS_ELEMENT_KEYS = {"A": "a", "EC": "e", "IN": "i",
                          "OM": "om", "W": "w", "MA": "ma"}


def horizons_elements(
    command: str,
    epoch: date,
    *,
    session: requests.Session | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """One body's orbit around the Sun on a given date, from Horizons.

    ``command`` is what Horizons calls the body: ``"99942;"`` for a numbered asteroid, ``"DES=2008
    EV5;"`` for a designation. Returns ``{a, e, i, om, w, ma, epoch}``, with ``epoch`` the Julian
    date the orbit is quoted for. These are the same columns an SBDB row carries, so they drop
    straight in. Raises if the fetch fails or the response cannot be read, leaving the caller to
    decide what to do about it.
    """
    jd = _julian_date(epoch)
    owns_session = session is None
    session = session or requests.Session()
    try:
        response = session.get(HORIZONS_API, params={
            "format": "text", "COMMAND": f"'{command}'", "OBJ_DATA": "NO",
            "MAKE_EPHEM": "YES", "EPHEM_TYPE": "ELEMENTS", "CENTER": "'500@10'",
            "TLIST": f"'{jd}'", "OUT_UNITS": "AU-D", "REF_PLANE": "ECLIPTIC",
        }, timeout=timeout)
        response.raise_for_status()
        text = response.text
    finally:
        if owns_session:
            session.close()
    return parse_horizons_elements(text)


def parse_horizons_elements(text: str) -> dict:
    """Parse the ``$$SOE`` element block of a Horizons ELEMENTS response."""
    import re

    start, end = text.find("$$SOE"), text.find("$$EOE")
    if start < 0 or end < 0:
        raise ValueError(f"no element block in Horizons response; began: {text[:200]!r}")
    block = text[start:end]
    out = {}
    for key, col in _HORIZONS_ELEMENT_KEYS.items():
        # Keys are short and overlap (W vs AD/OM): anchor on a non-letter boundary.
        m = re.search(rf"(?<![A-Z]){key}\s*=\s*([0-9.Ee+-]+)", block)
        if m is None:
            raise ValueError(f"Horizons element {key!r} missing from response block")
        out[col] = float(m.group(1))
    m = re.search(r"\$\$SOE\s*\n\s*([0-9.]+)\s*=", block)
    if m is None:
        raise ValueError("Horizons element epoch line missing from response block")
    out["epoch"] = float(m.group(1))
    return out


def refresh_target_elements(
    row: dict,
    epoch: date,
    *,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    session: requests.Session | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    allow_stale: bool = False,
) -> dict:
    """A population row with its orbit refreshed for ``epoch`` from Horizons, cached on disk.

    Returns a copy of ``row`` with ``a, e, i, om, w, ma, epoch`` replaced by Horizons' values, plus
    ``elements_source`` and ``elements_epoch`` recording where they came from. The fetch is cached
    per body and date, so only the first call for a mission touches the network.

    Raises on any failure, whether from being offline, an ambiguous designation, or Horizons having
    a bad day, because the alternative is worse than stopping. Solving against a stale orbit does
    not give a slightly worse trajectory; across a close approach it gives a rendezvous with an
    orbit the body does not fly, and the only trace is a field nobody reads. That is a wrong answer
    dressed up as a right one.

    ``allow_stale=True`` accepts the stale orbit for a caller that genuinely prefers a rough answer
    to none, such as an offline exploratory screen. It warns, and the returned row carries
    ``elements_source: "sbdb-cache"`` so that travels with the result. It has to be asked for.

    A planet comes back unchanged. Its elements are not what gets propagated: the solvers read a
    planet's position from PyKEP's fitted ``jpl_lp`` series instead
    (:func:`prospector.solvers.lambert.planet_from_row`), which already accounts for everything a
    refresh would pick up. Fetching a fresh orbit for one would spend a network round trip refining
    numbers nothing downstream reads.
    """
    from prospector.population.planets import is_planet

    row = dict(row)
    if is_planet(row):
        return {**row, "elements_source": "jpl_lp", "elements_epoch": epoch.isoformat()}
    spkid = row.get("spkid")
    pdes = str(row.get("pdes", "")).strip()
    command = f"DES={pdes};" if pdes and not pdes.isdigit() else f"{pdes};"
    body_key = str(spkid or pdes or "unknown")

    cache_path = None
    if cache_dir is not None:
        cache_path = (Path(cache_dir) / "horizons_elements"
                      / f"{body_key}_{epoch.isoformat()}.json")
        if cache_path.is_file():
            try:
                elements = json.loads(cache_path.read_text())
                return {**row, **elements, "elements_source": "horizons",
                        "elements_epoch": epoch.isoformat()}
            except (OSError, json.JSONDecodeError):
                pass

    try:
        elements = horizons_elements(command, epoch, session=session, timeout=timeout)
    except Exception as exc:
        if not allow_stale:
            raise RuntimeError(
                f"could not refresh the orbit for {body_key} at {epoch}: {exc}. The cached "
                f"orbit is quoted for a different date, so solving against it can aim at an "
                f"orbit the body no longer flies. Pass allow_stale=True to accept that "
                f"knowingly.") from exc
        warnings.warn(f"Horizons element refresh failed for {body_key} at {epoch}: "
                      f"{exc} - using the cached elements at the caller's request (their epoch "
                      f"may predate a close approach)", stacklevel=2)
        return {**row, "elements_source": "sbdb-cache"}

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(elements))
        tmp.replace(cache_path)              # atomic: sweep children race this file
    return {**row, **elements, "elements_source": "horizons",
            "elements_epoch": epoch.isoformat()}


def _julian_date(d: date) -> float:
    """Julian date of 00:00 UTC on ``d`` (the resolution element epochs need)."""
    # Days from the MJD2000 epoch (2000-01-01 = JD 2451544.5).
    return 2451544.5 + (d - date(2000, 1, 1)).days
