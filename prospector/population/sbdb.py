"""The small-body population: one SBDB element query, cached on disk.

The front door of the data layer. Pulls orbital elements, absolute brightness and the orbit-quality
code from JPL's Small-Body Database Query API, so the element screen can run over the whole set in
numpy.

It fetches only what that screen needs, being semimajor axis, eccentricity, inclination, brightness
and the quality code, and no positions, so the fast path stays light enough for any laptop with
just requests and pandas. Results are cached on disk against the query, so working on the screen
does not hammer the API.

Note what that implies about scope: the query covers asteroids only. A planet is not in this
database and cannot be looked up through here.

Two slices are named: ``"neo"``, the near-Earth group, and ``"mba"``, the main belt (inner,
middle and outer classes together). SBDB groups only the near-Earth and hazardous sets, so the
main belt is asked for by orbit class instead; :func:`fetch_population` hides that difference
behind one ``group`` argument.

Units: a in AU, i in degrees, matching solvers/edelbaum.py.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import requests

from prospector.paths import DATA_DIR

DEFAULT_TIMEOUT = 60  # seconds


SBDB_QUERY_API = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"


SBDB_QUERY_API_VERSION = "1.0"


# Columns fetched from SBDB. `a, e, i` drive the element screen; `H` is the brightness the cutoff
# uses as a stand-in for size; `condition_code` runs 0 for a well-known orbit to 9 for a badly
# known one, and keeps the worst out. `om`, `w`, `ma` and `epoch` complete the orbit, being the two
# angles that orient it, where the body sits along it, and the date all of that is quoted for.
# Together they let Lambert place the body on any date.
FIELDS = ["spkid", "pdes", "full_name", "H", "condition_code",
          "a", "e", "i", "om", "w", "ma", "epoch"]


# Columns the screen needs: a row is dropped only if one of these will not parse. The remaining
# orbit angles (om, w, ma, epoch) are needed only by the downstream Lambert solve, so a target is
# never dropped from the screen for lacking them.
ELEMENT_COLS = ["a", "e", "i", "H"]


ORBIT_COLS = ["om", "w", "ma", "epoch"]


# The named population slices, and how SBDB is asked for each. "neo"/"pha" are SBDB groups; the
# main belt has no group, so it is the union of its three orbit classes (inner, middle, outer).
NEAR_EARTH = "neo"
MAIN_BELT = "mba"
_GROUP_PARAMS: dict[str, dict[str, str]] = {
    NEAR_EARTH: {"sb-group": "neo"},
    "pha": {"sb-group": "pha"},
    MAIN_BELT: {"sb-class": "IMB,MBA,OMB"},
}


DEFAULT_CACHE_DIR = DATA_DIR / "cache"


def fetch_population(
    h_max: float = 25.0,
    condition_code_max: int = 7,
    group: str = "neo",
    *,
    session: requests.Session | None = None,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
) -> pd.DataFrame:
    """Fetch the small-body population as a tidy, numeric DataFrame.

    Queries SBDB for the requested group (default near-Earth asteroids; ``"mba"`` is the main
    belt, see :data:`MAIN_BELT`), keeping objects brighter than ``h_max`` (the H cutoff is applied
    server-side) and with an orbit no worse than ``condition_code_max``. The H bound and group key
    the on-disk cache; pass ``refresh=True`` to bypass it.

    Returns one row per object with numeric ``a, e, i, H`` (rows whose elements don't parse are
    dropped) plus the ``spkid / pdes / full_name`` identifiers. This frame is the direct input to
    :func:`prospector.solvers.edelbaum.evaluate_dataframe`.
    """
    cache_path = _cache_path(cache_dir, group, h_max) if cache_dir else None
    if cache_path and cache_path.is_file() and not refresh:
        return pd.read_parquet(cache_path)

    # A network fetch of tens of thousands of orbits takes a minute or two on a first run, during
    # which the app shows only a placeholder. Say so where the person launched it from.
    what = {NEAR_EARTH: "near-Earth", MAIN_BELT: "main-belt"}.get(group, group)
    where = f", caching it under {cache_path.parent}" if cache_path is not None else ""
    print(f"prospector: downloading the {what} small-body catalogue (H <= {h_max:g}) from JPL SBDB"
          f"{where} - a minute or two on a first run...", file=sys.stderr, flush=True)
    raw = _query(h_max, group, session=session, timeout=timeout)
    df = _clean(raw, condition_code_max)
    print(f"prospector: {what} catalogue ready, {len(df):,} bodies.", file=sys.stderr, flush=True)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
    return df


def _query(h_max: float, group: str, *, session: requests.Session | None,
           timeout: int) -> pd.DataFrame:
    """Run the SBDB query and return its rows verbatim as a DataFrame."""
    params = {
        "sb-kind": "a",          # asteroids
        # A named slice maps to its SBDB selector; anything else is passed through as a group.
        **_GROUP_PARAMS.get(group, {"sb-group": group}),
        "fields": ",".join(FIELDS),
        # Server-side H cutoff, so the faint tail is never transferred only to be dropped.
        "sb-cdata": json.dumps({"AND": [f"H|LE|{h_max}"]}),
    }
    owns_session = session is None
    session = session or requests.Session()
    try:
        response = session.get(SBDB_QUERY_API, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    finally:
        if owns_session:
            session.close()

    version = payload.get("signature", {}).get("version")
    if version != SBDB_QUERY_API_VERSION:
        warnings.warn(f"SBDB query API version {version!r}; expected {SBDB_QUERY_API_VERSION!r}",
                      stacklevel=2)
    return pd.DataFrame(payload.get("data", []), columns=payload["fields"])


def _clean(df: pd.DataFrame, condition_code_max: int) -> pd.DataFrame:
    """Convert the numeric columns, drop rows that will not parse or whose orbit is too
    uncertain, and reindex.

    Only a, e, i and H can cause a row to be dropped. The orbit angles are converted too, but a
    missing one never loses a target; it only means Lambert cannot solve for it.

    Every dropped row is reported. Losing a body from the population is the one thing this tool
    must never do quietly: the expensive mistake is throwing away something reachable, and a body
    removed here is invisible everywhere downstream with nothing to show it was ever a candidate.
    The counts are split by reason so a surprising total can be traced. Elements that will not
    parse are a data problem, while a rejected quality code is the caller's ``condition_code_max``
    doing its job.
    """
    out = df.copy()
    for col in ELEMENT_COLS + ORBIT_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    code = pd.to_numeric(out.get("condition_code"), errors="coerce").fillna(999)
    parseable = out[ELEMENT_COLS].notna().all(axis=1)
    well_determined = code <= condition_code_max
    n_unparseable = int((~parseable).sum())
    n_uncertain = int((parseable & ~well_determined).sum())
    if n_unparseable or n_uncertain:
        warnings.warn(
            f"population fetch dropped {n_unparseable + n_uncertain} of {len(out)} bodies: "
            f"{n_unparseable} with unparseable {'/'.join(ELEMENT_COLS)}, "
            f"{n_uncertain} with condition_code > {condition_code_max}",
            stacklevel=3)
    return out[parseable & well_determined].reset_index(drop=True)


# Bumped when the fetched column set changes, so an older cache (missing new columns) is ignored
# rather than silently served.
_CACHE_SCHEMA = "v2"


def _cache_path(cache_dir: Path, group: str, h_max: float) -> Path:
    """Cache file for a (group, H-cutoff) query; H rounded to 0.1 to match the slider."""
    return Path(cache_dir) / f"sbdb_{group}_H{h_max:.1f}_{_CACHE_SCHEMA}.parquet"
