"""AstorbDB: Lowell Observatory's small-body physical-property database.

Two different things are asked of it, and they have very different shapes.

Per target, :func:`lookup` pulls every survey determination of albedo, taxonomy and rotational
lightcurve for one body, including two flags that matter to the grade and appear nowhere
else: whether the body is a non-principal-axis (tumbling) rotator, and the quality code the
lightcurve was assigned.

Once per run, :func:`spin_catalog` pulls the rotation period and brightness of every body that has
a measured period. That is the reference population :mod:`.models.spin` fits a distribution to, so
it is fetched whole rather than per target: it is the same tens of thousands of rows whichever
targets are being looked at, and fetching it per target is where all the time went.

Both are cached on disk. The per-target cache stops a screen of overlapping populations re-querying
bodies it already knows, and the catalogue cache stops a multi-megabyte download happening more
than once.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import requests

from prospector.enrichment.measurements import Measurement
from prospector.paths import DATA_DIR

GRAPHQL_URL = "https://astorbdb.lowell.edu/v1/graphql"

# Every survey determination for the requested bodies. Numbered and provisionally-designated
# bodies are matched on different keys, so both go in and the server unions them.
_TARGET_QUERY = """
query TargetQuery($designations: [String!], $numbers: [bigint!]) {
  minorplanet(
    where: {_or: [{ast_number: {_in: $numbers}},
                  {designameByIdDesignationPrimary: {str_designame: {_in: $designations}}}]}
  ) {
    ast_number
    designameByIdDesignationPrimary { str_designame }
    surveydata {
      albedo { albedo }
      lightcurve {
        period
        ambiguous_period
        non_principal_axis_rotator
        code_lc_quality { code }
      }
      taxonomy { taxonomy_sys_type { tax_type } }
    }
  }
}
"""

# Absolute magnitude and rotation period for every body that has one: the reference
# population for the spin-distribution fit.
_CATALOG_QUERY = """
query SpinCatalog {
  minorplanet(where: {surveydata: {lightcurve: {period: {_is_null: false}}}}) {
    h
    surveydata { lightcurve { period } }
  }
}
"""

_TARGET_TIMEOUT_S = 120.0
_CATALOG_TIMEOUT_S = 600.0     # tens of thousands of rows in one response

# Properties this source contributes. Named here so a caller can build an empty result without
# knowing the query, and so a typo in one of the keys below fails a test rather than silently
# producing a body with no albedo.
PROPERTIES: tuple[str, ...] = ("albedo", "taxonomy", "period_h", "tumbling",
                               "lightcurve_quality")

_catalog_lock = threading.Lock()
_catalog: tuple[list[float], list[float]] | None = None


def cache_dir() -> Path:
    """Where this source caches. A function, not a constant, so the path is assembled
    per call, but note ``DATA_DIR`` itself is bound at import: redirecting
    ``paths.DATA_DIR`` does not move this. Pass an explicit path to relocate a cache."""
    return DATA_DIR / "enrichment" / "astorb"


def _post(query: str, variables: dict | None, timeout: float) -> dict:
    response = requests.post(GRAPHQL_URL, json={"query": query, "variables": variables or {}},
                             timeout=timeout)
    response.raise_for_status()
    return response.json()


def _empty() -> dict[str, list[Measurement]]:
    return {name: [] for name in PROPERTIES}


def _survey_measurements(records) -> dict[str, list[Measurement]]:
    """One body's survey rows, split into per-property measurement lists.

    Nothing here is flagged preferred: AstorbDB reports what each survey found without ranking
    them, so the choice falls to the source ordering in :mod:`prospector.enrichment.measurements`.
    """
    out = _empty()
    for entry in records or []:
        albedo = (entry.get("albedo") or {}).get("albedo")
        if albedo is not None:
            out["albedo"].append(Measurement(albedo, source="astorb"))

        taxonomy = ((entry.get("taxonomy") or {}).get("taxonomy_sys_type") or {}).get("tax_type")
        if taxonomy is not None:
            out["taxonomy"].append(Measurement(taxonomy, source="astorb"))

        lightcurve = entry.get("lightcurve")
        if lightcurve:
            if lightcurve.get("period") is not None:
                out["period_h"].append(Measurement(lightcurve["period"], source="astorb"))
            if lightcurve.get("non_principal_axis_rotator") is not None:
                out["tumbling"].append(
                    Measurement(lightcurve["non_principal_axis_rotator"], source="astorb"))
            quality = (lightcurve.get("code_lc_quality") or {}).get("code")
            if quality is not None:
                out["lightcurve_quality"].append(Measurement(quality, source="astorb"))
    return out


def _cache_path(identifier: str) -> Path:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", identifier).strip("_") or "id"
    return cache_dir() / f"{slug}.json"


def _as_measurements(cached: dict) -> dict[str, list[Measurement]]:
    return {name: [Measurement(v, source="astorb") for v in cached.get(name, [])]
            for name in PROPERTIES}


def lookup(identifier: str) -> dict[str, list[Measurement]]:
    """Every AstorbDB survey determination for one body, by property.

    An unreachable database or an unknown body both come back as empty lists. The screen keeps
    targets with unknown properties, so a missing record costs a target nothing.
    """
    path = _cache_path(identifier)
    if path.is_file():
        try:
            return _as_measurements(json.loads(path.read_text()))
        except Exception:
            pass    # unreadable cache entry: refetch this one

    numbers = [int(identifier)] if re.fullmatch(r"\d+", identifier.strip()) else []
    designations = [] if numbers else [identifier.strip()]
    try:
        payload = _post(_TARGET_QUERY,
                        {"designations": designations, "numbers": numbers}, _TARGET_TIMEOUT_S)
    except Exception:
        return _empty()

    bodies = (payload.get("data") or {}).get("minorplanet") or []
    result = _survey_measurements(bodies[0].get("surveydata") if bodies else None)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {name: [m.value for m in result[name]] for name in PROPERTIES}))
    except Exception:
        pass        # caching is best-effort; never fail a lookup over it
    return result


def spin_catalog(refresh: bool = False) -> tuple[list[float], list[float]]:
    """``(absolute magnitudes, rotation periods)`` for every body with a measured period.

    Fetched at most once per process and cached on disk between runs. Returns empty lists if the
    database cannot be reached, which leaves the spin-distribution fit unknown rather than failing
    the run.

    Only the first period per body is used. A body measured several times still counts once, so the
    distribution describes the population of bodies rather than the population of measurements.
    Otherwise the handful of heavily studied targets would dominate it.
    """
    global _catalog
    with _catalog_lock:
        if _catalog is not None and not refresh:
            return _catalog

        path = cache_dir() / "spin_catalog.json"
        if path.is_file() and not refresh:
            try:
                cached = json.loads(path.read_text())
                _catalog = (cached["abs_mag"], cached["period_h"])
                return _catalog
            except Exception:
                pass

        try:
            payload = _post(_CATALOG_QUERY, None, _CATALOG_TIMEOUT_S)
        except Exception:
            return [], []

        magnitudes: list[float] = []
        periods: list[float] = []
        for body in (payload.get("data") or {}).get("minorplanet") or []:
            abs_mag = body.get("h")
            if abs_mag is None:
                continue
            for entry in body.get("surveydata") or []:
                period = (entry.get("lightcurve") or {}).get("period")
                if period is not None:
                    magnitudes.append(float(abs_mag))
                    periods.append(float(period))
                    break

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"abs_mag": magnitudes, "period_h": periods}))
        except Exception:
            pass
        _catalog = (magnitudes, periods)
        return _catalog
