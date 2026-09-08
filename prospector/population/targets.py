"""
Target lookup: resolve any body in the population by designation or name.

A thin lookup over the table :func:`prospector.population.load_population` returns. It lets the UI,
or a script, pick a specific body before any screening has run and whether or not the body is
reachable. The first thing to use it was the Launch tab's plane-change panel, which needs a
target's orbital tilt to price the plane change before any screen results exist.

Both kinds of body work the same way, so ``get_target("Mars")`` behaves like
``get_target("Apophis")``: the planets are in the table under their names.

Deliberately not a query engine: no network calls of its own, and no matching beyond
case-insensitive substrings. What can be found is what is in the table, so an asteroid fainter than
the cached brightness cutoff simply will not turn up. The fix is to raise ``h_max``, or pass a
wider ``population``, which re-keys the cache in ``fetch_population`` and pulls the fainter tail
once. The cutoff never hides a planet, because it does not apply to them (see
:mod:`prospector.population.catalog`).
"""
from __future__ import annotations

import pandas as pd

from prospector.population.catalog import load_population
from prospector.population.sbdb import ELEMENT_COLS, ORBIT_COLS

# Population cutoff used when no frame is supplied. Matches the Screening default
# (config.Screening.h_max = 25.0) so anything the screen would ever consider is findable here too,
# and comfortably includes the usual reference bodies (Apophis H ~ 19.7, 2008 EV5 H ~ 20.0).
# Raising it only costs one wider SBDB fetch, then it's cached.
DEFAULT_H_MAX = 25.0

# Columns returned as plain Python floats in a target dict (the orbital elements the solvers
# consume plus H). Everything else passes through verbatim.
_FLOAT_COLS = tuple(ELEMENT_COLS) + tuple(ORBIT_COLS)

# A target can arrive as a population row (pandas Series) or an already-built dict.
TargetLike = pd.Series | dict


def search_targets(query: str, *, population: pd.DataFrame | None = None,
                   limit: int = 20) -> pd.DataFrame:
    """Find small bodies whose designation or name contains ``query``.

    Case-insensitive substring match over the ``pdes`` (primary designation) and ``full_name``
    columns of the population frame. When ``population`` is None, the cached SBDB population at
    ``DEFAULT_H_MAX`` is loaded (one network fetch the first time, parquet cache after).

    Returns up to ``limit`` matching rows, best match first: an exact designation hit outranks a
    designation prefix, which outranks any other substring hit. An empty or whitespace query
    returns an empty frame.
    """
    if population is None:
        population = load_population(h_max=DEFAULT_H_MAX)
    key = str(query).strip().lower()
    if not key or population.empty:
        return population.iloc[0:0].copy()

    pdes = _normalized(population, "pdes")
    names = _normalized(population, "full_name")
    mask = pdes.str.contains(key, regex=False) | names.str.contains(key, regex=False)
    hits = population[mask]
    if hits.empty:
        return hits.copy()

    # Rank: exact designation, then designation prefix, then any substring. The sort is
    # stable, so within a rank the population's own order (SBDB query order) is kept.
    rank = pd.Series(2, index=hits.index)
    rank[pdes[mask].str.startswith(key)] = 1
    rank[pdes[mask] == key] = 0
    return hits.loc[rank.sort_values(kind="stable").index].head(limit).copy()


def get_target(designation: str, *, population: pd.DataFrame | None = None
               ) -> dict | None:
    """Resolve one small body to a target dict, or None if it isn't in the population.

    Matching is case-insensitive: an exact ``pdes`` match wins, then an exact ``full_name`` match,
    and finally a substring search, for the convenience of bare names like "Apophis", which
    resolves only if it is unambiguous (exactly one body matches). An ambiguous name returns None
    rather than guessing; use :func:`search_targets` to disambiguate.

    The returned dict carries every column of the population row (so it is directly usable as a
    solver ``target_row``: ``pdes``, ``full_name``, ``spkid``, ``a``, ``e``, ``i``, ``om``, ``w``,
    ``ma``, ``epoch``, ``H``, ...) with the numeric elements as plain floats, plus a ``name``
    display string from :func:`display_name`.
    """
    if population is None:
        population = load_population(h_max=DEFAULT_H_MAX)
    key = str(designation).strip().lower()
    if not key or population.empty:
        return None

    hit = population[_normalized(population, "pdes") == key]
    if hit.empty:
        hit = population[_normalized(population, "full_name") == key]
    if hit.empty:
        found = search_targets(key, population=population, limit=2)
        if len(found) == 1:
            hit = found
    if hit.empty:
        return None
    return _row_to_target(hit.iloc[0])


def display_name(target: TargetLike) -> str:
    """A human label for a target row or dict, e.g. ``"99942 Apophis (2004 MN4)"``.

    Prefers ``full_name``, stripped because SBDB pads it, then ``name``, then the bare ``pdes``
    designation, so every caller renders targets the same way regardless of which dict shape it
    holds.
    """
    for field in ("full_name", "name", "pdes"):
        value = target.get(field)
        if value is not None and not pd.isna(value):
            text = str(value).strip()
            if text:
                return text
    return "target"


def _normalized(population: pd.DataFrame, column: str) -> pd.Series:
    """The column lowercased and stripped for matching; empty strings where absent."""
    if column not in population.columns:
        return pd.Series("", index=population.index, dtype=str)
    return population[column].fillna("").astype(str).str.strip().str.lower()


def _row_to_target(row: pd.Series) -> dict:
    """A population row as a plain dict: numeric elements as floats, plus ``name``."""
    out = {}
    for col, value in row.items():
        if col in _FLOAT_COLS and pd.notna(value):
            out[col] = float(value)
        else:
            out[col] = value
    out["name"] = display_name(out)
    return out
