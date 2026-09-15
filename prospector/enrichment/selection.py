"""Narrowing down to the targets worth going to. Touches nothing outside.

Reachability says a target can be reached; this says it is worth reaching. The module turns a
:class:`~prospector.config.Desirability`, which holds a minimum grade plus composition, spin and
size terms, into a true/false mask over the described population, and combines it with the
reachability mask into the single ``selected`` column the solve step reads.

The rule throughout: never quietly drop a reachable target. A row that is not characterized,
or that a filter in force cannot test, is flagged ``uncharacterized`` and, by default, left out
of ``selected`` so the list holds only measured matches; with ``keep_unknown`` it stays in. The
flag is always there, so the count of such rows is always visible. A filter left unset keeps
every characterized row, and a table that has not been described at all is not filtered.
"""
from __future__ import annotations

import pandas as pd

from prospector.config import Desirability
from prospector.enrichment.schema import TIER_ORDER, taxonomy_major


def _tier_rank(tier):
    """Position in TIER_ORDER (lower = more desirable), or None if unknown."""
    if isinstance(tier, str) and tier.strip().upper() in TIER_ORDER:
        return TIER_ORDER.index(tier.strip().upper())
    return None


def desirability_masks(df: pd.DataFrame, des: Desirability) -> tuple[pd.Series, pd.Series]:
    """Two boolean Series over ``df``: ``(failed, unknown)``.

    ``failed`` is True where a known value fails a filter in force; ``unknown`` where a filter in
    force found no value to test. A row that is neither met every filter on measured data.

    A filter applies only when its term is set and its column exists; a table that has not been
    described yet has no such columns, so nothing is filtered. Each filter runs in one pass over
    its column so an unknown value is never turned into a NaN that reads as a failed test.
    """
    failed = pd.Series(False, index=df.index)
    unknown = pd.Series(False, index=df.index)

    def _apply(missing: pd.Series, ok: pd.Series) -> None:
        nonlocal failed, unknown
        missing = missing.astype(bool)
        failed |= ~missing & ~ok.astype(bool)
        unknown |= missing

    if des.min_tier is not None and "tier" in df.columns:
        floor = TIER_ORDER.index(des.min_tier)
        rank = df["tier"].map(_tier_rank)
        _apply(rank.isna(), rank.fillna(-1) <= floor)

    if des.taxonomy_include and "taxonomy" in df.columns:
        allowed = {str(c).strip().upper() for c in des.taxonomy_include}
        major = df["taxonomy"].map(taxonomy_major)
        _apply(major.isna(), major.isin(allowed))

    if "period_h" in df.columns and (des.min_period_h is not None or des.max_period_h is not None):
        period = pd.to_numeric(df["period_h"], errors="coerce")
        ok = pd.Series(True, index=df.index)
        if des.min_period_h is not None:
            ok &= period >= des.min_period_h
        if des.max_period_h is not None:
            ok &= period <= des.max_period_h
        _apply(period.isna(), ok)

    if "diameter_m" in df.columns and (des.min_diameter_m is not None or des.max_diameter_m is not None):
        diameter = pd.to_numeric(df["diameter_m"], errors="coerce")
        ok = pd.Series(True, index=df.index)
        if des.min_diameter_m is not None:
            ok &= diameter >= des.min_diameter_m
        if des.max_diameter_m is not None:
            ok &= diameter <= des.max_diameter_m
        _apply(diameter.isna(), ok)

    return failed.astype(bool), unknown.astype(bool)


def desirability_mask(df: pd.DataFrame, des: Desirability) -> pd.Series:
    """Boolean Series over ``df``: True where the target meets the desirability terms, counting
    an unknown value as met when ``des.keep_unknown`` and as unmet otherwise."""
    failed, unknown = desirability_masks(df, des)
    keep = ~failed if des.keep_unknown else ~failed & ~unknown
    return keep.astype(bool)


def apply_selection(df: pd.DataFrame, des: Desirability) -> pd.DataFrame:
    """Return ``df`` with a ``selected`` column: reachable and worth reaching.

    ``selected`` is the list handed to the solve step. It is never wider than ``reachable``, since
    a target has to be reachable first, and with nothing set it equals ``reachable``.

    ``uncharacterized`` marks the reachable rows that no known value ruled out but that are not
    characterized (no tier) or that a filter in force could not test. With ``keep_unknown`` they
    are selected; without it they are left out. Either way the count of measured matches is
    ``selected & ~uncharacterized``.
    """
    out = df.copy()
    reachable = out["reachable"] if "reachable" in out.columns else pd.Series(True, index=out.index)
    reachable = reachable.fillna(False).astype(bool)
    failed, unknown = desirability_masks(out, des)
    if "tier" in out.columns:
        # Not characterized at all: nothing about it has been looked up yet.
        unknown |= out["tier"].isna()
    out["uncharacterized"] = reachable & ~failed & unknown
    keep = ~failed if des.keep_unknown else ~failed & ~unknown
    out["selected"] = reachable & keep
    return out
