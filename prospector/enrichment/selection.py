"""Narrowing down to the targets worth going to. Touches nothing outside.

Reachability says a target can be reached; this says it is worth reaching. The module turns a
:class:`~prospector.config.Desirability`, which holds a minimum grade plus composition, spin and
size terms, into a true/false mask over the described population, and combines it with the
reachability mask into the single ``selected`` column the solve step reads.

The rule throughout: never quietly drop a reachable target. Every filter keeps rows whose property
is unknown, because missing data is not grounds for rejection. Tightening is something the user
chooses; a filter left unset keeps everything, so ``selected`` is just ``reachable`` until a real
constraint is applied.
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


def desirability_mask(df: pd.DataFrame, des: Desirability) -> pd.Series:
    """Boolean Series over ``df``: True where the target meets the desirability terms.

    A filter applies only when its term is set and its column exists; a table that has not been
    described yet has no such columns, so nothing is filtered. Within a filter that does apply,
    rows whose value is unknown are kept, and only a known value that fails the test is dropped.

    Each filter runs in one pass over its column so an unknown value is never turned into a NaN
    that could be mistaken for a failed test, which would quietly drop a reachable target.
    """
    mask = pd.Series(True, index=df.index)

    if des.min_tier is not None and "tier" in df.columns:
        floor = TIER_ORDER.index(des.min_tier)

        def _keep_tier(tier):
            rank = _tier_rank(tier)
            return True if rank is None else rank <= floor

        mask &= df["tier"].map(_keep_tier)

    if des.taxonomy_include and "taxonomy" in df.columns:
        allowed = {str(c).strip().upper() for c in des.taxonomy_include}

        def _keep_taxonomy(taxonomy):
            major = taxonomy_major(taxonomy)
            return True if major is None else major in allowed

        mask &= df["taxonomy"].map(_keep_taxonomy)

    if "period_h" in df.columns and (des.min_period_h is not None or des.max_period_h is not None):
        period = pd.to_numeric(df["period_h"], errors="coerce")
        if des.min_period_h is not None:
            mask &= period.isna() | (period >= des.min_period_h)
        if des.max_period_h is not None:
            mask &= period.isna() | (period <= des.max_period_h)

    if "diameter_m" in df.columns and (des.min_diameter_m is not None or des.max_diameter_m is not None):
        diameter = pd.to_numeric(df["diameter_m"], errors="coerce")
        if des.min_diameter_m is not None:
            mask &= diameter.isna() | (diameter >= des.min_diameter_m)
        if des.max_diameter_m is not None:
            mask &= diameter.isna() | (diameter <= des.max_diameter_m)

    return mask.astype(bool)


def apply_selection(df: pd.DataFrame, des: Desirability) -> pd.DataFrame:
    """Return ``df`` with a ``selected`` column: reachable and worth reaching.

    ``selected`` is the list handed to the solve step. It is never wider than ``reachable``, since
    a target has to be reachable first, and with nothing set it equals ``reachable``.
    """
    out = df.copy()
    reachable = out["reachable"] if "reachable" in out.columns else pd.Series(True, index=out.index)
    out["selected"] = reachable.fillna(False).astype(bool) & desirability_mask(out, des)
    return out
