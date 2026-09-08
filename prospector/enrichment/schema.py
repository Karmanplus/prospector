"""The normalized enrichment column contract.

Whatever produced the data, whether the models in this package, a row cached months ago, or a table
handed in from outside, it goes through :func:`normalize` into one fixed set of columns, so the
rest of the app never has to know where a number came from.

Missing or uncertain values become NaN for numbers and None for text, so a filter downstream can
handle "unknown" explicitly and keep the target instead of quietly dropping it. That is what stops
a gap in the data from reading like a failed test.

Units: rotation period in hours, diameter in meters, cohesion in pascals, q_min in AU.
"""
from __future__ import annotations

import pandas as pd

from prospector.enrichment.measurements import is_missing
from prospector.enrichment.measurements import major_class as taxonomy_major
from prospector.enrichment.tiers import TIER_ORDER

__all__ = ["ENRICH_COLUMNS", "TIER_ORDER", "normalize", "taxonomy_major"]

# The join key carried verbatim from the caller (prospector's `pdes`) plus the physical properties
# the filters and plots read. `name` is the resolved name, kept for display only, never as a join
# key, since it diverges from the input for named bodies.
ENRICH_COLUMNS = [
    "input_id",      # the identifier passed in (prospector pdes); the join key
    "name",          # the resolved name, for display only
    "tier",          # mission-target tier, one of TIER_ORDER
    "taxonomy",      # spectral taxonomy string (e.g. "C", "Cb", "X")
    "albedo",        # geometric albedo (0-1)
    "period_h",      # rotation period (hours)
    "diameter_m",    # diameter (meters), measured-albedo-derived where possible
    "cohesion_pa",   # minimum cohesion to resist rotational disruption (Pa)
    "p_gt_pcrit",    # P(rotation period > critical period): structural-stability margin
    "q_min_au",      # expected minimum perihelion distance (AU)
    "hydration_class",  # hydration / aqueous-alteration class (composition proxy)
]

# Unit-free names some upstream sources use for the same quantities, mapped onto the contract. The
# contract names carry their unit and these do not, which is why they are translated at the
# boundary rather than adopted. normalize() is idempotent: against a frame that already uses the
# contract names these renames no-op.
_RAW_ALIASES = {
    "period": "period_h",
    "diameter": "diameter_m",
    "cohesion": "cohesion_pa",
    "p_crit_cdf": "p_gt_pcrit",
    "q_min": "q_min_au",
    "hydra_class": "hydration_class",
}

# Columns that must end up numeric; their source strings may carry sentinels for "missing".
_NUMERIC_COLUMNS = ["albedo", "period_h", "diameter_m", "cohesion_pa", "p_gt_pcrit", "q_min_au"]


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw enrichment frame onto :data:`ENRICH_COLUMNS`.

    Renames the known unit-free aliases, turns upstream "missing" sentinels into NaN/None, coerces
    the numeric columns, normalizes the tier to a known letter, and guarantees every contract
    column is present. Extra columns the backend supplies are preserved. Idempotent.
    """
    df = raw.copy()

    for src, dst in _RAW_ALIASES.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    if "input_id" not in df.columns:
        raise ValueError("enrichment frame is missing the 'input_id' join key")

    for col in _NUMERIC_COLUMNS:
        if col in df.columns:
            cleaned = df[col].map(lambda v: None if is_missing(v) else v)
            df[col] = pd.to_numeric(cleaned, errors="coerce")

    if "tier" in df.columns:
        df["tier"] = df["tier"].map(
            lambda t: None if is_missing(t) else (
                str(t).strip().upper() if str(t).strip().upper() in TIER_ORDER else None
            )
        )

    for col in ("name", "taxonomy", "hydration_class"):
        if col in df.columns:
            df[col] = df[col].map(lambda v: None if is_missing(v) else str(v).strip())

    for col in ENRICH_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Contract columns first (in order), then any backend extras.
    extras = [c for c in df.columns if c not in ENRICH_COLUMNS]
    return df[ENRICH_COLUMNS + extras]
