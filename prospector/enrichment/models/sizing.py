"""Diameter from absolute magnitude and albedo.

The standard photometric relation: a body's absolute magnitude H fixes the product of its size and
its reflectivity, so a diameter follows once an albedo is assumed. Most small bodies have no
measured albedo, so the albedo used is, in order of preference:

1. a measured value (from any of the enrichment sources), 2. the median albedo of the body's
spectral complex, if a taxonomy is known, 3. :data:`DEFAULT_ALBEDO`, an all-population compromise.

Steps 2 and 3 are guesses and the diameter inherits their error, roughly a factor
``sqrt(assumed / true)``, so a C-type sized at the default 0.15 comes out about 1.6x too
small. That direction matters for the screen: sizing a dark body with a bright albedo
*under*-states its diameter, which is the safe direction for a minimum-size filter (it keeps
the target as "possibly too small" instead of dropping a body that is in fact large).
"""
from __future__ import annotations

import numpy as np

# Median geometric albedo per spectral complex (Marsset et al.). Used to size a body whose taxonomy
# is known but whose albedo was never measured.
ALBEDO_BY_COMPLEX: dict[str, float] = {
    "A": 0.27, "B": 0.07, "C": 0.06, "P": 0.05, "D": 0.09, "S": 0.26,
    "Q": 0.24, "K": 0.16, "L": 0.18, "V": 0.35, "E": 0.51, "M": 0.14,
}

# Used when neither an albedo nor a taxonomy is available: near the small-body population median,
# between the dark carbonaceous and bright silicate complexes.
DEFAULT_ALBEDO = 0.15


def diameter_km(abs_mag: float, albedo: float) -> float:
    """Diameter in kilometres from absolute magnitude ``abs_mag`` and geometric ``albedo``."""
    return 10 ** (3.1236 - 0.5 * np.log10(albedo) - 0.2 * abs_mag)


def albedo_for_sizing(albedo: float | None, taxonomy: str | None) -> float:
    """The albedo to size a body with: measured, else its complex median, else the default."""
    if albedo is not None and not (isinstance(albedo, float) and np.isnan(albedo)):
        return float(albedo)
    if isinstance(taxonomy, str) and taxonomy.strip():
        return ALBEDO_BY_COMPLEX.get(taxonomy.strip()[0].upper(), DEFAULT_ALBEDO)
    return DEFAULT_ALBEDO


def diameter_m(abs_mag: float | None, albedo: float | None,
               taxonomy: str | None = None) -> float | None:
    """Diameter in metres, or None when no absolute magnitude is available.

    The albedo falls back through :func:`albedo_for_sizing`, so a body with an H and nothing else
    still gets a diameter. It is coarse, but the screen's size filter keeps unknowns anyway and a
    guessed diameter is more useful than none.
    """
    if abs_mag is None or (isinstance(abs_mag, float) and np.isnan(abs_mag)):
        return None
    return float(diameter_km(abs_mag, albedo_for_sizing(albedo, taxonomy)) * 1000.0)
