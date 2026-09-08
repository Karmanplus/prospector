"""Measurements, who made them, and the rule for choosing between them.

Every property collected here has usually been measured several times, by different surveys that
disagree: reflectivity, spectral class, rotation period, brightness. They are not
merged into one number as they arrive. Each keeps its value, who measured it, and whether that
survey marked it as the one it prefers, so the choice happens once, in one place, in
:func:`preferred`.

The rule: an explicitly preferred measurement wins; failing that, the last one in the list. Order
therefore carries meaning, and :data:`SOURCE_ORDER` fixes it. Later sources are the larger, more
systematically curated databases, which is the better guess when nobody has declared a preference.
A local override is always marked preferred, so it wins outright.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

# Sources in increasing order of what to fall back on when nothing is flagged preferred.
# Deliberately not alphabetical: it is a ranking, and reordering it changes which value the tool
# reports for a body measured twice.
SOURCE_ORDER: tuple[str, ...] = ("ssodnet", "override", "astorb", "ondrejov")

# Case-insensitive strings the upstream catalogues use to mean "no value". They arrive as text
# because the sources are text, and coercing them numerically would turn a stated absence into a
# NaN indistinguishable from a failed parse.
MISSING_TOKENS = frozenset({"", "n/a", "na", "nan", "none", "null", "-"})


@dataclass(frozen=True)
class Measurement:
    """One survey's determination of one property."""

    value: Any
    source: str
    preferred: bool = False


def is_missing(value: Any) -> bool:
    """Whether a value carries no information: absent, NaN, or a stated-absence token."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip().lower() in MISSING_TOKENS


def major_class(taxonomy: Any) -> str | None:
    """The major spectral class letter (``"Cb"`` -> ``"C"``), or None if unknown.

    Keyed off the first alphabetic character so that a subclass, a compound class, or a
    parenthesised note all reduce to the complex the composition filters are expressed in.
    """
    if is_missing(taxonomy):
        return None
    for character in str(taxonomy).strip():
        if character.isalpha():
            return character.upper()
    return None


def _usable(value: Any) -> bool:
    """Whether a measurement is worth returning: it has a value that means something."""
    return not is_missing(value)


def preferred(measurements: Sequence[Measurement] | None) -> Any:
    """The value to use for a property, or None when nothing usable was measured."""
    if not measurements:
        return None
    for measurement in measurements:
        if measurement.preferred and _usable(measurement.value):
            return measurement.value
    for measurement in reversed(measurements):
        if _usable(measurement.value):
            return measurement.value
    return None


def preferred_float(measurements: Sequence[Measurement] | None) -> float | None:
    """:func:`preferred`, coerced to a float, or None when absent or non-numeric."""
    value = preferred(measurements)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def preferred_str(measurements: Sequence[Measurement] | None) -> str | None:
    """:func:`preferred`, coerced to a non-empty string, or None."""
    value = preferred(measurements)
    if not isinstance(value, str):
        return None
    return value.strip() or None


def consolidate(*groups: Iterable[Measurement] | None) -> list[Measurement]:
    """Concatenate one property's measurements from several sources into :data:`SOURCE_ORDER`.

    Sorting is stable, so measurements from one source keep the order that source reported them in,
    and any source not named in :data:`SOURCE_ORDER` sorts last, so a new source is trusted as a
    fallback ahead of nothing, rather than being dropped.
    """
    combined = [m for group in groups if group for m in group]
    rank = {name: index for index, name in enumerate(SOURCE_ORDER)}
    return sorted(combined, key=lambda m: rank.get(m.source, len(rank)))
