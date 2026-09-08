"""Turning photometry into a spectrum, and reading the hydration band out of one.

Most small bodies have never had a spectrum taken, but many have been measured in two or more
broadband filters. A colour index is the difference of two magnitudes, and once the Sun's own
colour is divided out that difference gives the ratio of the body's reflectance in two bands.
Chaining such ratios across overlapping filter pairs reconstructs a coarse relative reflectance
curve: a few points instead of hundreds, but enough to place a body on the hydration axis when
nothing else exists.

The chaining is a graph walk. Each colour is an edge between two filters carrying a multiplicative
factor; a connected component of that graph can be normalised to any one of its filters.
Disconnected components (say B-V measured but only g-r otherwise) cannot be put on a common scale,
so they come back separately rather than being spliced on a guess.

:func:`band_depth_07um` then measures the 0.7 um absorption feature, the diagnostic for aqueously
altered phyllosilicates, the signature of a body that once held liquid water.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

# Solar magnitudes per filter (Willmer 2018 and references therein). Dividing these out is what
# converts an observed colour into a reflectance ratio rather than a flux ratio.
SOLAR_COLORS: dict[str, float] = {
    "U": 5.61, "B": 5.44, "V": 4.83, "R": 4.42, "I": 4.08,
    "g": 5.12, "r": 4.64, "i": 4.53, "z": 4.51, "Y": 4.43,
}

# Effective central wavelength per filter, in microns.
FILTER_WAVELENGTHS_UM: dict[str, float] = {
    "U": 0.365, "B": 0.445, "V": 0.551, "R": 0.658, "I": 0.806,
    "g": 0.475, "r": 0.622, "i": 0.763, "z": 0.905, "Y": 1.02,
}

# Filter pairs recognised as colour indices when scanning a source record. Restricting to known
# pairs keeps an arbitrary hyphenated column name from being read as photometry.
COLOR_PAIRS: frozenset[tuple[str, str]] = frozenset({
    ("B", "V"), ("V", "R"), ("R", "I"), ("V", "I"),
    ("g", "r"), ("g", "i"), ("r", "i"), ("i", "z"),
})


@dataclass(frozen=True)
class Color:
    """A colour index ``m(filter1) - m(filter2)`` for one body."""

    filter1: str
    filter2: str
    magnitude_difference: float

    def reflectance_ratio(self, solar_colors: Mapping[str, float] = SOLAR_COLORS) -> float:
        """Reflectance in ``filter1`` relative to ``filter2``, with the solar colour removed."""
        try:
            solar_delta = solar_colors[self.filter1] - solar_colors[self.filter2]
        except KeyError as exc:
            raise ValueError(f"no solar colour for filter '{exc.args[0]}'") from exc
        return 10 ** (-0.4 * (self.magnitude_difference - solar_delta))


def _ratio_graph(colors: Iterable[Color],
                 solar_colors: Mapping[str, float]) -> dict[str, list[tuple[str, float]]]:
    """Filters as nodes, colours as edges carrying ``R_neighbour = factor * R_current``."""
    graph: dict[str, list[tuple[str, float]]] = {}
    for color in colors:
        ratio = color.reflectance_ratio(solar_colors)     # R(filter1) / R(filter2)
        graph.setdefault(color.filter1, []).append((color.filter2, 1.0 / ratio))
        graph.setdefault(color.filter2, []).append((color.filter1, ratio))
    return graph


def colors_to_reflectance(
    colors: Iterable[Color],
    *,
    solar_colors: Mapping[str, float] | None = None,
    anchor_filter: str | None = "V",
    filter_wavelengths: Mapping[str, float] | None = None,
) -> list[dict[str, tuple[float, float | None]]]:
    """Relative reflectance per filter, one dict per connected group of filters.

    Each group is normalised to unity at ``anchor_filter`` when that filter is in the group, and
    otherwise at whichever filter the walk started from. Values are ``{filter: (reflectance,
    wavelength_um)}``; the wavelength is None for a filter with no tabulated effective wavelength,
    which the caller drops rather than guesses.
    """
    colors = list(colors)
    if not colors:
        raise ValueError("at least one colour measurement is required")

    solar = {**SOLAR_COLORS, **(solar_colors or {})}
    wavelengths = {**FILTER_WAVELENGTHS_UM, **(filter_wavelengths or {})}
    graph = _ratio_graph(colors, solar)

    visited: set[str] = set()

    def walk(start: str) -> dict[str, float]:
        reflectance = {start: 1.0}
        queue = deque([start])
        visited.add(start)
        while queue:
            current = queue.popleft()
            for neighbour, factor in graph.get(current, []):
                if neighbour in reflectance:
                    continue
                reflectance[neighbour] = reflectance[current] * factor
                queue.append(neighbour)
                visited.add(neighbour)
        return reflectance

    groups: list[dict[str, float]] = []
    if anchor_filter is not None and anchor_filter in graph:
        groups.append(walk(anchor_filter))         # anchored group first, so callers see it first
    for node in graph:
        if node not in visited:
            groups.append(walk(node))

    normalized: list[dict[str, tuple[float, float | None]]] = []
    for group in groups:
        anchor = anchor_filter if anchor_filter in group else next(iter(group))
        scale = group[anchor]
        normalized.append({filt: (value / scale, wavelengths.get(filt))
                           for filt, value in group.items()})
    return normalized


def colors_to_spectrum(colors: Iterable[Color], *, anchor_filter: str = "V"
                       ) -> tuple[list[float], list[float]]:
    """The best-connected colour group as wavelength-ordered ``(wave_um, reflectance)``.

    Raises when fewer than two filters end up with a known wavelength: a classifier needs a slope,
    and a single point has none.
    """
    groups = colors_to_reflectance(colors, anchor_filter=anchor_filter)
    if not groups:
        raise ValueError("no reflectance could be derived from the colours")
    group = next((g for g in groups if anchor_filter in g), groups[0])

    points = sorted((wavelength, value) for value, wavelength in group.values()
                    if wavelength is not None)
    if len(points) < 2:
        raise ValueError("not enough wavelength coverage to build a spectrum")
    return [p[0] for p in points], [p[1] for p in points]


# The 0.7 um feature is measured against a continuum drawn between 0.6 and 0.8 um, on a fixed
# resampling grid so the depth does not depend on how densely the source spectrum was sampled.
_BAND_SHORT_UM, _BAND_CENTRE_UM, _BAND_LONG_UM = 0.6, 0.7, 0.8
_RESAMPLE_GRID_UM = np.concatenate((np.arange(0.45, 1.05 + 0.025, 0.025),
                                    np.arange(1.1, 2.45 + 0.05, 0.05)))


def band_depth_07um(wave_um: Sequence[float], reflectance: Sequence[float]) -> float:
    """Depth of the 0.7 um absorption band: 0 for a featureless spectrum, higher when hydrated.

    The band marks Fe-bearing phyllosilicates, which form only in the presence of liquid water, so
    its depth is the most direct spectral evidence that a body was aqueously altered.
    """
    wave = np.asarray(wave_um, dtype=float)
    refl = np.asarray(reflectance, dtype=float)
    order = np.argsort(wave)
    resampled = np.interp(_RESAMPLE_GRID_UM, wave[order], refl[order])

    r_short, r_centre, r_long = (float(np.interp(x, _RESAMPLE_GRID_UM, resampled))
                                 for x in (_BAND_SHORT_UM, _BAND_CENTRE_UM, _BAND_LONG_UM))
    long_weight = ((_BAND_CENTRE_UM - _BAND_SHORT_UM)
                   / (_BAND_LONG_UM - _BAND_SHORT_UM))
    continuum = (1 - long_weight) * r_short + long_weight * r_long
    return float(1 - r_centre / continuum)
