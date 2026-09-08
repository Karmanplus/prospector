"""Expected minimum perihelion distance over a body's dynamical past.

A near-Earth object's orbit is not stable on million-year timescales; it is fed in from the main
belt and random-walked by planetary encounters until it hits the Sun, is ejected, or strikes a
planet. How close it came to the Sun along the way governs whether volatiles and hydrated minerals
survived: below roughly 0.2 AU thermal cycling and cracking strip the surface, so a body that once
passed inside that distance is a poorer mining prospect than its present-day orbit suggests.

Toliou, Granvik & Tsirvoulis (2021) integrated a synthetic near-Earth population and tabulated, on
a grid of ``(a, e, i, H)``, the probability that a body reached each of 26 perihelion thresholds
from 0.05 to 1.3 AU. This module reads that table, finds the grid cell closest to a real body's
elements, and reduces its cumulative distribution to a single expected value.

The result is an expectation over a *population* that shares this body's orbit, not a
reconstruction of this body's own history, since two objects on identical orbits today can have had
very different pasts. Read it as a prior, so it informs the display rather than any rejection.
"""
from __future__ import annotations

import gzip
import warnings
from pathlib import Path

import numpy as np
import requests

from prospector.paths import DATA_DIR

# Published table. It is ~1.3 GB uncompressed, far too large to vendor, so it is fetched once on
# demand and cached alongside the other downloaded population data.
TOLIOU_URL = ("https://www.mv.helsinki.fi/home/mgranvik/data/"
              "Toliou+_2021_MNRAS/Toliou+_2021_MNRAS.dat.gz")
TOLIOU_FILENAME = "Toliou+_2021_MNRAS.dat"

# The grid axes, and the ranges used to make distances along them comparable. Without the
# normalisation, inclination in degrees would swamp semimajor axis in AU and the "closest"
# cell would be chosen almost entirely on inclination.
_AXIS_RANGES = np.array([4.2, 1.0, 180.0, 10.0])   # a (AU), e, i (deg), H (mag)

# Columns 5-30 hold P(q_min <= q_s) for q_s = 0.05, 0.10, ... 1.30 AU.
_THRESHOLD_COLUMNS = slice(4, 30)
_THRESHOLDS_AU = np.arange(0.05, 1.35, 0.05)


def toliou_path() -> Path:
    """Where the cached table lives. A function, not a constant, so the path is assembled
    per call, but note ``DATA_DIR`` itself is bound at import: redirecting
    ``paths.DATA_DIR`` does not move this. Pass an explicit path to relocate a cache."""
    return DATA_DIR / TOLIOU_FILENAME


def load_toliou_table(path: Path | None = None) -> np.ndarray | None:
    """The published grid as an array, downloading and caching it on first use.

    Returns None if the table is neither present nor fetchable. Perihelion history is one display
    column among many, so a body still enriches without it.
    """
    target = Path(path) if path is not None else toliou_path()
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.get(TOLIOU_URL, timeout=600)
            response.raise_for_status()
            target.write_bytes(gzip.decompress(response.content))
        except Exception:
            return None
    try:
        lines = target.read_text().splitlines(keepends=True)
        with warnings.catch_warnings():
            # A truncated file warns about having no data on its way to returning an empty array.
            # That is handled below as a normal "no table"; the warning would only surface as noise
            # somewhere far from the cause.
            warnings.simplefilter("ignore", UserWarning)
            # Three header lines describe the columns; the last is a trailing record marker.
            table = np.loadtxt(lines[:-1], skiprows=3)
    except Exception:
        return None
    # A truncated or wrong-format file parses without raising and yields something with no rows or
    # too few columns. Returning it would index past the end of every row, so it is rejected here
    # rather than surfacing as an error deep inside the lookup.
    if table.ndim != 2 or table.shape[0] == 0 or table.shape[1] < _THRESHOLD_COLUMNS.stop:
        return None
    return table


def expected_min_perihelion_au(a_au: float, ecc: float, inc_deg: float, abs_mag: float,
                               table: np.ndarray) -> float:
    """Expected minimum perihelion distance (AU) for a body at these elements.

    The nearest grid cell's cumulative curve is differenced into per-bin probabilities and averaged
    over the bin midpoints, giving the mean of the distribution rather than its mode, so a body
    with a small chance of a very close approach is scored accordingly.
    """
    elements = np.array([a_au, ecc, inc_deg, abs_mag], dtype=float)
    distances = np.sqrt(np.sum(((table[:, :4] - elements) / _AXIS_RANGES) ** 2, axis=1))
    cumulative = table[np.argmin(distances)][_THRESHOLD_COLUMNS]

    per_bin = np.diff(cumulative, prepend=0)
    midpoints = _THRESHOLDS_AU - 0.025
    return float(np.sum(midpoints * per_bin))
