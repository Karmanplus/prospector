"""Targeted small-body surveys: rotation periods and reflectance spectra.

Two surveys contribute what the big aggregated catalogues do not.

**Ondrejov** publishes a running table of photometrically determined rotation periods,
including many small near-Earth objects that reach the aggregators late or not at all. It is
one flat file for the whole survey, so it is fetched once and indexed.

**MANOS** (the Mission Accessible Near-Earth Object Survey) targets the same population this
tool screens, small low-delta-v bodies, and publishes reflectance spectra for them.
For a typical target it is the only spectrum in existence, and without it the hydration
classifier has nothing to work from.

Both are cached to disk on first use. Neither is required: an unreachable survey leaves those
properties unknown, which the filters treat as "keep".
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import requests

from prospector.enrichment.measurements import Measurement
from prospector.paths import DATA_DIR

ONDREJOV_URL = "https://space.asu.cas.cz/~ppravec/newres.txt"
MANOS_STATUS_URL = "https://manos.lowell.edu/api/v1/observations/statuses"

_TIMEOUT_S = 60.0

# Fixed-width column spans in the Ondrejov table: the body's designation, then its period.
_ONDREJOV_ID_SPAN = slice(0, 29)
_ONDREJOV_PERIOD_SPAN = slice(29, 44)

_ondrejov_index: dict[str, float] | None = None
_manos_index: dict[str, dict] | None = None


def cache_dir() -> Path:
    """Where these surveys cache. A function, not a constant, so the path is assembled
    per call, but note ``DATA_DIR`` itself is bound at import: redirecting
    ``paths.DATA_DIR`` does not move this. Pass an explicit path to relocate a cache."""
    return DATA_DIR / "enrichment" / "surveys"


def _cached_text(name: str, url: str) -> str | None:
    """The survey file, from disk if it is there and from the network otherwise."""
    path = cache_dir() / name
    if path.is_file():
        try:
            return path.read_text()
        except Exception:
            pass
    try:
        response = requests.get(url, timeout=_TIMEOUT_S)
        response.raise_for_status()
    except Exception:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text)
    except Exception:
        pass
    return response.text


def _canonical_designation(text: str) -> str | None:
    """The bare designation inside a survey's display name.

    Survey tables label a body however they please: ``(341843) 2008 EV5``, ``2008 EV5``, ``341843
    2008EV5``. The provisional designation is extracted when present, falling back to a
    parenthesised catalogue number, so the index keys match what a caller passes in.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    provisional = re.search(r"\b\d{4}\s?[A-Z]{2}\d*\b", text)
    if provisional:
        return provisional.group(0)
    numbered = re.search(r"\((\d+)\)", text)
    if numbered:
        return numbered.group(1)
    return text.strip()


def ondrejov_periods(refresh: bool = False) -> dict[str, float]:
    """Rotation period in hours by designation, for the whole Ondrejov table.

    Loaded once per process. Empty if the survey cannot be reached.
    """
    global _ondrejov_index
    if _ondrejov_index is not None and not refresh:
        return _ondrejov_index

    text = _cached_text("ondrejov.txt", ONDREJOV_URL)
    index: dict[str, float] = {}
    for line in (text or "").splitlines()[1:]:        # first line is the column header
        if not line.strip():
            continue
        designation = _canonical_designation(line[_ONDREJOV_ID_SPAN].strip())
        try:
            period = float(line[_ONDREJOV_PERIOD_SPAN].strip())
        except ValueError:
            continue
        # First entry wins: the table lists refinements after the original determination and the
        # leading row is the survey's own headline value.
        if designation and designation not in index:
            index[designation] = period
    _ondrejov_index = index
    return index


def ondrejov_period(identifier: str) -> list[Measurement]:
    """The Ondrejov rotation period for one body, as a measurement list (possibly empty)."""
    period = ondrejov_periods().get(identifier.strip())
    return [Measurement(period, source="ondrejov")] if period is not None else []


def manos_observations(refresh: bool = False) -> dict[str, dict]:
    """The MANOS observation-status table, indexed by primary designation."""
    global _manos_index
    if _manos_index is not None and not refresh:
        return _manos_index

    text = _cached_text("manos_statuses.json", MANOS_STATUS_URL)
    index: dict[str, dict] = {}
    try:
        for entry in json.loads(text or "[]"):
            designation = str(entry.get("primary_designation", "")).strip()
            if designation:
                index[designation.lower()] = entry
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    _manos_index = index
    return index


def _spectrum_urls(entry: dict) -> list[str]:
    """Data-file URLs for a MANOS record, best spectral coverage first."""
    urls = []
    for key in ("vis_spectrum", "nir_spectrum", "color_spectrum"):
        spectrum = entry.get(key)
        if not spectrum or not spectrum.get("exists"):
            continue
        url = (spectrum.get("products") or {}).get("data")
        if isinstance(url, str) and url.endswith(".dat"):
            urls.append(url)
    return urls


def _parse_spectrum(text: str) -> tuple[list[float], list[float], list[float]]:
    """A MANOS ``.dat`` spectrum: a metadata block, ``###``, then wavelength/reflectance rows."""
    lines = text.splitlines()
    if "###" not in lines:
        raise ValueError("spectrum file has no ### separator")
    wave: list[float] = []
    reflectance: list[float] = []
    error: list[float] = []
    for line in lines[lines.index("###") + 1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            values = [float(p) for p in parts[:3]]
        except ValueError:
            continue
        wave.append(values[0])
        reflectance.append(values[1])
        error.append(values[2])
    return wave, reflectance, error


def manos_spectrum(identifier: str):
    """The best MANOS spectrum for one body as ``(wave_um, reflectance, error)``, or None.

    "Best" is the file with the most points: a near-infrared spectrum covers more of the diagnostic
    range than a handful of colour-derived points, and more points give the classifier's resampling
    less to interpolate across.
    """
    entry = manos_observations().get(identifier.strip().lower())
    if not entry:
        return None

    target_dir = cache_dir() / "manos" / re.sub(r"[^0-9A-Za-z]+", "_", identifier).strip("_")
    best = None
    for url in _spectrum_urls(entry):
        path = target_dir / Path(url.split("?")[0]).name
        if not path.is_file():
            try:
                response = requests.get(url, timeout=_TIMEOUT_S)
                response.raise_for_status()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(response.content)
            except Exception:
                continue
        try:
            wave, reflectance, error = _parse_spectrum(path.read_text())
        except Exception:
            continue
        if len(wave) >= 2 and (best is None or len(wave) > len(best[0])):
            # An all-zero error column means "not reported", not "perfectly measured".
            best = (wave, reflectance, error if any(error) else None)
    return best
