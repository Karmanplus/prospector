"""Working out which targets are worth going to, as opposed to which can be reached.

``solvers/edelbaum`` says which targets a vehicle can reach. This package looks at those survivors
and describes what makes one worth mining: what it is made of, how fast it spins, how big it is,
and how well it holds together. The user can then narrow down to the good ones, and the shorter
list feeds the solve step as the reachable list did.

The package is arranged so the expensive, unreliable part stays in one place:

- :mod:`sources`      -- one module per outside catalogue or survey, each allowed to fail
- :mod:`models`       -- the physics and statistics over those measurements, offline
- :mod:`tiers`        -- the letter grade, and the reasoning behind it
- :mod:`backend`      -- putting it together: sources in, one described body out
- :mod:`schema`       -- the fixed set of columns everything downstream reads
- :mod:`selection`    -- the filter itself, which touches nothing outside

Only :func:`enrich` goes to the network, and only the sources can fail. The filters and the column
set do not, so the parts of the app that use them can be tested without any of this, and an install
missing the optional dependencies still has the reachability screen.
"""
from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from prospector.enrichment import backend
from prospector.enrichment.backend import EnrichmentFailed, EnrichmentUnavailable
from prospector.enrichment.schema import ENRICH_COLUMNS, TIER_ORDER, normalize, taxonomy_major
from prospector.enrichment.selection import apply_selection, desirability_mask
from prospector.paths import DATA_DIR

# Enrichment caches **per object** (one file per identifier), not per requested set, so that
# re-screening an overlapping reachable set reuses everything already fetched and only the
# newly-reachable targets hit the network. Re-screens therefore refetch very little.
_DEFAULT_CACHE_DIR = DATA_DIR / "enrichment"
_CACHE_SCHEMA = "v3"  # bump when ENRICH_COLUMNS / normalize change, to invalidate old caches

__all__ = [
    "enrich",
    "load_cached",
    "cached_ids",
    "EnrichmentUnavailable",
    "EnrichmentFailed",
    "ENRICH_COLUMNS",
    "TIER_ORDER",
    "normalize",
    "taxonomy_major",
    "apply_selection",
    "desirability_mask",
]


def _obj_cache_path(identifier: str, cache_dir: str | Path | None) -> Path:
    base = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
    slug = re.sub(r"[^0-9A-Za-z]+", "_", identifier).strip("_") or "id"
    digest = hashlib.sha1(identifier.encode()).hexdigest()[:8]   # disambiguate slug collisions
    return base / _CACHE_SCHEMA / f"{slug}__{digest}.parquet"


def _cache_rows(frame: pd.DataFrame, cache_dir: str | Path | None) -> dict[str, pd.DataFrame]:
    """Write each row of a normalized frame to its per-object cache file; return them by id."""
    out: dict[str, pd.DataFrame] = {}
    for identifier, group in frame.groupby(frame["input_id"].astype(str)):
        row = group.head(1).reset_index(drop=True)
        out[identifier] = row
        path = _obj_cache_path(identifier, cache_dir)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            row.to_parquet(path, index=False)
        except Exception:
            pass  # caching is best-effort; never fail enrichment over it
    return out


def load_cached(identifiers: list[str], cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Return the normalized rows already on disk for ``identifiers`` (no network, no fetch).

    The screen calls this once a background enrichment job has finished to pull the results out of
    the per-object cache without re-characterizing anything. Uncached ids are simply absent.
    """
    seen: set[str] = set()
    ids = [s for s in (str(x).strip() for x in identifiers) if s and not (s in seen or seen.add(s))]
    frames = []
    for identifier in ids:
        path = _obj_cache_path(identifier, cache_dir)
        if path.exists():
            try:
                frames.append(pd.read_parquet(path))
            except Exception:
                pass
    if not frames:
        return normalize(pd.DataFrame(columns=["input_id"]))
    return normalize(pd.concat(frames, ignore_index=True))


def cached_ids(identifiers: list[str], cache_dir: str | Path | None = None) -> set[str]:
    """The identifiers already characterized on disk (no network, no read)."""
    return {s for s in (str(x).strip() for x in identifiers)
            if s and _obj_cache_path(s, cache_dir).exists()}


def enrich(
    identifiers: list[str],
    *,
    on_progress: Callable[[int, int, str], None] | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    n_workers: int | None = None,
    should_continue: Callable[[], bool] | None = None,
    **backend_kwargs,
) -> pd.DataFrame:
    """Characterize ``identifiers`` (prospector ``pdes`` strings) and return a normalized frame.

    Per-object cached: identifiers already on disk load instantly; only the rest are characterized,
    and each is cached the moment it completes, so a long run checkpoints and a re-run refetches
    little. ``on_progress(done, total, msg)`` reports per-target counts across the whole request
    (cached + freshly done). Raises :class:`EnrichmentUnavailable` if the optional dependencies are
    absent, or :class:`EnrichmentFailed` if nothing could be characterized, unless some results are
    already cached, which are then kept.
    """
    seen: set[str] = set()
    ids = [s for s in (str(x).strip() for x in identifiers)
           if s and not (s in seen or seen.add(s))]   # dedupe, preserve order
    if not ids:
        return normalize(pd.DataFrame(columns=["input_id"]))

    cached: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for identifier in ids:
        path = _obj_cache_path(identifier, cache_dir)
        if path.exists() and not refresh:
            try:
                cached[identifier] = pd.read_parquet(path)
                continue
            except Exception:
                pass  # unreadable cache entry -> refetch this one
        missing.append(identifier)

    total, base = len(ids), len(cached)
    if on_progress and base:
        on_progress(base, total, f"characterized {base}/{total} (cached)")

    fresh: dict[str, pd.DataFrame] = {}
    fresh_lock = threading.Lock()
    if missing:
        def _progress(done: int, _missing_total: int, _msg: str) -> None:
            if on_progress:
                got = base + done
                on_progress(got, total, f"characterized {got}/{total}")

        def _on_result(raw_slice: pd.DataFrame) -> None:
            # Called per target from concurrent worker threads; serialize the dict update.
            cached_rows = _cache_rows(normalize(raw_slice), cache_dir)
            with fresh_lock:
                fresh.update(cached_rows)

        try:
            backend.run(missing, on_progress=_progress, on_result=_on_result,
                        n_workers=n_workers, should_continue=should_continue, **backend_kwargs)
        except (EnrichmentUnavailable, EnrichmentFailed):
            # The new identifiers couldn't be characterized. Keep whatever is cached or was
            # captured before the failure; only propagate when there is nothing to fall back on.
            if not cached and not fresh:
                raise

    by_id = {**cached, **fresh}
    frames = [by_id[i] for i in ids if i in by_id]
    if not frames:
        return normalize(pd.DataFrame(columns=["input_id"]))
    return normalize(pd.concat(frames, ignore_index=True))
