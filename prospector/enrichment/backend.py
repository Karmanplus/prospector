"""Describing one target: where every source and model comes together.

:func:`characterize` is the whole of this for a single body: identify it, gather what every
catalogue and survey says about it, run the models over those measurements, and boil the result
down to one letter grade. :func:`run` does that for many bodies at once.

Two rules shape the error handling throughout.

A source that fails costs one property, never a target. Every lookup here is allowed to come back
empty, and the row is built from whatever did arrive. The filters keep rows whose properties are
unknown, so a target described only halfway is still a target. A target dropped because a survey
timed out is the mistake this tool cannot afford.

A target that fails costs one target, never the run. Bodies are handled independently, and one that
cannot be identified at all is skipped, so a single bad designation in a screen of four hundred
does not take the other three hundred and ninety-nine with it.

The work is nearly all waiting on the network, being several requests per body and hardly any
arithmetic, so targets run on a thread pool. The one big download they share, the reference spin
catalogue, is fetched once before the pool starts rather than raced for by every worker.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from prospector.enrichment import tiers
from prospector.enrichment.measurements import (
    consolidate,
    preferred,
    preferred_float,
    preferred_str,
)
from prospector.enrichment.models import cohesion, perihelion, sizing, spin
from prospector.enrichment.sources import astorb, overrides, ssodnet, surveys

# Concurrent targets, the vehicle sweep's rule (one per core, less one). Each target is mostly
# waiting on two remote catalogues, at 4-5 s a body; four at a time made a 5,000 batch an
# eighty-minute job. Measured at 23 the services answered without errors or slowdown.
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 1)

# Bodies whose remote lookups are fetched together before the pool finishes them. Each bulk
# stage opens a connection per body, and a few hundred at once is where the server starts
# dropping them.
PREFETCH_CHUNK = 100


def _prefetch(fetch, chunk: list[str]) -> dict:
    """A source's bulk lookup for ``chunk``; an empty dict if it fails, so each body falls back
    to its own lookup. A source reporting itself unavailable ends the run: the package was
    checked up front, so this is the remote service, and caching bodies characterized without it
    would record nothing measured for all of them."""
    try:
        return fetch(chunk) or {}
    except ssodnet.SourceUnavailable as exc:
        raise EnrichmentFailed(str(exc)) from exc
    except Exception:
        return {}

# Cohesion the critical spin is computed for (Pa). One pascal is the boundary between a body that
# is gravity-bound and one that needs real strength, so the critical period is "the spin at which
# this body would start to need to be more than a rubble pile".
_CRITICAL_COHESION_PA = 1.0


class EnrichmentUnavailable(RuntimeError):
    """Enrichment cannot run here: an optional dependency is missing.

    Reachability is unaffected: the screen, the solvers and the trajectory all work without this,
    so it makes the app less capable rather than stopping it.
    """


class EnrichmentFailed(RuntimeError):
    """Enrichment ran but characterized nothing."""

    def __init__(self, message: str, debug: str = ""):
        super().__init__(message)
        self.debug = debug


def _hydration_class(identifier: str, albedo: float | None, taxonomy: str | None,
                     colors: dict) -> str | None:
    """The body's hydration class, or None when it cannot be determined.

    Imported here rather than at module scope: the classifier needs published spectra, which come
    from an optional package, and enrichment must not fail to import without it.
    """
    try:
        from prospector.enrichment.models import hydration
        from prospector.enrichment.models.reflectance import COLOR_PAIRS, Color
    except ImportError:
        return None

    measurements = []
    for key, values in (colors or {}).items():
        filters = tuple(part.strip() for part in key.split("-", 1))
        value = preferred_float(values)
        if len(filters) == 2 and filters in COLOR_PAIRS and value is not None:
            measurements.append(Color(filters[0], filters[1], value))

    try:
        spectrum = surveys.manos_spectrum(identifier)
        best, _probabilities = hydration.classify(
            identifier,
            albedo=sizing.albedo_for_sizing(albedo, taxonomy),
            colors=measurements or None,
            spectrum=spectrum)
        return best
    except Exception:
        return None      # display-only; a body is never dropped for want of a hydration class


def characterize(identifier: str, *, spin_catalog=None, toliou_table=None,
                 body=None, catalog_data=None) -> dict | None:
    """Everything enrichment knows about one body, or None if it could not be resolved.

    ``spin_catalog`` and ``toliou_table`` are the two population-wide datasets the models need.
    They are passed in rather than fetched here because they are identical for every target;
    omitting them makes this function self-contained at the cost of loading them per call.
    ``body`` and ``catalog_data`` are the two remote lookups when :func:`run` has already fetched
    them in bulk; left None they are fetched here, one body at a time.
    """
    if body is None:
        body = ssodnet.lookup(identifier)
    if not body.resolved:
        return None

    if catalog_data is None:
        catalog_data = astorb.lookup(identifier)
    override_data = overrides.for_target(identifier)

    abs_mag = preferred_float(consolidate(body.abs_mag, override_data.get("abs_mag")))
    albedo = preferred_float(consolidate(body.albedo, override_data.get("albedo"),
                                         catalog_data["albedo"]))
    taxonomy = preferred_str(consolidate(body.taxonomy, override_data.get("taxonomy"),
                                         catalog_data["taxonomy"]))
    period_h = preferred_float(consolidate(body.period_h, override_data.get("period_h"),
                                           catalog_data["period_h"],
                                           surveys.ondrejov_period(identifier)))
    tumbling = bool(preferred(consolidate(catalog_data["tumbling"])))

    diameter_m = sizing.diameter_m(abs_mag, albedo, taxonomy)

    # Both structural measures need a size. Without one there is nothing to say about whether the
    # body holds together, and the tier treats that as unknown rather than as a failure.
    cohesion_pa = stability = None
    if diameter_m is not None:
        if period_h is not None:
            cohesion_pa = cohesion.min_cohesion_pa(diameter_m / 2.0, period_h)
        critical_period_h = cohesion.critical_period_h(diameter_m / 2.0, _CRITICAL_COHESION_PA)
        magnitudes, periods = spin_catalog if spin_catalog is not None else astorb.spin_catalog()
        distribution = spin.fit_period_distribution(abs_mag, magnitudes, periods)
        stability = spin.probability_slower_than(distribution, critical_period_h)

    q_min_au = None
    elements = (body.semi_major_axis_au, body.eccentricity, body.inclination_deg, abs_mag)
    if toliou_table is not None and all(value is not None for value in elements):
        q_min_au = perihelion.expected_min_perihelion_au(*elements, toliou_table)

    tier = overrides.tier_override(identifier) or tiers.assign(
        taxonomy=taxonomy, albedo=albedo, cohesion_pa=cohesion_pa,
        stability=stability, period_h=period_h, tumbling=tumbling)

    return {
        "input_id": body.input_id,
        "name": body.name,
        "tier": tier,
        "taxonomy": taxonomy,
        "albedo": albedo,
        "period_h": period_h,
        "diameter_m": diameter_m,
        "cohesion_pa": cohesion_pa,
        "p_gt_pcrit": stability,
        "q_min_au": q_min_au,
        "hydration_class": _hydration_class(identifier, albedo, taxonomy, body.colors),
    }


def _preflight() -> None:
    """Fail fast and clearly when the desirability axis simply cannot run here."""
    try:
        import rocks  # noqa: F401  (checking the optional dependency is installed)
    except ImportError as exc:
        raise EnrichmentUnavailable(
            "enrichment needs the 'rocks' package to resolve small-body identities; "
            "install the enrichment extra to enable the desirability axis"
        ) from exc


def run(
    identifiers: list[str],
    *,
    n_workers: int | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    on_result: Callable[[pd.DataFrame], None] | None = None,
    should_continue: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    """Characterize ``identifiers`` and return one row per body that resolved.

    ``on_result(row_frame)`` fires as each body finishes, so the caller can cache it straight away
    and stopping partway loses at most the targets still running. ``should_continue()`` is checked
    as each target is picked up: once it turns False the queued targets return without doing any
    work and the running ones finish, so a job that has been replaced winds down within one target
    rather than running the whole batch out.

    Raises :class:`EnrichmentUnavailable` if the optional dependencies are absent, or
    :class:`EnrichmentFailed` if every target failed to resolve and the run was not superseded.
    """
    _preflight()

    seen: set[str] = set()
    ids = [s for s in (str(x).strip() for x in identifiers)
           if s and not (s in seen or seen.add(s))]
    if not ids:
        return pd.DataFrame(columns=["input_id"])

    # The two population-wide datasets, fetched once for the whole run. Both degrade to None/empty
    # rather than raising, so a target still characterizes without them.
    spin_catalog = astorb.spin_catalog()
    toliou_table = perihelion.load_toliou_table()

    total = len(ids)
    workers = max(1, min(n_workers or DEFAULT_WORKERS, total))
    lock = threading.Lock()
    done = 0
    rows: list[dict] = []
    failures: list[str] = []
    superseded = False

    def _one(identifier: str, body, catalog_data) -> dict | None:
        if should_continue is not None and not should_continue():
            return None
        try:
            return characterize(identifier, spin_catalog=spin_catalog, toliou_table=toliou_table,
                                body=body, catalog_data=catalog_data)
        except ssodnet.SourceUnavailable:
            raise
        except Exception as exc:
            with lock:
                failures.append(f"{identifier}: {type(exc).__name__}: {exc}")
            return None

    # The two remote lookups are fetched for a chunk at a time in bulk (one resolution and one
    # concurrent fetch per chunk instead of several round trips per body), then each body is
    # finished on the pool. Chunking keeps progress flowing and a superseded job stopping soon.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, total, PREFETCH_CHUNK):
            if should_continue is not None and not should_continue():
                superseded = True
                break
            chunk = ids[start:start + PREFETCH_CHUNK]
            bodies = _prefetch(ssodnet.lookup_many, chunk)
            catalog = _prefetch(astorb.lookup_many, chunk)
            futures = {pool.submit(_one, identifier, bodies.get(identifier),
                                   catalog.get(identifier)): identifier
                       for identifier in chunk}
            for future in as_completed(futures):
                row = future.result()
                with lock:
                    done += 1
                    if row is not None:
                        rows.append(row)
                    progress = done
                if row is not None and on_result is not None:
                    on_result(pd.DataFrame([row]))
                if on_progress:
                    on_progress(progress, total, f"characterized {progress}/{total}")
                if should_continue is not None and not should_continue():
                    superseded = True

    if not rows and not superseded:
        raise EnrichmentFailed(
            "no targets could be characterized (the small-body catalogues may be unreachable)",
            debug="\n".join(failures[:50]))
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["input_id"])
