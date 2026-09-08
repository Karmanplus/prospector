"""What rotation period a body of a given size is likely to have.

A single published rotation period can be wrong, ambiguous by a factor of two, or missing entirely,
so testing one measurement against the disruption limit gives a brittle yes/no. The alternative
used here is statistical: gather the measured periods of every body of similar size, fit a
distribution to them, and ask what fraction of that population spins more slowly than this body's
critical period (:mod:`.cohesion`). The answer, ``P(period > P_crit)``, is a margin on structural
stability rather than a verdict, and it degrades gracefully: a body with no period of its own still
gets one.

"Similar size" is done through absolute magnitude, within :data:`MAGNITUDE_WINDOW`, because H is
known for essentially every catalogued body while diameter is not.

The fitted family is Johnson's S_B, chosen for its bounded support and skew: rotation periods have
a hard floor near the spin barrier and a long slow tail, which the usual two-parameter families
reproduce poorly.
"""
from __future__ import annotations

import numpy as np
import scipy.stats

# Half-width of the absolute-magnitude window defining "bodies of similar size" (mag). Wide enough
# to gather a fittable sample for sparsely-observed sizes, narrow enough that the sample still
# reflects this body's size class.
MAGNITUDE_WINDOW = 0.75

# Passes of the outlier clip below. Two is enough to remove the handful of extreme slow rotators;
# more starts eating the genuine tail.
CLIP_PASSES = 2


class UnknownSpinDistribution:
    """Stands in for a distribution that could not be fitted.

    Its CDF is NaN, so a stability margin computed from it is NaN rather than a number -- "we do
    not know" propagates instead of turning into a passing or failing score.
    """

    def cdf(self, _value):
        return float("nan")


def _clip_slow_tail(periods: np.ndarray) -> np.ndarray:
    """Drop extreme slow rotators, which drag the fit far more than they inform it.

    A handful of catalogued bodies rotate on timescales of weeks. Their tail is real but so long
    that including it dominates the fitted shape for every body in the window.

    Each pass keeps periods below twice the sample's standard deviation, recomputing the spread on
    what survived. Note that the bound is twice the *spread*, not the mean plus twice the spread,
    so how much it removes depends on how skewed the sample is: on the real catalogue, whose long
    tail makes the standard deviation large, it trims only the extremes. On a tightly clustered
    sample it can remove everything, which leaves the distribution unfitted and the stability
    margin unknown. That is the safe direction, since the margin can only promote a tier.

    This is the calibrated behaviour and the tiers in use were assigned under it. Widening the
    bound would shift every margin, so it is pinned by test rather than tidied.
    """
    kept = np.asarray(periods, dtype=float)
    for _ in range(CLIP_PASSES):
        if kept.size == 0:
            break
        kept = kept[kept <= 2 * np.std(kept)]
    return kept


def similar_size_periods(abs_mag: float, catalog_mags, catalog_periods) -> np.ndarray:
    """The measured periods of catalogued bodies within :data:`MAGNITUDE_WINDOW` of ``abs_mag``."""
    mags = np.asarray(catalog_mags, dtype=float)
    periods = np.asarray(catalog_periods, dtype=float)
    finite = np.isfinite(mags) & np.isfinite(periods)
    in_window = finite & (np.abs(mags - float(abs_mag)) < MAGNITUDE_WINDOW)
    return _clip_slow_tail(periods[in_window])


def fit_period_distribution(abs_mag: float | None, catalog_mags, catalog_periods):
    """Fit a Johnson S_B period distribution for a body of absolute magnitude ``abs_mag``.

    Returns an :class:`UnknownSpinDistribution`, rather than raising, whenever there is no
    magnitude, no sample, or a fit that fails to converge. An unfittable body must still come out
    of enrichment with the rest of its properties intact.

    A thin sample is fitted rather than refused. The fit is then poor, but the margin it produces
    is only ever compared against a threshold that *promotes* a tier, so a weak number can lift a
    target into consideration and never drop one out of it. Refusing to fit would demote every
    sparsely observed size class.
    """
    if abs_mag is None or not np.isfinite(float(abs_mag)):
        return UnknownSpinDistribution()
    sample = similar_size_periods(abs_mag, catalog_mags, catalog_periods)
    if sample.size == 0:
        return UnknownSpinDistribution()
    try:
        return scipy.stats.johnsonsb(*scipy.stats.johnsonsb.fit(sample))
    except Exception:
        return UnknownSpinDistribution()


def probability_slower_than(distribution, period_h: float | None) -> float:
    """``P(rotation period > period_h)`` under ``distribution``; NaN when either is unknown.

    Read as a stability margin against the critical period: near 1 the body almost certainly
    rotates more slowly than its disruption limit, near 0 it almost certainly does not.
    """
    if period_h is None or not np.isfinite(float(period_h)):
        return float("nan")
    try:
        return float(1.0 - distribution.cdf(float(period_h)))
    except Exception:
        return float("nan")
