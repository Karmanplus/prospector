"""Tests for the size-matched rotation-period distribution.

The point of this model is to produce a stability margin for a body whose own period is missing or
unreliable, so the tests care most about what happens when the data is thin: an unfittable body
must come back "unknown", and unknown must stay unknown all the way through rather than turning
into a number a tier could act on.

The synthetic catalogue below is long-tailed on purpose. The outlier clip bounds periods by twice
the sample's standard deviation, so its effect depends on the sample's skew; a tidy normal sample
would exercise a regime the real catalogue never produces.
"""
import numpy as np

from prospector.enrichment.models import spin


def _catalog(n=600, seed=0):
    """A synthetic reference population: a few-hour mode with the real tail of slow rotators."""
    rng = np.random.default_rng(seed)
    magnitudes = rng.uniform(18.0, 22.0, n)
    periods = rng.lognormal(mean=np.log(6.0), sigma=1.1, size=n)
    return magnitudes, periods


def test_only_bodies_of_similar_size_are_sampled():
    magnitudes = np.array([15.0, 19.9, 20.0, 20.1, 25.0])
    periods = np.array([1.0, 5.0, 6.0, 7.0, 2.0])
    sample = spin.similar_size_periods(20.0, magnitudes, periods)
    # The two bodies outside the magnitude window are excluded regardless of their periods.
    assert set(sample) <= {5.0, 6.0, 7.0}
    assert 1.0 not in sample and 2.0 not in sample


def test_extreme_slow_rotators_are_clipped_out():
    magnitudes = np.full(60, 20.0)
    periods = np.concatenate([np.linspace(2.0, 20.0, 59), [10_000.0]])
    sample = spin.similar_size_periods(20.0, magnitudes, periods)
    assert 10_000.0 not in sample and len(sample) > 0


def test_a_fitted_distribution_gives_a_margin_between_zero_and_one():
    distribution = spin.fit_period_distribution(20.0, *_catalog())
    margin = spin.probability_slower_than(distribution, 3.0)
    assert 0.0 < margin < 1.0


def test_a_shorter_critical_period_is_easier_to_beat():
    distribution = spin.fit_period_distribution(20.0, *_catalog())
    # P(period > x) falls as x rises: a body whose limit sits deep in the population's range is
    # less likely to be safely slower than it.
    assert (spin.probability_slower_than(distribution, 1.0)
            > spin.probability_slower_than(distribution, 6.0))


def test_no_magnitude_means_no_distribution():
    distribution = spin.fit_period_distribution(None, *_catalog())
    assert np.isnan(spin.probability_slower_than(distribution, 5.0))


def test_no_matching_population_means_no_distribution():
    # Nothing within the magnitude window: unknown, not a fit of an empty sample.
    distribution = spin.fit_period_distribution(2.0, *_catalog())
    assert np.isnan(spin.probability_slower_than(distribution, 5.0))


def test_an_empty_catalogue_leaves_the_margin_unknown():
    # The reference catalogue is a network fetch and is allowed to come back empty; that must leave
    # the margin unknown rather than fail the run.
    distribution = spin.fit_period_distribution(20.0, [], [])
    assert np.isnan(spin.probability_slower_than(distribution, 5.0))


def test_an_unknown_critical_period_leaves_the_margin_unknown():
    distribution = spin.fit_period_distribution(20.0, *_catalog())
    assert np.isnan(spin.probability_slower_than(distribution, None))
    assert np.isnan(spin.probability_slower_than(distribution, float("inf")))


def test_a_sample_the_clip_empties_leaves_the_margin_unknown():
    """The clip bounds periods by twice the sample spread, so a tightly clustered sample can
    be removed entirely. That must read as "unknown", which only ever withholds a tier
    promotion -- rather than as a fitted distribution over nothing."""
    magnitudes, periods = np.full(3, 20.0), np.array([5.0, 6.0, 7.0])
    distribution = spin.fit_period_distribution(20.0, magnitudes, periods)
    assert np.isnan(spin.probability_slower_than(distribution, 4.0))
