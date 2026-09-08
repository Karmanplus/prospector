"""Tests for diameter-from-magnitude sizing and its albedo fallbacks.

The fallback chain is what these mostly check. Which albedo a body is sized with decides whether it
passes a minimum-diameter filter, and the chain must degrade in a defined order rather than
reaching for whatever happens to be non-null.
"""
import pytest

from prospector.enrichment.models import sizing


def test_diameter_matches_the_published_relation():
    # The standard form, D = 1329 / sqrt(albedo) * 10^(-0.2 H). Checking against the published
    # constant rather than against this implementation's own output catches a mistyped exponent,
    # which would otherwise size every body in the population wrongly and consistently.
    for abs_mag, albedo in ((15.6, 0.15), (22.0, 0.05), (18.0, 0.40)):
        expected = 1329.0 / (albedo ** 0.5) * 10 ** (-0.2 * abs_mag)
        assert sizing.diameter_km(abs_mag, albedo) == pytest.approx(expected, rel=1e-3)


def test_a_darker_body_is_larger_for_the_same_brightness():
    assert sizing.diameter_km(18.0, 0.05) > sizing.diameter_km(18.0, 0.25)


def test_five_magnitudes_fainter_is_ten_times_smaller():
    assert (sizing.diameter_km(20.0, 0.15)
            == pytest.approx(sizing.diameter_km(15.0, 0.15) / 10.0, rel=1e-9))


def test_measured_albedo_wins_over_the_taxonomy_median():
    assert sizing.albedo_for_sizing(0.42, "C") == 0.42


def test_taxonomy_median_is_used_when_no_albedo_was_measured():
    assert sizing.albedo_for_sizing(None, "Cb") == sizing.ALBEDO_BY_COMPLEX["C"]
    assert sizing.albedo_for_sizing(None, "v") == sizing.ALBEDO_BY_COMPLEX["V"]


def test_unknown_taxonomy_falls_through_to_the_default():
    assert sizing.albedo_for_sizing(None, None) == sizing.DEFAULT_ALBEDO
    assert sizing.albedo_for_sizing(None, "") == sizing.DEFAULT_ALBEDO
    assert sizing.albedo_for_sizing(None, "Zz") == sizing.DEFAULT_ALBEDO


def test_no_magnitude_means_no_diameter():
    # Not a zero and not a guess: a body with no H has no size, and the size filter keeps unknowns
    # rather than dropping them.
    assert sizing.diameter_m(None, 0.1) is None
    assert sizing.diameter_m(float("nan"), 0.1) is None


def test_diameter_in_metres_follows_the_fallback_chain():
    measured = sizing.diameter_m(18.0, 0.05)
    by_taxonomy = sizing.diameter_m(18.0, None, "C")
    by_default = sizing.diameter_m(18.0, None, None)
    # A C-type sized on the population default comes out smaller than on its own complex median,
    # the safe direction, since it reads as "possibly too small" rather than dropping a body that
    # is actually large.
    assert by_default < by_taxonomy
    assert measured == pytest.approx(sizing.diameter_km(18.0, 0.05) * 1000.0)
