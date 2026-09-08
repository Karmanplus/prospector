"""Tests for colour-to-reflectance reconstruction and the 0.7 um band depth.

Two things matter here. The chaining must be self-consistent: a body measured in B-V and V-R has to
come out with the same V-R ratio it went in with, whichever filter the walk anchors on. And
disconnected filter groups must stay separate: splicing them onto a common scale would invent a
slope that was never measured, and slope is exactly what the classifier reads.
"""
import numpy as np
import pytest

from prospector.enrichment.models.reflectance import (
    SOLAR_COLORS,
    Color,
    band_depth_07um,
    colors_to_reflectance,
    colors_to_spectrum,
)


def test_a_solar_coloured_body_is_flat():
    # A body whose colour matches the Sun's reflects equally in both filters by definition.
    solar_bv = SOLAR_COLORS["B"] - SOLAR_COLORS["V"]
    assert Color("B", "V", solar_bv).reflectance_ratio() == pytest.approx(1.0)


def test_a_redder_than_solar_body_reflects_more_at_the_longer_wavelength():
    solar_vr = SOLAR_COLORS["V"] - SOLAR_COLORS["R"]
    redder = Color("V", "R", solar_vr + 0.3)          # fainter in V relative to R than the Sun
    assert redder.reflectance_ratio() < 1.0           # i.e. R reflects more than V


def test_chained_colours_share_one_scale():
    colors = [Color("B", "V", SOLAR_COLORS["B"] - SOLAR_COLORS["V"] + 0.2),
              Color("V", "R", SOLAR_COLORS["V"] - SOLAR_COLORS["R"] + 0.1)]
    groups = colors_to_reflectance(colors, anchor_filter="V")
    assert len(groups) == 1
    group = groups[0]
    assert group["V"][0] == pytest.approx(1.0), "the anchor is normalised to unity"
    # The reconstructed ratios must reproduce the measured colours exactly.
    assert group["B"][0] / group["V"][0] == pytest.approx(colors[0].reflectance_ratio())
    assert group["V"][0] / group["R"][0] == pytest.approx(colors[1].reflectance_ratio())


def test_disconnected_filter_groups_stay_separate():
    # B-V and g-r share no filter, so there is no measurement relating them. Splicing them would
    # invent a slope across the join.
    colors = [Color("B", "V", 0.9), Color("g", "r", 0.5)]
    groups = colors_to_reflectance(colors, anchor_filter="V")
    assert len(groups) == 2
    assert {frozenset(group) for group in groups} == {frozenset({"B", "V"}),
                                                      frozenset({"g", "r"})}


def test_an_unknown_filter_is_rejected_rather_than_guessed():
    with pytest.raises(ValueError, match="solar colour"):
        colors_to_reflectance([Color("B", "Q", 0.5)])


def test_no_colours_is_an_error_not_an_empty_spectrum():
    with pytest.raises(ValueError):
        colors_to_reflectance([])


def test_a_spectrum_comes_out_in_wavelength_order():
    colors = [Color("B", "V", 0.9), Color("V", "R", 0.4)]
    wave, refl = colors_to_spectrum(colors)
    assert wave == sorted(wave) and len(wave) == len(refl) == 3


def test_the_spectrum_comes_from_the_anchored_group():
    # With two disconnected groups only one can be used, and it must be the one containing the
    # anchor, the visible band the classifier was trained on, not whichever group the walk happened
    # to reach first.
    colors = [Color("g", "r", 0.5), Color("B", "V", 0.9), Color("V", "R", 0.4)]
    wave, _refl = colors_to_spectrum(colors, anchor_filter="V")
    assert len(wave) == 3, "the B-V-R group, not the two-filter g-r one"


def test_a_featureless_spectrum_has_no_band():
    wave = np.linspace(0.4, 1.0, 50)
    assert band_depth_07um(wave, np.ones_like(wave)) == pytest.approx(0.0, abs=1e-12)


def test_a_straight_slope_has_no_band():
    # The band is measured against a continuum drawn between 0.6 and 0.8 um, so a linear spectrum,
    # however steep, must read as no absorption at all.
    wave = np.linspace(0.4, 1.0, 50)
    assert band_depth_07um(wave, 1.0 + 0.8 * wave) == pytest.approx(0.0, abs=1e-12)


def test_an_absorption_at_07um_reads_as_a_positive_depth():
    wave = np.linspace(0.4, 1.0, 121)
    absorbed = np.ones_like(wave) - 0.2 * np.exp(-((wave - 0.7) / 0.04) ** 2)
    assert band_depth_07um(wave, absorbed) > 0.15


def test_the_depth_is_independent_of_how_densely_the_spectrum_was_sampled():
    def absorbed(wave):
        return np.ones_like(wave) - 0.2 * np.exp(-((wave - 0.7) / 0.06) ** 2)

    coarse = np.linspace(0.4, 1.0, 25)
    fine = np.linspace(0.4, 1.0, 400)
    assert band_depth_07um(coarse, absorbed(coarse)) == pytest.approx(
        band_depth_07um(fine, absorbed(fine)), abs=5e-3)
