"""Pin the solar-array performance physics.

The constants in source are ILLUSTRATIVE, a real array's measured optical response, cell fit, and
power-to-mass curve are entered in the config library, not here. So these tests pin the PHYSICS and
the MACHINERY rather than any particular vendor's numbers: the normalization invariant, the
direction and scaling of each effect, and the piecewise curve's interpolation, floor, extrapolation
and seam handling. A value that only reflects the illustrative defaults is asserted loosely and
labelled as such.
"""
import numpy as np
import pytest

from prospector.spacecraft.arrays import (
    EARTH_MEAN_RADIUS_KM,
    ArrayModel,
    default_array_model,
)


@pytest.fixture
def am():
    return default_array_model()


# ---------------------------------------------------------------------------- power


def test_power_fraction_is_exactly_one_at_reference(am):
    # The normalization invariant: vehicle solar_power_W is quoted at 1 AU BOL, so the fraction
    # there must be exactly 1 (deep space, no Earth term).
    assert am.power_fraction(1.0) == pytest.approx(1.0, abs=1e-12)


def test_power_fraction_falls_slower_than_irradiance(am):
    """The behaviour that matters, and it is nearly independent of the cell constants because the
    fraction is normalized at 1 AU: power falls with distance, but slower than 1/r^2, because the
    panel runs cooler and cold cells convert better."""
    assert am.power_fraction(1.5) > 1.0 / 1.5**2
    assert am.power_fraction(1.2) > am.power_fraction(1.5)
    # Roughly three-quarters at 1.2 AU and a half at 1.5 AU for any sane cell fit.
    assert am.power_fraction(1.2) == pytest.approx(0.74, abs=0.03)
    assert am.power_fraction(1.5) == pytest.approx(0.50, abs=0.03)


def test_panel_temperature_scaling(am):
    """The radiative balance: a sun-pointed panel near Earth's distance runs warm (tens of degC,
    set by the absorptivity/emissivity ratio) and cools as the inverse fourth root of irradiance.
    The scaling law is the assertion; the absolute value depends on the optical constants."""
    assert 20.0 < am.panel_temperature_C(1.0) < 100.0
    assert am.panel_temperature_C(1.5) < am.panel_temperature_C(1.0)
    # T scales as (1/r^2)^(1/4) = r^-0.5 in deep space.
    t1 = am.panel_temperature_C(1.0) + 273.15
    t15 = am.panel_temperature_C(1.5) + 273.15
    assert t15 == pytest.approx(t1 / np.sqrt(1.5), rel=1e-9)


def test_near_earth_term_warms_panel_and_dies_off(am):
    # Rear-face albedo + Earth IR warm the panel (lower efficiency -> lower fraction) and attenuate
    # as the view factor (R_E/r_geo)^2: pronounced in LEO, negligible by ~10 Earth radii.
    leo = am.power_fraction(1.0, r_geo_km=EARTH_MEAN_RADIUS_KM + 500.0)
    far = am.power_fraction(1.0, r_geo_km=65_000.0)
    assert leo < far <= 1.0
    assert 1.0 - far < 0.01
    assert 1.0 - leo > 0.02


def test_vectorized(am):
    r = np.array([1.0, 1.2, 1.5])
    frac = am.power_fraction(r)
    assert frac.shape == (3,)
    assert frac[0] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(frac) < 0)
    # Mixed vector sun distance + vector geocentric distance also broadcasts.
    both = am.power_fraction(np.full(3, 1.0), r_geo_km=np.array([7e3, 7e4, 7e5]))
    assert both.shape == (3,)
    assert np.all(np.diff(both) > 0)      # receding from Earth recovers power


def test_cell_efficiency_clamped(am):
    # A panel hot enough to null the linear fit produces nothing, never negative power.
    assert am.cell_efficiency(1000.0) == 0.0


# ---------------------------------------------------------------------------- mass


def test_mass_curve_interpolates_between_its_anchors(am):
    """Every anchor of the active curve is reproduced exactly, and points between interpolate
    linearly. Read off the curve itself so the test follows the config rather than a fixed table."""
    for seg in am.mass_curve:
        assert am.mass_kg(seg.w0) == pytest.approx(seg.kg0)
        assert am.mass_kg(seg.w1) == pytest.approx(seg.kg1)
        mid_w = 0.5 * (seg.w0 + seg.w1)
        assert am.mass_kg(mid_w) == pytest.approx(0.5 * (seg.kg0 + seg.kg1))
    # More power always costs more panel on a monotone curve.
    powers = np.linspace(am.mass_curve[0].lo_W, am.mass_curve[-1].hi_W, 50)
    assert np.all(np.diff([am.mass_kg(w) for w in powers]) > 0)


def test_mass_curve_supports_a_discontinuous_family_seam():
    """The machinery a real vendor curve needs: adjacent brackets may STEP, because a rollout wing
    is not a scaled rigid panel. The illustrative default is smooth, so the seam is
    exercised with an explicit curve rather than assumed of the default."""
    from prospector.spacecraft.arrays import ArrayMassSegment
    stepped = ArrayModel(mass_curve=[
        ArrayMassSegment(lo_W=0.0, hi_W=1000.0, w0=0.0, kg0=0.0, w1=1000.0, kg1=10.0),
        ArrayMassSegment(lo_W=1000.0, hi_W=2000.0, w0=1000.0, kg0=6.0, w1=2000.0, kg1=12.0),
    ])
    assert stepped.mass_kg(999.0) == pytest.approx(9.99, abs=0.01)
    assert stepped.mass_kg(1001.0) == pytest.approx(6.01, abs=0.01)   # steps DOWN across the seam


def test_mass_curve_edges(am):
    first, last = am.mass_curve[0], am.mass_curve[-1]
    # Below the smallest bracket: priced at that bracket's floor (a minimum buildable wing).
    assert am.mass_kg(1.0) == pytest.approx(am.mass_kg(first.lo_W))
    # Above the largest: extrapolated along the last segment's fit line.
    slope = (last.kg1 - last.kg0) / (last.w1 - last.w0)
    over = last.hi_W * 1.2
    assert am.mass_kg(over) == pytest.approx(last.kg1 + slope * (over - last.w1))


def test_mass_scale_scales_curve(am):
    heavy = ArrayModel(mass_scale=2.0)
    assert heavy.mass_kg(5000.0) == pytest.approx(2.0 * am.mass_kg(5000.0))


def test_specific_power_reporting(am):
    # Reporting sugar: watts of start-of-life power per kilogram of panel, off the active curve.
    assert am.specific_power_W_kg(5000.0) == pytest.approx(5000.0 / am.mass_kg(5000.0), rel=1e-9)
    assert am.specific_power_W_kg(0.0) == 0.0 or am.specific_power_W_kg(0.0) >= 0.0


# ---------------------------------------------------------------------------- rated vs operating


def test_operating_over_rated_is_the_efficiency_ratio_between_the_two_temperatures(am):
    # A datasheet wattage is quoted at rating_temp_C (28 degC); the panel facing the Sun at 1 AU
    # runs hotter, so it delivers less than its rating there. The ratio is the cell fit at the
    # operating temperature over the fit at the rating temperature, nothing else.
    t_op = am.operating_temp_C()
    assert t_op == pytest.approx(float(am.panel_temperature_C(1.0)))
    assert t_op > am.rating_temp_C                          # ordinary panels run hot at 1 AU
    k = am.operating_over_rated()
    assert k == pytest.approx(float(am.cell_efficiency(t_op)) / float(am.cell_efficiency(28.0)))
    assert 0.85 < k < 1.0                                   # a several-percent derate (illustrative fit)
    # The rated wattage behind an operating figure is the inverse conversion.
    assert am.rated_power_W(1000.0) == pytest.approx(1000.0 / k)
    assert am.rated_power_W(1000.0) > 1000.0


def test_rating_at_the_operating_temperature_makes_rated_and_operating_coincide(am):
    # Setting the rating temperature to the panel's own 1 AU temperature says "my wattages are
    # already flight figures": no derate at all.
    flat = am.model_copy(update={"rating_temp_C": am.operating_temp_C()})
    assert flat.operating_over_rated() == pytest.approx(1.0)
    assert flat.rated_power_W(1234.0) == pytest.approx(1234.0)
    # And a colder rating temperature derates more than a warmer one.
    assert (am.model_copy(update={"rating_temp_C": 20.0}).operating_over_rated()
            < am.model_copy(update={"rating_temp_C": 40.0}).operating_over_rated())


def test_rating_temperature_never_touches_the_flight_physics(am):
    # power_fraction is relative to the OPERATING output at 1 AU, which is what a vehicle's
    # solar_power_W means, so the rating temperature (a purchasing convention) cannot move it, the
    # panel temperature, or the cell efficiency at any distance.
    other = am.model_copy(update={"rating_temp_C": 60.0})
    r = np.array([0.7, 1.0, 1.2, 1.5])
    assert np.allclose(other.power_fraction(r), am.power_fraction(r))
    assert np.allclose(other.power_fraction(1.0, 6771.0), am.power_fraction(1.0, 6771.0))
    assert np.allclose(other.panel_temperature_C(r), am.panel_temperature_C(r))
    assert np.allclose(other.cell_efficiency(other.panel_temperature_C(r)),
                       am.cell_efficiency(am.panel_temperature_C(r)))


def test_reference_constants_derate_a_28C_rating_by_about_six_percent():
    # The team's reference thermal code: alpha_f 0.8856, eps_f 0.8795, alpha_r 0.3437,
    # eps_r 0.8673, Microlink fit eta = (-0.0599 T + 31.3)/100. Facing the Sun at 1 AU the panel
    # sits near 59 degC, where the cells make ~93.7% of their 28 degC datasheet figure. Pinned so
    # the derate the sizing charges cannot drift.
    ref = ArrayModel(alpha_front=0.8856, epsilon_front=0.8795, alpha_rear=0.3437,
                     epsilon_rear=0.8673, cell_eff_slope_pct_per_C=-0.0599,
                     cell_eff_intercept_pct=31.3, rating_temp_C=28.0)
    assert ref.operating_temp_C() == pytest.approx(59.0, abs=0.5)
    assert ref.operating_over_rated() == pytest.approx(0.937, abs=0.002)
    # Near Earth (400 km) the back face is warmed by albedo and Earth IR: hotter still.
    assert float(ref.panel_temperature_C(1.0, EARTH_MEAN_RADIUS_KM + 400.0)) == pytest.approx(79.0, abs=1.0)
