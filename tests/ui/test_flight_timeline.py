"""Tests for the mission-profile engine series in ``ui.workspaces.flight``.

The cruise's available thrust and Isp are drawn per segment, as the allowance the solve flew
under, so the flown line never sits above the available one and a segment the array cannot power
reads as a gap rather than a number.
"""
import numpy as np
import pytest

from prospector.config import EngineMount, Mission, ResolvedConfig, Screening, Vehicle
from prospector.launch import LaunchOrbit
from prospector.spacecraft.propulsion import Engine
from ui.workspaces import flight

LEO = LaunchOrbit(name="leo", perigee_alt_km=400, apogee_alt_km=400, inclination_deg=28.5)


def _rc() -> ResolvedConfig:
    eng = Engine(name="E", isp_s=1900.0, thrust_mN=250.0, power_W=5000.0)
    veh = Vehicle(name="v", dry_mass=333, fuel_mass=320, solar_power_W=9296.0,
                  engines=[EngineMount(type="E", count=1)])
    return ResolvedConfig.build(Mission(launch_orbit="LEO"), veh, Screening(), {"E": eng},
                                launches={"LEO": LEO})


def test_cruise_series_is_the_per_segment_allowance():
    rc = _rc()
    days = np.linspace(0.0, 400.0, 41)               # 4 segments of 100 days
    sf = {"thrust_N": 0.249, "isp_s": 1798.0, "nseg": 4, "tof_days": 400.0,
          "seg_caps": np.array([0.64, 0.0, 0.8, 1.0]),
          "seg_isp_s": np.array([1712.0, 1712.0, 1783.0, 1881.0]),
          "fine_throttle": np.concatenate([np.full(10, 0.5), np.zeros(10), np.full(10, 0.8),
                                           np.full(11, 1.0)])}
    avail, isp, flown = flight._segment_engine_series(rc, sf, days)
    assert avail.shape == isp.shape == flown.shape == days.shape
    # Segment 1 (days 0-100): 0.64 x 249 mN available, held flat; Isp 1712.
    assert np.allclose(avail[:10], 0.64 * 249.0) and np.allclose(isp[:10], 1712.0)
    # Segment 2: the array cannot run the stack. Zero available, NO Isp, and nothing flown.
    assert np.all(avail[10:20] == 0.0) and np.all(np.isnan(isp[10:20]))
    assert np.all(flown[10:20] == 0.0)
    # Segments 3 and 4 step up, and the flown thrust never exceeds the allowance anywhere.
    assert np.allclose(avail[20:30], 0.8 * 249.0) and np.allclose(isp[20:30], 1783.0)
    assert np.allclose(avail[30:], 249.0) and np.allclose(isp[30:], 1881.0)
    assert np.all(flown <= avail + 1e-9)


def test_available_trace_is_the_array_ceiling_without_the_duty_factor():
    """Stored caps carry the duty-cycle limit as well as the array's (they are on the throttle's
    basis). The "available" line means what the array can supply, so the duty factor comes back
    out; a segment the array can fully power at 80 % duty reads as the full leg thrust."""
    rc = _rc()
    days = np.linspace(0.0, 200.0, 21)
    sf = {"thrust_N": 0.249, "isp_s": 1798.0, "nseg": 2, "tof_days": 200.0,
          "max_duty_cycle": 0.8, "seg_caps": np.array([0.8, 0.4]),
          "seg_isp_s": np.array([1881.0, 1712.0])}
    avail, _isp, _flown = flight._segment_engine_series(rc, sf, days)
    assert np.allclose(avail[:10], 249.0)                   # 0.8 / 0.8 -> the whole leg thrust
    assert np.allclose(avail[10:], 0.5 * 249.0)             # 0.4 / 0.8 -> half of it


def test_cruise_series_without_caps_is_the_leg_operating_point():
    """A leg solved with no array model (or an older run without per-segment terms) draws its one
    operating point flat, as before."""
    rc = _rc()
    days = np.linspace(0.0, 100.0, 5)
    sf = {"thrust_N": 0.159, "isp_s": 1712.0, "nseg": 12, "tof_days": 100.0, "seg_caps": None}
    avail, isp, flown = flight._segment_engine_series(rc, sf, days)
    assert np.allclose(avail, 159.0) and np.allclose(isp, 1712.0)
    assert flown is None                              # no throttle history recorded
    # No per-segment Isp recorded (an older run): the leg's one Isp is held across the segments.
    sf = {"thrust_N": 0.159, "isp_s": 1712.0, "nseg": 2, "tof_days": 100.0,
          "seg_caps": np.array([1.0, 0.5])}
    avail, isp, _ = flight._segment_engine_series(rc, sf, days)
    assert avail[0] == pytest.approx(159.0) and avail[-1] == pytest.approx(79.5)
    assert np.allclose(isp, 1712.0)


def test_array_thermal_series_is_the_panel_temperature_and_absolute_efficiency():
    # The mission profile's thermal panel draws the panel temperature and the fraction of sunlight
    # the cells convert, from the same ArrayModel the solvers fly: hotter (and less efficient) at
    # 400 km over Earth than in deep space at 1 AU, colder and more efficient going out.
    from prospector.spacecraft.arrays import EARTH_MEAN_RADIUS_KM, default_array_model
    am = default_array_model()
    sun = np.array([1.0, 1.0, 1.5])
    geo = np.array([EARTH_MEAN_RADIUS_KM + 400.0, 1e9, 1e9])
    temp_C, eff_pct = flight._array_thermal(am, sun, geo)
    assert temp_C.shape == eff_pct.shape == (3,)
    assert temp_C[0] > temp_C[1] > temp_C[2]
    assert eff_pct[0] < eff_pct[1] < eff_pct[2]
    assert np.allclose(temp_C, am.panel_temperature_C(sun, geo))
    assert np.allclose(eff_pct, np.asarray(am.cell_efficiency(temp_C)) * 100.0)
    # Deep space when no Earth distance is given.
    t_deep, _e = flight._array_thermal(am, 1.0)
    assert float(t_deep) == pytest.approx(am.operating_temp_C())
