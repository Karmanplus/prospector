"""Tests for the launch library and the analytic Earth-escape estimate (prospector.launch).

These pin how the escape delta-v is worked out rather than typed in: the shipped launch types load
and resolve, the estimate comes out at the circular speed when leaving with nothing to spare, it
grows more slowly than linearly with departure speed, and the fingerprint that guards a refinement
changes whenever anything it depends on does.
"""
import math

import pytest
from pydantic import ValidationError

from prospector.config import EngineMount, Mission, ResolvedConfig, Screening, Vehicle
from prospector.launch import (
    EARTH_EQUATORIAL_RADIUS_KM,
    MU_EARTH_KM3_S2,
    LaunchOrbit,
    escape_dv_estimate,
    escape_fingerprint,
    list_launch_orbits,
    load_launch_orbits,
    spiral_time_estimate_days,
)
from prospector.spacecraft.propulsion import Engine

LEO = LaunchOrbit(name="leo", perigee_alt_km=400, apogee_alt_km=400, inclination_deg=28.5)


def test_shipped_launch_library_loads():
    catalog = load_launch_orbits()
    assert {"LEO", "SSO", "GTO", "TLI", "ESCAPE"} <= set(catalog)
    assert catalog["TLI"].escape_provided and catalog["ESCAPE"].escape_provided
    assert not catalog["LEO"].escape_provided
    assert list_launch_orbits() == sorted(catalog)


def test_bare_escape_is_circular_speed():
    # The classic continuous-thrust limit: spiral-to-escape costs the circular speed.
    v_circ = math.sqrt(MU_EARTH_KM3_S2 / (EARTH_EQUATORIAL_RADIUS_KM + 400.0))
    assert math.isclose(escape_dv_estimate(LEO), v_circ, rel_tol=1e-12)
    assert math.isclose(LEO.v_circ_kms, v_circ, rel_tol=1e-12)


def test_lv_provided_escape_costs_nothing():
    tli = LaunchOrbit(name="tli", perigee_alt_km=185, apogee_alt_km=400000,
                      inclination_deg=28.5, escape_provided=True)
    assert escape_dv_estimate(tli) == 0.0
    assert escape_dv_estimate(tli, vinf_kms=2.0) == 0.0


def test_vinf_grows_cost_sublinearly():
    # Optimistic by construction: more v-infinity always costs more, but less than 1:1 (the Oberth
    # credit), the safe direction for a screen that must not drop targets.
    bare = escape_dv_estimate(LEO)
    one = escape_dv_estimate(LEO, vinf_kms=1.0)
    two = escape_dv_estimate(LEO, vinf_kms=2.0)
    assert bare < one < two
    assert two - bare < 2.0


def test_elliptical_orbit_uses_sma():
    gto = LaunchOrbit(name="gto", perigee_alt_km=250, apogee_alt_km=35786, inclination_deg=27)
    assert math.isclose(gto.sma_km, EARTH_EQUATORIAL_RADIUS_KM + 0.5 * (250 + 35786), rel_tol=1e-12)
    assert escape_dv_estimate(gto) < escape_dv_estimate(LEO)   # higher orbit, shallower well


def test_apogee_below_perigee_rejected():
    with pytest.raises(ValidationError):
        LaunchOrbit(name="bad", perigee_alt_km=500, apogee_alt_km=400, inclination_deg=0)


def test_time_estimate_scales_with_thrust():
    slow = spiral_time_estimate_days(LEO, wet_mass_kg=1000, thrust_N=0.1, isp_s=2000)
    fast = spiral_time_estimate_days(LEO, wet_mass_kg=1000, thrust_N=1.0, isp_s=2000)
    assert slow > fast > 0
    assert math.isclose(slow / fast, 10.0, rel_tol=1e-9)


def _rc(launch="LEO", fuel=500.0, solar_power_W=0.0):
    cat = {"E": Engine(name="E", isp_s=2000, thrust_mN=100, power_W=1000)}
    launches = {
        "LEO": LEO,
        "GTO": LaunchOrbit(name="gto", perigee_alt_km=250, apogee_alt_km=35786,
                           inclination_deg=27.0),
    }
    veh = Vehicle(name="v", dry_mass=500, fuel_mass=fuel, solar_power_W=solar_power_W,
                  engines=[EngineMount(type="E", count=1)])
    return ResolvedConfig.build(Mission(launch_orbit=launch), veh, Screening(), cat,
                                launches=launches)


def test_fingerprint_tracks_determining_inputs():
    base = escape_fingerprint(_rc())
    assert escape_fingerprint(_rc()) == base                          # deterministic
    assert escape_fingerprint(_rc(launch="GTO")) != base              # launch type matters
    assert escape_fingerprint(_rc(fuel=600)) != base                  # masses matter
    assert escape_fingerprint(_rc(solar_power_W=2000)) != base        # power model matters


def test_fingerprint_ignores_target_vinf():
    # The spiral is identical up to the escape crossing however far past it the run went, so any
    # converged run may refine the bare-escape budget term.
    rc = _rc()
    assert escape_fingerprint(rc, {"target_vinf_kms": 0.0}) == \
        escape_fingerprint(rc, {"target_vinf_kms": 2.0})
    assert escape_fingerprint(rc, {"duty_cycle": 0.9}) != \
        escape_fingerprint(rc, {"duty_cycle": 1.0})


# ---------------------------------------------------------------------------
# plane-change geometry (spiral steering vs asymptote vs cruise)
# ---------------------------------------------------------------------------

def test_max_departure_declination_caps_what_launch_can_aim():
    """The exact cone the cruise optimizer's departure v∞ must stay inside: a plane of
    inclination i contains no direction beyond EQUATORIAL declination i, no matter the
    node -- and everything within it is launch-targetable (node + in-plane phase). The
    cone is measured in the equatorial frame on purpose: the ecliptic-latitude band the
    UI quotes is this seen from the ecliptic, wider only at favorable longitudes."""
    from prospector.launch import LaunchOrbit, max_departure_declination_deg
    leo = LaunchOrbit(name="leo", perigee_alt_km=500, apogee_alt_km=500,
                      inclination_deg=28.5)
    assert max_departure_declination_deg(leo) == pytest.approx(28.5)
    # Steering widens the cone to the steered plane's reach.
    assert max_departure_declination_deg(leo, steer_inc_deg=60.0) == pytest.approx(60.0)
    # A retrograde near-polar plane reaches up to 180 - i.
    sso = LaunchOrbit(name="sso", perigee_alt_km=500, apogee_alt_km=500,
                      inclination_deg=97.4)
    assert max_departure_declination_deg(sso) == pytest.approx(82.6)
    # The LV aims the asymptote itself: unconstrained.
    tli = LaunchOrbit(name="tli", perigee_alt_km=200, apogee_alt_km=200,
                      inclination_deg=28.5, escape_provided=True)
    assert max_departure_declination_deg(tli) == 90.0
