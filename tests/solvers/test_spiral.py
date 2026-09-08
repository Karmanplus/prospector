"""Tests for the propagated escape spiral (prospector.solvers.spiral).

The real use case is a months-long spiral that runs as a background job; these tests use a
deliberately torquey vehicle (high thrust-to-mass) so escape arrives in days of simulated
time and seconds of wall clock, then check the physics invariants rather than pinned
numbers: it escapes, the cost sits in the analytic neighborhood, the dv-vs-v-infinity
curve is monotone and interpolable, fuel exhaustion and the LV-escape guard are honest.
"""
import math

import numpy as np
import pytest

from prospector.config import EngineMount, Mission, ResolvedConfig, Screening, Vehicle
from prospector.launch import LaunchOrbit
from prospector.solvers import spiral
from prospector.spacecraft.propulsion import Engine

# 400 km equatorial start; circular speed ~7.67 km/s sets the escape-cost scale.
LEO_KW = dict(perigee_alt_km=400.0, apogee_alt_km=400.0, inclination_deg=0.0)
V_CIRC = math.sqrt(spiral.MU / (spiral.EARTH_EQUATORIAL_RADIUS_M + 400e3)) / 1e3

# High thrust-to-mass so the test integrates in seconds: 10 N on 500 kg.
FAST = dict(mass_kg=500.0, dry_mass_kg=100.0, thrust_N=10.0, isp_s=1500.0)


@pytest.fixture(scope="module")
def escaped():
    return spiral.solve(**LEO_KW, **FAST, target_vinf_kms=1.0)


def test_escapes_with_cost_near_the_analytic_scale(escaped):
    sol = escaped
    assert sol.status == "escaped"
    # High accel undercuts the slow-spiral limit (v_circ) but stays above the impulsive floor; the
    # band brackets both regimes without pinning solver noise.
    assert 0.3 * V_CIRC < sol.dv_at_escape_kms < 1.1 * V_CIRC
    assert sol.final_mass_kg < sol.initial_mass_kg
    assert sol.propellant_kg > 0
    assert sol.tof_days > 0


def test_reaches_the_target_vinf(escaped):
    sol = escaped
    assert sol.vinf_kms == pytest.approx(1.0, abs=0.05)
    assert sol.dv_kms > sol.dv_at_escape_kms          # excess speed costs extra
    assert sol.tof_days >= sol.tof_at_escape_days


def test_tradeoff_curve_is_monotone_and_interpolable(escaped):
    sol = escaped
    assert sol.curve_vinf_kms[0] == 0.0
    assert sol.curve_vinf_kms[-1] == pytest.approx(sol.vinf_kms, abs=1e-6)
    assert np.all(np.diff(sol.curve_vinf_kms) >= 0)
    assert np.all(np.diff(sol.curve_dv_kms) >= -1e-9)
    mid = sol.dv_at_vinf(0.5)
    assert sol.dv_at_escape_kms <= mid <= sol.dv_kms
    assert sol.dv_at_vinf(5.0) is None                 # beyond what this run reached


def test_path_arrays_are_consistent(escaped):
    sol = escaped
    n = sol.times_days.size
    assert sol.positions_km.shape == (n, 3)
    assert sol.mass_kg.shape == (n,) and sol.energy_km2_s2.shape == (n,)
    assert sol.energy_km2_s2[0] < 0 < sol.energy_km2_s2[-1]   # bound -> hyperbolic
    assert np.linalg.norm(sol.positions_km[0]) == pytest.approx(
        spiral.EARTH_EQUATORIAL_RADIUS_M / 1e3 + 400.0, rel=1e-6)
    assert sol.eclipse_days >= 0 and sol.belt_days >= 0


def test_belt_power_profile_declines_then_holds(escaped):
    # The reconstructed array-power curve: starts at full power, never recovers (belt damage is
    # permanent), and ends near the run's own end-of-escape power fraction.
    sol = escaped
    in_belt, belt_cum, power = spiral.belt_power_profile(sol.positions_km, sol.times_days)
    n = sol.times_days.size
    assert in_belt.shape == (n,) and power.shape == (n,)
    assert power[0] == pytest.approx(1.0)
    assert np.all(np.diff(power) <= 1e-12)                       # monotone non-increasing
    assert np.all(np.diff(belt_cum) >= -1e-12)                   # belt residence only grows
    assert power[-1] == pytest.approx(sol.power_fraction_end, rel=0.15)
    # A path too short to read gives empty arrays rather than raising.
    assert spiral.belt_power_profile(np.zeros((1, 3)), np.zeros(1))[2].size == 0


def test_reports_a_revolution_count(escaped):
    # Orbits flown to escape, the build-stage ignition proxy. Positive and finite, and below the
    # constant-period upper bound (the orbit raises, so its period only grows).
    sol = escaped
    assert sol.revolutions > 0 and math.isfinite(sol.revolutions)
    p0_days = 2 * math.pi * math.sqrt(
        (spiral.EARTH_EQUATORIAL_RADIUS_M + 400e3) ** 3 / spiral.MU) / 86400.0
    assert sol.revolutions <= sol.tof_at_escape_days / p0_days + 1.0


def test_power_limited_flight_is_throttled_not_failed():
    # An array that degrades below the full-thrust demand is flown at reduced thrust (Isp
    # constant), not rejected: power_limited is an informational verdict and
    # available_thrust_fraction is the linear thrust ceiling the cruise inherits. Power-rich (no
    # array model): never power-limited, full thrust available.
    rich = spiral.solve(**LEO_KW, **FAST, target_vinf_kms=0.0)
    assert rich.power_limited is False
    assert rich.power_required_W == 0.0
    assert rich.available_thrust_fraction == pytest.approx(1.0)

    # A generously sized array stays above the demand through belt degradation, not limited, and
    # the reported available power is the BOL output scaled by the end-of-life fraction.
    ample = spiral.solve(**LEO_KW, **FAST, target_vinf_kms=0.0,
                         bol_power_W=8000.0, power_req_W=4000.0)
    assert ample.status == "escaped"
    assert not ample.power_limited
    assert ample.power_available_end_W >= ample.power_required_W
    assert ample.power_available_end_W == pytest.approx(
        8000.0 * ample.power_fraction_end, rel=1e-9)
    assert ample.available_thrust_fraction == pytest.approx(1.0)

    # An undersized array falls below the full-thrust demand -> power-limited, but it still flies
    # (a real result), throttled to the fraction of full thrust the aged array can power.
    starved = spiral.solve(**LEO_KW, **FAST, target_vinf_kms=0.0,
                          bol_power_W=3900.0, power_req_W=4000.0)
    assert starved.power_limited is True
    assert starved.status in ("escaped", "timed_out", "out_of_fuel")
    assert starved.power_available_end_W < starved.power_required_W
    assert 0.0 < starved.available_thrust_fraction < 1.0
    assert starved.available_thrust_fraction == pytest.approx(
        min(1.0, starved.power_available_end_W / starved.power_required_W), rel=1e-9)


def test_radiation_model_selection_changes_the_loss():
    # Selecting a gentler proton environment leaves MORE array power at the end of the same climb.
    from prospector.spacecraft.radiation import load_radiation_model
    worst = spiral.solve(**LEO_KW, **FAST, target_vinf_kms=0.0,
                         radiation_model=load_radiation_model("ap8min-worstcase"))
    gentle = spiral.solve(**LEO_KW, **FAST, target_vinf_kms=0.0,
                          radiation_model=load_radiation_model("ap8max-nominal"))
    assert gentle.power_fraction_end > worst.power_fraction_end
    # The solution records the scenario it was flown with (self-describing for the report).
    assert worst.radiation_model and worst.coverglass_um == pytest.approx(212.5)
    assert worst.coverglass_density_g_cm3 == pytest.approx(1.64)


def test_power_curve_lowers_isp_when_power_limited():
    # With an engine throttle curve, a power-limited spiral runs at reduced thrust and reduced Isp;
    # without one it throttles thrust at constant (rated) Isp. The curve below is built so its
    # thrust matches that linear throttle exactly at every power (60 mN x P/1000), which isolates
    # the one difference that matters: the curve's lower Isp (1350 s at 500 W, against the rated
    # 1500 s) burns MORE propellant for the same climb.
    from prospector.spacecraft.propulsion import Engine, PowerPoint, assembly_power_grid
    eng = Engine(name="C", isp_s=1500.0, thrust_mN=60.0, power_W=1000.0,
                 power_curve=[PowerPoint(power_W=400.0, thrust_mN=24.0, isp_s=1300.0),
                              PowerPoint(power_W=500.0, thrust_mN=30.0, isp_s=1350.0),
                              PowerPoint(power_W=1000.0, thrust_mN=60.0, isp_s=1500.0)])
    grid = assembly_power_grid([("C", 1)], {"C": eng})
    # Start above the inner proton belt so the array stays near its 500 W (power-limited but above
    # the engine's ~400 W shed floor) the whole climb, both cases escape, isolating the Isp effect.
    # (Deep in the belt the array would degrade below the floor, shed to zero thrust, and coast;
    # that is a genuine "does not escape" verdict, tested elsewhere, not the comparison wanted
    # here.)
    kw = dict(perigee_alt_km=20000.0, apogee_alt_km=20000.0, inclination_deg=10.0,
              mass_kg=120.0, dry_mass_kg=60.0, thrust_N=60.0e-3, isp_s=1500.0,
              power_req_W=1000.0, bol_power_W=500.0, area_m2=0.0, target_vinf_kms=0.0, max_years=4.0)
    with_curve = spiral.solve(power_grid=grid, **kw)
    legacy = spiral.solve(**kw)                        # no grid -> constant Isp, linear throttle
    assert with_curve.status == "escaped" and legacy.status == "escaped"
    assert with_curve.propellant_kg > legacy.propellant_kg * 1.02


def test_power_starved_spiral_bails_early_not_grinds():
    # An array that can't power even one thruster above its floor makes ZERO thrust: the spiral
    # coasts and never escapes. The no-progress bailout must catch this and return timed_out after
    # a short stall, not integrate every step out to max_years (the pathological run time that a
    # curve-limited, deeply-degraded design would otherwise hit).
    from prospector.spacecraft.propulsion import Engine, PowerPoint, assembly_power_grid
    eng = Engine(name="C", isp_s=1500.0, thrust_mN=60.0, power_W=1000.0,
                 power_curve=[PowerPoint(power_W=400.0, thrust_mN=24.0, isp_s=1300.0),
                              PowerPoint(power_W=1000.0, thrust_mN=60.0, isp_s=1500.0)])
    grid = assembly_power_grid([("C", 1)], {"C": eng})
    sol = spiral.solve(perigee_alt_km=8000.0, apogee_alt_km=8000.0, inclination_deg=10.0,
                       mass_kg=120.0, dry_mass_kg=60.0, thrust_N=60.0e-3, isp_s=1500.0,
                       power_req_W=1000.0, bol_power_W=300.0, area_m2=0.0,   # 300 W < 400 W floor
                       target_vinf_kms=0.0, max_years=8.0, power_grid=grid)
    assert sol.status == "timed_out"
    assert sol.tof_days < 500.0            # bailed on the stall, did not grind ~8 years of coast


def test_array_to_thruster_eff_scales_delivered_power():
    # The array powers the thrusters through the conversion chain (transmission x HV->LV x PPU). A
    # lower chain efficiency delivers proportionally less power to the thruster. With a power-RICH
    # array (never throttled), both runs fly the identical trajectory, so the end-of-life delivered
    # power scales exactly with the efficiency; default 1.0 = no chain.
    kw = dict(perigee_alt_km=20000.0, apogee_alt_km=20000.0, inclination_deg=10.0,
              mass_kg=120.0, dry_mass_kg=60.0, thrust_N=60.0e-3, isp_s=1500.0,
              power_req_W=1000.0, bol_power_W=6000.0, area_m2=0.0,   # >> demand -> never power-limited
              target_vinf_kms=0.0, max_years=4.0)
    full = spiral.solve(array_to_thruster_eff=1.0, **kw)
    lossy = spiral.solve(array_to_thruster_eff=0.7533, **kw)
    assert full.status == "escaped" and lossy.status == "escaped"
    assert full.power_available_end_W > 0.0
    assert lossy.power_available_end_W == pytest.approx(0.7533 * full.power_available_end_W, rel=1e-6)
    assert full.propellant_kg == pytest.approx(lossy.propellant_kg, rel=1e-6)   # same flight (power-rich)


def test_array_model_default_is_legacy_full_output(escaped):
    # No array model flown -> the sun-distance/thermal factor is identically 1 (the legacy
    # full-output assumption), with one sample per path point.
    sol = escaped
    assert sol.power_fraction_sun.shape == sol.times_days.shape
    assert np.all(sol.power_fraction_sun == 1.0)


def test_array_model_heats_the_panel_near_earth():
    # With the sun-distance/thermal physics flown, the near-Earth albedo/IR term warms the panel
    # early in the climb (a few % output loss in LEO) and dies off as the spiral recedes, so the
    # per-sample factor starts depressed and recovers toward ~1. The geocentric sun distance stays
    # ~1 AU, so the factor never strays far from unity.
    from prospector.spacecraft.arrays import default_array_model
    sol = spiral.solve(**LEO_KW, **FAST, target_vinf_kms=0.0,
                       array_model=default_array_model())
    assert sol.status == "escaped"
    frac = sol.power_fraction_sun
    assert frac.shape == sol.times_days.shape
    assert frac[0] < 0.99                      # LEO: warmed by albedo + Earth IR
    assert frac[-1] > frac[0]                  # recovers as the Earth term dies off
    assert np.all((0.9 < frac) & (frac < 1.05))
    # The end-of-flight available power folds the factor in.
    with_power = spiral.solve(**LEO_KW, **FAST, target_vinf_kms=0.0,
                              bol_power_W=8000.0, power_req_W=4000.0,
                              array_model=default_array_model())
    assert with_power.power_available_end_W == pytest.approx(
        8000.0 * with_power.power_fraction_end * with_power.power_fraction_sun[-1], rel=1e-9)


def test_out_of_fuel_reported_honestly():
    sol = spiral.solve(**LEO_KW, mass_kg=105.0, dry_mass_kg=100.0, thrust_N=10.0,
                       isp_s=1500.0, target_vinf_kms=0.0)
    assert sol.status == "out_of_fuel"
    assert sol.dv_at_escape_kms is None                # never escaped -> no refinement term
    assert sol.curve_vinf_kms.size == 0
    assert sol.final_mass_kg == pytest.approx(100.0, abs=0.5)


def test_progress_reports_a_climb():
    fractions = []
    spiral.solve(**LEO_KW, **FAST, target_vinf_kms=0.0, chunk_days=0.5,
                 progress=lambda frac, msg: fractions.append(frac))
    assert fractions and all(0.0 <= f <= 1.0 for f in fractions)
    assert fractions == sorted(fractions)              # the energy climb is monotone


def test_solve_for_config_pulls_the_vehicle_and_guards_lv_escape():
    cat = {"E": Engine(name="E", isp_s=1500, thrust_mN=10000, power_W=1000)}
    launches = {
        "LEO": LaunchOrbit(name="leo", perigee_alt_km=400, apogee_alt_km=400,
                           inclination_deg=0.0),
        "TLI": LaunchOrbit(name="tli", perigee_alt_km=185, apogee_alt_km=400000,
                           inclination_deg=28.5, escape_provided=True),
    }
    veh = Vehicle(name="v", dry_mass=100, fuel_mass=400,
                  engines=[EngineMount(type="E", count=1)])
    rc = ResolvedConfig.build(Mission(launch_orbit="LEO"), veh, Screening(), cat,
                              launches=launches)
    sol = spiral.solve_for_config(rc, target_vinf_kms=0.0)
    assert sol.status == "escaped"

    rc_tli = ResolvedConfig.build(Mission(launch_orbit="TLI"), veh, Screening(), cat,
                                  launches=launches)
    with pytest.raises(ValueError, match="provides escape"):
        spiral.solve_for_config(rc_tli)


def test_injection_orientation_parameters():
    """raan/argp orient the injection rigidly: the node (launch time of day) and the
    in-plane phase (ascent targeting) are launch's free choices; the inclination is the
    launch type's and stays put. The defaults give the standard orientation."""
    inc = dict(LEO_KW, inclination_deg=28.5)
    base = spiral.solve(**inc, **FAST, target_vinf_kms=0.0, max_years=0.001)
    p0 = base.positions_km[0]
    # The standard orientation: the low point on the +x axis.
    assert p0[0] == pytest.approx(np.linalg.norm(p0), rel=1e-9)
    # raan=90 swings the node a quarter turn: the initial position lands on +y.
    turned = spiral.solve(**inc, **FAST, target_vinf_kms=0.0, max_years=0.001,
                          raan_deg=90.0)
    q0 = turned.positions_km[0]
    assert q0[1] == pytest.approx(np.linalg.norm(q0), rel=1e-9)
    # argp=180 starts at the opposite in-plane phase: initial position on -x.
    flipped = spiral.solve(**inc, **FAST, target_vinf_kms=0.0, max_years=0.001,
                           argp_deg=180.0)
    f0 = flipped.positions_km[0]
    assert f0[0] == pytest.approx(-np.linalg.norm(f0), rel=1e-9)


def test_solve_targeted_plane_contains_the_requested_exit():
    """The exit-aimed re-fly: every pass is a genuine propagation with a corrected
    injection node, Newton-stepped off the previous pass's flown geometry, never a
    post-hoc rotation of the picture. Convergence is plane CONTAINMENT of the requested
    direction (where ALONG the plane the exit lands is launch-date timing, anchored to
    the Sun by eclipse gating in this model -- not chased). The cost must
    match the untargeted run at the percent level (orientation only shifts perturbation
    phasing)."""
    inc = dict(LEO_KW, inclination_deg=28.5)
    plain = spiral.solve(**inc, **FAST, target_vinf_kms=1.0)
    assert plain.exit_lat_deg is not None          # escaped runs record their asymptote

    # A direction the 28.5 deg plane can reach: ecliptic (5, 140) sits at declination ~19.6 deg
    # (note (15, 140) would be ~29 deg, just past the plane; the frame bites).
    target_lat, target_lon = 5.0, 140.0
    assert (spiral.plane_alignment_error_deg(plain.positions_km, target_lat, target_lon)
            > 1.5)                                  # the standard orientation misses it
    aimed = spiral.solve_targeted(exit_lat_deg=target_lat, exit_lon_deg=target_lon,
                                  tol_deg=1.5, **inc, **FAST, target_vinf_kms=1.0)
    assert aimed.status == "escaped"
    err = spiral.plane_alignment_error_deg(aimed.positions_km, target_lat, target_lon)
    assert err <= 1.5
    # Genuinely flown, comparably priced: same physics, different phasing only.
    assert aimed.dv_kms == pytest.approx(plain.dv_kms, rel=2e-2)
    # The orientation that flew it is recorded (reproducible without re-aiming).
    redo = spiral.solve(**inc, **FAST, target_vinf_kms=1.0,
                        raan_deg=aimed.raan_deg, argp_deg=aimed.argp_deg)
    assert (spiral.plane_alignment_error_deg(redo.positions_km, target_lat, target_lon)
            <= 1.5)


def test_solve_targeted_reports_unreachable_directions_honestly():
    """A direction beyond the plane's declination reach can't be contained by node
    targeting; the best-effort run comes back with its true flown geometry, never a
    faked alignment. (The cruise solve's departure cone prevents asking for this.)"""
    inc = dict(LEO_KW, inclination_deg=10.0)
    sol = spiral.solve_targeted(exit_lat_deg=60.0, exit_lon_deg=0.0, tol_deg=1.5,
                                max_aim_iter=2, **inc, **FAST, target_vinf_kms=1.0)
    assert sol.status == "escaped"
    assert spiral.plane_alignment_error_deg(sol.positions_km, 60.0, 0.0) > 1.5
