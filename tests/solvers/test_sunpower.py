"""Tests for the cruise operating point following the Sun (prospector.solvers.sunpower), and for
the Sims-Flanagan leg terms built from it.

The array's output moves with distance from the Sun; the thrusters follow it along their throttle
curve both ways, up to their rated draw. The leg optimizer carries one thrust and one Isp, so the
model supplies a leg thrust (the closest approach), per-segment ceilings (the mean over each
segment's path) and a leg Isp (the throttle-weighted mean of the segments'). These pin that
arithmetic on a discrete-mode stack, where every one of those choices is visible as a step.
"""
from datetime import date

import numpy as np
import pytest

from prospector.config import EngineMount, Mission, ResolvedConfig, Screening, Vehicle
from prospector.launch import LaunchOrbit
from prospector.solvers import sunpower
from prospector.spacecraft.propulsion import Engine, PowerPoint

# Modes at 3, 4 and 5 kW, in the shape of a 5 kW Hall thruster: thrust and Isp both rise with power.
MODES = [PowerPoint(power_W=3000, thrust_mN=150.0, isp_s=1700.0),
         PowerPoint(power_W=4000, thrust_mN=200.0, isp_s=1800.0),
         PowerPoint(power_W=5000, thrust_mN=250.0, isp_s=1900.0)]
STEPPED = Engine(name="E", isp_s=1900.0, thrust_mN=250.0, power_W=5000.0, power_curve=MODES,
                 throttle_mode="discrete")
SMOOTH = STEPPED.model_copy(update={"throttle_mode": "continuous"})
LEO = LaunchOrbit(name="leo", perigee_alt_km=400, apogee_alt_km=400, inclination_deg=28.5)


class _InverseSquare:
    """An array whose output scales purely as 1/r^2: the sun-distance factor with no thermal term,
    so the arithmetic below is checkable by hand."""

    def power_fraction(self, r_au, *_args):
        r = np.asarray(r_au, float)
        return 1.0 / (r * r)


def _rc(engine: Engine, n: int = 1) -> ResolvedConfig:
    veh = Vehicle(name="v", dry_mass=333, fuel_mass=320, solar_power_W=9296.0,
                  engines=[EngineMount(type="E", count=n)])
    return ResolvedConfig.build(Mission(launch_orbit="LEO"), veh, Screening(), {"E": engine},
                                launches={"LEO": LEO})


def test_the_operating_point_follows_the_sun_both_ways():
    # 3.8 kW at the thruster at 1 AU (a post-escape figure from a study run).
    model = sunpower.SunPowerModel(_rc(STEPPED), 3800.0, _InverseSquare())
    assert model.at_one_au() == (pytest.approx(0.150), pytest.approx(1700.0))     # the 3 kW mode
    t, isp = model.thrust_isp(0.9)                       # 4.69 kW: the 4.5.. no, the 4 kW mode
    assert (t, isp) == (pytest.approx(0.200), pytest.approx(1800.0))
    t, isp = model.thrust_isp(0.85)                      # 5.26 kW: rated, capped there
    assert (t, isp) == (pytest.approx(0.250), pytest.approx(1900.0))
    t, isp = model.thrust_isp(0.5)                       # 15 kW buys nothing more than rated
    assert (t, isp) == (pytest.approx(0.250), pytest.approx(1900.0))
    t, _ = model.thrust_isp(1.2)                         # 2.64 kW: below the floor, off
    assert t == 0.0
    # The same array on the continuous engine slides instead of stepping.
    smooth = sunpower.SunPowerModel(_rc(SMOOTH), 3800.0, _InverseSquare())
    t, isp = smooth.thrust_isp(0.9)
    assert 0.200 < t < 0.250 and 1800.0 < isp < 1900.0


def test_leg_thrust_is_the_closest_approach_within_bounds():
    model = sunpower.SunPowerModel(_rc(STEPPED), 3800.0, _InverseSquare())
    assert model.leg_thrust_N(0.9) == pytest.approx(0.200)
    assert model.leg_thrust_N(0.75) == pytest.approx(0.250)          # a perihelion inside 0.85 AU: rated
    assert model.leg_thrust_N(None) == pytest.approx(model.thrust_N(sunpower.DEFAULT_R_MIN_AU))
    assert model.leg_thrust_N(1.5) == pytest.approx(model.thrust_N(sunpower.DEFAULT_R_MIN_AU)), \
        "a target further out than Earth still lets the leg pass near 1 AU"
    assert model.leg_thrust_N(0.01) == pytest.approx(model.thrust_N(sunpower.R_MIN_FLOOR_AU))


def test_segment_ceilings_are_means_over_the_segment_path():
    """A segment whose path crosses the floor part-way gets the impulse it really has: the mean of
    the available thrust along it, not the value at one instant (which read as zero available
    with thrust still being flown when the midpoint sat just inside the floor)."""
    model = sunpower.SunPowerModel(_rc(STEPPED), 3800.0, _InverseSquare())
    au = 1.495978707e11
    # One segment along +x from 1.0 AU (3 kW mode, 150 mN) to 1.3 AU (off). Three nodes: start,
    # midpoint 1.15 AU (2.87 kW: off), end.
    nodes = np.array([[1.0, 0, 0], [1.15, 0, 0], [1.3, 0, 0]]) * au
    caps, isps = model.segment_terms(nodes, leg_thrust_N=0.250)
    assert caps.shape == (1,) and isps.shape == (1,)
    # Only the part inside ~1.125 AU fires, at 150 mN: the mean sits between 0 and 150/250.
    assert 0.0 < caps[0] < 0.6
    assert isps[0] == pytest.approx(1700.0)              # every firing sample is the 3 kW mode
    # A segment drifting outward from 0.90 to 0.94 AU stays in the 4 kW mode throughout
    # (4.7 kW down to 4.3 kW at the thruster).
    nodes = np.array([[0.90, 0, 0], [0.92, 0, 0], [0.94, 0, 0]]) * au
    caps, isps = model.segment_terms(nodes, leg_thrust_N=0.250)
    assert caps[0] == pytest.approx(0.8) and isps[0] == pytest.approx(1800.0)
    # A segment entirely beyond the floor has no operating point: zero ceiling, the floor's Isp.
    nodes = np.array([[1.3, 0, 0], [1.35, 0, 0], [1.4, 0, 0]]) * au
    caps, isps = model.segment_terms(nodes, leg_thrust_N=0.250)
    assert caps[0] == 0.0 and isps[0] == pytest.approx(1700.0)


def test_the_leg_isp_is_the_throttle_weighted_harmonic_mean():
    """Equal-duration segments spend propellant as |u_i| / Isp_i, so the one Isp that reproduces
    the total is the throttle-weighted harmonic mean; segments that coast do not vote."""
    isps = np.array([1700.0, 1900.0, 1800.0])
    u = np.array([1.0, 1.0, 0.0])
    assert sunpower.effective_isp(u, isps, 1000.0) == pytest.approx(2 / (1 / 1700 + 1 / 1900))
    assert sunpower.effective_isp(np.zeros(3), isps, 1234.0) == 1234.0
    # Propellant check: 1 N for two equal segments of 1 s at the two Isps vs the mean.
    g0 = 9.80665
    direct = 1.0 / (g0 * 1700) + 1.0 / (g0 * 1900)
    mean = 2.0 / (g0 * sunpower.effective_isp(np.array([1.0, 1.0]), isps[:2], 0.0))
    assert mean == pytest.approx(direct)


def test_no_power_model_when_nothing_limits():
    rc = _rc(STEPPED)
    # An escape that left the thrusters no power at all is a real, limiting model: zero power, so
    # zero ceilings, rather than "nothing to limit" and a cruise at rated thrust.
    starved = sunpower.sun_power_model(rc, 0.0, _InverseSquare())
    assert starved is not None and starved.power_W(1.0) == 0.0 and starved.thrust_N(0.5) == 0.0
    # No array modelled on the vehicle at all: power-rich, nothing to limit.
    powerless = rc.model_copy(update={"vehicle": rc.vehicle.model_copy(update={"solar_power_W": 0.0})})
    assert sunpower.sun_power_model(powerless, None) is None
    assert sunpower.sun_power_model(rc, 3800.0, _InverseSquare()) is not None


@pytest.mark.slow
def test_the_isp_pass_resolves_at_the_isp_the_path_implies_and_keeps_a_converged_answer():
    """The ceilings are part of the problem, so the one term settled after the solve is the leg
    Isp. When the path implies a different Isp from the one the leg was optimized at, the solve
    re-solves once at the new value and reports it; when the path then agrees, it is settled. And
    if a pass cannot converge, the converged answer before it is returned rather than a failure."""
    pytest.importorskip("pykep")
    from datetime import date as _date

    from prospector.solvers import lambert as lb
    from prospector.solvers import simsflanagan as sf

    ev5 = dict(a_au=0.95982, e=0.082830, i_deg=7.4478, om_deg=93.1897, w_deg=235.9248,
               ma_deg=122.8655, epoch_jd=2461000.5)
    target = lb.target_planet(name="2008 EV5", **ev5)
    window, arrive = (_date(2021, 12, 26), _date(2022, 1, 15)), _date(2023, 8, 25)
    dep, arr = lb.launch_arrival_grids(window, arrive, step_days=10.0)
    seed = lb.porkchop(target, dep, arr, max_revs=2).best
    kw = dict(mass_kg=10800.0, thrust_N=1.77, launch_window=window, arrive_by=arrive,
              seed=seed, nseg=12, maxeval=600, restarts=1, vinf_dep_kms=1.5, vinf_arr_kms=0.1,
              thrust_cap_fn=lambda r: np.full(np.shape(r), 0.9))

    def isps(node_positions_m):
        return np.full((len(node_positions_m) - 1) // 2, 2800.0)     # the path says 2800 s

    sol = sf.solve(target, isp_s=3000.0, seg_isp_fn=isps, **kw)
    assert sol.feasible, sol.mismatch
    assert sol.isp_s == pytest.approx(2800.0)                        # re-solved at the path's Isp
    assert sol.refresh_settled is True
    assert sol.refresh_log and sol.refresh_log[0]["isp_before"] == pytest.approx(3000.0)
    assert sol.refresh_log[0]["isp_after"] == pytest.approx(2800.0)
    assert np.all(sol.seg_isp_s == 2800.0)
    np.testing.assert_allclose(sol.seg_caps, 0.9, atol=1e-6)         # the ceiling, everywhere

    # A pass that cannot converge: the first look says 2800 s (a pass that converges), every look
    # after says 40 s, at which no trajectory keeps the final mass above the problem's floor. The
    # answer returned is the converged 2800 s one, flagged unsettled since its own path disagrees.
    calls = {"n": 0}

    def wild(node_positions_m):
        calls["n"] += 1
        n = (len(node_positions_m) - 1) // 2
        return np.full(n, 2800.0 if calls["n"] == 1 else 40.0)

    sol = sf.solve(target, isp_s=3000.0, seg_isp_fn=wild, isp_iters=3, **kw)
    assert sol.feasible, "the converged answer must not be lost"
    assert sol.isp_s == pytest.approx(2800.0)
    assert sol.refresh_settled is False


@pytest.mark.slow
def test_a_warm_start_is_never_worse_than_the_answer_it_came_from():
    """A converged vector handed back as a warm start, under the same ceilings, must come back at
    least as good: its own throttles are admitted at the outset and the refresh tightens from
    there. Folding it into the 1 AU ceilings first dropped grid-cell refines into worse basins
    than the cells they started from."""
    pytest.importorskip("pykep")
    from datetime import date as _date

    from prospector.solvers import lambert as lb
    from prospector.solvers import simsflanagan as sf

    ev5 = dict(a_au=0.95982, e=0.082830, i_deg=7.4478, om_deg=93.1897, w_deg=235.9248,
               ma_deg=122.8655, epoch_jd=2461000.5)
    target = lb.target_planet(name="2008 EV5", **ev5)
    window, arrive = (_date(2021, 12, 26), _date(2022, 1, 15)), _date(2023, 8, 25)
    dep, arr = lb.launch_arrival_grids(window, arrive, step_days=10.0)
    seed = lb.porkchop(target, dep, arr, max_revs=2).best

    def cap(r_au):
        # Sunward segments may use full thrust, outer ones a fraction: a real-shaped ceiling.
        return np.where(np.asarray(r_au, float) < 1.0, 1.0, 0.6)

    kw = dict(mass_kg=10800.0, thrust_N=1.77, isp_s=3000.0, launch_window=window,
              arrive_by=arrive, nseg=12, maxeval=800, restarts=1, vinf_dep_kms=1.5,
              vinf_arr_kms=0.1, thrust_cap_fn=cap)
    first = sf.solve(target, seed=seed, **kw)
    assert first.feasible
    again = sf.solve(target, x0=first.decision_vector, **kw)
    assert again.feasible
    assert again.final_mass_kg >= first.final_mass_kg - 1e-6


@pytest.mark.slow
def test_the_cruise_solve_rises_above_the_frozen_point_and_rebuilds():
    """End to end on a real solve: a power-limited discrete stack aimed at a target inside 1 AU
    flies a leg whose thrust ceiling exceeds the 1 AU operating point, records per-segment Isps,
    spends propellant at their mean, and rebuilds exactly from the stored terms."""
    pytest.importorskip("pykep")
    from prospector.solvers import lambert as lb
    from prospector.solvers import simsflanagan as sf

    rc = _rc(STEPPED)
    rc = rc.model_copy(update={"mission": Mission(
        launch_orbit="LEO", launch_window=(date(2028, 7, 1), date(2028, 7, 31)),
        arrive_by=date(2030, 8, 1))})
    target = lb.target_planet(name="Apophis", a_au=0.9224, e=0.1911, i_deg=3.34,
                              om_deg=203.9, w_deg=126.7, ma_deg=215.5, epoch_jd=2461000.5)
    dep, arr = lb.launch_arrival_grids(rc.departure_window, rc.mission.arrive_by, step_days=10.0)
    seed = lb.porkchop(target, dep, arr, max_revs=2).best
    kw = dict(available_power_W=3800.0, array_model=_InverseSquare(), nseg=12,
              vinf_dep_kms=1.0, max_dep_decl_deg=None)
    sol = sf.solve_for_config(rc, target, seed=seed, maxeval=1500, restarts=2, **kw)
    assert sol.feasible, sol.mismatch
    t1, isp1 = sunpower.SunPowerModel(rc, 3800.0, _InverseSquare()).at_one_au()
    assert sol.thrust_N > t1 + 1e-6, "the leg may use more than the frozen 1 AU point"
    assert sol.seg_caps is not None and sol.seg_isp_s is not None
    assert sol.seg_isp_s.shape == (12,)
    assert set(np.round(sol.seg_isp_s[sol.seg_caps > 0], 0)) <= {1700.0, 1800.0, 1900.0} or \
        np.all((sol.seg_isp_s >= 1700.0 - 1e-6) & (sol.seg_isp_s <= 1900.0 + 1e-6))
    assert 1700.0 - 1e-6 <= sol.isp_s <= 1900.0 + 1e-6
    for i in range(12):
        assert sol.throttle[2 * i] <= sol.seg_caps[i] + 1e-4
    re = sf.rebuild_for_config(rc, target, sol.decision_vector, seg_caps=sol.seg_caps,
                               thrust_N=sol.thrust_N, isp_s=sol.isp_s, seg_isp_s=sol.seg_isp_s,
                               **kw)
    assert re.feasible == sol.feasible and re.isp_s == sol.isp_s and re.thrust_N == sol.thrust_N
    np.testing.assert_allclose(re.trajectory, sol.trajectory, rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(re.seg_isp_s, sol.seg_isp_s)


# ---------------------------------------------------------------------------
# the smooth ceiling the optimizer differentiates
# ---------------------------------------------------------------------------


def test_smooth_cap_turns_a_mode_step_into_a_ramp_below_it_with_a_consistent_slope():
    """A discrete stack's ceiling is a staircase in distance. SmoothCap rounds each step into a
    ramp a few hundredths of an AU wide that never sits ABOVE the staircase (eroded first, so an
    optimizer cannot draw power just past a step), holds the plateaus, clamps outside its range,
    and reports a slope that matches finite differences of its own value."""
    exact = lambda r: np.where(np.asarray(r) < 1.0, 1.0, 0.6)  # noqa: E731
    step = sunpower.SmoothCap.from_function(exact, sigma_au=0.02)
    r = np.array([0.5, 0.8, 0.97, 1.2, 1.5, 3.0])
    cap, slope = step.with_derivative(r)
    np.testing.assert_allclose(cap[[0, 1]], 1.0, atol=1e-3)         # plateaus, to the tail
    np.testing.assert_allclose(cap[[3, 4, 5]], 0.6, atol=1e-3)
    assert 0.6 < cap[2] < 1.0 and slope[2] < 0                      # on the ramp, falling
    assert np.all(np.abs(slope[[0, 1, 4, 5]]) < 1e-3)
    # Never above the true ceiling, anywhere; and the ramp is done by the step itself, so the
    # dim side is credited nothing it does not have, while 0.12 AU sunward the plateau holds.
    rr = np.linspace(0.3, 3.0, 2000)
    assert np.all(step(rr) <= exact(rr) + 1e-3)
    assert step(1.0) < 0.61 and step(0.88) > 0.99
    # Slope against a central difference of the value itself, on and off the ramp.
    for x in (0.97, 1.0, 1.03, 2.0):
        fd = (step(x + 1e-5) - step(x - 1e-5)) / 2e-5
        assert step.with_derivative(x)[1] == pytest.approx(fd, abs=1e-3)
    # Outside the sampled range the end value holds with zero slope.
    c, s = step.with_derivative(np.array([0.05, 20.0]))
    np.testing.assert_allclose(c, [1.0, 0.6], atol=1e-3)
    assert np.all(s == 0.0)
    # Values are clipped to [0, 1] whatever the function said.
    wild = sunpower.SmoothCap.from_function(lambda r: 3.0 - 2.0 * np.asarray(r))
    assert np.all(wild(np.array([0.3, 1.0, 2.0, 5.0])) <= 1.0)
    assert np.all(wild(np.array([0.3, 1.0, 2.0, 5.0])) >= 0.0)
    flat = sunpower.SmoothCap.constant(0.0)
    assert np.all(flat(np.array([0.5, 1.0, 2.0])) == 0.0)


def test_the_model_supplies_a_smooth_cap_and_segment_isps():
    """The config model hands the solver a SmoothCap of its own thrust-against-distance, as a
    fraction of the leg thrust, and the Isp each segment of a path runs at."""
    rc = _rc(STEPPED)
    model = sunpower.SunPowerModel(rc, 3800.0, _InverseSquare())
    leg_thrust = model.leg_thrust_N(0.9)
    cap = model.smooth_cap(leg_thrust)
    assert isinstance(cap, sunpower.SmoothCap)
    # Far out the array cannot run the stack: the ceiling falls to zero; at the closest approach
    # it is the leg thrust itself.
    assert cap(3.0) == 0.0
    assert cap(0.5) == pytest.approx(1.0, abs=1e-3)      # power to spare: the leg thrust itself
    assert 0.0 < cap(0.9) <= 1.0                         # the closest approach, next to a mode step
    assert 0.0 <= cap(1.0) <= 1.0
    # Equal to the exact staircase away from its steps, within a narrow ramp at each: the solve
    # constrains the mean over a segment's arc, where the ramps' over- and under-credit cancel.
    rr = np.linspace(0.3, 3.0, 2000)
    exact = np.minimum(np.asarray(model.thrust_N(rr), float) / leg_thrust, 1.0)
    close = np.abs(cap(rr) - exact) < 1e-3
    assert close.mean() > 0.9, "the ramps must be narrow against the distances a cruise spans"
    assert np.all(np.abs(cap(rr) - exact) <= 1.0)
    au = 1.495978707e11
    nodes = np.array([[1.0, 0, 0], [1.0, 0.05, 0], [1.0, 0.1, 0]]) * au
    isps = model.segment_isps(nodes, leg_thrust)
    assert isps.shape == (1,) and 1700.0 - 1e-6 <= isps[0] <= 1900.0 + 1e-6
    assert model.smooth_cap(0.0)(1.0) == 0.0
