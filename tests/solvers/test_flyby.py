"""The two-leg gravity-assist problem: gradients, the flyby constraints, a cell, and Dawn.

The analytic gradient is checked against central differences, with a controlled step on the three
time columns (pygmo's default step on an epoch of a few thousand days is too coarse) and the
residual allowed is the ephemeris's own departure from Keplerian motion, the approximation pykep's
single-leg wrapper also makes. Dawn's Mars flyby is the convergence check: the flown mission is the
answer key.
"""
from datetime import date

import numpy as np
import pygmo as pg
import pykep as pk
import pytest

from prospector.population import targets
from prospector.solvers import flyby as fb
from prospector.solvers import lambert as lb


def _mars():
    return pk.planet(pk.udpla.jpl_lp("mars"))


def _dawn_udp(nseg=(4, 4), **over):
    kw = dict(mass_kg=1217.7, thrust_N=0.0927, isp_s=3127.0, nseg=nseg,
              launch_window=(date(2007, 9, 26), date(2007, 10, 15)), arrive_by=date(2011, 7, 16),
              vinf_dep_kms=3.3, vinf_arr_kms=0.1, window_slack_days=0.0, min_tof_days=(60.0, 120.0),
              max_tof_days=None, max_duty_cycle=1.0, max_dep_decl_deg=20.0, depart_body=None,
              min_flyby_alt_km=500.0, flyby_mu=None)
    kw.update(over)
    vesta = lb.planet_from_row(targets.get_target("4 Vesta"))
    return fb._make_udp(vesta, _mars(), **kw)


def _point(udp, seed=3):
    """An arbitrary interior point: not a solution, just somewhere every constraint has slope."""
    rng = np.random.default_rng(seed)
    lo, hi = (np.asarray(b, float) for b in udp.get_bounds())
    x = np.zeros(udp.dim)
    x[fb._T0], x[fb._TOF1], x[fb._TOF2] = lo[0] + 5.0, 500.0, 850.0
    x[fb._MFB], x[fb._MF] = 1150.0, 980.0
    x[fb._VDEP] = rng.normal(0, 1500, 3)
    x[fb._VIN] = rng.normal(0, 2500, 3)
    x[fb._VOUT] = x[fb._VIN] * 0.97 + rng.normal(0, 300, 3)
    x[fb._VARR] = rng.normal(0, 40, 3)
    x[udp._u1] = rng.uniform(-0.5, 0.5, 3 * udp.n1)
    x[udp._u2] = rng.uniform(-0.5, 0.5, 3 * udp.n2)
    return np.clip(x, lo, hi)


def _check_gradient(udp, rows=None, tol_time=1e-3, tol=2e-4):
    prob = pg.problem(udp)
    x = _point(udp)
    nrows = 1 + prob.get_nec() + prob.get_nic()
    G = np.asarray(udp.gradient(x)).reshape(nrows, udp.dim)
    sel = slice(None) if rows is None else rows
    for col in range(udp.dim):
        h = 1e-4 if col < 3 else (1e-3 if col < 5 else (0.5 if col < 17 else 1e-5))
        xp, xm = x.copy(), x.copy()
        xp[col] += h
        xm[col] -= h
        est = (np.asarray(prob.fitness(xp)) - np.asarray(prob.fitness(xm))) / (2 * h)
        ref = np.maximum(np.abs(est[sel]), 1e-6)
        rel = np.abs(G[sel, col] - est[sel]) / ref
        rel[np.abs(est[sel]) < 1e-9] = 0.0
        # The three epoch columns carry the ephemeris's non-Keplerian residual (the analytic
        # planets' velocity is not exactly d(position)/dt), about a part in a thousand.
        assert np.max(rel) < (tol_time if col < 3 else tol), \
            f"column {col}: worst relative error {np.max(rel):.2e}"
    return nrows


def test_gradient_matches_central_differences():
    _check_gradient(_dawn_udp())


def test_gradient_with_a_sun_distance_ceiling_matches_central_differences():
    """The ceiling rows run through both legs' chains; an inverse-square ceiling has slope
    everywhere, so every column the chain touches is exercised."""
    from prospector.solvers.sunpower import SmoothCap
    cap = SmoothCap.from_function(lambda r: np.clip(1.0 / np.asarray(r) ** 2, 0.0, 1.0),
                                  sigma_au=0.05, erode_sigmas=0.0)
    udp = _dawn_udp(cap_fn=cap)
    nrows = 1 + 15 + (8 + 3 + 1 + 1 + 8)
    assert _check_gradient(udp, rows=slice(nrows - 8, nrows)) == nrows
    # And the ceiling actually bites at this point: some segment sits under a ceiling below 1.
    caps = udp.segment_caps(_point(udp))
    assert caps is not None and min(c.min() for c in caps) < 1.0


def test_total_flight_time_bounds_are_two_more_rows():
    """A grid cell holds tof1 + tof2 to its flight time: two inequalities with unit slope on the
    two flight-time columns and none elsewhere."""
    udp = _dawn_udp(tof_total_bounds=(1340.0, 1360.0))
    prob = pg.problem(udp)
    assert prob.get_nic() == 8 + 3 + 1 + 2 + 1
    x = _point(udp)                                   # tof1 + tof2 = 1350: inside the range
    f = np.asarray(prob.fitness(x))
    rows = 1 + 15 + 8 + 3 + 1                         # after the deadline row
    assert f[rows] == pytest.approx(1350.0 - 1360.0) and f[rows + 1] == pytest.approx(1340.0 - 1350.0)
    _check_gradient(udp)
    tight = _dawn_udp(tof_total_bounds=(1400.0, 1402.0))
    assert not pg.problem(tight).feasibility_x(x)


def test_exact_departure_speed_is_one_more_row():
    """A launcher-provided escape pins the departure speed: the ceiling row gains a lower side,
    violated below the speed, met at it, with its gradient checked like the rest."""
    udp = _dawn_udp(vinf_dep_exact=True)
    assert pg.problem(udp).get_nic() == 8 + 3 + 1 + 1 + 1
    x = _point(udp)
    x[fb._VDEP] = [3300.0, 0.0, 0.0]
    f = np.asarray(pg.problem(udp).fitness(x))
    row = 1 + 15 + 8                                       # the ceiling row, then the floor row
    assert abs(f[row]) < 1e-6 and abs(f[row + 1]) < 1e-6
    x[fb._VDEP] = [2000.0, 0.0, 0.0]
    f = np.asarray(pg.problem(udp).fitness(x))
    assert f[row] < 0 < f[row + 1]
    _check_gradient(udp)


def test_flyby_constraint_gradient():
    udp = _dawn_udp()
    rng = np.random.default_rng(7)
    vin = rng.normal(0, 3000, 3)
    vout = rng.normal(0, 3000, 3)
    eq, ineq, d_eq, d_ineq = udp._flyby(vin, vout)
    for k in range(6):
        h = 1e-2
        p = np.concatenate((vin, vout)).astype(float)
        pp, pm = p.copy(), p.copy()
        pp[k] += h
        pm[k] -= h
        eqp, inp, _, _ = udp._flyby(pp[:3], pp[3:])
        eqm, inm, _, _ = udp._flyby(pm[:3], pm[3:])
        assert d_eq[k] == pytest.approx((eqp - eqm) / (2 * h), rel=1e-6, abs=1e-14)
        assert d_ineq[k] == pytest.approx((inp - inm) / (2 * h), rel=1e-6, abs=1e-14)


def test_flyby_constraints_mean_what_they_say():
    udp = _dawn_udp()
    v = np.array([3000.0, 0.0, 0.0])
    # Same speed, no turn: both constraints satisfied.
    eq, ineq, _, _ = udp._flyby(v, v)
    assert eq == 0.0 and ineq < 0
    # Same speed, a turn beyond what Mars can bend at 500 km: violated.
    rp, mu = udp.fb_rp, udp.fb_mu
    e = 1.0 + v @ v * rp / mu
    max_turn = 2.0 * np.arcsin(1.0 / e)
    for turn, ok in ((0.9 * max_turn, True), (1.1 * max_turn, False)):
        vout = np.array([np.cos(turn), np.sin(turn), 0.0]) * 3000.0
        _, ineq, _, _ = udp._flyby(v, vout)
        assert (ineq <= 0) == ok
    # A speed change is an equality violation however small the turn.
    eq, _, _, _ = udp._flyby(v, v * 1.01)
    assert eq != 0.0
    # The reported periapsis inverts the turn: at the maximum turn it is the floor.
    vout = np.array([np.cos(max_turn), np.sin(max_turn), 0.0]) * 3000.0
    x = np.zeros(udp.dim)
    x[fb._VIN], x[fb._VOUT] = v, vout
    geo = udp.flyby_geometry(x)
    assert geo["periapsis_alt_km"] == pytest.approx(500.0, abs=1e-3)
    assert geo["turn_deg"] == pytest.approx(np.degrees(max_turn), abs=1e-6)


def test_the_impulsive_start_is_a_consistent_geometry_at_the_days_asked():
    """The start for a cell: departure on the cell's day, the total flight time within a day of
    the cell's, the flyby somewhere between, engine off, every velocity inside the bounds."""
    udp = _dawn_udp(nseg=(3, 3))
    t0 = lb.mjd2000_from_date(date(2007, 10, 1))
    z, dv = fb._impulsive_start(udp, t0, t0 + 1370.0)
    assert 3.0 < dv < 12.0                           # km/s: the impulsive route, without launch
    assert abs(z[fb._T0] - t0) <= 0.5
    assert abs(z[fb._TOF1] + z[fb._TOF2] - 1370.0) <= 1.0
    assert 60.0 <= z[fb._TOF1] and 120.0 <= z[fb._TOF2]
    assert np.linalg.norm(z[fb._VDEP]) <= 3300.0 + 1e-6
    assert np.all(z[udp._u1] == 0.0) and np.all(z[udp._u2] == 0.0)
    assert np.array_equal(z, fb._impulsive_start(udp, t0, t0 + 1370.0)[0])   # fixed seed


def test_a_hopeless_cell_is_returned_unsolved_in_a_second():
    """A cell whose impulsive route needs more than the engine can deliver in its flight time
    comes back infeasible without a descent; the deliverable bound is the tank's or the engine's,
    whichever is less."""
    import time

    from prospector import paths
    from prospector.config import load_study, resolve_study
    assert fb._dv_deliverable(13.0, 1217.7, 0.0927, 3127.0, 1300.0) == pytest.approx(
        min(13.0, 0.0927 * 1300 * 86400 / (1217.7 * np.exp(-13000 / (3127 * fb.G0))) / 1000))
    rc = resolve_study(load_study("dawn", config_dir=paths.EXAMPLE_CONFIG_DIR),
                       config_dir=paths.EXAMPLE_CONFIG_DIR)
    rc.departure_vinf_kms = 3.3
    vesta = lb.planet_from_row(targets.get_target("4 Vesta"))
    t0 = lb.mjd2000_from_date(date(2007, 10, 1))
    t = time.time()
    sol = fb.solve_cell_for_config(rc, vesta, dep_mjd2000=t0, tof_days=400.0, nseg=4,
                                   max_duty_cycle=0.95, vinf_dep_kms=3.3)
    assert not sol.feasible and time.time() - t < 10.0


def test_a_cell_solve_is_deterministic_and_holds_its_flight_time():
    """Two runs of the same cell give the same vector; the trip stays within the cell's tolerance
    of the flight time asked for and departs on its day."""
    from prospector import paths
    from prospector.config import load_study, resolve_study
    rc = resolve_study(load_study("dawn", config_dir=paths.EXAMPLE_CONFIG_DIR),
                       config_dir=paths.EXAMPLE_CONFIG_DIR)
    rc.departure_vinf_kms = 3.3
    vesta = lb.planet_from_row(targets.get_target("4 Vesta"))
    t0 = lb.mjd2000_from_date(date(2007, 10, 1))
    kw = dict(dep_mjd2000=t0, tof_days=1370.0, nseg=4, restarts=1, max_duty_cycle=0.95,
              vinf_dep_kms=3.3, maxeval=400)
    a = fb.solve_cell_for_config(rc, vesta, **kw)
    b = fb.solve_cell_for_config(rc, vesta, **kw)
    assert np.array_equal(a.decision_vector, b.decision_vector)
    assert abs(a.tof_days - 1370.0) <= 0.75 + 1e-6
    assert abs(a.dep_mjd2000 - t0) <= 0.75 + 1e-6
    assert a.flyby_name == "mars" and a.nseg == (4, 4)


@pytest.mark.slow
def test_dawn_via_mars_reproduces_the_flown_propellant():
    """Dawn spent 247 kg of xenon from launch to Vesta with a Mars flyby; a direct transfer needs
    about 272 kg. From one cell of the window the two-leg solve must land in between or below,
    legally, with the flyby within a season of the flown one (17 Feb 2009 at 542 km)."""
    vesta = lb.planet_from_row(targets.get_target("4 Vesta"))
    window, deadline = (date(2007, 9, 26), date(2007, 10, 15)), date(2011, 7, 16)
    t0 = lb.mjd2000_from_date(window[0])
    sol = fb.solve(vesta, _mars(), mass_kg=1217.7, thrust_N=0.0927, isp_s=3127.0,
                   vinf_dep_kms=3.3, launch_window=window, arrive_by=deadline,
                   seed=(t0 + 5.0, lb.mjd2000_from_date(deadline)), max_duty_cycle=0.95,
                   min_flyby_alt_km=500.0, nseg=(10, 10), restarts=3)
    assert sol.feasible, sol.mismatch
    assert 190.0 < sol.propellant_kg < 260.0
    assert sol.flyby["periapsis_alt_km"] >= 500.0 - 1.0
    assert sol.flyby["vinf_kms"] == pytest.approx(sol.flyby["vinf_out_kms"], abs=1e-3)
    assert date(2007, 9, 26) <= sol.dep_date.date() <= date(2007, 10, 15)
    assert sol.arr_date.date() <= date(2011, 7, 16)
    assert date(2008, 9, 1) <= sol.flyby_date.date() <= date(2010, 3, 1)
    # Both legs' nodes end and start at Mars at the flyby epoch.
    n1, n2 = sol.leg_trajectories
    r_mars = np.asarray(_mars().eph(sol.flyby_mjd2000)[0], float)
    assert np.linalg.norm(n1[-1, 1:4] - r_mars) < 2e7 and np.linalg.norm(n2[0, 1:4] - r_mars) < 2e7
