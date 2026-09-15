"""Integration test for the Sims-Flanagan low-thrust solver (prospector.solvers.simsflanagan).

Runs one real PyKEP + pygmo MBH solve (deterministic seed) of a reference transfer -- Earth -> 2008
EV5 on a high-power SEP stack, and pins that it converges to a feasible, flyable rendezvous with a
sane low-thrust dV and flight time. This is the calibrated-physics regression for the low-thrust
solve; it takes 10-15 s. PyKEP and pygmo come from the pixi env.
"""
from datetime import date

import numpy as np
import pytest

pytest.importorskip("pykep")
pytest.importorskip("pygmo")
from prospector import config  # noqa: E402
from prospector.solvers import lambert as lb  # noqa: E402
from prospector.solvers import simsflanagan as sf  # noqa: E402

EV5 = dict(a_au=0.95982, e=0.082830, i_deg=7.4478, om_deg=93.1897, w_deg=235.9248,
           ma_deg=122.8655, epoch_jd=2461000.5)
LAUNCH = (date(2021, 12, 26), date(2022, 1, 15))
ARRIVE_BY = date(2023, 8, 25)
# A ~10.8 t wet vehicle, 1.77 N total thrust, ~3000 s Isp. Inline, not from the library:
MASS, THRUST, ISP = 10800.0, 1.77, 3000.0


@pytest.fixture(scope="module")
def solution():
    target = lb.target_planet(name="2008 EV5", **EV5)
    dep, arr = lb.launch_arrival_grids(LAUNCH, ARRIVE_BY, step_days=10.0)
    seed = lb.porkchop(target, dep, arr, max_revs=2).best
    calls = []
    sol = sf.solve(target, mass_kg=MASS, thrust_N=THRUST, isp_s=ISP,
                   launch_window=LAUNCH, arrive_by=ARRIVE_BY, seed=seed,
                   nseg=15, maxeval=1500, restarts=3, vinf_dep_kms=1.5, vinf_arr_kms=0.1,
                   max_duty_cycle=1.0, progress=lambda d, t: calls.append((d, t)))
    sol._progress_calls = calls
    return sol


def test_converges_to_feasible_rendezvous(solution):
    # The matchpoint mismatch is the robust convergence measure. The boolean `feasible`
    # flag sits at a 1e-4 tolerance that BLAS-threaded finite-diff gradients can jitter
    # across, so pin the unitless mismatch to the design tolerance instead.
    assert solution.mismatch < 1e-3          # matchpoint closed to design tolerance


def test_low_thrust_dv_and_time_are_sane(solution):
    # Below Lambert's impulsive ~5.7 km/s (continuous thrust matches the inclined orbit more
    # efficiently), and a multi-hundred-day cruise like the real ~425-day ARRM outbound.
    assert 2.0 < solution.dv_kms < 6.0
    assert 300 < solution.tof_days < 650
    assert 0 < solution.propellant_kg < (MASS - 5800)   # within the usable-Xe budget


def test_departs_within_the_launch_window(solution):
    lo = lb.mjd2000_from_date(LAUNCH[0]) - 25
    hi = lb.mjd2000_from_date(LAUNCH[1]) + 25
    assert lo <= solution.dep_mjd2000 <= hi


def test_return_leg_swaps_departure_body_and_disables_cone():
    # The return leg departs the asteroid and arrives at Earth, the exact mirror of the outbound's
    # arrival-body swap, and carries no launch escape cone (it is already in heliocentric space).
    # This pins the mechanism without running an optimization.
    earth = lb.earth_planet()
    asteroid = lb.target_planet(name="2008 EV5", **EV5)
    udp = sf._make_udp(earth, mass_kg=600.0, thrust_N=0.2, isp_s=3000.0, nseg=10,
                       launch_window=(date(2030, 1, 1), date(2030, 1, 1)),
                       arrive_by=date(2031, 6, 1), vinf_dep_kms=0.1, vinf_arr_kms=0.4,
                       window_slack_days=0.0, min_tof_days=120.0, max_tof_days=None,
                       max_duty_cycle=1.0, max_dep_decl_deg=None, depart_body=asteroid)
    assert udp.base.pls is asteroid and udp.base.plf is earth
    assert udp.max_dep_decl_deg is None                # no escape cone on the homebound leg
    # The fitness still evaluates (it only calls p0.eph()/pf.eph()).
    lo, hi = udp.get_bounds()
    f = udp.fitness(list((np.asarray(lo) + np.asarray(hi)) / 2.0))
    assert len(f) >= 1


def test_arrives_by_the_deadline(solution):
    # The flight-time cap is hard: the whole transfer must land within the mission window.
    assert solution.dep_mjd2000 + solution.tof_days <= lb.mjd2000_from_date(ARRIVE_BY) + 1e-6


def test_window_too_short_raises():
    # A deadline that leaves less than the minimum flight time, even for the earliest departure
    # -- must fail loudly, not silently return a transfer that arrives late.
    target = lb.target_planet(name="2008 EV5", **EV5)
    with pytest.raises(ValueError, match="too short"):
        sf.solve(target, mass_kg=MASS, thrust_N=THRUST, isp_s=ISP,
                 launch_window=(date(2028, 1, 1), date(2028, 1, 10)),
                 arrive_by=date(2028, 3, 1), nseg=8, maxeval=10, restarts=1)


def test_trajectory_and_throttle_are_physical(solution):
    assert solution.trajectory.shape == (2 * solution.nseg + 1, 12)
    assert solution.throttle.max() <= 1.0001          # throttle magnitude bounded
    assert solution.throttle.min() >= -1e-6
    # The solve records the thrust cap it ran under, and the throttle never exceeds it (the
    # array-power ceiling the UI draws). This fixture is power-rich (limit 1.0).
    assert solution.max_duty_cycle == pytest.approx(1.0)
    assert solution.throttle.max() <= solution.max_duty_cycle + 1e-4
    r_au = np.linalg.norm(solution.positions_au, axis=1)
    assert 0.85 < r_au[0] < 1.05                       # starts at Earth
    assert 0.80 < r_au[-1] < 1.10                      # ends on 2008 EV5's orbit


def test_thrust_limit_caps_the_throttle():
    # A sub-1 thrust limit (e.g. a power-limited cruise: 90% Max thrust on a 50%-power array ->
    # 0.45) is recorded on the solution and bounds the throttle, so the plotted ceiling is real.
    target = lb.target_planet(name="2008 EV5", **EV5)
    dep, arr = lb.launch_arrival_grids(LAUNCH, ARRIVE_BY, step_days=10.0)
    seed = lb.porkchop(target, dep, arr, max_revs=2).best
    sol = sf.solve(target, mass_kg=MASS, thrust_N=THRUST, isp_s=ISP,
                   launch_window=LAUNCH, arrive_by=ARRIVE_BY, seed=seed,
                   nseg=12, maxeval=400, restarts=1, vinf_dep_kms=1.5, vinf_arr_kms=0.1,
                   max_duty_cycle=0.45)
    assert sol.max_duty_cycle == pytest.approx(0.45)
    assert sol.throttle.max() <= 0.45 + 1e-4


def test_progress_was_reported(solution):
    assert solution._progress_calls and solution._progress_calls[-1] == (3, 3)


def _ecl_from_eq(v):
    """Rotate an equatorial vector into PyKEP's ecliptic frame (inverse of the UDP's)."""
    eps = sf._OBLIQUITY_RAD
    return np.array([v[0], v[1] * np.cos(eps) + v[2] * np.sin(eps),
                     -v[1] * np.sin(eps) + v[2] * np.cos(eps)])


def test_departure_cone_constraint_in_the_udp():
    """The launch-reachable departure cone: the escape exits in the spiral's plane, so
    the departure v∞'s EQUATORIAL declination is launch's to dictate, the optimizer
    gets an inequality, not a free 3-vector. Checked at the UDP level (no
    optimization): the constraint counts toward get_nic, signs correctly for
    in/out-of-cone aims measured in the right frame, and is absent at >= 90 deg."""
    target = lb.target_planet(name="2008 EV5", **EV5)
    common = dict(mass_kg=MASS, thrust_N=THRUST, isp_s=ISP, nseg=10,
                  launch_window=LAUNCH, arrive_by=ARRIVE_BY,
                  vinf_dep_kms=1.5, vinf_arr_kms=0.1, window_slack_days=0.0,
                  min_tof_days=120.0, max_tof_days=None, max_duty_cycle=1.0)
    capped = sf._make_udp(target, max_dep_decl_deg=30.0, **common)
    free = sf._make_udp(target, max_dep_decl_deg=None, **common)
    polar = sf._make_udp(target, max_dep_decl_deg=90.0, **common)
    assert capped.get_nic() == free.get_nic() + 1
    assert polar.get_nic() == free.get_nic()

    lo, _ = capped.get_bounds()
    x = np.zeros(len(lo))
    x[sf._I_T0], x[capped._i_tof], x[sf._I_MF] = 8030.0, 300.0, MASS * 0.9
    # Declination 20 deg at 1 km/s (built in the equatorial frame, handed to the UDP in ecliptic
    # components): inside the 30 deg cone -> constraint <= 0.
    d20 = 1000.0 * np.array([np.cos(np.radians(20.0)), 0.0, np.sin(np.radians(20.0))])
    x[sf._I_VINF_DEP] = _ecl_from_eq(d20)
    assert capped.fitness(x)[-1] <= 1e-12
    # Declination 45 deg: outside -> violated.
    d45 = 1000.0 * np.array([np.cos(np.radians(45.0)), 0.0, np.sin(np.radians(45.0))])
    x[sf._I_VINF_DEP] = _ecl_from_eq(d45)
    assert capped.fitness(x)[-1] > 0.0
    # The frame matters: an ECLIPTIC latitude of 40 deg pointed where the equator leans away can
    # still be inside a 30 deg declination cone, the old latitude-frame cone would wrongly forbid
    # it.
    lat = np.radians(40.0)
    x[sf._I_VINF_DEP] = 1000.0 * np.array([0.0, -np.cos(lat), np.sin(lat)])  # ecliptic
    vz_eq = x[3] * np.sin(sf._OBLIQUITY_RAD) + x[4] * np.cos(sf._OBLIQUITY_RAD)
    assert np.degrees(np.arcsin(vz_eq / 1000.0)) < 30.0           # truly inside, in-frame
    assert capped.fitness(x)[-1] <= 1e-12
    # The free problem's fitness length is one shorter, all else equal.
    assert len(capped.fitness(x)) == len(free.fitness(x)) + 1


def test_seed_projection_folds_into_the_cone():
    """An impulsive seed aiming outside the launch cone starts the descent infeasible;
    the seed builder folds its out-of-plane (equatorial) component onto the cone edge
    instead."""
    target = lb.target_planet(name="2008 EV5", **EV5)
    udp = sf._make_udp(target, mass_kg=MASS, thrust_N=THRUST, isp_s=ISP, nseg=10,
                       launch_window=LAUNCH, arrive_by=ARRIVE_BY,
                       vinf_dep_kms=1.5, vinf_arr_kms=0.1, window_slack_days=0.0,
                       min_tof_days=120.0, max_tof_days=None, max_duty_cycle=1.0,
                       max_dep_decl_deg=25.0)
    dep, arr = lb.launch_arrival_grids(LAUNCH, ARRIVE_BY, step_days=10.0)
    seed = lb.porkchop(target, dep, arr, max_revs=2).best
    # Force a steeply out-of-plane seed v∞ regardless of the real geometry.
    seed.v_transfer_dep_ms = seed.v_dep_body_ms + np.array([300.0, 0.0, 1200.0])
    z = sf._seed_decision_vector(seed, udp, MASS, 10)
    vx, vy, vz = z[sf._I_VINF_DEP]
    vy_eq = vy * np.cos(sf._OBLIQUITY_RAD) - vz * np.sin(sf._OBLIQUITY_RAD)
    vz_eq = vy * np.sin(sf._OBLIQUITY_RAD) + vz * np.cos(sf._OBLIQUITY_RAD)
    decl = np.degrees(np.arctan2(abs(vz_eq), np.hypot(vx, vy_eq)))
    assert decl <= 25.0 + 1e-6
    assert udp.fitness(z)[-1] <= 1e-12


def test_seg_caps_constraints_in_the_udp():
    """Per-segment thrust ceilings at the UDP level (no optimization): one inequality per
    segment counted by get_nic, signed correctly for throttles inside/outside their cap,
    and absent when no caps are given."""
    target = lb.target_planet(name="2008 EV5", **EV5)
    common = dict(mass_kg=MASS, thrust_N=THRUST, isp_s=ISP, nseg=10,
                  launch_window=LAUNCH, arrive_by=ARRIVE_BY,
                  vinf_dep_kms=1.5, vinf_arr_kms=0.1, window_slack_days=0.0,
                  min_tof_days=120.0, max_tof_days=None, max_duty_cycle=1.0,
                  max_dep_decl_deg=None)
    caps = np.linspace(1.0, 0.5, 10)
    capped = sf._make_udp(target, seg_caps=caps, **common)
    free = sf._make_udp(target, **common)
    assert capped.get_nic() == free.get_nic() + 10
    with pytest.raises(ValueError, match="one entry per segment"):
        sf._make_udp(target, seg_caps=[0.5, 0.5], **common)

    lo, _ = capped.get_bounds()
    x = np.zeros(len(lo))
    x[sf._I_T0], x[capped._i_tof], x[sf._I_MF] = 8030.0, 300.0, MASS * 0.9
    seg0, seg9 = sf._I_THROTTLE0, sf._I_THROTTLE0 + 3 * 9
    x[seg0:seg0 + 3] = [0.3, 0.0, 0.0]           # segment 0: |u| = 0.3 < cap 1.0
    x[seg9:seg9 + 3] = [0.0, 0.8, 0.0]           # segment 9: |u| = 0.8 > cap 0.5
    f = capped.fitness(x)
    # Appended after the base fitness (objective, equalities, base inequalities) and the
    # arrival-deadline row.
    k0 = 1 + capped.get_nec() + capped.base.get_nic() + 1
    seg_terms = f[k0: k0 + 10]
    assert seg_terms[0] <= 0.0                     # inside its cap
    assert seg_terms[9] > 0.0                      # violates its cap
    # Folding the champion inside the caps restores feasibility of the cap rows.
    z = sf._fold_throttles_to_caps(x, caps)
    f2 = capped.fitness(z)
    assert all(t <= 1e-12 for t in f2[k0: k0 + 10])
    assert np.allclose(z[2:8], x[2:8])             # v-infinities untouched


def test_sun_distance_caps_bound_the_converged_throttle():
    """The ceiling inside the solve: with a synthetic sun-distance cap function the converged
    throttle respects every segment's ceiling at its own midpoint, the solution records the caps
    (physical-max fractions), and a rebuild with the stored caps reproduces the solve exactly."""
    target = lb.target_planet(name="2008 EV5", **EV5)
    dep, arr = lb.launch_arrival_grids(LAUNCH, ARRIVE_BY, step_days=10.0)
    seed = lb.porkchop(target, dep, arr, max_revs=2).best
    def cap_fn(r_au):
        return 1.0 - 0.6 * np.clip(np.asarray(r_au, float) - 0.95, 0.0, None)

    kw = dict(mass_kg=MASS, thrust_N=THRUST, isp_s=ISP,
              launch_window=LAUNCH, arrive_by=ARRIVE_BY,
              nseg=12, maxeval=600, restarts=1, vinf_dep_kms=1.5, vinf_arr_kms=0.1,
              max_duty_cycle=0.9)
    sol = sf.solve(target, seed=seed, thrust_cap_fn=cap_fn, **kw)
    assert sol.seg_caps is not None and sol.seg_caps.shape == (12,)
    # Caps are on the throttle's basis (physical-max fractions), composed with the duty cap.
    assert np.all(sol.seg_caps <= 0.9 + 1e-9)
    assert sol.seg_caps.min() < 0.9 - 1e-3           # the far segments really are capped below
    # Each segment's throttle (constant over the segment, held at nodes 2i and 2i+1) stays
    # inside its own ceiling.
    for i in range(12):
        cap = sol.seg_caps[i]
        assert sol.throttle[2 * i] <= cap + 1e-4
        assert sol.throttle[2 * i + 1] <= cap + 1e-4
    # The stored caps rebuild the exact capped problem: same trajectory, same feasibility.
    re = sf.rebuild_solution(target, x=sol.decision_vector, seg_caps=sol.seg_caps,
                             mass_kg=MASS, thrust_N=THRUST, isp_s=ISP,
                             launch_window=LAUNCH, arrive_by=ARRIVE_BY,
                             nseg=12, vinf_dep_kms=1.5, vinf_arr_kms=0.1, max_duty_cycle=0.9)
    assert re.feasible == sol.feasible
    np.testing.assert_allclose(re.seg_caps, sol.seg_caps, rtol=1e-12)
    np.testing.assert_allclose(re.trajectory, sol.trajectory, rtol=1e-9, atol=1e-6)


def test_rebuild_solution_reproduces_the_solve(solution):
    """A stored decision vector reconstructs the same trajectory against the same
    problem terms, no re-optimization. This is how the v∞ sweep's winning point comes
    back as a full run: the sweep keeps only the vector, the worker rebuilds it."""
    target = lb.target_planet(name="2008 EV5", **EV5)
    re = sf.rebuild_solution(target, x=solution.decision_vector,
                             mass_kg=MASS, thrust_N=THRUST, isp_s=ISP,
                             launch_window=LAUNCH, arrive_by=ARRIVE_BY,
                             nseg=15, vinf_dep_kms=1.5, vinf_arr_kms=0.1,
                             max_duty_cycle=1.0,
                             maxeval=999, restarts=9, seed=None)   # solver-only knobs ignored
    assert re.feasible == solution.feasible
    assert re.dep_mjd2000 == pytest.approx(solution.dep_mjd2000, abs=1e-9)
    assert re.tof_days == pytest.approx(solution.tof_days, abs=1e-9)
    assert re.dv_kms == pytest.approx(solution.dv_kms, rel=1e-12)
    assert re.final_mass_kg == pytest.approx(solution.final_mass_kg, rel=1e-12)
    np.testing.assert_allclose(re.trajectory, solution.trajectory, rtol=1e-9, atol=1e-6)


def test_the_same_inputs_solve_to_the_same_trajectory():
    """Two identical solves must agree exactly, and a different seed must be free to differ.

    MBH perturbs from its own generator, which pygmo seeds randomly unless told otherwise. That
    made the solver non-reproducible while the seeded population made it look deterministic: the
    same vehicle evaluated twice could return a different trajectory and a different verdict, and
    the design search's reseeded retry (1337) only means something if the default is fixed.
    Measured before the fix, one converged quantity wandered from -0.82 to +2e-5 across runs; after
    it, every run is bit-identical.
    """
    target = lb.target_planet(name="2008 EV5", **EV5)
    dep, arr = lb.launch_arrival_grids(LAUNCH, ARRIVE_BY, step_days=10.0)
    seed = lb.porkchop(target, dep, arr, max_revs=2).best
    kw = dict(mass_kg=MASS, thrust_N=THRUST, isp_s=ISP, launch_window=LAUNCH,
              arrive_by=ARRIVE_BY, nseg=6, maxeval=120, restarts=1, max_duty_cycle=0.9)

    a = sf.solve(target, seed=seed, **kw)
    b = sf.solve(target, seed=seed, **kw)
    assert a.dv_kms == b.dv_kms
    assert a.final_mass_kg == b.final_mass_kg
    assert np.array_equal(a.throttle, b.throttle)

    # The seed is still a real knob: changing it is allowed to land elsewhere. (Not asserted as a
    # strict inequality; an easy problem may reach the same answer from either start.)
    other = sf.solve(target, seed=seed, rng_seed=1337, **kw)
    assert other.dv_kms == pytest.approx(other.dv_kms)


def test_leg_nodes_are_the_shape_every_consumer_reads():
    """The node array replaces what PyKEP 2.x handed back as ``get_traj``, and the whole
    plotting/report layer indexes it positionally, so its layout is pinned here: half-segment
    node spacing, each segment's throttle on BOTH of its nodes with the last repeated at the
    close, and the impulse taken at the midpoint (so a midpoint carries the mass on the way in
    to its own impulse, and the boundary after it carries the drawn-down mass)."""
    target = lb.target_planet(name="2008 EV5", **EV5)
    nseg = 6
    udp = sf._make_udp(target, mass_kg=MASS, thrust_N=THRUST, isp_s=ISP, nseg=nseg,
                       launch_window=LAUNCH, arrive_by=ARRIVE_BY,
                       vinf_dep_kms=1.5, vinf_arr_kms=0.1, window_slack_days=0.0,
                       min_tof_days=120.0, max_tof_days=None, max_duty_cycle=1.0,
                       max_dep_decl_deg=None)
    lo, hi = (np.asarray(b, float) for b in udp.get_bounds())
    x = (lo + hi) / 2.0
    rng = np.random.default_rng(5)
    for i in range(nseg):                       # a distinct, sub-unit throttle per segment
        j = sf._I_THROTTLE0 + 3 * i
        x[j:j + 3] = rng.uniform(-0.4, 0.4, 3)

    nodes = sf._leg_nodes(udp, x)
    assert nodes.shape == (2 * nseg + 1, 12)

    # Times: evenly spaced half-segments spanning exactly the flight time.
    t0, tof = x[sf._I_T0], x[udp._i_tof]
    np.testing.assert_allclose(nodes[:, 0], t0 + np.arange(2 * nseg + 1) * tof / (2 * nseg))

    # Throttles: a staircase, one step per segment, both nodes of a segment equal.
    for i in range(nseg):
        u = x[sf._I_THROTTLE0 + 3 * i: sf._I_THROTTLE0 + 3 * i + 3]
        np.testing.assert_allclose(nodes[2 * i, 9:12], u)
        np.testing.assert_allclose(nodes[2 * i + 1, 9:12], u)
    np.testing.assert_allclose(nodes[-1, 9:12], x[sf._I_THROTTLE0 + 3 * (nseg - 1):][:3])
    np.testing.assert_allclose(nodes[:, 8], np.linalg.norm(nodes[:, 9:12], axis=1))

    # Masses: the leg starts at the start mass and ends at the decision vector's final mass,
    # and every impulse costs propellant, so mass falls forward and rises backward.
    assert nodes[0, 7] == pytest.approx(MASS)
    assert nodes[-1, 7] == pytest.approx(x[sf._I_MF])
    fwd = nodes[:2 * (nseg // 2), 7]
    assert np.all(np.diff(fwd) <= 1e-9)          # forward half: only ever getting lighter
    # A midpoint shares its segment-start mass (the impulse has not been applied yet).
    for i in range(nseg // 2):
        assert nodes[2 * i + 1, 7] == pytest.approx(nodes[2 * i, 7])

    # The departure state is Earth's, offset by the decision vector's departure v-infinity.
    r_earth, v_earth = lb.earth_planet().eph(float(t0))
    np.testing.assert_allclose(nodes[0, 1:4], r_earth, rtol=1e-12)
    np.testing.assert_allclose(nodes[0, 4:7], np.asarray(v_earth) + x[sf._I_VINF_DEP],
                               rtol=1e-12)


def test_rebuild_rejects_a_vector_from_a_different_variable_order():
    """A stored decision vector is flown on trust, so one written against another layout
    must fail loudly. Reordered, it puts a flight time where a mass belongs and a mass where
    a velocity component belongs -- values those slots cannot hold."""
    target = lb.target_planet(name="2008 EV5", **EV5)
    common = dict(mass_kg=MASS, thrust_N=THRUST, isp_s=ISP, nseg=10,
                  launch_window=LAUNCH, arrive_by=ARRIVE_BY,
                  vinf_dep_kms=1.5, vinf_arr_kms=0.1)
    udp = sf._make_udp(target, window_slack_days=0.0, min_tof_days=120.0,
                       max_tof_days=None, max_duty_cycle=1.0, max_dep_decl_deg=None,
                       **common)
    lo, hi = (np.asarray(b, float) for b in udp.get_bounds())
    good = (lo + hi) / 2.0
    # A well-formed vector still rebuilds.
    sol = sf.rebuild_solution(target, x=good, **common)
    assert sol.trajectory.shape == (2 * 10 + 1, 12)

    # The old order was [t0, tof, mf, vinf_dep, vinf_arr, throttles].
    stale = np.concatenate([[good[sf._I_T0]], [good[udp._i_tof]], [good[sf._I_MF]],
                            good[2:8], good[sf._I_THROTTLE0:udp._i_tof]])
    with pytest.raises(ValueError, match="variable order"):
        sf.rebuild_solution(target, x=stale, **common)

    with pytest.raises(ValueError, match="different problem"):
        sf.rebuild_solution(target, x=good[:-1], **common)


def test_a_pinned_cell_solves_where_it_was_asked_and_warm_starts_faster():
    """A grid cell is a solve with the departure epoch and flight time PINNED rather than searched.

    Pinning is what makes a grid affordable: the free solve's hardest multimodality is in those two
    variables, and fixing them leaves only the control history to find. It is also what makes the
    answer a cell rather than "the best nearby", which is what a grid axis has to mean.

    The warm start is the other half. A converged neighbour's throttle history transfers because a
    throttle is a fraction of its own segment, so only the epoch and duration are retargeted.
    """
    rc = config.default_resolved()
    target = lb.target_planet(name="2008 EV5", **EV5)
    dep0 = lb.mjd2000_from_date(rc.departure_window[0])

    cold = sf.solve_cell_for_config(rc, target, dep_mjd2000=dep0 + 10.0, tof_days=380.0,
                                    nseg=10, restarts=1)
    # Solved where it was asked, inside the pin's own tolerance.
    assert abs(cold.dep_mjd2000 - (dep0 + 10.0)) <= 0.8
    assert abs(cold.tof_days - 380.0) <= 0.8

    warm = sf.solve_cell_for_config(rc, target, dep_mjd2000=dep0 + 20.0, tof_days=380.0,
                                    nseg=10, restarts=1, x0=cold.decision_vector)
    assert abs(warm.dep_mjd2000 - (dep0 + 20.0)) <= 0.8
    assert abs(warm.tof_days - 380.0) <= 0.8
    # The warm start is a legal point for the new cell: same length, inside the new bounds.
    assert warm.decision_vector.shape == cold.decision_vector.shape

    # Seeded, so a re-run reproduces, a grid whose cells wander is not a grid.
    again = sf.solve_cell_for_config(rc, target, dep_mjd2000=dep0 + 20.0, tof_days=380.0,
                                     nseg=10, restarts=1, x0=cold.decision_vector)
    assert again.dv_kms == pytest.approx(warm.dv_kms, rel=1e-9)


def test_warm_starting_across_segment_counts_is_refused():
    """Resampling a control history between segment counts is the one thing a warm start must not
    do. The resampled point satisfies the coarse discretization's constraints but not the finer
    one's, and the optimizer settles there and reports a plausible ΔV with a failed matchpoint --
    a trajectory that looks real and is not.

    Measured on a 2008 EV5 grid: seeding a 12-segment refine from an 8-segment solve regressed
    against a cold multi-start in 15 of 15 trials, worst case +0.62 km/s (+15.5%). Seeding it from
    a 12-segment solve regressed in none. So the mismatch raises instead of interpolating.
    """
    rc = config.default_resolved()
    target = lb.target_planet(name="2008 EV5", **EV5)
    dep0 = lb.mjd2000_from_date(rc.departure_window[0])
    coarse = sf.solve_cell_for_config(rc, target, dep_mjd2000=dep0 + 10.0, tof_days=380.0,
                                      nseg=6, restarts=1)
    with pytest.raises(ValueError, match="segment count|warm-start"):
        sf.solve_cell_for_config(rc, target, dep_mjd2000=dep0 + 10.0, tof_days=380.0,
                                 nseg=12, restarts=1, x0=coarse.decision_vector)


def test_the_sun_distance_ceiling_gradient_matches_finite_differences():
    """The ceiling's constraint rows depend on where each segment's midpoint is, which depends on
    the epoch, the excess velocities, the flight time and every earlier throttle through the
    propagation chain. Their gradient is accumulated analytically along that chain; here it is
    checked column by column against central differences of the constraint values, and the
    sparsity pattern is checked to name every column the values actually move with."""
    import pygmo as pg

    from prospector.solvers.sunpower import SmoothCap

    target = lb.target_planet(name="2008 EV5", **EV5)
    cap = SmoothCap.from_function(lambda r: 1.0 - 0.6 * np.clip(np.asarray(r, float) - 0.95, 0.0, None))
    nseg = 8
    udp = sf._make_udp(target, mass_kg=MASS, thrust_N=THRUST, isp_s=ISP, nseg=nseg,
                       launch_window=LAUNCH, arrive_by=ARRIVE_BY, vinf_dep_kms=1.5,
                       vinf_arr_kms=0.1, window_slack_days=0.0, min_tof_days=120.0,
                       max_tof_days=None, max_duty_cycle=0.9, max_dep_decl_deg=28.5, cap_fn=cap)
    prob = pg.problem(udp)
    rng = np.random.default_rng(3)
    lo, hi = (np.asarray(b, float) for b in udp.get_bounds())
    x = lo + rng.random(lo.size) * (hi - lo)
    x[sf._I_MF] = 0.85 * MASS
    x[sf._I_THROTTLE0:sf._I_THROTTLE0 + 3 * nseg] *= 0.5
    x[udp._i_tof] = 400.0
    row0 = 1 + udp.base.get_nec() + udp.base.get_nic() + 1          # after PyKEP's rows + deadline
    sp = np.asarray(prob.gradient_sparsity(), int)
    g = np.asarray(prob.gradient(x), float)
    assert sp.shape[0] == g.size
    J = np.zeros((nseg, x.size))
    declared = np.zeros_like(J, dtype=bool)
    for (r, c), v in zip(sp, g):
        if row0 <= r < row0 + nseg:
            J[r - row0, c] = v
            declared[r - row0, c] = True
    Jfd = np.zeros_like(J)
    for c in range(x.size):
        h = 1e-5 if c in (sf._I_T0, udp._i_tof) else 1e-6 * max(1.0, abs(x[c]))
        xp, xm = x.copy(), x.copy()
        xp[c] += h
        xm[c] -= h
        fp = np.asarray(prob.fitness(xp), float)[row0:row0 + nseg]
        fm = np.asarray(prob.fitness(xm), float)[row0:row0 + nseg]
        Jfd[:, c] = (fp - fm) / (2 * h)
    np.testing.assert_allclose(J, Jfd, rtol=1e-4, atol=1e-7)
    assert np.abs(Jfd[~declared]).max() < 1e-9, "a column the sparsity omits moves the constraint"
    # And the rows are what they claim: |u_i|^2 - capbar_i^2, capbar the mean ceiling over the
    # six chord samples between the chain's own nodes, the same points the exact ceiling uses.
    from prospector.solvers.sunpower import CHORD_WEIGHTS
    P, _J = udp._segment_nodes(x)
    pts = np.einsum("sk,ikj->isj", CHORD_WEIGHTS, P)
    caps = cap(np.linalg.norm(pts, axis=2) / 1.495978707e11).mean(axis=1)
    u = x[sf._I_THROTTLE0:sf._I_THROTTLE0 + 3 * nseg].reshape(nseg, 3)
    f = np.asarray(prob.fitness(x), float)[row0:row0 + nseg]
    np.testing.assert_allclose(f, (u * u).sum(axis=1) - caps ** 2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(udp.segment_caps(x), caps)
    # The nodes are the ones the trajectory reconstruction reports: midpoints everywhere, and the
    # boundaries except at the cut, where each half carries its own end.
    nodes = sf._leg_nodes(udp, x)
    nf = udp._nseg_fwd
    np.testing.assert_allclose(P[:, 1], nodes[1::2, 1:4], rtol=1e-9, atol=1e-3)
    np.testing.assert_allclose(P[:nf, 0], nodes[0:2 * nf:2, 1:4], rtol=1e-9, atol=1e-3)
    np.testing.assert_allclose(P[nf:, 2], nodes[2 * nf + 2::2, 1:4], rtol=1e-9, atol=1e-3)
    np.testing.assert_allclose(P[nf, 0], nodes[2 * nf, 1:4], rtol=1e-9, atol=1e-3)  # the cut


def test_a_flat_ceiling_takes_the_light_path_and_the_same_positions():
    """Where the ceiling has no slope at any sample (a power-rich leg), its gradient is the throttle
    term alone and the chain need only supply positions. The light path must give the positions the
    full chain gives, its rows must equal |u|^2 - cap^2, and the problem must remember which path
    it is on: flat stays light, a sloped sample switches back to the full chain."""
    import pygmo as pg

    from prospector.solvers.sunpower import SmoothCap

    target = lb.target_planet(name="2008 EV5", **EV5)
    nseg = 8
    kw = dict(mass_kg=MASS, thrust_N=THRUST, isp_s=ISP, nseg=nseg, launch_window=LAUNCH,
              arrive_by=ARRIVE_BY, vinf_dep_kms=1.5, vinf_arr_kms=0.1, window_slack_days=0.0,
              min_tof_days=120.0, max_tof_days=None, max_duty_cycle=1.0, max_dep_decl_deg=None)
    flat = sf._make_udp(target, cap_fn=SmoothCap.constant(0.8), **kw)
    rng = np.random.default_rng(5)
    lo, hi = (np.asarray(b, float) for b in flat.get_bounds())
    x = lo + rng.random(lo.size) * (hi - lo)
    x[sf._I_MF] = 0.85 * MASS
    x[sf._I_THROTTLE0:sf._I_THROTTLE0 + 3 * nseg] *= 0.5
    x[flat._i_tof] = 400.0
    P_light, J_light = flat._segment_nodes(x, jacobian=False)
    P_full, J_full = flat._segment_nodes(x, jacobian=True)
    assert J_light is None and J_full is not None
    np.testing.assert_allclose(P_light, P_full, rtol=1e-12, atol=1e-3)
    prob = pg.problem(flat)
    row0 = 1 + flat.base.get_nec() + flat.base.get_nic() + 1
    u = x[sf._I_THROTTLE0:sf._I_THROTTLE0 + 3 * nseg].reshape(nseg, 3)
    f = np.asarray(prob.fitness(x), float)[row0:row0 + nseg]
    np.testing.assert_allclose(f, (u * u).sum(axis=1) - 0.64, rtol=1e-12, atol=1e-12)
    assert prob.extract(sf._LowThrustUDP)._flat is True   # flat everywhere: on the light path now
    sp = np.asarray(prob.gradient_sparsity(), int)
    g = np.asarray(prob.gradient(x), float)
    J = np.zeros((nseg, x.size))
    for (r, c), v in zip(sp, g):
        if row0 <= r < row0 + nseg:
            J[r - row0, c] = v
    expect = np.zeros_like(J)
    for i in range(nseg):
        expect[i, sf._I_THROTTLE0 + 3 * i:sf._I_THROTTLE0 + 3 * i + 3] = 2.0 * u[i]
    np.testing.assert_allclose(J, expect, atol=1e-12)
    # A sloped ceiling on the same problem leaves the light path at its first evaluation.
    sloped = sf._make_udp(target, cap_fn=SmoothCap.from_function(
        lambda r: 1.0 - 0.6 * np.clip(np.asarray(r, float) - 0.9, 0.0, None)), **kw)
    sloped._flat = True
    sloped._cap_fn_rows(x)
    assert sloped._flat is False


def test_exact_departure_speed_constraint_in_the_udp():
    """A launcher delivers its departure speed exactly, so the speed bound gains its lower side:
    one more inequality, violated below the speed, met at it, with a gradient that matches the
    declared sparsity and central differences."""
    target = lb.target_planet(name="2008 EV5", **EV5)
    common = dict(mass_kg=MASS, thrust_N=THRUST, isp_s=ISP, nseg=10,
                  launch_window=LAUNCH, arrive_by=ARRIVE_BY,
                  vinf_dep_kms=1.5, vinf_arr_kms=0.1, window_slack_days=0.0,
                  min_tof_days=120.0, max_tof_days=None, max_duty_cycle=1.0,
                  max_dep_decl_deg=None)
    exact = sf._make_udp(target, vinf_dep_exact=True, **common)
    ceiling = sf._make_udp(target, **common)
    assert exact.get_nic() == ceiling.get_nic() + 1
    lo, _ = exact.get_bounds()
    x = np.zeros(len(lo))
    x[sf._I_T0], x[exact._i_tof], x[sf._I_MF] = 8030.0, 300.0, MASS * 0.9
    x[sf._I_VINF_DEP] = [1000.0, 0.0, 0.0]                      # 1.0 of 1.5 km/s: too slow
    assert exact.fitness(x)[-1] > 0.0
    x[sf._I_VINF_DEP] = [0.0, 1500.0, 0.0]                      # exactly the speed
    assert abs(exact.fitness(x)[-1]) < 1e-12
    g = np.asarray(exact.gradient(x))
    sp = exact.gradient_sparsity()
    assert len(g) == len(sp)
    last = [i for i, (r, _) in enumerate(sp) if r == 1 + exact.get_nec() + exact.get_nic() - 1]
    for i in last:
        _, col = sp[i]
        h = 0.5
        xp, xm = x.copy(), x.copy()
        xp[col] += h
        xm[col] -= h
        fd = (exact.fitness(xp)[-1] - exact.fitness(xm)[-1]) / (2 * h)
        assert g[i] == pytest.approx(fd, rel=1e-6, abs=1e-12)


def test_the_coast_seed_carries_the_transfer_arrival_speed():
    """For a flyby or impactor arrival the seed keeps the two-burn transfer's arrival speed, so a
    ballistic route starts matched; for a rendezvous the same seed clips to the small bound."""
    target = lb.target_planet(name="2008 EV5", **EV5)
    t0 = lb.mjd2000_from_date(LAUNCH[0]) + 10.0
    seed = lb.lambert_transfer(lb.earth_planet(), target, t0, t0 + 300.0, max_revs=0)
    common = dict(mass_kg=MASS, thrust_N=THRUST, isp_s=ISP, nseg=10, launch_window=LAUNCH,
                  arrive_by=ARRIVE_BY, vinf_dep_kms=5.0, window_slack_days=0.0,
                  min_tof_days=120.0, max_tof_days=None, max_duty_cycle=1.0,
                  max_dep_decl_deg=None)
    wide = sf._make_udp(target, vinf_arr_kms=8.0, **common)
    z = sf._seed_decision_vector(seed, wide, MASS, 10)
    assert np.linalg.norm(z[sf._I_VINF_ARR]) / 1e3 == pytest.approx(seed.vinf_arr_kms, rel=1e-6)
    tight = sf._make_udp(target, vinf_arr_kms=0.1, **common)
    z = sf._seed_decision_vector(seed, tight, MASS, 10)
    assert np.all(np.abs(z[sf._I_VINF_ARR]) <= 100.0 + 1e-9)
