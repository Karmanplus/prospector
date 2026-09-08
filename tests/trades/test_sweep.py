"""Tests for the departure-v-infinity sweep (prospector.trades.sweep) and its job channel.

Pricing the escape, reading off the curve, and extending it past its end with the estimate's
marginal cost, and rebuilding the physics at each point, the cruise's starting mass, and how far
its departure window shifts, are pinned directly. The solve itself is stood in for at the module's
one solver seam (``sweep._sf_solve``), so no optimization ever runs here. The solver's own
behaviour is pinned in test_simsflanagan.
"""
import json
import math
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from prospector import jobs, paths
from prospector.config import EngineMount, Mission, ResolvedConfig, Screening, Vehicle
from prospector.launch import LaunchOrbit, escape_dv_estimate
from prospector.spacecraft.propulsion import Engine
from prospector.trades import sweep

WINDOW = (date(2028, 1, 1), date(2028, 3, 31))


def _rc(launch: LaunchOrbit | None = None) -> ResolvedConfig:
    """A small, fully-derived config: a 650 kg stack spiraling out of LEO."""
    launch = launch or LaunchOrbit(name="leo", perigee_alt_km=500, apogee_alt_km=500,
                                   inclination_deg=28.5)
    cat = {"E": Engine(name="E", isp_s=2000, thrust_mN=500, power_W=1000)}
    vehicle = Vehicle(name="v", dry_mass=300, fuel_mass=350, unusable_prop=10,
                      engines=[EngineMount(type="E")])
    mission = Mission(launch_orbit=launch.name, launch_window=WINDOW,
                      arrive_by=date(2030, 6, 1))
    return ResolvedConfig.build(mission, vehicle, Screening(), cat,
                                launches={launch.name: launch})


def _curve(orbit: LaunchOrbit) -> dict:
    """A spiral tradeoff curve anchored at the orbit's analytic bare-escape cost."""
    vc = orbit.v_circ_kms
    return {"curve_vinf_kms": [0.0, 0.5, 1.0],
            "curve_dv_kms": [vc, vc + 0.14, vc + 0.34],
            "curve_tof_days": [200.0, 212.0, 230.0]}


_TARGET = {"pdes": "341843", "full_name": "341843 (2008 EV5)", "a": 0.95982, "e": 0.08283,
           "i": 7.4478, "om": 93.1897, "w": 235.9248, "ma": 122.8655, "epoch": 2461000.5}


def _fake_solution(rc_point, *, dep_mjd2000=10250.0, tof_days=400.0,
                   vinf_dep_ms=500.0) -> SimpleNamespace:
    """A deterministic stand-in for a converged SimsFlanaganSolution.

    The decision vector is laid out through the solver's own index names, because the sweep reads
    the departure v-infinity straight out of it, a stand-in built to a stale layout would hand the
    sweep a plausible speed assembled from the wrong components.
    """
    from prospector.solvers.simsflanagan import _I_MF, _I_T0, _I_VINF_DEP

    m0 = rc_point.cruise_start_mass_kg
    mf = m0 * 0.9
    nseg = 2
    z = np.zeros(9 + 3 * nseg)
    z[_I_T0], z[_I_MF], z[-1] = dep_mjd2000, mf, tof_days
    z[_I_VINF_DEP] = [vinf_dep_ms, 0.0, 0.0]
    return SimpleNamespace(
        feasible=True, mismatch=1e-6, dep_mjd2000=dep_mjd2000, tof_days=tof_days,
        dv_kms=2000.0 * 9.80665e-3 * math.log(m0 / mf),
        initial_mass_kg=m0, final_mass_kg=mf, propellant_kg=m0 - mf, decision_vector=z)


# ---------------------------------------------------------------------------
# escape pricing
# ---------------------------------------------------------------------------

def test_escape_cost_interpolates_inside_the_curve():
    orbit = _rc().launch
    curve = _curve(orbit)
    dv, tof = sweep.escape_cost(curve, 0.5, orbit)
    assert dv == pytest.approx(curve["curve_dv_kms"][1])
    assert tof == pytest.approx(212.0)
    dv, tof = sweep.escape_cost(curve, 0.25, orbit)        # between samples
    assert dv == pytest.approx(orbit.v_circ_kms + 0.07)
    assert tof == pytest.approx(206.0)


def test_escape_cost_extends_past_the_curve_with_marginal_estimate():
    # Beyond the curve's last v-infinity the cost is the curve end plus the ANALYTIC marginal cost,
    # continuous at the joint, never a raw slope extrapolation.
    orbit = _rc().launch
    curve = _curve(orbit)
    marginal = escape_dv_estimate(orbit, 2.0) - escape_dv_estimate(orbit, 1.0)
    dv, tof = sweep.escape_cost(curve, 2.0, orbit)
    assert dv == pytest.approx(curve["curve_dv_kms"][-1] + marginal)
    # Time extends at the curve end's time-per-dv rate: (230-212)/(0.34-0.14) = 90 d per km/s.
    assert tof == pytest.approx(230.0 + marginal * 90.0)
    # Continuity: just past the end is just past the end's numbers.
    dv_end, _ = sweep.escape_cost(curve, 1.0, orbit)
    dv_eps, _ = sweep.escape_cost(curve, 1.0 + 1e-9, orbit)
    assert dv_eps == pytest.approx(dv_end, abs=1e-6)


def test_escape_cost_is_vectorized():
    orbit = _rc().launch
    dv, tof = sweep.escape_cost(_curve(orbit), np.array([0.25, 2.0]), orbit)
    assert dv.shape == tof.shape == (2,)
    d0, t0 = sweep.escape_cost(_curve(orbit), 0.25, orbit)
    assert dv[0] == pytest.approx(d0) and tof[0] == pytest.approx(t0)


def test_escape_cost_without_a_curve_falls_back_to_the_estimate():
    orbit = _rc().launch
    dv, tof = sweep.escape_cost({}, 1.5, orbit)
    assert dv == pytest.approx(escape_dv_estimate(orbit, 1.5))
    assert math.isnan(tof)                                  # no propagated duration to offer


def test_escape_cost_is_free_when_the_launch_vehicle_escapes():
    tli = LaunchOrbit(name="TLI", escape_provided=True)
    leo = LaunchOrbit(name="leo", perigee_alt_km=185, apogee_alt_km=185, inclination_deg=28.5)
    dv, tof = sweep.escape_cost(_curve(leo), 2.0, tli)    # the curve is there to be ignored
    assert dv == 0.0 and tof == 0.0


# ---------------------------------------------------------------------------
# the sweep itself (solver stubbed)
# ---------------------------------------------------------------------------

def test_run_sweep_rebuilds_mass_and_window_per_point(monkeypatch):
    rc = _rc()
    curve = _curve(rc.launch)
    seen = []

    def fake_solve(rc_point, target_row, **kwargs):
        seen.append({"mass": rc_point.cruise_start_mass_kg,
                     "window": rc_point.departure_window,
                     "cap": kwargs["sf_options"]["vinf_dep_kms"]})
        return _fake_result(rc_point)

    monkeypatch.setattr(sweep, "_evaluate_candidate", fake_solve)
    points = sweep.run_sweep(rc.model_dump(mode="json"), _TARGET,
                             vinf_values=[0.0, 1.0], curve=curve)
    assert len(points) == len(seen) == 2

    veff = 2000.0 * 9.80665e-3
    wet = rc.vehicle.wet_mass
    for point, call, vinf in zip(points, seen, [0.0, 1.0]):
        # Tsiolkovsky at the full wet mass: the solve starts post-spiral, lighter.
        escape_dv = point["escape_dv_kms"]
        assert escape_dv == pytest.approx(sweep.escape_cost(curve, vinf, rc.launch)[0])
        assert call["mass"] == pytest.approx(wet * math.exp(-escape_dv / veff))
        assert point["cruise_start_mass_kg"] == pytest.approx(call["mass"])
        # The departure window shifts by this point's escape duration.
        shift = timedelta(days=round(point["escape_tof_days"]))
        assert call["window"] == (WINDOW[0] + shift, WINDOW[1] + shift)
    # The two points must not share a window: a faster departure spirals longer.
    assert seen[0]["window"] != seen[1]["window"]
    assert points[0]["escape_tof_days"] == pytest.approx(200.0)
    assert points[1]["escape_tof_days"] == pytest.approx(230.0)

    # The v-infinity = 0 point runs the solver at the tiny positive floor (the transcription scales
    # by the cap), priced at the true zero on the escape side.
    assert 0.0 < seen[0]["cap"] <= sweep.VINF_CAP_FLOOR_KMS
    assert seen[1]["cap"] == 1.0

    # Totals and the ledger of what the optimizer actually spent.
    p = points[1]
    assert p["feasible"] is True
    assert p["total_dv_kms"] == pytest.approx(p["escape_dv_kms"] + p["cruise_dv_kms"])
    assert p["propellant_kg"] == pytest.approx(
        p["escape_propellant_kg"] + p["cruise_propellant_kg"])
    assert p["vinf_dep_used_kms"] == pytest.approx(0.5)     # 500 m/s in the decision vector
    assert p["dep_mjd2000"] == 10250.0 and p["arr_mjd2000"] == 10650.0
    assert p["dep_date"] == "2028-01-24" and p["arr_date"] == "2029-02-27"
    assert json.dumps(points)                               # every point is JSON-safe

    # Each point records its own schedule. This is what makes a sweep's departure dates
    # readable: they differ point to point because each point's escape takes a different length of
    # time, so each point's window sits somewhere else. Without the schedule a reader cannot tell a
    # shifted window from a mission that loiters.
    for point in points:
        assert point["coast_days"] is not None and point["coast_days"] >= 0.0
        assert point["liftoff_date"] is not None
        assert point["liftoff_in_window"] in (True, False)
        # The seam closes: liftoff + escape + wait lands exactly on the cruise departure.
        liftoff = date.fromisoformat(point["liftoff_date"])
        escape = timedelta(days=round(point["escape_tof_days"]))
        assert liftoff + escape == date.fromisoformat(point["dep_date"])


def _rc_return() -> ResolvedConfig:
    """The small config of :func:`_rc`, but flying a round trip home to EML2."""
    launch = LaunchOrbit(name="leo", perigee_alt_km=500, apogee_alt_km=500, inclination_deg=28.5)
    cat = {"E": Engine(name="E", isp_s=2000, thrust_mN=500, power_W=1000)}
    vehicle = Vehicle(name="v", dry_mass=300, fuel_mass=350, unusable_prop=10,
                      engines=[EngineMount(type="E")])
    mission = Mission(launch_orbit=launch.name, launch_window=WINDOW, arrive_by=date(2030, 6, 1),
                      return_trip=True, return_by=date(2032, 1, 1), return_destination="EML2",
                      time_at_asteroid=60.0, asteroid_payload_mass=200.0)
    return ResolvedConfig.build(mission, vehicle, Screening(), cat,
                                launches={launch.name: launch})


def _fake_return(rc_point, target_row, outbound_sol, return_opts):
    """A deterministic laden-return stand-in + the EML2 destination object."""
    from prospector.launch import load_return_destinations
    dest = load_return_destinations()["EML2"]
    start = rc_point.return_start_mass_kg(outbound_sol.final_mass_kg)
    mf = start * 0.85
    z = np.zeros(15)
    z[2] = mf
    sol = SimpleNamespace(feasible=True, mismatch=1e-6, dep_mjd2000=11000.0, tof_days=500.0,
                          dv_kms=1.2, initial_mass_kg=start, final_mass_kg=mf,
                          propellant_kg=start - mf, decision_vector=z)
    return sol, dest


def _block(sol) -> dict:
    """The ``sf`` block the pipeline assembles for a solution, light mode (numbers only)."""
    return {"feasible": sol.feasible, "mismatch": sol.mismatch, "dep_mjd2000": sol.dep_mjd2000,
            "tof_days": sol.tof_days, "dv_kms": sol.dv_kms,
            "initial_mass_kg": sol.initial_mass_kg, "final_mass_kg": sol.final_mass_kg,
            "propellant_kg": sol.propellant_kg, "decision_vector": sol.decision_vector,
            "seg_caps": None, "seg_isp_s": None, "thrust_N": 1.0, "isp_s": 2000.0,
            "refresh_settled": True}


def _fake_result(rc_point, *, return_leg: bool = False, **kw) -> dict:
    """What ``pipeline.solve.evaluate_candidate`` hands back for one candidate: the outbound block
    and, when asked, the laden return's block and account, built from the same stand-ins."""
    sol = _fake_solution(rc_point, **kw)
    result = {"sf": _block(sol), "target": {}, "orbits": {}}
    if return_leg:
        ret_sol, dest = _fake_return(rc_point, None, sol, None)
        insertion = rc_point.insertion_propellant_kg(ret_sol.final_mass_kg, dest)
        total = float(ret_sol.propellant_kg) + insertion
        available = rc_point.return_available_propellant_kg(sol.propellant_kg)
        result["return"] = {
            "sf": _block(ret_sol), "insertion_propellant_kg": insertion,
            "total_return_propellant_kg": total, "available_propellant_kg": available,
            "feasible_propellant": bool(total <= available),
            "delivered_mass_kg": float(ret_sol.final_mass_kg) - insertion,
            "decision_vector": ret_sol.decision_vector,
        }
    return result


def test_run_sweep_adds_the_return_leg(monkeypatch):
    rc = _rc_return()
    curve = _curve(rc.launch)
    monkeypatch.setattr(sweep, "_evaluate_candidate",
                        lambda rc_point, target_row, **kw: _fake_result(rc_point, return_leg=True))
    (p,) = sweep.run_sweep(rc.model_dump(mode="json"), _TARGET, vinf_values=[1.0], curve=curve,
                           options={"return_options": {"nseg": 8}})
    assert p["return_dv_kms"] == pytest.approx(1.2)
    assert p["return_tof_days"] == pytest.approx(500.0)
    assert p["return_cruise_propellant_kg"] is not None and p["insertion_prop_kg"] is not None
    assert p["return_propellant_kg"] == pytest.approx(
        p["return_cruise_propellant_kg"] + p["insertion_prop_kg"])
    # The mission propellant now rolls up escape + outbound cruise + return (cruise + insertion).
    assert p["propellant_kg"] == pytest.approx(
        p["escape_propellant_kg"] + p["cruise_propellant_kg"] + p["return_propellant_kg"])
    assert p["return_decision_vector"] is not None
    assert p["round_trip_days"] == pytest.approx(p["tof_days"] + 60.0 + 500.0)
    # total_dv stays the escape/cruise split the sweep optimizes (return dV is separate).
    assert p["total_dv_kms"] == pytest.approx(p["escape_dv_kms"] + p["cruise_dv_kms"])
    assert json.dumps(p)                                   # still JSON-safe


def test_one_way_sweep_leaves_return_fields_none(monkeypatch):
    rc = _rc()                                             # return_trip is False
    monkeypatch.setattr(sweep, "_evaluate_candidate",
                        lambda rc_point, target_row, **kw: _fake_result(rc_point))
    (p,) = sweep.run_sweep(rc.model_dump(mode="json"), _TARGET, vinf_values=[1.0],
                           curve=_curve(rc.launch))
    assert p["return_dv_kms"] is None and p["return_propellant_kg"] is None
    # One-way propellant is escape + cruise only, as before the feature.
    assert p["propellant_kg"] == pytest.approx(
        p["escape_propellant_kg"] + p["cruise_propellant_kg"])


def test_run_sweep_without_a_curve_uses_the_analytic_fallbacks(monkeypatch):
    rc = _rc()
    monkeypatch.setattr(sweep, "_evaluate_candidate",
                        lambda rc_point, target_row, **kw: _fake_result(rc_point))
    (point,) = sweep.run_sweep(rc.model_dump(mode="json"), _TARGET,
                               vinf_values=[1.0], curve=None)
    assert point["escape_dv_kms"] == pytest.approx(escape_dv_estimate(rc.launch, 1.0))
    assert math.isfinite(point["escape_tof_days"]) and point["escape_tof_days"] > 0


def test_run_sweep_records_failed_points_and_continues(monkeypatch):
    # A point whose solve blows up is recorded infeasible with its escape numbers -- the sweep
    # never aborts (a missing point is a silently dropped option).
    rc = _rc()
    curve = _curve(rc.launch)

    def flaky_solve(rc_point, target_row, **kwargs):
        if kwargs["sf_options"]["vinf_dep_kms"] > 0.6:
            raise ValueError("mission window too short")
        return _fake_result(rc_point)

    monkeypatch.setattr(sweep, "_evaluate_candidate", flaky_solve)
    calls = []
    points = sweep.run_sweep(rc.model_dump(mode="json"), _TARGET,
                             vinf_values=[0.0, 1.0, 0.5], curve=curve,
                             progress=lambda i, n, p: calls.append((i, n, p)))
    assert [p["feasible"] for p in points] == [True, False, True]
    bad = points[1]
    assert bad["total_dv_kms"] is None and bad["cruise_dv_kms"] is None
    assert bad["escape_dv_kms"] > 0 and bad["escape_tof_days"] == pytest.approx(230.0)
    assert "mission window too short" in bad["error"]
    # Progress fired after every point, including the failed one.
    assert [(i, n) for i, n, _ in calls] == [(1, 3), (2, 3), (3, 3)]
    assert calls[1][2] is bad


def test_default_vinf_grid_includes_zero_and_the_cap():
    grid = sweep.default_vinf_grid(3.0)
    assert len(grid) == 7
    assert grid[0] == 0.0 and grid[-1] == 3.0
    assert np.allclose(np.diff(grid), 0.5)                  # roughly even: exactly, here
    assert sweep.default_vinf_grid(2.0, n=2) == [0.0, 2.0]
    assert sweep.default_vinf_grid(0.0) == [0.0]            # degenerate cap: just the baseline


# ---------------------------------------------------------------------------
# the job channel (runs/sweep/<id>/), mirroring the spiral channel
# ---------------------------------------------------------------------------

def test_submit_sweep_spawns_and_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    spawned = {}
    monkeypatch.setattr(jobs.subprocess, "Popen",
                        lambda args, **k: spawned.setdefault("args", list(args)))
    options = {"vinf_values": [0.0, 1.0], "curve": {"curve_vinf_kms": [0.0]}, "nseg": 10}
    run_id = jobs.submit_sweep({"mission": {}}, _TARGET, options=options, label="ARM")
    assert spawned["args"][-2:] == [run_id, "sweep"]         # worker.py <id> sweep
    spec = jobs.read_sweep_job(run_id)
    assert spec["options"] == options and spec["label"] == "ARM"
    assert spec["target"]["pdes"] == "341843"
    assert jobs.read_sweep_status(run_id)["state"] == jobs.QUEUED


def test_sweep_status_streams_the_latest_point(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    jobs.sweep_run_dir("s1").mkdir(parents=True)
    point = {"vinf_kms": 0.5, "total_dv_kms": 9.1, "feasible": True}
    jobs.write_sweep_status("s1", state=jobs.RUNNING, stage="sweep", progress=0.5,
                            message="point 1/2", point=point)
    st = jobs.read_sweep_status("s1")
    assert st["state"] == jobs.RUNNING and st["progress"] == 0.5
    assert st["point"] == point
    assert jobs.read_sweep_status("nope")["point"] is None   # placeholder carries the key


# ---------------------------------------------------------------------------
# worker mode (in-process, solver stubbed; the detached spawn is exercised above)
# ---------------------------------------------------------------------------

def test_worker_sweep_writes_artifacts(tmp_path, monkeypatch):
    pytest.importorskip("pykep")            # importing worker pulls in the pipeline stack
    import worker

    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: None)
    monkeypatch.setattr(sweep, "_evaluate_candidate",
                        lambda rc_point, target_row, **kw: _fake_result(rc_point))
    rc = _rc()
    options = {"vinf_values": [0.0, 1.0], "curve": _curve(rc.launch), "nseg": 10}
    run_id = jobs.submit_sweep(rc.model_dump(mode="json"), _TARGET,
                               options=options, label="ARM")

    assert worker.main(run_id, "sweep") == 0
    status = jobs.read_sweep_status(run_id)
    assert status["state"] == jobs.DONE and status["progress"] == 1.0
    summary = jobs.read_sweep_summary(run_id)
    assert summary["n_points"] == 2 and summary["n_feasible"] == 2
    assert len(summary["points"]) == 2
    assert summary["points"][0]["vinf_kms"] == 0.0          # the zero baseline is a point
    # The best point is the feasible total-dV minimum across the sweep.
    totals = {p["vinf_kms"]: p["total_dv_kms"] for p in summary["points"]}
    assert summary["best_total_dv_kms"] == min(totals.values())
    assert totals[summary["best_vinf_kms"]] == summary["best_total_dv_kms"]
    assert jobs.read_sweep_result(run_id)["points"] == summary["points"]


def test_each_flight_time_seeds_inside_the_points_own_window():
    """A sweep point has no click to start from, so it seeds each flight time it tries with the
    cheapest two-burn departure inside its own window. A departure outside that window seeds the
    solve at a date it is not allowed to fly, and the solve then reports a plausible answer for a
    mission nobody asked for."""
    pytest.importorskip("pykep")
    from prospector.solvers import lambert as lb
    from prospector.trades import sweep as sw

    rc = _rc()
    row = {"pdes": "x", "full_name": "x", "a": 0.96, "e": 0.083, "i": 7.4,
           "om": 93.2, "w": 235.9, "ma": 122.9, "epoch": 2461000.5}
    target = lb.planet_from_row(row)
    lo = lb.mjd2000_from_date(rc.departure_window[0])
    hi = lb.mjd2000_from_date(rc.departure_window[1])

    seen = 0
    for tof in (200.0, 400.0, 600.0):
        seed = sw._seed_at(rc, target, row, tof, vinf_cap_kms=1.0)
        if seed is None:
            continue
        seen += 1
        assert lo - 1e-6 <= seed.dep_mjd2000 <= hi + 1e-6
        assert seed.tof_days == pytest.approx(tof, abs=1e-6)
    assert seen, "no flight time produced a seed"


def test_run_sweep_parallel_matches_serial(monkeypatch):
    """workers>1 fans the points across processes: same points, cap order preserved,
    one progress call per completion. Relies on fork inheriting the stubbed solver
    seam, so it only runs where fork is the start method (Linux)."""
    import multiprocessing as mp
    if mp.get_start_method(allow_none=False) != "fork":
        pytest.skip("needs fork start method to inherit the stubbed solver seam")

    rc = _rc()
    curve = _curve(rc.launch)
    monkeypatch.setattr(sweep, "_evaluate_candidate",
                        lambda rc_point, target_row, **kw: _fake_result(rc_point))
    kwargs = dict(vinf_values=[0.0, 0.5, 1.0, 1.5], curve=curve)
    serial = sweep.run_sweep(rc.model_dump(mode="json"), _TARGET, **kwargs)
    calls = []
    parallel = sweep.run_sweep(rc.model_dump(mode="json"), _TARGET, **kwargs,
                               workers=3, progress=lambda i, n, p: calls.append((i, n)))
    assert [p["vinf_kms"] for p in parallel] == [0.0, 0.5, 1.0, 1.5]
    assert calls == [(1, 4), (2, 4), (3, 4), (4, 4)]
    for s, p in zip(serial, parallel):
        assert p == s


def test_worker_records_the_sweep_winner_as_a_run(tmp_path, monkeypatch):
    """The worker rebuilds the winning point's decision vector into a complete result and
    records it through the normal solve channel (DONE, result, summary), so the UI's
    "Apply best v∞" can open a trajectory the sweep already paid for. No winner -> None."""
    import worker
    from prospector import jobs

    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    fake_result = {
        "sf": {"feasible": True, "mismatch": 1e-6, "dep_mjd2000": 10650.0,
               "tof_days": 300.0, "dv_kms": 2.5, "propellant_kg": 40.0},
        "target": {"pdes": "341843", "name": "341843 (2008 EV5)", "a": 0.96, "e": 0.08,
                   "i": 7.4},
        "lambert": {"best_dv_kms": 2.1},
        "orbits": {},
    }
    seen = {}

    def fake_rebuild(rc_point, target, x, **kwargs):
        seen.update({"vinf": rc_point.departure_vinf_kms,
                     "mass": rc_point.cruise_start_mass_kg,
                     "cap": kwargs.get("vinf_dep_kms"), "x": list(x)})
        return fake_result

    monkeypatch.setattr(worker.pipeline, "result_from_decision", fake_rebuild)

    rc = _rc()
    spec = {"label": "study", "config": rc.model_dump(mode="json"), "target": _TARGET}
    winner = {"feasible": True, "total_dv_kms": 9.0, "vinf_kms": 1.0,
              "escape_dv_kms": 7.9, "escape_tof_days": 230.0,
              "escape_propellant_kg": 200.0, "dep_mjd2000": 10650.0,
              "arr_mjd2000": 10950.0, "decision_vector": [1.0, 2.0, 3.0]}
    loser = dict(winner, total_dv_kms=9.5, vinf_kms=0.5)
    failed = {"feasible": False, "vinf_kms": 2.0, "decision_vector": None}

    run_id = worker._record_best_sweep_run(spec, {"nseg": 15}, [loser, winner, failed])
    assert run_id is not None
    assert seen["vinf"] == 1.0 and seen["cap"] == 1.0 and seen["x"] == [1.0, 2.0, 3.0]
    assert seen["mass"] == pytest.approx(rc.vehicle.wet_mass - 200.0)
    assert jobs.read_status(run_id)["state"] == jobs.DONE
    assert jobs.read_result(run_id)["sf"]["dv_kms"] == 2.5
    assert jobs.read_summary(run_id)["dv_kms"] == 2.5
    job = jobs.read_job(run_id)
    assert job["target"]["pdes"] == "341843" and "sweep best" in job["label"]
    assert job["config"]["departure_vinf_kms"] == 1.0   # the run reloads at the winner's knob

    assert worker._record_best_sweep_run(spec, {}, [failed]) is None


def test_worker_grid_writes_the_surface_its_summary_and_a_streaming_status(tmp_path, monkeypatch):
    """The grid worker mode end to end: a job spec in, a converged surface out.

    Detached on purpose. The impulsive porkchop it replaces was instant and ran inline, which is
    why that surface had to be an approximation, and measured against converged solves it moved the
    wrong way against real cost. Paying tens of seconds behind a Run is the trade.
    """
    import worker
    from prospector.config import EngineMount, Mission, ResolvedConfig, Screening, Vehicle
    from prospector.launch import LaunchOrbit
    from prospector.solvers import lambert as lb
    from prospector.spacecraft.propulsion import Engine

    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: None)
    # No Horizons in a test. Patched on the grid module itself: a function resolves names in its
    # own module, so patching the population facade would leave the grid on the real thing.
    from prospector.trades.pipeline import grid as _G
    monkeypatch.setattr(_G, "_mission_elements", lambda rc_, r: dict(r))
    # A high-thrust bus so the cells close fast; the physics has its own tests.
    cat = {"E": Engine(name="E", isp_s=2000, thrust_mN=4000, power_W=5000)}
    launches = {"LV": LaunchOrbit(name="lv", perigee_alt_km=400, apogee_alt_km=400,
                                  inclination_deg=0.0, escape_provided=True)}
    mis = Mission(launch_orbit="LV", launch_window=(date(2028, 1, 1), date(2028, 2, 1)),
                  arrive_by=date(2029, 8, 1))
    veh = Vehicle(name="v", dry_mass=800, fuel_mass=600,
                  engines=[EngineMount(type="E", count=1)], solar_power_W=5000)
    rc = ResolvedConfig.build(mis, veh, Screening(), cat, launches=launches)
    row = {"pdes": "X", "full_name": "X", "a": 1.15, "e": 0.06, "i": 3.0,
           "om": 20.0, "w": 40.0, "ma": 10.0, "epoch": lb.mjd2000_from_date(date(2028, 1, 1)) + 51544.5}

    run_id = jobs.submit_grid(rc.model_dump(mode="json"), row,
                              options={"n_dep": 2, "n_tof": 2, "nseg": 6, "restarts": 1,
                                       "workers": 1, "polish": False})
    assert worker.main(run_id, "grid") == 0

    status = jobs.read_grid_status(run_id)
    assert status["state"] == jobs.DONE
    assert status["cells_total"] == 4 and status["cells_done"] == 4

    surface = jobs.read_grid_result(run_id)
    assert len(surface["dep_mjd2000"]) == 2 and len(surface["tof_days"]) == 2
    assert surface["nseg"] == 6
    # Flight times come back longest first: the order they get solved in.
    assert surface["tof_days"][0] > surface["tof_days"][-1]
    # The frontier and the spread ride with the surface, so the caller needs no second pass.
    assert len(surface["frontier"]) == 2 and len(surface["departure_spread"]) == 2
    # Every cell carries a decision vector or an explicit None; that cube is what makes a later
    # cell click a rebuild rather than a solve.
    assert len(surface["decision_vectors"]) == 2
    assert all(len(col) == 2 for col in surface["decision_vectors"])

    summary = jobs.read_grid_summary(run_id)
    assert summary["n_cells"] == 4 and summary["target_pdes"] == "X"
    assert summary["n_polished"] == 0, "polish was disabled for this run"


def test_worker_grid_reports_a_failure_without_losing_the_job(tmp_path, monkeypatch):
    """A grid that raises must leave a readable ERROR status with its traceback, not a job stuck
    at running -- the UI polls this and has nothing else to go on."""
    import worker

    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: None)
    run_id = jobs.submit_grid({"not": "a config"}, {"pdes": "X", "full_name": "X"}, options={})
    assert worker.main(run_id, "grid") == 1
    status = jobs.read_grid_status(run_id)
    assert status["state"] == jobs.ERROR
    assert status["error"] and "Traceback" in status["error"]
