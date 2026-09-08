"""Tests for the background-job contract (prospector.jobs) and pipeline result shape.

Reading and writing the status, and converting a row, are pinned directly. The full pipeline runs
with a tiny optimizer budget, enough to confirm the result has the shape the views and plots
expect, without paying for a real solve. The physics itself is pinned in test_simsflanagan and
test_lambert.
"""
import json

import numpy as np
import pandas as pd
import pytest

from prospector import jobs, paths
from tests import library as lib

pytest.importorskip("pykep")
pytest.importorskip("pygmo")


def _sf_run(tmp_path, run_id="r", *, done=True):
    """Lay down a minimal finished Sims-Flanagan run (job + status + result) under tmp RUNS_DIR."""
    d = jobs.run_dir(run_id)
    d.mkdir()
    (d / "job.json").write_text(json.dumps(
        {"config": {}, "target": {"spkid": 20341843, "a": 0.96, "full_name": "2008 EV5"}}))
    jobs.write_result(run_id, {"sf": {}, "vehicle": {}, "target": {"name": "2008 EV5"}})
    if done:
        jobs.write_status(run_id, state=jobs.DONE, progress=1.0)
    return d


def test_status_roundtrip_and_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    jobs.run_dir("r1").mkdir()
    jobs.write_status("r1", state=jobs.RUNNING, stage="lambert", progress=0.4, message="x")
    st = jobs.read_status("r1")
    assert st["state"] == jobs.RUNNING and st["progress"] == 0.4 and st["stage"] == "lambert"
    assert not jobs.is_terminal(st)
    jobs.write_status("r1", state=jobs.DONE, progress=1.0)
    assert jobs.is_terminal(jobs.read_status("r1"))


def test_read_status_missing_is_queued(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    jobs.run_dir("nope").mkdir()
    assert jobs.read_status("nope")["state"] == jobs.QUEUED


def test_jsonable_coerces_numpy_and_series():
    row = pd.Series({"pdes": "341843", "a": np.float64(0.96), "i": np.int64(7)})
    out = jobs.jsonable_row(row)
    assert out == {"pdes": "341843", "a": 0.96, "i": 7}
    assert all(not isinstance(v, np.generic) for v in out.values())


def test_pipeline_result_has_expected_shape():
    # Tiny budget: just verify orchestration + result contract, not convergence.
    from prospector.config import load_study, resolve_study
    from prospector.solvers import lambert as lb
    from prospector.trades import pipeline

    rc = resolve_study(load_study(lib.study_key()))
    ev5 = {"pdes": "341843", "full_name": "341843 (2008 EV5)", "a": 0.95982, "e": 0.082830,
           "i": 7.4478, "om": 93.1897, "w": 235.9248, "ma": 122.8655, "epoch": 2461000.5}
    dep = lb.mjd2000_from_date(rc.departure_window[0]) + 1.0
    res = pipeline.solve_from_cell(rc, ev5, dep, dep + 400.0, nseg=8, maxeval=120, restarts=1)

    assert {"target", "lambert", "sf", "orbits", "vehicle"} <= set(res)
    # The lambert block summarises the cell the solve started from, not a grid.
    assert np.isfinite(res["lambert"]["best_dv_kms"])
    assert res["lambert"]["best_dep_mjd2000"] == pytest.approx(dep, abs=1.0)
    sf = res["sf"]
    assert sf["positions_au"].shape == (2 * 8 + 1, 3)
    assert sf["node_times_days"].shape == (2 * 8 + 1,)
    assert sf["fine_positions_au"].shape[1] == 3 and len(sf["fine_positions_au"]) > len(sf["positions_au"])
    assert sf["fine_times_days"][0] == 0 and sf["fine_times_days"][-1] > 0
    assert sf["throttle"].max() <= 1.05      # tiny budget: not converged, just a shape check
    assert res["vehicle"]["thrust_mN"] > 0
    assert res["orbits"]["earth_au"].shape[1] == 3 and res["orbits"]["target_au"].shape[1] == 3


# ---- enrichment background job ----

def test_submit_enrich_spawns_per_job_and_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    spawned = {}
    monkeypatch.setattr(jobs.subprocess, "Popen",
                        lambda args, **k: spawned.setdefault("args", list(args)))
    run_id = jobs.submit_enrich(["341843", "2008 SG344"], options={"n_workers": 2})
    assert isinstance(run_id, str) and run_id
    assert spawned["args"][-2:] == [run_id, "enrich"]          # worker.py <id> enrich
    assert jobs.read_enrich_ids(run_id) == ["341843", "2008 SG344"]
    assert jobs.read_enrich_options(run_id) == {"n_workers": 2}
    st = jobs.read_enrich_status(run_id)
    assert st["state"] == jobs.QUEUED and st["total"] == 2
    assert jobs.enrich_is_current(run_id)


def test_new_enrich_job_supersedes_the_older(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: None)
    first = jobs.submit_enrich(["a", "b"])
    second = jobs.submit_enrich(["c"])              # a new screen: its own job, its own total
    assert first != second
    assert not jobs.enrich_is_current(first)        # the older worker will wind down
    assert jobs.enrich_is_current(second)           # the latest job owns the screen


def test_enrich_status_done_total_dropped_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    jobs.write_enrich_status("j1", state=jobs.RUNNING, done=7, total=20, dropped=2, message="m")
    st = jobs.read_enrich_status("j1")
    assert (st["done"], st["total"], st["dropped"], st["state"]) == (7, 20, 2, jobs.RUNNING)
    assert not jobs.is_terminal(st)


def test_worker_enrich_writes_done_and_reports_progress(tmp_path, monkeypatch):
    import worker
    from prospector import enrichment
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    d = tmp_path / "enrichment" / "j1"
    d.mkdir(parents=True)
    (d / "ids.json").write_text(json.dumps(["1", "2", "3"]))
    (d / "options.json").write_text("{}")
    (tmp_path / "enrichment" / "latest.txt").write_text("j1")
    seen = {}

    def fake_enrich(ids, on_progress=None, n_workers=None, should_continue=None):
        seen["ids"] = ids
        if on_progress:
            on_progress(2, 3, "characterized 2/3")
        return enrichment.normalize(pd.DataFrame({"input_id": ids}))

    monkeypatch.setattr(enrichment, "enrich", fake_enrich)
    assert worker._run_enrich("j1") == 0
    st = jobs.read_enrich_status("j1")
    assert st["state"] == jobs.DONE and st["total"] == 3 and seen["ids"] == ["1", "2", "3"]


def test_worker_enrich_records_unavailable(tmp_path, monkeypatch):
    import worker
    from prospector import enrichment
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    d = tmp_path / "enrichment" / "j1"
    d.mkdir(parents=True)
    (d / "ids.json").write_text(json.dumps(["1"]))
    (d / "options.json").write_text("{}")

    def boom(*a, **k):
        raise enrichment.EnrichmentUnavailable("the enrichment extra is not installed")

    monkeypatch.setattr(enrichment, "enrich", boom)
    assert worker._run_enrich("j1") == 1
    st = jobs.read_enrich_status("j1")
    assert st["state"] == jobs.ERROR and "unavailable" in (st["message"] + st["error"]).lower()


def test_cancel_marker_and_should_continue(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    (tmp_path / "enrichment").mkdir(parents=True)
    (tmp_path / "enrichment" / "latest.txt").write_text("j1")
    assert jobs.enrich_should_continue("j1")               # latest + not cancelled
    jobs.cancel_enrich("j1")
    assert jobs.enrich_cancelled("j1")
    assert not jobs.enrich_should_continue("j1")            # a worker checking this will stop


def test_worker_enrich_reports_cancelled(tmp_path, monkeypatch):
    import worker
    from prospector import enrichment
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    d = tmp_path / "enrichment" / "j1"
    d.mkdir(parents=True)
    (d / "ids.json").write_text(json.dumps(["1", "2"]))
    (d / "options.json").write_text("{}")
    (tmp_path / "enrichment" / "latest.txt").write_text("j1")
    jobs.cancel_enrich("j1")    # cancelled before/while running

    def fake_enrich(ids, on_progress=None, n_workers=None, should_continue=None):
        return enrichment.normalize(pd.DataFrame({"input_id": ids[:1]}))   # got 1 of 2

    monkeypatch.setattr(enrichment, "enrich", fake_enrich)
    assert worker._run_enrich("j1") == 0
    st = jobs.read_enrich_status("j1")
    assert st["state"] == jobs.DONE and "cancel" in st["message"].lower()


# ---------------------------------------------------------------------------
# escape-spiral jobs (the launch phase)
# ---------------------------------------------------------------------------

def test_submit_spiral_spawns_and_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    spawned = {}
    monkeypatch.setattr(jobs.subprocess, "Popen",
                        lambda args, **k: spawned.setdefault("args", list(args)))
    fp = {"launch": "SSO", "thrust_N": 0.27}
    run_id = jobs.submit_spiral({"mission": {}}, options={"target_vinf_kms": 1.0},
                                fingerprint=fp, label="HF")
    assert spawned["args"][-2:] == [run_id, "spiral"]          # worker.py <id> spiral
    spec = jobs.read_spiral_job(run_id)
    assert spec["options"] == {"target_vinf_kms": 1.0} and spec["fingerprint"] == fp
    assert jobs.read_spiral_status(run_id)["state"] == jobs.QUEUED


def _finished_spiral(run_id, fp, *, status="escaped", dv_escape=7.51, monkeypatch=None):
    """Lay down a finished spiral run (job + status + summary) for matching tests."""
    d = jobs.spiral_run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "job.json").write_text(json.dumps({"config": {}, "options": {}, "fingerprint": fp}))
    jobs.write_spiral_summary(run_id, {
        "status": status, "dv_at_escape_kms": dv_escape,
        "dv_kms": (dv_escape + 0.4) if dv_escape is not None else 1.0,
        "tof_days": 220.0, "vinf_kms": 1.0, "target_vinf_kms": 1.0, "propellant_kg": 233.0})
    jobs.write_spiral_status(run_id, state=jobs.DONE, progress=1.0)


def test_worker_spiral_writes_artifacts(tmp_path, monkeypatch):
    # The spiral worker mode runs the real (fast, high-thrust) propagation end to end and leaves
    # the result + summary + DONE status the Launch tab consumes.
    import worker
    from prospector.config import EngineMount, Mission, ResolvedConfig, Screening, Vehicle
    from prospector.launch import LaunchOrbit, escape_fingerprint
    from prospector.spacecraft.propulsion import Engine

    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: None)
    cat = {"E": Engine(name="E", isp_s=1500, thrust_mN=10000, power_W=1000)}
    launches = {"LEO": LaunchOrbit(name="leo", perigee_alt_km=400, apogee_alt_km=400,
                                   inclination_deg=0.0)}
    veh = Vehicle(name="v", dry_mass=100, fuel_mass=400, engines=[EngineMount(type="E")])
    rc = ResolvedConfig.build(Mission(launch_orbit="LEO"), veh, Screening(), cat,
                              launches=launches)
    options = {"target_vinf_kms": 0.0}
    run_id = jobs.submit_spiral(rc.model_dump(mode="json"), options=options,
                                fingerprint=escape_fingerprint(rc, options))

    assert worker.main(run_id, "spiral") == 0
    status = jobs.read_spiral_status(run_id)
    assert status["state"] == jobs.DONE
    summary = jobs.read_spiral_summary(run_id)
    assert summary["status"] == "escaped" and summary["dv_at_escape_kms"] > 0
    assert summary["curve_vinf_kms"][0] == 0.0
    result = jobs.read_spiral_result(run_id)
    assert result["spiral"]["status"] == "escaped"
    assert len(result["spiral"]["positions_km"]) == len(result["spiral"]["times_days"])
    # And the finished run refines the budget for exactly this setup.


def test_record_run_lays_down_the_channel_without_a_worker(tmp_path, monkeypatch):
    """record_run writes the same job/status scaffold submit does, but spawns nothing --
    the sweep's winning point is persisted through it after being solved in-process."""
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    run_id = jobs.record_run({"cfg": 1}, {"pdes": "99942", "full_name": "Apophis",
                                          "a": 0.922}, 10650.0, 10950.0,
                             label="sweep best", options={"nseg": 15})
    assert jobs.read_status(run_id)["state"] == jobs.QUEUED
    job = jobs.read_job(run_id)
    assert job["target"]["pdes"] == "99942"
    assert job["dep_mjd2000"] == 10650.0 and job["options"]["nseg"] == 15
    assert not (jobs.run_dir(run_id) / "worker.log").exists()   # no process was spawned


def _finished_grid(run_id: str, fingerprint=None, *, n_feasible=30, n_cells=36):
    """A grid run written straight to disk, without a worker."""
    d = jobs.grid_run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "job.json").write_text(json.dumps({
        "config": {}, "options": {"n_dep": 6, "n_tof": 6},
        "target": {"pdes": "99942", "full_name": "99942 Apophis"}, "label": ""}))
    jobs.write_grid_summary(run_id, {"target": "99942 Apophis", "target_pdes": "99942",
                                     "n_cells": n_cells, "n_feasible": n_feasible,
                                     "best_dv_kms": 2.93, "best_tof_days": 403.0})
    jobs.write_grid_status(run_id, state=jobs.DONE, progress=1.0,
                           cells_done=n_cells, cells_total=n_cells)


def test_the_grid_channel_streams_its_cell_count(tmp_path, monkeypatch):
    """A grid costs tens of seconds, so its status has to say how much of the surface exists yet --
    a bare progress fraction cannot tell a reader whether to keep waiting."""
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: None)
    run_id = jobs.submit_grid({"x": 1}, {"pdes": "99942", "full_name": "99942 Apophis"},
                              options={"n_dep": 6, "n_tof": 6})
    # Defaults are present before the worker writes anything, so a poll never KeyErrors.
    fresh = jobs.read_grid_status(run_id)
    assert fresh["cells_done"] == 0 and fresh["cells_total"] == 0
    jobs.write_grid_status(run_id, state=jobs.RUNNING, cells_done=12, cells_total=36,
                           progress=0.3, message="12/36 cells solved")
    mid = jobs.read_grid_status(run_id)
    assert (mid["cells_done"], mid["cells_total"]) == (12, 36)
    assert not jobs.is_terminal(mid)
    # The job spec keeps the grid terms, so a reload knows what shape the surface is.
    assert jobs.read_grid_job(run_id)["options"]["n_dep"] == 6

