"""Turning a result into figures: the out-of-date check, the derived numbers, and the propellant.

Building these in two places, once for the app and once for the PDF, lets them drift, and the PDF
ends up embedding a cruise the app has already discarded. There is one place for it now, and the
first test below is what keeps it that way.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from prospector import jobs, paths
from prospector.config import (
    EngineMount,
    Mission,
    ResolvedConfig,
    Screening,
    Vehicle,
    load_mission,
)
from prospector.figures import products
from prospector.spacecraft.propulsion import load_engines
from tests import library as lib


def _config(dry: float = 250.0, prop: float = 325.0, n: int = 4) -> ResolvedConfig:
    return ResolvedConfig.build(
        load_mission(lib.mission_key()),
        Vehicle(name="probe", dry_mass=dry, fuel_mass=prop, unusable_prop=0.0,
                engines=[EngineMount(type=lib.engine_key(), count=n)]),
        Screening(), load_engines())


def _lay_down_cruise(tmp_path, monkeypatch, rc, *, mismatch=1e-6, feasible=True,
                     available_power_W=None) -> str:
    """A finished, converged cruise run stored the way the worker stores one."""
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    run_id = jobs.record_run(rc.model_dump(mode="json"), {"pdes": "99942", "a": 0.92,
                                                          "e": 0.19, "i": 3.3},
                             10000.0, 10300.0,
                             options={"available_power_W": available_power_W})
    jobs.write_result(run_id, {"sf": {"feasible": feasible, "mismatch": mismatch,
                                      "propellant_kg": 120.0}})
    jobs.write_status(run_id, state=jobs.DONE, message="done")
    return run_id


# ---------------------------------------------------------------------------
# the regression: a stale trajectory must not survive into a document
# ---------------------------------------------------------------------------

def test_cruise_block_refuses_a_run_whose_config_has_moved(tmp_path, monkeypatch):
    """Edit the vehicle without re-solving and the stored cruise is no longer current.

    This is the bug this layer exists to close: the exporter used to check only that a run was
    finished and converged, so a PDF could embed a trajectory and a dV ledger the workspace had
    already cleared as stale, on paper, with nothing marking it.
    """
    rc = _config()
    run_id = _lay_down_cruise(tmp_path, monkeypatch, rc)
    assert products.cruise_block(run_id, rc) is not None       # current: usable

    heavier = _config(dry=300.0)                                # a vehicle edit, no re-solve
    assert products.cruise_block(run_id, heavier) is None


def test_signature_is_stable_for_an_identical_config():
    """The gate must not be so strict that a valid run is discarded -- that would be the
    false-negative direction, throwing away minutes of solve the user just paid for."""
    assert products.trajectory_signature(_config(), None) \
        == products.trajectory_signature(_config(), None)


def test_cruise_block_refuses_an_unconverged_run(tmp_path, monkeypatch):
    rc = _config()
    run_id = _lay_down_cruise(tmp_path, monkeypatch, rc, mismatch=1.0, feasible=False)
    assert products.cruise_block(run_id, rc) is None


def test_cruise_block_ignores_a_run_still_in_flight(tmp_path, monkeypatch):
    rc = _config()
    run_id = _lay_down_cruise(tmp_path, monkeypatch, rc)
    jobs.write_status(run_id, state=jobs.RUNNING, message="solving")
    assert products.cruise_block(run_id, rc) is None


def test_cruise_block_without_a_config_skips_the_gate(tmp_path, monkeypatch):
    """Passing no config means "do not check staleness" -- used where the caller has no resolved
    config to compare against. It must not silently reject everything."""
    rc = _config()
    run_id = _lay_down_cruise(tmp_path, monkeypatch, rc)
    assert products.cruise_block(run_id, None) is not None


def test_cruise_run_matches_treats_an_unreadable_job_as_matching(tmp_path, monkeypatch):
    """Best-effort: a run predating stored configs has nothing to compare, and must not be
    thrown away."""
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    d = tmp_path / "oldrun"
    d.mkdir()
    (d / "job.json").write_text(json.dumps({"label": "no config here"}))
    assert products.cruise_run_matches("oldrun", _config(), None) is True


# ---------------------------------------------------------------------------
# what the signature is sensitive to
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"dry": 260.0},          # mass ratio -> capability
    {"prop": 340.0},         # mass ratio -> capability
    {"n": 3},                # thrust, power, and the acceleration limit
])
def test_signature_changes_with_the_vehicle(kwargs):
    assert products.trajectory_signature(_config(**kwargs), None) \
        != products.trajectory_signature(_config(), None)


def test_signature_changes_with_the_frozen_array_power():
    """A re-flown escape moves the power the cruise inherits, so the old arc is stale."""
    rc = _config()
    assert products.trajectory_signature(rc, 6000.0) != products.trajectory_signature(rc, 5000.0)


def test_signature_changes_with_the_arrival_deadline():
    rc = _config()
    later = rc.model_copy(deep=True)
    later.mission = Mission(**{**rc.mission.model_dump(),
                               "arrive_by": rc.mission.arrive_by.replace(
                                   year=rc.mission.arrive_by.year + 2)})
    assert products.trajectory_signature(later, None) != products.trajectory_signature(rc, None)


def test_signature_ignores_the_vehicle_name():
    """Renaming a vehicle does not change the trajectory it flies, so it must not invalidate a
    solve the user waited minutes for."""
    rc, renamed = _config(), _config()
    renamed.vehicle = renamed.vehicle.model_copy(update={"name": "different label"})
    assert products.trajectory_signature(renamed, None) == products.trajectory_signature(rc, None)


# ---------------------------------------------------------------------------
# convergence
# ---------------------------------------------------------------------------

def test_converged_accepts_the_flag_or_a_tight_mismatch():
    assert products.converged(True, 5.0)            # the optimizer's own verdict wins
    assert products.converged(False, 1e-6)          # a closed matchpoint despite a jittery flag
    assert not products.converged(False, 1.0)
    assert not products.converged(False, None)


# ---------------------------------------------------------------------------
# derived power / thrust numbers
# ---------------------------------------------------------------------------

def test_power_floor_pct_is_the_loss_that_starves_full_thrust():
    """An array delivering 4000 W at end of life from 8000 W BOL, against a 6000 W demand:
    full thrust is lost once 25% of BOL is gone."""
    spiral = {"power_required_W": 6000.0, "power_available_end_W": 4000.0,
              "power_fraction_end": 0.5}
    assert products.power_floor_pct(spiral) == pytest.approx(25.0)


def test_power_floor_pct_is_none_when_the_power_loop_is_open():
    assert products.power_floor_pct({}) is None
    assert products.power_floor_pct({"power_required_W": 0.0}) is None
    assert products.power_floor_pct(None) is None


def test_cruise_thrust_ceiling_is_the_flown_fraction_of_rated():
    rc = _config()
    rated_N = rc.total_thrust_mN * 1e-3
    assert products.cruise_thrust_ceiling_pct(rc, {"thrust_N": rated_N}) == pytest.approx(100.0)
    assert products.cruise_thrust_ceiling_pct(rc, {"thrust_N": rated_N * 0.5}) \
        == pytest.approx(50.0)
    assert products.cruise_thrust_ceiling_pct(rc, None) is None
    assert products.cruise_thrust_ceiling_pct(rc, {"thrust_N": 0.0}) is None


def test_available_thrust_fraction_is_one_when_power_rich():
    """No escape flown means no frozen power, and the cruise runs at rated thrust."""
    assert products.available_thrust_fraction(_config(), None) == 1.0


def test_available_thrust_fraction_falls_when_power_limited():
    rc = _config()
    starved = products.available_thrust_fraction(rc, rc.total_power_W * 0.4)
    assert 0.0 <= starved < 1.0


# ---------------------------------------------------------------------------
# the propellant integral
# ---------------------------------------------------------------------------

def _thrust_history(days: float, throttle: float, n: int = 200) -> dict:
    t = np.linspace(0.0, days, n)
    return {"sf": {"fine_times_days": t, "fine_throttle": np.full(n, throttle),
                   "thrust_N": 0.156, "isp_s": 1400.0, "propellant_kg": 0.0}}


def test_propellant_integral_matches_the_closed_form():
    """Constant throttle over a known span: mass = F * u * t / (Isp * g0)."""
    days, u = 100.0, 0.75
    res = _thrust_history(days, u)
    _d, prop_cum, verification = products.cruise_propellant_profile(res)
    expected = 0.156 * u * days * 86400.0 / (1400.0 * 9.80665)
    assert prop_cum[-1] == pytest.approx(expected, rel=1e-6)
    assert prop_cum[0] == 0.0
    assert verification["avg_throttle"] == pytest.approx(u, rel=1e-9)


def test_propellant_integral_uses_the_operating_point_not_the_rated_one():
    """The throttle history is a fraction of the thrust the solve FLEW. Integrating against the
    vehicle's rated numbers would over-count and push the curve above the usable load."""
    res = _thrust_history(100.0, 1.0)
    res["vehicle"] = {"thrust_mN": 600.0, "isp_s": 1400.0}     # rated, much higher than flown
    _d, prop_cum, _v = products.cruise_propellant_profile(res)
    flown = 0.156 * 100.0 * 86400.0 / (1400.0 * 9.80665)
    assert prop_cum[-1] == pytest.approx(flown, rel=1e-6)


def test_propellant_integral_falls_back_to_rated_when_no_operating_point_stored():
    res = _thrust_history(100.0, 1.0)
    del res["sf"]["thrust_N"], res["sf"]["isp_s"]
    res["vehicle"] = {"thrust_mN": 156.0, "isp_s": 1400.0}
    _d, prop_cum, _v = products.cruise_propellant_profile(res)
    assert prop_cum[-1] == pytest.approx(0.156 * 100.0 * 86400.0 / (1400.0 * 9.80665), rel=1e-6)


def test_propellant_integral_is_empty_without_a_usable_history():
    for res in (None, {"sf": {}}, {"sf": {"fine_times_days": [1.0], "fine_throttle": [1.0]}}):
        days, prop, verification = products.cruise_propellant_profile(res)
        assert len(days) == 0 and len(prop) == 0 and verification is None


# ---------------------------------------------------------------------------
# figure builders degrade rather than raise
# ---------------------------------------------------------------------------

def test_escape_figures_are_empty_without_a_path():
    assert products.escape_figures(None) == []
    assert products.escape_figures({"positions_km": []}) == []


def test_cruise_figures_are_empty_without_a_result():
    assert products.cruise_figures(None) == []


def test_array_power_figure_is_none_without_a_path():
    assert products.array_power_figure(None) is None


def test_mass_allocation_figure_is_none_without_components():
    assert products.mass_allocation_figure(None, dry_mass_kg=250.0) is None
    assert products.mass_allocation_figure([], dry_mass_kg=250.0) is None


def test_spiral_block_is_none_without_a_run():
    assert products.spiral_block(None) is None


def test_spiral_power_is_none_without_a_run():
    assert products.spiral_power_available_W(None) is None


def test_spiral_power_reads_the_frozen_end_of_escape_value(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    run_id = jobs.SPIRAL.new_run({"label": "escape"})
    jobs.write_spiral_result(run_id, {"spiral": {"power_available_end_W": 5200.0}})
    assert products.spiral_power_available_W(run_id) == pytest.approx(5200.0)
    # A run whose power loop never closed reports zero, which must read as "no frozen power".
    jobs.write_spiral_result(run_id, {"spiral": {"power_available_end_W": 0.0}})
    assert products.spiral_power_available_W(run_id) is None


def test_pickled_results_round_trip_through_the_readers(tmp_path, monkeypatch):
    """The worker stores results as pickle; the readers must return them unchanged."""
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    rc = _config()
    run_id = _lay_down_cruise(tmp_path, monkeypatch, rc)
    raw = pickle.loads((jobs.SOLVE.path(run_id, "result.pkl")).read_bytes())
    assert products.cruise_block(run_id, rc)["sf"]["propellant_kg"] == raw["sf"]["propellant_kg"]


# ---------------------------------------------------------------------------
# the flight-time trade: kept while it is true, retired the moment it is not
# ---------------------------------------------------------------------------

_TRADE = {
    "points": [{"tof_days": 300.0, "margin_kms": 0.41, "feasible": True,
                "max_tof_days": 320.0, "total_dv_kms": 6.1},
               {"tof_days": 240.0, "margin_kms": -0.22, "feasible": True,
                "max_tof_days": 260.0, "total_dv_kms": 6.7},
               {"tof_days": 200.0, "margin_kms": -0.90, "feasible": False,
                "max_tof_days": 200.0, "total_dv_kms": 7.4}],
    "step_days": 20.0,
}


# ---------------------------------------------------------------------------
# duty cycle: what the engine was actually asked to do
# ---------------------------------------------------------------------------

def _throttle_history(days, thr):
    return {"fine_times_days": np.asarray(days, float),
            "fine_throttle": np.asarray(thr, float), "max_duty_cycle": 1.0}


def test_duty_cycle_reports_peak_and_whole_phase_separately():
    """The two numbers answer different questions and are usually far apart.

    An efficient low-thrust transfer is a few hard burns between long coasts, so the busiest
    segment runs at full thrust while the share of the whole cruise spent firing is small. A
    per-segment cap binds against the peak; what the mission asks of the engine is the mean.
    Conflating them is how a duty limit gets sized against the wrong number.
    """
    days = np.linspace(0.0, 300.0, 601)
    thr = np.where((days < 30.0) | ((days > 200.0) & (days < 230.0)), 1.0, 0.0)
    duty = products.cruise_duty_cycle(_throttle_history(days, thr))
    assert duty["peak"] == pytest.approx(1.0)
    assert duty["mean"] == pytest.approx(60.0 / 300.0, abs=0.01)


def test_duty_cycle_is_time_weighted_not_a_sample_average():
    """Samples are not guaranteed evenly spaced, so a plain mean of the throttle column would
    weight a densely-sampled burn more heavily than the coast around it."""
    # Full thrust for the first 10 days, sampled finely; coasting for 90, sampled coarsely.
    days = np.concatenate([np.linspace(0.0, 10.0, 101), np.linspace(20.0, 100.0, 5)])
    thr = np.concatenate([np.ones(101), np.zeros(5)])
    duty = products.cruise_duty_cycle(_throttle_history(days, thr))
    assert duty["mean"] < 0.2, "a sample average would have read ~0.95 here"


def test_duty_cycle_carries_the_cap_that_was_asked_for():
    duty = products.cruise_duty_cycle(
        {"fine_times_days": [0.0, 100.0], "fine_throttle": [0.5, 0.5], "max_duty_cycle": 0.8})
    assert duty["cap"] == pytest.approx(0.8)
    assert duty["mean"] == pytest.approx(0.5)


@pytest.mark.parametrize("sf", [
    None, {}, {"fine_times_days": [], "fine_throttle": []},
    {"fine_times_days": [5.0], "fine_throttle": [1.0]},          # one sample spans no time
    {"fine_times_days": [3.0, 3.0], "fine_throttle": [1.0, 1.0]},  # zero-length span
])
def test_duty_cycle_is_none_without_a_usable_history(sf):
    assert products.cruise_duty_cycle(sf) is None
