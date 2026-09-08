"""The shared candidate evaluation: one path for the interactive refine and the design sweep.

``pipeline.solve.evaluate_candidate`` runs the cruise from a spread of starts, the return when
planned, and assembles the block the app reads; ``sweep.price_escape`` prices the escape the one
way every cruise does; the design search's spiral flies through the app's own entry point. These
pin the seams so the two paths cannot drift apart again.
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from prospector.config import EngineMount, Mission, ResolvedConfig, Screening, Vehicle
from prospector.launch import LaunchOrbit
from prospector.spacecraft.propulsion import Engine
from prospector.trades import sweep

LEO = LaunchOrbit(name="leo", perigee_alt_km=400, apogee_alt_km=400, inclination_deg=28.5)
CAT = {"E": Engine(name="E", isp_s=2000, thrust_mN=100, power_W=1000, mass_kg=5)}


def _rc(**mission_kw) -> ResolvedConfig:
    mission = Mission(launch_orbit="LEO", launch_window=(date(2028, 1, 1), date(2028, 3, 31)),
                      arrive_by=date(2030, 1, 1), **mission_kw)
    veh = Vehicle(name="v", dry_mass=300, fuel_mass=200, engines=[EngineMount(type="E", count=2)])
    return ResolvedConfig.build(mission, veh, Screening(), CAT, launches={"LEO": LEO})


# ---------------------------------------------------------------------------
# escape pricing is one function
# ---------------------------------------------------------------------------

def test_price_escape_is_what_the_budget_refinement_and_the_sweep_both_use():
    rc = _rc()
    curve = {"curve_vinf_kms": [0.0, 1.0], "curve_dv_kms": [7.0, 7.4],
             "curve_tof_days": [200.0, 230.0]}
    rc_point, terms = sweep.price_escape(rc, curve, 0.5)
    assert terms["escape_dv_kms"] == pytest.approx(7.2)
    assert terms["escape_tof_days"] == pytest.approx(215.0)
    veff = rc.effective_isp * 9.80665e-3
    assert terms["escape_propellant_kg"] == pytest.approx(
        rc.vehicle.wet_mass * (1.0 - np.exp(-7.2 / veff)))
    assert rc_point.escape_dv_refined == pytest.approx(7.2)
    assert rc_point.escape_tof_refined == pytest.approx(215.0)
    assert rc_point.cruise_start_mass_kg == pytest.approx(
        rc.vehicle.wet_mass - terms["escape_propellant_kg"])
    # The app's budget refinement reads the same figures off a spiral summary.
    from ui import state
    S = state.S
    saved = (S.mission, S.vehicle, S.screening, S.desirability, S.departure_vinf)
    try:
        S.mission, S.vehicle, S.screening = rc.mission, rc.vehicle, rc.screening
        S.departure_vinf = 0.5
        rc_app = rc.model_copy(update={"departure_vinf_kms": 0.5})
        rc_ui, _ = sweep.price_escape(rc_app, curve, rc_app.departure_vinf_kms)
        assert rc_ui.escape_dv_refined == pytest.approx(rc_point.escape_dv_refined)
        assert rc_ui.escape_prop_refined == pytest.approx(rc_point.escape_prop_refined)
    finally:
        S.mission, S.vehicle, S.screening, S.desirability, S.departure_vinf = saved


def test_with_escape_terms_is_how_a_sweep_winner_is_rebuilt():
    rc = _rc()
    rc2 = sweep.with_escape_terms(rc, dv_kms=7.9, tof_days=230.0, propellant_kg=200.0)
    assert (rc2.escape_dv_refined, rc2.escape_tof_refined, rc2.escape_prop_refined) == (7.9, 230.0, 200.0)
    assert rc2.cruise_start_mass_kg == pytest.approx(rc.vehicle.wet_mass - 200.0)


# ---------------------------------------------------------------------------
# evaluate_candidate orchestrates the shared pieces
# ---------------------------------------------------------------------------

def _stub_pipeline(monkeypatch, sv, *, fail_starts=False):
    seen = {}
    fake_sol = SimpleNamespace(feasible=True, mismatch=1e-6, final_mass_kg=400.0)

    def fake_cells(rc, row, cells, sf_kwargs, workers, max_revs, scaled, x0=None, x0_cell=None):
        seen["cells"] = list(cells)
        seen["sf_kwargs"] = dict(sf_kwargs)
        seen["x0"] = x0
        seen["x0_cell"] = x0_cell
        seen["workers"] = workers
        return None if fail_starts else fake_sol

    def fake_finish(rc, target, earth, row, seed, progress, sf_kwargs, sol=None, **kw):
        seen["finish_light"] = kw.get("light")
        seen["finish_sol"] = sol
        return {"sf": {"dep_mjd2000": 10500.0, "tof_days": 400.0, "final_mass_kg": 400.0,
                       "propellant_kg": 20.0}, "target": {}, "orbits": {}}

    def fake_return(rc, row, asteroid, earth, sf_block, progress, return_options, **kw):
        seen["return_options"] = dict(return_options)
        seen["return_light"] = kw.get("light")
        return {"sf": {"dv_kms": 1.0}, "total_return_propellant_kg": 30.0}

    monkeypatch.setattr(sv, "_mission_elements", lambda rc, row: dict(row))
    monkeypatch.setattr(sv.lb, "planet_from_row", lambda row: object())
    monkeypatch.setattr(sv.lb, "earth_planet", lambda: object())
    monkeypatch.setattr(sv, "solve_cells", fake_cells)
    monkeypatch.setattr(sv, "_finish", fake_finish)
    monkeypatch.setattr(sv, "_solve_return", fake_return)
    return seen, fake_sol


def test_evaluate_candidate_runs_the_shared_multistart_and_assembles_light(monkeypatch):
    pytest.importorskip("pykep")
    from prospector.trades.pipeline import solve as sv
    seen, fake_sol = _stub_pipeline(monkeypatch, sv)
    rc = _rc()
    cells = [(10500.0, 10900.0), (10510.0, 10950.0)]
    out = sv.evaluate_candidate(rc, {"a": 0.9, "pdes": "x"}, cells=cells,
                                sf_options={"nseg": 12, "available_power_W": 3800.0,
                                            "n_starts": 9, "x0": [1.0]},
                                x0=[1.0, 2.0], workers=1, light=True)
    assert seen["cells"] == cells and seen["x0"] == [1.0, 2.0] and seen["x0_cell"] == cells[0]
    # The multi-start's own knobs never reach the solver; the physics options do.
    assert "n_starts" not in seen["sf_kwargs"] and "x0" not in seen["sf_kwargs"]
    assert seen["sf_kwargs"]["nseg"] == 12
    assert seen["finish_sol"] is fake_sol and seen["finish_light"] is True
    assert "return" not in out                      # one-way mission: no return leg


def test_evaluate_candidate_hands_the_array_power_to_the_return_and_keeps_a_failed_one(monkeypatch):
    pytest.importorskip("pykep")
    from prospector.trades.pipeline import solve as sv
    seen, _ = _stub_pipeline(monkeypatch, sv)
    rc = _rc(return_trip=True, return_by=date(2032, 1, 1), time_at_asteroid=30.0)
    out = sv.evaluate_candidate(rc, {"a": 0.9, "pdes": "x"}, cells=[(10500.0, 10900.0)],
                                sf_options={"available_power_W": 3800.0},
                                return_options={"nseg": 20}, light=True)
    assert seen["return_options"] == {"nseg": 20, "available_power_W": 3800.0}
    assert seen["return_light"] is True
    assert out["return"]["total_return_propellant_kg"] == 30.0
    # A return that raises is recorded, not fatal.
    monkeypatch.setattr(sv, "_solve_return",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("window too short")))
    out = sv.evaluate_candidate(rc, {"a": 0.9, "pdes": "x"}, cells=[(10500.0, 10900.0)],
                                sf_options={})
    assert out["return"] == {"error": "window too short"}


def test_evaluate_candidate_raises_when_no_start_produces_a_solution(monkeypatch):
    pytest.importorskip("pykep")
    from prospector.trades.pipeline import solve as sv
    _stub_pipeline(monkeypatch, sv, fail_starts=True)
    with pytest.raises(ValueError, match="no starting cell"):
        sv.evaluate_candidate(_rc(), {"a": 0.9}, cells=[(10500.0, 10900.0)], sf_options={})


def test_the_sweep_point_is_built_from_the_apps_result_block():
    """The sweep's row is a projection of the same block the app reads, so the two agree."""
    rc = _rc()
    rc_point, terms = sweep.price_escape(rc, None, 0.0)
    point = {"vinf_kms": 0.0, **terms}
    z = np.zeros(45)
    z[0], z[2], z[-1] = 10500.0, 380.0, 400.0
    sf = {"feasible": True, "mismatch": 1e-6, "dep_mjd2000": 10500.0, "tof_days": 400.0,
          "dv_kms": 3.0, "final_mass_kg": 380.0, "propellant_kg": 25.0, "decision_vector": z,
          "seg_caps": np.array([0.5, 1.0]), "seg_isp_s": np.array([1712.0, 1881.0]),
          "thrust_N": 0.249, "isp_s": 1798.0, "refresh_settled": False}
    out = sweep._point_from_result(point, rc_point, {"sf": sf, "target": {}, "orbits": {}})
    assert out["cruise_dv_kms"] == 3.0 and out["cruise_propellant_kg"] == 25.0
    assert out["propellant_kg"] == pytest.approx(terms["escape_propellant_kg"] + 25.0)
    assert out["total_dv_kms"] == pytest.approx(terms["escape_dv_kms"] + 3.0)
    assert out["settled"] is False
    assert out["cruise_thrust_N"] == 0.249 and out["cruise_isp_s"] == 1798.0
    assert out["seg_caps"] == [0.5, 1.0] and out["cruise_seg_isp_s"] == [1712.0, 1881.0]
    assert out["decision_vector"] == z.tolist()
    assert out["return_dv_kms"] is None if "return_dv_kms" in out else True


# ---------------------------------------------------------------------------
# the design search's escape flies through the app's entry point
# ---------------------------------------------------------------------------

def test_spiral_curve_flies_the_apps_spiral_with_a_probe_vehicle(tmp_path, monkeypatch):
    from prospector.solvers import spiral as sp
    from prospector.trades.design_search import evaluate as ev

    seen = {}

    def fake_solve_for_config(probe, **opts):
        seen["probe"] = probe
        seen["opts"] = opts
        n = 3
        return SimpleNamespace(
            status="escaped", dv_at_escape_kms=7.0, dv_at_escape_equiv_kms=7.3,
            tof_at_escape_days=200.0, belt_days=30.0, eclipse_days=5.0, revolutions=700.0,
            power_fraction_end=0.9, power_limited=True, power_required_W=2000.0,
            power_available_end_W=1800.0, curve_vinf_kms=np.linspace(0, 2, n),
            curve_dv_kms=np.linspace(7.3, 7.9, n), curve_tof_days=np.linspace(200, 260, n))

    monkeypatch.setattr(sp, "solve_for_config", fake_solve_for_config)
    rc = _rc()
    curve = ev.spiral_curve(rc, duty=0.9, vinf_max=2.0, cache_dir=tmp_path, bol_power_W=5000.0,
                            radiation_model="ap8min-worstcase", coverglass_um=150.0,
                            coverglass_density=2.0)
    probe, opts = seen["probe"], seen["opts"]
    # The probe: same wet mass, thrust and drag area; a 1 kg dry mass so the climb never runs dry;
    # the array THIS pass flies.
    assert probe.vehicle.wet_mass == pytest.approx(rc.vehicle.wet_mass)
    assert probe.vehicle.burnout_mass == pytest.approx(1.0)
    assert probe.vehicle.solar_power_W == 5000.0
    assert probe.vehicle.area_m2 == rc.vehicle.area_m2
    assert probe.total_power_W == rc.total_power_W
    # The app's knobs, by name, through the app's entry point.
    assert opts["duty_cycle"] == 0.9 and opts["target_vinf_kms"] == 2.0
    assert opts["max_years"] == 8.0
    assert opts["radiation_model"] == "ap8min-worstcase"
    assert opts["coverglass_um"] == 150.0 and opts["coverglass_density"] == 2.0
    assert opts["array_to_thruster_eff"] == pytest.approx(rc.thruster_chain_eff())
    assert opts["array_reserve_W"] == pytest.approx(rc.array_reserve_W())
    # The budget term is the rated-Isp equivalent; the flown figure rides along.
    assert curve["dv_at_escape_kms"] == 7.3 and curve["dv_at_escape_flown_kms"] == 7.0
    assert curve["key"]["v"] == 10 and curve["key"]["coverglass_um"] == 150.0
    # Cached: a second call with the same terms does not fly again.
    seen.clear()
    ev.spiral_curve(rc, duty=0.9, vinf_max=2.0, cache_dir=tmp_path, bol_power_W=5000.0,
                    radiation_model="ap8min-worstcase", coverglass_um=150.0, coverglass_density=2.0)
    assert not seen


def test_the_cruise_duty_is_its_own_setting_in_the_design_search():
    """One --duty used to throttle both the spiral and the cruise to 90%; the app throttles the
    spiral only. The cruise now has its own limit, unthrottled by default."""
    from prospector.trades.design_search.args import parse_args
    args = parse_args(["--mission", "m", "--duty", "0.8"])
    assert args.duty == 0.8 and args.cruise_duty == 1.0
    args = parse_args(["--mission", "m", "--cruise-duty", "0.7", "--radiation-model", "x",
                       "--coverglass-um", "150", "--coverglass-density", "2.0"])
    assert args.cruise_duty == 0.7 and args.radiation_model == "x"
    assert args.coverglass_um == 150.0 and args.coverglass_density == 2.0
