"""The gravity assist from the mission file to the result block.

A mission names a planet and the cruise becomes two legs: the grid's cells, the cruise, a stored
point's rebuild and the sweep all pick the two-leg solver through one dispatch and never see the
difference. The block a flyby produces has the same keys as a direct leg's, with the two legs
joined and a ``flyby`` sub-block, so the flight view, the return leg and the report read it as
one cruise. Dawn via Mars is the end-to-end check.
"""
from datetime import date

import numpy as np
import pykep as pk
import pytest
from pydantic import ValidationError

from prospector import paths
from prospector.config import Mission, load_mission, load_study, resolve_study, save_mission
from prospector.population import targets
from prospector.solvers import flyby as fb
from prospector.solvers import lambert as lb
from prospector.solvers import simsflanagan as sf
from prospector.solvers import transfer
from prospector.spacecraft.buildability import BusModel
from prospector.trades.pipeline import arcs


def test_the_mission_names_a_planet_or_flies_direct():
    assert Mission().gravity_assist is None
    assert Mission(gravity_assist="Mars").gravity_assist == "mars"
    assert Mission(gravity_assist="").gravity_assist is None
    with pytest.raises(ValidationError, match="venus, earth, mars, jupiter"):
        Mission(gravity_assist="pluto")


def test_the_choice_survives_the_library(tmp_path):
    save_mission(Mission(name="Via Mars", gravity_assist="mars"), "via-mars", config_dir=tmp_path)
    assert load_mission("via-mars", config_dir=tmp_path).gravity_assist == "mars"
    save_mission(Mission(name="Direct"), "direct", config_dir=tmp_path)
    assert load_mission("direct", config_dir=tmp_path).gravity_assist is None


def test_the_minimum_flyby_altitude_is_a_settings_field():
    from ui import settings
    assert BusModel().flyby_min_altitude_km == 500.0
    grouped = set().union(*settings._GROUPS.values())
    assert "flyby_min_altitude_km" in grouped and "flyby_min_altitude_km" in settings._LABELS


def test_flyby_body_keys():
    assert fb.flyby_body("mars").get_name().startswith("mars")
    assert fb.flyby_body("Earth").get_name().startswith("earth")


def test_the_arrival_speed_is_one_mission_field():
    """Zero is a rendezvous and solves at the rendezvous slack; above zero the cruise may pass the
    target at up to that speed, which is a flyby or an impactor; a trip that passes the target
    cannot come back."""
    assert Mission().arrival_vinf_kms == 0.0
    rc = resolve_study(load_study("dawn", config_dir=paths.EXAMPLE_CONFIG_DIR),
                       config_dir=paths.EXAMPLE_CONFIG_DIR)
    assert rc.arrival_vinf_kms == 0.1
    rc.mission = rc.mission.model_copy(update={"arrival_vinf_kms": 6.0})
    assert rc.arrival_vinf_kms == 6.0
    with pytest.raises(ValidationError, match="cannot stay at it or return"):
        Mission(arrival_vinf_kms=6.0, return_trip=True, return_by=date(2035, 1, 1))


def _dawn():
    rc = resolve_study(load_study("dawn", config_dir=paths.EXAMPLE_CONFIG_DIR),
                       config_dir=paths.EXAMPLE_CONFIG_DIR)
    rc.departure_vinf_kms = 3.3
    vesta = lb.planet_from_row(targets.get_target("4 Vesta"))
    mars = pk.planet(pk.udpla.jpl_lp("mars"))
    return rc, vesta, mars


def test_the_mission_picks_the_leg_solver():
    rc, _, _ = _dawn()
    assert transfer.for_config(rc) is fb
    rc.mission = rc.mission.model_copy(update={"gravity_assist": None})
    assert transfer.for_config(rc) is sf


def test_a_cell_owns_its_window_and_flight_time_and_is_its_own_start(monkeypatch):
    """The grid hands every cell the one-leg settings it uses for direct cells: a Lambert seed, a
    warm start from the neighbouring cell, whole-trip flight-time bounds. The two-leg cell holds
    its departure day and its total flight time, starts from the impulsive optimum between its own
    two days, and takes the neighbour's vector only when its flight time already fits."""
    rc, vesta, _ = _dawn()
    seen = {}

    def fake_solve(target, body, **kw):
        seen.update(kw)
        return "sentinel"

    monkeypatch.setattr(fb, "solve", fake_solve)
    t0 = lb.mjd2000_from_date(date(2007, 10, 1))
    neighbour = np.zeros(17 + 3 * 24)
    neighbour[fb._TOF1], neighbour[fb._TOF2] = 500.0, 950.0          # a 1450-day vector
    out = fb.solve_cell_for_config(rc, vesta, dep_mjd2000=t0, tof_days=1370.0, x0=neighbour,
                                   seed="a one-leg Lambert transfer", min_tof_days=150.0,
                                   max_tof_days=1400.0, nseg=12, restarts=2, max_duty_cycle=0.95,
                                   vinf_dep_kms=3.3)
    assert out == "sentinel"
    assert seen["launch_window"] == (date(2007, 10, 1), date(2007, 10, 1))
    assert seen["window_slack_days"] == 0.75
    assert seen["tof_total_bounds"] == (1370.0 - 0.75, 1370.0 + 0.75)
    assert seen["seed"] == (t0, t0 + 1370.0)
    assert seen["x0"] is None                       # the neighbour's flight time does not fit
    assert seen["nseg"] == (12, 12) and seen["restarts"] == 2
    assert seen["vinf_arr_kms"] == 0.1 and seen["min_flyby_alt_km"] == 500.0
    assert seen["vinf_dep_exact"] is True                    # a launcher-provided escape
    assert "min_tof_days" not in seen or seen["min_tof_days"] == (60.0, 120.0)
    assert "max_tof_days" not in seen


def test_a_warm_start_is_one_candidate_among_the_geometries(monkeypatch):
    """With both a cell and a warm start, the two-leg solve descends the warm start and still
    tries its impulsive geometries; the warm start alone (a re-solve) runs no geometries."""
    rc, vesta, mars = _dawn()
    calls = {"starts": 0, "descents": 0}
    x = np.zeros(17 + 3 * 6)
    x[fb._TOF1], x[fb._TOF2], x[fb._MFB], x[fb._MF] = 500.0, 850.0, 1100.0, 1000.0
    monkeypatch.setattr(fb, "_impulsive_start",
                        lambda udp, a, b, seed: calls.__setitem__("starts", calls["starts"] + 1) or None)
    monkeypatch.setattr(fb, "_descend",
                        lambda udp, z, **kw: (calls.__setitem__("descents", calls["descents"] + 1)
                                              or ((True, 1e-6, 1000.0), np.asarray(z, float))))
    kw = dict(mass_kg=1217.7, thrust_N=0.0927, isp_s=3127.0, nseg=(3, 3),
              launch_window=rc.departure_window, arrive_by=rc.mission.arrive_by, vinf_dep_kms=3.3)
    t0 = lb.mjd2000_from_date(rc.departure_window[0])
    fb.solve(vesta, mars, x0=x, seed=(t0, t0 + 1350.0), restarts=3, **kw)
    assert calls == {"starts": 3, "descents": 2}      # x0, then the polish; no geometry closed
    calls.update(starts=0, descents=0)
    fb.solve(vesta, mars, x0=x, restarts=3, **kw)
    assert calls == {"starts": 0, "descents": 2}      # rounds from x0, then the polish


def test_the_free_solve_maps_one_leg_bounds_onto_the_whole_trip(monkeypatch):
    """The cruise and the grid's polish pass a whole-trip flight-time range in the one-leg names;
    the two-leg solve reads them as bounds on the total and keeps its own per-leg minimums. A
    Lambert seed's two epochs are the start's days."""
    from types import SimpleNamespace
    rc, vesta, _ = _dawn()
    seen = {}
    monkeypatch.setattr(fb, "solve", lambda target, body, **kw: seen.update(kw) or "sentinel")
    seed = SimpleNamespace(dep_mjd2000=2830.0, arr_mjd2000=4200.0)
    fb.solve_for_config(rc, vesta, seed=seed, min_tof_days=1300.0, max_tof_days=1380.0, nseg=10)
    assert seen["seed"] == (2830.0, 4200.0)
    assert seen["tof_total_bounds"] == (1300.0, 1380.0)
    assert seen["nseg"] == (10, 10) and "max_tof_days" not in seen


def _point(udp):
    lo, hi = (np.asarray(b, float) for b in udp.get_bounds())
    x = np.zeros(udp.dim)
    x[fb._T0], x[fb._TOF1], x[fb._TOF2] = lo[0], 500.0, 850.0
    x[fb._MFB], x[fb._MF] = 1150.0, 980.0
    x[fb._VIN] = x[fb._VOUT] = [2000.0, 500.0, 0.0]
    x[udp._u1] = 0.3
    x[udp._u2] = 0.2
    return np.clip(x, lo, hi)


def test_a_flyby_block_reads_like_one_cruise():
    """Two legs joined: node and fine arrays run on across the flyby, times offset by the first
    leg, the flyby sub-block present, and the flyby body's track in the orbits."""
    rc, vesta, mars = _dawn()
    udp = fb._make_udp(vesta, mars, mass_kg=1217.7, thrust_N=0.0927, isp_s=3127.0, nseg=(3, 3),
                       launch_window=(date(2007, 9, 26), date(2007, 10, 15)),
                       arrive_by=date(2011, 7, 16), vinf_dep_kms=3.3, vinf_arr_kms=0.1,
                       window_slack_days=0.0, min_tof_days=(60.0, 120.0), max_tof_days=None,
                       max_duty_cycle=0.95, max_dep_decl_deg=None, depart_body=None,
                       min_flyby_alt_km=500.0, flyby_mu=None)
    x = _point(udp)
    fsol = fb._build_solution(udp, x, "Vesta", "mars", 0.0927, 0.95)
    block, orbits = arcs._assemble_flyby(fsol, lb.earth_planet(), mars, vesta, 2.36, lambda *a: None)
    n_fine = len(block["fine_times_days"])
    assert n_fine == len(block["fine_positions_au"]) == len(block["fine_throttle"])
    assert len(orbits["flyby_track_au"]) == n_fine == len(orbits["earth_track_au"])
    assert np.all(np.diff(block["fine_times_days"]) >= 0)
    assert block["fine_times_days"][-1] == pytest.approx(fsol.tof_days, rel=1e-6)
    assert block["tof_days"] == pytest.approx(1350.0) and block["dep_mjd2000"] == x[0]
    assert len(block["node_times_days"]) == 14            # 2*3+1 nodes per leg, both legs
    assert block["flyby"]["body"].startswith("mars")
    assert block["flyby"]["tof1_days"] == pytest.approx(500.0)
    assert block["flyby"]["leg_propellant_kg"] == pytest.approx([1217.7 - 1150.0, 1150.0 - 980.0])
    assert block["flyby"]["leg_isp_s"] == [3127.0, 3127.0]
    assert block["propellant_kg"] == pytest.approx(1217.7 - 980.0)
    assert isinstance(block["isp_s"], float) and block["isp_s"] > 0
    assert "direct" not in block
    assert orbits["flyby_name"].startswith("mars") and len(orbits["flyby_au"]) > 10


def test_a_rebuild_reproduces_a_point_from_its_stored_terms():
    """The grid and the sweep store a two-leg vector with its leg Isps as a pair and its per-leg
    arrays flat; the rebuild evaluates the same problem and reports the same numbers."""
    from prospector.trades.pipeline.grid import leg_terms
    rc, vesta, mars = _dawn()
    udp = fb._make_udp(vesta, mars, mass_kg=rc.cruise_start_mass_kg, thrust_N=0.0927,
                       isp_s=(3127.0, 3105.0), nseg=(3, 3), launch_window=rc.departure_window,
                       arrive_by=rc.mission.arrive_by, vinf_dep_kms=3.3, vinf_arr_kms=0.1,
                       window_slack_days=0.0, min_tof_days=(60.0, 120.0), max_tof_days=None,
                       max_duty_cycle=0.95, max_dep_decl_deg=None, depart_body=None,
                       min_flyby_alt_km=500.0, flyby_mu=None)
    x = _point(udp)
    sol = fb._build_solution(udp, x, "Vesta", "mars", 0.0927, 0.95,
                             seg_isp_s=(np.full(3, 3127.0), np.full(3, 3105.0)))
    terms = leg_terms(sol)
    assert terms["isp_s"] == [3127.0, 3105.0] and len(terms["seg_isp_s"]) == 6
    rebuilt = fb.rebuild_for_config(rc, vesta, x, nseg=3, max_duty_cycle=0.95, vinf_dep_kms=3.3,
                                    thrust_cap_fn=None, **terms)
    assert rebuilt.isp_s == (3127.0, 3105.0)
    assert rebuilt.mismatch == pytest.approx(sol.mismatch, rel=1e-9)
    assert rebuilt.final_mass_kg == pytest.approx(sol.final_mass_kg)
    assert rebuilt.flyby["periapsis_alt_km"] == pytest.approx(sol.flyby["periapsis_alt_km"])
    assert np.allclose(rebuilt.seg_isp_s[1], 3105.0)


def test_the_direct_comparison_is_a_sentence_beside_the_answer(monkeypatch):
    """A cruise run for a mission via a planet also flies the same cell direct, once, and records
    it under the block; the sweep and the grid, which call the evaluation directly, never do."""
    from types import SimpleNamespace

    from prospector.trades.pipeline import solve as pipeline
    rc, vesta, _ = _dawn()
    row = targets.get_target("4 Vesta")
    calls = []
    monkeypatch.setattr(pipeline, "evaluate_candidate",
                        lambda *a, **k: {"sf": {"propellant_kg": 215.0, "vinf_dep_kms": 3.3,
                                               "vinf_arr_kms": 0.1}})
    monkeypatch.setattr(pipeline.sf, "solve_for_config",
                        lambda *a, **k: calls.append(k) or SimpleNamespace(
                            feasible=True, propellant_kg=272.0, dv_kms=7.7, tof_days=1360.0,
                            dep_mjd2000=2830.0))
    t0 = lb.mjd2000_from_date(date(2007, 9, 27))
    res = pipeline.solve_from_cell(rc, row, t0, t0 + 1380.0, nseg=12, n_starts=3, x0=[0.0])
    assert res["sf"]["direct"]["propellant_kg"] == 272.0
    assert calls and "n_starts" not in calls[0] and "x0" not in calls[0]
    res = pipeline.solve_from_cell(rc, row, t0, t0 + 1380.0, nseg=12, compare_direct=False)
    assert "direct" not in res["sf"] and len(calls) == 1


@pytest.mark.slow
def test_dawn_via_mars_through_the_pipeline():
    """The app's cruise solve on the Dawn example with its Mars gravity assist: the cell is a
    two-leg start, the block carries the flyby, and there is no direct transfer underneath."""
    from prospector.trades.pipeline import solve as pipeline
    rc, _, _ = _dawn()
    row = targets.get_target("4 Vesta")
    t0 = lb.mjd2000_from_date(rc.mission.launch_window[0])
    t_arr = lb.mjd2000_from_date(rc.mission.arrive_by)
    res = pipeline.evaluate_candidate(
        rc, row, cells=[(t0 + 5.0, t0 + 5.0 + 0.985 * (t_arr - t0 - 5.0))],
        sf_options={"nseg": 12, "restarts": 3, "max_duty_cycle": 0.95, "vinf_dep_kms": 3.3},
        light=True)
    block = res["sf"]
    assert block["feasible"]
    assert block["flyby"]["body"].startswith("mars")
    assert 190.0 < block["propellant_kg"] < 260.0
    assert block["flyby"]["periapsis_alt_km"] >= 500.0 - 1.0
    assert lb.date_from_mjd2000(block["dep_mjd2000"] + block["tof_days"]).date() <= rc.mission.arrive_by
    assert "direct" not in block
