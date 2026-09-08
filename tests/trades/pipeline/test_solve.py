"""The solve entry points: the return-leg ledger, and how the legs are orchestrated.

The Sims-Flanagan calls and the arc assembly are stubbed, because neither is what was untested
here. What was untested is the arithmetic around them, who pays for what propellant, what counts as
enough to fly home, and whether a return that cannot close takes the outbound down with it. Those
are decisions a round-trip mission is judged on, and every one of them is plain bookkeeping that
can be wrong while everything still runs.
"""
from __future__ import annotations

import numpy as np
import pytest

from prospector.config import EngineMount, ResolvedConfig, Screening, Vehicle, load_mission
from prospector.trades.pipeline import solve as sv
from tests import library as lib


@pytest.fixture
def rc() -> ResolvedConfig:
    """A round-trip config: 60 kg collected at the asteroid, delivered to EML2."""
    mission = load_mission(lib.mission_key())
    mission = mission.model_copy(update={
        "launch_orbit": "TLI",          # LV-provided escape: no spiral draw on the tank
        "return_trip": True,
        "return_by": mission.arrive_by.replace(year=2032),
        "return_destination": "EML2",
        "time_at_asteroid": 30.0,
        "asteroid_payload_mass": 60.0,
    })
    return ResolvedConfig.build(
        mission,
        Vehicle(name="probe", dry_mass=250.0, fuel_mass=400.0, unusable_prop=10.0,
                engines=[EngineMount(type=lib.engine_key(), count=4)]),
        Screening(), load_engines_())


def load_engines_():
    from prospector.spacecraft.propulsion import load_engines
    return load_engines()


class _Sol:
    """A converged leg, carrying only what the ledger reads off it."""

    def __init__(self, propellant_kg=80.0, final_mass_kg=300.0):
        self.propellant_kg = propellant_kg
        self.final_mass_kg = final_mass_kg
        self.decision_vector = [1.0, 2.0, 3.0]
        self.dv_kms = 2.0
        self.tof_days = 350.0
        self.dep_mjd2000 = 10500.0


def _outbound_block(final_mass_kg=430.0, propellant_kg=120.0) -> dict:
    return {"final_mass_kg": final_mass_kg, "propellant_kg": propellant_kg,
            "dep_mjd2000": 10500.0, "tof_days": 400.0}


@pytest.fixture
def stub_return(monkeypatch):
    """Replace the return solve and the arc assembly; keep the ledger real."""
    calls: dict = {}

    def _fake_solve(rc_, asteroid, **kw):
        calls.update(kw)
        return calls.get("_sol", _Sol())

    def _fake_rebuild(rc_, asteroid, x, **kw):
        calls.update(kw)
        calls["rebuilt_from"] = x
        return calls.get("_sol", _Sol())

    monkeypatch.setattr(sv.sf, "solve_return_for_config", _fake_solve)
    monkeypatch.setattr(sv.sf, "rebuild_return_for_config", _fake_rebuild)
    monkeypatch.setattr(sv, "_assemble_leg", lambda sol, *a, **k: ({"dv_kms": sol.dv_kms}, {}))
    return calls


def _run_return(rc, stub_return, *, sol=None, outbound=None, return_x=None, options=None):
    if sol is not None:
        stub_return["_sol"] = sol
    return sv._solve_return(rc, {"a": 0.922}, object(), object(),
                            outbound or _outbound_block(), lambda *a: None,
                            options or {}, return_x=return_x)


# ---------------------------------------------------------------------------
# the laden return: what it starts with
# ---------------------------------------------------------------------------

def test_the_return_departs_heavier_by_the_collected_payload(rc, stub_return):
    """The spacecraft leaves the asteroid carrying what it collected, so the return flies a
    heavier vehicle than the one that arrived."""
    _run_return(rc, stub_return, outbound=_outbound_block(final_mass_kg=430.0))
    assert stub_return["start_mass_kg"] == pytest.approx(430.0 + 60.0)


def test_the_return_departs_after_the_stay(rc, stub_return):
    """Departure is the outbound arrival plus the time spent at the asteroid; leaving on the
    arrival date would plan a mission that never does the work it went for."""
    from prospector.solvers import lambert as lb
    _run_return(rc, stub_return, outbound=_outbound_block())
    expected = lb.date_from_mjd2000(10500.0 + 400.0 + 30.0).date()
    assert stub_return["depart_window"] == (expected, expected)


def test_the_return_is_bounded_by_the_mission_deadline(rc, stub_return):
    _run_return(rc, stub_return)
    assert stub_return["arrive_by"] == rc.mission.return_by


# ---------------------------------------------------------------------------
# the propellant ledger
# ---------------------------------------------------------------------------

def test_available_propellant_is_what_escape_and_the_outbound_left(rc, stub_return):
    """The return may only spend what is actually still in the tank."""
    out = _run_return(rc, stub_return, outbound=_outbound_block(propellant_kg=120.0))
    expected = rc.usable_propellant_kg - rc.escape_propellant_kg - 120.0
    assert expected > 0.0                       # the fixture leaves a real remainder to spend
    assert out["available_propellant_kg"] == pytest.approx(expected)


def test_a_tank_the_outbound_emptied_leaves_nothing_rather_than_a_negative(rc, stub_return):
    """A depleted tank cannot fly home, and it cannot owe propellant either: the remainder is
    floored at zero so a negative never propagates into the feasibility check as if it were
    spendable."""
    out = _run_return(rc, stub_return, outbound=_outbound_block(propellant_kg=10_000.0))
    assert out["available_propellant_kg"] == 0.0
    assert out["feasible_propellant"] is False


def test_the_return_bill_is_its_cruise_plus_the_insertion_burn(rc, stub_return):
    """Arriving near Earth is not arriving at the destination: capture into EML2 costs
    propellant that the cruise arc never flies, and it is charged on the laden arrival mass."""
    out = _run_return(rc, stub_return, sol=_Sol(propellant_kg=70.0, final_mass_kg=320.0))
    assert out["cruise_propellant_kg"] == pytest.approx(70.0)
    assert out["insertion_propellant_kg"] > 0.0
    assert out["total_return_propellant_kg"] == pytest.approx(
        70.0 + out["insertion_propellant_kg"])


def test_delivered_mass_is_the_arrival_mass_after_the_capture_burn(rc, stub_return):
    """What actually arrives is what is left once the insertion propellant is gone."""
    out = _run_return(rc, stub_return, sol=_Sol(final_mass_kg=320.0))
    assert out["delivered_mass_kg"] == pytest.approx(320.0 - out["insertion_propellant_kg"])


def test_a_return_within_the_remaining_propellant_is_feasible(rc, stub_return):
    out = _run_return(rc, stub_return, sol=_Sol(propellant_kg=5.0, final_mass_kg=320.0))
    assert out["total_return_propellant_kg"] <= out["available_propellant_kg"]
    assert out["feasible_propellant"] is True


def test_a_return_that_overspends_the_tank_is_not_feasible(rc, stub_return):
    """Converging is not the same as being able to fly it: a leg the tank cannot pay for has to
    be reported infeasible, or a design 'closes' and strands itself at the target."""
    out = _run_return(rc, stub_return, sol=_Sol(propellant_kg=10_000.0, final_mass_kg=320.0))
    assert out["feasible_propellant"] is False


def test_the_destination_terms_travel_with_the_result(rc, stub_return):
    """The heliocentric arc cannot see the destination, so its two terms -- the allowed arrival
    excess and the capture dV -- have to be reported alongside it."""
    out = _run_return(rc, stub_return)
    assert out["destination"] == "EML2"
    assert out["insertion_dv_kms"] > 0.0
    assert out["arrival_vinf_allow_kms"] >= 0.0
    assert out["payload_kg"] == pytest.approx(60.0)
    assert out["stay_days"] == pytest.approx(30.0)


def test_a_stored_decision_vector_rebuilds_instead_of_solving(rc, stub_return):
    """A swept return point is rebuilt in milliseconds rather than re-optimized; the ledger must
    come out the same either way."""
    out = _run_return(rc, stub_return, return_x=[9.0, 8.0], sol=_Sol(propellant_kg=70.0))
    assert stub_return["rebuilt_from"] == [9.0, 8.0]
    assert out["cruise_propellant_kg"] == pytest.approx(70.0)
    assert out["decision_vector"] == [1.0, 2.0, 3.0]


def test_return_solver_knobs_are_passed_through_and_defaulted(rc, stub_return):
    """The return leg has its own solver settings, independent of the outbound's."""
    _run_return(rc, stub_return, options={"nseg": 25, "max_duty_cycle": 0.7})
    assert stub_return["nseg"] == 25 and stub_return["max_duty_cycle"] == 0.7
    # The mission's return flight-time cap fills in when the caller does not set one.
    assert "max_tof_days" in stub_return


# ---------------------------------------------------------------------------
# orchestration: how a failing return is handled
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_outbound(monkeypatch):
    """Stub everything the outbound path needs, so solve_from_cell's control flow is the only
    thing under test."""
    seed = type("Seed", (), {"dep_mjd2000": 10500.0, "arr_mjd2000": 10900.0, "dv_kms": 4.0,
                             "tof_days": 400.0, "vinf_dep_kms": 1.0, "vinf_arr_kms": 0.2,
                             "c3_km2s2": 1.0})()
    monkeypatch.setattr(sv, "_mission_elements", lambda rc_, row: dict(row))
    monkeypatch.setattr(sv.lb, "planet_from_row", lambda row: object())
    monkeypatch.setattr(sv.lb, "earth_planet", lambda: object())
    monkeypatch.setattr(sv.lb, "lambert_transfer", lambda *a, **k: seed)
    monkeypatch.setattr(sv, "solve_cells", lambda *a, **k: object())   # the shared multi-start
    monkeypatch.setattr(sv, "_finish", lambda *a, **k: {"sf": _outbound_block(), "target": {}})
    return seed


def test_a_return_that_cannot_close_keeps_the_outbound(rc, stub_outbound, monkeypatch):
    """An unflyable return is a finding about the return, not a reason to throw away minutes of
    converged outbound trajectory. The reason is recorded instead."""
    monkeypatch.setattr(sv, "_solve_return",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("window too short")))
    out = sv.solve_from_cell(rc, {"a": 0.922, "pdes": "99942"}, 10500.0, 10900.0)
    assert out["sf"] == _outbound_block()                  # outbound survives intact
    assert out["return"] == {"error": "window too short"}


def test_a_one_way_mission_solves_no_return_leg(stub_outbound, monkeypatch):
    called = []
    monkeypatch.setattr(sv, "_solve_return", lambda *a, **k: called.append(1))
    # Forced one-way: the active library's mission may be a round trip.
    one_way = ResolvedConfig.build(
        load_mission(lib.mission_key()).model_copy(
            update={"return_trip": False, "return_by": None}),
        Vehicle(name="probe", dry_mass=250.0, fuel_mass=400.0,
                engines=[EngineMount(type=lib.engine_key(), count=4)]),
        Screening(), load_engines_())
    out = sv.solve_from_cell(one_way, {"a": 0.922, "pdes": "99942"}, 10500.0, 10900.0)
    assert called == [] and "return" not in out


def test_a_cell_with_no_transfer_is_an_error_not_an_empty_result(rc, stub_outbound, monkeypatch):
    """A cell whose arrival precedes departure has no transfer; returning an empty result would
    look like a solve that simply found nothing."""
    monkeypatch.setattr(sv.lb, "lambert_transfer", lambda *a, **k: None)
    with pytest.raises(ValueError, match="no valid Lambert transfer"):
        sv.solve_from_cell(rc, {"a": 0.922, "pdes": "99942"}, 10900.0, 10500.0)


def test_the_chosen_cell_is_reported_back_as_the_seed_baseline(rc, stub_outbound):
    """The lambert block is what the UI compares the converged low-thrust cost against, so it has
    to describe the cell actually seeded."""
    out = sv.solve_from_cell(rc, {"a": 0.922, "pdes": "99942"}, 10500.0, 10900.0)
    assert out["lambert"]["best_dep_mjd2000"] == 10500.0
    assert out["lambert"]["best_dv_kms"] == 4.0


def test_return_options_never_reach_the_outbound_solve(rc, stub_outbound, monkeypatch):
    """The two legs have independent solver settings; leaking the return's nseg into the
    outbound would silently solve the outbound at the wrong fidelity."""
    seen: dict = {}
    monkeypatch.setattr(sv, "_finish",
                        lambda rc_, t, e, row, seed, prog, sf_kwargs, **k:
                        (seen.update(sf_kwargs), {"sf": _outbound_block(), "target": {}})[1])
    monkeypatch.setattr(sv, "_solve_return", lambda *a, **k: {})
    sv.solve_from_cell(rc, {"a": 0.922, "pdes": "99942"}, 10500.0, 10900.0,
                       nseg=20, return_options={"nseg": 30})
    assert seen["nseg"] == 20 and "return_options" not in seen


def test_the_frozen_array_power_reaches_the_return_leg(rc, stub_outbound, monkeypatch):
    """The return flies after the belts too, so it inherits the same degraded array power as the
    outbound unless the caller overrode it."""
    seen: dict = {}
    monkeypatch.setattr(sv, "_solve_return",
                        lambda rc_, row, ast, earth, sfb, prog, opts, **k:
                        (seen.update(opts), {})[1])
    sv.solve_from_cell(rc, {"a": 0.922, "pdes": "99942"}, 10500.0, 10900.0,
                       available_power_W=5200.0)
    assert seen["available_power_W"] == pytest.approx(5200.0)


# ---------------------------------------------------------------------------
# rebuilding a swept point
# ---------------------------------------------------------------------------

def test_a_swept_point_rebuilds_into_the_same_result_shape(rc, stub_outbound, monkeypatch):
    """A sweep keeps only each point's decision vector; the winner must come back as the same
    dict an interactive solve produces, or the app cannot show it."""
    monkeypatch.setattr(sv.sf, "rebuild_for_config",
                        lambda rc_, target, x, **kw: _Sol())
    monkeypatch.setattr(sv, "_solve_return", lambda *a, **k: {"ok": True})
    out = sv.result_from_decision(rc, {"a": 0.922, "pdes": "99942"}, [1.0, 2.0],
                                  return_decision_vector=[3.0])
    assert set(out) >= {"sf", "target", "lambert", "return"}
    assert out["return"] == {"ok": True}


def test_a_rebuild_falls_back_to_the_solution_when_the_cell_has_no_lambert(
        rc, stub_outbound, monkeypatch):
    """The seed-baseline block is nice-to-have. When the solved cell has no impulsive geometry,
    the solution's own v-infinity split stands in rather than the block going missing."""
    monkeypatch.setattr(sv.sf, "rebuild_for_config", lambda rc_, target, x, **kw: _Sol())
    monkeypatch.setattr(sv.lb, "lambert_transfer", lambda *a, **k: None)
    monkeypatch.setattr(sv, "_finish", lambda *a, **k: {
        "sf": {**_outbound_block(), "vinf_dep_kms": 1.5, "vinf_arr_kms": 0.3}, "target": {}})
    out = sv.result_from_decision(rc, {"a": 0.922, "pdes": "99942"}, [1.0])
    assert out["lambert"]["best_dv_kms"] == pytest.approx(1.8)     # 1.5 + 0.3
    assert out["lambert"]["c3_km2s2"] == pytest.approx(1.5 ** 2)


def test_numpy_arrays_in_a_decision_vector_are_accepted(rc, stub_outbound, monkeypatch):
    """Sweep points store decision vectors as numpy arrays; rebuilding must not care."""
    monkeypatch.setattr(sv.sf, "rebuild_for_config", lambda rc_, target, x, **kw: _Sol())
    monkeypatch.setattr(sv, "_solve_return", lambda *a, **k: {})
    out = sv.result_from_decision(rc, {"a": 0.922, "pdes": "99942"}, np.array([1.0, 2.0]))
    assert "sf" in out
