"""The design search's decision logic: what counts as a vehicle that works.

These are the numbers that drive a vehicle choice, and they were the least-covered code in the
library; the argument parsing was tested, the verdict was not. Nothing here runs a solve: the sweep
is replaced with synthetic points so the verdict arithmetic can be exercised deterministically,
which is the part that can be wrong without anything crashing.
"""
from __future__ import annotations

import math
from datetime import date

import pytest

from prospector.config import EngineMount, ResolvedConfig, Screening, Vehicle, load_mission
from prospector.constants import G0_KM_S2
from prospector.spacecraft.propulsion import load_engines
from prospector.trades.design_search import evaluate as ev
from tests import library as lib


def _config(dry: float = 250.0, prop: float = 325.0, n: int = 4,
            return_trip: bool = False) -> ResolvedConfig:
    # Set the trip shape explicitly in both directions: the library's mission may itself be a round
    # trip, and only ever ADDING the flag left "one-way" cases still flying a return.
    # The mission shape these expectations were measured against: a launcher-provided escape
    # over a four-year window (the former direct Dawn example), whichever mission the library
    # lists first today.
    mission = load_mission(lib.mission_key()).model_copy(update={
        "launch_orbit": "ESCAPE", "departure_vinf_kms": None,
        "launch_window": (date(2007, 9, 26), date(2007, 10, 15)), "arrive_by": date(2011, 7, 16)})
    if return_trip:
        mission = mission.model_copy(update={
            "return_trip": True, "return_by": mission.arrive_by.replace(year=2032)})
    else:
        mission = mission.model_copy(update={"return_trip": False, "return_by": None})
    return ResolvedConfig.build(
        mission,
        Vehicle(name="probe", dry_mass=dry, fuel_mass=prop, unusable_prop=0.0,
                engines=[EngineMount(type=lib.engine_key(), count=n)]),
        Screening(dv_margin_factor=1.0), load_engines())


def _patch_sweep(monkeypatch, points):
    """Replace the Sims-Flanagan sweep with fixed points, so the verdict is the only variable."""
    from prospector.trades import sweep
    monkeypatch.setattr(sweep, "run_sweep", lambda *a, **k: list(points))


def _combo(rc, monkeypatch, points):
    _patch_sweep(monkeypatch, points)
    return ev.evaluate_combo(rc, {"pdes": "99942"}, {}, [0.0], {}, workers=1)


def _point(**kw):
    """A sweep point with the keys the verdict reads; overrides win."""
    return {"vinf_kms": 0.0, "feasible": True, "mismatch": 1e-9,
            "total_dv_kms": 5.0, "cruise_dv_kms": 4.0, "escape_dv_kms": 1.0,
            "tof_days": 400.0, **kw}


# ---------------------------------------------------------------------------
# what counts as closing
# ---------------------------------------------------------------------------

def test_a_converged_point_within_capability_closes(monkeypatch):
    rc = _config()
    out = _combo(rc, monkeypatch, [_point(total_dv_kms=rc.total_dv_capability - 0.5)])
    assert out["success"] is True
    assert out["margin_kms"] == pytest.approx(0.5)
    assert out["why"] == "ok"


def test_a_point_over_capability_does_not_close_and_reports_the_shortfall(monkeypatch):
    """A near miss must report how short it is: the shortfall is a first-class output, which is
    why the solve is not tank-limited."""
    rc = _config()
    out = _combo(rc, monkeypatch, [_point(total_dv_kms=rc.total_dv_capability + 0.75)])
    assert out["success"] is False
    assert out["margin_kms"] == pytest.approx(-0.75)
    assert out["dv_short_kms"] == pytest.approx(0.75)
    assert out["why"] == "over budget"


def test_an_unconverged_point_is_not_a_verdict(monkeypatch):
    """A leg whose matchpoint did not close is not a trajectory, however good its dV looks."""
    rc = _config()
    out = _combo(rc, monkeypatch,
                 [_point(feasible=False, mismatch=1.0, total_dv_kms=1.0)])
    assert out["success"] is False
    assert out["n_converged"] == 0
    assert out["why"] == "SF did not converge"


def test_convergence_keys_on_the_mismatch_not_the_boolean(monkeypatch):
    """The feasibility flag sits at a tolerance that threaded finite-difference gradients can
    jitter across; a closed matchpoint counts even when the flag says otherwise."""
    rc = _config()
    out = _combo(rc, monkeypatch,
                 [_point(feasible=False, mismatch=ev.MISMATCH_TOL / 10,
                         total_dv_kms=rc.total_dv_capability - 1.0)])
    assert out["n_converged"] == 1 and out["success"] is True


def test_the_headline_is_the_best_converged_point(monkeypatch):
    """Several v-infinity points are swept; the verdict quotes the one with the most margin, so
    one bad starting point does not decide the combination."""
    rc = _config()
    cap = rc.total_dv_capability
    out = _combo(rc, monkeypatch, [
        _point(vinf_kms=0.0, total_dv_kms=cap - 0.1),
        _point(vinf_kms=1.0, total_dv_kms=cap - 1.2),      # the best split
        _point(vinf_kms=2.0, feasible=False, mismatch=5.0, total_dv_kms=cap - 3.0),
    ])
    assert out["margin_kms"] == pytest.approx(1.2)
    assert out["best_point"]["vinf_kms"] == 1.0
    assert out["n_converged"] == 2                          # the unconverged one is excluded


def test_a_window_too_short_for_any_point_says_so(monkeypatch):
    rc = _config()
    out = _combo(rc, monkeypatch, [_point(total_dv_kms=None, error="window too short")])
    assert out["success"] is False and out["why"] == "window too short"


# ---------------------------------------------------------------------------
# the propellant check
# ---------------------------------------------------------------------------

def test_propellant_needed_is_the_rocket_equation_at_the_solved_total(monkeypatch):
    rc = _config()
    total_dv = 3.0
    out = _combo(rc, monkeypatch, [_point(total_dv_kms=total_dv)])
    expected = rc.vehicle.wet_mass * (1.0 - math.exp(-total_dv / (rc.effective_isp * G0_KM_S2)))
    assert out["best_point"]["prop_needed_kg"] == pytest.approx(expected)
    assert out["prop_margin_kg"] == pytest.approx(rc.vehicle.fuel_mass - expected)


def test_a_return_leg_adds_its_own_propellant_to_the_tank_check(monkeypatch):
    """A round trip must carry the return's draw too, or a design 'closes' on the outbound and
    strands itself at the target."""
    rc = _config(return_trip=True)
    total_dv = 2.0
    base = _combo(_config(), monkeypatch, [_point(total_dv_kms=total_dv)])
    with_ret = _combo(rc, monkeypatch,
                      [_point(total_dv_kms=total_dv, return_propellant_kg=60.0,
                              return_feasible=True)])
    assert (with_ret["best_point"]["prop_needed_kg"]
            == pytest.approx(base["best_point"]["prop_needed_kg"] + 60.0))


def test_a_round_trip_that_cannot_fly_home_does_not_close(monkeypatch):
    rc = _config(return_trip=True)
    out = _combo(rc, monkeypatch,
                 [_point(total_dv_kms=rc.total_dv_capability - 1.0,
                         return_propellant_kg=10.0, return_feasible=False)])
    assert out["success"] is False


def test_a_one_way_mission_ignores_return_fields(monkeypatch):
    """Success for a one-way mission keys on the outbound margin alone; a stray return flag from
    an older artifact must not change the verdict."""
    rc = _config(return_trip=False)
    out = _combo(rc, monkeypatch,
                 [_point(total_dv_kms=rc.total_dv_capability - 1.0, return_feasible=False)])
    assert out["success"] is True


def test_a_launch_vehicle_escape_flies_no_spiral(tmp_path):
    """A direct-escape mission must not be charged for a spiral it never flies.

    ``spiral_curve`` reaches the propagator through the low-level entry point, which does not carry
    the ``escape_provided`` guard that ``spiral.solve_for_config`` has. Without the short-circuit,
    a direct-escape study propagated a full SEP climb out of the injection orbit for every distinct
    wet mass and then charged its delta-v -- an escape the launch vehicle had already paid for.
    That cost both the run time and the answer.
    """
    from prospector.config import EngineMount, Mission, ResolvedConfig, Screening, Vehicle
    from prospector.spacecraft.propulsion import load_engines
    from prospector.trades.design_search.evaluate import spiral_curve

    rc = ResolvedConfig.build(
        Mission(launch_orbit="ESCAPE"),
        Vehicle(name="v", dry_mass=500.0, fuel_mass=400.0,
                engines=[EngineMount(type=lib.engine_key(), count=4)]),
        Screening(), load_engines())
    assert rc.launch.escape_provided

    curve = spiral_curve(rc, duty=0.9, vinf_max=2.0, cache_dir=tmp_path)
    assert curve["dv_at_escape_kms"] == 0.0 and curve["tof_at_escape_days"] == 0.0
    assert curve["status"] == "escaped"
    assert curve["belt_days"] == 0.0                     # never crosses the belts under thrust
    assert curve["power_fraction_end"] == 1.0            # so the array arrives pristine
    assert not curve["power_limited"]
    # The tradeoff curve is flat at zero: the launch vehicle hands over the excess speed, so every
    # cap on it costs the spacecraft nothing.
    assert set(curve["curve_dv_kms"]) == {0.0}
    assert max(curve["curve_vinf_kms"]) == 2.0
    # Nothing was propagated, so nothing was cached.
    assert not list(tmp_path.glob("spiral_*.json"))


def test_the_analytic_bound_actually_computes_a_cruise_floor():
    """The bound has to include a cruise floor. With the floor at 0.0 the bound is the escape
    estimate alone, which proves nothing about the cruise, and a bare ``except`` around the
    calculation is enough to leave it there without anything failing.

    The failure was in the SAFE direction, which is exactly why it survived: a rejection bound that
    never fires looks identical to a design space with nothing provably dead in it.
    """
    from prospector.launch import escape_dv_estimate
    from prospector.solvers import impulse_margin as im
    from prospector.solvers.edelbaum import lowthrust_dv

    rc = _config()
    row = {"a": 1.1026, "e": 0.1914, "i": 3.34, "pdes": "99942", "full_name": "Apophis"}
    vinf_grid = [0.0, 1.0, 2.0]
    out = ev.analytic_row(rc, row, vinf_grid, 0.9)

    assert out["bound_kms"] is not None
    # The bound must exceed the escape estimate at its chosen v-infinity: the difference is the
    # cruise floor, and its absence is what the dead call produced.
    esc = escape_dv_estimate(rc.launch, out["bound_vinf_kms"])
    assert out["bound_kms"] > esc + 1e-6, "the cruise floor is missing from the bound"

    element = float(lowthrust_dv(row["a"], row["e"], row["i"]))
    expected_floor = max(0.0, im.CERTIFIED_MARGIN * element - out["bound_vinf_kms"])
    assert out["bound_kms"] - esc == pytest.approx(expected_floor, abs=1e-9)
    # And it is held below the element estimate: the verdict is rejection-grade, so the term must
    # under-charge or the bound could kill a design that flies.
    assert im.CERTIFIED_MARGIN < 1.0
    assert out["bound_kms"] - esc < element


def test_the_analytic_bound_never_rejects_a_design_that_flies():
    """The bound's only verdict is rejection, so a false positive costs a viable vehicle. Apophis on
    this class of bus converges around 2.9 km/s in a real solve, so the bound must let it through --
    the previous surface's own rejection verdict fired on 16, 21 and 18 cells that fly across three
    reference targets, which is the failure this replaces."""
    rc = _config()
    row = {"a": 1.1026, "e": 0.1914, "i": 3.34, "pdes": "99942", "full_name": "Apophis"}
    out = ev.analytic_row(rc, row, [0.0, 1.0, 2.0], 0.9)
    assert out["provably_infeasible"] is False
    assert out["bound_margin_kms"] > 0.0

    # A vehicle with almost no capability is provably dead, so the bound is not merely inert.
    starved = _config(dry=250.0, prop=5.0, n=1)
    dead = ev.analytic_row(starved, row, [0.0, 1.0, 2.0], 0.9)
    if dead["bound_kms"] is not None:
        assert dead["provably_infeasible"] is True, dead


def test_a_positive_dv_margin_with_a_short_tank_does_not_close(monkeypatch):
    """The escape and the cruise run at different Isps, so the single-Isp delta-v margin can read
    positive while the propellant actually spent overfills the tank. The verdict is the kilograms;
    the delta-v margin is reported but does not decide. This is a case from a real study: +0.19
    km/s on paper, 3.5 kg short in the tank."""
    rc = _config()
    usable = rc.usable_propellant_kg
    out = _combo(rc, monkeypatch, [_point(total_dv_kms=rc.total_dv_capability - 0.2,
                                          propellant_kg=usable + 3.5)])
    assert out["margin_kms"] == pytest.approx(0.2)
    assert out["prop_margin_kg"] == pytest.approx(-3.5)
    assert out["success"] is False and out["why"] == "tank short"
    # With the propellant inside the tank the same point closes.
    out = _combo(rc, monkeypatch, [_point(total_dv_kms=rc.total_dv_capability - 0.2,
                                          propellant_kg=usable - 3.5)])
    assert out["success"] is True and out["prop_margin_kg"] == pytest.approx(3.5)


def test_the_headline_point_leaves_the_most_propellant(monkeypatch):
    """Two converged points: the one with the smaller delta-v margin but more propellant left is
    the headline, since propellant is what the verdict is judged on."""
    rc = _config()
    usable = rc.usable_propellant_kg
    cap = rc.total_dv_capability
    out = _combo(rc, monkeypatch, [
        _point(vinf_kms=0.0, total_dv_kms=cap - 1.0, propellant_kg=usable - 2.0),
        _point(vinf_kms=0.5, total_dv_kms=cap - 0.5, propellant_kg=usable - 9.0),
    ])
    assert out["best_point"]["vinf_kms"] == 0.5
    assert out["prop_margin_kg"] == pytest.approx(9.0) and out["margin_kms"] == pytest.approx(0.5)


def test_an_unsettled_point_does_not_close(monkeypatch):
    """A cruise whose thrust ceilings did not settle on its own path still fires where the array
    cannot power it, so its propellant reads low. It is not a closed design until a re-solve
    settles it, however good its kilograms look."""
    rc = _config()
    usable = rc.usable_propellant_kg
    out = _combo(rc, monkeypatch, [_point(total_dv_kms=rc.total_dv_capability - 0.5,
                                          propellant_kg=usable - 5.0, settled=False)])
    assert out["success"] is False and out["why"] == "Isp unsettled"
    assert out["prop_margin_kg"] == pytest.approx(5.0)     # the kilograms are still reported
    out = _combo(rc, monkeypatch, [_point(total_dv_kms=rc.total_dv_capability - 0.5,
                                          propellant_kg=usable - 5.0, settled=True)])
    assert out["success"] is True
