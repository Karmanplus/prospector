"""Tests for discrete throttle modes: a thruster that hard-switches between the points of its curve
rather than sliding along it (``Engine.throttle_mode == "discrete"``).

Also pins the two things a mode switch must move (the spiral fingerprint and the trajectory
signature), the thing it must NOT move (either of those for a continuous stack, so runs already on
disk stay valid), and that a propellant swap scales the curve along with the rated point.
"""
import numpy as np
import pytest

from prospector.config import EngineMount, Mission, ResolvedConfig, Screening, Vehicle
from prospector.figures import products
from prospector.launch import LaunchOrbit, discrete_modes, escape_fingerprint
from prospector.spacecraft.propellants import engine_on_propellant, load_propellants
from prospector.spacecraft.propulsion import (
    Engine,
    PowerPoint,
    assembly_performance_at_power,
    assembly_power_grid,
    load_engines,
    save_engine,
)

# Round invented numbers (see test_propulsion.CURVE); the modes are 400, 500 and 1000 W.
CURVE = [PowerPoint(power_W=400, thrust_mN=24.0, isp_s=1300.0),
         PowerPoint(power_W=500, thrust_mN=30.0, isp_s=1350.0),
         PowerPoint(power_W=1000, thrust_mN=60.0, isp_s=1500.0)]
SMOOTH = Engine(name="S", isp_s=1500, thrust_mN=60.0, power_W=1000, power_curve=CURVE)
STEPPED = SMOOTH.model_copy(update={"name": "D", "throttle_mode": "discrete"})
CAT = {"S": SMOOTH, "D": STEPPED}


def test_default_is_continuous_and_old_files_load_unchanged():
    assert Engine(name="x", isp_s=1, thrust_mN=1).throttle_mode == "continuous"
    assert not SMOOTH.discrete and STEPPED.discrete


def test_discrete_runs_the_highest_mode_the_power_allows():
    assert STEPPED.performance_at(1000) == (60.0, 1500.0)      # top mode at rated
    assert STEPPED.performance_at(1500) == (60.0, 1500.0)      # nothing above rated
    assert STEPPED.performance_at(750) == (30.0, 1350.0)       # rounds DOWN to 500 W, no blend
    assert STEPPED.performance_at(999.9) == (30.0, 1350.0)     # a hair short of a mode is the one below
    assert STEPPED.performance_at(500) == (30.0, 1350.0)       # exactly on a mode runs it
    assert STEPPED.performance_at(450) == (24.0, 1300.0)
    thrust, _ = STEPPED.performance_at(399)                    # below the lowest mode: off
    assert thrust == 0.0
    # The same curve read continuously blends between the points.
    thrust, isp = SMOOTH.performance_at(750)
    assert 30.0 < thrust < 60.0 and 1350 < isp < 1500


def test_a_discrete_thruster_without_a_curve_is_on_or_off():
    e = Engine(name="onoff", isp_s=2000, thrust_mN=100, power_W=1000, throttle_mode="discrete")
    assert e.modes() == [PowerPoint(power_W=1000, thrust_mN=100, isp_s=2000)]
    assert e.performance_at(1000) == (100.0, 2000.0)
    assert e.performance_at(999)[0] == 0.0
    assert e.min_power_W == 1000
    # A stack of them sheds whole engines instead of throttling linearly.
    r = assembly_performance_at_power([("onoff", 4)], {"onoff": e}, 2500.0)
    assert r["n_active"] == 2 and r["thrust_mN"] == pytest.approx(200.0)


def test_a_discrete_stack_runs_the_most_engines_each_rounded_down():
    mounts = [("D", 4)]                                         # 4 x 1000 W rated
    full = assembly_performance_at_power(mounts, CAT, 5000.0)
    assert full["thrust_mN"] == pytest.approx(240.0) and full["n_active"] == 4
    # 2700 W: all four can reach the 400 W floor (675 W each), and 675 rounds down to the 500 W
    # mode. 4 x 30 mN at 1350 s, with 700 W left unused rather than blended in.
    r = assembly_performance_at_power(mounts, CAT, 2700.0)
    assert r["n_active"] == 4
    assert r["thrust_mN"] == pytest.approx(120.0) and r["isp_s"] == pytest.approx(1350.0)
    # The continuous stack on the same power blends to something in between.
    c = assembly_performance_at_power([("S", 4)], CAT, 2700.0)
    assert 120.0 < c["thrust_mN"] < 240.0 and 1350.0 < c["isp_s"] < 1500.0
    # 1500 W: only three engines reach the floor (500 each), all at the 500 W mode.
    r = assembly_performance_at_power(mounts, CAT, 1500.0)
    assert r["n_active"] == 3 and r["thrust_mN"] == pytest.approx(90.0)
    # 4 x 1000 W exactly is rated; 3999 W is four engines at 999 W -> the 500 W mode each.
    r = assembly_performance_at_power(mounts, CAT, 3999.0)
    assert r["thrust_mN"] == pytest.approx(120.0)


def test_an_off_stack_reports_the_floor_isp_not_the_rated_one():
    """Below the floor nothing fires. The Isp reported alongside the zero thrust must not be the
    rated figure: on the mission timeline it drew as an Isp step UP exactly where the array had
    fallen too far to run the thruster at all, which read as the engine getting more efficient."""
    for key in ("D", "S"):
        off = assembly_performance_at_power([(key, 2)], CAT, 300.0)
        assert off["n_active"] == 0 and off["thrust_mN"] == 0.0
        assert off["isp_s"] == 1300.0                           # the floor's Isp, not 1500
    # And the grid, read at a power below the floor, says the same.
    pw, thrust_N, isp_s = assembly_power_grid([("D", 2)], CAT)
    assert np.interp(300.0, pw, thrust_N) == 0.0
    assert np.interp(300.0, pw, isp_s) == pytest.approx(1300.0)


def test_the_timeline_leaves_a_gap_in_isp_where_the_stack_is_off():
    from ui.workspaces import flight
    rc = _rc(STEPPED)                                           # 2 x D, floor 400 W each
    thrust_mN, isp = flight._grid_thrust_isp(rc, np.array([300.0, 1000.0, 2000.0]))
    assert thrust_mN[0] == 0.0 and np.isnan(isp[0])             # off: no operating point to draw
    assert thrust_mN[1] == pytest.approx(60.0) and isp[1] == pytest.approx(1350.0)
    assert thrust_mN[2] == pytest.approx(120.0) and isp[2] == pytest.approx(1500.0)


def test_a_mixed_stack_switches_off_a_discrete_type_below_its_lowest_mode():
    # S (continuous) + D (discrete), 2000 W rated. At 700 W each type gets 35% of its rated power,
    # 350 W: the continuous engine clamps to its 400 W floor, the discrete one is off.
    r = assembly_performance_at_power([("S", 1), ("D", 1)], CAT, 700.0)
    assert r["thrust_mN"] == pytest.approx(24.0)
    assert r["n_active"] == 2                                   # the mixed path never sheds


def test_the_power_grid_keeps_the_steps():
    """The grid is read by interpolation. For a discrete stack every mode change must be a pair of
    nodes a hair apart, or the step is smeared over the gap between two evenly spaced samples."""
    pw, thrust_N, isp_s = assembly_power_grid([("D", 2)], CAT, points=16)
    assert pw[0] == 0.0 and pw[-1] == pytest.approx(2000.0)
    # Just below two engines reaching the 500 W mode (1000 W) the stack is 2 x 24 mN; at it, 2 x 30.
    assert np.interp(1000.0 * (1 - 1e-9), pw, thrust_N) == pytest.approx(0.048, rel=1e-6)
    assert np.interp(1000.0, pw, thrust_N) == pytest.approx(0.060, rel=1e-6)
    assert np.interp(1000.0 * (1 - 1e-9), pw, isp_s) == pytest.approx(1300.0)
    assert np.interp(1000.0, pw, isp_s) == pytest.approx(1350.0)
    # In between two breakpoints the reading is flat, not a ramp toward the next mode.
    assert np.interp(1400.0, pw, thrust_N) == pytest.approx(0.060, rel=1e-6)
    assert np.interp(1900.0, pw, thrust_N) == pytest.approx(0.060, rel=1e-6)
    # A continuous stack's grid is the plain evenly spaced sample, exactly as before.
    pw_c, _, _ = assembly_power_grid([("S", 2)], CAT, points=16)
    assert len(pw_c) == 16


def test_throttle_mode_round_trips_through_the_library(tmp_path):
    save_engine(STEPPED, "stepped", tmp_path)
    save_engine(SMOOTH, "smooth", tmp_path)
    back = load_engines(tmp_path)
    assert back["stepped"].throttle_mode == "discrete" and back["stepped"].discrete
    assert back["smooth"].throttle_mode == "continuous"
    assert "throttle_mode: discrete" in (tmp_path / "stepped.yaml").read_text()


# ---------------------------------------------------------------------------
# what a mode switch invalidates, and what it leaves alone
# ---------------------------------------------------------------------------

_LEO = LaunchOrbit(name="leo", perigee_alt_km=400, apogee_alt_km=400, inclination_deg=28.5)


def _rc(engine: Engine) -> ResolvedConfig:
    cat = {"E": engine.model_copy(update={"name": "E"})}
    veh = Vehicle(name="v", dry_mass=500, fuel_mass=300, solar_power_W=1500,
                  engines=[EngineMount(type="E", count=2)])
    return ResolvedConfig.build(Mission(launch_orbit="LEO"), veh, Screening(), cat,
                                launches={"LEO": _LEO})


def test_continuous_fingerprint_and_signature_are_exactly_what_they_were():
    """Runs on disk were fingerprinted before modes existed; a continuous stack must still match
    them, so the mode terms appear only when a thruster is discrete."""
    rc = _rc(SMOOTH)
    assert discrete_modes(rc) == {}
    assert "discrete_modes" not in escape_fingerprint(rc)
    assert len(products.trajectory_signature(rc, 1000.0)) == 14


def test_switching_a_thruster_to_discrete_moves_both_staleness_keys():
    smooth, stepped = _rc(SMOOTH), _rc(STEPPED)
    assert discrete_modes(stepped) == {"E": [[400.0, 24.0, 1300.0], [500.0, 30.0, 1350.0],
                                             [1000.0, 60.0, 1500.0]]}
    assert escape_fingerprint(smooth) != escape_fingerprint(stepped)
    assert escape_fingerprint(stepped) == escape_fingerprint(_rc(STEPPED))       # deterministic
    assert products.trajectory_signature(smooth, 1000.0) != products.trajectory_signature(stepped, 1000.0)
    assert products.trajectory_signature(stepped, 1000.0) == products.trajectory_signature(_rc(STEPPED), 1000.0)


# ---------------------------------------------------------------------------
# the curve follows the rated point onto another gas
# ---------------------------------------------------------------------------

def test_propellant_swap_scales_the_curve_with_the_rated_point():
    cat = load_propellants()
    kr, estimated = engine_on_propellant(SMOOTH, "krypton", cat)
    assert estimated
    ft, fi = cat["krypton"].thrust_scale, cat["krypton"].isp_scale
    assert kr.thrust_mN == pytest.approx(60.0 * ft) and kr.isp_s == pytest.approx(1500.0 * fi)
    assert [pp.power_W for pp in kr.power_curve] == [400.0, 500.0, 1000.0]   # power untouched
    assert [pp.thrust_mN for pp in kr.power_curve] == pytest.approx([24.0 * ft, 30.0 * ft, 60.0 * ft])
    assert [pp.isp_s for pp in kr.power_curve] == pytest.approx([1300.0 * fi, 1350.0 * fi, 1500.0 * fi])
    # So the curve's top IS the rated point, and there is no jump at rated power.
    assert kr.performance_at(kr.power_W) == pytest.approx((kr.thrust_mN, kr.isp_s))
    # A curve-less engine still scales as before.
    bare, _ = engine_on_propellant(Engine(name="b", isp_s=2000, thrust_mN=100, power_W=1000),
                                   "krypton", cat)
    assert bare.power_curve is None and bare.thrust_mN == pytest.approx(100 * ft)
