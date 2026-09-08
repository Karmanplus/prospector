"""Tests for the propulsion layer (prospector.spacecraft.propulsion): rocket equation + engine library."""
import math

import pytest
from pydantic import ValidationError

from prospector.constants import G0_KM_S2
from prospector.spacecraft.propulsion import (
    Engine,
    PowerPoint,
    assembly_performance,
    assembly_performance_at_power,
    assembly_power_by_ppu_side,
    assembly_power_grid,
    list_engines,
    load_engines,
    save_engine,
    tsiolkovsky_dv,
)

CAT = {
    "A": Engine(name="A", isp_s=2000, thrust_mN=100, power_W=1000, mass_kg=5),
    "B": Engine(name="B", isp_s=4000, thrust_mN=200, power_W=2000, mass_kg=10),
}

# A curved engine: thrust and Isp fall together with input power. Round invented numbers, chosen so
# the interpolation and the shedding thresholds are checkable by eye, a measured datasheet curve
# would tie the suite to one vendor's performance data.
CURVE = [PowerPoint(power_W=400, thrust_mN=24.0, isp_s=1300.0),
         PowerPoint(power_W=500, thrust_mN=30.0, isp_s=1350.0),
         PowerPoint(power_W=1000, thrust_mN=60.0, isp_s=1500.0)]
CURVED = {"C": Engine(name="C", isp_s=1500, thrust_mN=60.0, power_W=1000, power_curve=CURVE)}


def test_tsiolkovsky_matches_rocket_equation():
    dv = tsiolkovsky_dv(isp_s=2000, wet_mass_kg=1000, dry_mass_kg=500)
    assert math.isclose(dv, G0_KM_S2 * 2000 * math.log(2.0), rel_tol=1e-12)


def test_no_usable_propellant_gives_zero_dv():
    assert tsiolkovsky_dv(2000, 500, 500) == 0.0


def test_single_type_assembly_keeps_its_isp():
    perf = assembly_performance([("A", 3)], CAT)
    assert perf["isp_s"] == 2000
    assert perf["thrust_mN"] == 300
    assert perf["power_W"] == 3000
    assert perf["mass_kg"] == 15


def test_mixed_assembly_blends_isp_thrust_weighted():
    # eff Isp = sum(F) / sum(F/Isp); bounded by the two engines' Isp values.
    perf = assembly_performance([("A", 1), ("B", 1)], CAT)
    expected = (100 + 200) / (100 / 2000 + 200 / 4000)
    assert math.isclose(perf["isp_s"], expected, rel_tol=1e-12)
    assert 2000 < perf["isp_s"] < 4000
    assert perf["thrust_mN"] == 300


def test_unknown_engine_key_raises():
    with pytest.raises(KeyError):
        assembly_performance([("Z", 1)], CAT)


def test_power_curve_interpolation_and_clamping():
    e = CURVED["C"]
    assert e.performance_at(1000) == (60.0, 1500.0)       # rated point
    assert e.performance_at(1500) == (60.0, 1500.0)       # clamped above rated
    assert e.performance_at(200) == (24.0, 1300.0)        # clamped below the operating floor
    thrust, isp = e.performance_at(750)                   # interpolated between 500 and 1000
    assert 30.0 < thrust < 60.0 and 1350 < isp < 1500
    assert e.min_power_W == 400
    # A curve-less engine is flat at its rated point at any power.
    assert CAT["A"].performance_at(300) == (100.0, 2000.0)


def test_stack_power_limited_uses_curve_and_sheds():
    mounts = [("C", 4)]                                    # rated 4000 W, 240 mN, 1500 s
    assert assembly_performance_at_power(mounts, CURVED, 5000)["thrust_mN"] == pytest.approx(240.0)
    mid = assembly_performance_at_power(mounts, CURVED, 2000)   # 4 engines at 500 W each
    assert mid["n_active"] == 4 and mid["isp_s"] == pytest.approx(1350, rel=1e-3)
    assert mid["thrust_mN"] == pytest.approx(4 * 30.0, rel=1e-3)
    shed = assembly_performance_at_power(mounts, CURVED, 1000)  # only 2 engines fit above 400 W
    assert shed["n_active"] == 2
    assert assembly_performance_at_power(mounts, CURVED, 300)["n_active"] == 0   # none can fire


def test_stack_power_limited_no_curve_is_linear_constant_isp():
    # Legacy engines: thrust scales linearly with power, Isp unchanged.
    r = assembly_performance_at_power([("A", 2)], CAT, 1000.0)   # half of 2000 W rated
    assert r["isp_s"] == 2000.0 and r["thrust_mN"] == pytest.approx(100.0)


def test_power_grid_spans_zero_to_rated():
    pw, thrust_N, isp_s = assembly_power_grid([("C", 4)], CURVED)
    assert pw[0] == 0.0 and pw[-1] == pytest.approx(4000.0)
    assert thrust_N[-1] == pytest.approx(0.240) and isp_s[-1] == pytest.approx(1500.0)
    assert thrust_N[0] == 0.0                              # no power -> no thrust


def test_engine_library_is_file_based(tmp_path):
    assert load_engines(tmp_path) == {}          # empty dir -> empty catalog (nothing hardcoded)
    save_engine(Engine(name="Custom", isp_s=3000, thrust_mN=150), "custom", tmp_path)
    cat = load_engines(tmp_path)
    assert list_engines(tmp_path) == ["custom"]
    assert cat["custom"].isp_s == 3000


def test_shipped_engine_library_loads():
    """The active library loads and every engine is usable. Run against the example library (see
    conftest), so this also asserts the shipped examples are self-consistent."""
    cat = load_engines()
    assert cat, "the active engine library is empty"
    for key, e in cat.items():
        assert e.isp_s > 0 and e.thrust_mN > 0, key


def test_every_engine_records_a_provenance():
    """No engine may leave its source blank, in ANY library.

    Provenance is what decides whether numbers may be redistributed, and an engine with none
    recorded is indistinguishable after the fact from one supplied under an agreement, which
    forces the whole library to be treated as restricted."""
    unrecorded = [k for k, e in load_engines().items() if not str(e.source or "").strip()]
    assert not unrecorded, f"engines with no source recorded: {unrecorded}"


def test_the_shippable_example_library_is_fully_shareable():
    """The example library specifically must be safe to hand over in full.

    Checked against ``examples/configs`` directly rather than the active library, because the
    repository's own library legitimately holds restricted engines; that is the whole reason the
    example set exists. If this set ever contains something unshareable, the distributable library
    is empty and nobody finds out until distribution time.
    """
    from prospector import paths
    from prospector.spacecraft.propulsion import is_shareable

    example_dir = paths.REPO_ROOT / "examples" / "configs" / "engines"
    cat = load_engines(example_dir)
    assert cat, f"no example engines at {example_dir}"
    assert all(is_shareable(e) for e in cat.values()), (
        f"unshareable engines in the example library: "
        f"{[k for k, e in cat.items() if not is_shareable(e)]}")


def test_ppu_bus_side_survives_the_library_round_trip(tmp_path):
    # The wiring sizes the solar array, so it has to persist with the engine rather than being
    # re-guessed on load.
    save_engine(Engine(name="High side", isp_s=3000, thrust_mN=150, power_W=4000,
                       ppu_bus_side="high"), "hv", tmp_path)
    assert load_engines(tmp_path)["hv"].ppu_bus_side == "high"


def test_unknown_ppu_bus_side_is_rejected():
    with pytest.raises(ValidationError):
        Engine(name="bad", isp_s=2000, thrust_mN=100, ppu_bus_side="medium")


def test_assembly_power_splits_by_ppu_side():
    """The array-sizing chain differs per side, so the assembly reports its power split rather
    than one total: a mixed stack draws off the array through two chains at once."""
    cat = {
        "lo": Engine(name="lo", isp_s=2000, thrust_mN=100, power_W=1000, ppu_bus_side="low"),
        "hi": Engine(name="hi", isp_s=2000, thrust_mN=100, power_W=1500, ppu_bus_side="high"),
        "off": Engine(name="off", isp_s=2000, thrust_mN=100, power_W=0.0),
    }
    assert assembly_power_by_ppu_side([("lo", 2), ("hi", 2)], cat) == {"low": 2000.0, "high": 3000.0}
    assert assembly_power_by_ppu_side([("lo", 3)], cat) == {"low": 3000.0}
    # A side that draws nothing gets no entry, so an empty split means a stack with no demand.
    assert assembly_power_by_ppu_side([("off", 4)], cat) == {}
