"""Pin the buildability/cost model.

The sizing chain makes three documented choices (worst-mode load, engine masses as full systems, no
mass-growth margin); these tests lock the numbers so a refactor can't silently drift the assessment
(the same discipline as the dV-estimate pins).
"""
import math

import pytest

from prospector.spacecraft.buildability import (
    SWEEPABLE,
    BusModel,
    apply_model_overrides,
    array_area_m2,
    array_rated_W,
    assess,
    assess_engine,
    load_bus_model,
    mission_eol_factor,
    required_bol_W,
    save_bus_model,
)
from prospector.spacecraft.propulsion import Engine


def test_bus_model_is_config_driven_and_round_trips(tmp_path):
    # The model is file-based: a saved model reloads identically.
    path = tmp_path / "build-model.yaml"
    tuned = BusModel(array_mass_scale=0.8, propellant_cost_per_kg=900.0)
    save_bus_model(tuned, path)
    assert load_bus_model(path) == tuned
    # An edited coefficient actually moves the assessment (a lighter array technology scale frees
    # payload). Compare two EXPLICIT models so the result doesn't depend on the live
    # configs/build-model.yaml.
    base = assess(dry_kg=300.0, prop_kg=350.0, n_engines=3, engine_power_W=5000.0,
                  engine_mass_kg=18.0, model=BusModel())
    lighter = assess(dry_kg=300.0, prop_kg=350.0, n_engines=3, engine_power_W=5000.0,
                     engine_mass_kg=18.0, model=tuned)
    assert lighter["array_kg"] < base["array_kg"]


def test_missing_build_model_raises_rather_than_using_field_defaults(tmp_path):
    """Every mass and cost this module reports is scaled by these coefficients, so a missing
    file must be an error; otherwise the assessment silently describes a different vehicle
    than the one configured."""
    with pytest.raises(FileNotFoundError, match="no build model"):
        load_bus_model(tmp_path / "not-there.yaml")


def test_a_partial_build_model_still_fills_missing_coefficients(tmp_path):
    """Within a file that EXISTS, a missing key falling back to its field default is safe and
    deliberate: the value is still the authored physics-of-record, and it lets an existing
    config survive a new coefficient being added. Unknown keys are ignored the same way."""
    path = tmp_path / "build-model.yaml"
    path.write_text("propellant_cost_per_kg: 900.0\nnot_a_real_coefficient: 12\n")
    m = load_bus_model(path)
    assert m.propellant_cost_per_kg == 900.0
    assert m.array_mass_scale == BusModel().array_mass_scale     # untouched field default


def test_sweepable_model_overrides_apply_only_to_model_fields():
    # The design-space search sweeps model settings by name; each "model"-kind key must be a real
    # BusModel field (else an override silently no-ops), and apply_model_overrides must move only
    # those fields, ignoring the duty knob the caller applies itself.
    assert all(key in BusModel.model_fields
               for key, (_, _, kind) in SWEEPABLE.items() if kind == "model")
    base = BusModel()
    tuned = apply_model_overrides(base, {"power_margin_pct": 35.0, "duty": 0.8})
    assert tuned.power_margin_pct == 35.0           # the model field moved
    assert base.power_margin_pct == 20.0            # the original is untouched (a copy)
    assert not hasattr(tuned, "duty")               # duty is not a model field, so ignored
    # A higher solar-array margin sizes a bigger array - the trade the sweep exposes.
    rich = apply_model_overrides(base, {"power_margin_pct": 40.0})
    lean = apply_model_overrides(base, {"power_margin_pct": 10.0})
    assert (required_bol_W(3, 5000.0, rich)
            > required_bol_W(3, 5000.0, lean))
    # No overrides -> the same model back (no needless copy churn in the hot search loop).
    assert apply_model_overrides(base, {}) is base


def test_array_mass_comes_from_the_configured_power_to_mass_curve():
    # The array line in an assessment is the piecewise power->mass curve from the build model (see
    # prospector.spacecraft.arrays), never a blanket W/kg, and it responds to the sweep knob.
    m = BusModel()
    out = assess(dry_kg=300.0, prop_kg=350.0, n_engines=3, engine_power_W=5000.0,
                 engine_mass_kg=18.0, model=m)
    bol = required_bol_W(3, 5000.0, m)
    # Read at the RATED wattage the operating power implies, since the curve is a catalogue curve
    # (see test_array_mass_and_area_are_charged_at_the_rated_wattage).
    assert out["array_kg"] == pytest.approx(m.array_model().mass_kg(array_rated_W(bol, m)), abs=0.01)
    scaled = assess(dry_kg=300.0, prop_kg=350.0, n_engines=3, engine_power_W=5000.0,
                    engine_mass_kg=18.0, model=BusModel(array_mass_scale=1.5))
    # abs, not rel: array_kg is rounded to 2 dp on the way out, so a 1e-6 relative compare is
    # tighter than the reported precision and passes only by luck.
    assert scaled["array_kg"] == pytest.approx(1.5 * out["array_kg"], abs=0.02)


def test_array_mass_and_area_are_charged_at_the_rated_wattage():
    # A vehicle's array power is what it delivers facing the Sun at 1 AU, where the panel runs
    # hotter than the 28 degC a datasheet quotes, so the array that delivers it must be RATED
    # higher. Mass (the vendor curve) and area (the areal power) are catalogue figures and are
    # charged at that rated wattage; rating the array at its own operating temperature removes the
    # derate entirely, which pins that the derate is the whole difference.
    m = BusModel()
    k = m.array_model().operating_over_rated()
    assert 0.85 < k < 1.0
    assert array_rated_W(5000.0, m) == pytest.approx(5000.0 / k)
    assert array_area_m2(5000.0, m) == pytest.approx(5000.0 / k / m.array_specific_power_W_m2)
    flat = BusModel(array_rating_temp_C=m.array_model().operating_temp_C())
    assert array_rated_W(5000.0, flat) == pytest.approx(5000.0)
    assert array_area_m2(5000.0, flat) == pytest.approx(5000.0 / m.array_specific_power_W_m2)
    rated = assess(dry_kg=300.0, prop_kg=350.0, n_engines=3, engine_power_W=5000.0,
                   engine_mass_kg=18.0, model=m)
    unrated = assess(dry_kg=300.0, prop_kg=350.0, n_engines=3, engine_power_W=5000.0,
                     engine_mass_kg=18.0, model=flat)
    assert rated["array_kg"] > unrated["array_kg"]
    bol = required_bol_W(3, 5000.0, m)
    assert unrated["array_kg"] == pytest.approx(m.array_model().mass_kg(bol), abs=0.01)
    # Zero power still means no array, whatever the rating.
    assert array_area_m2(0.0, m) == 0.0


def test_xenon_cost_matches_the_propellant_library():
    # The BusModel's xenon fallback cost must equal the library's xenon entry, or a gas-free
    # assessment prices differently from a xenon one.
    from prospector.spacecraft.propellants import load_propellants
    assert BusModel().propellant_cost_per_kg == pytest.approx(2500.0)
    assert load_propellants()["xenon"].cost_per_kg == pytest.approx(2500.0)


def test_power_chain_efficiencies():
    # Avionics run on transmission x HV->LV. A thruster additionally runs through its PPU, and
    # whether it ALSO pays the HV->LV converter depends on which bus its PPU is fed from: a
    # low-side PPU pays it, a high-side one taps the main bus and skips it. An unspecified side is
    # charged the deeper (low-side) chain.
    m = BusModel()
    assert m.avionics_power_eff() == pytest.approx(0.90 * 0.93, rel=1e-12)
    assert m.thruster_power_eff("low") == pytest.approx(0.90 * 0.93 * 0.90, rel=1e-12)
    assert m.thruster_power_eff("high") == pytest.approx(0.90 * 0.90, rel=1e-12)
    assert m.thruster_power_eff() == m.thruster_power_eff("low")
    assert m.thruster_power_eff("high") > m.thruster_power_eff("low")


def test_unknown_ppu_bus_side_raises():
    # A misspelled side must not silently price as one of the two: the array mass depends on it.
    with pytest.raises(ValueError, match="unknown PPU bus side"):
        BusModel().thruster_power_eff("medium")


def _ppu_side_catalog():
    """Two engines identical but for which bus their PPU is fed from."""
    from prospector.spacecraft.propulsion import Engine
    return {
        "lo": Engine(name="lo", isp_s=2000, thrust_mN=100, power_W=1000, ppu_bus_side="low"),
        "hi": Engine(name="hi", isp_s=2000, thrust_mN=100, power_W=1000, ppu_bus_side="high"),
    }


def test_engines_default_to_the_low_side_ppu():
    # The default must be the deeper chain, so an engine whose wiring nobody recorded is never
    # credited a conversion saving it may not have.
    from prospector.spacecraft.propulsion import Engine
    assert Engine(name="e", isp_s=2000, thrust_mN=100, power_W=1000).ppu_bus_side == "low"


def test_uniform_assembly_chain_matches_its_engines_side():
    from prospector.spacecraft.buildability import assembly_thruster_chain_eff
    m, cat = BusModel(), _ppu_side_catalog()
    assert assembly_thruster_chain_eff([("lo", 4)], cat, m) == pytest.approx(
        m.thruster_power_eff("low"), rel=1e-12)
    assert assembly_thruster_chain_eff([("hi", 4)], cat, m) == pytest.approx(
        m.thruster_power_eff("high"), rel=1e-12)


def test_mixed_ppu_sides_blend_conserving_array_demand():
    """The blend's defining property: a mixed stack's total power divided by the blended
    efficiency equals the sum of the per-side demands. That is what makes it the only correct
    blend; the array is sized to a demand, so an averaging rule that changed the demand would
    size the wrong array."""
    from prospector.spacecraft.buildability import assembly_thruster_chain_eff
    m, cat = BusModel(), _ppu_side_catalog()
    eff = assembly_thruster_chain_eff([("lo", 2), ("hi", 2)], cat, m)
    per_side = 2000.0 / m.thruster_power_eff("low") + 2000.0 / m.thruster_power_eff("high")
    assert 4000.0 / eff == pytest.approx(per_side, rel=1e-12)
    # It lands between the two chains, nearer the side carrying more power.
    assert m.thruster_power_eff("low") < eff < m.thruster_power_eff("high")
    lo_heavy = assembly_thruster_chain_eff([("lo", 3), ("hi", 1)], cat, m)
    assert lo_heavy < eff
    # A stack drawing no power has no demand to misprice; it reports the deeper chain.
    from prospector.spacecraft.propulsion import Engine
    passive = {"p": Engine(name="p", isp_s=2000, thrust_mN=100, power_W=0.0)}
    assert assembly_thruster_chain_eff([("p", 2)], passive, m) == pytest.approx(
        m.thruster_power_eff("low"), rel=1e-12)


def test_high_side_ppu_sizes_a_smaller_array():
    """The point of the flag: skipping the step-down converter cuts the thrusters' array-referred
    demand, so the same stack needs less panel. Everything else about the two vehicles is equal."""
    from prospector.spacecraft.buildability import assess_engine
    m, cat = BusModel(), _ppu_side_catalog()
    low = assess_engine(dry_kg=300.0, prop_kg=350.0, n_engines=4, engine=cat["lo"], model=m)
    high = assess_engine(dry_kg=300.0, prop_kg=350.0, n_engines=4, engine=cat["hi"], model=m)
    assert high["bol_power_W"] < low["bol_power_W"]
    # The saving is exactly the converter on the thruster share of the load, ~7% of a 4 kW stack.
    assert high["bol_power_W"] == pytest.approx(low["bol_power_W"] * 0.933, rel=0.01)
    # A smaller array is lighter, which leaves more dry mass for payload.
    assert high["array_kg"] < low["array_kg"]
    assert high["payload_capacity_kg"] > low["payload_capacity_kg"]


def test_mixed_assembly_sizes_between_its_two_wirings():
    from prospector.spacecraft.buildability import assess_assembly
    m, cat = BusModel(), _ppu_side_catalog()

    def bol(mounts):
        return assess_assembly(dry_kg=300.0, prop_kg=350.0, mounts=mounts, catalog=cat,
                               model=m)["bol_power_W"]

    assert bol([("hi", 4)]) < bol([("lo", 2), ("hi", 2)]) < bol([("lo", 4)])


def test_required_bol_default_case():
    # One 800 W thruster: thrusting mode (housekeeping through avionics chain + thruster through
    # the full PPU chain) beats ops mode; x1.2 margin (BusModel default). Belt degradation never
    # enters sizing (it is a flown result).
    m = BusModel()
    bol = required_bol_W(1, 800.0, m)
    avi, thr = m.avionics_power_eff(), m.thruster_power_eff()
    thrust_mode = 165.0 / avi + 800.0 / thr
    ops_mode = (165.0 + 600.0) / avi
    expected = max(thrust_mode, ops_mode) * 1.2
    assert bol == pytest.approx(expected, rel=1e-12)
    assert thrust_mode > ops_mode              # an 800 W thruster dominates the ops load


def test_ops_mode_floors_the_sizing_load():
    # A tiny thruster doesn't shrink the array below what payload + comms TX need: the ops mode
    # (all avionics, no thrust) sets the floor, priced through the avionics chain.
    m = BusModel()
    small = required_bol_W(1, 100.0, m)
    floor = (165.0 + 600.0) / m.avionics_power_eff() * 1.2
    assert small == pytest.approx(floor, rel=1e-12)


def test_eol_is_spiral_derived_only():
    # The end-of-life fraction is the spiral's PHYSICAL array fraction x transmission; there is NO
    # static %/day or %/yr rate (belt degradation is measured by the propagated escape). With no
    # spiral fraction supplied, only the transmission loss applies.
    m = BusModel()
    assert mission_eol_factor(m, eol_power_fraction=0.7) == pytest.approx(0.7 * 0.9, rel=1e-12)
    assert mission_eol_factor(m) == pytest.approx(0.9, rel=1e-12)
    # The retired static rate is gone from the model entirely.
    assert "belt_degradation_pct_day" not in BusModel.model_fields


def test_negative_margin_undersizes_the_array():
    # A negative power margin deliberately undersizes the array (BOL below the worst-mode load) --
    # a valid design to explore flying power-limited.
    base = required_bol_W(3, 800.0, BusModel(power_margin_pct=0.0))
    under = required_bol_W(3, 800.0, BusModel(power_margin_pct=-40.0))
    assert under < base
    assert under == pytest.approx(base * 0.6, rel=1e-9)


def test_belt_residence_does_not_size_the_arrays():
    # Belt degradation is a flown result, not a sizing input: the array is built to the worst-mode
    # load at beginning of life, so two designs differing only in belt residence get the same array
    # and payload capacity. With the static per-day rate retired, belt RESIDENCE alone no longer
    # even moves the reported eol_factor, only the spiral's PHYSICAL end-of-life fraction does.
    slow = assess(dry_kg=300.0, prop_kg=350.0, n_engines=3, engine_power_W=820.0,
                  engine_mass_kg=8.25, belt_days=220.0)
    fast = assess(dry_kg=300.0, prop_kg=350.0, n_engines=3, engine_power_W=820.0,
                  engine_mass_kg=8.25, belt_days=90.0)
    assert slow["array_kg"] == fast["array_kg"]                 # sizing is belt-independent
    assert slow["bol_power_W"] == fast["bol_power_W"]
    assert slow["payload_capacity_kg"] == fast["payload_capacity_kg"]
    assert slow["eol_factor"] == fast["eol_factor"]             # belt DAYS is no longer a lever
    # The spiral's physical end-of-life fraction is the lever that lowers eol_factor.
    degraded = assess(dry_kg=300.0, prop_kg=350.0, n_engines=3, engine_power_W=820.0,
                      engine_mass_kg=8.25, eol_power_fraction=0.5)
    assert degraded["eol_factor"] < slow["eol_factor"]


def test_fixed_bus_is_a_ratio_of_dry_mass():
    # The fixed bus (GNC, C&DH, comms, batteries, RCS) is 13% of dry, not an itemized CBE list - a
    # 400 kg vehicle carries twice the bus of a 200 kg one.
    small = assess(dry_kg=200.0, prop_kg=250.0, n_engines=3,
                   engine_power_W=820.0, engine_mass_kg=8.25)
    big = assess(dry_kg=400.0, prop_kg=450.0, n_engines=3,
                 engine_power_W=820.0, engine_mass_kg=8.25)
    assert small["fixed_bus_kg"] == pytest.approx(0.13 * 200.0, rel=1e-9)
    assert big["fixed_bus_kg"] == pytest.approx(0.13 * 400.0, rel=1e-9)


def test_payload_capacity_inverts_the_dry_identity():
    # assess() is one margin-free identity both ways: the dry mass at which capacity falls to zero
    # is min_dry, so re-running at min_dry must report ~0 capacity.
    base = assess(dry_kg=300.0, prop_kg=350.0, n_engines=3,
                  engine_power_W=820.0, engine_mass_kg=8.25)
    again = assess(dry_kg=base["min_dry_kg"], prop_kg=350.0, n_engines=3,
                   engine_power_W=820.0, engine_mass_kg=8.25)
    assert again["payload_capacity_kg"] == pytest.approx(0.0, abs=0.05)
    assert base["buildable"]


def test_power_hungry_engines_cost_dry_mass_and_dollars():
    # The trade this exists for: a 5 kW thruster forces a much bigger array than an 820 W one,
    # eating payload capacity and raising cost at the same dry budget.
    small = assess(dry_kg=300.0, prop_kg=350.0, n_engines=3,
                   engine_power_W=820.0, engine_mass_kg=8.25)
    hungry = assess(dry_kg=300.0, prop_kg=350.0, n_engines=3,
                    engine_power_W=5000.0, engine_mass_kg=18.0)
    assert hungry["array_kg"] > 2.5 * small["array_kg"]
    assert hungry["payload_capacity_kg"] < small["payload_capacity_kg"] - 30.0
    assert hungry["build_cost_musd"] > small["build_cost_musd"]


def test_too_small_a_dry_budget_is_not_buildable():
    tiny = assess(dry_kg=60.0, prop_kg=100.0, n_engines=3,
                  engine_power_W=820.0, engine_mass_kg=8.25)
    assert not tiny["buildable"]
    assert tiny["payload_capacity_kg"] < 0      # reads as "how overweight"
    assert tiny["min_dry_kg"] > 60.0
    # The verdict diagnoses itself: the dry floor and the items driving it.
    assert f"{tiny['min_dry_kg']:.0f} kg dry" in tiny["build_why"]
    assert "arrays" in tiny["build_why"]


def test_assess_engine_uses_the_library_spec():
    eng = Engine(name="Test thruster", isp_s=1500.0, thrust_mN=40.0, power_W=800.0,
                 mass_kg=8.0)      # full system per unit (thruster + PPU + feed)
    via_engine = assess_engine(dry_kg=250.0, prop_kg=300.0, n_engines=4, engine=eng)
    direct = assess(dry_kg=250.0, prop_kg=300.0, n_engines=4,
                    engine_power_W=800.0, engine_mass_kg=8.0)
    assert via_engine == direct


def test_thruster_cost_comes_from_the_engine():
    # Cost travels with the engine (Engine.cost_musd), not a blanket build-model constant: a
    # pricier thruster raises only the thrusters line, by n_engines x the per-unit delta.
    cheap = Engine(name="cheap", isp_s=1400.0, thrust_mN=39.0, power_W=820.0,
                   mass_kg=8.25, cost_musd=1.5)
    dear = cheap.model_copy(update={"cost_musd": 2.5})
    a = assess_engine(dry_kg=300.0, prop_kg=350.0, n_engines=3, engine=cheap)
    b = assess_engine(dry_kg=300.0, prop_kg=350.0, n_engines=3, engine=dear)
    assert b["cost_breakdown_musd"]["thrusters"] == pytest.approx(
        a["cost_breakdown_musd"]["thrusters"] + 3 * 1.0)
    # Mass-side outputs are untouched by a cost change.
    assert b["payload_capacity_kg"] == pytest.approx(a["payload_capacity_kg"])


def test_cost_breakdown_sums_to_total():
    a = assess(dry_kg=300.0, prop_kg=350.0, n_engines=3,
               engine_power_W=820.0, engine_mass_kg=8.25)
    parts = sum(a["cost_breakdown_musd"].values())
    assert a["build_cost_musd"] == pytest.approx(parts, abs=0.05)
    assert math.isfinite(a["build_cost_musd"]) and a["build_cost_musd"] > 0


# --- thruster lifetime: throughput + ignitions, per thruster, as a caution ---

def test_lifetime_unset_is_not_checked():
    # No qualified limits on file -> lifetime_ok is None (not a pass, not a fail) and the mass
    # verdict is untouched.
    a = assess(dry_kg=300.0, prop_kg=350.0, n_engines=2,
               engine_power_W=820.0, engine_mass_kg=8.25)
    assert a["lifetime_ok"] is None
    assert a["throughput_margin_kg"] is None
    assert a["prop_per_thruster_kg"] == pytest.approx(175.0)  # 350 / 2


def test_throughput_shared_across_thrusters():
    # 200 kg over 2 thrusters = 100 kg each: inside a 120 kg qual, over an 80 kg one.
    ok = assess(dry_kg=200.0, prop_kg=200.0, n_engines=2, engine_power_W=820.0,
                engine_mass_kg=8.25, throughput_kg=120.0)
    assert ok["lifetime_ok"] is True
    assert ok["throughput_margin_kg"] == pytest.approx(20.0)

    over = assess(dry_kg=200.0, prop_kg=200.0, n_engines=2, engine_power_W=820.0,
                  engine_mass_kg=8.25, throughput_kg=80.0)
    assert over["lifetime_ok"] is False
    assert over["throughput_margin_kg"] == pytest.approx(-20.0)
    assert "throughput" in over["lifetime_why"]
    # A lifetime overage is a CAUTION, not a mass-fit veto: buildable still reflects mass.
    assert over["buildable"] == ok["buildable"]
    assert "lifetime caution" in over["build_why"]


def test_ignitions_checked_only_with_a_mission_count():
    # Limit on file but no mission ignition count -> ignition leg is skipped.
    a = assess(dry_kg=200.0, prop_kg=120.0, n_engines=2, engine_power_W=820.0,
               engine_mass_kg=8.25, throughput_kg=100.0, lifetime_ignitions=500)
    assert a["lifetime_ok"] is True            # throughput passes, ignitions not evaluated
    assert a["mission_ignitions"] is None

    over = assess(dry_kg=200.0, prop_kg=120.0, n_engines=2, engine_power_W=820.0,
                  engine_mass_kg=8.25, throughput_kg=100.0,
                  lifetime_ignitions=500, ignitions=900.0)
    assert over["lifetime_ok"] is False
    assert over["mission_ignitions"] == pytest.approx(900.0)
    assert "ignition" in over["lifetime_why"]


def test_assess_engine_threads_the_library_limits():
    eng = Engine(name="X00", isp_s=1650.0, thrust_mN=75.0, power_W=1000.0,
                 mass_kg=8.0, throughput_kg=65.0, lifetime_ignitions=2000)
    # 100 kg over 2 thrusters = 50 kg each: inside the 65 kg qual.
    a = assess_engine(dry_kg=100.0, prop_kg=100.0, n_engines=2, engine=eng,
                      ignitions=1500.0)
    assert a["lifetime_ok"] is True
    assert a["prop_per_thruster_kg"] == pytest.approx(50.0)
    # The same engine on a single mount carries the whole load: over the qual.
    solo = assess_engine(dry_kg=100.0, prop_kg=100.0, n_engines=1, engine=eng)
    assert solo["lifetime_ok"] is False


def test_assess_engine_without_limits_matches_direct():
    # Engines with no lifetime fields must reproduce the bare assess() result.
    eng = Engine(name="Test thruster", isp_s=1500.0, thrust_mN=40.0, power_W=800.0, mass_kg=8.0)
    via = assess_engine(dry_kg=250.0, prop_kg=300.0, n_engines=4, engine=eng)
    direct = assess(dry_kg=250.0, prop_kg=300.0, n_engines=4,
                    engine_power_W=800.0, engine_mass_kg=8.0)
    assert via == direct
