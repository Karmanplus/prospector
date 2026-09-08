"""The working-gas library and how the gas choice reprices build cost + tank mass.

Authored Isp/thrust are per-gas on the engine; the gas sets the propellant's cost and
storage density (build-model inputs) and -- for an engine with NO authored data on a gas
-- the thrust/Isp scale factors used to ESTIMATE it (`engine_on_propellant`). These tests
pin that wiring and that xenon (the reference gas) reproduces the build-model defaults so
existing assessments don't drift.
"""
import pytest

from prospector.spacecraft.buildability import (
    REFERENCE_DENSITY_KG_M3,
    BusModel,
    assess,
    assess_engine,
    tank_dry_mass_kg,
)
from prospector.spacecraft.propellants import (
    DEFAULT_PROPELLANT,
    Propellant,
    engine_on_propellant,
    load_propellants,
    resolve_propellant,
    storage_density,
)
from prospector.spacecraft.propulsion import Engine, load_engines


def test_library_loads_and_xenon_is_the_reference():
    cat = load_propellants()
    assert {"xenon", "krypton", "iodine"} <= set(cat)
    xe = cat["xenon"]
    # Xenon must match the BusModel's reference cost/tank, or wiring it in drifts costs.
    assert xe.cost_per_kg == pytest.approx(BusModel().propellant_cost_per_kg)
    assert xe.cost_per_kg == pytest.approx(2500.0)
    # Xenon's storage density is the build model's tank-sizing reference.
    assert xe.density_kg_m3 == pytest.approx(REFERENCE_DENSITY_KG_M3)
    assert xe.density_kg_m3 == pytest.approx(1990.5)
    # Xenon is the scaling reference: both factors are exactly 1.0.
    assert xe.thrust_scale == pytest.approx(1.0)
    assert xe.isp_scale == pytest.approx(1.0)
    # Krypton: cheaper per kg, and far less dense -> a bigger, heavier tank; spreadsheet scales.
    kr = cat["krypton"]
    assert kr.cost_per_kg < xe.cost_per_kg
    assert kr.density_kg_m3 < xe.density_kg_m3
    assert kr.thrust_scale == pytest.approx(0.83)
    assert kr.isp_scale == pytest.approx(1.0377, abs=1e-4)


def test_engine_on_propellant_native_and_estimated():
    cat = load_propellants()
    eng = Engine(name="Xe", isp_s=3000, thrust_mN=600, power_W=12000, mass_kg=24.0,
                 propellant="xenon", throughput_kg=1700)
    # Native gas: unchanged, not flagged estimated.
    same, est = engine_on_propellant(eng, "xenon", cat)
    assert est is False
    assert same.thrust_mN == pytest.approx(600) and same.isp_s == pytest.approx(3000)
    # Cross-gas: thrust/Isp scaled by the gas's factors (xenon source = 1.0), flagged.
    kr, est = engine_on_propellant(eng, "krypton", cat)
    assert est is True
    assert kr.propellant == "krypton"
    assert kr.thrust_mN == pytest.approx(600 * cat["krypton"].thrust_scale)
    assert kr.isp_s == pytest.approx(3000 * cat["krypton"].isp_scale)
    # Power, mass, and the lifetime limit are not scaled (only performance).
    assert kr.power_W == eng.power_W and kr.mass_kg == eng.mass_kg
    assert kr.throughput_kg == eng.throughput_kg


def test_engine_on_propellant_from_nonxenon_source():
    # A krypton-authored engine estimated on iodine scales by the target/source ratio.
    cat = load_propellants()
    eng = Engine(name="Kr", isp_s=2200, thrust_mN=250, power_W=4500, mass_kg=12.0,
                 propellant="krypton")
    io, est = engine_on_propellant(eng, "iodine", cat)
    assert est is True
    ratio = cat["iodine"].thrust_scale / cat["krypton"].thrust_scale
    assert io.thrust_mN == pytest.approx(250 * ratio)


def test_resolve_falls_back_to_reference_gas():
    cat = load_propellants()
    assert resolve_propellant(None, cat).name == cat[DEFAULT_PROPELLANT].name
    assert resolve_propellant("not-a-gas", cat).name == cat[DEFAULT_PROPELLANT].name
    assert resolve_propellant("krypton", cat).name == "Krypton"
    # No catalog at all -> None (caller keeps the BusModel xenon defaults).
    assert resolve_propellant("krypton", {}) is None


def test_every_library_engine_names_a_known_gas():
    engines, props = load_engines(), load_propellants()
    for key, eng in engines.items():
        assert eng.propellant in props, f"engine {key} names unknown gas {eng.propellant!r}"


def test_xenon_engine_matches_the_default_pricing():
    # An engine on xenon, assessed with the catalog, stays close to the catalog-free result (xenon
    # is the BusModel reference gas). With CoolProp the catalog path prices the tank at the
    # real-fluid density for the tank conditions, a few percent off the static reference density,
    # so this is a refinement, not silent drift; the match is therefore approximate and the
    # propellant is identified as xenon.
    eng = Engine(name="Xe", isp_s=1600, thrust_mN=80, power_W=1000, mass_kg=8.0)
    cat = load_propellants()
    with_cat = assess_engine(dry_kg=200, prop_kg=200, n_engines=2, engine=eng,
                             propellants=cat)
    without = assess_engine(dry_kg=200, prop_kg=200, n_engines=2, engine=eng)
    assert with_cat["build_cost_musd"] == pytest.approx(without["build_cost_musd"], rel=0.02)
    assert with_cat["tank_kg"] == pytest.approx(without["tank_kg"], rel=0.05)
    assert with_cat["propellant"] == "Xenon"


def test_krypton_is_cheaper_but_grows_the_tank():
    cat = load_propellants()
    xe = Engine(name="Xe", isp_s=1600, thrust_mN=80, power_W=1000, mass_kg=8.0,
                propellant="xenon")
    kr = Engine(name="Kr", isp_s=1600, thrust_mN=80, power_W=1000, mass_kg=8.0,
                propellant="krypton")
    a_xe = assess_engine(dry_kg=300, prop_kg=300, n_engines=2, engine=xe, propellants=cat)
    a_kr = assess_engine(dry_kg=300, prop_kg=300, n_engines=2, engine=kr, propellants=cat)
    # Same propellant mass: krypton's tank is heavier (less dense) but its propellant line is far
    # cheaper, the two effects the gas choice is supposed to have.
    assert a_kr["tank_kg"] > a_xe["tank_kg"]
    assert a_kr["cost_breakdown_musd"]["propellant"] < a_xe["cost_breakdown_musd"]["propellant"]
    assert a_kr["propellant"] == "Krypton"
    # The heavier tank eats payload capacity (fewer kg left after the bigger tank).
    assert a_kr["payload_capacity_kg"] < a_xe["payload_capacity_kg"]


def test_assess_propellant_overrides_are_optional():
    # No propellant args -> prices as xenon (BusModel default); the field reads None.
    a = assess(dry_kg=200, prop_kg=200, n_engines=2, engine_power_W=1000,
               engine_mass_kg=8.0)
    assert a["propellant"] is None
    # Single-tank pressure-vessel model at xenon's reference density: the assessed tank mass is the
    # geometry helper applied to the load's volume.
    m = BusModel()
    expected = tank_dry_mass_kg(200 / REFERENCE_DENSITY_KG_M3, m)
    assert a["tank_kg"] == pytest.approx(expected, abs=0.01)


def test_tank_model_reproduces_prior_affine_fit_with_type_iii_wall():
    # The geometry math is calibrated against the prior affine model (base 6.77 kg + 116.15 kg per
    # m^3 of propellant volume): a 2:1 capsule with the heavier Type III wall (750 MPa, 1800
    # kg/m^3) and the 1.2x factor reproduces it. The default wall is the lighter Type IV, so the
    # next test pins that the default comes out lighter.
    m = BusModel(tank_wall_strength_pa=750e6, tank_wall_density_kg_m3=1800.0)
    for vol in (0.1, 0.5, 1.0, 2.0):
        prior = 6.77 + 116.15 * vol
        assert tank_dry_mass_kg(vol, m) == pytest.approx(prior, rel=0.01)


def test_storage_density_falls_back_without_a_coolprop_fluid():
    # Iodine names no CoolProp fluid -> its authored density stands at any pressure/temperature.
    io = load_propellants()["iodine"]
    assert io.coolprop_name is None
    assert storage_density(io, 150.0, 20.0) == pytest.approx(io.density_kg_m3)
    assert storage_density(io, 300.0, -40.0) == pytest.approx(io.density_kg_m3)


def test_storage_density_falls_back_on_unknown_fluid():
    # A bad CoolProp fluid name can't be evaluated -> fall back to the authored density, never
    # raise (the gas library must stay usable).
    bogus = Propellant(name="Bogus", cost_per_kg=1.0, density_kg_m3=500.0,
                       coolprop_name="Notafluid")
    assert storage_density(bogus, 150.0, 20.0) == pytest.approx(500.0)


def test_storage_density_responds_to_pressure_and_temperature():
    # With CoolProp present, xenon (supercritical at SEP storage) densifies as the tank is cooled
    # or pressurized, the real-fluid behavior the knobs are meant to expose.
    pytest.importorskip("CoolProp")
    xe = load_propellants()["xenon"]
    assert storage_density(xe, 150.0, -20.0) > storage_density(xe, 150.0, 60.0)   # colder = denser
    assert storage_density(xe, 300.0, 20.0) > storage_density(xe, 100.0, 20.0)    # higher P = denser
    # The live value is the EOS density, not the static fallback.
    assert storage_density(xe, 150.0, 20.0) != pytest.approx(xe.density_kg_m3)


def test_tank_temperature_knob_moves_the_assessment():
    # The build-model temperature knob flows through to the tank: colder xenon is denser, so the
    # same propellant load needs less volume and a lighter tank.
    pytest.importorskip("CoolProp")
    cat = load_propellants()
    eng = Engine(name="Xe", isp_s=1600, thrust_mN=80, power_W=1000, mass_kg=8.0,
                 propellant="xenon")
    cold = assess_engine(dry_kg=300, prop_kg=300, n_engines=2, engine=eng, propellants=cat,
                         model=BusModel(tank_temperature_C=-20.0))
    warm = assess_engine(dry_kg=300, prop_kg=300, n_engines=2, engine=eng, propellants=cat,
                         model=BusModel(tank_temperature_C=60.0))
    assert cold["tank_kg"] < warm["tank_kg"]


def test_default_type_iv_wall_is_lighter_than_type_iii():
    # The default Type IV wall (stronger and less dense) yields a lighter tank than the prior Type
    # III calibration for the same volume, a better tank frees payload.
    type_iv = BusModel()
    type_iii = BusModel(tank_wall_strength_pa=750e6, tank_wall_density_kg_m3=1800.0)
    assert tank_dry_mass_kg(1.0, type_iv) < tank_dry_mass_kg(1.0, type_iii)


def test_sphere_is_the_lightest_tank_shape():
    # A sphere (ratio 1) has the least wall mass for a given volume; elongating it adds mass.
    sphere = BusModel(tank_length_diameter_ratio=1.0)
    capsule = BusModel(tank_length_diameter_ratio=2.0)
    long_tank = BusModel(tank_length_diameter_ratio=4.0)
    ms = tank_dry_mass_kg(1.0, sphere)
    mc = tank_dry_mass_kg(1.0, capsule)
    ml = tank_dry_mass_kg(1.0, long_tank)
    assert ms < mc < ml


def test_tank_walls_scale_with_pressure_and_safety_factor():
    # Wall mass (everything above the fixed base) is linear in burst pressure: doubling the
    # operating pressure, or the safety factor, doubles the wall mass.
    base = BusModel(tank_pressure_bar=150.0, tank_safety_factor=1.5).tank_base_mass_kg
    low = tank_dry_mass_kg(1.0, BusModel(tank_pressure_bar=150.0, tank_safety_factor=1.5))
    hi_p = tank_dry_mass_kg(1.0, BusModel(tank_pressure_bar=300.0, tank_safety_factor=1.5))
    hi_sf = tank_dry_mass_kg(1.0, BusModel(tank_pressure_bar=150.0, tank_safety_factor=3.0))
    assert (hi_p - base) == pytest.approx(2.0 * (low - base), rel=1e-9)
    assert (hi_sf - base) == pytest.approx(2.0 * (low - base), rel=1e-9)


def test_tank_mass_is_linear_in_propellant_volume():
    # For a fixed shape and pressure the pressure-vessel wall mass is exactly linear in volume
    # (thickness grows with radius), so the per-m^3 slope is constant.
    m = BusModel()
    base = m.tank_base_mass_kg
    s1 = tank_dry_mass_kg(1.0, m) - base
    s2 = tank_dry_mass_kg(2.0, m) - base
    assert s2 == pytest.approx(2.0 * s1, rel=1e-9)
