"""Tests for the layered config model (prospector.config).

These pin what the rest of the tool relies on: vehicles are physical only and name their engines by
reference; the delta-v budget always follows from those parts rather than being stored, and is
loosened in the direction that keeps targets; the escape charge follows from the launch type, as an
estimate and then from a flown spiral, rather than being typed in; a study resolves its parts into
one config; validation rejects impossible setups; and every kind of file survives a round trip
through YAML.
"""
import math
from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from prospector.config import (
    Desirability,
    EngineMount,
    Mission,
    ResolvedConfig,
    Screening,
    Study,
    Vehicle,
    default_mission,
    default_vehicle,
    list_studies,
    load_study,
    resolve_study,
    save_mission,
    save_study,
    save_vehicle,
)
from prospector.constants import G0_KM_S2
from prospector.launch import LaunchOrbit, escape_dv_estimate
from prospector.spacecraft.propulsion import Engine
from tests import library as lib

CAT = {"E": Engine(name="E", isp_s=2000, thrust_mN=100, power_W=1000, mass_kg=5)}
# A minimal launch catalog: an LV-provided escape and a 400 km SEP-spiral start.
LAUNCHES = {
    "TLI": LaunchOrbit(name="tli", perigee_alt_km=185, apogee_alt_km=400000,
                       inclination_deg=28.5, escape_provided=True),
    "LEO": LaunchOrbit(name="leo", perigee_alt_km=400, apogee_alt_km=400,
                       inclination_deg=28.5),
}


def _resolved(dry=500.0, fuel=500.0, unusable=0.0, launch="TLI", margin=1.0,
              return_trip=False, **mission_kw):
    vehicle = Vehicle(name="v", dry_mass=dry, fuel_mass=fuel, unusable_prop=unusable,
                      engines=[EngineMount(type="E", count=1)])
    mission = Mission(launch_orbit=launch, return_trip=return_trip, **mission_kw)
    return ResolvedConfig.build(mission, vehicle, Screening(dv_margin_factor=margin), CAT,
                                launches=LAUNCHES)


def test_build_sizes_the_array_a_vehicle_never_got():
    """A vehicle with no array power is one nobody has sized, not one with power to spare.

    Left at zero it used to reach the cruise as "unlimited power, no throttling, no drag", so a
    study loaded straight off disk flew a different spacecraft from the same study after the
    vehicle editor had touched it. Resolving sizes it from the engines it mounts, and the answer
    has to match the sizing model everywhere else uses.
    """
    from prospector.spacecraft import buildability

    rc = _resolved()
    model = buildability.load_bus_model()
    expected = buildability.required_bol_W(
        1, CAT["E"].power_W, model,
        thruster_chain_eff=buildability.assembly_thruster_chain_eff(rc.vehicle.mounts, CAT, model))
    assert rc.vehicle.solar_power_W == pytest.approx(round(expected, 1))
    assert rc.vehicle.area_m2 > 0.0


def test_build_leaves_a_typed_array_alone():
    """A power figure somebody entered is their number, not a placeholder to overwrite."""
    vehicle = Vehicle(name="v", dry_mass=500.0, fuel_mass=500.0, solar_power_W=4321.0,
                      area_m2=12.5, engines=[EngineMount(type="E", count=1)])
    rc = ResolvedConfig.build(Mission(launch_orbit="TLI"), vehicle, Screening(), CAT,
                              launches=LAUNCHES)
    assert rc.vehicle.solar_power_W == 4321.0 and rc.vehicle.area_m2 == 12.5


def test_budget_is_derived_and_margin_loosened():
    rc = _resolved(dry=500, fuel=500, margin=1.2)
    cap = G0_KM_S2 * 2000 * math.log(1000 / 500)
    assert math.isclose(rc.total_dv_capability, cap, rel_tol=1e-9)
    assert math.isclose(rc.cruise_dv_limit, cap, rel_tol=1e-9)
    assert math.isclose(rc.dv_budget, cap * 1.2, rel_tol=1e-9)


def test_escape_charge_derived_from_launch_type():
    # An LV-provided escape charges nothing; a SEP launch type charges the spiral estimate.
    assert _resolved(launch="TLI").escape_dv == 0.0
    rc = _resolved(launch="LEO")
    est = escape_dv_estimate(LAUNCHES["LEO"])
    assert math.isclose(rc.escape_dv, est, rel_tol=1e-12)
    assert math.isclose(rc.cruise_dv_limit, rc.total_dv_capability - est, rel_tol=1e-9)
    assert rc.escape_source == "estimate"


def test_spiral_refinement_overrides_estimate():
    rc = _resolved(launch="LEO")
    rc.escape_dv_refined = 8.1
    assert rc.escape_dv == 8.1
    assert rc.escape_source == "spiral"
    assert math.isclose(rc.cruise_dv_limit, rc.total_dv_capability - 8.1, rel_tol=1e-9)


def test_unknown_launch_type_rejected():
    with pytest.raises(ValueError, match="unknown launch type"):
        _resolved(launch="NOPE")


def test_return_capability_only_when_return_trip():
    assert _resolved(return_trip=False).return_dv_capability is None
    rc = _resolved(return_trip=True, return_by="2030-01-01", return_orbit_insert_dv=0.1,
                   asteroid_payload_mass=1000)
    assert rc.return_dv_capability is not None
    assert rc.return_dv_capability < rc.total_dv_capability    # payload aboard costs dV


def test_return_propellant_accounting():
    from prospector.launch import load_return_destinations
    rc = _resolved(launch="TLI", dry=500, fuel=500, return_trip=True,
                   return_by="2030-01-01", return_destination="EML2",
                   asteroid_payload_mass=150.0)
    dest = load_return_destinations()["EML2"]
    assert rc.usable_propellant_kg == pytest.approx(500.0)
    # Laden start = outbound arrival mass + the payload collected at the asteroid.
    assert rc.return_start_mass_kg(700.0) == pytest.approx(850.0)
    # Available = usable - escape (0 for TLI) - outbound cruise, floored at zero.
    assert rc.return_available_propellant_kg(120.0) == pytest.approx(380.0)
    assert rc.return_available_propellant_kg(600.0) == 0.0
    # Insertion propellant: Tsiolkovsky on the laden ARRIVAL mass at the catalog dV.
    veff = G0_KM_S2 * 2000
    expect = 600.0 * (1.0 - math.exp(-dest.insertion_dv_kms / veff))
    assert rc.insertion_propellant_kg(600.0, dest) == pytest.approx(expect)
    assert rc.insertion_dv_kms(dest) == pytest.approx(dest.insertion_dv_kms)


def test_insertion_dv_override_beats_the_catalog():
    from prospector.launch import load_return_destinations
    dest = load_return_destinations()["EML2"]
    rc = _resolved(launch="TLI", return_trip=True, return_by="2030-01-01",
                   return_destination="EML2", return_orbit_insert_dv=1.5)
    assert rc.insertion_dv_kms(dest) == 1.5


def test_unknown_return_destination_rejected():
    with pytest.raises(ValueError, match="unknown return destination"):
        _resolved(return_trip=True, return_by="2030-01-01", return_destination="NOPE")


def test_mixed_assembly_isp_is_between_engine_isps():
    cat = {"A": Engine(name="A", isp_s=2000, thrust_mN=100),
           "B": Engine(name="B", isp_s=4000, thrust_mN=100)}
    veh = Vehicle(name="v", dry_mass=500, fuel_mass=500,
                  engines=[EngineMount(type="A", count=1), EngineMount(type="B", count=1)])
    rc = ResolvedConfig.build(default_mission(), veh, Screening(), cat)
    assert 2000 < rc.effective_isp < 4000


def test_thruster_chain_eff_follows_the_engines_ppu_wiring():
    """The one accessor every consumer reads for the array-to-thruster fraction -- the escape
    propagator, the cruise thrust cap, and the array sizing all call it, so a vehicle can never be
    flown on one chain and sized on another. It follows the engines' wiring, not the bus model
    alone."""
    from prospector.spacecraft.buildability import BusModel

    m = BusModel()
    cat = {"lo": Engine(name="lo", isp_s=2000, thrust_mN=100, power_W=1000, ppu_bus_side="low"),
           "hi": Engine(name="hi", isp_s=2000, thrust_mN=100, power_W=1000, ppu_bus_side="high")}

    def chain(*mounts):
        veh = Vehicle(name="v", dry_mass=500, fuel_mass=500,
                      engines=[EngineMount(type=k, count=n) for k, n in mounts])
        return ResolvedConfig.build(default_mission(), veh, Screening(), cat).thruster_chain_eff(m)

    assert chain(("lo", 4)) == pytest.approx(m.thruster_power_eff("low"), rel=1e-12)
    assert chain(("hi", 4)) == pytest.approx(m.thruster_power_eff("high"), rel=1e-12)
    assert chain(("lo", 2), ("hi", 2)) == pytest.approx(
        4000.0 / (2000.0 / m.thruster_power_eff("low") + 2000.0 / m.thruster_power_eff("high")),
        rel=1e-12)


def test_build_rejects_unknown_engine_reference():
    veh = Vehicle(name="v", dry_mass=500, fuel_mass=500,
                  engines=[EngineMount(type="DoesNotExist", count=1)])
    with pytest.raises(ValueError, match="unknown engine"):
        ResolvedConfig.build(default_mission(), veh, Screening(), CAT)


def test_vehicle_requires_an_engine():
    with pytest.raises(ValidationError):
        Vehicle(name="v", dry_mass=500, fuel_mass=500, engines=[])


def test_no_usable_propellant_rejected():
    with pytest.raises(ValidationError):
        Vehicle(name="v", dry_mass=300, fuel_mass=10, unusable_prop=10,
                engines=[EngineMount(type="E")])


def test_backwards_launch_window_rejected():
    with pytest.raises(ValidationError):
        Mission(launch_window=("2028-03-31", "2028-01-01"))


def test_return_trip_requires_return_by():
    with pytest.raises(ValidationError):
        Mission(return_trip=True, return_orbit_insert_dv=0.1)


def test_library_round_trip(tmp_path):
    # Each kind saves and reloads unchanged, and a study resolves its named parts. Seed the engine
    # library first: the default vehicle mounts whatever the ACTIVE library offers rather than a
    # key hardcoded in source, so the library has to exist before it is built.
    from prospector.spacecraft.propulsion import save_engine
    save_engine(Engine(name="Thruster A", isp_s=1820, thrust_mN=90), "thruster-a", tmp_path / "engines")
    save_vehicle(default_vehicle(tmp_path), "v1", tmp_path)
    save_mission(default_mission(), "m1", tmp_path)
    save_study(Study(name="S", mission="m1", vehicle="v1", screening=Screening(h_max=22)), "s1", tmp_path)

    study = load_study("s1", tmp_path)
    assert study.name == "S"
    rc = resolve_study(study, tmp_path)
    assert rc.screening.h_max == 22
    assert rc.dv_budget > 0


def test_desirability_defaults_permissive():
    # An unset desirability keeps everything; no term is active, so it can never drop a target.
    d = Desirability()
    assert d.min_tier is None and d.taxonomy_include is None
    assert d.min_period_h is None and d.min_diameter_m is None
    # build() defaults the desirability when the caller omits it (back-compatible signature).
    assert _resolved().desirability == d


def test_study_round_trips_desirability(tmp_path):
    from prospector.spacecraft.propulsion import save_engine
    save_engine(Engine(name="Thruster A", isp_s=1820, thrust_mN=90), "thruster-a", tmp_path / "engines")
    save_vehicle(default_vehicle(tmp_path), "v1", tmp_path)
    save_mission(default_mission(), "m1", tmp_path)
    from prospector.spacecraft.propulsion import save_engine
    save_engine(Engine(name="Test thruster", isp_s=1820, thrust_mN=90), "test-thruster",
                tmp_path / "engines")
    des = Desirability(min_tier="A", taxonomy_include=["C", "B", "D"], min_period_h=2.5, min_diameter_m=50.0)
    save_study(Study(name="S", mission="m1", vehicle="v1", desirability=des), "s1", tmp_path)

    study = load_study("s1", tmp_path)
    assert study.desirability == des
    assert resolve_study(study, tmp_path).desirability == des


def test_desirability_min_period_and_diameter_nonnegative():
    with pytest.raises(ValidationError):
        Desirability(min_period_h=-1.0)
    with pytest.raises(ValidationError):
        Desirability(min_diameter_m=-5.0)


def test_shipped_studies_resolve():
    names = list_studies()
    assert names, "the active library ships no studies"
    for name in names:
        rc = resolve_study(load_study(name))
        assert rc.dv_budget > 0
        assert rc.mission.name and rc.vehicle.name        # names present (not 'Untitled')
    # Every study in the active library must fully resolve; that is the check worth having, and
    # it runs against whichever library is configured. Properties of any ONE study belong to that
    # library, not to the tool, so they are not asserted here.
    one = resolve_study(load_study(lib.study_key()))
    assert one.total_dv_capability > 0.0
    assert one.escape_source in {"launch vehicle", "estimate", "spiral"}


def test_departure_window_shifts_by_escape_time():
    # The cruise can only start once the spacecraft has left Earth: the date-resolved solvers plan
    # against liftoff + escape time, not the raw liftoff window.
    rc = _resolved(launch="LEO")
    start, end = rc.mission.launch_window
    d0, d1 = rc.departure_window
    shift = (d0 - start).days
    assert shift == round(rc.escape_tof_days) > 0
    assert (d1 - end).days == shift
    # A rocket-provided escape departs at liftoff.
    tli = _resolved(launch="TLI")
    assert tli.escape_tof_days == 0.0
    assert tli.departure_window == tli.mission.launch_window
    # A matching spiral run's real duration overrides the idealized estimate.
    rc.escape_tof_refined = 220.0
    assert (rc.departure_window[0] - start).days == 220


def test_escape_estimate_prices_the_departure_vinf():
    # The escape charge is priced AT the planned departure v-infinity: a faster departure costs
    # more, and the charge is exactly the analytic estimate at that v-infinity.
    rc = _resolved(launch="LEO")
    base = rc.escape_dv_estimate
    rc.departure_vinf_kms = 2.0
    assert rc.escape_dv_estimate > base
    assert math.isclose(rc.escape_dv_estimate,
                        escape_dv_estimate(LAUNCHES["LEO"], 2.0), rel_tol=1e-12)


def test_dv_budget_credits_the_departure_vinf_back():
    # The screen budget is exactly (cruise limit + planned v-infinity) x margin: the escape charge
    # pays for the v-infinity and the budget credits the same speed back. Since the spiral's
    # marginal cost per km/s of v-infinity is under 1, planning a faster departure can only GROW
    # the budget; it can never silently shrink the screen.
    rc0 = _resolved(launch="LEO", margin=1.2)
    rc = _resolved(launch="LEO", margin=1.2)
    rc.departure_vinf_kms = 1.5
    assert math.isclose(rc.dv_budget, (rc.cruise_dv_limit + 1.5) * 1.2, rel_tol=1e-12)
    assert rc.dv_budget > rc0.dv_budget


def test_higher_vinf_means_longer_spiral_and_lighter_cruise_start():
    # A faster departure takes a longer, hungrier spiral (estimate path): the departure window
    # shifts further out and the cruise starts lighter.
    rc0 = _resolved(launch="LEO")
    rc2 = _resolved(launch="LEO")
    rc2.departure_vinf_kms = 2.0
    assert rc2.escape_tof_days > rc0.escape_tof_days
    assert rc2.departure_window[0] > rc0.departure_window[0]
    assert rc2.cruise_start_mass_kg < rc0.cruise_start_mass_kg


def test_cruise_starts_at_post_spiral_mass():
    # The spiral burns real propellant: the cruise solvers start lighter than liftoff.
    tli = _resolved(launch="TLI")
    assert tli.escape_propellant_kg == 0.0
    assert tli.cruise_start_mass_kg == tli.vehicle.wet_mass     # rocket escape: untouched

    rc = _resolved(launch="LEO")
    # Estimate fallback: rocket equation at the escape estimate, so the dV chain and the mass chain
    # agree, Tsiolkovsky from the cruise-start mass equals the cruise limit.
    assert 0 < rc.escape_propellant_kg < rc.vehicle.fuel_mass
    remaining = G0_KM_S2 * 2000 * math.log(rc.cruise_start_mass_kg / rc.vehicle.burnout_mass)
    assert math.isclose(remaining, rc.cruise_dv_limit, rel_tol=1e-9)

    # A matching spiral run's reported burn overrides the estimate.
    rc.escape_prop_refined = 233.0
    assert math.isclose(rc.cruise_start_mass_kg, rc.vehicle.wet_mass - 233.0, rel_tol=1e-12)


def test_vehicle_propellant_scales_the_resolved_engine():
    # A vehicle that loads a non-native gas resolves to the SCALED thrust/Isp (the same conversion
    # the sweep applies), so a trajectory planned from it uses that gas, not xenon. None leaves the
    # engine on its authored gas.
    from prospector.spacecraft.propellants import load_propellants

    cat = load_propellants()
    native = Vehicle(name="v", dry_mass=500, fuel_mass=500,
                     engines=[EngineMount(type="E", count=2)])
    kr = Vehicle(name="v", dry_mass=500, fuel_mass=500, propellant="krypton",
                 engines=[EngineMount(type="E", count=2)])
    rc_n = ResolvedConfig.build(Mission(launch_orbit="TLI"), native, Screening(), CAT,
                                launches=LAUNCHES)
    rc_k = ResolvedConfig.build(Mission(launch_orbit="TLI"), kr, Screening(), CAT,
                                launches=LAUNCHES)
    assert rc_n.effective_isp == pytest.approx(2000)            # native xenon, unscaled
    assert rc_k.effective_isp == pytest.approx(2000 * cat["krypton"].isp_scale)
    assert rc_k.total_thrust_mN == pytest.approx(2 * 100 * cat["krypton"].thrust_scale)
    assert rc_k.engines["E"].propellant == "krypton"


def test_scaled_engine_propagates_to_config():
    # A non-native gas injected into the catalog (the sweep's mechanism) must move the resolved
    # performance: effective Isp and total thrust both scale by the gas factors.
    from prospector.spacecraft.propellants import engine_on_propellant, load_propellants

    cat = load_propellants()
    scaled, estimated = engine_on_propellant(CAT["E"], "krypton", cat)
    assert estimated
    catalog = {"E": scaled}
    vehicle = Vehicle(name="v", dry_mass=500.0, fuel_mass=500.0,
                      engines=[EngineMount(type="E", count=2)])
    rc = ResolvedConfig.build(Mission(launch_orbit="TLI"),
                              vehicle, Screening(dv_margin_factor=1.0), catalog,
                              launches=LAUNCHES)
    assert rc.effective_isp == pytest.approx(2000 * cat["krypton"].isp_scale)
    assert rc.total_thrust_mN == pytest.approx(2 * 100 * cat["krypton"].thrust_scale)


# ---------------------------------------------------------------------------
# the escape -> cruise seam
# ---------------------------------------------------------------------------

def _scheduled(escape_days: float, launch_window=(date(2028, 1, 1), date(2028, 3, 31))):
    """A config whose escape takes a known number of days, so the handover arithmetic is checkable.

    A SEP launch type ("LEO"), not an LV-provided escape: with escape provided the duration is zero
    by definition and a refined value is correctly ignored.
    """
    rc = _resolved(launch="LEO")
    rc = rc.model_copy(update={
        "mission": rc.mission.model_copy(update={"launch_window": launch_window}),
        "escape_tof_refined": escape_days, "escape_dv_refined": 6.0,
        "escape_prop_refined": 100.0})
    return rc


def test_the_cruise_window_is_the_liftoff_window_shifted_by_the_escape():
    """The relationship the schedule rests on, and the reason a departure date appears to move for
    no reason: the solver plans in a window derived from the liftoff window, not in it."""
    rc = _scheduled(200.0)
    sched = rc.departure_schedule()
    assert (sched["launch_open"], sched["launch_close"]) == rc.mission.launch_window
    assert sched["cruise_open"] == date(2028, 1, 1) + timedelta(days=200)
    assert sched["cruise_close"] == date(2028, 3, 31) + timedelta(days=200)
    assert (sched["cruise_open"], sched["cruise_close"]) == rc.departure_window
    # Nothing about the handover is known until a cruise departure exists.
    assert sched["coast_days"] is None and sched["liftoff_if_no_coast"] is None


def test_departing_at_the_earliest_handover_means_no_wait():
    # The common case: the cruise leaves the moment the spiral can hand it over.
    rc = _scheduled(200.0)
    sched = rc.departure_schedule(rc.departure_window[0])
    assert sched["coast_days"] == 0.0
    assert sched["liftoff_if_no_coast"] == date(2028, 1, 1)
    assert sched["liftoff_in_window"] is True


def test_a_later_departure_is_reported_as_slack_not_absorbed():
    """The bug this exists to fix: a cruise departing later than the earliest handover used to be
    shown by silently sliding the liftoff date. The slack is a real schedule item and both ways of
    spending it are reported."""
    rc = _scheduled(200.0)
    sched = rc.departure_schedule(rc.departure_window[0] + timedelta(days=41))
    assert sched["coast_days"] == 41.0
    # Launching 41 days later removes the slack entirely, and that liftoff is inside the window.
    assert sched["liftoff_if_no_coast"] == date(2028, 2, 11)
    assert sched["liftoff_in_window"] is True


def test_a_departure_the_liftoff_window_cannot_reach_is_flagged():
    # Past the window's close the no-coast liftoff falls outside it, so that option is not
    # available and the caller must be told rather than shown an impossible date as if it were
    # fine.
    rc = _scheduled(200.0)
    sched = rc.departure_schedule(rc.departure_window[1] + timedelta(days=10))
    assert sched["liftoff_in_window"] is False
    assert sched["coast_days"] == float((rc.departure_window[1] - rc.departure_window[0]).days) + 10


def test_a_departure_before_the_handover_never_reports_negative_slack():
    # A cruise epoch earlier than the escape can deliver is not "negative waiting"; the slack
    # floors at zero so no consumer ever renders a negative duration.
    rc = _scheduled(200.0)
    sched = rc.departure_schedule(rc.departure_window[0] - timedelta(days=30))
    assert sched["coast_days"] == 0.0


def test_a_launch_vehicle_escape_has_no_seam_to_explain():
    # With escape provided by the launch vehicle the spiral takes no time, so the cruise window is
    # the liftoff window and liftoff coincides with departure.
    rc = _resolved(launch="TLI")
    assert rc.escape_tof_days == 0.0
    sched = rc.departure_schedule(rc.mission.launch_window[0])
    assert sched["cruise_open"] == sched["launch_open"]
    assert sched["coast_days"] == 0.0
    assert sched["liftoff_if_no_coast"] == sched["launch_open"]
