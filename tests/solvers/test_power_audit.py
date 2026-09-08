"""Regression tests from the thrust / Isp / power audit.

Each pins one finding that was verified against the code before it was fixed:

* the spiral handed the thrusters the array's whole output, though the array was sized to carry
  the bus's housekeeping on top of them;
* the spiral's config entry point left the array-to-thruster chain efficiency at 1.0 unless the
  caller remembered it, so raw array watts were compared against thruster watts;
* a cruise leg the array could not power at all fell back to RATED thrust with no limits;
* the sun-power model has to keep the housekeeping reserve fixed while the array output scales;
* the ledger recomputed a cruise's propellant from its delta-v at the RATED Isp, understating a
  power-limited cruise whose delta-v is defined with its own lower Isp;
* the report quoted a hardcoded duty cycle.
"""
import math

import numpy as np
import pytest

from prospector.config import EngineMount, Mission, ResolvedConfig, Screening, Vehicle
from prospector.figures import products
from prospector.launch import LaunchOrbit, escape_fingerprint
from prospector.reporting.sections import escape_section
from prospector.solvers import simsflanagan as sf
from prospector.solvers import spiral, sunpower
from prospector.spacecraft.propulsion import Engine, PowerPoint

LEO = LaunchOrbit(name="leo", perigee_alt_km=400, apogee_alt_km=400, inclination_deg=28.5)
MODES = [PowerPoint(power_W=3000, thrust_mN=150.0, isp_s=1700.0),
         PowerPoint(power_W=4000, thrust_mN=200.0, isp_s=1800.0),
         PowerPoint(power_W=5000, thrust_mN=250.0, isp_s=1900.0)]
STEPPED = Engine(name="E", isp_s=1900.0, thrust_mN=250.0, power_W=5000.0, power_curve=MODES,
                 throttle_mode="discrete")


def _rc(engine: Engine = STEPPED, solar_power_W: float = 9296.0) -> ResolvedConfig:
    veh = Vehicle(name="v", dry_mass=333, fuel_mass=320, solar_power_W=solar_power_W,
                  engines=[EngineMount(type="E", count=1)])
    return ResolvedConfig.build(Mission(launch_orbit="LEO"), veh, Screening(), {"E": engine},
                                launches={"LEO": LEO})


class _InverseSquare:
    def power_fraction(self, r_au, *_args):
        r = np.asarray(r_au, float)
        return 1.0 / (r * r)


# ---------------------------------------------------------------------------
# the bus keeps its housekeeping power
# ---------------------------------------------------------------------------

def test_the_array_reserve_is_the_housekeeping_load_through_the_avionics_chain():
    """What the array was sized to carry for the bus (buildability.bol_power_terms: housekeeping /
    avionics_eff) is exactly what the spiral and the cruise must now take off the top."""
    from prospector.spacecraft.buildability import load_bus_model
    bm = load_bus_model()
    rc = _rc()
    assert rc.array_reserve_W(bm) == pytest.approx(bm.housekeeping_W / bm.avionics_power_eff())
    assert rc.array_reserve_W(bm) > 0.0


def test_the_spiral_takes_the_reserve_off_the_array_before_the_chain():
    """Power-rich array, so both runs fly the same trajectory and the end-of-life delivered power
    differs by exactly the reserve, through the chain, at the array's end-of-life fraction."""
    kw = dict(perigee_alt_km=20000.0, apogee_alt_km=20000.0, inclination_deg=10.0,
              mass_kg=120.0, dry_mass_kg=60.0, thrust_N=60.0e-3, isp_s=1500.0,
              power_req_W=1000.0, bol_power_W=6000.0, area_m2=0.0,
              target_vinf_kms=0.0, max_years=4.0, array_to_thruster_eff=0.8)
    plain = spiral.solve(array_reserve_W=0.0, **kw)
    reserved = spiral.solve(array_reserve_W=500.0, **kw)
    assert plain.status == "escaped" and reserved.status == "escaped"
    assert plain.propellant_kg == pytest.approx(reserved.propellant_kg, rel=1e-6)  # same flight
    # Same trajectory, so the same end-of-life array fraction f: plain delivers bol*f*eff and the
    # reserved run (bol*f - R)*eff. The reserve is a fixed draw at the array, not a share of its
    # output, so it does not degrade with the cells: the drop is exactly R*eff.
    assert plain.power_available_end_W - reserved.power_available_end_W == pytest.approx(
        500.0 * 0.8, rel=1e-6)
    assert plain.power_available_end_W == pytest.approx(6000.0 * plain.power_fraction_end * 0.8,
                                                        rel=1e-3)


def test_solve_for_config_defaults_the_chain_efficiency_and_the_reserve(monkeypatch):
    """The config entry point flies every caller on the same power model: the chain efficiency the
    engines' wiring implies and the bus's reserve, unless the caller overrides them. It used to
    leave the efficiency at 1.0, comparing array watts straight against thruster watts."""
    rc = _rc()
    seen = {}

    def fake_solve(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(spiral, "solve", fake_solve)
    with pytest.raises(RuntimeError):
        spiral.solve_for_config(rc)
    assert seen["array_to_thruster_eff"] == pytest.approx(rc.thruster_chain_eff())
    assert seen["array_reserve_W"] == pytest.approx(rc.array_reserve_W())
    seen.clear()
    with pytest.raises(RuntimeError):
        spiral.solve_for_config(rc, array_to_thruster_eff=1.0, array_reserve_W=0.0)
    assert seen["array_to_thruster_eff"] == 1.0 and seen["array_reserve_W"] == 0.0


def test_the_reserve_is_in_the_spiral_fingerprint():
    rc = _rc()
    fp = escape_fingerprint(rc)
    assert fp["array_reserve_W"] == pytest.approx(rc.array_reserve_W())
    lean = escape_fingerprint(rc.model_copy(update={"vehicle": rc.vehicle}))
    assert lean == fp                                            # deterministic


def test_sun_power_model_keeps_the_reserve_fixed_while_the_array_scales():
    """3.8 kW reaches the thruster at 1 AU through a 0.8 chain with 200 W reserved at the array.
    Nearer the Sun the array output scales, the reserve does not."""
    rc = _rc()
    model = sunpower.SunPowerModel(rc, 3800.0, _InverseSquare(), chain_eff=0.8, reserve_W=200.0)
    assert model.power_W(1.0) == pytest.approx(3800.0)
    array_1au = 3800.0 / 0.8 + 200.0                              # 4950 W at the array
    assert model.power_W(0.9) == pytest.approx((array_1au / 0.81 - 200.0) * 0.8)
    assert model.power_W(1.2) == pytest.approx((array_1au / 1.44 - 200.0) * 0.8)
    far = model.power_W(20.0)                                    # the reserve eats it all
    assert far == 0.0
    # Built from a config: the 1 AU figure is the given at-thruster power, and without one it is
    # the start-of-life array less the reserve, through the chain.
    auto = sunpower.sun_power_model(rc, None, _InverseSquare())
    assert auto.base_power_W == pytest.approx(
        (rc.vehicle.solar_power_W - rc.array_reserve_W()) * rc.thruster_chain_eff())
    assert sunpower.sun_power_model(rc, 3800.0, _InverseSquare()).base_power_W == 3800.0


# ---------------------------------------------------------------------------
# a cruise the array cannot power fails, rather than flying at rated
# ---------------------------------------------------------------------------

def test_a_leg_the_array_cannot_power_gets_zero_ceilings_not_rated_thrust():
    """2.4 kW at the thruster at 1 AU on a stack whose lowest mode wants 3 kW: even at the closest
    approach the model allows (0.98 AU) the stack cannot run. The leg keeps its rated basis but
    every ceiling starts at zero, so the solve cannot pretend otherwise."""
    rc = _rc()
    from prospector.solvers import lambert as lb
    terms = sf._leg_power_terms(rc, (lb.earth_planet(), lb.earth_planet()), 2400.0,
                                _InverseSquare(), when_mjd2000=10000.0)
    assert terms["thrust_N"] == pytest.approx(0.250)
    assert np.all(terms["thrust_cap_fn"](np.array([0.5, 0.98, 1.0, 1.5])) == 0.0)
    isps = terms["seg_isp_fn"](np.array([[1.0, 0, 0], [1.0, 0.05, 0], [1.0, 0.1, 0]])
                               * 1.495978707e11)
    assert isps.shape == (1,)
    # With enough power the same call is the ordinary leg: a positive ceiling at 1 AU.
    terms = sf._leg_power_terms(rc, (lb.earth_planet(), lb.earth_planet()), 3800.0,
                                _InverseSquare(), when_mjd2000=10000.0)
    assert 0.0 < float(terms["thrust_cap_fn"](1.0)) <= 1.0


# ---------------------------------------------------------------------------
# the ledger spends what the leg spent
# ---------------------------------------------------------------------------

def test_cruise_propellant_is_the_solved_figure_not_a_rated_isp_inversion():
    rc = _rc()
    m0 = rc.cruise_start_mass_kg
    leg_isp, rated_isp = 1712.0, rc.effective_isp
    mf = m0 * 0.85
    dv = leg_isp * 9.80665e-3 * math.log(m0 / mf)             # the leg's dv, at ITS Isp
    sf_block = {"dv_kms": dv, "propellant_kg": m0 - mf, "isp_s": leg_isp}
    assert products.cruise_propellant_kg(rc, sf_block) == pytest.approx(m0 - mf)
    # The rated-Isp inversion the ledger used to do understates it when the leg Isp is lower.
    rated_inversion = m0 * (1.0 - math.exp(-dv / (rated_isp * 9.80665e-3)))
    assert rated_inversion < (m0 - mf) * 0.995
    # An older result without the solved figure falls back to that inversion; no result, None.
    assert products.cruise_propellant_kg(rc, {"dv_kms": dv}) == pytest.approx(rated_inversion)
    assert products.cruise_propellant_kg(rc, None) is None


# ---------------------------------------------------------------------------
# the velocity change flown is not the propellant in rated-Isp units
# ---------------------------------------------------------------------------

def test_spiral_reports_the_flown_dv_and_the_rated_equivalent_separately():
    """At constant rated Isp the two coincide (the integrated a·dt IS Isp·g0·ln(m0/m)); with a
    throttle curve that drops the Isp when power-limited, the flown velocity change is smaller
    than the propellant expressed at the rated Isp. The budget curve stays in the equivalent."""
    from prospector.spacecraft.propulsion import assembly_power_grid
    kw = dict(perigee_alt_km=20000.0, apogee_alt_km=20000.0, inclination_deg=10.0,
              mass_kg=120.0, dry_mass_kg=60.0, thrust_N=60.0e-3, isp_s=1500.0,
              power_req_W=1000.0, bol_power_W=500.0, area_m2=0.0, target_vinf_kms=0.0,
              max_years=4.0)
    flat = spiral.solve(**kw)                                  # no curve: rated Isp throughout
    assert flat.status == "escaped"
    assert flat.dv_kms == pytest.approx(flat.dv_equiv_kms, rel=1e-4)
    assert flat.dv_at_escape_kms == pytest.approx(flat.dv_at_escape_equiv_kms, rel=1e-4)
    # The equivalent is the rocket equation on the mass history, exactly.
    assert flat.dv_equiv_kms == pytest.approx(
        1500.0 * 9.80665 * math.log(flat.initial_mass_kg / flat.final_mass_kg) / 1e3, rel=1e-9)

    eng = Engine(name="C", isp_s=1500.0, thrust_mN=60.0, power_W=1000.0,
                 power_curve=[PowerPoint(power_W=400.0, thrust_mN=24.0, isp_s=1300.0),
                              PowerPoint(power_W=500.0, thrust_mN=30.0, isp_s=1350.0),
                              PowerPoint(power_W=1000.0, thrust_mN=60.0, isp_s=1500.0)])
    limited = spiral.solve(power_grid=assembly_power_grid([("C", 1)], {"C": eng}), **kw)
    assert limited.status == "escaped"
    # Ran at ~1350 s the whole way: the flown dV is ~1350/1500 of the rated-Isp figure.
    ratio = limited.dv_kms / limited.dv_equiv_kms
    assert 0.88 < ratio < 0.92
    assert limited.dv_at_escape_kms < limited.dv_at_escape_equiv_kms
    # The trade curve anchors at the EQUIVALENT escape figure, since it is read against the budget.
    assert limited.curve_dv_kms[0] == pytest.approx(limited.dv_at_escape_equiv_kms)


def test_closure_is_judged_in_kilograms_and_quotes_the_flown_dv():
    """A power-limited escape spent more propellant than its rated-Isp dV suggests. The report's
    ledger closes on the kilograms and shows the flown velocity change, and it must not call a
    trip that overspends the tank closed because a dV subtraction came out positive."""
    from prospector.reporting import wording as W
    from prospector.reporting.sections import closure_section
    rc = _rc()
    usable = rc.usable_propellant_kg
    spiral_block = {"status": "escaped", "dv_at_escape_kms": 7.0, "dv_at_escape_equiv_kms": 7.6,
                    "dv_kms": 7.0, "tof_days": 300.0, "tof_at_escape_days": 300.0,
                    "initial_mass_kg": rc.vehicle.wet_mass,
                    "final_mass_kg": rc.vehicle.wet_mass - 0.7 * usable}
    sf_block = {"dv_kms": 3.0, "tof_days": 400.0, "propellant_kg": 0.5 * usable,
                "dep_mjd2000": 10400.0}
    sec = closure_section(rc, spiral=spiral_block, sf=sf_block)
    rows = dict(sec.table)
    assert rows["Total for the mission"] == "10.00 km/s"          # 7.0 flown + 3.0, not 7.6
    assert "Spacecraft capability" not in rows                     # no single-Isp dV capability
    assert rows["Reserve"].startswith("-")                          # 120 % of the tank: short
    body = " ".join(sec.body)
    assert W.CLOSURE["verdict_does_not_close"] in body
    assert "more than is aboard" in body
    # Spend less and it closes, with the reserve in kilograms.
    sf_block["propellant_kg"] = 0.2 * usable
    sec = closure_section(rc, spiral=spiral_block, sf=sf_block)
    assert W.CLOSURE["verdict_closes"] in " ".join(sec.body)
    assert "in reserve" in " ".join(sec.body)


# ---------------------------------------------------------------------------
# the report quotes the duty cycle that ran
# ---------------------------------------------------------------------------

def test_report_quotes_the_flown_duty_cycle():
    rc = _rc()
    base = {"status": "escaped", "dv_at_escape_kms": 7.6, "dv_kms": 7.9, "tof_days": 120.0,
            "tof_at_escape_days": 110.0, "vinf_kms": 1.0, "target_vinf_kms": 1.0,
            "initial_mass_kg": 1000.0, "final_mass_kg": 820.0, "belt_days": 35.0,
            "eclipse_days": 12.0, "revolutions": 850.0, "power_fraction_end": 0.94,
            "inc_deg_end": 28.5}
    rows = dict(escape_section(rc, {**base, "duty_cycle": 0.9}).table)
    assert rows["Max duty cycle"] == "90%"
    rows = dict(escape_section(rc, base).table)
    assert rows["Max duty cycle"] == "-"                       # an older run: not recorded, not invented
