"""The written sections of the feasibility document.

Each builder takes a resolved config plus whatever results exist and returns a
:class:`~prospector.reporting.document.ReportSection`: the words, the numbers, the tables, and the
captions for figures the caller has already built. Rendering is
:mod:`prospector.reporting.document`'s job; deciding which sections have data is the caller's.

The document is meant to read as an analysis of a vehicle and a trajectory in its own right, so the
text names the methods and cites the literature rather than describing the software that produced
it. Where a number is an estimate, a placeholder or an optimistic bound, the section says so. The
assumptions section exists so that nothing in the document can be quoted without its caveat.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go

from prospector.config import ResolvedConfig
from prospector.figures import products
from prospector.reporting import wording as W
from prospector.reporting.document import ReportSection
from prospector.spacecraft.arrays import EXAMPLE_MASS_CURVE


def parameters_section(rc: ResolvedConfig) -> ReportSection:
    """The parameters appendix: the exact mission, spacecraft, and launch inputs the analysis
    used, with the derived ΔV figures, so the document is fully specified on its face."""
    m, v = rc.mission, rc.vehicle
    assembly = " + ".join(f"{count} × {rc.engines[key].name}" for key, count in v.mounts)
    gas = v.propellant or "engine-native (xenon)"
    dep0, dep1 = rc.departure_window

    rows: list[tuple[str, str | None]] = [
        ("Mission", None),
        ("Name", m.name),
        ("Liftoff window", f"{m.launch_window[0].isoformat()} to {m.launch_window[1].isoformat()}"),
        ("Arrive by", m.arrive_by.isoformat()),
        ("Launch type", f"{rc.launch.name}"),
        ("Return trip", "yes" if m.return_trip else "no"),
    ]
    if m.return_trip:
        if m.return_by is not None:
            rows.append(("Return by", m.return_by.isoformat()))
        rows.append(("Collected payload", f"{m.asteroid_payload_mass:,.0f} kg"))

    rows += [
        ("Spacecraft", None),
        ("Engine assembly", assembly),
        ("Working propellant", gas),
        ("Effective specific impulse", f"{rc.effective_isp:,.0f} s"),
        ("Total thrust", f"{rc.total_thrust_mN:,.0f} mN"),
        ("Total electrical power", f"{rc.total_power_W:,.0f} W"),
        ("Dry mass", f"{v.dry_mass:,.1f} kg"),
        ("Propellant", f"{v.fuel_mass:,.1f} kg"
         + (f" ({v.unusable_prop:,.1f} kg unusable)" if v.unusable_prop > 0 else "")),
        ("Wet mass at liftoff", f"{v.wet_mass:,.1f} kg"),
    ]

    rows += [
        ("Launch and Earth escape", None),
        ("Escape ΔV budget (rated-Isp equivalent)", f"{rc.escape_dv:.2f} km/s ({rc.escape_source})"),
        ("Escape duration", f"{rc.escape_tof_days:,.0f} days"),
        ("Escape propellant", f"{rc.escape_propellant_kg:,.1f} kg"),
        ("Mass at Earth departure", f"{rc.cruise_start_mass_kg:,.1f} kg"),
        # The window the trajectory solvers plan in: the liftoff window above, shifted by the
        # escape duration. Naming it as derived keeps the two windows from reading as independent
        # inputs, which is what makes a shifting departure date look unexplained.
        ("Cruise departure window",
         f"{dep0.isoformat()} to {dep1.isoformat()}"),
    ]

    rows += [
        ("Delta-v", None),
        ("Total ΔV capability", f"{rc.total_dv_capability:.2f} km/s"),
        ("ΔV available for cruise", f"{rc.cruise_dv_limit:.2f} km/s"),
    ]
    if rc.return_dv_capability is not None:
        rows.append(("Return ΔV capability (laden)", f"{rc.return_dv_capability:.2f} km/s"))

    return ReportSection(
        title="Mission and vehicle parameters",
        page_break_before=True,
        caption=W.PARAMETERS["caption"],
        table=rows,
    )


Figures = Sequence[tuple[str, go.Figure]]


def _km_s(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f} km/s"


def _kg(x: float | None, digits: int = 0) -> str:
    return "-" if x is None else f"{x:,.{digits}f} kg"


def _days(x: float | None) -> str:
    return "-" if x is None else f"{x:,.0f} days"


def _date_mjd2000(mjd2000: float) -> str:
    """MJD2000 (days since 2000-01-01) -> 'YYYY-MM-DD' calendar date."""
    return (datetime(2000, 1, 1) + timedelta(days=float(mjd2000))).strftime("%Y-%m-%d")


def _date_from_mjd2000(mjd2000: float):
    """MJD2000 -> a ``date``, for the schedule arithmetic on the resolved config."""
    return (datetime(2000, 1, 1) + timedelta(days=float(mjd2000))).date()


def _seam_clause(sched: dict) -> str:
    """Which launch date the quoted cruise departure implies.

    Told only "N days of spiral, then departure on this date", a reader has to do the subtraction
    themselves to find out when the vehicle leaves the pad. The two dates are one choice separated
    by the spiral's duration, so giving both completes the schedule. There is no gap between them
    to account for.
    """
    liftoff = sched.get("liftoff_if_no_coast")
    if liftoff is None:
        return ""
    if sched.get("liftoff_in_window"):
        return (f" That departure implies liftoff on {liftoff.isoformat()}, "
                f"{_days(sched['escape_days'])} earlier, and the escape hands straight over to "
                f"the cruise.")
    return (f" That departure would need liftoff on {liftoff.isoformat()}, which falls outside "
            f"the launch window ({sched['launch_open'].isoformat()} to "
            f"{sched['launch_close'].isoformat()}), so it is not reachable from the pad as "
            f"planned: it requires either a different departure or a slower spiral.")


def _working_gas(rc: ResolvedConfig, build: dict | None = None) -> str:
    """The propellant the spacecraft flies on, for display (title-cased)."""
    if build and build.get("propellant"):
        return str(build["propellant"])
    if rc.vehicle.propellant:
        return str(rc.vehicle.propellant).capitalize()
    gases = {getattr(e, "propellant", None) or "xenon" for e in rc.engines.values()}
    return gases.pop().capitalize() if len(gases) == 1 else "mixed"


def _assembly_label(rc: ResolvedConfig) -> str:
    """The engine assembly as 'N × Name (+ ...)' using the library display names."""
    return " + ".join(f"{count} × {rc.engines[key].name}" for key, count in rc.vehicle.mounts)


def _belt_remaining(power_fraction_end: float | None) -> float:
    """The single belt-degradation figure used throughout the report: the spiral's PHYSICAL
    end-of-life array-power fraction (the selectable radiation model integrated along the flown
    path), clamped to [0, 1]. This is the one number of record, quoted by the escape section, the
    array-power chart and the array sizing alike. It is purely the belt loss, with the separate
    wiring/transmission loss kept out of it. Returns 1.0 when no spiral fraction is available."""
    if power_fraction_end is None:
        return 1.0
    return max(0.0, min(1.0, float(power_fraction_end)))


def _ppu_wiring_clause(ppu_bus_side: str) -> str:
    """One plain-language clause saying which conversion stages the thrusters pay for.

    A power-processing unit fed from the high-voltage main bus never sees the step-down converter;
    one fed from the low-voltage bus pays it on top of its own loss. A stack can mount both, in
    which case the delivered fraction is the power-weighted blend of the two chains.
    """
    key = ppu_bus_side if ppu_bus_side in ("high", "low") else "mixed"
    return W.BUILDABILITY[f"ppu_wiring_{key}"]


def summary_section(rc: ResolvedConfig, *, spiral: dict | None = None,
                    sf: dict | None = None, build: dict | None = None,
                    target_name: str = "the target", figures: Figures = ()) -> ReportSection:
    """The opening summary: the headline numbers for the spacecraft and its journey on a single
    page, in plain language, ahead of the detailed cases that follow."""
    v = rc.vehicle
    gas = _working_gas(rc, build)
    usable = rc.usable_propellant_kg
    n_engines = sum(c for _, c in v.mounts)
    # The bare engine name (a trailing "(...)" descriptor is dropped so the card stays short).
    first_name = (rc.engines[v.mounts[0][0]].name.split("(")[0].strip() if v.mounts else "-")
    engine_card = (f"{n_engines} × {first_name}" if len({k for k, _ in v.mounts}) == 1
                   else f"{n_engines} thrusters")
    array_W = (build or {}).get("bol_power_W") or v.solar_power_W or 0.0

    if rc.launch.escape_provided:
        esc_value, esc_sub, esc_days, esc_prop = "n/a", "provided by launch vehicle", 0.0, 0.0
    elif spiral and spiral.get("dv_at_escape_kms") is not None:
        esc_days = spiral.get("tof_days")
        esc_prop = (spiral.get("initial_mass_kg", 0.0) - spiral.get("final_mass_kg", 0.0)) or None
        esc_value, esc_sub = _days(esc_days), f"{_kg(esc_prop)} propellant"
    else:
        esc_days, esc_prop = rc.escape_tof_days, rc.escape_propellant_kg
        esc_value, esc_sub = _days(esc_days), f"{_kg(esc_prop)} propellant (estimated)"

    cruise_dv = (sf or {}).get("dv_kms")
    cruise_days = (sf or {}).get("tof_days")
    cruise_prop = (sf or {}).get("propellant_kg")
    # Velocity change as flown (the spiral's integrated figure when one flew; the analytic estimate
    # otherwise) and the propellant it took; whether the trip closes is a question of kilograms.
    escape_dv = _flown_escape_dv(rc, spiral)
    used = None if cruise_dv is None else escape_dv + cruise_dv
    used_prop = (None if (cruise_prop is None or esc_prop is None)
                 else float(esc_prop) + float(cruise_prop))
    margin = None if used_prop is None else usable - used_prop

    cards = [
        (_kg(v.dry_mass), "Dry mass", "current best estimate"),
        (_kg(v.fuel_mass), f"Propellant · {gas}",
         (f"{usable:,.0f} kg usable" if v.unusable_prop > 0 else "")),
        (engine_card, "Electric thrusters", f"{rc.total_thrust_mN:,.0f} mN total"),
        (f"{array_W / 1000.0:.1f} kW" if array_W else "-", "Solar array (start of life)",
         f"powers {rc.total_power_W:,.0f} W of thrusters"),
        (esc_value, "Earth escape", esc_sub),
        (_days(cruise_days), f"Cruise to {target_name}",
         f"{_kg(cruise_prop)} propellant" if cruise_prop is not None else "not yet solved"),
    ]
    if used is not None:
        cards.append((f"{used:.1f} km/s", "Mission ΔV used", "escape + cruise, as flown"))
    if margin is not None:
        cards.append((f"{margin:+,.0f} kg", "Propellant margin", f"of {usable:,.0f} kg usable"))

    leg = (W.SUMMARY["leg_solved"].format(target_name=target_name, cruise_days=_days(cruise_days))
           if cruise_days is not None
           else W.SUMMARY["leg_unsolved"].format(target_name=target_name))
    escape_phrase = (W.SUMMARY["escape_by_lv"] if rc.launch.escape_provided else
                     W.SUMMARY["escape_by_spiral"].format(launch_name=rc.launch.name,
                                                          escape_days=_days(esc_days)))
    overview = [
        W.SUMMARY["opening"].format(leg=leg, escape_phrase=escape_phrase),
        W.SUMMARY["masses"].format(dry_mass=_kg(v.dry_mass), fuel_mass=_kg(v.fuel_mass), gas=gas),
    ]
    if used is not None and used_prop is not None:
        verdict = W.SUMMARY["verdict_comfortable" if (margin or 0) > 0 else "verdict_tight"]
        overview.append(W.SUMMARY["budget"].format(
            used=f"{used:.1f} km/s", used_prop=_kg(used_prop), usable=_kg(usable),
            verdict=verdict))
    else:
        overview.append(W.SUMMARY["budget_unsolved"])

    return ReportSection(
        title="Summary",
        cards=cards, body=overview, figures=list(figures))


def _gas_note(gas: str) -> str:
    """A short, gas-specific line on how the propellant choice affects tank sizing."""
    g = (gas or "").lower()
    for name in ("krypton", "xenon", "iodine"):
        if name in g:
            return W.SPACECRAFT[f"gas_note_{name}"]
    return ""


def spacecraft_section(rc: ResolvedConfig, figures: Figures = ()) -> ReportSection:
    """The spacecraft itself: its mass breakdown, its propulsion, and the velocity change the
    propellant load can produce, which is the basis both legs of the journey are flown on."""
    v = rc.vehicle
    assembly = _assembly_label(rc)
    usable = rc.usable_propellant_kg
    gas = _working_gas(rc)
    single_type = len({k for k, _ in v.mounts}) == 1

    masses = W.SPACECRAFT["masses_with_residual" if v.unusable_prop > 0 else "masses_all_usable"]
    mass_intro = masses.format(
        wet_mass=_kg(v.wet_mass, 1), dry_mass=_kg(v.dry_mass, 1), fuel_mass=_kg(v.fuel_mass, 1),
        unusable=_kg(v.unusable_prop, 1), usable=_kg(usable, 1),
        burnout_mass=_kg(v.burnout_mass, 1), capability=_km_s(rc.total_dv_capability))
    isp = W.SPACECRAFT["isp_single_type" if single_type else "isp_mixed_types"].format(
        isp=f"{rc.effective_isp:,.0f} seconds")
    body = [
        mass_intro,
        W.SPACECRAFT["thrust"].format(assembly=assembly,
                                      thrust=f"{rc.total_thrust_mN:,.0f} mN",
                                      power=f"{rc.total_power_W:,.0f} W") + isp,
    ]
    gas_note = _gas_note(gas)
    body.append(W.SPACECRAFT["propellant"].format(
        gas=gas, gas_note=(gas_note + " " if gas_note else "")))

    rows: list[tuple[str, str | None]] = [
        ("Propulsion", None),
        ("Engine assembly", assembly),
        ("Working propellant", gas),
        ("Effective specific impulse", f"{rc.effective_isp:,.0f} s"),
        ("Total thrust", f"{rc.total_thrust_mN:,.0f} mN"),
        ("Electrical power (full thrust)", f"{rc.total_power_W:,.0f} W"),
        ("Total ΔV capability (no payload)", _km_s(rc.total_dv_capability)),
        ("Mass", None),
        ("Dry spacecraft", _kg(v.dry_mass, 1)),
    ]
    if v.unusable_prop > 0:
        rows += [("Usable propellant", _kg(usable, 1)),
                 ("Unusable residual", _kg(v.unusable_prop, 1))]
    else:
        rows += [("Propellant", _kg(v.fuel_mass, 1))]
    rows += [("Wet mass at liftoff", _kg(v.wet_mass, 1))]
    notes = W.SPACECRAFT["notes"]
    if v.unusable_prop > 0:
        notes += W.SPACECRAFT["notes_residual"]
    return ReportSection(
        title="The spacecraft", caption=W.SPACECRAFT["caption"],
        body=body, table=rows, figures=list(figures), notes=notes)


def _measured_array_curve(m) -> bool:
    """Whether the build model's power-to-mass curve came from the library rather than the
    illustrative one the software falls back to. The array mass drives the payload capacity, so a
    report has to say which of the two it is quoting."""
    return list(m.array_mass_curve) != list(EXAMPLE_MASS_CURVE)


def buildability_section(rc: ResolvedConfig, build: dict | None = None,
                         figures: Figures = ()) -> ReportSection:
    """Whether the spacecraft can be built: how the dry mass divides across the real
    hardware a solar-electric bus needs, with the assumptions behind each sizing stated."""
    v = rc.vehicle
    n_engines = sum(c for _, c in v.mounts)
    gas = _working_gas(rc, build)
    if build is None:
        return ReportSection(
            title="Building the spacecraft", page_break_before=True,
            caption="How the dry mass divides across the bus.",
            body=["A component-level mass breakdown was not available for this configuration."])

    m = rc.bus_model()
    buildable = build.get("buildable")
    capacity = build.get("payload_capacity_kg")
    array_kg = build.get("array_kg")
    bol_W = build.get("bol_power_W", 0.0)
    belt_days = build.get("belt_days")
    per_thruster = build.get("prop_per_thruster_kg")

    # The belt loss, recovered from the build's eol_factor (= transmission x belt fraction), so
    # this section quotes the same belt loss as the escape section and the array-power chart. The
    # array is not sized against it (that is a flown result); it is quoted to explain the power the
    # panels lose after the climb, and whether the vehicle ends up flying power-limited.
    eol_factor = build.get("eol_factor")
    transmission = m.transmission_eff_pct / 100.0
    belt_remaining = _belt_remaining(eol_factor / transmission if eol_factor and transmission else None)
    belt_loss_pct = (1.0 - belt_remaining) * 100.0
    avionics_reach_pct = m.avionics_power_eff() * 100.0    # array->bus x HV/LV, reaches every system
    # The thrusters' chain depends on which bus side each engine's PPU is wired to, so it is read
    # off the assembly rather than assumed; a mixed stack blends the two chains.
    thruster_reach_pct = rc.thruster_chain_eff(m) * 100.0
    ppu_sides = {e.ppu_bus_side for e in rc.engines.values()}
    ppu_side = ppu_sides.pop() if len(ppu_sides) == 1 else "mixed"
    reserve = m.power_margin_pct if v.array_margin_pct is None else float(v.array_margin_pct)
    belt_clause = (W.BUILDABILITY["belt_known"].format(belt_loss=f"{belt_loss_pct:.0f}%",
                                                       belt_days=f"{belt_days:.0f} days")
                   if belt_days else W.BUILDABILITY["belt_unknown"])
    # The margin is a reserve on the worst-mode beginning-of-life load, not an end-of-life
    # gross-up, and a negative margin means an array undersized on purpose, to fly power-limited.
    if reserve >= 0.5:
        margin_note = W.BUILDABILITY["margin_reserve"].format(reserve=f"{reserve:.0f}%")
    elif reserve <= -0.5:
        margin_note = W.BUILDABILITY["margin_undersized"].format(reserve=f"{reserve:.0f}%")
    else:
        margin_note = W.BUILDABILITY["margin_none"]

    body = [
        W.BUILDABILITY["opening"].format(dry_mass=_kg(v.dry_mass)),
        W.BUILDABILITY["arrays"].format(
            bol_power=f"{bol_W:,.0f} W", array_mass=_kg(array_kg),
            converter_eff=f"{m.hvlv_converter_eff_pct:.0f}%",
            transmission_eff=f"{m.transmission_eff_pct:.0f}%",
            avionics_reach=f"{avionics_reach_pct:.0f}%", ppu_eff=f"{m.ppu_eff_pct:.0f}%",
            ppu_wiring=_ppu_wiring_clause(ppu_side),
            thruster_reach=f"{thruster_reach_pct:.0f}%",
            margin_note=margin_note, belt_clause=belt_clause),
        W.BUILDABILITY["array_mass_curve"].format(
            specific_power=f"{bol_W / array_kg if array_kg else 0.0:,.0f} watts",
            illustrative=("" if _measured_array_curve(m) else
                          " This figure comes from the illustrative curve built into the software,"
                          " not from a measured array: the build model carries no array_mass_curve,"
                          " so the array mass and the payload capacity below are placeholders.")),
    ]
    tank_note = _gas_note(gas)
    body.append(W.BUILDABILITY["tank"].format(
        gas=gas, shape_ratio=f"{m.tank_length_diameter_ratio:.1f}:1",
        gas_note=(tank_note + " " if tank_note else ""),
        tank_mass=_kg(build.get("tank_kg")), pressure=f"{m.tank_pressure_bar:.0f} bar",
        safety_factor=f"{m.tank_safety_factor:.1f}\u00d7"))
    if build.get("lifetime_ok") is False:
        body.append(W.BUILDABILITY["life_exceeded"].format(
            n_engines=n_engines, per_thruster=f"{per_thruster:,.0f} kg",
            why=build.get("lifetime_why", "")))
    elif build.get("lifetime_ok") is True:
        body.append(W.BUILDABILITY["life_ok"].format(
            n_engines=n_engines, per_thruster=f"{per_thruster:,.0f} kg"))

    if buildable:
        body.append(W.BUILDABILITY["closes"].format(
            dry_mass=_kg(v.dry_mass), capacity=_kg(capacity)))
    else:
        body.append(W.BUILDABILITY["does_not_close"].format(
            dry_mass=_kg(v.dry_mass), min_dry=_kg(build.get("min_dry_kg"))))

    rows: list[tuple[str, str | None]] = [
        ("Dry-mass allocation", None),
        ("Solar array", _kg(array_kg)),
        ("Power distribution", _kg(build.get("pcdu_kg"))),
        (f"Thrusters (×{n_engines})", _kg(build.get("thruster_sys_kg"))),
        ("Propellant tank", _kg(build.get("tank_kg"))),
        ("Avionics & bus", _kg(v.dry_mass * m.fixed_bus_ratio_pct / 100.0)),
        ("Structure", _kg(v.dry_mass * m.structure_ratio_pct / 100.0)),
        ("Thermal & harness",
         _kg(v.dry_mass * (m.thermal_ratio_pct + m.harness_ratio_pct) / 100.0)),
        ("Payload + margin", _kg(capacity)),
        ("Smallest dry mass that closes", _kg(build.get("min_dry_kg"))),
    ]
    notes = W.BUILDABILITY["notes"]
    return ReportSection(
        title="Building the spacecraft", page_break_before=True,
        caption=W.BUILDABILITY["caption"].format(dry_mass=_kg(v.dry_mass)),
        body=body, table=rows, figures=list(figures), notes=notes)


def escape_section(rc: ResolvedConfig, spiral: dict | None = None,
                   figures: Figures = ()) -> ReportSection:
    """Earth escape: how the spacecraft climbs out of Earth's gravity on its own thrusters,
    what shapes the climb, and what it costs. ``spiral`` is a serialized propagated-escape
    result; ``None`` or a launch-vehicle-provided escape degrade to a short note."""
    if rc.launch.escape_provided:
        return ReportSection(
            title="Leaving Earth", page_break_before=True,
            caption=W.ESCAPE["caption_by_launch_vehicle"],
            body=[W.ESCAPE["by_launch_vehicle"].format(launch_name=rc.launch.name)])

    vinf0 = float(rc.departure_vinf_kms)
    near_zero = vinf0 < 0.05
    est_tail = (W.ESCAPE["estimate_tail_zero_vinf"] if near_zero else
                W.ESCAPE["estimate_tail_with_vinf"].format(vinf=f"{vinf0:.2f} km/s"))
    body = [
        W.ESCAPE["why_a_spiral"].format(estimate_tail=est_tail,
                                        estimate=_km_s(rc.escape_dv_estimate)),
        W.ESCAPE["forces"],
        W.ESCAPE["belts"],
    ]

    rows: list[tuple[str, str | None]] = []
    notes = W.ESCAPE["notes"]
    footnotes = list(W.ESCAPE["footnotes"])

    if spiral is None:
        body.append(W.ESCAPE["not_propagated"])
    else:
        status = spiral.get("status")
        escaped = spiral.get("dv_at_escape_kms") is not None
        prop = None
        if spiral.get("initial_mass_kg") is not None and spiral.get("final_mass_kg") is not None:
            prop = spiral["initial_mass_kg"] - spiral["final_mass_kg"]
        revs = spiral.get("revolutions")
        sp_vinf = float(spiral.get("vinf_kms", 0.0))
        sp_near_zero = sp_vinf < 0.05
        laps = "" if revs is None else W.ESCAPE["laps"].format(revolutions=f"{revs:,.0f}")
        if not escaped:
            body.append(W.ESCAPE["did_not_escape"].format(status=status.replace("_", " ")))
        elif sp_near_zero:
            body.append(W.ESCAPE["escaped"].format(
                dv_at_escape=_km_s(spiral.get("dv_at_escape_kms")),
                time_to_escape=_days(spiral.get("tof_at_escape_days")),
                laps=laps, propellant=_kg(prop)))
        else:
            body.append(W.ESCAPE["escaped_with_vinf"].format(
                dv_at_escape=_km_s(spiral.get("dv_at_escape_kms")),
                time_to_escape=_days(spiral.get("tof_at_escape_days")), laps=laps,
                vinf=f"{sp_vinf:.2f} km/s", dv_total=_km_s(spiral.get("dv_kms")),
                propellant=_kg(prop)))
        if spiral.get("power_limited"):
            body.append(W.ESCAPE["power_limited"])
        # One belt-degradation figure for the whole report (purely the belt loss; the separate
        # wiring/transmission loss is accounted for in the array sizing, not here).
        belt_remaining = _belt_remaining(spiral.get("power_fraction_end"))
        rows = [("Velocity change to escape", _km_s(spiral.get("dv_at_escape_kms")))]
        if not sp_near_zero:
            rows.append((f"To {sp_vinf:.2f} km/s departure speed", _km_s(spiral.get("dv_kms"))))
        rows += [
            ("Time to escape", _days(spiral.get("tof_at_escape_days"))),
            ("Max duty cycle", ("-" if spiral.get("duty_cycle") is None
                                else f"{float(spiral['duty_cycle']) * 100:.0f}%")),
            ("Total time leaving Earth", _days(spiral.get("tof_days"))),
            ("Propellant used", _kg(prop)),
            ("Laps of Earth", "-" if revs is None else f"{revs:,.0f}"),
            ("Days in the radiation belts", _days(spiral.get("belt_days"))),
            ("Days in eclipse", _days(spiral.get("eclipse_days"))),
            ("Panel power at end of escape (belt loss)", f"{belt_remaining * 100:.0f}% of start of life"),
            ("Final orbit tilt",
             "-" if spiral.get("inc_deg_end") is None else f"{spiral['inc_deg_end']:.1f}°"),
        ]
        if spiral.get("radiation_model"):
            cover = spiral.get("coverglass_um") or 0.0
            dens = spiral.get("coverglass_density_g_cm3") or 0.0
            if cover and dens:
                cover_txt = (f", {cover:.0f} µm cover @ {dens:.2f} g/cm³ "
                             f"({cover * dens * 0.1:.1f} mg/cm²)")
            elif cover:
                cover_txt = f", {cover:.0f} µm cover"
            else:
                cover_txt = ""
            rows.append(("Radiation model", f"{spiral['radiation_model']}{cover_txt}"))
        if spiral.get("belt_days") and spiral["belt_days"] > 60:
            notes += W.ESCAPE["notes_long_belt_dwell"].format(
                belt_days=f"{spiral['belt_days']:.0f} days")

    return ReportSection(
        title="Leaving Earth",
        page_break_before=True,
        caption=W.ESCAPE["caption"],
        body=body, table=rows or None, figures=list(figures), notes=notes, footnotes=footnotes)


def cruise_section(rc: ResolvedConfig, sf: dict | None = None,
                   figures: Figures = (), *, target_name: str = "the target",
                   verification: dict | None = None) -> ReportSection:
    """The heliocentric cruise: how the powered transfer is designed and why a converged
    result is flyable, with the run's numbers and figures. ``sf`` is a serialized low-thrust
    solution; ``verification`` (when supplied) carries the thrust-history propellant cross-check
    (``avg_throttle``, ``thrust_N``, ``integral_prop_kg``, ``reported_prop_kg``)."""
    nseg = (sf or {}).get("nseg", 15)
    vinf_arr = (sf or {}).get("vinf_arr_kms", 0.1)
    arrive_by = rc.mission.arrive_by.isoformat()

    cvinf = float(rc.departure_vinf_kms)
    c_near_zero = cvinf < 0.05
    depart_state = W.CRUISE["depart_state_zero_vinf" if c_near_zero else "depart_state_with_vinf"]
    vinf_arrival = f"{vinf_arr:.2f} km/s"
    constraints = W.CRUISE["constraints_zero_vinf" if c_near_zero else "constraints_with_vinf"] \
        .format(vinf_arrival=vinf_arrival, arrive_by=arrive_by)
    body = [
        W.CRUISE["opening"].format(depart_state=depart_state, target_name=target_name,
                                   arrive_by=arrive_by),
        W.CRUISE["model"].format(nseg=nseg),
        W.CRUISE["why_flyable"],
        W.CRUISE["global_search"],
        constraints,
        W.CRUISE["why_low_thrust_wins"],
    ]

    notes = W.CRUISE["notes"].format(nseg=nseg, vinf_arrival=vinf_arrival)
    footnotes = list(W.CRUISE["footnotes"])

    rows: list[tuple[str, str | None]] = []
    if sf is None:
        body.append(W.CRUISE["not_solved"])
    else:
        dep = sf.get("dep_mjd2000")
        tof = sf.get("tof_days")
        mismatch = sf.get("mismatch")
        converged = bool(sf.get("feasible")) or (mismatch is not None and mismatch < 1e-3)
        if converged and verification:
            ip = verification.get("integral_prop_kg")
            rp = verification.get("reported_prop_kg")
            avg = verification.get("avg_throttle")
            if ip and rp:
                agree = abs(ip - rp) / rp * 100.0
                close = (W.CRUISE["agreement_exact"].format(reported=f"{rp:,.1f} kg")
                         if agree < 0.5 else
                         W.CRUISE["agreement_within"].format(error=f"{agree:.1f}%",
                                                             reported=f"{rp:,.1f} kg"))
                body.append(W.CRUISE["propellant_check"].format(
                    avg_throttle=f"{(avg or 0) * 100:.0f}%",
                    thrust=f"{verification.get('thrust_N', 0):.3f} N",
                    integral_propellant=f"{ip:,.1f} kg", agreement=close))
        rows = [
            ("Cruise velocity change", _km_s(sf.get("dv_kms"))),
            ("Flight time",
             "-" if tof is None else f"{tof:,.0f} days ({tof / 365.25:.2f} yr)"),
            ("Departure", "-" if dep is None else _date_mjd2000(dep) + (
                f" (at {sf.get('vinf_dep_kms', 0.0):.2f} km/s past Earth)"
                if sf.get("vinf_dep_kms", 0.0) >= 0.05 else "")),
            ("Arrival", "-" if dep is None or tof is None else _date_mjd2000(dep + tof)),
            ("Mass at departure", _kg(sf.get("initial_mass_kg"))),
            ("Mass at arrival", _kg(sf.get("final_mass_kg"))),
            ("Cruise propellant", _kg(sf.get("propellant_kg"))),
            ("Segments modeled", str(nseg)),
        ]
        fb = sf.get("flyby")
        if fb:
            body = str(fb.get("body", "planet")).capitalize()
            rows.insert(4, (f"{body} flyby",
                            f"{_date_mjd2000(fb['mjd2000'])} at {fb['periapsis_alt_km']:,.0f} km, "
                            f"{fb['vinf_kms']:.2f} km/s relative, turned {fb['turn_deg']:.0f}°"))
            direct = sf.get("direct") or {}
            if direct.get("propellant_kg") is not None:
                rows.append(("Flying direct instead", _kg(direct["propellant_kg"])))
        if not converged:
            body.append(W.CRUISE["not_converged"])

    return ReportSection(
        title=f"The cruise to {target_name}",
        caption=W.CRUISE["caption"],
        body=body, table=rows or None, figures=list(figures), notes=notes, footnotes=footnotes)


def _flown_escape_dv(rc: ResolvedConfig, spiral: dict | None) -> float:
    """The escape's velocity change for a reader: the spiral's integrated figure when one flew
    (or zero when the launch vehicle escapes), else the analytic estimate. Not ``rc.escape_dv``,
    which is the budget's propellant-equivalent at rated Isp."""
    if rc.launch.escape_provided:
        return 0.0
    if spiral and spiral.get("dv_at_escape_kms") is not None:
        return float(spiral["dv_at_escape_kms"])
    return float(rc.escape_dv)


def _flown_escape_propellant(rc: ResolvedConfig, spiral: dict | None) -> float:
    """The propellant the escape took: the spiral's mass history when one flew, else the estimate."""
    if rc.launch.escape_provided:
        return 0.0
    if spiral and spiral.get("initial_mass_kg") is not None and spiral.get("final_mass_kg") is not None:
        return float(spiral["initial_mass_kg"]) - float(spiral["final_mass_kg"])
    return float(rc.escape_propellant_kg)


def closure_section(rc: ResolvedConfig, *, spiral: dict | None = None,
                    sf: dict | None = None, figures: Figures = ()) -> ReportSection:
    """The end-to-end propellant and velocity-change ledger, and whether the journey fits
    inside the spacecraft's capability. Reads the budget from the resolved configuration and
    the solved cruise from ``sf``; a cumulative-propellant figure is passed in."""
    # Velocity change as flown, for the reader; kilograms for the verdict. The two legs run at
    # different Isps (and the escape at a changing one), so a single-Isp dV capability no longer
    # describes the trip; the propellant does.
    escape_dv = _flown_escape_dv(rc, spiral)
    cruise_dv = (sf or {}).get("dv_kms")
    total_dv = None if cruise_dv is None else escape_dv + cruise_dv

    escape_prop = _flown_escape_propellant(rc, spiral)
    ledger = products.closure_ledger(rc, sf, None, escape_propellant_kg=escape_prop)
    cruise_prop = ledger["cruise_kg"]
    used_prop, usable, margin = ledger["used_kg"], ledger["usable_kg"], ledger["margin_kg"]
    closes = ledger["closes"]

    body = [
        W.CLOSURE["velocity_ledger"].format(
            escape_dv=_km_s(escape_dv), cruise_dv=_km_s(cruise_dv), total_dv=_km_s(total_dv)),
        W.CLOSURE["propellant_ledger"].format(
            escape_prop=_kg(escape_prop), cruise_prop=_kg(cruise_prop),
            used_prop=_kg(used_prop), usable=_kg(usable),
            reserve_clause=(W.CLOSURE["no_reserve_clause"] if margin is None
                            else W.CLOSURE["reserve_clause"].format(margin=_kg(margin))
                            if margin >= 0.0
                            else W.CLOSURE["short_clause"].format(short=_kg(-margin)))),
    ]
    if sf is not None and sf.get("dep_mjd2000") is not None:
        dep = sf["dep_mjd2000"]
        tof = sf.get("tof_days", 0.0)
        sched = rc.departure_schedule(_date_from_mjd2000(dep))
        body.append(W.CLOSURE["schedule"].format(
            launch_open=sched["launch_open"].isoformat(),
            launch_close=sched["launch_close"].isoformat(),
            escape_days=_days(rc.escape_tof_days), departure=_date_mjd2000(dep),
            cruise_days=_days(tof), arrival=_date_mjd2000(dep + tof),
            deadline=rc.mission.arrive_by.isoformat()) + _seam_clause(sched))
    body.append(W.CLOSURE["methods"].format(
        verdict=W.CLOSURE["verdict_closes" if closes else "verdict_does_not_close"]))

    rows: list[tuple[str, str | None]] = [
        ("Velocity change (as flown)", None),
        ("Earth escape", _km_s(escape_dv)),
        ("Cruise", _km_s(cruise_dv)),
        ("Total for the mission", _km_s(total_dv)),
        ("Propellant", None),
        ("Earth escape", _kg(escape_prop)),
        ("Cruise", _kg(cruise_prop)),
        ("Total used", _kg(used_prop)),
        ("Usable propellant aboard", _kg(usable)),
        ("Reserve", _kg(margin)),
    ]

    example = W.CLOSURE["example_zero_vinf" if rc.departure_vinf_kms < 0.05
                        else "example_with_vinf"]
    return ReportSection(
        title="Does it close? The propellant and velocity-change ledger",
        caption=W.CLOSURE["caption"],
        body=body, table=rows, figures=list(figures),
        notes=W.CLOSURE["notes"].format(example=example))


def assumptions_section(nseg: int | None = None, *,
                        departure_vinf_kms: float = 0.0) -> ReportSection:
    """The consolidated table of modeling simplifications, why each is acceptable for a
    preliminary design, and what would confirm it, plus the document's overall caveat.

    ``nseg`` is the segment count the cruise was solved at, and None when no cruise converged: the
    count then goes unstated rather than being filled in with a default, which put a number on the
    page that no solve had produced. The departure-speed row is dropped when
    ``departure_vinf_kms`` is essentially zero (it then has no bearing).
    """
    segments = (W.ASSUMPTIONS["cruise_segments_unsolved"] if not nseg
                else W.ASSUMPTIONS["cruise_segments_known"].format(nseg=int(nseg)))
    rows = [[cell.format(nseg=segments) for cell in row]
            for row in W.ASSUMPTIONS["rows_before_departure_speed"]]
    if departure_vinf_kms >= 0.05:
        rows.append(list(W.ASSUMPTIONS["row_departure_speed"]))
    rows += [list(row) for row in W.ASSUMPTIONS["rows_after_departure_speed"]]
    df = pd.DataFrame(rows, columns=list(W.ASSUMPTIONS["columns"]))
    return ReportSection(
        title="Assumptions and recommended verification",
        caption=W.ASSUMPTIONS["caption"], table=df, notes=W.ASSUMPTIONS["notes"])
