"""The application Model-settings window: edit the build/cost model and propellant library.

These belong to the application rather than to any project or vehicle: the numbers everything about
a vehicle other than its trajectory is worked out from, covering solar arrays, bus proportions,
tank and unit costs, plus the properties of each working gas: what it costs, how much tank it
needs, and the factors used to estimate an engine's performance on a gas it was not measured on.
They live in config files tracked in the repo (``configs/build-models/<profile>.yaml`` and
``configs/propellants/``), so editing them here changes what every project uses.

One tabbed dialog, opened from the top bar. Save writes each part out through its own library
writer (``buildability.save_bus_model`` and ``propellants.save_propellant``). Since the build model
is re-read on every assessment, edits take effect on the next render or sweep with no restart.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from nicegui import ui

from prospector.launch import (
    LaunchOrbit,
    ReturnDestination,
    delete_launch_orbit,
    delete_return_destination,
    load_launch_orbits,
    load_return_destinations,
    save_launch_orbit,
    save_return_destination,
)
from prospector.spacecraft import buildability
from prospector.spacecraft.propellants import list_propellants, load_propellants, save_propellant
from prospector.spacecraft.radiation import (
    RadiationModel,
    list_radiation_models,
    load_radiation_models,
    save_radiation_model,
)
from ui import state
from ui.state import S
from ui.theme import ACCENT, BORDER, MUTED, PANEL, PANEL2, TEXT

# The build-model fields grouped into tabs by component, so each component's mass and cost
# properties sit together (the power chain's sizing and its $/W in one tab, etc.), rather than
# split into separate "mass" and "cost" tabs. The thruster is missing because its mass, cost, and
# lifetime live in the engine library (Project tab), not the build model. Any field not listed here
# still appears (under "Other"), so a coefficient added to BusModel is editable without touching
# this map.
_GROUPS: dict[str, list[str]] = {
    "Power & arrays": [
        "housekeeping_W", "ops_load_W", "power_margin_pct",
        "array_specific_power_W_m2", "transmission_eff_pct",
        "hvlv_converter_eff_pct", "ppu_eff_pct", "pcdu_kg_per_kW", "cell_cost_per_W",
    ],
    "Array physics": [
        "array_rating_temp_C",
        "array_alpha_front", "array_epsilon_front",
        "array_alpha_rear", "array_epsilon_rear",
        "array_earth_albedo_W_m2", "array_earth_ir_W_m2",
        "array_cell_eff_slope_pct_per_C", "array_cell_eff_intercept_pct",
        "array_mass_scale",
    ],
    "Propellant tank": [
        "tank_base_mass_kg", "tank_length_diameter_ratio",
        "tank_pressure_bar", "tank_temperature_C",
        "tank_safety_factor", "tank_wall_strength_pa",
        "tank_wall_density_kg_m3", "tank_wall_factor",
    ],
    "Bus & structure": [
        "fixed_bus_ratio_pct", "structure_ratio_pct", "thermal_ratio_pct",
        "harness_ratio_pct", "bus_cost_per_kg_kusd",
    ],
    "Programmatics": [
        "propellant_cost_per_kg", "iat_pct", "pm_pct",
    ],
    "Flight rules": [
        "flyby_min_altitude_km",
    ],
}

# Explicit labels with units for each build-model field, which read better than the raw slug. A
# field with no entry falls back to the suffix-humanizer below.
_LABELS: dict[str, str] = {
    # power & arrays
    "housekeeping_W": "Housekeeping power (W)",
    "ops_load_W": "Ops-mode load (W)",
    "power_margin_pct": "Power margin (%)",
    "array_specific_power_W_m2": "Array areal power (W/m²)",
    # array physics
    "array_rating_temp_C": "Array rating temperature (°C)",
    "array_alpha_front": "Front absorptivity (α)",
    "array_epsilon_front": "Front emissivity (ε)",
    "array_alpha_rear": "Rear absorptivity (α)",
    "array_epsilon_rear": "Rear emissivity (ε)",
    "array_earth_albedo_W_m2": "Earth albedo (W/m²)",
    "array_earth_ir_W_m2": "Earth infrared (W/m²)",
    "array_cell_eff_slope_pct_per_C": "Cell efficiency slope (%/°C)",
    "array_cell_eff_intercept_pct": "Cell efficiency at 0 °C (%)",
    "array_mass_scale": "Array mass scale (×)",
    "transmission_eff_pct": "Array→bus wiring efficiency (%)",
    "hvlv_converter_eff_pct": "HV→LV converter efficiency (%)",
    "ppu_eff_pct": "PPU efficiency (%, thrusters)",
    "pcdu_kg_per_kW": "PCDU mass (kg/kW)",
    # propellant tank
    "tank_base_mass_kg": "Tank base mass (kg)",
    "tank_length_diameter_ratio": "Tank length : diameter",
    "tank_pressure_bar": "Tank pressure (bar)",
    "tank_temperature_C": "Tank temperature (°C)",
    "tank_safety_factor": "Burst safety factor",
    "tank_wall_strength_pa": "Wall strength (Pa)",
    "tank_wall_density_kg_m3": "Wall density (kg/m³)",
    "tank_wall_factor": "Wall-thickness margin",
    # bus & structure
    "fixed_bus_ratio_pct": "Fixed bus (% of dry)",
    "structure_ratio_pct": "Structure (% of dry)",
    "thermal_ratio_pct": "Thermal (% of dry)",
    "harness_ratio_pct": "Harness (% of dry)",
    "bus_cost_per_kg_kusd": "Bus cost ($k/kg)",
    # power & arrays (cost)
    "cell_cost_per_W": "Array cost ($/W)",
    # programmatics
    "propellant_cost_per_kg": "Propellant cost ($/kg)",
    "iat_pct": "Integration & test (%)",
    "pm_pct": "Program management (%)",
    # flight rules
    "flyby_min_altitude_km": "Minimum flyby altitude (km)",
}

# Suffix -> the unit shown in a field's label (longest match first). Fallback humanizer for any
# field not named explicitly in _LABELS (e.g. a coefficient added to BusModel later).
_UNITS = [
    ("_pct_day", "%/day"), ("_pct", "%"), ("_W_kg", "W/kg"), ("_per_kW", "kg/kW"),
    ("_per_kg_kusd", "$k/kg"), ("_per_kg", "$/kg"), ("_per_W", "$/W"),
    ("_musd", "$M"), ("_kg", "kg"), ("_W", "W"),
]

# Plain-language help for each build-model field (shown as a hover tooltip). A field with no entry
# falls back to no tooltip; keep these in step with prospector.spacecraft.buildability.BusModel.
_HELP: dict[str, str] = {
    # flight rules
    "flyby_min_altitude_km": "Closest a gravity-assist flyby may pass above the planet's surface. "
                             "A lower pass bends the path more; this is the margin kept for "
                             "navigation.",
    # power & arrays
    "housekeeping_W": "Always-on bus power: avionics, wheels, cameras, thermal.",
    "ops_load_W": "Payload and comms power, thrusters off. Arrays are sized on whichever mode "
                  "draws more.",
    "power_margin_pct": "Added to the worst-mode draw before the arrays are sized.",
    "array_specific_power_W_m2": "Array power per square metre when new. Also sets the array "
                                 "area used for drag and solar pressure.",
    "transmission_eff_pct": "Wiring (ohmic) loss in the harness from the array to the main bus, "
                            "paid by everything on board. Wiring only: the array's temperature "
                            "and sun-distance response are modelled under Array physics, so do "
                            "not fold them in here as well.",
    "hvlv_converter_eff_pct": "Step down from the main bus to the low-voltage bus, which feeds "
                              "the avionics and any PPU wired to it.",
    "ppu_eff_pct": "Drives the thrusters only, so the avionics do not pay it.",
    "pcdu_kg_per_kW": "Per kilowatt of array power (power control and distribution unit).",
    "cell_cost_per_W": "Per watt of array power when new.",
    # array physics (prospector.spacecraft.arrays.ArrayModel; the solvers fly these)
    "array_rating_temp_C": "The cell temperature the array's wattage is quoted at. Datasheets "
                           "quote 28 °C; facing the Sun at 1 AU the panel runs hotter (about "
                           "60 °C), where the cells convert several percent worse. A vehicle's "
                           "array power is what it delivers in flight, so the array bought, "
                           "weighed and given area is rated higher by that ratio. Set this to "
                           "the panel's operating temperature to treat wattages as flight "
                           "figures instead.",
    "array_alpha_front": "Fraction of sunlight the front face absorbs, which sets how hot it "
                         "runs.",
    "array_epsilon_front": "How well the front face sheds that heat.",
    "array_alpha_rear": "Fraction of Earth-reflected sunlight the rear face absorbs, near Earth "
                        "only.",
    "array_epsilon_rear": "How well the rear face sheds heat, and how much Earth infrared it "
                          "takes in.",
    "array_earth_albedo_W_m2": "Sunlight reflected off Earth onto the rear face, at the surface "
                               "(~30% of the solar constant). Fades with distance squared.",
    "array_earth_ir_W_m2": "Earth's own infrared glow warming the rear face at low altitude.",
    "array_cell_eff_slope_pct_per_C": "How fast cell efficiency changes with temperature, "
                                      "negative because hotter is worse.",
    "array_mass_scale": "Multiplier on the power to mass curve below; 1.0 is the vendor curve "
                        "as measured.",
    # propellant tank (a single spherocylindrical pressure vessel; see tank_dry_mass_kg)
    "tank_base_mass_kg": "Valves, mounts and feed lines, before any volume is added.",
    "tank_length_diameter_ratio": "1.0 is a sphere, the lightest tank for a given volume; "
                                  "higher is longer, slimmer and heavier, but packs better.",
    "tank_pressure_bar": "Higher pressure means thicker walls but a denser gas in a smaller "
                         "tank.",
    "tank_temperature_C": "Colder storage packs the gas denser, so the tank is smaller.",
    "tank_safety_factor": "Burst pressure ÷ operating pressure the walls are sized to (1.5 is "
                          "the usual uncrewed standard).",
    "tank_wall_strength_pa": "Wall material strength; the default is a composite-overwrapped "
                             "tank with a plastic liner.",
    "tank_wall_factor": "Extra thickness over the pressure requirement, for welds and fittings "
                        "(≈1.2 typical).",
    # bus & structure (ratios are a percentage of dry mass; no growth margin is applied)
    "fixed_bus_ratio_pct": "GNC, computer, comms, batteries, RCS and the secondary tank.",
    "bus_cost_per_kg_kusd": "A comparison placeholder, not a quote.",
    # programmatics
    "propellant_cost_per_kg": "Fallback price; the Propellants tab value wins when a gas is "
                              "set.",
    "iat_pct": "Added on top of hardware cost.",
    "pm_pct": "Added on top of hardware cost.",
}

# Help for the per-gas propellant fields (Propellants tab).
_PROP_HELP = {
    "density_kg_m3": "Used only when CoolProp cannot model the gas; otherwise density comes "
                     "from the tank pressure and temperature.",
    "thrust_scale": "Thrust relative to xenon, used only when the engine has no measured data "
                    "for this gas.",
    "isp_scale": "Isp relative to xenon, used only when the engine has no measured data for "
                 "this gas.",
}


def _label_for(field: str) -> str:
    """A readable label for a model field: the explicit ``_LABELS`` name, else a
    unit-suffixed humanization of the slug (so a newly added field still reads sensibly)."""
    if field in _LABELS:
        return _LABELS[field]
    for suffix, unit in _UNITS:
        if field.endswith(suffix):
            stem = field[: -len(suffix)].replace("_", " ").strip()
            return f"{stem or field} ({unit})"
    return field.replace("_", " ")


def adopt_profile(name: str) -> bool:
    """Make ``name`` the open project's build-model profile. Returns whether anything changed;
    the choice is saved with the project, so it marks the project dirty."""
    if S.project is None or name == S.build_model:
        return False
    S.build_model = name
    state.mark_dirty()
    return True


def open_settings(profile: str | None = None) -> None:
    """Open the Global-settings dialog: one build/cost profile (the open project's unless
    ``profile`` says otherwise), plus the propellant, radiation, launch-type and return-destination
    libraries, which are shared across projects."""
    profile = profile or S.build_model
    model = buildability.load_bus_model(name=profile)
    grouped = set().union(*_GROUPS.values())
    # The mass CURVE is structured (a list of segments), not a scalar coefficient: it is edited in
    # the profile's YAML and shown read-only below the Array-physics grid, so it must not
    # fall into the auto-rendered "Other" number grid.
    other = [f for f in buildability.BusModel.model_fields
             if f not in grouped and f != "array_mass_curve"]
    tabs_spec = {**_GROUPS, **({"Other": other} if other else {})}

    fields: dict[str, ui.number] = {}        # build-model field -> input
    gas_fields: dict[str, dict[str, ui.number]] = {}  # gas key -> {prop field -> input}
    rad_fields: dict[str, dict] = {}                  # scenario key -> {model field -> input}
    launches = _Catalog(
        entries=load_launch_orbits(), fields={}, noun="launch type", blurb="",
        blank=lambda key: LaunchOrbit(name=key, perigee_alt_km=400.0, apogee_alt_km=400.0,
                                      inclination_deg=28.5),
        writer=save_launch_orbit, remover=delete_launch_orbit,
        text_fields=("name",), bool_fields=("escape_provided",), note=_launch_note)
    returns = _Catalog(
        entries=load_return_destinations(), fields={}, noun="return destination",
        blurb="Where a return leg ends, and what capture there costs.",
        blank=lambda key: ReturnDestination(name=key, arrival_vinf_kms=0.4,
                                            insertion_dv_kms=0.8),
        writer=save_return_destination, remover=delete_return_destination,
        text_fields=("name", "description"))

    with ui.dialog() as dlg, ui.card().style(
            f"background:{PANEL};border:1px solid {BORDER};min-width:560px;max-width:680px"):
        with ui.row().classes("items-center w-full no-wrap"):
            ui.icon("tune").style(f"color:{ACCENT}")
            ui.label("Global settings").style(f"color:{TEXT};font-weight:600")

        def _switch(name: str) -> None:
            adopt_profile(name)
            dlg.close()
            open_settings(name)

        def _copy_as() -> None:
            key = state.slug(copy_name.value or "")
            if not key:
                err.text = "Name the new profile first."
                err.set_visibility(True)
                return
            if key in buildability.list_bus_models():
                err.text = f"A profile named '{key}' already exists."
                err.set_visibility(True)
                return
            buildability.save_bus_model(model, name=key)
            _switch(key)

        # The sizing and cost coefficients are one profile of several; the picker names the one this
        # dialog edits, and an open project is built with the one it names.
        with ui.row().classes("items-center w-full no-wrap gap-2"):
            ui.select(buildability.list_bus_models(), value=profile, label="Build-model profile",
                      on_change=lambda e: _switch(e.value)).props("dense outlined") \
                .classes("col").tooltip(
                    "Which set of sizing and cost coefficients the open project is built with. "
                    "Saved with the project. Other tabs are shared by every project.")
            copy_name = ui.input(placeholder="new profile name").props("dense outlined") \
                .classes("col")
            ui.button("Copy as", icon="content_copy", on_click=_copy_as).props("flat no-caps") \
                .style(f"color:{MUTED}").tooltip("Save this profile's numbers under a new name and switch to it.")

        with ui.tabs().props("dense").classes("w-full") as tabs:
            for name in tabs_spec:
                ui.tab(name)
            ui.tab("Propellants")
            ui.tab("Radiation")
            ui.tab("Launch types")
            ui.tab("Returns")
        with ui.tab_panels(tabs, value=next(iter(tabs_spec))).style(
                f"background:{PANEL2};border-radius:6px").classes("w-full"):
            for name, field_names in tabs_spec.items():
                with ui.tab_panel(name):
                    _model_grid(model, field_names, fields)
                    if name == "Array physics":
                        _mass_curve_summary(model)
                        _mass_curve_editor(model)
            with ui.tab_panel("Propellants"):
                _propellant_grid(gas_fields)
            with ui.tab_panel("Radiation"):
                _radiation_panel(rad_fields)
            with ui.tab_panel("Launch types"):
                _catalog_panel(launches)
            with ui.tab_panel("Returns"):
                _catalog_panel(returns)

        err = ui.label("").style(f"color:{ACCENT};font-size:.75rem")
        err.set_visibility(False)

        def _save() -> None:
            try:
                merged = {**model.model_dump(),
                          **{f: w.value for f, w in fields.items() if w.value is not None}}
                curve, why = segments_from_sextets(_mass_sextets())
                if curve is not None:
                    merged["array_mass_curve"] = curve
                elif why:
                    raise ValueError(why)
                new_model = buildability.BusModel.model_validate(merged)
                buildability.save_bus_model(new_model, name=profile)
                _save_propellants(gas_fields)
                _save_radiation(rad_fields)
                _commit_catalog(launches, LaunchOrbit)
                _commit_catalog(returns, ReturnDestination)
            except Exception as exc:  # noqa: BLE001  (show the validation message inline)
                err.text = str(exc).splitlines()[0]
                err.set_visibility(True)
                return
            ui.notify("Saved global settings", type="positive")
            dlg.close()

        with ui.row().classes("justify-end w-full gap-2 mt-2 no-wrap"):
            ui.button("Cancel", on_click=dlg.close).props("flat no-caps").style(f"color:{MUTED}")
            ui.button("Save", icon="save", on_click=_save).props("unelevated no-caps")
    dlg.open()


def _model_grid(model, field_names: list[str], fields: dict[str, ui.number]) -> None:
    """Two-up number inputs for a group of build-model fields."""
    for i in range(0, len(field_names), 2):
        with ui.row().classes("gap-3 w-full no-wrap"):
            for field in field_names[i:i + 2]:
                box = ui.number(_label_for(field), value=getattr(model, field)).props(
                    "dense outlined").classes("flex-grow")
                if field in _HELP:
                    box.tooltip(_HELP[field])
                fields[field] = box


def _mass_curve_summary(model) -> None:
    """What the stored curve implies, as a sanity read on the segments edited below: the specific
    power it lands on at a few reference array sizes. Reflects the SAVED curve, so it is a
    before-picture while the rows are being edited, not a live preview of them."""
    am = model.array_model()
    implied = " · ".join(f"{p / 1000:g} kW → {am.mass_kg(p):.1f} kg "
                         f"({am.specific_power_W_kg(p):.0f} W/kg)"
                         for p in (500.0, 2000.0, 5000.0, 20000.0))
    ui.label("Stored curve implies").style(
        f"color:{MUTED};font-size:.7rem;margin-top:6px")
    ui.label(implied).style(f"color:{TEXT};font-size:.72rem")


def _propellant_grid(gas_fields: dict[str, dict[str, ui.number]]) -> None:
    """A per-gas editor: cost, fallback density, and the thrust/Isp estimation scales.

    The scales are how an engine authored on xenon is estimated on this gas (xenon = 1.0). The
    density edited here is the fallback used when CoolProp can't model the gas; when it can (xenon,
    krypton), the live density comes from the tank pressure/temperature instead.
    """
    catalog = load_propellants()
    for key in list_propellants():
        gas = catalog[key]
        widgets: dict[str, ui.number] = {}
        with ui.row().classes("items-end gap-2 w-full no-wrap"):
            widgets["name"] = ui.input("Gas", value=gas.name).props(
                "dense outlined").style("width:110px;flex:0 0 auto")
            widgets["cost_per_kg"] = ui.number("Cost ($/kg)", value=gas.cost_per_kg).props(
                "dense outlined").classes("flex-grow")
            widgets["density_kg_m3"] = ui.number("Density (kg/m³)", value=gas.density_kg_m3,
                                                 step=10).props("dense outlined").classes("flex-grow")
            widgets["thrust_scale"] = ui.number("Thrust ×Xe", value=gas.thrust_scale,
                                                 step=0.01).props("dense outlined").classes("flex-grow")
            widgets["isp_scale"] = ui.number("Isp ×Xe", value=gas.isp_scale,
                                             step=0.01).props("dense outlined").classes("flex-grow")
            # The real-fluid name, and the reason the density box above is a FALLBACK: with a
            # CoolProp name the stored density is only used when CoolProp cannot be reached. Blank
            # means this gas has no real-fluid model and the authored density always holds.
            widgets["coolprop_name"] = ui.input(
                "CoolProp", value=gas.coolprop_name or "").props(
                "dense outlined").style("width:110px;flex:0 0 auto")
            for fname, box in widgets.items():
                if fname in _PROP_HELP:
                    box.tooltip(_PROP_HELP[fname])
        gas_fields[key] = widgets


def _save_propellants(gas_fields: dict[str, dict[str, ui.number]]) -> None:
    """Persist each edited gas back to its library YAML."""
    catalog = load_propellants()
    for key, widgets in gas_fields.items():
        base = catalog[key]
        edits = {}
        for name, widget in widgets.items():
            raw = widget.value
            if raw is None:
                continue
            if name == "coolprop_name":
                # Cleared means "no real-fluid model for this gas", which is a real setting rather
                # than a blank to ignore.
                edits[name] = raw.strip() or None
            elif not (isinstance(raw, str) and not raw.strip()):
                edits[name] = raw
        save_propellant(base.model_copy(update=edits), key)


# The belt-physics fields, grouped the way the model reads: where the belts sit, how hard they hit,
# and what the cell behind the coverglass does about it. `name` and `cell` are the scenario's own
# labels and are edited as text; everything else is a coefficient.
_RADIATION_GROUPS: dict[str, list[str]] = {
    "Belt geometry": ["re_km", "proton_plateau_l", "proton_shoulder_l",
                      "electron_peak_l", "electron_sigma_l", "bb0_scale",
                      "belt_floor_lo_km", "belt_floor_hi_km"],
    "Dose": ["proton_ddd_core", "electron_ddd_core", "cell_ddd_ref", "cell_ddd_coef"],
    "Coverglass": ["coverglass_um", "coverglass_density_g_cm3",
                   "coverglass_ref_um", "coverglass_ref_density_g_cm3",
                   "proton_shield_exponent", "electron_shield_exponent"],
}


def _radiation_panel(rad_fields: dict[str, dict]) -> None:
    """Every radiation scenario in the library, each fully editable.

    The belt model decides how much array a design must carry to survive its climb, so it is a
    tuning knob like any other, but it was reachable only by editing YAML, and sixteen of its
    twenty coefficients appeared nowhere in the app at all. A coefficient nobody can see is a
    coefficient nobody checks.

    One expansion per scenario rather than a picker: there are only a handful, and seeing that the
    worst case and the nominal differ by two numbers is worth more than hiding all but one behind a
    dropdown.
    """
    catalog = load_radiation_models()
    if not catalog:
        ui.label("No radiation scenarios in this config library.").style(
            f"color:{MUTED};font-size:.75rem")
        return
    ui.label("Belt physics per scenario: sets how much array a spiral has to carry.").style(
        f"color:{MUTED};font-size:.7rem;margin-bottom:.2rem")

    for key in list_radiation_models():
        model = catalog[key]
        widgets: dict[str, object] = {}
        rad_fields[key] = widgets
        with ui.expansion(key, caption=model.name).classes("w-full").props("dense"):
            with ui.row().classes("gap-3 w-full no-wrap"):
                widgets["name"] = ui.input("Scenario name", value=model.name).props(
                    "dense outlined").classes("flex-grow")
                widgets["cell"] = ui.input("Cell", value=model.cell).props(
                    "dense outlined").classes("flex-grow")
            for group, names in _RADIATION_GROUPS.items():
                ui.label(group).style(f"color:{MUTED};font-size:.68rem;margin-top:.35rem")
                _radiation_grid(model, names, widgets)


def _radiation_grid(model, field_names: list[str], widgets: dict) -> None:
    """Two-up inputs for a group of scenario coefficients.

    ``proton_plateau_l`` is the one pair rather than a single number, being the band the proton
    plateau spans, so it gets a text box holding both ends, the way a range is written in the file.
    Everything else is a number.
    """
    for i in range(0, len(field_names), 2):
        with ui.row().classes("gap-3 w-full no-wrap"):
            for field in field_names[i:i + 2]:
                value = getattr(model, field)
                if isinstance(value, (tuple, list)):
                    widgets[field] = ui.input(
                        _label_for(field), value=", ".join(f"{v:g}" for v in value)).props(
                        "dense outlined").classes("flex-grow")
                    widgets[field].tooltip("The two L-shell bounds of the band, comma-separated.")
                else:
                    widgets[field] = ui.number(_label_for(field), value=value).props(
                        "dense outlined").classes("flex-grow")


def _save_radiation(rad_fields: dict[str, dict]) -> None:
    """Persist every edited scenario through the library writer.

    A blank input is left at the stored value rather than written as zero: clearing a box is how
    someone asks a question, not how they assert that a belt has no protons in it.
    """
    catalog = load_radiation_models()
    for key, widgets in rad_fields.items():
        stored = catalog.get(key)
        if stored is None:
            continue
        merged = stored.model_dump()
        for name, widget in widgets.items():
            raw = widget.value
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                continue
            if isinstance(getattr(stored, name), (tuple, list)):
                parts = [p for p in str(raw).replace(",", " ").split() if p]
                merged[name] = [float(p) for p in parts]
            else:
                merged[name] = raw
        save_radiation_model(RadiationModel.model_validate(merged), key)


# ---------------------------------------------------------------------------------------
# Launch types, return destinations, and the array mass curve
#
# All three were reachable only by editing YAML. A knob the app will not let you turn is one that
# gets turned somewhere the app cannot see.
# ---------------------------------------------------------------------------------------

def _launch_note(orbit: LaunchOrbit) -> str:
    """Why a launch type's orbit boxes are empty, when they are: the vehicle escapes by itself, so
    there is no closed orbit to describe and the three terms are not stored."""
    if not orbit.escape_provided:
        return ""
    return "No drop-off orbit while this is on. Turn the switch off to spiral out of one."


@dataclass
class _Catalog:
    """One editable keyed model catalog: what is in it, what the panel is doing to it, and how it
    is written back.

    ``entries`` is the working copy the panel edits, so a New or a Delete shows up straight away
    and only reaches the config library when the dialog is saved. ``removed`` collects the keys
    whose files that save has to unlink.
    """

    entries: dict                       # key -> model instance, the working copy
    fields: dict[str, dict]             # key -> {model field -> widget}
    blurb: str
    blank: Callable[[str], object]      # key -> a new entry with sensible starting numbers
    writer: Callable[[object, str], object]
    remover: Callable[[str], None]
    noun: str                           # what one entry is called, for the buttons and messages
    text_fields: tuple[str, ...] = ()
    bool_fields: tuple[str, ...] = ()
    note: Callable[[object], str] | None = None   # a line above an entry's fields, when it needs one
    removed: set[str] = field(default_factory=set)


# A catalog key names the file it is written to, so it has to survive being a filename.
_KEY_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@ui.refreshable
def _catalog_panel(cat: _Catalog) -> None:
    """Every entry of a keyed model catalog, each fully editable, one expansion per entry, over a
    New and a per-entry Delete.

    Field kind follows the model: a bool renders as a switch, the named text fields as inputs,
    everything else as a number. Listing the exceptions rather than introspecting keeps the panel
    honest about what it can render, since a field type it does not know shows up as a number box
    that will not validate, rather than silently vanishing.
    """
    cat.fields.clear()
    if cat.blurb:
        ui.label(cat.blurb).style(f"color:{MUTED};font-size:.7rem;margin-bottom:.2rem")
    if not cat.entries:
        ui.label(f"No {cat.noun}s left. Cancel to keep the ones on disk.").style(
            f"color:{ACCENT};font-size:.75rem")

    for key in sorted(cat.entries):
        entry = cat.entries[key]
        widgets: dict[str, object] = {}
        cat.fields[key] = widgets
        names = list(type(entry).model_fields)
        with ui.expansion(key, caption=getattr(entry, "name", "")).classes("w-full").props("dense"):
            note = cat.note(entry) if cat.note else None
            if note:
                ui.label(note).style(f"color:{MUTED};font-size:.68rem")
            for i in range(0, len(names), 2):
                with ui.row().classes("gap-3 w-full no-wrap items-center"):
                    for name in names[i:i + 2]:
                        value = getattr(entry, name)
                        label = _label_for(name)
                        if name in cat.bool_fields:
                            widgets[name] = ui.switch(label, value=bool(value)).props(
                                "dense").classes("flex-grow")
                        elif name in cat.text_fields:
                            widgets[name] = ui.input(label, value=str(value or "")).props(
                                "dense outlined").classes("flex-grow")
                        else:
                            widgets[name] = ui.number(label, value=value).props(
                                "dense outlined").classes("flex-grow")
            ui.button(f"Delete {key}", icon="delete",
                      on_click=lambda _, k=key: _delete(cat, k)).props(
                "flat dense no-caps").style(f"color:{MUTED};font-size:.7rem;margin-top:.3rem")

    with ui.row().classes("items-center gap-2 w-full no-wrap").style("margin-top:.4rem"):
        box = ui.input(placeholder=f"new {cat.noun} key").props("dense outlined").classes("w-40")
        ui.button(f"Add {cat.noun}", icon="add",
                  on_click=lambda: _create(cat, box)).props("flat dense no-caps").style(
            f"color:{ACCENT}")


def _create(cat: _Catalog, box: ui.input) -> None:
    """The Add button: take the typed key, complain in place if it will not do, redraw if it will."""
    complaint = _add_entry(cat, str(box.value or "").strip())
    if complaint:
        ui.notify(complaint, type="warning")
        return
    box.value = ""
    _catalog_panel.refresh()


def _delete(cat: _Catalog, key: str) -> None:
    """The Delete button: drop the entry and redraw without it."""
    _remove_entry(cat, key)
    _catalog_panel.refresh()


def _add_entry(cat: _Catalog, key: str) -> str:
    """Put a new entry into the working copy under ``key``, starting from the blank the catalog
    supplies so its numbers are editable rather than empty. Returns why it could not, if it
    could not: the key names the file the entry is written to, so it has to survive being one.
    """
    if not _KEY_OK.match(key):
        return ("A key is letters, digits, dashes and underscores, starting with a letter or "
                "digit - it names the file.")
    if key in cat.entries:
        return f"{key} is already in the library."
    cat.entries[key] = cat.blank(key)
    cat.removed.discard(key)
    return ""


def _remove_entry(cat: _Catalog, key: str) -> None:
    """Drop an entry from the working copy and mark its file for deletion on save."""
    cat.entries.pop(key, None)
    cat.fields.pop(key, None)
    cat.removed.add(key)


def _save_catalog(fields: dict[str, dict], catalog: dict, model_cls, writer) -> None:
    """Persist every edited catalog entry. A blank number keeps its stored value."""
    for key, widgets in fields.items():
        stored = catalog.get(key)
        if stored is None:
            continue
        merged = stored.model_dump()
        for name, widget in widgets.items():
            raw = widget.value
            if raw is None or (isinstance(raw, str) and not raw.strip()
                               and isinstance(getattr(stored, name), (int, float))):
                continue
            merged[name] = raw
        writer(model_cls.model_validate(merged), key)


def _commit_catalog(cat: _Catalog, model_cls) -> None:
    """Write the working copy back to the config library: every entry saved, every deleted key's
    file unlinked. Writes come first, so a validation error leaves the library untouched."""
    _save_catalog(cat.fields, cat.entries, model_cls, cat.writer)
    for key in cat.removed:
        cat.remover(key)


# The array mass-curve segments the settings dialog is editing, and the label describing them. Held
# at module level for the same reason the engine throttle curve's rows are: a refreshable defined
# inside the dialog closure does not redraw when refreshed from a button in that dialog.
_MASS_COLS = ("lo_W", "hi_W", "w0", "kg0", "w1", "kg1")
_mass_rows: list[dict] = []
_mass_status = None


def _mass_sextets() -> list[tuple[float, ...]]:
    """The rows as ``(lo_W, hi_W, w0, kg0, w1, kg1)``, dropping any row still entirely empty.

    A partly-filled row is kept, with its blanks read as zero, so the validator names it rather
    than the curve quietly losing a band."""
    out = []
    for r in _mass_rows:
        vals = tuple(r[c] for c in _MASS_COLS)
        if all(v is None for v in vals):
            continue
        out.append(tuple(0.0 if v is None else float(v) for v in vals))
    return out


def segments_from_sextets(rows: list[tuple[float, ...]]) -> tuple[list[dict] | None, str]:
    """Validate edited segments into what :class:`BusModel` stores, plus a line describing them.

    Returns ``(None, "")`` for no rows at all (the stored curve stands) and ``(None, message)`` for
    a row that cannot be a segment. The anchor check is the load-bearing one: the mass is
    interpolated as ``kg0 + (kg1 - kg0)/(w1 - w0) * (P - w0)``, so two anchors at the same power
    divide by zero: a curve that validates as a model and then fails when something asks it for a
    mass.
    """
    if not rows:
        return None, ""
    for lo, hi, w0, kg0, w1, kg1 in rows:
        band = f"(row {lo:g}-{hi:g} W)"
        if lo < 0 or hi <= 0:
            return None, f"a band's powers must be positive {band}"
        if hi <= lo:
            return None, f"a band's upper power must exceed its lower {band}"
        if w0 == w1:
            return None, f"the two anchors must sit at different powers {band}"
        if kg0 <= 0 or kg1 <= 0:
            return None, f"an anchor mass must be > 0 {band}"
    out = sorted(({c: v for c, v in zip(_MASS_COLS, r, strict=True)} for r in rows),
                 key=lambda s: s["lo_W"])
    bands = " · ".join(f"{s['lo_W'] / 1000:g}-{s['hi_W'] / 1000:g} kW" for s in out)
    return out, f"{len(out)} segment{'s' if len(out) > 1 else ''} · {bands}"


def _refresh_mass_status() -> None:
    """Update the summary under the rows. Kept out of the refreshable, because refreshing on
    every keystroke would rebuild the number fields under the cursor."""
    if _mass_status is None:
        return
    segs, msg = segments_from_sextets(_mass_sextets())
    bad = segs is None and bool(msg)
    _mass_status.text = ("⚠ " + msg) if bad else (msg or "no rows, keeps the stored curve")
    _mass_status.style(f"color:{ACCENT if bad else MUTED};font-size:.7rem")


def _redraw_mass_curve() -> None:
    """Rebuild the rows and their summary. Kept apart from the mutations below so the row state
    can be exercised without a live client."""
    _mass_curve_rows.refresh()
    _refresh_mass_status()


def _set_mass_cell(i: int, key: str, value) -> None:
    _mass_rows[i][key] = value


def _add_mass_segment() -> None:
    _mass_rows.append(dict.fromkeys(_MASS_COLS))


def _remove_mass_segment(i: int) -> None:
    _mass_rows.pop(i)


def _set_mass_rows(segments) -> None:
    """Load a stored curve into the rows. Accepts the model's segment objects or plain dicts."""
    _mass_rows.clear()
    for s in segments or []:
        get = s.get if isinstance(s, dict) else lambda c, s=s: getattr(s, c)
        _mass_rows.append({c: get(c) for c in _MASS_COLS})


def _apply_mass_paste(paste) -> None:
    """Fill the rows from a pasted table. The rows are the editor; this is the fast way in for a
    vendor curve carrying several array families."""
    try:
        segments = parse_mass_curve(paste.value)
    except ValueError as exc:
        ui.notify(str(exc), type="warning")
        return
    if segments is None:
        ui.notify("nothing to paste", type="warning")
        return
    _set_mass_rows(segments)
    _redraw_mass_curve()
    paste.value = ""
    ui.notify(f"Loaded {len(segments)} segments", type="positive")


@ui.refreshable
def _mass_curve_rows() -> None:
    """One row per power band, plus an Add button. Six numbers is wide for this dialog, so the
    headings carry the field names the YAML uses rather than a label on every field."""
    if _mass_rows:
        with ui.row().classes("w-full no-wrap gap-1"):
            for head in _MASS_COLS:
                ui.label(head).style(
                    f"color:{MUTED};font-size:.6rem;font-family:monospace;flex:1 1 0;min-width:0")
            ui.element("div").style("width:26px;flex:0 0 auto")      # over the remove column
    for idx, row in enumerate(_mass_rows):
        with ui.row().classes("items-center gap-1 w-full no-wrap"):
            # No min= on these: a bound clamps on blur, editing the number just typed. The
            # validator reports a bad value instead of moving it.
            for col in _MASS_COLS:
                ui.number(value=row[col],
                          on_change=lambda e, i=idx, c=col: (_set_mass_cell(i, c, e.value),
                                                             _refresh_mass_status())
                          ).props("dense outlined").style("flex:1 1 0;min-width:0").classes(
                    "text-xs")
            ui.button(icon="close",
                      on_click=lambda i=idx: (_remove_mass_segment(i), _redraw_mass_curve())).props(
                "flat dense round").style(f"color:{MUTED};flex:0 0 auto")
    ui.button("Add segment", icon="add",
              on_click=lambda: (_add_mass_segment(), _redraw_mass_curve())).props(
        "outline dense no-caps").style(f"color:{ACCENT}")


def _mass_curve_editor(model) -> None:
    """The array power->mass curve as a row per power band.

    A segment maps a power band onto the two (watts, kilograms) anchors the array mass is
    interpolated between, so the curve is a piecewise line through vendor sizes rather than one
    specific power. Anchors need not sit on the band edges, which is what lets adjacent bands step
    to a different array family.
    """
    ui.label("Array mass curve - a power band and the two (W, kg) anchors its mass "
             "interpolates between").style(f"color:{MUTED};font-size:.7rem;margin-top:.4rem")
    _set_mass_rows(model.array_mass_curve)
    _mass_curve_rows()
    global _mass_status
    _mass_status = ui.label("").style(f"color:{MUTED};font-size:.7rem")
    _refresh_mass_status()

    with ui.expansion("Paste a table", icon="content_paste").classes("w-full").props("dense"):
        paste = ui.textarea(
            placeholder="50, 500, 50, 1.2, 500, 5\n500, 5000, 500, 5, 5000, 50").props(
            'dense outlined autogrow input-style="font-family:monospace;font-size:.75rem"'
        ).classes("w-full")
        paste.tooltip("Comma, tab or space separated; a header line is fine.")
        ui.button("Replace rows with this", icon="playlist_add",
                  on_click=lambda: _apply_mass_paste(paste)).props(
            "outline dense no-caps").style(f"color:{ACCENT}")


def parse_mass_curve(text: str) -> list[dict] | None:
    """Pasted rows into array-mass segments, or None to keep whatever is stored.

    Raises :class:`ValueError` with a readable message on a malformed row rather than silently
    dropping it: a curve missing a band would size every array in that band off the wrong segment,
    and nothing downstream would look wrong.
    """
    segments = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p for p in re.split(r"[,\t ]+", line) if p]
        try:
            numbers = [float(p) for p in parts]
        except ValueError:
            continue                      # a pasted header row, not a segment
        if len(numbers) != 6:
            raise ValueError(f"each segment needs 6 numbers (lo_W, hi_W, w0, kg0, w1, kg1); "
                             f"got {len(numbers)} in {line!r}")
        lo, hi, w0, kg0, w1, kg1 = numbers
        if hi <= lo:
            raise ValueError(f"segment {line!r}: the band's upper power must exceed its lower")
        segments.append({"lo_W": lo, "hi_W": hi, "w0": w0, "kg0": kg0, "w1": w1, "kg1": kg1})
    if not segments:
        return None
    segments.sort(key=lambda s: s["lo_W"])
    return segments
