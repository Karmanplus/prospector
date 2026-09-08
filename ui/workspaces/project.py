"""Project (config): the study umbrella and its two building-block editors.

A project is a :class:`~prospector.config.Study`: one mission, one vehicle, and both screens. This
workspace edits the parts the study saves: the mission, meaning the launch window, the launch type
and whether there is a return trip, and the vehicle, meaning its masses and its engines. The engine
library itself is edited in a separate dialog. The screening terms are tuned live over in Find
targets, so they do not appear here.

Everything updates as you type. Editing any field rebuilds the working model in :mod:`ui.state`,
marks the project as having unsaved changes so the top-bar Save lights up, and refreshes the
figures and the delta-v budget straight away. The budget always follows from the masses, the engine
and the launch type; it is never typed in. The launch type only decides where the spacecraft gets
dropped off, and what escaping from there costs is an estimate at this point, refined later in Plan
trajectory. Saving is always deliberate: the top-bar Save overwrites the open study, and Save as
here writes a new one, storing the mission and vehicle alongside it so the whole thing loads again
cleanly.
"""
from __future__ import annotations

import re
from datetime import date

from nicegui import ui
from pydantic import ValidationError

from prospector.config import EngineMount, Mission, Vehicle
from prospector.launch import escape_dv_estimate, load_launch_orbits, load_return_destinations
from prospector.spacecraft import buildability
from prospector.spacecraft.propellants import list_propellants, load_propellants
from prospector.spacecraft.propulsion import (
    Engine,
    PowerPoint,
    list_engines,
    load_engines,
    save_engine,
)
from ui import state
from ui.components import section, vehicle_perf_rows
from ui.state import S
from ui.theme import ACCENT, AMBER, BORDER, GREEN, MUTED, PANEL, PANEL2, TEXT

# Widget handles for the current render, so a field's on_change can rebuild the whole model from
# every field's live value (repopulated each render; read via .get for optional fields).
_w: dict = {}
# Tracked outside the widget set: the launch type (chosen by clicking the launch table, not a
# value-bearing input) and the engine type (a select whose row is rebuilt on library edits).
_launch_orbit: str = "TLI"
_engine_type: str = ""
# The working gas LOADED on the vehicle (Vehicle.propellant). None = run each engine on its
# authored (native) gas; a different gas estimates thrust/Isp by scaling the native numbers.
_propellant: str | None = None
# The solar-array sizing reserve (Vehicle.array_margin_pct). None = the build model's default. The
# array's power when new follows from it, as the loads through the power chain times (1+margin),
# rather than being typed in.
_array_margin: float | None = None
# Engine assembly blocks beyond the first (head) one: preserved verbatim across edits so a
# multi-type YAML assembly isn't silently flattened to one block by the single-engine editor.
_extra_mounts: list = []
# Inline validation lines, shown only when a rebuild fails (toggled, not rebuilt).
_mission_err = None
_vehicle_err = None


# ======================================================================================
# layout
# ======================================================================================

def render() -> None:
    """Build the Project workspace: study header, Mission/Vehicle tabs, derived stats."""
    with ui.element("div").style("width:100%;height:100%;overflow:auto"):
        with ui.column().classes("w-full p-6 gap-4").style("max-width:880px;margin:0 auto"):
            _header()
            with ui.tabs().props(
                    "dense no-caps align=left active-color=primary indicator-color=primary").style(
                    f"color:{MUTED}") as tabs:
                ui.tab("Mission", icon="public")
                ui.tab("Vehicle", icon="satellite_alt")
            with ui.tab_panels(tabs, value="Mission").classes("w-full").style(
                    f"background:{PANEL};border:1px solid {BORDER};border-radius:8px"):
                with ui.tab_panel("Mission"):
                    _mission_panel()
                with ui.tab_panel("Vehicle"):
                    _vehicle_panel()
            _stats()


def _header() -> None:
    """The study name (editable) + a Save-as button. The top-bar Save overwrites in place."""
    with ui.row().classes("items-center w-full no-wrap gap-2"):
        ui.icon("tune").style(f"color:{ACCENT}")
        ui.input(value=S.study_name, on_change=lambda e: _set_study_name(e.value)).props(
            "dense outlined").style("font-size:1.05rem;min-width:280px").classes("flex-grow")
        ui.button("Save as…", icon="save_as", on_click=_save_as_dialog).props(
            "outline dense no-caps").style(f"color:{TEXT}")


# ======================================================================================
# Mission
# ======================================================================================

def _mission_panel() -> None:
    global _launch_orbit, _mission_err
    m = S.mission
    _launch_orbit = m.launch_orbit
    with ui.column().classes("w-full gap-3"):
        _w["mis_name"] = ui.input("Mission name", value=m.name,
                                  on_change=_apply_mission).props("dense outlined").classes("w-full")
        with ui.row().classes("gap-4 w-full no-wrap"):
            _w["mis_open"] = _date_field("Launch opens", m.launch_window[0].isoformat(),
                                          lambda: _apply_mission(edited="mis_open"))
            _w["mis_close"] = _date_field("Launch closes", m.launch_window[1].isoformat(),
                                          lambda: _apply_mission(edited="mis_close"))
        _w["mis_arrive"] = _date_field("Arrive by", m.arrive_by.isoformat(),
                                      lambda: _apply_mission(edited="mis_arrive"))

        section("How it leaves Earth", "north_east")
        _launch_table()

        ui.separator().style(f"background:{BORDER}")
        section("Default target", "my_location")
        _target_section()

        ui.separator().style(f"background:{BORDER}")
        _w["mis_return"] = ui.switch("Return trip", value=m.return_trip,
                                     on_change=_on_return_toggle).props("dense")
        _return_fields()

        _mission_err = ui.label("").style(f"color:{ACCENT};font-size:.75rem")
        _mission_err.set_visibility(False)


@ui.refreshable
def _target_section() -> None:
    """The project's default focus target: a searchable designation/name field. Setting it
    adopts the target as the active focus too (the top bar then flags any later override)."""
    current = S.project_target
    if current:
        bits = [str(current.get("full_name") or current.get("pdes") or "target")]
        for key, fmt in (("i", "i = {:.1f}°"), ("a", "a = {:.2f} AU")):
            v = current.get(key)
            if isinstance(v, (int, float)) and v == v:
                bits.append(fmt.format(float(v)))
        ui.label("Current: " + "  ·  ".join(bits)).style(f"color:{MUTED};font-size:.74rem")
    else:
        ui.label("No default target set.").style(
            f"color:{MUTED};font-size:.74rem")
    with ui.row().classes("gap-2 w-full no-wrap items-center"):
        box = ui.input("Designation or name", value=state.target_designation(current) or "").props(
            "dense outlined").classes("flex-grow")
        box.on("keydown.enter", lambda: _set_target(box.value))
        ui.button("Set", on_click=lambda: _set_target(box.value)).props(
            "outline dense no-caps").style(f"color:{TEXT}")


def _set_target(query: str) -> None:
    target = state.resolve_target(query)
    if target is None:
        ui.notify(f"No target matched '{query}'", type="negative")
        return
    state.set_project_target(target)
    ui.notify(f"Default target → {target.get('full_name') or target.get('pdes')}")
    _target_section.refresh()
    from ui import topbar
    topbar.header.refresh()


@ui.refreshable
def _return_fields() -> None:
    """The return-leg fields, shown only when the return-trip toggle is on."""
    m = S.mission
    if not (_w.get("mis_return") and _w["mis_return"].value):
        return
    dests = load_return_destinations()
    dest_options = {k: v.name for k, v in dests.items()}
    default_key = m.return_destination if m.return_destination in dests else (
        next(iter(dests), "EML2"))
    with ui.column().classes("w-full gap-3"):
        _w["mis_dest"] = ui.select(dest_options, value=default_key, label="Return destination",
                                   on_change=_apply_mission).props(
            "dense outlined options-dense").classes("w-full")
        _w["mis_rby"] = _date_field("Return by", (m.return_by or m.arrive_by).isoformat(),
                                    lambda: _apply_mission(edited="mis_rby"))
        with ui.row().classes("gap-4 w-full no-wrap"):
            _w["mis_rins"] = ui.number("Insertion ΔV override (km/s, 0 = default)",
                                       value=float(m.return_orbit_insert_dv or 0.0), min=0.0, step=0.05,
                                       on_change=_apply_mission).props("dense outlined").classes("flex-grow")
            _w["mis_stay"] = ui.number("Stay (days)", value=float(m.time_at_asteroid or 0.0),
                                       min=0.0, step=5.0,
                                       on_change=_apply_mission).props("dense outlined").classes("flex-grow")
        _w["mis_pay"] = ui.number("Payload collected (kg)", value=float(m.asteroid_payload_mass),
                                  min=0.0, step=10.0, on_change=_apply_mission).props(
            "dense outlined").classes("w-full")


def _on_return_toggle() -> None:
    _return_fields.refresh()
    _apply_mission()


def _launch_table() -> None:
    """Pick the launch type: an altitude-sorted table of the launch library, each row showing
    what leaving Earth costs the SEP system from there (LV-provided escapes sort last)."""
    orbits = load_launch_orbits()
    rows = []
    for key, o in orbits.items():
        if o.escape_provided:
            starts, cost, order = "leaves Earth already", "free (0.00)", float("inf")
        else:
            starts = (f"{o.perigee_alt_km:.0f} km circular" if o.apogee_alt_km == o.perigee_alt_km
                      else f"{o.perigee_alt_km:.0f} × {o.apogee_alt_km:.0f} km")
            cost, order = f"spiral ≈ {escape_dv_estimate(o):.2f} km/s", o.sma_km
        rows.append({"key": key, "type": f"{key} - {o.name}", "starts": starts,
                     "escape": cost, "_order": order})
    rows.sort(key=lambda r: r["_order"])
    columns = [{"name": "type", "label": "Launch option", "field": "type", "align": "left"},
               {"name": "starts", "label": "Starts at", "field": "starts", "align": "left"},
               {"name": "escape", "label": "Cost to leave Earth (estimate)", "field": "escape",
                "align": "left"}]
    table = ui.table(columns=columns, rows=rows, row_key="key").props(
        "dense flat selection=single").classes("w-full pf-rowsel").style("background:transparent")
    table.selected = [r for r in rows if r["key"] == _launch_orbit]

    def _click(e):
        row = e.args[1]
        table.selected = [row]
        _set_launch_orbit(row["key"])
    table.on("rowClick", _click)


def _set_launch_orbit(key: str) -> None:
    global _launch_orbit
    _launch_orbit = key
    _apply_mission()


def _as_date(value, fallback: date) -> date:
    """A date widget's text as a ``date``, or ``fallback`` while it is still being typed."""
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip())
    except (TypeError, ValueError):
        return fallback


def _reconcile_mission_dates(prev: Mission, opens: date, closes: date, arrives: date,
                             returns: date | None, edited: str | None):
    """Move whichever dates the edited one just put out of order, and name the ones that moved.

    The mission's dates are ordered by definition: the window opens before it closes, arrival
    cannot precede the last launch it could have come from, and a return follows the arrival.
    Editing any one of them therefore passes through combinations the model refuses, as pushing the
    window out a year inverts it until the closing date is edited too. Refusing the edit means the
    field just typed into keeps its text while the model keeps the old value, so the date reappears
    the next time the panel is built and there is no order of edits that gets anywhere.

    The edited date is authoritative. Dates after it shift by the same amount it moved, which
    preserves the window's length and the transfer duration, so moving a mission later moves all of
    it. Pulling arrival earlier is a deadline rather than a shift, so the window closes at the new
    arrival instead of sliding backwards.

    Returns ``(opens, closes, arrives, returns, moved)``."""
    moved: list[str] = []
    if edited == "mis_arrive":
        if closes > arrives:                     # cannot still be launching after arrival is due
            closes = arrives
            moved.append("launch closes")
            if opens > closes:
                opens = closes
                moved.append("launch opens")
    elif edited == "mis_close":
        if closes < opens:
            opens = closes
            moved.append("launch opens")
        if arrives < closes:
            arrives = max(arrives + (closes - prev.launch_window[1]), closes)
            moved.append("arrive by")
    elif edited == "mis_open":
        if closes < opens:                       # carry the window and the transfer along
            shift = opens - prev.launch_window[0]
            closes = max(closes + shift, opens)
            arrives = max(arrives + shift, closes)
            moved += ["launch closes", "arrive by"]
    if returns is not None and returns < arrives:
        returns = arrives
        moved.append("return by")
    return opens, closes, arrives, returns, moved


def _apply_mission(_=None, edited: str | None = None) -> None:
    """Rebuild the working Mission from the live fields; on a validation error keep the prior
    model and surface the message inline rather than crashing the editor."""
    ret = bool(_w["mis_return"].value)
    m_prev = S.mission                       # preserve fields not edited here (return max-TOF)
    opens = _as_date(_w["mis_open"].value, m_prev.launch_window[0])
    closes = _as_date(_w["mis_close"].value, m_prev.launch_window[1])
    arrives = _as_date(_w["mis_arrive"].value, m_prev.arrive_by)
    returns = (_as_date(_w["mis_rby"].value, m_prev.return_by or arrives)
               if ret and _w.get("mis_rby") else None)
    opens, closes, arrives, returns, moved = _reconcile_mission_dates(
        m_prev, opens, closes, arrives, returns, edited)
    try:
        m = Mission(
            name=_w["mis_name"].value or "Untitled mission",
            launch_window=(opens, closes),
            launch_orbit=_launch_orbit,
            arrive_by=arrives,
            return_trip=ret,
            # The return-leg widgets are built by a refreshable that the toggle rebuilds; on the
            # rebuild tick they may not be in ``_w`` yet, so each field falls back to the prior
            # mission's value (and return_by to arrive_by) - that keeps the mission VALID the
            # instant return is enabled, rather than failing "return_by is missing" and silently
            # dropping the toggle. Once the fields render, edits flow through normally.
            return_by=(returns if ret and returns is not None
                       else (m_prev.return_by or arrives) if ret else None),
            return_destination=(_w["mis_dest"].value if ret and _w.get("mis_dest")
                                else m_prev.return_destination),
            # 0 means "use the destination's catalog insertion dV" -> store None (no override).
            return_orbit_insert_dv=((float(_w["mis_rins"].value) or None) if ret and _w.get("mis_rins")
                                    else (m_prev.return_orbit_insert_dv if ret else None)),
            return_max_tof_days=m_prev.return_max_tof_days if ret else None,
            time_at_asteroid=(_w["mis_stay"].value if ret and _w.get("mis_stay")
                              else (m_prev.time_at_asteroid if ret else None)),
            asteroid_payload_mass=(_w["mis_pay"].value if ret and _w.get("mis_pay")
                                   else (m_prev.asteroid_payload_mass if ret else 0.0)),
        )
    except (ValidationError, ValueError) as exc:
        _show_error(_mission_err, exc)
        return
    _show_error(_mission_err, None)
    S.mission = m
    for key, value in (("mis_open", m.launch_window[0]), ("mis_close", m.launch_window[1]),
                       ("mis_arrive", m.arrive_by), ("mis_rby", m.return_by)):
        widget = _w.get(key)
        if widget is not None and key != edited and value is not None:
            widget.value = value.isoformat()
    if moved:
        ui.notify(f"Moved {', '.join(moved)} to keep the dates in order", type="info")
    _mark_dirty()
    _stats.refresh()


# ======================================================================================
# Vehicle
# ======================================================================================

def _vehicle_panel() -> None:
    global _engine_type, _extra_mounts, _vehicle_err, _propellant, _array_margin
    v = S.vehicle
    _extra_mounts = list(v.engines[1:])
    head = v.engines[0] if v.engines else EngineMount(type="", count=1)
    _engine_type = head.type
    _propellant = v.propellant
    _array_margin = v.array_margin_pct
    with ui.column().classes("w-full gap-3"):
        _w["veh_name"] = ui.input("Vehicle name", value=v.name,
                                  on_change=_apply_vehicle).props("dense outlined").classes("w-full")
        with ui.row().classes("gap-4 w-full no-wrap"):
            _w["veh_dry"] = ui.number("Dry mass (kg)", value=float(v.dry_mass), min=1.0, step=10.0,
                                      on_change=_apply_vehicle).props("dense outlined").classes("flex-grow")
            _w["veh_fuel"] = ui.number("Propellant (kg)", value=float(v.fuel_mass), min=1.0, step=10.0,
                                       on_change=_apply_vehicle).props("dense outlined").classes("flex-grow")
            _w["veh_unus"] = ui.number("Unusable prop. (kg)", value=float(v.unusable_prop), min=0.0,
                                       step=1.0, on_change=_apply_vehicle).props(
                "dense outlined").classes("flex-grow")

        section("Engine", "bolt")
        _engine_row()
        if _extra_mounts:
            ui.label(f"+{len(_extra_mounts)} more engine blocks, edit in YAML").style(
                f"color:{MUTED};font-size:.7rem;font-style:italic")
        _propellant_row()

        section("Solar array", "wb_sunny")
        with ui.row().classes("gap-3 w-full no-wrap items-center"):
            _w["veh_margin"] = ui.number(
                "Array margin (%)",
                value=(None if _array_margin is None else float(_array_margin)),
                step=5.0, on_change=lambda e: _set_array_margin(e.value)).props(
                "dense outlined").style("width:170px")
            _w["veh_margin"].tooltip(
                "Reserve on top of the worst-mode power load. Negative means the array is "
                "undersized on purpose and the vehicle flies at reduced thrust.")
            ui.label("blank = build-model default").style(
                f"color:{MUTED};font-size:.68rem;font-style:italic")
        _array_derivation()

        ui.separator().style(f"background:{BORDER}")
        _tank_derivation()

        _vehicle_err = ui.label("").style(f"color:{ACCENT};font-size:.75rem")
        _vehicle_err.set_visibility(False)


@ui.refreshable
def _engine_row() -> None:
    """The single engine type + count, plus access to the engine library. (One type per
    vehicle here; sweep different pairings in Compare vehicles.)"""
    catalog = load_engines()
    options = sorted(set(catalog) | ({_engine_type} if _engine_type else set()))
    with ui.row().classes("gap-3 w-full items-center no-wrap"):
        # A plain dropdown rather than a filterable combobox, for the same reason as the engine
        # library's picker: a combobox looks like a text box, throws away what is typed into it,
        # and drops its menu over the field beside it.
        ui.select(options, value=_engine_type if _engine_type in options else (options[0] if options else None),
                  label="Engine type",
                  on_change=lambda e: _set_engine_type(e.value)).props(
            "dense outlined").classes("flex-grow")
        _w["veh_count"] = ui.number("How many", value=int(S.vehicle.engines[0].count if S.vehicle.engines else 1),
                                    min=1, step=1, on_change=_apply_vehicle).props(
            "dense outlined").style("width:120px")
        ui.button("Engine library", icon="bolt", on_click=_engine_dialog).props(
            "outline dense no-caps").style(f"color:{MUTED}")


def _set_engine_type(value: str | None) -> None:
    global _engine_type
    if value:
        _engine_type = value
        _propellant_row.refresh()    # the native (measured) gas may have changed
        _apply_vehicle()


def _native_gas() -> str:
    """The gas the current head engine's thrust/Isp are authored on (its native propellant)."""
    return getattr(load_engines().get(_engine_type), "propellant", None) or "xenon"


@ui.refreshable
def _propellant_row() -> None:
    """Pick the working gas LOADED on the vehicle. When it differs from the engine's native
    (measured) gas, the thrust and Isp are estimated by scaling the native numbers, flagged
    here, the same convention the design sweep uses."""
    props = list_propellants() or ["xenon"]
    native = _native_gas()
    selected = _propellant or native
    if selected not in props:
        selected = native if native in props else props[0]
    with ui.row().classes("gap-3 w-full items-center no-wrap"):
        ui.select(props, value=selected, label="Propellant loaded",
                  on_change=lambda e: _set_propellant(e.value)).props(
            "dense outlined").style("width:200px")
        if selected != native:
            ui.label(f"⚠ estimated - thrust/Isp scaled from {native}").style(
                f"color:{AMBER};font-size:.72rem")
        else:
            ui.label("measured, not scaled").style(
                f"color:{MUTED};font-size:.72rem")


def _set_propellant(value: str | None) -> None:
    global _propellant
    # Store None when the loaded gas is the engine's native gas (no scaling - the default);
    # otherwise store the chosen gas so resolving scales the engine to it.
    _propellant = None if (not value or value == _native_gas()) else value
    _propellant_row.refresh()
    _apply_vehicle()


def _apply_vehicle(_=None) -> None:
    try:
        head = EngineMount(type=_engine_type, count=int(_w["veh_count"].value or 1))
        v = Vehicle(name=_w["veh_name"].value or "Untitled vehicle",
                    dry_mass=_w["veh_dry"].value, fuel_mass=_w["veh_fuel"].value,
                    unusable_prop=_w["veh_unus"].value,
                    propellant=_propellant,        # the loaded gas (None = engine's native)
                    array_margin_pct=_array_margin,
                    engines=[head, *_extra_mounts])
    except (ValidationError, ValueError, TypeError) as exc:
        _show_error(_vehicle_err, exc)
        return
    # The array power and the drag area are sized from the loads, the power chain and the margin
    # rather than typed, and written onto the vehicle here so the editor's own cards show the same
    # array the spiral and the cruise will fly.
    v = buildability.with_sized_array(v, load_engines())
    _show_error(_vehicle_err, None)
    S.vehicle = v
    _mark_dirty()
    _stats.refresh()
    _array_derivation.refresh()
    _tank_derivation.refresh()
    # The top-bar vehicle chip mirrors the name; refresh it on a rename (cheap, only on edit).
    from ui import topbar
    topbar.header.refresh()


def _set_array_margin(value) -> None:
    global _array_margin
    _array_margin = None if value is None or value == "" else float(value)
    _apply_vehicle()


def _tank_sizing_terms(v: Vehicle, model, catalog) -> dict:
    """The tank-sizing derivation: the propellant volume at its storage density sets a
    pressure-vessel of fixed shape, whose wall mass follows from the burst pressure. The
    intermediate terms feed the explainer; the mass is :func:`buildability.tank_dry_mass_kg`."""
    from prospector.spacecraft.propellants import (
        load_propellants,
        resolve_propellant,
        storage_density,
    )
    prop_kg = float(v.fuel_mass)
    head_type = v.engines[0].type if v.engines else None
    gas_key = v.propellant or getattr(catalog.get(head_type), "propellant", None) or "xenon"
    props = load_propellants()
    prop = resolve_propellant(gas_key, props) if props else None
    density = (storage_density(prop, model.tank_pressure_bar, model.tank_temperature_C)
               if prop else None) or buildability.REFERENCE_DENSITY_KG_M3
    volume = prop_kg / density if density > 0 else 0.0
    return {"prop_kg": prop_kg, "gas": (prop.name if prop else str(gas_key)), "density": density,
            "volume_m3": volume, "tank_kg": buildability.tank_dry_mass_kg(volume, model),
            "ld_ratio": model.tank_length_diameter_ratio, "pressure_bar": model.tank_pressure_bar,
            "temp_C": model.tank_temperature_C, "safety": model.tank_safety_factor}


def _derivation_card(title: str, icon: str, rows: list[tuple], notes: list[str]) -> None:
    """A compact 'where this number came from' explainer: a titled panel of label→value rows over one
    or more plain-language note lines. Used for the array- and tank-sizing sections.

    A row is ``(label, value)``, or ``(label, value, tooltip)`` where the derivation behind the
    number is worth having on hover but not on screen."""
    with ui.element("div").classes("w-full").style(
            f"background:{PANEL2};border:1px solid {BORDER};border-radius:8px;padding:10px 12px"):
        with ui.row().classes("items-center gap-1"):
            ui.icon(icon).style(f"color:{ACCENT}").classes("text-sm")
            ui.label(title).style(f"color:{TEXT};font-weight:600;font-size:.8rem")
        for label, val, *tip in rows:
            with ui.row().classes("items-center w-full no-wrap justify-between").style(
                    "gap:.5rem") as row:
                if tip and tip[0]:
                    row.tooltip(tip[0])
                ui.label(label).style(f"color:{MUTED};font-size:.74rem")
                ui.label(val).style(f"color:{TEXT};font-family:monospace;font-size:.74rem;text-align:right")
        for note in notes:
            ui.label(note).style(f"color:{MUTED};font-size:.68rem;margin-top:.25rem;line-height:1.35")


@ui.refreshable
def _array_derivation() -> None:
    """Where the array power comes from: the heaviest operating mode's load, scaled up for the
    losses each system sits behind, times (1 + margin)."""
    model = buildability.load_bus_model()
    t = buildability.array_sizing_terms(S.vehicle, load_engines(), model)
    if t is None:
        _derivation_card("How the array is sized", "wb_sunny", [],
                         ["Pick a valid engine to size the array."])
        return
    mode = "thrusting" if t["thrust_limited"] else "full-ops (payload + comms)"
    load_line = (f"housekeeping {t['housekeeping_W']:.0f} W + thrusters {t['thrusters_W']:.0f} W"
                 if t["thrust_limited"]
                 else f"housekeeping {t['housekeeping_W']:.0f} W + payload/comms {t['ops_load_W']:.0f} W")
    loss_tip = ("Grossed up for the losses on the way to each load: array-to-bus wiring "
                f"{model.transmission_eff_pct:.0f}%, high-to-low-voltage converter "
                f"{model.hvlv_converter_eff_pct:.0f}%, thruster power-processing unit "
                f"{model.ppu_eff_pct:.0f}%.")
    rating_tip = (f"The wattage the array must be rated for on its datasheet. Datasheets quote "
                  f"cells at {t['rating_temp_C']:.0f} °C; facing the Sun at 1 AU the panel runs at "
                  f"about {t['operating_temp_C']:.0f} °C, where the cells convert worse, so it "
                  f"delivers {t['operating_over_rated'] * 100:.0f}% of its rating there. Mass and "
                  f"area are charged at the rated figure; the flight uses the operating one.")
    rows = [
        ("Sized for", mode),
        ("Loads at that mode", load_line),
        ("Array output to meet loads", f"{max(t['thrust_mode'], t['ops_mode']):.0f} W", loss_tip),
        ("→ Array power when new, at 1 AU", f"{t['bol']:.0f} W"),
        ("→ Rated (datasheet) power", f"{t['rated_W']:.0f} W", rating_tip),
        ("→ Solar array mass", f"{t['array_kg']:.0f} kg  ·  {t['area_m2']:.1f} m²"),
    ]
    notes = ["Belt degradation is not sized against: a degraded array throttles down in flight."]
    _derivation_card("How the array is sized", "wb_sunny", rows, notes)


@ui.refreshable
def _tank_derivation() -> None:
    """How the propellant tank mass is derived: the propellant volume at its storage density sets
    a fixed-shape pressure vessel whose wall mass follows from the burst pressure."""
    model = buildability.load_bus_model()
    t = _tank_sizing_terms(S.vehicle, model, load_engines())
    rows = [
        ("Propellant loaded", f"{t['prop_kg']:.0f} kg  ({t['gas']})"),
        (f"Storage density @ {t['pressure_bar']:.0f} bar, {t['temp_C']:.0f}°C", f"{t['density']:.0f} kg/m³"),
        ("Propellant volume", f"{t['volume_m3'] * 1000:.1f} L"),
        ("Tank shape (length ÷ diameter)", f"{t['ld_ratio']:.1f}  (1.0 = sphere, lightest)"),
        ("Burst / safety factor", f"×{t['safety']:.1f}"),
        ("→ Tank dry mass", f"{t['tank_kg']:.1f} kg"),
    ]
    notes = ["A less dense gas needs a bigger, heavier tank."]
    _derivation_card("How the tank is sized", "local_gas_station", rows, notes)


# ======================================================================================
# Engine library editor (modal)
# ======================================================================================

# A pasted cell often keeps its unit ("4000 W") and a number often keeps a thousands separator
# ("4,000"). The comma is also a column separator, so the thousands separator is removed before the
# columns are read rather than being disentangled from them afterwards.
_THOUSANDS_SEP = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
# No sign is matched: none of power, thrust or Isp is ever negative, and it keeps a part number in
# a title line ("HT-2200 performance") from reading as a value.
_NUMBER = re.compile(r"(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _row_numbers(line: str) -> list[float]:
    """The numbers on one pasted row, ignoring whatever punctuation or units surround them.

    Reading the numbers out is what makes comma, semicolon, tab and space separators all work
    without asking which one a spreadsheet chose, and lets a cell that kept its unit still
    yield its value."""
    return [float(m.group()) for m in _NUMBER.finditer(_THOUSANDS_SEP.sub("", line))]


def _parse_power_curve(text: str) -> tuple[list[PowerPoint] | None, str]:
    """Parse pasted ``power_W, thrust_mN, isp_s`` rows into sorted :class:`PowerPoint`s.

    Returns ``(points, message)``: a point list + a one-line summary on success, or ``(None,
    error)`` on a bad row. Blank input gives ``(None, "")``, meaning no curve and the constant-Isp
    linear-throttle fallback.

    A row carrying fewer than two numbers is a header or a title, and is skipped so a table
    copied straight from a datasheet parses. Anything else that fails is reported by name:
    text that arrived looking like data and left no curve behind has to say so, because the box
    still shows what was pasted, and a quiet "no curve" there reads as the engine having none.
    Points are sorted by power and must be distinct and positive."""
    rows: list[tuple[float, float, float]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        nums = _row_numbers(line)
        if len(nums) < 2:
            continue                          # a header or a title line, not a data row
        if len(nums) != 3:
            return None, f"each row needs 3 numbers (power, thrust, Isp) - got {len(nums)} in '{line}'"
        rows.append((nums[0], nums[1], nums[2]))
    if not rows and (text or "").strip():
        return None, ("no data rows found - expected one row of  power_W, thrust_mN, isp_s  "
                      "per line")
    return _curve_from_triples(rows)


def _curve_from_triples(rows: list[tuple[float, float, float]]
                        ) -> tuple[list[PowerPoint] | None, str]:
    """Validate ``(power, thrust, Isp)`` triples into a sorted curve, and describe the result.

    Shared by the row editor and the paste box, so a curve is held to the same rules and
    summarised the same way however it was entered. An empty list gives ``(None, "")``: no curve,
    which is a constant-Isp engine rather than an error."""
    if not rows:
        return None, ""
    for p, t, i in rows:
        if p <= 0 or t < 0 or i <= 0:
            return None, f"power and Isp must be > 0, thrust >= 0 (row {p:g}, {t:g}, {i:g})"
    if len(rows) < 2:
        return None, "a curve needs at least 2 points (or leave blank for constant Isp)"
    rows = sorted(rows, key=lambda r: r[0])
    powers = [r[0] for r in rows]
    if len(set(powers)) != len(powers):
        return None, "each power point must be distinct"
    pts = [PowerPoint(power_W=p, thrust_mN=t, isp_s=i) for p, t, i in rows]
    return pts, f"{len(pts)} points · {powers[0]:.0f}–{powers[-1]:.0f} W"


# The throttle-curve rows the engine dialog is editing, and the label describing them. Held at
# module level, like the vehicle workspace's engine pairings: a refreshable defined inside the
# dialog closure does not redraw when refreshed from a button inside that same dialog.
_curve_rows: list[dict] = []
_curve_status = None


def _curve_triples() -> list[tuple[float, float, float]]:
    """The rows as ``(power, thrust, Isp)``, dropping any row still entirely empty.

    A partly-filled row is kept, with its blanks read as zero, so the validator names it rather
    than the curve quietly losing a point."""
    out = []
    for r in _curve_rows:
        vals = (r["p"], r["t"], r["i"])
        if all(v is None for v in vals):
            continue
        out.append(tuple(0.0 if v is None else float(v) for v in vals))
    return out


def _refresh_curve_status() -> None:
    """Update the summary under the rows. Kept out of the refreshable, because refreshing on
    every keystroke would rebuild the number fields under the cursor."""
    if _curve_status is None:
        return
    pts, msg = _curve_from_triples(_curve_triples())
    bad = pts is None and bool(msg)
    _curve_status.text = ("⚠ " + msg) if bad else (msg or "no curve, constant Isp")
    _curve_status.style(f"color:{ACCENT if bad else MUTED};font-size:.7rem")


def _redraw_curve() -> None:
    """Rebuild the rows and their summary.

    Kept apart from the mutations below so the row state can be exercised without a live client,
    and so a keystroke can update the summary alone. A full refresh would rebuild the number
    fields under the cursor."""
    _curve_editor.refresh()
    _refresh_curve_status()


def _set_curve_point(i: int, key: str, value) -> None:
    _curve_rows[i][key] = value


def _add_curve_point() -> None:
    _curve_rows.append({"p": None, "t": None, "i": None})


def _remove_curve_point(i: int) -> None:
    _curve_rows.pop(i)


def _set_curve_rows(pts) -> None:
    """Load an engine's curve into the rows (``None``/empty clears them)."""
    _curve_rows.clear()
    _curve_rows.extend({"p": pt.power_W, "t": pt.thrust_mN, "i": pt.isp_s} for pt in (pts or []))


def _apply_curve_paste(paste) -> None:
    """Fill the rows from a pasted table. The rows are the editor; this is the fast way in for a
    vendor sheet carrying a dozen operating points."""
    pts, msg = _parse_power_curve(paste.value)
    if pts is None:
        ui.notify(msg or "nothing to paste", type="warning")
        return
    _set_curve_rows(pts)
    _redraw_curve()
    paste.value = ""
    ui.notify(f"Loaded {len(pts)} points", type="positive")


@ui.refreshable
def _curve_editor() -> None:
    """One row of (power, thrust, Isp) per measured operating point, plus an Add button."""
    if _curve_rows:                      # column headings once, not a label on all three fields
        with ui.row().classes("w-full no-wrap gap-2"):
            for head in ("Power (W)", "Thrust (mN)", "Isp (s)"):
                ui.label(head).style(f"color:{MUTED};font-size:.64rem;flex:1 1 0;min-width:0")
            ui.element("div").style("width:30px;flex:0 0 auto")      # over the remove column
    for idx, row in enumerate(_curve_rows):
        with ui.row().classes("items-center gap-2 w-full no-wrap"):
            # No min= on these: a bound clamps on blur, editing the number just typed. The
            # validator reports a bad value instead of moving it.
            for key, step in (("p", 100.0), ("t", 5.0), ("i", 10.0)):
                ui.number(value=row[key], step=step,
                          on_change=lambda e, i=idx, k=key: (_set_curve_point(i, k, e.value),
                                                             _refresh_curve_status())
                          ).props("dense outlined").style("flex:1 1 0;min-width:0")
            ui.button(icon="close",
                      on_click=lambda i=idx: (_remove_curve_point(i), _redraw_curve())).props(
                "flat dense round").style(f"color:{MUTED};flex:0 0 auto")
    ui.button("Add point", icon="add",
              on_click=lambda: (_add_curve_point(), _redraw_curve())).props(
        "outline dense no-caps").style(f"color:{ACCENT}")


def _engine_dialog() -> None:
    """Add or edit an engine in the shared library. Saving writes the YAML and refreshes the
    vehicle's engine picker so a newly-added type is immediately selectable."""
    catalog = load_engines()
    keys = list_engines()
    props = list_propellants() or ["xenon"]
    start = _engine_type if _engine_type in catalog else (keys[0] if keys else None)

    with ui.dialog() as dlg, ui.card().style(
            f"background:{PANEL};border:1px solid {BORDER};min-width:430px"):
        ui.label("Engine library").style(f"color:{TEXT};font-weight:600")
        # A plain dropdown rather than a filterable combobox. A combobox renders as an editable
        # text box sitting directly above the Name field, so it reads as the place to type an
        # engine's name, but its text only filters the list and is discarded on blur, and while it
        # is open the menu covers the fields beneath it, where a click meant for Name or the
        # throttle curve lands on a list item and silently swaps the engine being edited. Keyboard
        # type-ahead still jumps to a match, which is what the filter was worth.
        picker = ui.select(keys, value=start, label="Edit engine").props(
            "dense outlined").classes("w-full")
        name = ui.input("Name").props("dense outlined").classes("w-full")
        with ui.row().classes("gap-3 w-full no-wrap"):
            isp = ui.number("Isp (s)", min=1.0, step=10.0).props("dense outlined").classes("flex-grow")
            thrust = ui.number("Thrust (mN)", min=0.1, step=5.0).props("dense outlined").classes("flex-grow")
        with ui.row().classes("gap-3 w-full no-wrap"):
            power = ui.number("Power (W)", min=0.0, step=50.0).props("dense outlined").classes("flex-grow")
            mass = ui.number("Mass (kg)", min=0.0, step=0.1).props("dense outlined").classes("flex-grow")
            cost = ui.number("Unit cost ($M)", min=0.0, step=0.1).props("dense outlined").classes("flex-grow")
            cost.tooltip("Rough cost including the power-processing and feed systems, for "
                         "comparison only.")
        ui.label("Propellant these numbers were measured on").style(
            f"color:{MUTED};font-size:.7rem;margin-top:.2rem")
        prop = ui.radio(props, value=props[0]).props("inline dense").classes("w-full")
        ui.label("Power-processing unit fed from").style(
            f"color:{MUTED};font-size:.7rem;margin-top:.2rem")
        ppu_side = ui.radio({"high": "High voltage", "low": "Low voltage"},
                            value="low").props("inline dense").classes("w-full")
        ppu_side.tooltip("High voltage skips the converter, so more of the array's power "
                         "reaches the thruster.")
        with ui.row().classes("gap-3 w-full no-wrap"):
            thru = ui.number("Throughput (kg)", min=0.0, step=10.0).props(
                "dense outlined").classes("flex-grow")
            thru.tooltip("Qualified propellant throughput per thruster (kg); blank = unspecified.")
            ign = ui.number("Ignition cycles", min=0, step=100).props("dense outlined").classes("flex-grow")
            ign.tooltip("Qualified on/off cycles per thruster; blank = unspecified.")

        # Optional power->(thrust, Isp) throttle curve, a row per measured operating point. With a
        # curve, an array that cannot supply full power drops both thrust and Isp along it, in the
        # escape spiral and the cruise. No rows = the rated point only, with thrust throttled
        # linearly at constant Isp when power-limited.
        ui.label("Power throttle curve (optional): thrust and Isp against available power").style(
            f"color:{MUTED};font-size:.7rem;margin-top:.3rem")
        mode = ui.radio({"continuous": "Continuous throttle", "discrete": "Discrete modes"},
                        value="continuous").props("inline dense").classes("w-full")
        mode.tooltip("Continuous: any power between the points, thrust and Isp interpolated. "
                     "Discrete: the points are the only settings; short of power the thruster "
                     "drops to the highest mode it can run and holds that mode's power, thrust "
                     "and Isp. With no curve, a discrete thruster is on at rated or off.")
        _curve_editor()
        global _curve_status
        _curve_status = ui.label("").style(f"color:{MUTED};font-size:.7rem")
        _refresh_curve_status()

        with ui.expansion("Paste a datasheet table", icon="content_paste").classes(
                "w-full").props("dense"):
            # The placeholder is a shape, not data: round illustrative numbers that show the column
            # order and that thrust and Isp both rise with power. A real datasheet row here would
            # put one vendor's measured performance on every screen the app is on.
            paste = ui.textarea(
                placeholder="4000, 120, 3200\n7000, 210, 3700\n10000, 300, 4000").props(
                'dense outlined autogrow input-style="font-family:monospace;font-size:.75rem"'
            ).classes("w-full")
            paste.tooltip("One row per point, any separator; header lines and units in the "
                          "cells are ignored.")
            ui.button("Replace rows with this", icon="playlist_add",
                      on_click=lambda: _apply_curve_paste(paste)).props(
                "outline dense no-caps").style(f"color:{ACCENT}")

        err = ui.label("").style(f"color:{ACCENT};font-size:.75rem")
        err.set_visibility(False)

        def _load(key: str | None) -> None:
            e = catalog.get(key) if key else None
            if e is None:
                return
            name.value, isp.value, thrust.value = e.name, e.isp_s, e.thrust_mN
            power.value, mass.value = e.power_W, e.mass_kg
            cost.value = e.cost_musd
            prop.value = e.propellant if e.propellant in props else props[0]
            ppu_side.value = e.ppu_bus_side
            mode.value = e.throttle_mode
            thru.value = e.throughput_kg
            ign.value = e.lifetime_ignitions
            _set_curve_rows(e.power_curve)
            _redraw_curve()

        picker.on_value_change(lambda e: _load(e.value))

        def _build() -> Engine | None:
            pts, msg = _curve_from_triples(_curve_triples())
            if pts is None and msg:
                err.text = f"Power curve: {msg}"
                err.set_visibility(True)
                return None
            try:
                return Engine(name=(name.value or "").strip() or "New engine",
                              isp_s=isp.value, thrust_mN=thrust.value,
                              power_W=power.value or 0.0, mass_kg=mass.value or 0.0,
                              cost_musd=cost.value if cost.value is not None else 1.5,
                              propellant=prop.value or "xenon",
                              ppu_bus_side=ppu_side.value or "low",
                              throttle_mode=mode.value or "continuous",
                              throughput_kg=thru.value or None,
                              lifetime_ignitions=int(ign.value) if ign.value else None,
                              power_curve=pts)
            except (ValidationError, ValueError, TypeError) as exc:
                err.text = str(exc).splitlines()[0]
                err.set_visibility(True)
                return None

        def _save(key: str) -> None:
            eng = _build()
            if eng is None:
                return
            save_engine(eng, key)
            ui.notify(f"Saved engine '{key}'", type="positive")
            _engine_row.refresh()
            dlg.close()

        def _new() -> None:
            for el in (name, isp, thrust, power, mass, cost, thru, ign):
                el.value = None
            prop.value = props[0]
            ppu_side.value = "low"
            mode.value = "continuous"
            picker.value = None
            _set_curve_rows([])
            _redraw_curve()
            err.set_visibility(False)

        with ui.row().classes("justify-between w-full gap-2 mt-2 no-wrap"):
            ui.button("New", icon="add", on_click=_new).props("flat dense no-caps").style(f"color:{MUTED}")
            ui.space()
            ui.button("Cancel", on_click=dlg.close).props("flat no-caps").style(f"color:{MUTED}")
            ui.button("Save as new", on_click=lambda: _save(state.slug(name.value or ""))).props(
                "outline no-caps").style(f"color:{TEXT}")
            ui.button("Save", on_click=lambda: _save(picker.value or state.slug(name.value or ""))).props(
                "unelevated no-caps")

        if start:
            _load(start)
    dlg.open()


# ======================================================================================
# Vehicle summary dialog (shown when adopting a traded vehicle as the project default)
# ======================================================================================

def vehicle_summary_dialog(on_confirm) -> None:
    """A confirm dialog summarizing the working vehicle before it becomes the project default:
    the derived performance (wet mass, thrust, power, Isp, ΔV) and the spacecraft-configurator
    build sizing (array power, solar array, payload capacity, rough cost). ``on_confirm`` commits."""
    v = S.vehicle
    try:
        rc = state.resolved()
    except (ValidationError, ValueError) as exc:
        ui.notify(f"Vehicle does not resolve: {str(exc).splitlines()[0]}", type="negative")
        return
    build = assess_build(rc)

    with ui.dialog() as dlg, ui.card().style(
            f"background:{PANEL};border:1px solid {BORDER};min-width:460px"):
        ui.label("Make this the project default vehicle?").style(
            f"color:{TEXT};font-weight:600")
        ui.label(v.name).style(f"color:{MUTED};font-size:.82rem")

        _summary_cards(vehicle_perf_rows(rc))

        if build is not None:
            section("Build sizing", "wb_sunny")
            sizing = [("Array power (new)", f"{build['bol_power_W']:.0f} W"),
                      ("Solar array", f"{build['array_kg']:.0f} kg"),
                      ("Payload + margin", f"{build['payload_capacity_kg']:.0f} kg"),
                      ("Build cost", f"${build['build_cost_musd']:.1f}M")]
            _summary_cards(sizing)
            why = str(build.get("build_why") or "")
            if why:
                color = GREEN if build.get("buildable") else AMBER
                with ui.row().classes("items-center gap-1 mt-1"):
                    ui.icon("construction").style(f"color:{color}").classes("text-sm")
                    ui.label(why).style(f"color:{color};font-size:.76rem")
        else:
            ui.label("Build sizing unavailable for this engine.").style(
                f"color:{MUTED};font-size:.74rem")

        with ui.row().classes("justify-end w-full gap-2 mt-2"):
            ui.button("Cancel", on_click=dlg.close).props("flat no-caps").style(f"color:{MUTED}")
            ui.button("Make default", icon="save_as",
                      on_click=lambda: (on_confirm(), dlg.close())).props("unelevated no-caps")
    dlg.open()


def assess_build(rc) -> dict | None:
    """The spacecraft-configurator build assessment for the vehicle's head engine block."""
    v = rc.vehicle
    if not v.engines:
        return None
    head = v.engines[0]
    engine = rc.engines.get(head.type)
    if engine is None:
        return None
    try:
        return buildability.assess_engine(
            dry_kg=float(v.dry_mass), prop_kg=float(v.fuel_mass), n_engines=int(head.count),
            engine=engine, propellants=load_propellants(), margin_pct=v.array_margin_pct)
    except Exception:  # noqa: BLE001  (a sizing gap never blocks the confirm dialog)
        return None


def _summary_cards(cards) -> None:
    with ui.row().classes("w-full gap-2 no-wrap mt-1"):
        for label, val in cards:
            with ui.column().classes("gap-0 p-2 rounded-lg flex-grow items-center").style(
                    f"background:{PANEL2};border:1px solid {BORDER}"):
                ui.label(val).style(f"color:{TEXT};font-family:monospace;font-size:.95rem")
                ui.label(label).style(f"color:{MUTED};font-size:.66rem")


# ======================================================================================
# Derived stats + delta-v budget (always derived, never typed)
# ======================================================================================

@ui.refreshable
def _stats() -> None:
    section("Calculated", "calculate")
    try:
        rc = state.resolved()
    except (ValidationError, ValueError) as exc:
        ui.label(f"Config does not resolve: {str(exc).splitlines()[0]}").style(
            f"color:{ACCENT};font-size:.8rem")
        return
    with ui.row().classes("w-full gap-3 no-wrap"):
        for label, val in vehicle_perf_rows(rc):
            with ui.column().classes("gap-0 p-3 rounded-lg flex-grow items-center").style(
                    f"background:{PANEL2};border:1px solid {BORDER}"):
                ui.label(val).style(f"color:{TEXT};font-family:monospace;font-size:1.05rem")
                ui.label(label).style(f"color:{MUTED};font-size:.7rem")
    _budget_line(rc)


def _budget_line(rc) -> None:
    """One-line ΔV trade: total capability minus the Earth-escape charge leaves the cruise
    budget the screen works with. The escape tag says where the charge came from."""
    tag = {"launch vehicle": "rocket", "spiral": "spiral ✓", "estimate": "estimate"}[rc.escape_source]
    with ui.row().classes("items-center w-full no-wrap gap-1").style("margin-top:.2rem"):
        ui.icon("bolt").style(f"color:{ACCENT}").classes("text-sm")
        ui.label(f"ΔV {rc.total_dv_capability:.1f}").style(f"color:{TEXT};font-size:.8rem;font-weight:600")
        ui.label(f"− escape {rc.escape_dv:.1f} ({tag})  →  cruise budget").style(
            f"color:{MUTED};font-size:.78rem")
        ui.label(f"{rc.cruise_dv_limit:.1f} km/s").style(f"color:{GREEN};font-size:.8rem;font-weight:600")


# ======================================================================================
# behaviour
# ======================================================================================

def _set_study_name(name: str) -> None:
    S.study_name = name
    _mark_dirty()
    from ui import topbar
    topbar.header.refresh()


def flush() -> None:
    """Commit the live mission/vehicle editor widgets into the working models.

    Called right before a save so a value the user typed but whose change event hasn't been
    applied yet (focus still in the field, or an in-flight event) is captured rather than
    lost. A no-op unless the Project editor is the mounted workspace; it simply re-runs the
    same rebuild the field on_change handlers do, reading every widget's current value."""
    if S.workspace != "project":
        return
    if _w.get("mis_return") is not None:
        _apply_mission()
    if _w.get("veh_dry") is not None:
        _apply_vehicle()


def _mark_dirty() -> None:
    """Flip the dirty flag once (enabling the top-bar Save) and refresh the header that hosts
    it, but only on the first edit, so per-field changes do not rebuild the top bar each time."""
    if not S.dirty:
        S.dirty = True
        from ui import topbar
        topbar.header.refresh()


def _save_as_dialog() -> None:
    from ui import topbar
    with ui.dialog() as dlg, ui.card().style(
            f"background:{PANEL};border:1px solid {BORDER};min-width:360px"):
        ui.label("Save study as").style(f"color:{TEXT};font-weight:600")
        name = ui.input("Study name", value=S.study_name).props(
            "dense outlined autofocus").classes("w-full")

        def _ok() -> None:
            nm = (name.value or "").strip()
            if not nm:
                return
            flush()                       # capture any field edited but not yet committed
            try:
                notes = state.persist_study(nm, state.slug(nm))
            except (ValidationError, ValueError) as exc:
                ui.notify(f"Save failed: {str(exc).splitlines()[0]}", type="negative")
                return
            ui.notify(f"Saved study '{nm}'", type="positive")
            topbar.report_forks(notes)
            dlg.close()
            topbar.header.refresh()

        with ui.row().classes("justify-end w-full gap-2 mt-2"):
            ui.button("Cancel", on_click=dlg.close).props("flat no-caps").style(f"color:{MUTED}")
            ui.button("Save", on_click=_ok).props("unelevated no-caps")
    dlg.open()


# ======================================================================================
# helpers
# ======================================================================================

def _date_field(label: str, value: str, on_change):
    """A text date field with a calendar-picker popup; ``on_change`` fires on edit or pick."""
    with ui.input(label, value=value).props("dense outlined").classes("flex-grow") as d:
        with ui.menu().props("no-parent-event") as menu:
            ui.date().bind_value(d)
            with ui.row().classes("justify-end w-full"):
                ui.button("Close", on_click=menu.close).props("flat dense no-caps").style(f"color:{MUTED}")
        with d.add_slot("append"):
            ui.icon("event").classes("cursor-pointer").style(f"color:{MUTED}").on("click", menu.open)
    d.on_value_change(lambda e: on_change())
    return d


def _show_error(label, exc) -> None:
    if label is None:
        return
    if exc is None:
        label.set_visibility(False)
        return
    label.text = str(exc).splitlines()[0]
    label.set_visibility(True)
