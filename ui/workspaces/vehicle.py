"""Compare vehicles: browse the vehicle design-space sweeps, live or finished, and launch new ones.

A read-only view of the sessions :mod:`prospector.trades.design_search` writes under
``runs/vehicle_search/<session>/``, with ``ui.vehicle_search`` as the data layer, plus the one
thing this workspace can change: starting a fresh sweep as a detached subprocess. The sweep runs
the whole launch-and-cruise pipeline for each candidate vehicle, varying dry mass, propellant and
engine count, against the open study's mission, so the canvas shows the trade across whichever
session is selected: the scatter, the grid of designs, and the trend charts.

The canvas carries the Plot, Table and Trends tabs, the right-hand rail holds the filters that
narrow what is shown, and the dock along the bottom is where sweeps get started and recent ones
listed. Picking a design in the grid floats a card whose button hands that vehicle to Plan
trajectory. A running session streams its progress into the dock. Starting a sweep is the only
thing that changes anything, and solves never run inside the app.
"""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from nicegui import ui

from prospector import figures
from prospector.config import EngineMount, Vehicle
from prospector.spacecraft import buildability
from prospector.spacecraft.propellants import list_propellants
from prospector.spacecraft.propulsion import load_engines
from ui import state
from ui import vehicle_search as vs
from ui.components import (
    canvas_box,
    labeled_range,
    labeled_slider,
    section,
    selectable_table,
    subtabs,
    workspace_frame,
)
from ui.state import S
from ui.theme import ACCENT, AMBER, BORDER, GREEN, MUTED, PANEL, PANEL2, RED, TEXT

_PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": True}
_VIEWS = [("Table", "grid_on"), ("Plot", "scatter_plot"), ("Trends", "show_chart")]

# Readable labels for the columns worth featuring at the top of the axis pickers (in this order).
# Any OTHER numeric column the rows carry is offered too - swept model settings (``param_<key>``)
# and the internal numerics - so the scatter plots over all of them; this map only controls naming
# and ordering of the featured few. Column -> label.
_AXIS_LABELS = {
    "dry_kg": "Dry mass (kg)", "prop_kg": "Propellant (kg)", "wet_kg": "Wet mass (kg)",
    "prop_margin_kg": "Fuel margin (kg)", "margin_kms": "ΔV margin at rated Isp (km/s)",
    "total_dv_kms": "Total ΔV (km/s)", "payload_capacity_kg": "Payload + margin (kg)",
    "total_tof_days": "Mission time (d)", "build_cost_musd": "Build cost ($M)",
    "prop_dry_ratio": "Prop/dry", "n_engines": "Engine count", "array_kg": "Array (kg)",
    "return_dv_kms": "Return ΔV (km/s)", "total_prop_kg": "Total prop (kg)",
    "delivered_payload_kg": "Delivered (kg)", "round_trip_days": "Round trip (d)",
}

# Numeric-but-uninformative columns (pure identity/flags) kept out of the auto-discovered axes.
# ``power_limited`` is folded into the flyable verdict (a power-limited escape is not flyable), so
# it is not offered as its own axis.
_AXIS_SKIP = {"anchor", "propellant_estimated", "power_limited"}


def _axis_label(col: str) -> str:
    """A readable axis label for any column: curated where known, the swept-setting label
    (with unit) for ``param_<key>`` columns, else the column name humanized."""
    if col in _AXIS_LABELS:
        return _AXIS_LABELS[col]
    if col.startswith("param_"):
        spec = buildability.SWEEPABLE.get(col[len("param_"):])
        if spec:
            label, unit, _ = spec
            return f"{label} ({unit})" if unit else label
    return col.replace("_", " ").strip().capitalize()

# The per-design grid, in one ordered ledger: (label, source column, kind, width-px, tone,
# category). The order reads left-to-right as identity → verdicts → masses → build sub-masses → ΔV
# → return → propellant → durations → power/efficiency → cost. Numeric kinds carry rounded numbers
# so the table sorts on real values; ``bool`` renders a ✓/✗ glyph, ``kw`` shows watts as kilowatts,
# ``g`` is a general number (trailing zeros trimmed, used for swept model params). ``tone`` drives
# cell rendering: chip_engine / chip_count -> a tinted category chip; verdict -> green/red ✓·✗;
# margin_dv / margin_kg -> green/amber/red health by sign + a tight band; short -> red when the
# design falls short; None -> plain text. ``category`` groups the column for the rail's per-group
# "Show …" toggles. The ``decision`` group (identity, verdict, headline masses + ΔV) is always
# shown; every other group is an additive toggle, so the table is tuned per question rather than
# all-or-nothing. The swept model settings form a ``params`` group rendered dynamically from each
# row's ``param_<key>`` columns (see :func:`_param_columns`), not listed here.
_COLUMNS = [
    # identity + verdicts (decision)
    ("Name", "slug", "str", 118, None, "decision"),
    ("Engine", "engine", "str", 130, "chip_engine", "decision"),
    ("Propellant", "propellant", "str", 92, "chip_prop", "decision"),
    ("Est.", "propellant_estimated", "estflag", 48, "estflag", "decision"),
    ("Closes", "success", "bool", 58, "verdict", "decision"),
    ("Builds", "buildable", "bool", 58, "verdict", "decision"),
    # masses (decision)
    ("Dry (kg)", "dry_kg", "f0", 62, None, "decision"),
    ("Prop (kg)", "prop_kg", "f0", 64, None, "decision"),
    ("Wet (kg)", "wet_kg", "f0", 64, None, "decision"),
    ("Eng", "n_engines", "int", 56, "chip_count", "decision"),
    # The verdict's own number: kilograms left in the tank after the whole mission. The legs run
    # at different Isps, so this, not a dV margin, is what "closes" means.
    ("Fuel margin (kg)", "prop_margin_kg", "f0", 84, "margin_kg", "decision"),
    # build sub-masses
    ("Array (kg)", "array_kg", "f1", 70, None, "build"),
    ("Thruster (kg)", "thruster_sys_kg", "f1", 78, None, "build"),
    ("Tank (kg)", "tank_kg", "f1", 64, None, "build"),
    ("PCDU (kg)", "pcdu_kg", "f1", 66, None, "build"),
    ("Bus (kg)", "fixed_bus_kg", "f0", 60, None, "build"),
    ("Payload+margin (kg)", "payload_capacity_kg", "f0", 96, None, "decision"),
    ("Min dry (kg)", "min_dry_kg", "f0", 76, None, "build"),
    # ΔV ledger, on the budget's basis (the escape term is propellant at the RATED Isp, the
    # capability likewise), so its margin can read positive while the tank comes up short. Detail,
    # not the verdict.
    ("ΔV margin, rated Isp (km/s)", "margin_kms", "f2", 104, "margin_dv", "dv"),
    ("Total ΔV (km/s)", "total_dv_kms", "f2", 80, None, "dv"),
    ("Escape ΔV (km/s)", "escape_dv_kms", "f2", 82, None, "dv"),
    ("Cruise ΔV (km/s)", "cruise_dv_kms", "f2", 82, None, "dv"),
    ("Capability (km/s)", "capability_kms", "f2", 86, None, "dv"),
    ("Short by (km/s)", "dv_short_kms", "f2", 80, "short", "dv"),
    # return leg (None for one-way missions)
    ("Return ΔV (km/s)", "return_dv_kms", "f2", 84, None, "return"),
    ("Insertion ΔV (km/s)", "return_insertion_dv_kms", "f2", 92, None, "return"),
    ("Return prop (kg)", "return_prop_kg", "f0", 80, None, "return"),
    ("Total prop (kg)", "total_prop_kg", "f0", 78, None, "return"),
    ("Delivered (kg)", "delivered_payload_kg", "f0", 78, None, "return"),
    # propellant
    ("Prop/dry", "prop_dry_ratio", "f2", 64, None, "prop"),
    # durations
    ("Spiral (d)", "spiral_tof_days", "f0", 62, None, "dur"),
    ("Cruise (d)", "cruise_tof_days", "f0", 62, None, "dur"),
    ("Return (d)", "return_tof_days", "f0", 62, None, "dur"),
    ("Mission (d)", "total_tof_days", "f0", 66, None, "dur"),
    ("Round trip (d)", "round_trip_days", "f0", 80, None, "dur"),
    # power + efficiency
    ("Array kW (new)", "bol_power_W", "kw", 76, None, "power"),
    ("Array left", "eol_factor", "f2", 66, None, "power"),
    ("Belt (d)", "belt_days", "f0", 56, None, "power"),
    ("Thrust/mass (mN/kg)", "thrust_mN_per_kg", "f2", 88, None, "power"),
    ("Power/mass (W/kg)", "power_W_per_kg", "f1", 84, None, "power"),
    ("Thrust/power (mN/W)", "thrust_mN_per_W", "f3", 88, None, "power"),
    # cost
    ("Cost ($M)", "build_cost_musd", "f1", 66, None, "cost"),
]

# Health bands for the margin tones: at/above the band is green, a thin positive band is amber
# (tight but positive), and negative is red.
_MARGIN_DV_BAND = 0.3      # km/s of spare ΔV
_MARGIN_KG_BAND = 10.0     # kg of fuel margin
# Stable category chip colors (engine reuses the scatter's cycle so the two views agree).
_ENGINE_CYCLE = figures.ENGINE_CYCLE
_COUNT_CYCLE = ("#5b8def", "#40df92", "#fbaf2a", "#ff6a3c", "#b07cff", "#ff5566")
# Stable per-gas chip colors (keyed by lowercase propellant name); unknown gases -> muted.
_PROP_COLOR = {"xenon": "#5b8def", "krypton": "#40df92", "iodine": "#b07cff"}
# Session-state chip colors in the recent-sweeps table.
_STATE_COLOR = {"done": GREEN, "running": AMBER, "error": RED, "stopped": MUTED}
# The always-shown base: the columns a design decision turns on (identity, verdict, headline
# masses, and headline ΔV) - derived from the ``decision`` category so the ledger above is the
# single source of truth.
_BASE_LABELS = {label for label, *_rest in _COLUMNS if _rest[-1] == "decision"}

# The toggleable column groups (key -> rail label), in display order. Each adds its columns on top
# of the always-shown base, so the grid is tuned per question. ``params`` is the swept model
# settings, rendered dynamically and only offered when a session carries them.
_CATEGORIES = [
    ("build", "Build masses"),
    ("dv", "ΔV detail"),
    ("return", "Return leg"),
    ("prop", "Propellant"),
    ("dur", "Durations"),
    ("power", "Power & efficiency"),
    ("cost", "Cost"),
    ("params", "Model params (swept)"),
]

# Column kinds that take a numeric min/max filter (everything but the str/bool/int identity
# columns; engine count is filtered as a multiselect instead). ``g`` (swept model params) is
# filterable too, so a sweep can be narrowed to a value of interest.
_NUMERIC_KINDS = ("f0", "f1", "f2", "f3", "kw", "g")


def _param_columns(rows: pd.DataFrame) -> list[tuple]:
    """Grid columns for the swept model settings (``param_<key>``) the rows carry, so the
    INPUT value behind each design is visible and sortable next to its outcome - the point of a
    model-setting sweep is reading the optimal knob value off the winning rows."""
    cols = sorted(c for c in rows.columns if str(c).startswith("param_"))
    return [(_axis_label(c), c, "g", 104, None, "params") for c in cols]


def _visible_spec(rows: pd.DataFrame) -> list[tuple]:
    """The grid columns currently shown: the always-on base plus every enabled category, in
    ledger order, with the swept-param columns appended when their group is on."""
    spec = [c for c in _COLUMNS if c[5] == "decision" or c[5] in _cols_on]
    if "params" in _cols_on:
        spec = spec + _param_columns(rows)
    return spec


def _numeric_columns(spec: list[tuple]) -> list[tuple]:
    return [(label, col, kind) for label, col, kind, _w, _t, _c in spec
            if kind in _NUMERIC_KINDS]

# ---- workspace-local state (persists across re-renders within the open project) ----
_session: str | None = None         # the selected session name
# Show every project's sweeps in the list, not just the open project's own (off per project).
_all_projects: bool = False
_view: str = "Table"                   # canvas sub-tab (the per-design grid is the landing view)
_cols_on: set = set()                  # enabled column groups beyond the always-on base
_sel_design: str | None = None      # the highlighted design (a design_id)
_pairings: list[dict] = []             # the launcher's engine/count-range pairings
# The rest of the new-sweep form, held here for the same reason as the pairings: the form is a
# modal that Quasar dismisses on any click outside it, and a sweep is expensive enough to define
# that losing a half-filled form to a stray click is not acceptable. Every widget reads its
# starting value from here and writes back on change, so a dismissed dialog reopens as it
# was left. Seeded by :func:`_seed_form` from the last sweep that ran (see _FORM_DEFAULTS).
_form: dict = {}
_scatter: dict = {"x": "Dry mass (kg)", "y": "Fuel margin (kg)", "c": "Engine config"}
# Buildable + flyable default ON, so the nominal view is just the viable designs (closes and
# builds); turning one off widens the table toward the near-misses on that axis.
_filters: dict = {"engines": [], "counts": [], "buildable": True, "flyable": True}
# Per-column numeric min/max filters keyed by source column -> {"min": v, "max": v}. The curated
# few render always; the full set appears alongside "Show all columns".
_num_filters: dict = {}
# True while the selected session was last seen running, so poll() can fire one final render on the
# running -> terminal edge.
_was_running = False
# The change-signature of the running session poll() last rendered (see _progress_sig). While a
# session is running poll() only rebuilds when this moves, so an idle or crashed sweep (frozen
# files) no longer thrashes the canvas every timer tick.
_last_sig: tuple | None = None
# Handle to the live grid table (Table view only). While a sweep streams, poll() updates this
# component's rows in place rather than rebuilding the canvas, so the user's sort + pagination
# (NiceGUI syncs both onto the component) survive each new batch of evaluations. None whenever the
# Table grid isn't the current canvas content (Plot/Trends, empty state, or before render).
_live_table = None


# ======================================================================================
# layout
# ======================================================================================

def render() -> None:
    workspace_frame(_canvas, _rail, _runner)


def _sessions() -> list[dict]:
    """The sweeps this workspace lists: the open project's own unless "all projects" is on."""
    sessions = vs.list_sessions()
    if _all_projects:
        return sessions
    return vs.sessions_for_project(sessions, S.project, S.mission_key,
                                   state.target_designation(S.project_target))


def _n_other_projects() -> int:
    """How many sweeps on disk belong to other projects (hidden unless "all projects" is on)."""
    return len(vs.list_sessions()) - len(vs.sessions_for_project(
        vs.list_sessions(), S.project, S.mission_key, state.target_designation(S.project_target)))


def _current(sessions: list[dict] | None = None) -> dict | None:
    """The selected session, defaulting to the newest when nothing valid is picked."""
    sessions = _sessions() if sessions is None else sessions
    if not sessions:
        return None
    hit = next((s for s in sessions if s["name"] == _session), None)
    return hit or sessions[0]


def _load(session: dict) -> dict:
    """Rows + metadata for one session, filtered to the live rail. The frame is loaded fresh
    each render so a running session's table grows as evaluations stream in."""
    meta = session.get("meta") or {}
    rows = vs.with_buildability(vs.load_rows(session["dir"]), meta.get("engine"), S.build_model)
    if len(rows):
        # Sessions predating per-row engine keys carry one in their metadata; falling back to a
        # literal would name an engine a different library may not hold.
        default_engine = meta.get("engine")
        rows = rows.assign(engine=(rows["engine"].fillna(default_engine)
                                   if "engine" in rows.columns else default_engine))
    rows = vs.with_design_id(rows)
    return {"rows": rows, "shown": _apply_filters(rows), "meta": meta}


# ======================================================================================
# canvas: in-view tabs (Plot / Table / Trends) + the floating selection HUD
# ======================================================================================

@ui.refreshable
def _canvas() -> None:
    global _live_table
    _live_table = None          # a full canvas rebuild invalidates any prior live table
    box = canvas_box()
    session = _current()
    data = _load(session) if session is not None else None
    with box:
        with ui.column().classes("w-full h-full p-2 gap-2").style("box-sizing:border-box"):
            with ui.row().classes("items-center w-full no-wrap").style("flex:0 0 auto;gap:.5rem"):
                subtabs(_VIEWS, _view, _set_view, full_width=False)
                ui.space()
                _stat_cards(data)
                # The new-sweep form lives in a modal (not the dock) so the workspace stays
                # uncramped; this button is its only entry point.
                ui.button("New sweep", icon="rocket_launch",
                          on_click=_open_sweep_dialog).props(
                    "unelevated dense no-caps").style("margin-left:.5rem")
            if data is None:
                _empty("balance", "no sweeps yet")
                return
            if not len(data["rows"]):
                _empty("hourglass_empty",
                       "evaluating… the first spiral + cruise solve takes a minute or two")
            elif _view == "Table":
                _table_view(session, data)
            elif _view == "Trends":
                _trends_view(data)
            else:
                _plot_view(data)
        _hud()


@ui.refreshable
def _stat_cards(data: dict | None) -> None:
    if data is None:
        return
    shown, total = data["shown"], data["rows"]
    n_close = int(vs.flag(shown, "success").sum()) if len(shown) else 0
    margin = (pd.to_numeric(shown["prop_margin_kg"], errors="coerce")
              if len(shown) and "prop_margin_kg" in shown.columns else None)
    best = f"{margin.max():+.0f}" if margin is not None and margin.notna().any() else "-"
    with ui.row().classes("items-center no-wrap gap-4"):
        _stat(f"{len(shown)}", f"of {len(total)} shown")
        _stat(f"{n_close}", "close", color=GREEN)
        _stat(best, "best fuel margin kg")


def _stat(value: str, label: str, color: str = TEXT) -> None:
    with ui.column().classes("items-center gap-0").style("padding:0 .3rem"):
        ui.label(value).style(f"color:{color};font-family:monospace;font-size:1.05rem;line-height:1")
        ui.label(label).style(f"color:{MUTED};font-size:.62rem")


def _empty(icon: str, text: str) -> None:
    with ui.column().classes("w-full items-center justify-center gap-2").style("flex:1 1 0"):
        ui.icon(icon).style(f"color:{BORDER};font-size:4.5rem")
        ui.label(text).style(f"color:{MUTED};font-size:.85rem;max-width:520px;text-align:center")


@ui.refreshable
def _hud() -> None:
    """The floating selection card (top-right of the canvas) handing a design to Plan trajectory."""
    sel = S.selected
    if not sel:
        return
    with ui.card().classes("absolute top-3 right-3 p-3 gap-1").style(
            f"background:{PANEL2};border:1px solid {ACCENT};border-radius:10px;"
            f"box-shadow:0 6px 24px rgba(0,0,0,.45);min-width:215px;max-width:300px"):
        with ui.row().classes("items-center w-full no-wrap"):
            ui.icon(sel.get("icon", "satellite_alt")).style(f"color:{ACCENT}")
            ui.label(sel["title"]).classes("truncate").style(
                f"color:{TEXT};font-weight:700;font-size:.85rem;max-width:220px")
            ui.space()
            ui.button(icon="close", on_click=_clear_selection).props("flat dense round").style(
                f"color:{MUTED}")
        for line in sel["lines"]:
            ui.label(line).style(f"color:{MUTED};font-size:.78rem")
        ui.button(sel["action"], icon=sel.get("action_icon", "rocket"),
                  on_click=sel.get("on_action", lambda: None)).props(
            "dense no-caps unelevated").classes("mt-1")


# ---- Plot view: the configurable trade scatter (its axis pickers live in the rail) ----

def _plot_view(data: dict) -> None:
    shown = data["shown"]
    if len(shown) < 2:
        _empty("scatter_plot", "at least two designs are needed for the trade scatter")
        return
    available = _scatter_columns(shown)
    if not available:
        _empty("scatter_plot", "no numeric columns to plot yet")
        return
    labels = list(available)
    x_lbl = _scatter["x"] if _scatter["x"] in labels else labels[0]
    y_lbl = _scatter["y"] if _scatter["y"] in labels else labels[-1]
    color_opts = {"Engine config": "engine", "Propellant": "propellant", **available}
    c_lbl = _scatter["c"] if _scatter["c"] in color_opts else "Engine config"
    fig = figures.trade_scatter(shown, available[x_lbl], available[y_lbl],
                              color_opts[c_lbl], x_label=x_lbl, y_label=y_lbl)
    _plot(fig)


def _scatter_columns(shown: pd.DataFrame) -> dict:
    """Axis label to column, for every numeric column the rows carry.

    Featured columns (``_AXIS_LABELS``) come first in their curated order, then the swept
    model settings (``param_<key>``), then any remaining numeric column - so the panel can
    plot over all of them, not a fixed list. Identity/flag numerics are skipped."""
    def numeric(col: str) -> bool:
        return (col in shown.columns
                and pd.to_numeric(shown[col], errors="coerce").notna().any())

    featured = [c for c in _AXIS_LABELS if numeric(c)]
    other = [c for c in shown.columns
             if c not in _AXIS_LABELS and c not in _AXIS_SKIP and numeric(c)]
    params = sorted(c for c in other if c.startswith("param_"))
    rest = sorted(c for c in other if not c.startswith("param_"))
    return {_axis_label(c): c for c in [*featured, *params, *rest]}


# ---- Table view: the per-design grid (the Show-all toggle lives in the rail) ----

def _table_view(session: dict, data: dict) -> None:
    global _live_table
    shown = data["shown"]
    columns, rows, toned = _grid(shown)
    with ui.element("div").style("flex:1 1 0;min-height:0;width:100%;overflow:auto"):
        # The row-click resolves the clicked design from the CURRENT session data (by id), not a
        # captured snapshot - so selection still works after poll() swaps rows in place.
        table = selectable_table(columns, rows, "design_id", _sel_design,
                                 lambda r: _pick_design_by_id(r.get("design_id")),
                                 rows_per_page=20)
        table.props("wrap-cells")     # wrap the header labels so each column keeps its set width
        for name, tone in toned:      # render category chips + health colors via cell slots
            _tone_slot(table, name, tone)
        _live_table = table           # poll() grows this in place while a sweep streams


def _grid(shown: pd.DataFrame):
    """NiceGUI (columns, rows, toned) for the per-design grid. Columns carry a slug ``name`` (so a
    cell slot can target them) and the display label as ``field``; every row carries label ->
    formatted value, the ``design_id`` row key, and a ``<name>_clr`` colour for each toned column.
    ``toned`` lists ``(name, tone)`` so the caller can attach the chip / colour slots."""
    spec = _visible_spec(shown)
    palette = _engine_palette(shown)
    columns, toned = [], []
    for i, (label, _col, kind, w, tone, _cat) in enumerate(spec):
        name = f"c{i}"
        columns.append({"name": name, "label": label, "field": label, "sortable": True,
                        "align": "left" if kind in ("str", "bool") else "right",
                        "style": f"width:{w}px", "headerStyle": f"width:{w}px"})
        if tone:
            toned.append((name, tone))
    rows = []
    for rec in shown.to_dict("records"):
        row = {"design_id": rec.get("design_id")}
        for i, (label, col, kind, _w, tone, _cat) in enumerate(spec):
            row[label] = _cell(rec.get(col), kind)
            if tone:
                row[f"c{i}_clr"] = _tone_color(tone, rec.get(col), palette)
        rows.append(row)
    return columns, rows, toned


def _engine_palette(shown: pd.DataFrame) -> dict:
    """Stable engine -> colour, matching the scatter (sorted engines cycled through one palette)."""
    if "engine" not in shown.columns:
        return {}
    values = sorted(shown["engine"].dropna().astype(str).unique())
    return {v: _ENGINE_CYCLE[i % len(_ENGINE_CYCLE)] for i, v in enumerate(values)}


def _tone_color(tone: str, raw, palette: dict) -> str | None:
    """The cell colour for a toned column: a category hue (chips) or a health hue (margins)."""
    if tone == "chip_engine":
        return palette.get(str(raw), MUTED)
    if tone == "chip_prop":
        return _PROP_COLOR.get(str(raw).strip().lower(), MUTED)
    if tone == "estflag":
        # Estimated (scaled, not measured) reads as a caution; native is unremarkable.
        return AMBER if (raw is True or str(raw).strip().lower() == "true") else None
    if tone == "chip_count":
        n = _safe_int(raw)
        return _COUNT_CYCLE[(n - 1) % len(_COUNT_CYCLE)] if n else MUTED
    if tone == "verdict":
        return GREEN if (raw is True or str(raw).strip().lower() == "true") else RED
    if tone in ("margin_dv", "margin_kg"):
        v = _safe_float(raw)
        if v is None:
            return MUTED
        band = _MARGIN_DV_BAND if tone == "margin_dv" else _MARGIN_KG_BAND
        return GREEN if v >= band else (AMBER if v >= 0 else RED)
    if tone == "short":
        v = _safe_float(raw)
        return RED if (v is not None and v > 0) else None
    return None


def _tone_slot(table, name: str, tone: str) -> None:
    """Attach a body-cell slot that renders ``name`` as a tinted chip (categories) or coloured
    text (verdicts / margins), reading the precomputed ``<name>_clr`` colour off the row."""
    clr = f"props.row.{name}_clr"
    if tone in ("chip_engine", "chip_count", "chip_prop"):
        template = (f'<q-td :props="props">'
                    f'<q-chip dense square :style="`background:${{{clr}}}26;color:${{{clr}}}`" '
                    f'class="q-px-sm text-weight-medium">{{{{ props.value }}}}</q-chip></q-td>')
    else:
        template = (f'<q-td :props="props">'
                    f'<span :style="{clr} ? `color:${{{clr}}}` : null" '
                    f'class="text-weight-medium">{{{{ props.value }}}}</span></q-td>')
    table.add_slot(f"body-cell-{name}", template)


def _cell(value, kind: str):
    if kind == "str":
        return "-" if value is None or (isinstance(value, float) and value != value) else str(value)
    if kind == "bool":
        return "✓" if (value is True or str(value).strip().lower() == "true") else "✗"
    if kind == "estflag":
        # "est" only when the gas is estimated; native rows stay blank (no ✓/✗ noise).
        return "est" if (value is True or str(value).strip().lower() == "true") else ""
    num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(num):
        return None
    if kind == "int":
        return int(num)
    if kind == "kw":
        return round(float(num) / 1000.0, 2)
    if kind == "g":
        # A swept model param: show it cleanly - integers bare, fractions to 3 dp without
        # trailing-zero noise (20 stays "20", 0.9 stays "0.9", 1.2 stays "1.2").
        f = round(float(num), 3)
        return int(f) if f == int(f) else f
    return round(float(num), {"f0": 0, "f1": 1, "f2": 2, "f3": 3}[kind])


# ---- Trends view: the design-space analysis charts ----

def _trends_view(data: dict) -> None:
    shown = data["shown"]
    if len(shown) < 8:
        _empty("show_chart", "analysis charts appear once ≥ 8 designs are evaluated")
        return
    try:
        specs = {k: {"thrust_mN": e.thrust_mN, "power_W": e.power_W, "isp_s": e.isp_s}
                 for k, e in load_engines().items()}
        charts = figures.vehicle_search_analysis(shown, specs)
    except Exception as exc:  # noqa: BLE001  (charts are a bonus; say so rather than vanish)
        _empty("show_chart", f"analysis charts unavailable for this session ({exc})")
        return
    if not charts:
        _empty("show_chart", "not enough comparable designs for the trend charts yet")
        return
    with ui.element("div").style("flex:1 1 0;min-height:0;width:100%;overflow:auto"):
        for fig, takeaway in charts:
            _plot(fig, height=340)
            ui.label(takeaway).style(f"color:{MUTED};font-size:.74rem;margin:0 .5rem .6rem")


def _plot(fig, *, height: int | None = None):
    fig.update_layout(title_text="", paper_bgcolor=PANEL, plot_bgcolor=PANEL, autosize=True,
                      height=height, font=dict(color=MUTED, size=11))
    fig.update_scenes(xaxis_backgroundcolor=PANEL, yaxis_backgroundcolor=PANEL,
                      zaxis_backgroundcolor=PANEL)
    data = fig.to_plotly_json()
    data["config"] = _PLOTLY_CONFIG
    el = ui.plotly(data).classes("w-full")
    el.style("min-width:0" + (f";height:{height}px" if height else ";flex:1 1 0;min-height:0"))
    return el


# ======================================================================================
# right rail: live filters + the controls for whichever view is active
# ======================================================================================

@ui.refreshable
def _rail() -> None:
    session = _current()
    rows = _load(session)["rows"] if session is not None else pd.DataFrame()
    _filter_controls(rows)
    if _view == "Plot":
        ui.separator().style(f"background:{BORDER}")
        _scatter_controls(rows)
    elif _view == "Table":
        ui.separator().style(f"background:{BORDER}")
        _table_controls(rows)


def _filter_controls(rows: pd.DataFrame) -> None:
    section("Show only", "filter_alt")
    # Buildable + flyable as twin toggles (both default on -> the viable designs); turning one off
    # relaxes that requirement so the near-misses on that axis come back into view.
    ui.switch("Buildable", value=_filters["buildable"]).props("dense").on_value_change(
        lambda e: _set_filter("buildable", e.value))
    ui.switch("Flyable (closes)", value=_filters["flyable"]).props("dense").on_value_change(
        lambda e: _set_filter("flyable", e.value))
    engines = (sorted(rows["engine"].dropna().astype(str).unique())
               if "engine" in rows.columns and len(rows) else [])
    if engines:
        ui.select(engines, value=_filters["engines"], label="Engines", multiple=True).props(
            "dense outlined").classes("w-full").on_value_change(
            lambda e: _set_filter("engines", e.value or []))
    counts = (sorted(int(c) for c in pd.to_numeric(rows.get("n_engines"), errors="coerce")
                     .dropna().unique()) if len(rows) and "n_engines" in rows.columns else [])
    if counts:
        ui.select(counts, value=_filters["counts"], label="Engine count", multiple=True).props(
            "dense outlined").classes("w-full").on_value_change(
            lambda e: _set_filter("counts", e.value or []))

    # Numeric range filters mirror the VISIBLE columns: the always-on base, plus a row for each
    # column a column-group toggle has revealed (swept model params included), so enabling a group
    # brings its filters along with its columns.
    cols = _numeric_columns(_visible_spec(rows))
    if cols:
        ui.label("Column ranges").style(
            f"color:{MUTED};font-size:.66rem;margin-top:.4rem")
    for label, col, _kind in cols:
        _num_filter_row(label, col)


def _num_filter_row(label: str, col: str) -> None:
    """A column's min/max range filter: the label and two compact number fields. An empty field
    means that bound is unset."""
    f = _num_filters.setdefault(col, {"min": None, "max": None})
    with ui.row().classes("items-center w-full no-wrap gap-1").style("margin-top:.1rem"):
        ui.label(label).classes("truncate").style(
            f"color:{MUTED};font-size:.66rem;flex:1 1 0;min-width:0")
        ui.number(placeholder="min", value=f["min"]).props("dense outlined").style(
            "width:58px;flex:0 0 auto").on_value_change(lambda e: _set_num(col, "min", e.value))
        ui.number(placeholder="max", value=f["max"]).props("dense outlined").style(
            "width:58px;flex:0 0 auto").on_value_change(lambda e: _set_num(col, "max", e.value))


def _scatter_controls(rows: pd.DataFrame) -> None:
    section("Scatter axes", "scatter_plot")
    available = _scatter_columns(rows)
    if not available:
        ui.label("no plottable columns yet").style(f"color:{MUTED};font-size:.72rem")
        return
    labels = list(available)
    color_opts = ["Engine config", "Propellant", *labels]
    _axis_select("X axis", labels, _scatter["x"] if _scatter["x"] in labels else labels[0], "x")
    _axis_select("Y axis", labels, _scatter["y"] if _scatter["y"] in labels else labels[-1], "y")
    _axis_select("Colour by", color_opts,
                 _scatter["c"] if _scatter["c"] in color_opts else "Engine config", "c")


def _axis_select(label: str, options: list[str], value: str, key: str) -> None:
    ui.select(options, value=value, label=label,
              on_change=lambda e, key=key: _set_scatter(key, e.value)).props(
        "dense outlined").classes("w-full")


def _table_controls(rows: pd.DataFrame) -> None:
    section("Columns", "grid_on")
    ui.label("Add column groups:").style(f"color:{MUTED};font-size:.66rem")
    has_params = any(str(c).startswith("param_") for c in rows.columns)
    for key, label in _CATEGORIES:
        if key == "params" and not has_params:
            continue                      # no model-setting sweep in this session
        ui.switch(label, value=key in _cols_on,
                  on_change=lambda e, k=key: _toggle_category(k, e.value)).props("dense")


def _apply_filters(rows: pd.DataFrame) -> pd.DataFrame:
    if not len(rows):
        return rows

    def num(col):
        return pd.to_numeric(rows.get(col, pd.Series(index=rows.index)), errors="coerce")

    mask = pd.Series(True, index=rows.index)
    f = _filters
    if f["engines"] and "engine" in rows.columns:
        mask &= rows["engine"].astype(str).isin([str(e) for e in f["engines"]])
    if f["counts"]:
        mask &= num("n_engines").isin([int(c) for c in f["counts"]])
    if f["buildable"]:
        mask &= vs.flag(rows, "buildable")
    if f["flyable"]:
        mask &= vs.flag(rows, "success")
    for col, mm in _num_filters.items():
        lo, hi = mm.get("min"), mm.get("max")
        if lo is None and hi is None:
            continue
        series = num(col)
        if col == "bol_power_W":          # array power is shown and filtered in kilowatts
            series = series / 1000.0
        if lo is not None:
            mask &= series >= lo
        if hi is not None:
            mask &= series <= hi
    return rows[mask]


# ======================================================================================
# bottom dock: the sweep runner (new-sweep form + recent sweeps)
# ======================================================================================

def _runner() -> None:
    """The bottom dock: the recent-sweeps table + live progress, full width. The new-sweep
    form lives in a modal (the 'New sweep' button, top-right of the canvas) so the dock isn't
    cramped; the sessions panel refreshes on selection and while a sweep streams."""
    with ui.element("div").style(
            f"flex:0 0 260px;height:260px;width:100%;min-width:0;display:flex;"
            f"flex-direction:column;background:{PANEL};border:1px solid {BORDER};border-radius:8px;"
            f"padding:10px;gap:.3rem;box-sizing:border-box"):
        _sessions_panel()


def _open_sweep_dialog() -> None:
    """Open the new-sweep form as a modal. Built fresh each open, reading its values back out of
    the persistent form state (``_form`` + ``_pairings``); the form closes itself once a sweep is
    submitted.

    ``persistent`` means only the form's own close button dismisses it - a click on the backdrop
    does not - so a stray click while filling in a sweep cannot shut the form. The state would
    survive that anyway; this keeps the user from losing their place as well.
    """
    with ui.dialog().props("persistent") as dlg, ui.card().style(
            f"background:{PANEL};border:1px solid {BORDER};min-width:540px;max-width:620px;"
            f"gap:.3rem"):
        _launcher(on_started=dlg.close)
    dlg.open()


def _launcher(on_started: Callable[[], None] | None = None) -> None:
    """The new-sweep form. Every field maps to a flag of the design-search CLI; the
    sweep target is always the focused asteroid and the flight profile is the study's mission.
    ``on_started`` (when given) is called after a sweep is submitted - the modal uses it to
    close itself."""
    with ui.row().classes("items-center gap-2 w-full no-wrap"):
        ui.icon("rocket_launch").style(f"color:{ACCENT}").classes("text-sm")
        ui.label("New sweep").style(f"color:{TEXT};font-weight:600;font-size:.82rem")
        ui.space()
        if on_started is not None:
            ui.button(icon="close", on_click=on_started).props("flat dense round").style(
                f"color:{MUTED}")

    target = S.focus
    pdes = str(target.get("pdes") or "") if target else ""
    if target and pdes:
        bits = [str(target.get("full_name") or target.get("pdes"))]
        if target.get("i") is not None and target.get("i") == target.get("i"):
            bits.append(f"i = {float(target['i']):.1f}°")
        if target.get("a") is not None and target.get("a") == target.get("a"):
            bits.append(f"a = {float(target['a']):.2f} AU")
        ui.label("→ " + "  ·  ".join(bits)).style(f"color:{MUTED};font-size:.72rem")
    else:
        ui.label("Pick a target in Find targets first.").style(f"color:{AMBER};font-size:.72rem")

    engine_keys = sorted(load_engines())
    _seed_form()
    _seed_pairings(engine_keys)

    dry_range, dry_step = _range_with_step(
        "Dry mass (kg)", 50, 1000, _form["dry_min"], _form["dry_max"], _form["dry_step"],
        "dry_min", "dry_max", "dry_step", on_change=_plan_line.refresh)
    prop_range, prop_step = _range_with_step(
        "Propellant (kg)", 0, 2000, _form["prop_min"], _form["prop_max"], _form["prop_step"],
        "prop_min", "prop_max", "prop_step", on_change=_plan_line.refresh)

    max_wet = ui.number("Max wet (kg)", value=_form["max_wet"], min=100, step=50,
                        on_change=lambda e: _set_form("max_wet", e.value)).props(
        "dense outlined").classes("w-full")

    thrust = labeled_slider("Thrust limit", 0, 100, _form["thrust_pct"], 5, "{:.0f}", " %",
                            on_change=lambda e: _set_form("thrust_pct", e.value))

    propellant_keys = list_propellants() or ["xenon"]
    seeded_gases = [g for g in _form["gases"] if g in propellant_keys]
    gases = ui.select(
        propellant_keys, multiple=True, label="Propellants",
        value=seeded_gases or propellant_keys[:1],
        on_change=lambda e: (_set_form("gases", list(e.value or [])),
                             _plan_line.refresh())).props(
        "dense outlined use-chips").classes("w-full")
    gases.tooltip("A gas the engine has no data for is estimated from its xenon thrust and "
                  "Isp, flagged 'est' in the table.")

    # Optional model-setting sweep: one build-model knob varied over a list of values, evaluated
    # against every vehicle combo. This is how a multivariate trade like "how much solar-array
    # margin is ideal" gets answered - more margin grows the array (heavier, less payload) yet
    # powers a faster escape, so the sweep plots the knob against mission success. Duty cycle is
    # omitted here (the Thrust-limit slider already sets it).
    model_keys = {k: spec for k, spec in buildability.SWEEPABLE.items()
                  if spec[2] == "model"}
    bus = buildability.load_bus_model(name=S.build_model)
    ui.label("Sweep a model setting").style(
        f"color:{MUTED};font-size:.7rem;margin-top:.3rem")
    with ui.row().classes("items-end no-wrap w-full gap-2"):
        sweep_opts = {"": "- none -", **{k: (f"{spec[0]} ({spec[1]})" if spec[1]
                                             else spec[0]) for k, spec in model_keys.items()}}
        seeded_key = _form["sweep_key"] if _form["sweep_key"] in sweep_opts else ""
        sweep_key = ui.select(sweep_opts, value=seeded_key, label="Setting").props(
            "dense outlined").style("flex:1 1 0;min-width:0")
        sweep_vals = ui.input("Values", value=_form["sweep_vals"],
                              placeholder="e.g. 10, 20, 30",
                              on_change=lambda e: _set_form("sweep_vals", e.value)).props(
            "dense outlined").style("flex:1 1 0;min-width:0")
    sweep_hint = ui.label("").style(f"color:{MUTED};font-size:.68rem")

    def _on_sweep_key() -> None:
        k = sweep_key.value
        _set_form("sweep_key", k or "")
        if k and k in model_keys:
            label, unit, _ = model_keys[k]
            cur = getattr(bus, k, None)
            sweep_hint.set_text(f"Configured {label.lower()}: {cur} {unit}".rstrip())
        else:
            sweep_hint.set_text("")
    sweep_key.on_value_change(lambda e: _on_sweep_key())
    _on_sweep_key()                     # a reopened form carries its axis, so show its hint now

    ui.label("Engine pairings").style(
        f"color:{MUTED};font-size:.7rem;margin-top:.3rem")
    _pairings_ui(engine_keys)
    _plan_line()

    mission_key = S.mission_key
    if not mission_key:
        ui.label("Save the study (top-bar Save) so the sweep can load its mission.").style(
            f"color:{RED};font-size:.7rem")

    def _start() -> None:
        pairs = ["--pairs", *[f"{p['engine']}:{int(p['cmin'])}-{int(p['cmax'])}"
                              for p in _pairings]]
        dr, pr = dry_range.value, prop_range.value
        # Thrust limit is shown as 0-100 % but --duty takes a fraction.
        duty = max(0.01, min(1.0, float(thrust.value or 90) / 100.0))
        gas_vals = list(gases.value or ["xenon"]) or ["xenon"]
        # An optional model-setting sweep axis (e.g. solar-array margin over a few values).
        sweep_args: list[str] = []
        sk = sweep_key.value
        raw = (sweep_vals.value or "").strip()
        if sk and raw:
            try:
                vals = [float(v) for v in raw.split(",") if v.strip() != ""]
            except ValueError:
                ui.notify(f"Sweep values must be numbers, got '{raw}'", type="negative")
                return
            if not vals:
                ui.notify("Enter at least one sweep value", type="negative")
                return
            sweep_args = ["--sweep-param",
                          f"{sk}={','.join(str(v) for v in vals)}"]
        # Resolve the sweep target at the study's screening cutoff, so a faint focus target (say a
        # minimoon at H~29.5) the screen found is findable here too, since the lookup default
        # (H<=25) would drop it and the sweep would die before its first combo.
        args = ["--target", pdes, "--h-max", str(S.screening.h_max),
                "--mission", mission_key, *pairs,
                *(["--study", S.project] if S.project else []),
                "--prop-min", str(pr["min"]), "--prop-max", str(pr["max"]),
                "--prop-step", str(prop_step.value or 100),
                "--dry-min", str(dr["min"]), "--dry-max", str(dr["max"]),
                "--dry-step", str(dry_step.value or 10), "--max-wet", str(max_wet.value or 750),
                "--duty", str(duty), "--vinf", "0.0", "--propellants", *gas_vals,
                # The same solver terms the app's own cruise and escape use, so a sweep point is
                # the trajectory the app would fly for that vehicle rather than a stand-in.
                "--nseg", str(int(S.solve_nseg)),
                "--cruise-duty", str(float(S.solve_duty_pct) / 100.0),
                "--radiation-model", str(S.launch_radiation_model),
                "--coverglass-um", str(float(S.launch_coverglass_um)),
                "--coverglass-density", str(float(S.launch_coverglass_density)),
                *_stated_array_args(), *sweep_args]
        global _session, _was_running
        _session = vs.submit_search(args)
        _was_running = True
        ui.notify("Sweep started")
        if on_started is not None:
            on_started()
        _refresh_all()

    ready = bool(target and pdes and mission_key and _pairings and gases.value)
    ui.button("Run sweep", icon="play_arrow", on_click=_start).props(
        "unelevated no-caps").classes("w-full").style("margin-top:.3rem").set_enabled(ready)


# The factory new-sweep form: a mid-sized bus over a few engine counts, the starting point when
# there is no previous sweep to copy. Kept in one literal so _seed_form and reset() agree.
_FORM_DEFAULTS: dict = {
    "dry_min": 150.0, "dry_max": 350.0, "dry_step": 10.0,
    "prop_min": 200.0, "prop_max": 400.0, "prop_step": 25.0,
    "max_wet": 750.0, "thrust_pct": 90.0,
    "gases": ["xenon"], "sweep_key": "", "sweep_vals": "",
}


def _seed_form() -> None:
    """Fill the form state once per project, preferring the last sweep's own settings.

    Reopening the form after a run should show what was just run - that is the thing a user
    iterates from - so the newest session's ``session.json`` seeds it when one exists. The factory
    defaults are the fallback for a project with no sweeps yet. Pairings are seeded separately
    (:func:`_seed_pairings`), which needs the engine list.
    """
    if _form:
        return
    _form.update({k: (list(v) if isinstance(v, list) else v)
                  for k, v in _FORM_DEFAULTS.items()})
    sessions = _sessions()
    meta = (sessions[0].get("meta") or {}) if sessions else {}
    dry, prop = meta.get("dry_range_kg"), meta.get("prop_range_kg")
    if dry and len(dry) == 3:
        _form["dry_min"], _form["dry_max"], _form["dry_step"] = (float(x) for x in dry)
    if prop and len(prop) == 3:
        _form["prop_min"], _form["prop_max"], _form["prop_step"] = (float(x) for x in prop)
    if meta.get("max_wet_kg"):
        _form["max_wet"] = float(meta["max_wet_kg"])
    if meta.get("duty"):
        _form["thrust_pct"] = float(meta["duty"]) * 100.0
    if meta.get("propellants"):
        _form["gases"] = list(meta["propellants"])
    # A swept model axis is a single key -> values pair in the form; a session that swept more than
    # one cannot be represented, so it seeds nothing rather than half of itself.
    axes = meta.get("sweep_axes") or {}
    if len(axes) == 1:
        key, values = next(iter(axes.items()))
        _form["sweep_key"] = key
        _form["sweep_vals"] = ", ".join(f"{v:g}" for v in values)


def _set_form(key: str, value) -> None:
    _form[key] = value


def _seed_pairings(engine_keys: list[str]) -> None:
    """Ensure at least one engine pairing exists, seeded from the project vehicle's engine -
    pairings are the only path (the form sweeps every pairing in the list)."""
    if _pairings:
        return
    try:
        engine = S.vehicle.engines[0].type
    except Exception:  # noqa: BLE001
        engine = ""
    if engine not in engine_keys:
        engine = engine_keys[0] if engine_keys else ""
    _pairings.append({"engine": engine, "cmin": 3, "cmax": 6})


def _range_with_step(label, lo, hi, vmin, vmax, step, min_key, max_key, step_key,
                     on_change=None):
    """A dual-handle range slider beside a small step input, laid out so the step stays inside
    the form column: the range goes in a ``flex:1;min-width:0`` cell (which clamps its own
    full-width children) and the fixed-width step sits to its right.

    The three ``*_key`` names are where each handle writes back into the persistent form state,
    so the range survives the modal being dismissed. ``on_change`` runs after every write, which
    is how the form keeps its combo count in step with the axes."""
    def _touched(e, key=None) -> None:
        if key is None:
            _set_form(min_key, e.value["min"])
            _set_form(max_key, e.value["max"])
        else:
            _set_form(key, e.value)
        if on_change is not None:
            on_change()

    with ui.row().classes("items-end no-wrap w-full gap-2"):
        with ui.element("div").style("flex:1 1 0;min-width:0"):
            rng = labeled_range(label, lo, hi, vmin, vmax, step, "{:.0f}", " kg",
                                on_change=_touched)
        step_input = ui.number("step", value=step, min=1,
                               on_change=lambda e: _touched(e, step_key)).props(
            "dense outlined").style("width:70px;flex:0 0 auto")
    return rng, step_input


@ui.refreshable
def _plan_line() -> None:
    """How many trajectory solves the form as filled in comes to, so the cost of widening an axis
    is on screen before the sweep starts rather than discovered from the progress bar."""
    dry = _axis_steps(_form["dry_min"], _form["dry_max"], _form["dry_step"])
    prop = _axis_steps(_form["prop_min"], _form["prop_max"], _form["prop_step"])
    counts = sum(max(0, int(p["cmax"]) - int(p["cmin"]) + 1) for p in _pairings)
    gases = max(1, len(_form["gases"]))
    params = max(1, len([v for v in (_form["sweep_vals"] or "").split(",") if v.strip()])
                 if _form["sweep_key"] else 1)
    axes = (f"{dry} dry × {prop} propellant × {counts} engine "
            f"{'setups' if counts != 1 else 'setup'} × {gases} "
            f"{'gases' if gases != 1 else 'gas'}"
            + (f" × {params} setting values" if params > 1 else ""))
    total = dry * prop * counts * gases * params
    ui.label(f"{total:,} trajectory solves: {axes}, minus anything over the wet limit.").style(
        f"color:{MUTED};font-size:.68rem")


def _axis_steps(lo: float, hi: float, step: float) -> int:
    """How many points an inclusive lo..hi axis of this step has, matching the grid the search
    itself builds (``numpy.arange`` with a half-step of slack on the ceiling)."""
    step = abs(float(step or 0))
    if step <= 0:
        return 1
    return max(1, int((float(hi) - float(lo)) / step + 0.5) + 1)


@ui.refreshable
def _pairings_ui(engine_keys: list[str]) -> None:
    for i, p in enumerate(_pairings):
        with ui.row().classes("items-center gap-2 w-full no-wrap"):
            ui.select(engine_keys, value=p["engine"],
                      on_change=lambda e, i=i: _set_pairing(i, "engine", e.value)).props(
                "dense outlined").classes("flex-grow")
            ui.number("×min", value=p["cmin"], min=1,
                      on_change=lambda e, i=i: _set_pairing(i, "cmin", e.value)).props(
                "dense outlined").style("width:74px")
            ui.number("×max", value=p["cmax"], min=1,
                      on_change=lambda e, i=i: _set_pairing(i, "cmax", e.value)).props(
                "dense outlined").style("width:74px")
            ui.button(icon="close", on_click=lambda i=i: _remove_pairing(i)).props(
                "flat dense round").style(f"color:{MUTED}").set_enabled(len(_pairings) > 1)
    ui.button("Add pairing", icon="add",
              on_click=lambda: _add_pairing(engine_keys)).props("outline dense no-caps").style(
        f"color:{ACCENT}")


@ui.refreshable
def _sessions_panel() -> None:
    sessions = _sessions()
    selected = _current(sessions)
    hidden = 0 if _all_projects else _n_other_projects()
    with ui.row().classes("items-center w-full no-wrap gap-2").style("flex:0 0 auto"):
        ui.label("Recent sweeps").style(f"color:{TEXT};font-weight:600;font-size:.82rem")
        if hidden or _all_projects:
            ui.switch("all projects", value=_all_projects,
                      on_change=lambda e: _set_all_projects(bool(e.value))).props(
                "dense size=xs").style(f"color:{MUTED};font-size:.7rem").tooltip(
                f"{hidden} sweep{'s' if hidden != 1 else ''} from other projects hidden"
                if hidden else "Showing every project's sweeps")
        ui.space()
        if selected is not None:
            ui.label(vs.session_label(selected)).classes("truncate").style(
                f"color:{MUTED};font-size:.72rem;max-width:340px")
    if selected is not None and selected["state"] == "running":
        _progress(selected)
    elif selected is not None and selected["state"] == "error":
        ui.label(f"✕ {selected['message'] or 'stopped with an error'}").style(
            f"color:{RED};font-size:.72rem")
    if not sessions:
        ui.label("No sweeps for this project yet." + (
            f" {hidden} from other projects are hidden; flip 'all projects' to see them."
            if hidden else "")).style(f"color:{MUTED};font-size:.72rem")
        return
    # Each row carries only the settings that differ between these sweeps; everything they agree on
    # is stated once underneath, so the row spends its width on what identifies a sweep.
    setup = vs.session_setup(sessions)
    # Fixed widths + wrap-cells (below) keep every row inside the panel: the wide Engines and Setup
    # columns wrap instead of forcing a horizontal scrollbar.
    cols = [{"name": "when", "label": "When", "field": "when", "align": "left",
             "sortable": True, "style": "width:104px;white-space:nowrap"},
            {"name": "state", "label": "State", "field": "state", "align": "left",
             "style": "width:74px"},
            {"name": "target", "label": "Target", "field": "target", "align": "left",
             "style": "width:96px"},
            {"name": "engine", "label": "Engines", "field": "engine", "align": "left",
             "style": "width:190px"},
            {"name": "setup", "label": "Differs by", "field": "setup", "align": "left"},
            {"name": "n", "label": "Designs", "field": "n", "align": "right",
             "sortable": True, "style": "width:64px"}]
    rows = []
    for s in sessions:
        status = vs.read_json(s["dir"] / "status.json") or {}
        rows.append({"name": s["name"], "when": s["when"],
                     "state": vs.state_tag(s),
                     "state_clr": _STATE_COLOR.get(s["state"], MUTED),
                     "target": s.get("target") or "-", "engine": _engine_short(s.get("meta") or {}),
                     "setup": setup["rows"].get(s["name"]) or "—",
                     "n": status.get("n_evaluated")})
    with ui.element("div").style("flex:1 1 0;min-height:0;width:100%;overflow:auto"):
        table = selectable_table(cols, rows, "name", selected["name"] if selected else None,
                                 lambda r: _select_session(r["name"]), rows_per_page=8)
        table.props("wrap-cells").classes("pf-fixed")
        # The run state reads at a glance as a tinted chip (done green, running amber, error red).
        table.add_slot("body-cell-state",
                       '<q-td :props="props"><q-chip dense square '
                       ':style="`background:${props.row.state_clr}26;color:${props.row.state_clr}`" '
                       'class="q-px-sm">{{ props.value }}</q-chip></q-td>')
    if setup["common"]:
        prefix = "This sweep" if len(sessions) == 1 else "Every sweep above"
        ui.label(f"{prefix}:  {setup['common']}").classes("truncate").style(
            f"color:{MUTED};font-size:.68rem;flex:0 0 auto")


def _progress(session: dict) -> None:
    status = vs.read_json(session["dir"] / "status.json") or {}
    n_eval = int(status.get("n_evaluated") or 0)
    n_planned = status.get("n_planned")
    if n_planned:
        frac, text = min(n_eval / float(n_planned), 1.0), f"{n_eval} of {int(n_planned)} combos"
    else:
        frac, text = None, f"{n_eval} vehicles evaluated"
    if frac is not None:
        ui.linear_progress(value=frac, show_value=False).props("rounded").style("margin:.2rem 0")
    now = vs.in_flight(session["dir"])
    with ui.row().classes("items-center justify-between w-full no-wrap"):
        ui.label(f"{text}" + (f" · now {now}" if now else "")).style(
            f"color:{MUTED};font-size:.72rem")
        if vs.stop_requested(session["dir"]):
            ui.label("⏹ stopping…").style(f"color:{AMBER};font-size:.7rem")
        else:
            ui.button("Stop", on_click=lambda: _stop(session)).props("flat dense no-caps").style(
                f"color:{MUTED};font-size:.7rem")


def _engine_short(meta: dict) -> str:
    pairs = meta.get("pairs")
    if pairs:
        by_engine: dict = {}
        for key, n in pairs:
            by_engine.setdefault(key, []).append(int(n))
        return " + ".join(f"{k} ×{'/'.join(str(n) for n in sorted(ns))}"
                          for k, ns in by_engine.items())
    engine = meta.get("engine") or "-"
    if engine != "-" and meta.get("engines"):
        engine = f"{engine} ×{'/'.join(str(int(n)) for n in meta['engines'])}"
    return engine


# ======================================================================================
# behaviour
# ======================================================================================

def _refresh_all() -> None:
    _canvas.refresh()
    _rail.refresh()
    _sessions_panel.refresh()


def _set_view(value: str) -> None:
    global _view
    _view = value
    _canvas.refresh()
    _rail.refresh()         # the rail carries this view's controls (scatter axes / columns)


def _set_scatter(key: str, value: str) -> None:
    _scatter[key] = value
    _canvas.refresh()


def _toggle_category(key: str, on) -> None:
    if on:
        _cols_on.add(key)
    else:
        _cols_on.discard(key)
    _canvas.refresh()
    _rail.refresh()       # the rail's range filters mirror the now-visible columns


def _set_filter(key: str, value) -> None:
    _filters[key] = value
    _canvas.refresh()       # the stat cards live inside _canvas, so this re-counts them too


def _set_num(col: str, bound: str, value) -> None:
    _num_filters.setdefault(col, {"min": None, "max": None})[bound] = value
    _canvas.refresh()


def _select_session(name: str) -> None:
    global _session, _sel_design, _was_running
    if name == _session:
        return
    _session = name
    _sel_design = None
    S.selected = None
    _was_running = any(s["name"] == name and s["state"] == "running" for s in _sessions())
    _refresh_all()
    _hud.refresh()


def _pick_design_by_id(design_id) -> None:
    """Resolve a clicked design from the CURRENT session data and select it. Loading fresh
    (rather than trusting a row captured at render time) keeps selection correct after the
    live poll swaps the grid's rows in place."""
    if design_id is None:
        return
    session = _current()
    if session is None:
        return
    _pick_design(session, _load(session)["shown"], {"design_id": design_id})


def _pick_design(session: dict, shown: pd.DataFrame, row: dict) -> None:
    global _sel_design
    _sel_design = row.get("design_id")
    match = shown[shown["design_id"].astype(str) == str(_sel_design)]
    if not len(match):
        return
    rec = match.iloc[0].to_dict()
    engine, n = rec.get("engine"), _safe_int(rec.get("n_engines"))
    dry, prop = _safe_float(rec.get("dry_kg")), _safe_float(rec.get("prop_kg"))
    slug = rec.get("slug")
    # The memorable name leads the card; the numeric spec becomes the first detail line so the
    # config is still readable at a glance.
    lines = [f"{dry:.0f} kg dry · {prop:.0f} fuel · {n}×{engine}"]
    if _safe_float(rec.get("prop_margin_kg")) is not None:
        lines.append(f"fuel margin {float(rec['prop_margin_kg']):+.0f} kg")
    if _safe_float(rec.get("payload_capacity_kg")) is not None:
        lines.append(f"payload+margin {float(rec['payload_capacity_kg']):.0f} kg · "
                     f"builds {'✓' if vs.flag(match, 'buildable').iloc[0] else '✗'}")
    # The round-trip story, when this session flew a return leg (spent shown with a minus sign).
    if _safe_float(rec.get("return_prop_kg")) is not None:
        lines.append(f"return −{float(rec['return_prop_kg']):.0f} kg · "
                     f"total {float(rec['total_prop_kg']):.0f} kg prop"
                     if _safe_float(rec.get("total_prop_kg")) is not None
                     else f"return −{float(rec['return_prop_kg']):.0f} kg")
    if _safe_float(rec.get("delivered_payload_kg")) is not None:
        lines.append(f"delivers {float(rec['delivered_payload_kg']):.0f} kg home")
    S.selected = {
        "icon": "satellite_alt",
        "title": slug or f"{dry:.0f} kg dry · {prop:.0f} fuel · {n}×{engine}",
        "lines": lines,
        "action": "Plan trajectory", "action_icon": "rocket",
        "on_action": lambda rec=rec, session=session: _open_in_flight(rec, session)}
    _hud.refresh()


def _vehicle_from_trade(rec: dict, meta: dict) -> Vehicle:
    """Build the working Vehicle from a swept design row + its session meta.

    Carries the design's dry/propellant masses, engine type and count, working gas, and --
    crucially, the array power it was sized to when new. The sweep sized ``bol_power_W`` to carry
    the engine load through the belt degradation this design saw; without it the planned vehicle
    defaults to 0 W, which the escape spiral reads as "power-rich, model no degradation", so the
    array reads as 0 and belt-degradation throttling silently vanishes from the trajectory and the
    report. The working gas is carried so the trajectory solve uses the same (possibly scaled)
    thrust + Isp the sweep evaluated, not the engine's xenon numbers (a row with no gas recorded ->
    the engine's native gas). Raises on an unknown engine key, which the caller surfaces.
    """
    engine = rec.get("engine") or meta.get("engine")
    if not engine:
        raise ValueError("this evaluation records no engine key, and its session metadata carries "
                         "none either, so the vehicle cannot be rebuilt")
    engine = str(engine)
    dry = _safe_float(rec.get("dry_kg")) or 0.0
    prop = _safe_float(rec.get("prop_kg")) or 0.0
    n = _safe_int(rec.get("n_engines")) or 1
    gas = rec.get("propellant_key") or None
    gas_tag = f" {gas}" if gas and gas != "xenon" else ""
    # The memorable sweep name becomes the vehicle's name (and, on Save, its filename slug) so a
    # good config stays referenceable; its numeric spec is derivable from the masses/engines and
    # surfaces in the top-bar detail. A row with no slug falls back to the numeric name.
    name = str(rec.get("slug") or f"trade {dry:.0f}/{prop:.0f} ×{n}{gas_tag}")
    bol_power_W = _safe_float(rec.get("bol_power_W")) or 0.0
    # Size the spiral's drag/SRP cross-section from the array too: for a SEP bus the solar array
    # dominates that area, so a power-sized design implies a drag area. Without this the drag field
    # would also read 0 in the editor and report. 0 power -> 0 area (unmodeled).
    area_m2 = buildability.array_area_m2(bol_power_W)
    # Carry the array margin the sweep sized this design at, so the vehicle page's array
    # derivation reproduces its BOL power. Present only when the sweep varied margin as a
    # --sweep-param (``param_power_margin_pct``); otherwise None = the build model's default.
    margin = _safe_float(rec.get("param_power_margin_pct"))
    return Vehicle(name=name, dry_mass=dry,
                   fuel_mass=prop, unusable_prop=0.0, propellant=gas,
                   solar_power_W=bol_power_W, area_m2=area_m2, array_margin_pct=margin,
                   engines=[EngineMount(type=engine, count=n)])


def _open_in_flight(rec: dict, session: dict) -> None:
    """Hand a swept design to Plan trajectory: build the working vehicle from its dry/prop/
    engine (and its sized array), ensure the session's target is focused, switch workspaces."""
    try:
        S.vehicle = _vehicle_from_trade(rec, session.get("meta") or {})
    except Exception as exc:  # noqa: BLE001  (an unknown engine should not crash the handoff)
        ui.notify(f"Could not build that vehicle: {exc}", type="negative")
        return
    # A swept design is a new vehicle, not an edit of the project's: drop the library key so a
    # later Save writes it under a fresh slug rather than overwriting the study's vehicle.
    S.vehicle_key = None
    state.mark_dirty()
    S.workspace = "flight"
    S.selected = None
    from ui import topbar
    topbar.header.refresh()
    topbar.main_body.refresh()


def _stated_array_args() -> list[str]:
    """The study vehicle's stated array, held on every swept design as the project page holds
    it. Nothing when the app sizes the array."""
    stated = state.stated_array()
    if not stated:
        return []
    return ["--array-W", str(stated[0]), *(["--array-m2", str(stated[1])] if stated[1] else [])]


def _clear_selection() -> None:
    S.selected = None
    _hud.refresh()


def _set_pairing(i: int, key: str, value) -> None:
    _pairings[i][key] = value
    _plan_line.refresh()


def _add_pairing(engine_keys: list[str]) -> None:
    _pairings.append({"engine": engine_keys[0] if engine_keys else "", "cmin": 1, "cmax": 2})
    _pairings_ui.refresh()
    _plan_line.refresh()


def _remove_pairing(i: int) -> None:
    _pairings.pop(i)
    _pairings_ui.refresh()
    _plan_line.refresh()


def _stop(session: dict) -> None:
    vs.request_stop(session["dir"])
    ui.notify("Stop requested")
    _sessions_panel.refresh()


def _safe_float(value):
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(v) else float(v)


def _safe_int(value):
    v = _safe_float(value)
    return None if v is None else int(v)


def _set_all_projects(value: bool) -> None:
    global _all_projects, _session
    _all_projects = value
    _session = None                     # the selection may not be in the new list
    _refresh_all()


def reset() -> None:
    """Clear the workspace-local selection when a new project opens."""
    global _session, _sel_design, _view, _was_running, _live_table, _last_sig, _all_projects
    _session = _sel_design = None
    _all_projects = False
    _view = "Table"
    _cols_on.clear()
    _was_running = False
    _live_table = None
    _last_sig = None
    _pairings.clear()
    # The form belongs to the project whose sweeps seeded it, so it is cleared with the rest; the
    # next open re-seeds from the new project's own sweep history.
    _form.clear()
    S.selected = None


# ======================================================================================
# page timer hook
# ======================================================================================

def _grow_live_table() -> bool:
    """Stream new evaluations into the existing grid: swap its rows and update in place,
    leaving the component (and so the user's sort + pagination, which NiceGUI syncs onto it)
    intact. Returns True if it handled the update, False if the canvas must be rebuilt
    instead (not on the Table view, empty grid, or no table yet).

    Recomputes from the current session + filters, so streamed rows respect the live rail.
    The stat cards refresh alongside (they live outside the table); selection is re-pinned by
    design id so the highlight survives the row swap."""
    if _view != "Table" or _live_table is None:
        return False
    session = _current()
    if session is None:
        return False
    data = _load(session)
    _, rows, _ = _grid(data["shown"])
    _live_table.rows = rows
    if _sel_design is not None:
        _live_table.selected = [r for r in rows if r.get("design_id") == _sel_design]
    _live_table.update()
    _stat_cards.refresh(data)
    return True


def _progress_sig(session: dict) -> tuple:
    """A cheap change-signal for the selected session: its name and state plus the mtimes of the
    two files a live sweep advances - the status heartbeat and the evaluations stream. When this
    is unchanged between poll ticks nothing new has streamed, so poll() can skip the rebuild.
    A crashed sweep leaves both files frozen, so its signature never moves and the canvas stops
    thrashing (the state flip to 'stopped', from the PID/staleness guard, still changes it)."""
    d = session["dir"]

    def mtime(name: str) -> float | None:
        try:
            return (d / name).stat().st_mtime
        except OSError:
            return None

    return (session["name"], session["state"], mtime("status.json"), mtime("evaluations.jsonl"))


def poll() -> None:
    """Page-level timer hook: while the selected session runs, stream its progress and grow
    the grid in place, keeping the sort and the page; when a run finishes, land the
    final result the same way, falling back to a full render off the Table view.

    Renders only when the session's change-signature moves, so a session that is idle between
    evaluations - or whose process has died - does not rebuild the canvas on every timer tick."""
    global _was_running, _last_sig
    if S.workspace != "vehicle":
        return
    session = _current()
    if session is None:
        _was_running = False
        _last_sig = None
        return
    if session["state"] == "running":
        _was_running = True
        sig = _progress_sig(session)
        if sig == _last_sig:         # nothing new since last tick -> leave the page untouched
            return
        _last_sig = sig
        if not _grow_live_table():   # Plot/Trends (no sort state to lose) -> full refresh
            _canvas.refresh()
        _sessions_panel.refresh()    # advance the progress bar + session-table counts
    elif _was_running:
        _was_running = False
        _last_sig = None
        if _grow_live_table():       # land the final rows without dropping the user's sort
            _sessions_panel.refresh()
            _rail.refresh()
        else:
            _refresh_all()
