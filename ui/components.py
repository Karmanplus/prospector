"""Reusable layout primitives shared by every workspace.

Generic pieces with no data of their own: the full-screen workspace frame, made of a canvas, a
right-hand rail and an optional dock along the bottom; the table you can pick a row from; the
floating info card; the loading overlay; and a couple of dialogs. Workspaces put these together,
and the real wiring is in each ``workspaces/<name>.py``.
"""
from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

from ui.theme import ACCENT, BORDER, MUTED, PANEL, TEXT


def kg_plain(value) -> str:
    """A bare mass, such as ``"480 kg"``: an amount on board rather than a rate."""
    return f"{value:.0f} kg" if value is not None else "- kg"


def kg_spent(value) -> str:
    """Propellant a leg SPENT, shown as a negative flow, e.g. ``"−127 kg"``."""
    return f"−{value:.0f} kg" if value is not None else "- kg"


def kg_rem(value) -> str:
    """Propellant REMAINING between sections, e.g. ``"210 rem"`` (the ledger level after a leg)."""
    return f"{value:.0f} rem" if value is not None else "- rem"


def kg_added(value) -> str:
    """Mass ADDED (payload picked up), shown as a positive flow, e.g. ``"+150 kg"``."""
    return f"+{value:.0f} kg" if value is not None else "- kg"


def vehicle_drive_rows(rc) -> list[tuple[str, str]]:
    """The resolved vehicle's non-mass performance stats: thrust, array power, Isp, total ΔV.

    The power stat is the solar array's power when new, which is what limits an electric thruster
    short of power, rather than the engines' rated draw. That draw, the sum of thruster power
    is not a useful headline. Its sizing margin rides in the label when the vehicle carries one
    (``array_margin_pct``; None = the build model's default, so no explicit figure to show)."""
    v = rc.vehicle
    array_label = "Array power"
    if v.array_margin_pct is not None:
        array_label = f"Array power · {v.array_margin_pct:+.0f}% margin"
    # 0 W means "not modeled / power-rich" (no array sizing), so a bare "0.0 kW" would mislead.
    array_val = f"{v.solar_power_W * 1e-3:.1f} kW" if v.solar_power_W > 0 else "—"
    return [("Thrust", f"{rc.total_thrust_mN * 1e-3:.2f} N"),
            (array_label, array_val),
            ("Engine Isp", f"{rc.effective_isp:.0f} s"),
            ("Total ΔV", f"{rc.total_dv_capability:.1f} km/s")]


def vehicle_perf_rows(rc) -> list[tuple[str, str]]:
    """The compact performance summary for a resolved vehicle: wet mass plus the drive stats.
    Defined once so every surface that shows the summary (the Project stat strip and the
    make-default confirm dialog) reads identical numbers. ``rc`` is a ``ResolvedConfig``."""
    return [("Wet mass", f"{rc.vehicle.wet_mass:.0f} kg"), *vehicle_drive_rows(rc)]


def _kg(val: float) -> str:
    """A mass in kilograms, whole numbers except under 1 kg. The exception is there for the
    payload-and-margin reserve: a design a few hundred grams short is over budget, and rounding
    it to a bare "-0 kg" would show that shortfall as if it were break-even."""
    return f"{val:.1f} kg" if 0 < abs(val) < 1 else f"{val:.0f} kg"


def vehicle_build_mass_rows(build: dict | None) -> list[tuple[str, str]]:
    """The bus component masses from a ``buildability.assess_engine`` result (kg), or [] when the
    build model couldn't size the bus. Labels are kept short so the rows tile in a compact grid."""
    if not build:
        return []
    out = []
    for label, key in (("Array", "array_kg"), ("Thrusters", "thruster_sys_kg"),
                       ("Tank", "tank_kg"), ("PCDU", "pcdu_kg"), ("Bus", "fixed_bus_kg"),
                       ("Min dry", "min_dry_kg"), ("Payload+mgn", "payload_capacity_kg")):
        val = build.get(key)
        if val is not None:
            out.append((label, _kg(float(val))))
    return out


def section(label: str, icon: str | None = None) -> None:
    """A small accented section heading inside the right rail / panels."""
    with ui.row().classes("items-center gap-2").style("margin:.4rem 0 .2rem 0"):
        if icon:
            ui.icon(icon).style(f"color:{ACCENT}").classes("text-sm")
        ui.label(label).style(f"color:{TEXT};font-weight:600;font-size:.8rem")


def labeled_slider(label: str, lo, hi, val, step=1, fmt="{:.0f}", suffix="", on_change=None):
    """A slider whose live value is always shown beside its label.

    ``on_change`` is called with the Quasar change event after the value label updates.
    """
    with ui.row().classes("items-center justify-between w-full").style("gap:.5rem;margin-top:.2rem"):
        ui.label(label).style(f"color:{MUTED};font-size:.74rem")
        vlabel = ui.label(fmt.format(val) + suffix).style(
            f"color:{TEXT};font-size:.78rem;font-family:monospace")
    s = ui.slider(min=lo, max=hi, value=val, step=step).props("dense").classes("w-full")

    def _upd(e):
        vlabel.text = fmt.format(e.value) + suffix
        if on_change:
            on_change(e)
    s.on_value_change(_upd)
    return s


def labeled_range(label: str, lo, hi, vmin, vmax, step=1, fmt="{:.0f}", suffix="", on_change=None):
    """A dual-handle range slider whose live ``min–max`` is always shown beside the label.

    ``on_change`` receives the Quasar event (``e.value`` is a ``{'min','max'}`` dict).
    """
    with ui.row().classes("items-center justify-between w-full").style("gap:.5rem;margin-top:.2rem"):
        ui.label(label).style(f"color:{MUTED};font-size:.74rem")
        vlabel = ui.label(f"{fmt.format(vmin)}–{fmt.format(vmax)}{suffix}").style(
            f"color:{TEXT};font-size:.78rem;font-family:monospace")
    r = ui.range(min=lo, max=hi, value={"min": vmin, "max": vmax}, step=step).props("dense").classes("w-full")

    def _upd(e):
        vlabel.text = f"{fmt.format(e.value['min'])}–{fmt.format(e.value['max'])}{suffix}"
        if on_change:
            on_change(e)
    r.on_value_change(_upd)
    return r


def canvas_box():
    """The viewport-filling surface a workspace draws its main view into."""
    return ui.element("div").style(
        f"position:relative;width:100%;flex:1 1 0;min-height:0;border-radius:8px;"
        f"overflow:hidden;background:{PANEL};border:1px solid {BORDER}")


def workspace_frame(canvas_fn: Callable, rail_fn: Callable, dock_fn: Callable | None = None) -> None:
    """The viewport-centric layout: a flexible canvas (+ optional bottom dock) and a fixed
    right rail of live controls. Nothing overflows the viewport."""
    with ui.element("div").style("width:100%;height:100%;display:flex;flex-direction:row;overflow:hidden"):
        with ui.element("div").style(
                "flex:1 1 0;min-width:0;height:100%;display:flex;flex-direction:column;"
                "padding:12px;gap:12px;box-sizing:border-box"):
            canvas_fn()
            if dock_fn:
                dock_fn()
        with ui.element("div").style(
                f"flex:0 0 300px;width:300px;height:100%;overflow:auto;padding:12px;gap:.4rem;"
                f"display:flex;flex-direction:column;background:{PANEL};border-left:1px solid {BORDER};"
                f"box-sizing:border-box"):
            rail_fn()


def selectable_table(columns, rows, row_key, selected_value, on_row, rows_per_page: int = 0):
    """A table where clicking anywhere on a row selects + highlights it (no checkbox).

    The highlight survives re-renders via ``selected_value`` (a value of ``row_key``).
    ``rows_per_page`` (>0) paginates so a several-thousand-row set stays light in the browser.
    """
    table = ui.table(columns=columns, rows=rows, row_key=row_key,
                     pagination=rows_per_page or None).props(
        "dense flat selection=single").classes("w-full pf-rowsel").style("background:transparent")
    if selected_value is not None:
        table.selected = [r for r in rows if r[row_key] == selected_value]

    def _click(e):
        row = e.args[1]
        table.selected = [row]
        on_row(row)
    table.on("rowClick", _click)
    return table


def dock(title: str, columns, rows, row_key, selected_value, on_row, count_label: str = "",
         header_extra=None) -> None:
    """The bottom dock: a fixed-height, searchable, row-select table under the canvas.
    ``header_extra`` renders extra controls into the title row, between the count and the search."""
    with ui.element("div").style(
            f"flex:0 0 230px;height:230px;width:100%;min-width:0;display:flex;flex-direction:column;"
            f"background:{PANEL};border:1px solid {BORDER};border-radius:8px;padding:8px;box-sizing:border-box"):
        with ui.row().classes("items-center w-full no-wrap gap-2").style("flex:0 0 auto;margin-bottom:.3rem"):
            ui.icon("table_rows").style(f"color:{MUTED}").classes("text-sm")
            ui.label(title).style(f"color:{TEXT};font-weight:600;font-size:.82rem")
            if count_label:
                ui.label(count_label).style(f"color:{MUTED};font-size:.74rem")
            if header_extra is not None:
                header_extra()
            ui.space()
            search = ui.input(placeholder="Search…").props("dense outlined clearable").style("width:240px")
        with ui.element("div").style("flex:1 1 0;min-height:0;min-width:0;width:100%;overflow:auto"):
            table = selectable_table(columns, rows, row_key, selected_value, on_row, rows_per_page=15)
            search.bind_value_to(table, "filter")


def subtabs(items, active, on_pick, disabled=None, disabled_tip="", full_width=True):
    """Native Quasar tabs used as in-canvas view switchers (clean alignment, greyed disable).

    ``full_width=False`` lets the tabs share their row with other content (e.g. stat cards).
    """
    disabled = disabled or {}
    with ui.tabs(value=active, on_change=lambda e: on_pick(e.value)).props(
            "dense no-caps align=left active-color=primary indicator-color=primary").classes(
            "w-full" if full_width else "").style(f"color:{MUTED};flex:0 0 auto"):
        for name, icon in items:
            tab = ui.tab(name, icon=icon)
            if disabled.get(name):
                tab.props("disable")
                if disabled_tip:
                    tab.tooltip(disabled_tip)

