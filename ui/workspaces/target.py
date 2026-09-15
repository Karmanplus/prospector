"""Find targets: what the vehicle can reach, and which of those are worth reaching.

Two separate questions, kept apart:
  * Can it be reached? The element-based ``edelbaum`` estimate against the vehicle's budget,
    worked out over the whole catalogue in numpy. Instant, and it re-runs as the screening
    terms move: debounced, computed off the UI thread, with a progress overlay while an
    uncached slice of the catalogue is fetched.
  * Is it worth reaching? What the survivors are made of and how big they are, plus
    their grade and spin. That runs as a background job (``jobs.submit_enrich``), reporting
    per-target progress into the status line, and once it finishes the results are folded into
    the same views so the filters bite immediately.

The plots update in place rather than being redrawn, so moving a slider never resets a 3D camera or
a zoom. Neither question drops anything on its own: a reachable target whose properties are unknown
is kept, and the real solver decides later.
"""
from __future__ import annotations

import asyncio
import math

import pandas as pd
from nicegui import context, run, ui

from prospector import enrichment, figures, jobs, population
from prospector.config import Desirability, Screening
from prospector.solvers import edelbaum
from ui import state
from ui.components import (
    canvas_box,
    dock,
    labeled_range,
    labeled_slider,
    section,
    subtabs,
    workspace_frame,
)
from ui.state import S
from ui.theme import ACCENT, BORDER, GREEN, MUTED, PANEL, PANEL2, RED, TEXT

# Cap on how many reachable targets one screen sends for characterization, so an enormous reachable
# set can't queue an unbounded job.
_ENRICH_CAP = 5000

# Live-recompute debounce: a slider drag fires many value changes; only the last survives this
# delay to do real work (earlier tasks are cancelled here before computing).
_DEBOUNCE_S = 0.25

# The reachability dome is a dense isosurface; this resolution keeps the per-update payload light
# enough to feel instant while still reading as a smooth volume.
_DOME_GRID = 40

# Hide Plotly's modebar and allow scroll-zoom: a clean canvas, drag to rotate/pan, wheel to zoom,
# and no cluttered toolbar.
_PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": True}

_VIEWS = [("Reachability", "public"), ("Orbits", "bar_chart"), ("Value", "diamond")]

# Composition chips -> spectral major classes (lay-friendly groupings of the SBDB taxonomy).
_COMPOSITION_GROUPS = {
    "Carbon-rich": ["C", "B", "D", "P"],
    "Stony": ["S", "Q", "K", "L", "A", "V"],
    "Metallic": ["M", "X", "E", "T"],
}
_TIER_CHOICES = ["(any)", "S", "A", "B", "C", "D"]

# Full-scale ends of the size/spin range sliders; a handle parked at the end means "no bound".
_DIAM_HI = 1000.0
_PERIOD_HI = 40.0

# Sentinel for "this filter wasn't touched, keep its current value" in _set_desirability.
_UNSET = object()

# Live element handles (set on canvas render) so the recompute can update plots in place and drive
# the loading overlay without rebuilding the canvas. ``_overlay_mode`` tracks which busy state is
# showing ("updating" = light dim, instant feedback; "fetching" = stronger, with a progress bar) so
# repeated calls during a drag don't rebuild/flicker it.
_overlay = None
_overlay_mode: str | None = None
_plots: list = []
_recompute_task: asyncio.Task | None = None
# An id that only counts up, marking the latest screen; a run only touches the UI if it is still so
# a superseded run can never hide the overlay or stomp the live view with stale data.
_screen_gen = 0
# In-memory cache of the loaded characterization frame, keyed by (reachable pdes, job-done) so a
# several-thousand-file disk read happens at most once per reachable set, not per base rebuild.
_enrich_frame_key = None
_enrich_frame = None
# The composition chips currently active (workspace-local UI state).
_composition: set[str] = set()


# ======================================================================================
# layout
# ======================================================================================

def render() -> None:
    """Build the Find-targets workspace. Kicks off the first screen if none has run yet."""
    _sync_composition()
    workspace_frame(_canvas, _rail, _dock)
    if S.screen_df is None and S.population is None:
        ui.timer(0.05, lambda: _schedule_screen(rescreen=True), once=True)


def _sync_composition() -> None:
    """Light up the composition chips that the loaded project's desirability already includes
    (a group is active when all its classes are in the taxonomy filter)."""
    included = {str(c).strip().upper() for c in (S.desirability.taxonomy_include or [])}
    _composition.clear()
    for label, classes in _COMPOSITION_GROUPS.items():
        if included and set(classes) <= included:
            _composition.add(label)


@ui.refreshable
def _canvas() -> None:
    global _overlay
    box = canvas_box()
    with box:
        with ui.column().classes("w-full h-full p-2 gap-1").style("box-sizing:border-box"):
            _view_header()
            _plot_area()
        _hud()
        _overlay = ui.element("div").classes(
            "absolute inset-0 items-center justify-center flex")
        _overlay.set_visibility(False)


def _view_header() -> None:
    """The view tabs (left) and the live stat cards (top-right): ΔV budget + reachable /
    desirable rings. Only the cards refresh on a data update, so the tabs never flicker."""
    with ui.row().classes("items-center w-full no-wrap").style("flex:0 0 auto;gap:.5rem"):
        subtabs(_VIEWS, S.target_view, _set_view, full_width=False)
        ui.space()
        _stat_cards()


@ui.refreshable
def _stat_cards() -> None:
    try:
        rc = state.resolved()
    except Exception:  # noqa: BLE001
        return
    df = S.screen_df
    total = len(df) if df is not None else 0
    n_reach = int(df["reachable"].sum()) if df is not None and "reachable" in df.columns else 0
    n_sel = int(df["selected"].sum()) if df is not None and "selected" in df.columns else 0
    with ui.row().classes("items-center no-wrap gap-4"):
        with ui.column().classes("items-center gap-0").style("padding:0 .3rem"):
            ui.label(f"{rc.dv_budget:.1f}").style(
                f"color:{TEXT};font-family:monospace;font-size:1.1rem;line-height:1")
            ui.label("ΔV km/s").style(f"color:{MUTED};font-size:.62rem")
        _ring("reachable", n_reach, total, "positive")   # Quasar token -> GREEN (see ui.colors)
        tip = None
        if df is not None and "uncharacterized" in df.columns and _enriched(df):
            unknown = int(df["uncharacterized"].sum())
            matched = int((df["selected"] & ~df["uncharacterized"]).sum())
            tip = (f"{matched:,} characterized targets meet the filters; {unknown:,} "
                   f"uncharacterized {'listed' if S.desirability.keep_unknown else 'hidden'}")
        _ring("desirable", n_sel, n_reach, "primary", tip)     # Quasar token -> ACCENT


def _ring(label: str, value: int, maximum: int, color: str, tip: str | None = None) -> None:
    with ui.column().classes("items-center gap-0") as col:
        with ui.circular_progress(value=value, min=0, max=max(maximum, 1), size="2.6em",
                                  show_value=False, color=color):
            ui.label(f"{value:,}").style(f"color:{TEXT};font-size:.62rem;font-family:monospace")
        ui.label(label).style(f"color:{MUTED};font-size:.62rem")
    if tip:
        col.tooltip(tip)


@ui.refreshable
def _plot_area() -> None:
    """Render the current view's Plotly element(s) once; data updates patch them in place."""
    _plots.clear()
    df = S.screen_df
    if df is None:
        _empty("public", "screening the catalog…")
        return
    try:
        rc = state.resolved()
    except Exception as exc:  # noqa: BLE001
        ui.label(f"Config does not resolve: {exc}").style(f"color:{RED};font-size:.8rem")
        return
    figs = _figs_for_view(df, rc)
    if not figs:                       # Value view, nothing characterized to draw
        _value_placeholder()
        return
    if len(figs) == 1:
        _plots.append(_make_plot(figs[0]))
    else:
        with ui.row().classes("w-full no-wrap gap-2").style("flex:1 1 0;min-height:0;min-width:0"):
            for fig in figs:
                _plots.append(_make_plot(fig))


def _empty(icon: str, text: str, detail: str | None = None) -> None:
    with ui.column().classes("w-full items-center justify-center gap-2").style("flex:1 1 0"):
        ui.icon(icon).style(f"color:{BORDER};font-size:4.5rem")
        ui.label(text).style(f"color:{MUTED};font-size:.85rem")
        if detail:
            ui.label(detail).style(f"color:{MUTED};font-size:.75rem;max-width:36rem;"
                                   "text-align:center")


def _value_placeholder() -> None:
    """The Value view before there is anything to draw: what the characterization job is doing,
    or why there is none.

    A default install has no characterization at all, since it needs the optional enrichment
    environment, and the job it submits ends in ERROR within seconds. Showing "characterizing..."
    regardless read as a hang to the first person who installed it fresh; the right rail said
    "unavailable" in small grey type, which is not where anyone looks.
    """
    status = _enrich_status() if S.enrich_reachable is not None else {"state": jobs.DONE}
    st = status.get("state")
    if st in (jobs.QUEUED, jobs.RUNNING):
        done, total = status.get("done", 0), status.get("total", 0)
        _empty("diamond", f"characterizing {total:,} new targets... {done}/{total}",
               "The reachable targets nearest by ΔV are looked up in public small-body "
               "catalogues, new ones only; the view fills in when the job finishes. Cancel is "
               "in the right rail.")
    elif st == jobs.ERROR and _is_unavailable(status):
        _empty("diamond", "Target characterization is not installed in this environment.",
               "Reachability and every trajectory solve work without it. To rank targets by "
               "value, launch the app with:  pixi run -e enrichment app")
    elif st == jobs.ERROR:
        _empty("diamond", "Characterization failed.",
               _error_reason(status) or "see the worker log")
    else:
        _empty("diamond", "No characterization for these targets yet.")


def _make_plot(fig):
    return ui.plotly(_payload(fig)).classes("w-full").style("flex:1 1 0;min-height:0;min-width:0")


def _payload(fig) -> dict:
    """A Plotly figure as a JSON payload with the cleaned-up (modebar-free) config."""
    data = fig.to_plotly_json()
    data["config"] = _PLOTLY_CONFIG
    return data


def _figs_for_view(df: pd.DataFrame, rc) -> list:
    """The figure(s) for the active view, restyled to the app and tagged with a constant
    ``uirevision`` so an in-place update preserves the user's camera/zoom.

    Figures come from the pure-core ``plots`` (shared with the PDF report), so the app-specific
    cleanup, which is a panel-matched background, no verbose title and a compact legend, is applied
    here in the UI layer rather than mutating the core.
    """
    view = S.target_view
    if view == "Orbits":
        sel = df["selected"].to_numpy(bool) if _enriched(df) else None
        figs = [figures.element_distributions(df, dv_budget=rc.dv_budget, selected_mask=sel)]
    elif view == "Value":
        if not _enriched(df):
            return []
        reach = df[df["reachable"]]
        figs = [figures.tier_breakdown(reach), figures.desirability_scatter(reach)]
    else:
        figs = [figures.reachability_dome_3d(rc.dv_budget, df, grid=_DOME_GRID,
                                             highlights=_plot_highlights())]
    for fig in figs:
        _restyle(fig, view)
    if view == "Orbits":
        # The three panels carry their own titles along the top edge, exactly where the compact
        # legend sits, so "reachable" landed on top of "semimajor axis a (AU)". The panels leave
        # room below their axis titles; the legend goes there.
        figs[0].update_layout(margin=dict(b=52),
                              legend=dict(orientation="h", yanchor="top", y=-0.22,
                                          xanchor="left", x=0))
    return figs


def _plot_highlights() -> list[dict]:
    """The bodies to mark in the reachable volume: the focus target and the picked table row.

    Both are marked because they are different things and can disagree. Picking a row inspects a
    body while setting focus commits the whole app to it, so seeing where each one sits matters.
    When they are the same body it is marked once as both (the ring drawn around the diamond), and
    only the focus label is drawn so the name is not printed twice on top of itself.

    The focus target's elements come from ``S.focus``, not the screen frame, so a focus that the
    current screening terms exclude is still shown. That is deliberate: a target vanishing from the
    plot when a filter tightens is the moment its position matters most.
    """
    focus, picked = S.focus, _full_record(S.sel_target) if S.sel_target else None
    same = bool(focus and picked and str(focus.get("pdes")) == str(picked.get("pdes")))
    out = []
    if picked:
        out.append({**_elements(picked), "role": "picked",
                    "label": None if same else _short_name(picked)})
    if focus:
        out.append({**_elements(focus), "role": "focus", "label": _short_name(focus)})
    return [h for h in out if h]


def _elements(row: dict) -> dict:
    return {"a": row.get("a"), "e": row.get("e"), "i": row.get("i")}


def _short_name(row: dict) -> str:
    """A label short enough to sit beside a 3D marker: the designation, not the padded full name."""
    return str(row.get("pdes") or row.get("name") or row.get("full_name") or "target").strip()


def _refresh_highlights() -> None:
    """Redraw the plot after the focus or the picked row moved.

    Only the reachable-volume view marks them, and the other views' figures are expensive to
    rebuild over a large population, so a row click on Orbits or Value changes nothing visible and
    does no work.
    """
    if S.target_view == "Reachability":
        _update_plots()


def _restyle(fig, view: str) -> None:
    """De-clutter a core figure for the app: drop the title, match the panel background, and
    compact the legend into a thin transparent strip. ``uirevision`` is keyed to the view so
    repeated in-place updates within a view keep the camera/zoom."""
    fig.update_layout(
        title_text="",
        height=None, autosize=True,
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        margin=dict(l=8, r=8, t=26, b=8),
        font=dict(size=11),
        uirevision=f"view-{view}",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    # Blend the 3D panes into the panel too (a no-op on 2D figures, which have no scene).
    fig.update_scenes(xaxis_backgroundcolor=PANEL, yaxis_backgroundcolor=PANEL,
                      zaxis_backgroundcolor=PANEL)


def _update_plots() -> None:
    """Patch the current view's existing plot element(s) with fresh data (no element rebuild,
    so 3D camera and zoom persist). Falls back to a structural rebuild if the view's plot
    count changed (e.g. characterization just turned the Value placeholder into two charts)."""
    df = S.screen_df
    if df is None:
        _plot_area.refresh()
        return
    try:
        rc = state.resolved()
    except Exception:  # noqa: BLE001
        return
    figs = _figs_for_view(df, rc)
    if len(figs) != len(_plots):
        _plot_area.refresh()
        return
    for el, fig in zip(_plots, figs):
        el.update_figure(_payload(fig))


@ui.refreshable
def _hud() -> None:
    """The floating selection card (top-right of the canvas) with a 'Set as focus' action."""
    sel = S.selected
    if not sel:
        return
    with ui.card().classes("absolute top-3 right-3 p-3 gap-1").style(
            f"background:{PANEL2};border:1px solid {ACCENT};border-radius:10px;"
            f"box-shadow:0 6px 24px rgba(0,0,0,.45);min-width:215px;max-width:280px"):
        with ui.row().classes("items-center w-full no-wrap"):
            ui.icon(sel.get("icon", "place")).style(f"color:{ACCENT}")
            ui.label(sel["title"]).classes("truncate").style(
                f"color:{TEXT};font-weight:700;font-size:.85rem;max-width:200px")
            ui.space()
            ui.button(icon="close", on_click=_clear_selection).props("flat dense round").style(
                f"color:{MUTED}")
        for line in sel["lines"]:
            ui.label(line).style(f"color:{MUTED};font-size:.78rem")
        ui.button(sel["action"], icon=sel.get("action_icon", "arrow_forward"),
                  on_click=sel.get("on_action", lambda: None)).props(
            "dense no-caps unelevated").classes("mt-1")


# ======================================================================================
# right rail: live filters + characterization status
# ======================================================================================

def _rail() -> None:
    s = S.screening
    section("Asteroids to search", "filter_alt")
    labeled_slider("Brightness cutoff (H ≤)", 15.0, 30.0, float(s.h_max), 0.1, "{:.1f}",
                   on_change=lambda e: _set_screening(h_max=e.value))
    labeled_slider("Reachability margin", 1.0, 3.0, float(s.dv_margin_factor), 0.05, "{:.2f}", "×",
                   on_change=lambda e: _set_screening(margin=e.value))

    ui.separator().style(f"background:{BORDER}")
    section("Worth visiting", "diamond")
    d = S.desirability
    tier_val = d.min_tier if d.min_tier in _TIER_CHOICES else "(any)"
    ui.select(_TIER_CHOICES, value=tier_val, label="Minimum quality (tier)").props(
        "dense outlined").classes("w-full").on_value_change(lambda e: _set_desirability(tier=e.value))

    ui.label("Composition").style(f"color:{MUTED};font-size:.72rem;margin-top:.3rem")
    _composition_chips()

    labeled_range("Diameter (m)", 0, _DIAM_HI, float(d.min_diameter_m or 0),
                  float(d.max_diameter_m or _DIAM_HI), 10, "{:.0f}", " m",
                  on_change=lambda e: _set_desirability(diam=(e.value["min"], e.value["max"])))
    labeled_range("Rotation period (h)", 0.0, _PERIOD_HI, float(d.min_period_h or 0),
                  float(d.max_period_h or _PERIOD_HI), 0.5, "{:.1f}", " h",
                  on_change=lambda e: _set_desirability(period=(e.value["min"], e.value["max"])))

    ui.separator().style(f"background:{BORDER}")
    _status()


@ui.refreshable
def _composition_chips() -> None:
    with ui.row().classes("gap-1 flex-wrap"):
        for label in _COMPOSITION_GROUPS:
            active = label in _composition
            ui.button(label, on_click=lambda label=label: _toggle_composition(label)).props(
                "dense flat no-caps").style(
                f"color:{ACCENT if active else MUTED};border:1px solid "
                f"{ACCENT if active else BORDER}")


@ui.refreshable
def _status() -> None:
    """One-line characterization status + progress (counts now live in the stat cards).

    See :func:`_is_unavailable` for why the two ERROR causes are told apart.
    """
    if S.screen_df is None or S.enrich_reachable is None:
        return
    status = _enrich_status()
    st = status.get("state")
    if st in (jobs.QUEUED, jobs.RUNNING):
        done, total = status.get("done", 0), status.get("total", 0)
        frac = min(done / total, 1.0) if total else 0.0
        ui.linear_progress(value=frac, show_value=False).props("rounded").style("margin-top:.3rem")
        with ui.row().classes("items-center justify-between w-full"):
            ui.label(f"characterizing {done}/{total} new").style(
                f"color:{MUTED};font-size:.72rem")
            ui.button("Cancel", on_click=_cancel_enrich).props("flat dense no-caps").style(
                f"color:{MUTED};font-size:.7rem")
    elif st == jobs.ERROR:
        if _is_unavailable(status):
            ui.label("characterization unavailable").style(
                f"color:{MUTED};font-size:.72rem")
        else:
            ui.label("characterization failed").style(f"color:{RED};font-size:.72rem").tooltip(
                _error_reason(status) or "see the job's worker log")
    elif _enriched(S.screen_df) or S.enrich_reachable:
        n = int(S.screen_df["tier"].notna().sum()) if "tier" in S.screen_df.columns else 0
        reach = len(S.enrich_reachable or [])
        with ui.row().classes("items-center gap-1"):
            ui.icon("check_circle").style(f"color:{GREEN}").classes("text-sm")
            ui.label(f"characterized {n:,} of {reach:,} reachable").style(
                f"color:{GREEN};font-size:.72rem")
        remaining = reach - min(reach, S.enrich_window)
        if remaining > 0:
            ui.button(f"Characterize next {min(_ENRICH_CAP, remaining):,}",
                      on_click=_characterize_more).props("flat dense no-caps").style(
                f"color:{ACCENT};font-size:.7rem").tooltip(
                f"The {S.enrich_window:,} reachable targets nearest by ΔV are looked up "
                f"automatically; each press adds the next batch. Results are kept on disk, so "
                f"nothing is looked up twice.")


# ======================================================================================
# bottom dock: the targets table
# ======================================================================================

def _dock() -> None:
    _dock_table()


@ui.refreshable
def _dock_table() -> None:
    df = S.screen_df
    if df is None:
        dock("Targets", [{"name": "n", "label": "", "field": "n"}], [], "n", None, lambda r: None)
        return
    if S.target_show_all:
        # Every body in the catalog, the reachable ones first, then by how far over the budget the
        # rest sit, so an out-of-reach body (Venus, a main-belt asteroid) can still be picked.
        show = [c for c in ["full_name", "tier", "H", "a", "i", "lowthrust_dv", "dv_short"]
                if c in df.columns]
        sub = df.sort_values("lowthrust_dv")
        title, n = "All targets", int(sub.shape[0])
    elif _enriched(df):
        show = [c for c in ["full_name", "tier", "taxonomy", "diameter_m", "period_h", "lowthrust_dv"]
                if c in df.columns]
        # Measured matches first, then the rows kept only because a value is unknown.
        sub = (df[df["selected"]].sort_values(
            [c for c in ["uncharacterized", "tier", "lowthrust_dv"] if c in df.columns])
               if "selected" in df.columns else df[df["reachable"]])
        title, n = "Desirable targets", int(sub.shape[0])
        count = _selection_count(df)
    else:
        show = ["full_name", "H", "a", "e", "i", "lowthrust_dv"]
        sub = df[df["reachable"]].sort_values("lowthrust_dv")
        title, n = "Reachable targets", int(sub.shape[0])
    columns, rows = _table_data(sub, show)
    dock(title, columns, rows, "pdes", S.sel_target, _pick_target,
         count_label=count if _enriched(df) and not S.target_show_all else f"({n})",
         header_extra=_show_all_toggle)


def _selection_count(df: pd.DataFrame) -> str:
    """The dock's count for a described table: measured matches apart from the unknowns, and
    whether the unknowns are in the list or hidden by the switch."""
    if "uncharacterized" not in df.columns or "selected" not in df.columns:
        return f"({int(df['selected'].sum()) if 'selected' in df.columns else 0})"
    unknown = int(df["uncharacterized"].sum())
    matched = int((df["selected"] & ~df["uncharacterized"]).sum())
    if unknown == 0:
        return f"({matched})"
    if S.desirability.keep_unknown:
        return f"({matched:,} match · {unknown:,} uncharacterized)"
    return f"({matched:,} match · {unknown:,} uncharacterized hidden)"


def _show_all_toggle() -> None:
    ui.switch("show all", value=S.target_show_all, on_change=_set_show_all).props(
        "dense size=xs").style(f"color:{MUTED};font-size:.7rem").tooltip(
        "Include the bodies over the ΔV budget, with how far over they are, so one can still be "
        "set as the focus target")
    if S.screen_df is not None and _enriched(S.screen_df) and not S.target_show_all:
        ui.switch("show uncharacterized", value=bool(S.desirability.keep_unknown),
                  on_change=lambda e: _set_desirability(keep=bool(e.value))).props(
            "dense size=xs").style(f"color:{MUTED};font-size:.7rem").tooltip(
            "Targets not characterized yet cannot be tested against the filters. Off: the list "
            "holds only measured matches. On: they are listed too, marked ?.")


def _set_show_all(e) -> None:
    S.target_show_all = bool(e.value)
    _dock_table.refresh()


_LABELS = {"full_name": "Target", "tier": "Tier", "taxonomy": "Type", "diameter_m": "Size (m)",
           "period_h": "Spin (h)", "lowthrust_dv": "ΔV (km/s)", "dv_short": "Over budget (km/s)",
           "H": "Brightness (H)", "a": "a (AU)", "e": "e", "i": "i (°)"}
_ROUND = {"diameter_m": 0, "period_h": 1, "lowthrust_dv": 2, "dv_short": 2, "H": 1, "a": 3,
          "e": 3, "i": 1}
# Described properties that read "?" when not measured, so an empty cell is never mistaken for a
# value that met the filter. Numeric ones sort the "?" rows below every number.
_UNKNOWN_MARK = ("tier", "taxonomy", "diameter_m", "period_h")
_SORT_UNKNOWN_LAST = "(a, b) => (a === '?' ? -1e30 : a) - (b === '?' ? -1e30 : b)"


def _table_data(df: pd.DataFrame, show: list[str]):
    """Build NiceGUI (columns, rows) from a frame. Rows carry only ``pdes`` (the row key) plus
    the displayed, JSON-safe values; a click re-derives the full record from the frame, so the
    payload sent to the browser stays light even for a several-thousand-row reachable set."""
    columns = [{"name": c, "label": _LABELS.get(c, c), "field": c, "sortable": True,
                "align": "left" if c in ("full_name", "taxonomy", "tier") else "right",
                **({":sort": _SORT_UNKNOWN_LAST} if c in ("diameter_m", "period_h") else {})}
               for c in show]
    rows = []
    for rec in df.to_dict("records"):
        full = jobs.jsonable_row(rec)
        row = {"pdes": str(full.get("pdes"))}
        for c in show:
            v = full.get(c)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                row[c] = "?" if c in _UNKNOWN_MARK else None
            elif isinstance(v, (int, float)) and c in _ROUND:
                row[c] = round(float(v), _ROUND[c])
            else:
                row[c] = v
        rows.append(row)
    return columns, rows


def _full_record(pdes: str) -> dict:
    """The full population record (elements + any characterization) for a designation."""
    df = S.screen_df
    hit = df[df["pdes"].astype(str) == str(pdes)]
    return jobs.jsonable_row(hit.iloc[0]) if len(hit) else {"pdes": pdes}


def _pick_target(row: dict) -> None:
    S.sel_target = row.get("pdes")
    full = _full_record(row.get("pdes"))
    name = full.get("full_name") or full.get("pdes") or "target"
    lines = []
    if full.get("lowthrust_dv") == full.get("lowthrust_dv"):
        lines.append(f"ΔV ~ {float(full['lowthrust_dv']):.2f} km/s")
    if full.get("tier"):
        lines.append(f"tier {full['tier']} · {full.get('taxonomy') or '-'}")
    if isinstance(full.get("diameter_m"), (int, float)) and full["diameter_m"] == full["diameter_m"]:
        lines.append(f"{float(full['diameter_m']):.0f} m")
    short = full.get("dv_short")
    if isinstance(short, (int, float)) and short == short and short > 0:
        lines.append(f"over the budget by {float(short):.2f} km/s")
    S.selected = {"icon": "place", "title": str(name), "lines": lines or ["reachable"],
                  "action": "Set as focus", "action_icon": "my_location",
                  "on_action": lambda f=full: _set_focus(f)}
    _hud.refresh()
    _refresh_highlights()       # move the picked-row marker in the reachable volume


def _set_focus(full: dict) -> None:
    state.set_focus(dict(full))
    S.selected = None
    ui.notify(f"Focus → {full.get('full_name') or full.get('pdes')}")
    _hud.refresh()
    _refresh_highlights()       # the focus marker moves with it
    # The top bar's focus chip lives in topbar; refresh it (lazy import avoids an import cycle).
    from ui import topbar
    topbar.header.refresh()


def _clear_selection() -> None:
    S.selected = None
    _hud.refresh()


def _set_view(value: str) -> None:
    S.target_view = value
    _plot_area.refresh()


# ======================================================================================
# the live screen + its characterization
# ======================================================================================

def _set_screening(*, h_max: float | None = None, margin: float | None = None) -> None:
    s = S.screening
    try:
        S.screening = Screening(
            h_max=h_max if h_max is not None else s.h_max,
            dv_margin_factor=margin if margin is not None else s.dv_margin_factor,
            reference=s.reference)
    except Exception:  # noqa: BLE001  (a passing out-of-range value mid-drag is ignored)
        return
    state.mark_dirty()
    _schedule_screen(rescreen=True)     # H / margin change reachability -> rebuild the base


def _set_desirability(*, tier=_UNSET, diam=_UNSET, period=_UNSET, keep=_UNSET) -> None:
    """Rebuild the working Desirability from the rail. ``diam``/``period`` are ``(min, max)``
    tuples in slider units (a handle at 0 / the full-scale end means 'no bound'); ``_UNSET``
    keeps the current value so a control only edits its own term."""
    d = S.desirability
    dmin, dmax = _bounds(diam, (d.min_diameter_m, d.max_diameter_m), _DIAM_HI)
    pmin, pmax = _bounds(period, (d.min_period_h, d.max_period_h), _PERIOD_HI)
    if tier is _UNSET:
        min_tier = d.min_tier
    else:
        min_tier = None if tier == "(any)" else tier
    try:
        S.desirability = Desirability(
            keep_unknown=d.keep_unknown if keep is _UNSET else bool(keep),
            min_tier=min_tier,
            taxonomy_include=_composition_classes() or None,
            min_diameter_m=dmin, max_diameter_m=dmax,
            min_period_h=pmin, max_period_h=pmax)
    except Exception:  # noqa: BLE001
        return
    state.mark_dirty()
    _schedule_screen(rescreen=False)    # desirability is a pure re-select off the cached base


def _bounds(value, current, hi):
    """Turn a slider ``(min, max)`` tuple into ``(min_or_None, max_or_None)``: a min at 0 and a
    max at the full-scale end are treated as 'unbounded'. ``_UNSET`` returns the current pair."""
    if value is _UNSET:
        return current
    lo_v, hi_v = value
    return (lo_v or None, hi_v if hi_v < hi else None)


def _toggle_composition(label: str) -> None:
    if label in _composition:
        _composition.discard(label)
    else:
        _composition.add(label)
    _composition_chips.refresh()
    _set_desirability()


def _composition_classes() -> list[str]:
    classes: set[str] = set()
    for label in _composition:
        classes.update(_COMPOSITION_GROUPS[label])
    return sorted(classes)


def _schedule_screen(*, rescreen: bool) -> None:
    """Debounce: dim the canvas IMMEDIATELY (instant 'recomputing' feedback so the user isn't
    staring at stale data), supersede any in-flight recompute, and start a fresh one. Captures
    the client here (in a valid UI context) so the detached task can re-enter it to notify.

    ``rescreen=True`` rebuilds the expensive base (reachability + enrichment merge, off-thread);
    ``rescreen=False`` re-applies the filter off the cached base: no disk, and instant."""
    global _recompute_task, _screen_gen
    _show_overlay("updating")
    _screen_gen += 1
    if _recompute_task and not _recompute_task.done():
        _recompute_task.cancel()
    _recompute_task = asyncio.ensure_future(run_screen(context.client, _screen_gen, rescreen))


async def run_screen(client, gen: int, rescreen: bool) -> None:
    """Recompute the screen and patch the views in place.

    A ``rescreen`` (or a cold start) rebuilds the base off the UI thread: fetch the catalog
    (uncached H escalates to a stronger fetch overlay) and merge characterization. A pure
    desirability change skips all of that and just re-selects the cached base on the spot.

    Only the LATEST scheduled run (``gen == _screen_gen``) touches the UI, so a superseded run
    can't hide the overlay mid-drag or render stale data. UI calls that need a client context
    (notify) re-enter the captured ``client``, since a detached task's slot stack is empty."""
    try:
        await asyncio.sleep(_DEBOUNCE_S)            # a newer change cancels this before any work
    except asyncio.CancelledError:
        return
    try:
        rc = state.resolved()
    except Exception as exc:  # noqa: BLE001
        _finish_error(client, gen, f"Config does not resolve: {exc}")
        return

    if rescreen or S.screen_base is None:
        h = round(rc.screening.h_max, 1)
        if (S.population is None or S.population_h != h) and gen == _screen_gen:
            _show_overlay("fetching", f"Fetching catalog (brightness H ≤ {h:.1f})…")
        try:
            result = await run.io_bound(
                _compute_base, rc, S.population, S.population_h, S.enrich_reachable)
        except asyncio.CancelledError:
            return                      # a newer run is in flight and owns the overlay
        except Exception as exc:  # noqa: BLE001  (report a network failure; do not crash)
            _finish_error(client, gen, f"SBDB fetch failed: {exc}")
            return
        if result is None or gen != _screen_gen:
            return                      # superseded mid-compute -> the newer run owns the UI
        base, pop, pop_h = result
        S.population, S.population_h, S.screen_base = pop, pop_h, base
        _dispatch_enrichment(base)

    if gen != _screen_gen or S.screen_base is None:
        return

    S.screen_df = enrichment.apply_selection(S.screen_base, rc.desirability)
    # The data is in place; only touch the UI if Find targets is still the live workspace. A run
    # that finishes after the user navigated away would otherwise drive deleted elements (the next
    # visit rebuilds the canvas/dock from S.screen_df anyway).
    if S.workspace != "target":
        _overlay_mode_reset()
        return
    _hide_overlay()
    _stat_cards.refresh()
    _update_plots()
    _dock_table.refresh()
    _status.refresh()


def _finish_error(client, gen: int, message: str) -> None:
    """Surface a screen error from the latest run only, re-entering the client for ``ui.notify``."""
    if gen != _screen_gen:
        return
    _hide_overlay()
    try:
        with client:
            ui.notify(message, type="negative")
    except Exception:  # noqa: BLE001  (the client navigated away; nothing to notify)
        pass


def _compute_base(rc, cached, cached_h, enrich_reachable):
    """Pure (UI-free) base compute, safe to run in a worker thread: fetch the catalog slice
    (cached by H), score reachability, and merge any cached characterization. The desirability
    filter is applied separately, and cheaply, so moving one never touches disk again.

    ``cached`` / ``cached_h`` are the previously loaded frame and the cutoff it was loaded at;
    ``None`` loads it. They are not named for the module they come from, because a parameter called
    ``population`` shadows the ``population`` module inside this function, which is how the catalog
    load was once silently turned into an attribute lookup on the parameter.
    """
    h = round(rc.screening.h_max, 1)
    if cached is None or cached_h != h:
        # The whole candidate set: near-Earth asteroids brighter than the cutoff, the bright end
        # of the main belt, plus the major planets; the cutoff applies to the near-Earth slice only
        # (see population.catalog).
        cached = population.load_population(h_max=h)
        cached_h = h
    df = edelbaum.evaluate_dataframe(cached, dv_budget=rc.dv_budget)
    # How far over the budget each body sits (0 when reachable), for the "show all" table.
    df["dv_short"] = (df["lowthrust_dv"] - float(rc.dv_budget)).clip(lower=0.0)
    df = _merge_enrichment(df, _enrichment_frame(enrich_reachable))
    return df, cached, cached_h


def _show_overlay(mode: str, message: str = "") -> None:
    """Show the busy overlay. ``mode='updating'`` is a light dim + small spinner for instant
    feedback on any tweak; ``mode='fetching'`` is a stronger overlay with a message + an
    indeterminate progress bar for the slow uncached-catalog fetch. Re-showing the same mode
    is a no-op, so a slider drag doesn't rebuild/flicker it."""
    global _overlay_mode
    if _overlay is None or _overlay.is_deleted or _overlay_mode == mode:
        return
    _overlay_mode = mode
    strong = mode == "fetching"
    _overlay.clear()
    _overlay.style(f"background:rgba(20,26,35,{0.55 if strong else 0.32});"
                   + ("backdrop-filter:blur(1.5px)" if strong else ""))
    with _overlay:
        with ui.column().classes("items-center gap-2"):
            ui.spinner(size="lg" if strong else "md").style(f"color:{ACCENT}")
            if message:
                ui.label(message).style(f"color:{TEXT};font-size:.82rem")
            if strong:
                ui.linear_progress(show_value=False).props("indeterminate rounded").style("width:220px")
    _overlay.set_visibility(True)


def _hide_overlay() -> None:
    global _overlay_mode
    _overlay_mode = None
    if _overlay is not None and not _overlay.is_deleted:
        _overlay.set_visibility(False)


def _overlay_mode_reset() -> None:
    """Clear the tracked overlay mode without touching the element (used when the workspace is no
    longer live, so the deleted overlay is never driven)."""
    global _overlay_mode
    _overlay_mode = None


def _dispatch_enrichment(df: pd.DataFrame, *, more: bool = False) -> None:
    """Submit a background characterization for this screen's reachable ASTEROIDS if it changed.

    The nearest ``S.enrich_window`` by ΔV are covered (``more`` widens it by one batch); only the
    ids not already on disk are sent, so the progress shown is new lookups and a slider moved
    mid-run replaces the job with the still-missing remainder rather than starting over.

    Planets are excluded. What gets described is mining value, covering size, composition,
    structural stability and observability, assessed against a small-body database; asking it about
    Mars would either fail the lookup or invent a tier for a body the whole model does not
    describe. A planet is reachable or not, and its worth is not this axis's to judge.
    """
    reachable = df[df["reachable"]].dropna(subset=["pdes"]).sort_values("lowthrust_dv")
    if population.BODY_CLASS_COL in reachable.columns:
        reachable = reachable[reachable[population.BODY_CLASS_COL] != population.PLANET]
    pdes = reachable["pdes"].astype(str).tolist()
    if more:
        S.enrich_window += _ENRICH_CAP
    _covered, todo = _enrichment_plan(pdes, enrichment.cached_ids(pdes), S.enrich_window)
    if pdes == S.enrich_reachable and todo == S.enrich_todo:
        return                      # nothing new to look up; the running/finished job applies
    S.enrich_reachable = pdes       # everything on disk among these is merged, window or not
    S.enrich_todo = todo
    S.enrich_loaded = False
    S.enrich_job = jobs.submit_enrich(todo) if todo else None


def _enrichment_plan(pdes: list[str], cached: set[str], window: int) -> tuple[list[str], list[str]]:
    """Of the reachable ids nearest by ΔV first, the ``window`` the characterization covers and
    the ids in it not yet on disk, which are all a job looks up. A finished batch leaves nothing
    to do, so it never runs again; a wider window or a body newly in reach adds only the new."""
    covered = pdes[:max(int(window), 0)]
    return covered, [p for p in covered if p not in cached]


def _characterize_more() -> None:
    if S.screen_base is not None:
        _dispatch_enrichment(S.screen_base, more=True)
    _status.refresh()


def _is_unavailable(status: dict) -> bool:
    """Whether this characterization job failed only because the optional dependencies are absent.

    Two very different things land in ERROR and they must not read alike. "The enrichment extra is
    not installed here" is the expected state of a default install and says nothing about the
    screen the user is looking at; a real failure is worth alarming about.

    Both status fields are consulted. The message is the deliberate signal the worker records, the
    error is the raw exception text, and keying on only one of them makes the calm path depend on
    how an exception happens to be worded in another package, which is how it silently stops
    matching.
    """
    detail = f"{status.get('message', '')} {status.get('error', '')}".lower()
    return "unavailable" in detail or "not installed" in detail


def _error_reason(status: dict) -> str:
    """The last line of what the worker recorded for a failed job: the exception's own words,
    without the traceback above them."""
    text = str(status.get("error") or status.get("message") or "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    last = lines[-1]
    return last.split(": ", 1)[1] if ": " in last and not last.startswith("File ") else last


def _enrich_status() -> dict:
    return jobs.read_enrich_status(S.enrich_job) if S.enrich_job else {"state": jobs.DONE}


def _value_columns() -> list[str]:
    return [c for c in enrichment.ENRICH_COLUMNS if c not in ("name", "input_id")]


def _enrichment_frame(pdes):
    """The per-object characterization on disk for ``pdes``, loaded at most once per (set,
    completion-state). ``enrichment.load_cached`` reads one parquet PER id, so for a several-
    thousand-target reachable set this is expensive, so caching it here keeps repeated base
    rebuilds (e.g. nudging the margin) off the disk. Invalidated when the reachable set changes
    or the characterization job finishes (``S.enrich_loaded`` flips)."""
    global _enrich_frame_key, _enrich_frame
    key = (tuple(pdes) if pdes else (), bool(S.enrich_loaded))
    if key != _enrich_frame_key:
        _enrich_frame = enrichment.load_cached(pdes) if pdes else None
        _enrich_frame_key = key
    return _enrich_frame


def _merge_enrichment(df: pd.DataFrame, enriched) -> pd.DataFrame:
    """Left-join a (pre-loaded) characterization frame onto the reachability frame by ``pdes``."""
    base = df.drop(columns=[c for c in enrichment.ENRICH_COLUMNS if c in df.columns], errors="ignore")
    if enriched is None or not len(enriched):
        return base
    cols = [c for c in _value_columns() if c in enriched.columns]
    return base.merge(enriched[["input_id", *cols]], how="left",
                      left_on="pdes", right_on="input_id").drop(columns=["input_id"], errors="ignore")


def poll_enrich() -> None:
    """Page-level timer hook: stream characterization progress and merge it when the job ends."""
    if S.workspace != "target" or S.enrich_reachable is None or S.screen_df is None:
        return
    status = _enrich_status()
    if not jobs.is_terminal(status):
        _status.refresh()           # advance the progress bar
        if S.target_view == "Value" and not _plots:
            _plot_area.refresh()    # the placeholder carries the count too
        return
    if S.enrich_loaded:
        return
    # Terminal: mark done (invalidates the enrichment-frame cache) and rebuild the base once so
    # the finished characterization is woven in. The rescreen runs off-thread with the overlay.
    S.enrich_loaded = True
    _schedule_screen(rescreen=True)


def _cancel_enrich() -> None:
    if S.enrich_job:
        jobs.cancel_enrich(S.enrich_job)
    _status.refresh()


def _enriched(df: pd.DataFrame) -> bool:
    return "tier" in df.columns
