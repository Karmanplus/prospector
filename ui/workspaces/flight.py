"""Plan trajectory: flying one vehicle to the focus target, in order.

Sub-tabs across the canvas run Earth escape, then Trajectory, then Diagnostics, each unlocked by
the one before. The porkchop sits as a small inset on the Trajectory canvas rather than having a
tab of its own, the right-hand rail carries only the controls for whichever sub-tab is open, and a
timeline along the bottom colours the escape and the cruise across real dates. Everything the
flight workflow needs is here: running the escape spiral and reading its result and trade curve,
the porkchop with its clickable grid of launch dates and its budget comparison, the cruise solve
with live progress, the finished 3D trajectory and its per-node diagnostics, and the mission's
total delta-v and propellant.

Each of the four solves runs as its own detached job (``worker.py``, through
``prospector.jobs``), polled by a page-level timer (:func:`poll`). The escape sub-tab gates the
rest: an escape flown by the thrusters has to be planned, or the launch vehicle has to provide
one, before the cruise budget means anything. The Earth escape therefore comes first.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from nicegui import run, ui

from prospector import figures, jobs, trajectory_export
from prospector.config import ResolvedConfig
from prospector.figures import products
from prospector.figures.trajectory import flyby_marker
from prospector.solvers.spiral import belt_power_profile
from prospector.spacecraft.radiation import load_radiation_models
from prospector.trades import sweep
from ui import state
from ui.components import (
    canvas_box,
    kg_added,
    kg_plain,
    kg_rem,
    kg_spent,
    labeled_slider,
    section,
    subtabs,
    workspace_frame,
)
from ui.state import S
from ui.theme import ACCENT, AMBER, BORDER, GREEN, MUTED, PANEL, PANEL2, RED, TEXT

# Mission-phase palette for the journey strip + timeline. Escape (amber) and cruise (accent)
# already have theme colors; the round trip adds the stay at the asteroid as a grey coast and the
# laden RETURN leg (blue, the hue the vehicle table already ships for its first chip).
STAY = "#8b96a9"
RETURN = "#5b8def"

_ELEMENTS = ["a", "e", "i", "om", "w", "ma", "epoch"]
_PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": True}
# No "Flight-time trade" tab: that curve is the transfer grid's per-column minimum, so it is drawn
# on the same axes as the field it came from rather than re-solved as a separate study.
_SUBTABS = [("Earth escape", "north_east"), ("Trajectory", "rocket"),
            ("Mission profile", "stacked_line_chart"), ("Diagnostics", "insights")]

# Channels whose detached job was running on the last poll tick, so the running→terminal edge
# triggers one full re-render, with cheap progress ticks otherwise and no heavy plot redraw.
_running: set[str] = set()

# Live mission-timeline handles, so the play tick scrubs the thumb + readout in place (no bar
# rebuild every 0.1 s); ``_tl_span`` caches the current span so the tick needn't re-resolve.
_tl_slider = None
_tl_label = None
_tl_span: dict = {"esc_days": 0, "cruise_days": 0, "stay_days": 0, "return_days": 0,
                  "total_days": 0, "liftoff": None}

# The timeline scrubs the plots by moving only their marker traces (Plotly.restyle on a single
# 1-point trace) and never recomputes the heavy 3D path, so a high-resolution one stays cheap, and
# the thumb + dot advance on one clock (the tick), so they can never drift apart. These hold the
# live plot element, its marker trace indices, and the full path data to interpolate over. All
# None/empty unless that view is on screen.
_traj_plot = None
_traj_marker: dict = {}                 # trace name -> index (spacecraft / Earth / asteroid / cursor)
_traj_sc = None                         # spacecraft path (AU), one row per fine sample
_traj_earth = None                      # Earth track (AU), aligned with _traj_sc
_traj_ast = None                        # target track (AU), aligned with _traj_sc
_traj_fb = None                         # gravity-assist body track (AU), aligned with _traj_sc; None without one
_traj_days = None                       # day-from-departure per sample (drives the thrust cursor)
_traj_esc_frac = 0.0                    # where the cruise phase begins on the global timeline
_traj_cruise_end_frac = 1.0             # where the cruise phase ends (1.0 when there is no return)

# The laden return arc scrubs the same way, over the timeline's return phase.
_ret_plot = None
_ret_marker: dict = {}
_ret_sc = None                          # return spacecraft path (AU)
_ret_earth = None                       # Earth track over the return (the destination body)
_ret_ast = None                         # asteroid track over the return (the origin body)
_ret_days = None
_ret_start_frac = 1.0                   # where the return phase begins on the global timeline

_spiral_plot = None
_spiral_marker = None                   # the spiral's moving-marker trace index
_spiral_pos = None                      # the full geocentric spiral path (km)
_spiral_esc_frac = 1.0                  # the escape phase's share of the global timeline


# ======================================================================================
# entry + shared render plumbing
# ======================================================================================

def render() -> None:
    """Build the Plan-trajectory workspace (canvas + right rail + timeline dock)."""
    workspace_frame(_canvas, _rail, _timeline)


def _refresh_all() -> None:
    """Re-render the whole workspace + the top-bar budget chips after a state change."""
    _canvas.refresh()
    _rail.refresh()
    _timeline.refresh()
    from ui import topbar
    topbar.header.refresh()


def _resolve() -> ResolvedConfig | None:
    try:
        return state.resolved()
    except Exception:  # noqa: BLE001  (the caller reports a config that will not resolve)
        return None


def _set_subtab(value: str) -> None:
    S.flight_tab = value
    S.timeline_playing = False          # the plot is rebuilt on switch; don't orphan a play loop
    # The mission timeline follows the phase being viewed: escape ↔ cruise.
    rc = _resolve()
    if rc is not None:
        span = _timeline_span(rc)
        if value == "Earth escape":
            S.timeline_t = 0.0
        elif value == "Trajectory" and span["total_days"] > 0:
            S.timeline_t = min(1.0, span["esc_days"] / span["total_days"])
    _refresh_all()


def _escape_ready(rc) -> bool:
    """True once the Earth escape is settled: provided by the launch vehicle, or an escaped
    spiral is in force for this config. The cruise budget is only real past this point. The
    spiral is the one this project points at, whether flown here or restored with the project, and
    still matching its escape fingerprint; never one merely found on disk."""
    if rc.launch.escape_provided:
        return True
    return state.session_spiral_run(rc) is not None


def _fig(fig, key: str, *, height: int | None = None, paper: str = PANEL,
         plot: str | None = None):
    """Render a pure-core Plotly figure into the dark canvas: panel-matched background, a
    constant ``uirevision`` so an in-place update keeps the camera/zoom, modebar hidden.

    ``paper`` is the surround, ``plot`` the axes area, and they are the same colour unless a
    caller separates them. :func:`_fig_card` does, so that a figure of stacked subplots reads as
    tiles rather than one wash of colour.
    """
    # Drop the verbose figure title (the metrics live in the cards) and match the panel; leave
    # each figure's own margins (the porkchop needs axis room the 3D plots don't).
    plot = paper if plot is None else plot
    fig.update_layout(title_text="", paper_bgcolor=paper, plot_bgcolor=plot, autosize=True,
                      height=None, uirevision=key, font=dict(color=MUTED, size=11))
    fig.update_scenes(xaxis_backgroundcolor=paper, yaxis_backgroundcolor=paper,
                      zaxis_backgroundcolor=paper)
    data = fig.to_plotly_json()
    data["config"] = _PLOTLY_CONFIG
    # A height pins the figure to a fixed band (e.g. a 2D chart beneath the 3D views); without one
    # it flexes to fill the remaining canvas height.
    style = (f"flex:0 0 {height}px;height:{height}px;min-width:0" if height
             else "flex:1 1 0;min-height:0;min-width:0")
    return ui.plotly(data).classes("w-full").style(style)


def _fig_card(title: str, icon: str, fig, key: str, *, height: int | None = None):
    """One chart in its own titled panel, the same card the sizing explainers use.

    The timeline tabs stack several charts down one scrolling column, and drawn straight onto the
    canvas they run together: same background, no edge, nothing saying where one ends. The card
    gives each an edge and a name, and holding the axes area at the canvas colour inside the
    lighter card separates the subplots within a chart too.

    ``height`` pins the card to a band, for a column of several. Without one the card fills the
    height it is given, which is what a single chart on its own wants.
    """
    grow = ("flex:0 0 auto" if height else "flex:1 1 0;min-height:0")
    with ui.element("div").classes("w-full").style(
            f"background:{PANEL2};border:1px solid {BORDER};border-radius:8px;"
            f"padding:8px 10px 4px;display:flex;flex-direction:column;{grow}"):
        with ui.row().classes("items-center gap-1").style("flex:0 0 auto"):
            ui.icon(icon).style(f"color:{ACCENT}").classes("text-sm")
            ui.label(title).style(f"color:{TEXT};font-weight:600;font-size:.78rem")
        _fig(fig, key, height=height, paper=PANEL2, plot=PANEL)


def _empty(icon: str, text: str) -> None:
    with ui.column().classes("w-full items-center justify-center gap-2").style("flex:1 1 0"):
        ui.icon(icon).style(f"color:{BORDER};font-size:4.5rem")
        ui.label(text).style(f"color:{MUTED};font-size:.85rem;max-width:520px;text-align:center")


# ======================================================================================
# canvas: the budget strip header, the gated sub-tabs, and the active view
# ======================================================================================

@ui.refreshable
def _canvas() -> None:
    global _traj_plot, _spiral_plot
    # Only the views that draw these (re)set them to a live plot; cleared so a timeline scrub on
    # any other sub-tab safely no-ops instead of driving a stale element.
    _traj_plot = _spiral_plot = None
    box = canvas_box()
    with box:
        with ui.column().classes("w-full h-full p-2 gap-2").style("box-sizing:border-box"):
            rc = _resolve()
            if rc is None:
                _empty("error_outline", "The configuration does not resolve - fix it in Project.")
                return
            gated = not _escape_ready(rc)
            with ui.row().classes("items-center w-full no-wrap").style("flex:0 0 auto;gap:.6rem"):
                subtabs(_SUBTABS, S.flight_tab, _set_subtab,
                        disabled={"Trajectory": gated, "Mission profile": gated,
                                  "Diagnostics": gated},
                        disabled_tip="Plan the Earth escape first", full_width=False)
                ui.space()
                _journey_strip(rc)
                ui.button(icon="public", on_click=_export_trajectory).props(
                    "flat dense no-caps").style(f"color:{MUTED}").tooltip(
                    "Export the trajectory bundle.").set_enabled(
                    _cruise_result(S.solve_run_id) is not None)
            if S.flight_tab == "Earth escape":
                _escape_view(rc)
            elif S.flight_tab == "Trajectory":
                _trajectory_view(rc)
            elif S.flight_tab == "Mission profile":
                _mission_profile_view(rc)
            else:
                _diagnostics_view(rc)


def _journey_strip(rc) -> None:
    """The headline mission journey, in line with the sub-tabs (matching the other workspaces):
    the ΔV capability on the LEFT, the dated Departure → Escape → Arrival (→ Stay → Return →
    Destination, for a round trip) timeline in the middle, and the ΔV used plus margin on
    the RIGHT once solved.

    Nothing is invented: a leg's time/ΔV/propellant shows only once it is real. The propellant
    ledger reads across the strip with one convention, where a leg's arrow shows what it spent
    (``−NNN kg``), a node shows what REMAINS after it (``NNN rem``); the asteroid Stay shows the
    payload it ADDS (``+NNN kg``)."""
    from datetime import timedelta
    res = _cruise_result(S.solve_run_id)
    # A run that did not converge has no propellant figure to quote: the strip shows its dates and
    # nothing about the tank, the same as before a solve.
    solved = res is not None and _converged(res["sf"]["feasible"], res["sf"]["mismatch"])
    # Escape time is real only once a spiral has been flown this session (or the LV provides it).
    escape_known = rc.launch.escape_provided or state.session_spiral_run(rc) is not None
    esc_days = 0.0 if rc.launch.escape_provided else float(rc.escape_tof_days)
    # The escape leg's label is the velocity change the spiral actually produced when one flew
    # (the budget term, rc.escape_dv, is the propellant in rated-Isp units, which is larger when
    # the engine ran power-limited); the analytic estimate otherwise.
    spiral_run = None if rc.launch.escape_provided else state.session_spiral_run(rc)
    flown = products.spiral_block(spiral_run) if spiral_run else None
    escape_dv = (float(flown["dv_kms"]) if flown and flown.get("dv_kms") is not None
                 else rc.escape_dv)

    if solved:
        leg = res["sf"]
        escape_dt = figures.mjd2000_to_datetime(float(leg["dep_mjd2000"])).date()
        cruise_days = float(leg["tof_days"])
        arrival_dt = escape_dt + timedelta(days=round(cruise_days))
        cruise_dv = float(leg["dv_kms"])
    else:
        escape_dt = rc.departure_window[0]           # planned earliest cruise departure
        arrival_dt = None                            # arrival is a prediction -> blank until solved
        cruise_days = cruise_dv = None

    # The escape/cruise seam. There is no gap to report: the launch date and the cruise departure
    # are one choice offset by the spiral, so liftoff is back-dated from the solved departure and
    # the escape hands straight over. See ResolvedConfig.departure_schedule.
    sched = rc.departure_schedule(escape_dt if solved else None)
    liftoff = (sched["liftoff_if_no_coast"] or sched["launch_open"]) if escape_known else None

    # Propellant ledger: usable load onboard, spent per leg, remaining after it. Each piece is
    # known only once its leg is real (escape after the spiral, cruise after the solve).
    usable = max(0.0, rc.vehicle.fuel_mass - rc.vehicle.unusable_prop)
    esc_prop = rc.escape_propellant_kg if escape_known else None
    # The cruise's propellant is what the solve spent at the operating points it flew; the rocket
    # equation at the RATED Isp is only the fallback for a result without it, and understates a
    # power-limited cruise, whose Isp sits below rated.
    cruise_prop = products.cruise_propellant_kg(rc, leg) if solved else None
    prop_left = (usable - (esc_prop or 0.0) - cruise_prop
                 if (solved and cruise_prop is not None) else None)

    # The return half (only when the mission plans a round trip). Values fill in once the chained
    # return solve has produced its block; until then the Stay/Return/Destination render as dashes.
    return_planned = rc.mission.return_trip
    ret = res.get("return") if (solved and return_planned) else None
    ret_ok = bool(ret and "error" not in ret and ret.get("sf"))
    payload = float(rc.mission.asteroid_payload_mass) if return_planned else None
    stay_days = float(rc.mission.time_at_asteroid or 0.0) if return_planned else None
    dest_label = rc.mission.return_destination if return_planned else None
    return_days = return_dv = return_prop = insertion_dv = delivered = None
    stay_date = dest_date = prop_after_return = None
    if ret_ok:
        rsf = ret["sf"]
        return_days = float(rsf["tof_days"])
        return_dv = float(rsf["dv_kms"])
        return_prop = float(ret["total_return_propellant_kg"])
        insertion_dv = float(ret["insertion_dv_kms"])
        delivered = float(ret["delivered_mass_kg"])
        if arrival_dt is not None:
            stay_date = arrival_dt + timedelta(days=round(stay_days or 0.0))
            dest_date = stay_date + timedelta(days=round(return_days))
        prop_after_return = (prop_left - return_prop) if prop_left is not None else None

    total_days = (esc_days + cruise_days) if (escape_known and solved) else None
    if total_days is not None and ret_ok:
        total_days = total_days + (stay_days or 0.0) + return_days

    with ui.row().classes("items-center no-wrap").style("gap:.5rem"):
        # LEFT - the propellant the whole mission must fit inside. In kilograms rather than a dV
        # capability, because the legs fly at different Isps and only the kilograms add up.
        _journey_stat(f"{usable:.0f} kg", "Propellant",
                      "Usable propellant aboard - escape + cruise"
                      + (" + return + insertion" if return_planned else "") + " must fit inside this")
        ui.separator().props("vertical").style(f"background:{BORDER};height:34px")
        # MIDDLE - the dated timeline, with the propellant ledger underneath
        _journey_node(_date(liftoff), "Liftoff", kg_plain(usable),
                      _liftoff_tip(rc, sched, escape_known, solved)
                      + f" · {usable:.0f} kg propellant onboard")
        _journey_leg("escape", esc_days if escape_known else None,
                     f"{escape_dv:.1f} km/s" if escape_known else None,
                     kg_spent(esc_prop),
                     f"Earth escape priced at v∞ {rc.departure_vinf_kms:.2f} ({rc.escape_source})"
                     + ("" if escape_known else " · run the spiral for its duration, ΔV, and propellant"))
        # Nothing between them: the escape ends where the cruise begins.
        _journey_node(_date(escape_dt), "Cruise start", None,
                      _cruise_start_tip(sched, solved))
        fb = leg.get("flyby") if solved else None
        if fb:
            # Two legs joined at the flyby: each shows its own time and propellant, the node the
            # planet, the date and what remains after it.
            body = str(fb.get("body", "planet")).capitalize()
            leg_prop = fb.get("leg_propellant_kg") or [None, None]
            after_fb = (usable - (esc_prop or 0.0) - float(leg_prop[0])
                        if leg_prop[0] is not None else None)
            _journey_leg("cruise", float(fb["tof1_days"]), None, kg_spent(leg_prop[0]),
                         f"Heliocentric cruise to {body}")
            _journey_node(_date(figures.mjd2000_to_datetime(float(fb["mjd2000"])).date()),
                          f"{body} flyby", kg_rem(after_fb),
                          f"unpowered flyby at {fb['periapsis_alt_km']:,.0f} km, "
                          f"{fb['vinf_kms']:.2f} km/s relative, turned {fb['turn_deg']:.0f}°"
                          + (f" · {after_fb:.0f} kg propellant remaining" if after_fb is not None else ""))
            _journey_leg("cruise", float(fb["tof2_days"]), None, kg_spent(leg_prop[1]),
                         f"Heliocentric cruise from {body} to the target")
        else:
            _journey_leg("cruise", cruise_days, None if cruise_dv is None else f"{cruise_dv:.1f} km/s",
                         kg_spent(cruise_prop),
                         "Heliocentric cruise"
                         + ("" if solved else " - run the cruise solve for its time, ΔV, and propellant"))
        _journey_node(_date(arrival_dt), "Arrival", kg_rem(prop_left),
                      ("predicted arrival at the target" if solved else "predicted after the cruise solve")
                      + (f" · {prop_left:.0f} kg propellant remaining" if prop_left is not None else ""))
        if return_planned:
            _journey_node(_date(stay_date), f"Stay {stay_days:.0f}d" if stay_days else "Stay",
                          kg_added(payload),
                          (f"{stay_days:.0f} d at the target · " if stay_days else "")
                          + f"+{payload:.0f} kg payload collected" if payload is not None
                          else "payload collected at the target")
            _journey_leg("return", return_days,
                         None if return_dv is None else f"{return_dv:.1f} km/s",
                         kg_spent(return_prop),
                         (f"Laden return to {dest_label}"
                          + (f" · insertion ΔV {insertion_dv:.1f} km/s" if insertion_dv is not None else "")
                          if ret_ok else
                          (f"return could not be planned: {ret['error']}" if (ret and "error" in ret)
                           else f"return to {dest_label} - runs right after the cruise solve")))
            _journey_node(_date(dest_date), dest_label or "Return", kg_rem(prop_after_return),
                          (f"arrive {dest_label}" if ret_ok else "predicted after the return solve")
                          + (f" · {prop_after_return:.0f} kg propellant remaining"
                             if prop_after_return is not None else ""),
                          extra=(f"delivered {delivered:.0f} kg" if delivered is not None else None))
        ui.separator().props("vertical").style(f"background:{BORDER};height:34px")
        # RIGHT - the propellant used plus margin (only after a solve), smaller; it restates the legs
        _used_block(solved, esc_prop, cruise_prop, usable, total_days, return_prop=return_prop)


def _liftoff_tip(rc, sched: dict, escape_known: bool, solved: bool) -> str:
    """Why the liftoff date is what it is. Nobody picks it: the mission owns a launch window,
    and within that the date follows from when the cruise departs and how long the escape takes.
    Saying so is the point, since otherwise the date slides silently as the solve moves."""
    window = f"launch window {_date(sched['launch_open'])} – {_date(sched['launch_close'])}"
    if not escape_known:
        return f"{window} · set once the escape spiral is flown"
    if not solved:
        return (f"{window} · the earliest liftoff, shown until a cruise is solved · "
                f"{rc.cruise_start_mass_kg:.0f} kg at cruise start")
    note = ("" if sched["liftoff_in_window"] else
            " ⚠ outside the launch window - the mission window cannot deliver this departure")
    return (f"Back-dated from the cruise departure by the {sched['escape_days']:.0f}-day escape. "
            f"The launch date and the departure are one choice, not two · {window}{note}")


def _cruise_start_tip(sched: dict, solved: bool) -> str:
    if not solved:
        return (f"Earliest cruise departure ({_date(sched['cruise_open'])}) - the launch window "
                f"shifted by the escape duration; the solver plans inside "
                f"{_date(sched['cruise_open'])} – {_date(sched['cruise_close'])}")
    return (f"The solved heliocentric departure · the solver's window was "
            f"{_date(sched['cruise_open'])} – {_date(sched['cruise_close'])}")


def _journey_node(value: str, label: str, prop: str | None, tip: str, *,
                  extra: str | None = None) -> None:
    """A dated timeline node. ``prop`` (when given) adds the propellant onboard/remaining line;
    ``extra`` adds a second small line (e.g. the Destination node's delivered mass)."""
    with ui.column().classes("items-center gap-0").style("min-width:54px").tooltip(tip):
        ui.label(value).style(
            f"color:{TEXT if value != '-' else MUTED};font-family:monospace;"
            "font-size:.9rem;line-height:1.1")
        ui.label(label).style(f"color:{MUTED};font-size:.6rem")
        if prop is not None:
            ui.label(prop).style(f"color:{MUTED};font-size:.58rem;font-family:monospace")
        if extra is not None:
            ui.label(extra).style(f"color:{MUTED};font-size:.58rem;font-family:monospace")


def _journey_leg(name: str, days, dv, prop: str, tip: str) -> None:
    """One leg between two nodes: its name, an arrow, 'days · ΔV', and the propellant it spends
    (``−NNN kg``). A None for days, ΔV or propellant renders a muted dash, meaning not known until
    its solve runs."""
    days_txt = f"{days:.0f} d" if days is not None else "-"
    dv_txt = dv if dv else "-"
    known = days is not None or (dv not in (None, "-"))
    with ui.column().classes("items-center gap-0").style("min-width:84px").tooltip(tip):
        ui.label(name).style(f"color:{MUTED};font-size:.55rem;text-transform:uppercase;"
                             "letter-spacing:.06em")
        ui.icon("arrow_forward").style(f"color:{ACCENT if known else BORDER};font-size:1rem")
        ui.label(f"{days_txt} · {dv_txt}").style(
            f"color:{TEXT if known else MUTED};font-size:.6rem")
        ui.label(prop).style(f"color:{MUTED};font-size:.58rem;font-family:monospace")


def _journey_stat(value: str, label: str, tip: str) -> None:
    with ui.column().classes("items-center gap-0").style("padding:0 .2rem").tooltip(tip):
        ui.label(value).style(f"color:{TEXT};font-family:monospace;font-size:1.15rem;line-height:1")
        ui.label(label).style(f"color:{MUTED};font-size:.62rem")


def _used_block(solved: bool, esc_prop, cruise_prop, usable: float, total_days,
                *, return_prop=None) -> None:
    """The right-hand summary: propellant used and the margin left in the tank, plus the
    whole-mission duration. With a return solved the used total folds in the return leg's
    propellant (cruise + insertion), so the margin reflects the whole round trip. Kilograms,
    not a dV margin: each leg spends at its own operating point's Isp, so only the mass adds up."""
    with ui.column().classes("gap-0").style("min-width:104px"):
        if solved and cruise_prop is not None:
            used = (esc_prop or 0.0) + cruise_prop + (return_prop or 0.0)
            margin = usable - used
            ui.label(f"used {used:.0f} kg").style(
                f"color:{TEXT};font-family:monospace;font-size:.72rem")
            ui.label(f"margin {margin:+.0f} kg").style(
                f"color:{GREEN if margin >= 0 else RED};font-size:.66rem")
        else:
            ui.label("propellant used").style(f"color:{MUTED};font-size:.66rem")
        ui.label(f"mission {total_days:.0f} d" if total_days is not None else "mission -").style(
            f"color:{MUTED};font-size:.6rem")


def _date(value) -> str:
    return value.strftime("%m/%d/%y") if value else "-"


# ======================================================================================
# ① EARTH ESCAPE - the spiral result, tradeoff curve, and run status
# ======================================================================================

def _escape_view(rc) -> None:
    """The escape view reflects only the spiral run launched this session. It never adopts a
    prior run from disk, so the user always sees the result of the run they kicked off here."""
    if rc.launch.escape_provided:
        _empty("rocket_launch",
               f"{rc.launch.name} provides escape, so there is nothing to fly here.")
        return
    run_id = S.launch_run_id
    if not run_id:
        _empty("north_east", "No escape spiral yet. Run one from the right.")
        return
    status = jobs.read_spiral_status(run_id)
    if not jobs.is_terminal(status):
        _empty("north_east", "Escape spiral running - see the progress bar on the right.")
        return
    if status.get("state") == jobs.ERROR:
        _empty("error_outline", "The spiral propagation failed - see the worker log.")
        return
    res = jobs.read_spiral_result(run_id)
    if res is None:
        _empty("north_east", "Finishing up…")
        return
    # A spiral no longer matching the live escape setup (a vehicle / launch-type / duty / radiation
    # edit) is stale: clear the view rather than leave the old climb up with a warning, so what's
    # shown always belongs to the current config. (The departure-v∞ knob is excluded from the
    # fingerprint, since one run's curve prices every departure speed and moving it never stales
    # the spiral.) A matching-but-failed run keeps its own diagnostic below.
    if not _spiral_matches_config(run_id, rc):
        _empty("north_east", "Config changed. Re-run the spiral from the right.")
        return
    _spiral_result(rc, res["spiral"])


def _spiral_matches_config(run_id, rc) -> bool:
    """Whether a finished spiral was flown for the live escape setup (same fingerprint the
    budget refinement keys on). Best-effort: an unreadable/old job is treated as matching."""
    try:
        return jobs.read_spiral_job(run_id).get("fingerprint") == state.spiral_fingerprint(rc)
    except (OSError, KeyError, ValueError):
        return True


def _spiral_result(rc, sp: dict) -> None:
    status = sp.get("status")
    if status == "out_of_fuel":
        ui.label("⛽ Ran out of propellant before escaping. Try a higher drop-off or more "
                 "propellant.").style(f"color:{RED};font-size:.8rem")
    elif status == "timed_out":
        ui.label(f"⏱ Timed out after {sp['tof_days']:.0f} days - raise the max spiral time "
                 f"or lower the departure speed.").style(f"color:{AMBER};font-size:.8rem")
    if sp.get("power_limited"):
        avail = float(sp.get("power_available_end_W") or 0.0)
        req = float(sp.get("power_required_W") or 1.0)
        thrust_N, isp_s = rc.thrust_isp_at_power(avail)
        rated_mN, rated_isp = rc.total_thrust_mN, rc.effective_isp
        tpct = 100.0 * (thrust_N * 1e3) / rated_mN if rated_mN else 100.0
        isp_clause = (f", Isp {isp_s:.0f} s (vs {rated_isp:.0f} rated)"
                      if isp_s < rated_isp - 1.0 else "")
        ui.label(f"⚡ Power-limited: the array delivers {avail:.0f} W of {req:.0f} W rated, so "
                 f"the stack makes {thrust_N * 1e3:.0f} mN ({tpct:.0f}% of rated)"
                 f"{isp_clause}.").style(f"color:{AMBER};font-size:.78rem")
    global _spiral_plot, _spiral_marker, _spiral_pos, _spiral_esc_frac
    pos = np.asarray(sp["positions_km"], float)
    times = np.asarray(sp["times_days"], float)
    # Permanent array degradation as flown: the spiral records the per-sample power fraction from
    # the selected radiation model, so the headline card and the chart track whatever
    # model/coverglass was run. Older runs (no stored array) fall back to a default-model
    # reconstruction.
    power_fraction = np.asarray(sp.get("power_fraction", []), float)
    if power_fraction.size != len(times) or not power_fraction.size:
        _in_belt, _belt_cum, power_fraction = belt_power_profile(pos, times)
    power_left = (float(power_fraction[-1]) if len(power_fraction)
                  else float(sp.get("power_fraction_end", 1.0)))
    _spiral_cards(sp, power_left)
    _handover_line(rc)
    span = _timeline_span(rc)
    _spiral_esc_frac = _phase_fracs(span)["escape_end"]
    _spiral_pos = pos
    # One geocentric 3D view: the full climb to escape with the Van Allen belts overlaid and the
    # Moon for scale. Zoom in to inspect the belt crossings and the low-altitude winding. The
    # mission timeline scrubs the spacecraft marker along it (controls=False: marker only, no
    # Play/slider chrome).
    fig = figures.spiral_3d(pos, times, title=f"Escape spiral - {rc.launch.name}",
                          dv_kms=sp["dv_kms"], tof_days=sp["tof_days"], animate=True,
                          controls=False, show_belts=True, show_moon=True)
    _spiral_marker = _trace_index(fig, "scrub-marker")
    _spiral_plot = _fig(fig, "escape-3d")
    # The two 2D charts sit SIDE BY SIDE in a fixed-height band, so the 3D spiral above keeps the
    # rest of the canvas. Left: permanent array degradation over the mission clock (the L-shell
    # model integrated along the flown path. The steep rise is the inner proton belt, the flat
    # segment the gentler electron belt; when the power loop is closed, the loss beyond which full
    # thrust can't be held is marked, so a power-limited design is obvious). Right: altitude band +
    # local dose, showing where in altitude/time the dose is taken.
    with ui.row().classes("w-full no-wrap").style("flex:0 0 250px;height:250px;gap:.5rem"):
        with ui.column().classes("h-full").style("flex:1 1 0;min-width:0;gap:0"):
            if len(power_fraction):
                _fig(figures.degradation_profile(times, power_fraction,
                                               power_floor_pct=products.power_floor_pct(sp)), "escape-degrad")
        with ui.column().classes("h-full").style("flex:1 1 0;min-width:0;gap:0"):
            _fig(figures.altitude_radiation_profile(times, pos), "escape-altrad")
    _scrub_spiral(S.timeline_t)



def _handover_line(rc) -> None:
    """What the spiral's duration means for the schedule: which launch date reaches the cruise.

    The spiral's own numbers say how long the escape takes; this reads that as a launch date. The
    two are the same choice: the cruise departure is the launch date plus the spiral, and the
    departure window is the launch window shifted by that much, so there is no third quantity and
    nothing to wait through. Naming the pair is what keeps the escape and the cruise legible as one
    schedule rather than two.
    """
    res = _cruise_result(S.solve_run_id)
    dep = None
    if res is not None:
        leg = res["sf"]
        dep = figures.mjd2000_to_datetime(float(leg["dep_mjd2000"])).date()
    sched = rc.departure_schedule(dep)
    days = sched["escape_days"]
    if dep is None:
        text = (f"Cruise departs {days:.0f} d after liftoff, so the solver plans inside "
                f"{_date(sched['cruise_open'])} – {_date(sched['cruise_close'])}.")
        color = MUTED
    elif sched["liftoff_in_window"]:
        text = (f"Departing {_date(dep)} means lifting off "
                f"{_date(sched['liftoff_if_no_coast'])}.")
        color = GREEN
    else:
        # The one real violation, and it can only arise when the departure bound was widened past
        # the launch window (the solve's window slack). The launch window cannot deliver this
        # departure at all: the only remedy is a slower spiral, or a different departure.
        text = (f"⚠ Departing {_date(dep)} would need liftoff on "
                f"{_date(sched['liftoff_if_no_coast'])}, which is outside the launch window "
                f"({_date(sched['launch_open'])} – {_date(sched['launch_close'])}). This "
                f"departure is not reachable from the pad: lower the departure-window slack, or "
                f"fly a slower spiral so the same launch date hands over later.")
        color = RED
    ui.label(text).style(f"color:{color};font-size:.72rem;line-height:1.35")


def _spiral_cards(sp: dict, power_fraction_end: float) -> None:
    prop = sp["initial_mass_kg"] - sp["final_mass_kg"]
    escaped = sp.get("dv_at_escape_kms") is not None
    cards = [
        ("Escape ΔV", f"{sp['dv_at_escape_kms']:.2f} km/s" if escaped else "-"),
        (f"ΔV to v∞ {sp['vinf_kms']:.2f}", f"{sp['dv_kms']:.2f} km/s"),
        ("Spiral time", f"{sp['tof_days']:.0f} d"),
        ("Propellant", f"{prop:.0f} kg"),
        ("Array power left", f"{power_fraction_end * 100:.0f} %"),
        ("Final inclination", f"{sp['inc_deg_end']:.1f}°"),
    ]
    _stat_cards(cards)


# ======================================================================================
# ② TRAJECTORY - porkchop inset + the converged low-thrust 3D path + mission ΔV stack
# ======================================================================================

def _trajectory_view(rc) -> None:
    res = _cruise_result(S.solve_run_id)
    with ui.element("div").classes("relative w-full").style("flex:1 1 0;min-height:0"):
        if res is not None:
            _converged_trajectory(rc, res)
        else:
            _trajectory_placeholder()
        _porkchop_inset(rc)


def _trajectory_placeholder() -> None:
    run_id = S.solve_run_id
    # A run solved for a DIFFERENT config no longer belongs to this trajectory: treat it as if
    # there were no run (a clean "pick a cell and solve" prompt), never "did not converge".
    live = run_id and _run_matches_config(run_id, _resolve())
    if run_id and live and not jobs.is_terminal(jobs.read_status(run_id)):
        msg = "Solve running - progress on the right."
    elif run_id and live and jobs.read_status(run_id).get("state") == jobs.ERROR:
        msg = "The solve failed - see the worker log on the right."
    elif run_id and live:
        msg = "Did not converge - retune on the right and re-run."
    elif _grid_result() is None:
        msg = "Run the transfer grid on the right, then pick a cell and refine it."
    else:
        msg = "Pick a cell on the transfer grid, then refine it on the right."
    with ui.column().classes("absolute inset-0 items-center justify-center"):
        _empty("rocket", msg)


def _converged_trajectory(rc, res: dict) -> None:
    """The converged design: the Sims-Flanagan cruise.
    A compact card row + the 3D trajectory; mission totals close it out."""
    block = res["sf"]
    tgt, lam = res["target"], res["lambert"]
    span = _timeline_span(rc)
    # When a return converged, show the two legs SIDE BY SIDE; otherwise the outbound fills the
    # canvas and any return status (unconverged / errored) renders beneath it.
    two_up = bool(rc.mission.return_trip and _return_plottable(res))
    with ui.column().classes("absolute inset-0 gap-1").style("padding:2px"):
        title = f'{tgt["name"]} - low-thrust design'
        ui.label(title).style(f"color:{TEXT};font-weight:600;font-size:.85rem")
        with ui.row().classes("w-full no-wrap").style("flex:1 1 0;min-height:0;gap:.5rem"):
            with ui.column().classes("h-full").style("flex:1 1 0;min-width:0;gap:.25rem"):
                if two_up:
                    ui.label("Outbound").style(
                        f"color:{TEXT};font-weight:600;font-size:.82rem")
                _lowthrust_cards(block, lam)
                _planet_arrival_note(tgt)
                _outbound_plot(rc, res, block, tgt, span)
            if two_up:
                with ui.column().classes("h-full").style("flex:1 1 0;min-width:0;gap:.25rem"):
                    _return_trajectory(rc, res, span)
        if not two_up:
            _return_trajectory(rc, res, span)    # error/unconverged status below (no plot)


def _outbound_plot(rc, res: dict, block: dict, tgt: dict, span: dict) -> None:
    """Build the outbound 3D arc + wire its scrub globals (the timeline scrubs it in place)."""
    # controls=False: the figure carries just the moving markers (spacecraft / Earth /
    # asteroid and thrust cursor) with no Play button or slider, since the mission timeline moves
    # them
    # via Plotly.restyle (see _scrub_traj), never recomputing the arc.
    global _traj_plot, _traj_marker, _traj_sc, _traj_earth, _traj_ast, _traj_fb, _traj_days
    global _traj_esc_frac, _traj_cruise_end_frac
    fr = _phase_fracs(span)
    # The cruise starts after the escape and any wait at the handover.
    _traj_esc_frac, _traj_cruise_end_frac = fr["cruise_start"], fr["cruise_end"]
    _traj_sc = np.asarray(block["fine_positions_au"], float)
    _traj_earth = np.asarray(res["orbits"]["earth_track_au"], float)
    _traj_ast = np.asarray(res["orbits"]["target_track_au"], float)
    _traj_days = np.asarray(block["fine_times_days"], float)
    # A gravity-assist body moves through the scene like Earth and the target, so the encounter
    # is visible when the timeline is scrubbed to the flyby.
    orbits = res["orbits"]
    fb_track = orbits.get("flyby_track_au")
    _traj_fb = None if fb_track is None else np.asarray(fb_track, float)
    fig = figures.trajectory_3d(
        block["fine_positions_au"], block["fine_throttle"], block["fine_times_days"],
        block["node_times_days"], block["throttle"], orbits["earth_au"],
        orbits["target_au"], orbits["earth_track_au"],
        orbits["target_track_au"], _fmt(block["dep_mjd2000"]),
        _fmt(block["dep_mjd2000"] + block["tof_days"]), tgt["name"], block["dv_kms"],
        block["tof_days"], target_i_deg=tgt["i"], controls=False,
        max_throttle_pct=_available_thrust_fraction(rc) * 100.0,
        flyby_orbit_au=orbits.get("flyby_au"), flyby_track_au=fb_track,
        flyby_name=str(orbits.get("flyby_name") or "").capitalize(),
        flyby_day=(block["flyby"]["tof1_days"] if block.get("flyby") else None))
    _traj_marker = {n: _trace_index(fig, n)
                    for n in ("scrub-sc", "scrub-earth", "scrub-ast", "scrub-fb", "scrub-cursor")}
    _traj_plot = _fig(fig, "traj-3d")
    _scrub_traj(S.timeline_t)              # sync the markers to the current playhead


def _return_trajectory(rc, res: dict, span: dict) -> None:
    """The laden return arc, stacked below the outbound one when the mission flies home.

    A second 3D path (asteroid → Earth/destination) with its own card row. The mission
    timeline scrubs it over the return phase via :func:`_scrub_return`. Resets the return
    scrub globals to None when there is no return so a stale arc never scrubs."""
    global _ret_plot, _ret_marker, _ret_sc, _ret_earth, _ret_ast, _ret_days, _ret_start_frac
    _ret_plot = _ret_sc = _ret_earth = _ret_ast = _ret_days = None
    _ret_marker = {}
    ret = res.get("return") if rc.mission.return_trip else None
    if not (ret and "error" not in ret and ret.get("sf")):
        if ret and "error" in ret:
            ui.label(f"Return to {ret.get('destination', '')} could not be planned: {ret['error']}").style(
                f"color:{AMBER};font-size:.74rem")
        return
    rb, orb = ret["sf"], ret["orbits"]
    # Do not plot a return the optimizer never closed. An unconverged path is not a flyable
    # trajectory. Report it instead, and leave the scrub globals cleared (set above).
    if not _converged(rb["feasible"], rb["mismatch"]):
        dest = ret.get("destination_name") or ret.get("destination") or "destination"
        ui.label(f"Return to {dest} did not converge (mismatch {rb['mismatch']:.1e}).").style(
            f"color:{AMBER};font-size:.74rem;margin-top:.2rem")
        return
    dest = ret.get("destination_name") or ret.get("destination") or "destination"
    ast_name = res["target"]["name"]
    _ret_start_frac = _phase_fracs(span)["return_start"]
    ui.label(f"Return - laden, {ast_name} → {dest}"
             f"{'' if rb['feasible'] or ret['feasible_propellant'] else '  ⚠ over propellant budget'}").style(
        f"color:{TEXT};font-weight:600;font-size:.82rem;margin-top:.2rem")
    _return_cards(ret)
    # Orbit colors are body-consistent with the outbound (Earth blue, asteroid orange):
    # earth_au/earth_track = Earth, target_au/target_track = the asteroid. The spacecraft path
    # departs the asteroid, so fine[0] sits ON the orange asteroid orbit, and the orange asteroid
    # sphere tracks it there (the scrub markers feed the matching figure traces).
    _ret_sc = np.asarray(rb["fine_positions_au"], float)
    _ret_earth = np.asarray(orb["earth_track_au"], float)    # Earth -> blue "scrub-earth"
    _ret_ast = np.asarray(orb["target_track_au"], float)     # asteroid -> orange "scrub-ast"
    _ret_days = np.asarray(rb["fine_times_days"], float)
    fig = figures.trajectory_3d(
        rb["fine_positions_au"], rb["fine_throttle"], rb["fine_times_days"],
        rb["node_times_days"], rb["throttle"], orb["earth_au"], orb["target_au"],
        orb["earth_track_au"], orb["target_track_au"], f"depart {_fmt(rb['dep_mjd2000'])}",
        f"{dest} {_fmt(rb['dep_mjd2000'] + rb['tof_days'])}", ast_name, rb["dv_kms"],
        rb["tof_days"], target_i_deg=res["target"]["i"], controls=False,
        max_throttle_pct=_available_thrust_fraction(rc) * 100.0)
    _ret_marker = {n: _trace_index(fig, n)
                   for n in ("scrub-sc", "scrub-earth", "scrub-ast", "scrub-cursor")}
    _ret_plot = _fig(fig, "return-3d")
    _scrub_return(S.timeline_t)


def _return_cards(ret: dict) -> None:
    rb = ret["sf"]
    dep, arr = rb["dep_mjd2000"], rb["dep_mjd2000"] + rb["tof_days"]
    _stat_cards([
        ("Return ΔV", f"{rb['dv_kms']:.2f} km/s"),
        ("Flight time", f"{rb['tof_days']:.0f} d"),
        ("Insertion ΔV", f"{ret['insertion_dv_kms']:.2f} km/s"),
        ("Depart asteroid", _fmt(dep)),
        ("Arrive", _fmt(arr)),
        ("Delivered", f"{ret['delivered_mass_kg']:.0f} kg"),
    ])


def _planet_arrival_note(tgt: dict) -> None:
    """Say what "arrival" means when the target is a planet.

    The solvers work out a rendezvous with the target's orbit. Alongside an asteroid that is the
    whole job; at a planet it is capture at infinity, and orbit insertion or entry is extra. The
    omission is invisible in the delta-v, since a transfer that closes with margin has said nothing
    about whether the vehicle can stop, so it is stated wherever the arrival number is shown.
    """
    from prospector.population import planets as planet_lib

    if not planet_lib.is_planet(tgt):
        return
    name = tgt.get("name") or "the planet"
    ui.label(f"↳ Rendezvous with {name}'s orbit. Capture and descent are not "
             f"included.").style(f"color:{AMBER};font-size:.7rem;line-height:1.3")


def _lowthrust_cards(block: dict, lam: dict) -> None:
    dep, arr = block["dep_mjd2000"], block["dep_mjd2000"] + block["tof_days"]
    cards = [
        ("ΔV", f"{block['dv_kms']:.2f} km/s"),
        ("Flight time", f"{block['tof_days']:.0f} d"),
        ("Departure", _fmt(dep)),
        ("Arrival", _fmt(arr)),
    ]
    # A rendezvous arrives within the solver's small slack; anything above it is a pass the
    # mission asked for, so the speed of the pass is part of the answer.
    if float(block.get("vinf_arr_kms") or 0.0) > 0.2:
        cards.append(("Arrival speed", f"{float(block['vinf_arr_kms']):.2f} km/s relative"))
    # The share of the whole cruise spent firing. The busiest single segment is a per-segment
    # figure, not a mission one, so it belongs on the thrust profile's hover next to the segment it
    # describes rather than beside the mission totals.
    duty = products.cruise_duty_cycle(block)
    if duty is not None:
        cards.append(("Duty cycle · whole cruise", f"{duty['mean'] * 100:.1f} %"))
    fb = block.get("flyby")
    if fb:
        body = str(fb.get("body", "planet")).capitalize()
        direct = block.get("direct") or {}
        saved = (float(direct["propellant_kg"]) - float(block["propellant_kg"])
                 if direct.get("propellant_kg") is not None else None)
        cards.append((f"{body} flyby", _fmt(fb["mjd2000"])))
        cards.append(("Flyby altitude", f"{fb['periapsis_alt_km']:,.0f} km"))
        cards.append(("Flyby speed · turn", f"{fb['vinf_kms']:.2f} km/s · {fb['turn_deg']:.0f}°"))
        if saved is not None:
            cards.append(("Against flying direct", f"{saved:+.0f} kg" if saved < 0 else f"saves {saved:.0f} kg"))
    _stat_cards(cards)


def _porkchop_inset(rc) -> None:
    """The transfer grid as a minimizable inset, bottom-left of the canvas: the surface a departure
    and a flight time are chosen from, with the best-per-flight-time curve drawn across it."""
    grid = _grid_result()
    if grid is None:
        return
    if S.porkchop_min:
        with ui.card().classes("absolute bottom-2 left-2 p-1").style(
                f"background:{PANEL2};border:1px solid {BORDER};border-radius:8px"):
            with ui.row().classes("items-center gap-1"):
                ui.icon("grid_on").style(f"color:{MUTED}").classes("text-xs")
                ui.label("transfer options").style(f"color:{MUTED};font-size:.7rem")
                ui.button(icon="open_in_full", on_click=lambda: _set_porkchop_min(False)).props(
                    "flat dense round").style(f"color:{MUTED}").tooltip("Expand")
        return
    with ui.card().classes("absolute bottom-2 left-2 p-1 gap-1").style(
            f"background:{PANEL2};border:1px solid {BORDER};border-radius:8px;"
            f"width:460px;height:360px"):
        with ui.row().classes("items-center w-full no-wrap").style("flex:0 0 auto"):
            ui.icon("grid_on").style(f"color:{MUTED}").classes("text-xs")
            ui.label("click a cell").style(f"color:{MUTED};font-size:.72rem")
            ui.space()
            ui.button(icon="close_fullscreen", on_click=lambda: _set_porkchop_min(True)).props(
                "flat dense round").style(f"color:{MUTED}").tooltip("Minimize")
        _grid_plot(rc, grid)


def _grid_plot(rc, grid: dict) -> None:
    """The clickable converged grid. Every drawn cell is a trajectory the optimizer closed; a cell
    it did not close reads "not found", never infeasible."""
    sel = S.grid_sel
    # Built through products, the same call the report goes through, so the surface on screen and
    # the one on paper cannot drift apart. The caption is for print; the workspace has its own.
    built = products.grid_figure(grid, rc, colour_by=S.grid_colour,
                                 sel_dep=sel[0] if sel else None,
                                 sel_tof=sel[1] if sel else None)
    if built is None:
        return
    fig = built[1]
    fig.update_layout(height=None, margin=dict(l=40, r=8, t=8, b=30), showlegend=False)
    el = _fig(fig, f"grid-{grid.get('target_pdes')}")
    # Emit only the clicked cell's coordinates. Plotly's click event carries the whole trace it
    # fired on, meaning every x, y and customdata array, which the websocket refuses for a grid of
    # any size, and selecting a cell then failed outright. The js_handler runs in the browser and
    # sends a handful of numbers instead.
    el.on("plotly_click", _on_grid_click,
          js_handler="(e) => { const p = e.points && e.points[0];"
                     " if (p) emit({cd: p.customdata, x: p.x, y: p.y}); }")


def _on_grid_click(e) -> None:
    """Pick a cell: (departure epoch, flight time) in MJD2000 days.

    The browser sends the clicked marker's customdata, which carries the cell's exact coordinates.
    A cell the search did not close is still selectable, since re-running it with more effort is a
    reasonable thing to want, and refusing the click would read as the cell being forbidden.
    """
    args = e.args if isinstance(e.args, dict) else {}
    cd = args.get("cd")
    if isinstance(cd, dict):
        cd = list(cd.values())
    cell = None
    if isinstance(cd, (list, tuple)) and len(cd) >= 2:
        try:
            cell = (float(cd[0]), float(cd[1]))
        except (TypeError, ValueError):
            cell = None
    if cell is None:
        return
    if cell != S.grid_sel:
        S.grid_sel = cell
        _show_grid_cell(cell)
        _refresh_all()


def _show_grid_cell(cell) -> None:
    """Show the clicked cell's trajectory immediately, with no new solve.

    The grid already converged this cell and kept its solution vector, so the full result rebuilds
    from it in milliseconds: the cards, the 3D path, the diagnostics and the propellant. That is
    what the vector cube is for: a click used to mean "seed a two-minute solve", and now it means
    "show me the one you already have".

    It is recorded through the ordinary run channel rather than shown by a side path, so every
    consumer downstream (the trajectory view, the mission profile, the PDF) reads it unchanged.

    Refining is still a different question, not a slower version of this one: this cell is pinned
    to its exact departure and duration, while a refine lets the optimizer search off the pin and
    can find something better. Measured, doing that improved 7 of 8 columns on Apophis.
    """
    grid = _grid_result()
    if grid is None:
        return
    rc = _resolve()
    if rc is None:
        return
    from prospector.trades import pipeline

    refined = _polished_point(grid, cell[0], cell[1])
    if refined is not None:
        x = refined["decision_vector"]
        dep_m, tof = float(refined["dep_mjd2000"]), float(refined["solved_tof_days"])
        terms = refined.get("leg_terms") or {}
    else:
        x = _grid_vector(grid, cell[0], cell[1])
        if x is None:
            return                 # a cell the search never closed has nothing to show
        dep_m, tof = float(cell[0]), float(cell[1])
        terms = _grid_leg_terms(grid, cell[0], cell[1])
    # The terms the CELL was solved under, echoed back by the grid. Using the rail's current ones
    # would rebuild against different bounds from the ones the vector was optimized in, which is
    # how a stored vector comes back "outside this problem's bounds".
    sf_kwargs = dict(grid.get("sf_kwargs") or {})
    sf_kwargs.setdefault("nseg", int(grid.get("nseg") or S.solve_nseg))
    # And the thrust the cell settled on: its per-segment ceilings and Isps and its leg operating
    # point. The vector only reproduces the cell's trajectory against those; rebuilt against terms
    # derived afresh it flies a different thrust history and reads as not converged while the grid
    # beside it still shows the cell closing.
    sf_kwargs.update({k: v for k, v in terms.items() if v is not None})
    # Rebuild against the cell's own narrow bounds rather than the mission-wide window. A cell's
    # converged epoch may sit inside its pin and outside a day-rounded window edge, and the wider
    # bounds then reject a vector that is perfectly valid for the problem it was solved in.
    tol = float(grid.get("cell_tol_days") or 0.75)
    pin_date = figures.mjd2000_to_datetime(dep_m).date()
    sf_kwargs.update({
        "launch_window": (pin_date, pin_date),
        "window_slack_days": tol + abs(dep_m - _to_mjd2000(np.datetime64(pin_date))),
        "min_tof_days": max(1.0, tof - tol), "max_tof_days": tof + tol,
    })
    # The row the GRID solved against, re-osculated at the arrival era. Using the focus row would
    # rebuild against different elements than the vector was optimized on.
    target_row = (grid.get("target_row")
                  or jobs.read_grid_job(S.grid_run_id).get("target") or dict(S.focus or {}))
    try:
        result = pipeline.result_from_decision(rc, target_row, np.asarray(x, float), **sf_kwargs)
        # Recorded as the sweep records its winning point: the caller writes the result, summary
        # and terminal status, so the run lands in the same channel an interactive solve does. The
        # narrow bounds are dropped from what gets recorded, since a launch_window is a pair of
        # dates and the spec is JSON, and the pin is reconstructible from the epochs beside it
        # anyway. Recording it raised "Object of type date is not JSON serializable" on every tick.
        recorded = {k: v for k, v in sf_kwargs.items()
                    if k not in ("launch_window", "min_tof_days", "max_tof_days",
                                 "window_slack_days")}
        run_id = jobs.record_run(rc.model_dump(mode="json"), target_row, dep_m, dep_m + tof,
                                 label=f"{S.study_name} · grid cell", options=recorded)
        jobs.write_result(run_id, result)
        jobs.write_summary(run_id, products.run_summary(result))
        jobs.write_status(run_id, state=jobs.DONE, stage="done", progress=1.0,
                          message=f"grid cell · {tof:.0f} d flight")
    except Exception as exc:  # noqa: BLE001  (a failure here leaves the previous run shown)
        ui.notify(f"Could not rebuild that cell: {exc}", type="warning")
        return
    S.solve_run_id = run_id



def _diagnostics_view(rc) -> None:
    res = _cruise_result(S.solve_run_id)
    if res is None:
        _empty("insights", "Converge a cruise solve first.")
        return
    block = res["sf"]
    tgt = res["target"]
    if "node_thrust_normal" not in block:
        _empty("insights", "This run carries no per-node thrust decomposition to chart.")
        return
    # Only chart the return if its optimization converged (a flyable arc), matching the plot gate.
    ret = res["return"] if _return_plottable(res) else None
    # One leg is a figure of stacked panels on a shared clock, and it fills the canvas the way it
    # always has. Two legs each get a band and the column scrolls between them.
    height = 760 if ret else None
    with ui.column().classes("w-full").style(
            "flex:1 1 0;min-height:0;overflow-y:auto;gap:.6rem;padding-right:.2rem"):
        _fig_card(f"Outbound · Earth → {tgt['name']}", "trending_up",
                  figures.trajectory_diagnostics(
                      block["node_times_days"], block["throttle"], block["node_thrust_radial"],
                      block["node_thrust_transverse"], block["node_thrust_normal"],
                      block["node_i_deg"], block["node_a_au"], block["node_e"],
                      speed_kms=block.get("node_speed_kms"),
                      earth_speed_dep=block.get("earth_speed_dep_kms"),
                      target_speed_arr=block.get("target_speed_arr_kms"), target_a=tgt["a"],
                      target_e=tgt["e"], target_i=tgt["i"], target_name=tgt["name"],
                      max_throttle_pct=_available_thrust_fraction(rc) * 100.0,
                      flyby_day=(block["flyby"]["tof1_days"] if block.get("flyby") else None),
                      flyby_name=str((block.get("flyby") or {}).get("body", "")).capitalize()),
                  "diag-sf", height=height)
        # The range-to-Earth/target/Sun profile has moved to the Mission profile sub-tab, where it
        # sits on the whole-mission clock beside the power and engine timelines.
        if ret and "node_thrust_normal" in ret["sf"]:
            rb = ret["sf"]
            dest = ret.get("destination_name") or ret.get("destination") or "destination"
            _fig_card(f"Return · {tgt['name']} → {dest}", "trending_down",
                      figures.trajectory_diagnostics(
                          rb["node_times_days"], rb["throttle"], rb["node_thrust_radial"],
                          rb["node_thrust_transverse"], rb["node_thrust_normal"],
                          rb["node_i_deg"], rb["node_a_au"], rb["node_e"],
                          speed_kms=rb.get("node_speed_kms"),
                          earth_speed_dep=rb.get("earth_speed_dep_kms"),
                          target_speed_arr=rb.get("target_speed_arr_kms"), target_name=dest,
                          max_throttle_pct=_available_thrust_fraction(rc) * 100.0),
                      "diag-return", height=height)



def _grid_thrust_isp(rc, power_W):
    """Available thrust (mN) and Isp (s) at input power(s), off the stack's throttle curve.

    Vectorized over a power array via the config's ``(power_W, thrust_N, isp_s)`` grid; falls
    back to the rated point (flat) when the stack exposes no grid. The whole-mission engine
    plot reads its available-thrust ceiling and its Isp from here."""
    p = np.asarray(power_W, float)
    grid = rc.power_thrust_isp_grid()
    if grid is None:
        return np.full(p.shape, rc.total_thrust_mN), np.full(p.shape, rc.effective_isp)
    pw, thrust_N, isp_s = grid
    capped = np.minimum(p, pw[-1])
    thrust_mN = np.interp(capped, pw, thrust_N) * 1e3
    isp = np.interp(capped, pw, isp_s)
    # Where the power cannot run the stack at all there is no operating point, so there is no
    # Isp to draw: leave a gap rather than a line at whatever the grid records for zero thrust.
    isp = np.where(thrust_mN > 0.0, isp, np.nan)
    return thrust_mN, isp


def _array_effect_pcts(am, sun_au, r_geo_km=None):
    """The two sun-driven array effects at heliocentric distance(s), each as a percent of its
    1-AU value: ``(irradiance_pct, cell_efficiency_pct)``.

    These are the physically distinct halves of :meth:`ArrayModel.power_fraction` (whose value
    is their product / 1e4): the 1/r^2 IRRADIANCE (a sunlight-availability multiplier, above
    100% sunward), and the CELL EFFICIENCY vs temperature relative to 1 AU (a true efficiency,
    above 100% far out where the cells run cold). ``r_geo_km`` adds the near-Earth albedo/IR
    heating to the temperature (escape only)."""
    r = np.asarray(sun_au, float)
    irr_pct = 1.0 / r ** 2 * 100.0
    eta = np.asarray(am.cell_efficiency(am.panel_temperature_C(r, r_geo_km)), float)
    eta1 = float(am.cell_efficiency(am.panel_temperature_C(1.0)))
    return irr_pct, eta / eta1 * 100.0


def _array_thermal(am, sun_au, r_geo_km=None):
    """The panel's temperature (degC) and absolute cell efficiency (% of sunlight converted) at
    heliocentric distance(s), with the near-Earth albedo/IR heating when ``r_geo_km`` is given:
    the two quantities the mission profile's thermal panel draws."""
    temp_C = np.asarray(am.panel_temperature_C(np.asarray(sun_au, float), r_geo_km), float)
    return temp_C, np.asarray(am.cell_efficiency(temp_C), float) * 100.0


def _escape_sun_distance_au(positions_km, times_days):
    """Per-sample heliocentric distance (AU) over the geocentric escape spiral, reusing the
    spiral's own circular-Sun model, so the escape's sunlight and temperature split is computed
    on the same geometry it flew (no need to store it in the result)."""
    from prospector.constants import AU_M
    from prospector.solvers import spiral as _sp
    t = np.asarray(times_days, float) * 86400.0
    sun = np.array([_sp._sun_position(float(s)) for s in t], float)     # equatorial, metres
    sc = np.asarray(positions_km, float) * 1e3
    return np.linalg.norm(sun - sc, axis=1) / AU_M


def _segment_engine_series(rc, sf: dict, days):
    """The engine allowance a cruise leg was solved under, per fine sample: each segment's thrust
    ceiling (its cap times the leg's maximum thrust) and the Isp of its operating point, held
    across the segment, plus the thrust the solve actually flew there.

    Drawn per segment rather than from the instantaneous array power because that is what the
    optimizer saw: a Sims-Flanagan segment fires at one constant thrust, limited to the MEAN of
    what the array can supply across the segment. Sampled finely instead, the ceiling drops to
    zero the instant the array crosses the thruster's floor while the segment is still firing,
    which reads as thrust from nothing. Returns ``(avail_mN, isp_s, flown_mN)``; the Isp is NaN
    where the ceiling is zero (no operating point), and ``flown_mN`` None without a throttle
    history."""
    days = np.asarray(days, float)
    op_thrust_mN = float(sf.get("thrust_N") or rc.total_thrust_mN * 1e-3) * 1e3
    leg_isp = float(sf.get("isp_s") or rc.effective_isp)
    nseg = int(sf.get("nseg") or 0)
    tof = float(sf.get("tof_days") or (days[-1] if days.size else 0.0))
    caps = sf.get("seg_caps")
    if nseg <= 0 or caps is None or tof <= 0.0:
        avail = np.full(days.size, op_thrust_mN)
        isp = np.full(days.size, leg_isp)
    else:
        idx = np.clip(np.floor(days / tof * nseg).astype(int), 0, nseg - 1)
        # Stored caps carry the duty-cycle limit as well (fractions of maximum thrust, the
        # throttle's basis); the array's ceiling alone is what "available" means here.
        duty = max(float(sf.get("max_duty_cycle") or 1.0), 1e-9)
        caps = np.minimum(np.asarray(caps, float) / duty, 1.0)
        avail = caps[idx] * op_thrust_mN
        seg_isp = sf.get("seg_isp_s")
        isp = (np.asarray(seg_isp, float)[idx] if seg_isp is not None
               else np.full(days.size, leg_isp))
        isp = np.where(avail > 0.0, isp, np.nan)
    throttle = np.asarray(sf.get("fine_throttle", []), float)
    flown = throttle * op_thrust_mN if throttle.size == days.size else None
    return avail, isp, flown


def _cruise_sun_series(rc, block, am, teff, reserve: float = 0.0):
    """The per-sample sun distance and the two array efficiency factors over a cruise/return leg.

    The array is clear of the belts in cruise, so its belt degradation is fixed at the
    post-escape value while only the sun-distance factor (:meth:`ArrayModel.power_fraction`)
    still varies. Returns ``(days, sun_au, deg_factor, sun_frac)`` on the leg's own 0-based
    clock, keeping the two efficiency effects SEPARATE: ``deg_factor`` is the frozen belt
    degradation (a scalar, 1.0 = no loss), ``sun_frac`` is the per-sample sun-distance
    factor, uncapped, so above 1 closer in than 1 AU and below it further out. The caller forms
    the raw array output as ``nameplate x deg_factor x sun_frac`` and the power at the thrusters as
    ``nameplate x deg_factor x sun_frac x teff``; what the thrusters can use of that is capped at
    their rated draw by the throttle curve. ``deg_factor`` is None when the vehicle carries no
    sized array (power-rich)."""
    days = np.asarray(block.get("fine_times_days", []), float)
    pos = np.asarray(block.get("fine_positions_au", []), float)
    if not len(days) or pos.ndim != 2:
        return np.empty(0), np.empty(0), None, np.empty(0)
    sun_au = np.linalg.norm(pos, axis=1)
    sun_frac = np.asarray(am.power_fraction(sun_au), float)
    bol = float(rc.vehicle.solar_power_W)
    thruster_base = _available_cruise_power_W(rc)     # at-thruster power the solve flew on
    if thruster_base is not None and bol > 0.0:
        # Belt degradation reflected in the flown power: the at-thruster figure back through the
        # chain, plus the housekeeping the bus took first, over the nameplate.
        deg_factor = (thruster_base / teff + reserve) / bol
    elif bol > 0.0:
        deg_factor = 1.0                              # LV escape / power-rich: no belt loss
    else:
        return days, sun_au, None, sun_frac
    return days, sun_au, float(deg_factor), sun_frac


def _mission_profile_view(rc) -> None:
    """Whole-mission time series on one clock: effective array power, engine performance, and the
    ranges to Earth, the target and the Sun, with escape, cruise and return stitched together
    with dashed markers at the phase seams.

    Each phase contributes what it flew: the escape spiral's per-sample radiation damage
    x near-Earth heating, the cruise/return's frozen post-escape power breathing with heliocentric
    distance. It answers how power, thrust, Isp and geometry change across the mission, which is
    the
    story the individual 3D views can't show."""
    res = _cruise_result(S.solve_run_id)
    if res is None:
        _empty("stacked_line_chart", "Converge a cruise solve first.")
        return
    from prospector.spacecraft import buildability
    bm = buildability.load_bus_model(name=rc.build_model)
    am = bm.array_model()
    teff = rc.thruster_chain_eff(bm)
    reserve = rc.array_reserve_W(bm)                # array output the bus keeps for housekeeping
    bol = float(rc.vehicle.solar_power_W)

    span = _timeline_span(rc)
    esc_days = float(span["esc_days"])
    ret = res["return"] if _return_plottable(res) else None
    cruise_end = esc_days + float(span["cruise_days"])
    ret_start = cruise_end + float(span["stay_days"])
    transitions = [(esc_days, "Earth departure")] if esc_days > 0 else []
    fb = res["sf"].get("flyby")
    if fb:
        transitions.append(flyby_marker(esc_days + float(fb["tof1_days"]),
                                        str(fb.get("body", "")).capitalize()))
    if ret:
        transitions.append((ret_start, "return"))

    # The escape spiral leg, if one flew this session. Power at each point is the new-array power
    # times the radiation damage, the sun-distance and temperature factor, and the conversion
    # losses.
    run_id = state.session_spiral_run(rc)
    esc_sp = (jobs.read_spiral_result(run_id) or {}).get("spiral") if run_id else None

    power_phases, engine_phases = [], []
    dist_days = dist_sc = dist_earth = dist_tgt = dist_sun = None

    if esc_sp is not None and bol > 0.0:
        et = np.asarray(esc_sp.get("times_days", []), float)
        deg = np.asarray(esc_sp.get("power_fraction", []), float)          # belt degradation
        sun = np.asarray(esc_sp.get("power_fraction_sun", []), float)      # sun distance & temp
        pos = np.asarray(esc_sp.get("positions_km", []), float)
        if sun.size != et.size:
            sun = np.ones(et.size)
        if deg.size == et.size and et.size:
            esc_array = bol * deg * sun               # array raw output (belt deg x sun/thermal)
            esc_thruster = np.maximum(esc_array - reserve, 0.0) * teff   # less housekeeping, through the chain
            # Split the escape's sun factor into its irradiance and cell-temperature halves on
            # the flown geometry (recovers the heliocentric distance + the near-Earth heating).
            if pos.ndim == 2 and len(pos) == et.size:
                esc_r = _escape_sun_distance_au(pos, et)
                geo_km = np.linalg.norm(pos, axis=1)
                irr_pct, cell_pct = _array_effect_pcts(am, esc_r, geo_km)
                temp_C, eff_pct = _array_thermal(am, esc_r, geo_km)
            else:
                irr_pct = cell_pct = np.full(et.size, 100.0)
                temp_C, eff_pct = _array_thermal(am, np.ones(et.size))
            power_phases.append(("Earth escape", et, esc_array, esc_thruster,
                                 deg * 100.0, irr_pct, cell_pct, temp_C, eff_pct))
            # Thrust and Isp track the delivered power down the throttle curve; the spiral flies
            # tangentially at the duty cycle, so the flown thrust is that fraction of the available
            # (eclipse outages, which briefly gate it to zero, are not drawn here).
            avail_mN, esc_isp = _grid_thrust_isp(rc, esc_thruster)
            duty = float((jobs.read_spiral_job(run_id).get("options") or {}).get("duty_cycle", 1.0))
            engine_phases.append(("Earth escape", et, avail_mN, avail_mN * duty, esc_isp))

    # The outbound cruise leg. Belt degradation is frozen post-escape (a scalar); only the
    # sun-distance effects vary. The raw array output is unclamped (rises above nameplate sunward);
    # the thruster power carries the delivery limit the solve flew under.
    c_days, c_sun, c_deg, c_sunfrac = _cruise_sun_series(rc, res["sf"], am, teff, reserve)
    if len(c_days):
        off = esc_days
        if c_deg is not None:
            c_array = bol * c_deg * c_sunfrac
            # Power at the thrusters follows the Sun both ways; what they can use of it is capped
            # at their rated draw on the throttle curve, not here.
            c_thruster = np.maximum(bol * c_deg * c_sunfrac - reserve, 0.0) * teff
            deg_pct = np.full(c_days.size, c_deg * 100.0)
            irr_pct, cell_pct = _array_effect_pcts(am, c_sun)
            temp_C, eff_pct = _array_thermal(am, c_sun)
            power_phases.append(("cruise", c_days + off, c_array, c_thruster,
                                 deg_pct, irr_pct, cell_pct, temp_C, eff_pct))
            avail_mN, isp, demand_mN = _segment_engine_series(rc, res["sf"], c_days)
            engine_phases.append(("cruise", c_days + off, avail_mN, demand_mN, isp))
        # Distances ride the same mission clock (offset past the escape phase).
        orbits = res.get("orbits") or {}
        if "earth_track_au" in orbits:
            dist_days = c_days + off
            dist_sc = np.asarray(res["sf"]["fine_positions_au"], float)
            dist_earth = np.asarray(orbits["earth_track_au"], float)
            dist_tgt = np.asarray(orbits["target_track_au"], float)
            dist_sun = c_sun

    # The laden return leg, when it converged.
    if ret:
        r_days, r_sun, r_deg, r_sunfrac = _cruise_sun_series(rc, ret["sf"], am, teff, reserve)
        if len(r_days) and r_deg is not None:
            r_array = bol * r_deg * r_sunfrac
            r_thruster = np.maximum(bol * r_deg * r_sunfrac - reserve, 0.0) * teff
            deg_pct = np.full(r_days.size, r_deg * 100.0)
            irr_pct, cell_pct = _array_effect_pcts(am, r_sun)
            temp_C, eff_pct = _array_thermal(am, r_sun)
            power_phases.append(("return", r_days + ret_start, r_array, r_thruster,
                                 deg_pct, irr_pct, cell_pct, temp_C, eff_pct))
            avail_mN, isp, demand_mN = _segment_engine_series(rc, ret["sf"], r_days)
            engine_phases.append(("return", r_days + ret_start, avail_mN, demand_mN, isp))

    # The three timelines stack taller than the canvas, so host them in a scrollable column (the
    # canvas itself is overflow:hidden). Each card gets a fixed height so they are evenly spaced
    # and the column scrolls through them.
    with ui.column().classes("w-full").style(
            "flex:1 1 0;min-height:0;overflow-y:auto;gap:.6rem;padding-right:.2rem"):
        if power_phases:
            _fig_card("Array power", "wb_sunny",
                      figures.mission_power_timeline(power_phases, transitions=transitions,
                                                     nameplate_W=(bol if bol > 0 else None)),
                      "mp-power", height=540)
        else:
            ui.label("No sized solar array on this design, so there is no power "
                     "profile.").style(f"color:{MUTED};font-size:.78rem")
        if engine_phases:
            _fig_card("Engine performance", "bolt",
                      figures.engine_performance_timeline(engine_phases, transitions=transitions,
                                                          rated_thrust_mN=rc.total_thrust_mN),
                      "mp-engine", height=360)
        if dist_days is not None:
            tgt = res["target"]
            _fig_card("Range", "straighten",
                      figures.distance_profile(dist_days, dist_sc, dist_earth, dist_tgt,
                                               sun_au=dist_sun, transitions=transitions,
                                               target_name=tgt["name"]),
                      "mp-range", height=320)


# ======================================================================================
# right rail: the active sub-tab's controls
# ======================================================================================

@ui.refreshable
def _rail() -> None:
    rc = _resolve()
    if rc is None:
        return
    if S.flight_tab == "Earth escape":
        _escape_rail(rc)
    elif S.flight_tab == "Trajectory":
        _trajectory_rail(rc)
    elif S.flight_tab == "Mission profile":
        section("Mission profile", "stacked_line_chart")
    else:
        section("Diagnostics", "insights")


# ---- ① escape rail ----

def _escape_rail(rc) -> None:
    section("Leaving Earth", "north_east")
    if rc.launch.escape_provided:
        ui.label(f"{rc.launch.name} provides escape - v∞ is free to the cruise.").style(
            f"color:{MUTED};font-size:.74rem")
        _vinf_slider(rc)
        return

    _vinf_slider(rc)
    labeled_slider("Thruster on-time", 10, 100, float(S.launch_duty), 5, "{:.0f}", " %",
                   on_change=lambda e: _set_launch(duty=e.value))
    labeled_slider("Max spiral time", 0.5, 15.0, float(S.launch_years), 0.5, "{:.1f}", " yr",
                   on_change=lambda e: _set_launch(years=e.value))
    ui.switch("Steer the orbit plane (advanced)", value=S.launch_steer,
              on_change=lambda e: _set_steer(e.value)).props("dense")
    if S.launch_steer:
        labeled_slider("Target inclination", 0.0, 180.0,
                       float(S.launch_target_inc if S.launch_target_inc is not None
                             else rc.launch.inclination_deg or 0.0), 0.5, "{:.1f}", "°",
                       on_change=lambda e: _set_launch(target_inc=e.value))

    _radiation_controls()

    run_id = S.launch_run_id
    running = run_id and not jobs.is_terminal(jobs.read_spiral_status(run_id))
    if running:
        _spiral_progress(run_id)
        return

    ui.button("Run escape spiral", icon="play_arrow", on_click=_run_spiral).props(
        "unelevated no-caps").classes("w-full").style("margin-top:.5rem").tooltip(
        "Propagates the climb-out, including belt wear and eclipses.")
    ui.label(f"From {rc.launch.name} · {rc.launch.perigee_alt_km:.0f}×"
             f"{rc.launch.apogee_alt_km:.0f} km · i {rc.launch.inclination_deg:.1f}°").style(
        f"color:{MUTED};font-size:.7rem;margin-top:.3rem")
    if run_id and jobs.is_terminal(jobs.read_spiral_status(run_id)):
        ui.button("Delete run", icon="delete", on_click=_delete_spiral).props(
            "flat dense no-caps").style(f"color:{MUTED};font-size:.7rem")


def _radiation_controls() -> None:
    """The solar-array radiation scenario for the escape spiral: which belt-degradation model to
    fly and the cell cover, meaning its thickness and density, which shield by areal density.
    Collapsed by default. Changing any of these restages the spiral (they join the escape
    fingerprint), so the budget falls back to the analytic estimate until it is re-run."""
    models = load_radiation_models()
    if not models:
        return
    with ui.expansion("Solar-array radiation (advanced)", icon="solar_power").classes(
            "w-full").props("dense"):
        ui.select({k: m.name for k, m in models.items()}, value=S.launch_radiation_model,
                  label="Degradation model",
                  on_change=lambda e: _set_launch(rad_model=e.value)).props(
            "dense outlined options-dense").classes("w-full").tooltip(
            "How fast the belts degrade the array. AP8-MIN is the worst case for a low SSO, "
            "AP8-MAX is gentler.")
        # Cover thickness + density -> areal density (g/cm^2), the true proton-shielding variable.
        # Numbers (not sliders) so a real multi-layer stack (e.g. 212.5 um at 1.64 g/cm^3) is
        # exact.
        with ui.row().classes("w-full no-wrap gap-2"):
            ui.number("Cover thickness", value=float(S.launch_coverglass_um), min=10, step=5,
                      suffix="µm", on_change=lambda e: _set_launch(coverglass=e.value)).props(
                "dense outlined").classes("flex-grow").tooltip(
                "Total cover thickness over the cell, the main shielding lever.")
            ui.number("Cover density", value=float(S.launch_coverglass_density), min=0.5, step=0.05,
                      suffix="g/cm³", on_change=lambda e: _set_launch(coverglass_density=e.value)).props(
                "dense outlined").classes("flex-grow").tooltip(
                "Mean density of the cover layers. Shielding goes with thickness times density.")
        cur = models.get(S.launch_radiation_model)
        if cur is not None:
            areal = float(S.launch_coverglass_um) * float(S.launch_coverglass_density) * 1e-4
            ui.label(f"{cur.cell} · cover {areal * 1e3:.1f} mg/cm²").style(
                f"color:{MUTED};font-size:.68rem")


def _vinf_slider(rc) -> None:
    labeled_slider("Departure speed v∞", 0.0, 6.0, float(S.departure_vinf), 0.25, "{:.2f}", " km/s",
                   on_change=lambda e: _set_launch(vinf=e.value)).tooltip(
        "Speed left over after escape, relative to Earth. The cruise gets it for free.")


def _spiral_progress(run_id: str) -> None:
    status = jobs.read_spiral_status(run_id)
    stage = status.get("stage") or status.get("state", "")
    ui.linear_progress(value=_clip(status.get("progress", 0.0)), show_value=False).props(
        "rounded").style("margin-top:.5rem")
    ui.label(f"spiral · {stage} - {status.get('message', '')}").style(
        f"color:{MUTED};font-size:.72rem")
    ui.button("Dismiss", on_click=_dismiss_spiral).props("flat dense no-caps").style(
        f"color:{MUTED};font-size:.7rem")


# ---- ② trajectory rail (porkchop + cruise) ----

def _trajectory_rail(rc) -> None:
    _porkchop_controls(rc)
    ui.separator().style(f"background:{BORDER}")
    _cruise_controls(rc)
    if rc.mission.return_trip:
        ui.separator().style(f"background:{BORDER}")
        _return_controls(rc)


def _return_controls(rc) -> None:
    """Return-leg controls: the destination + its solver knobs. The return is auto-chained into
    the cruise job with no separate Run, since these controls feed the same submit, so this
    section
    reads the planned return and, once solved, its converged figures."""
    section("Return trip", "keyboard_return")
    res = _cruise_result(S.solve_run_id)
    # Destination picker (mission definition; editing it reprices the round-trip budget).
    dests = state.return_destinations()
    options = {k: v.name for k, v in dests.items()}
    ui.select(options, value=rc.mission.return_destination, label="Destination",
              on_change=lambda e: _set_return_dest(e.value)).props(
        "dense outlined options-dense").classes("w-full")
    labeled_slider("Return max flight time", 200, 2000,
                   float(rc.mission.return_max_tof_days or 900), 50, "{:.0f}", " d",
                   on_change=lambda e: _set_return(max_tof=e.value))
    labeled_slider("Return max duty cycle", 10, 100, float(S.return_duty_pct), 5, "{:.0f}", " %",
                   on_change=lambda e: _set_return(duty=e.value))
    labeled_slider("Return segments (fidelity)", 10, 40, int(S.return_nseg), 1, "{:.0f}", "",
                   on_change=lambda e: _set_return(nseg=e.value))
    ui.label(f"Payload collected: {rc.mission.asteroid_payload_mass:.0f} kg · "
             f"stay {float(rc.mission.time_at_asteroid or 0.0):.0f} d "
             f"(set in Project)").style(f"color:{MUTED};font-size:.7rem")

    if res is None:
        ui.label("The return flies automatically after the cruise converges.").style(
            f"color:{MUTED};font-size:.72rem;margin-top:.2rem")
        return
    ret = res.get("return")
    if not ret:
        return
    if "error" in ret:
        ui.label(f"⚠ Return could not be planned: {ret['error']}").style(
            f"color:{AMBER};font-size:.72rem")
        return
    rb = ret["sf"]
    ok = _converged(rb["feasible"], rb["mismatch"]) and ret["feasible_propellant"]
    color = GREEN if ok else AMBER
    note = "" if ret["feasible_propellant"] else " · ⚠ over propellant budget"
    ui.label(f"{'✓' if ok else '⚠'} ΔV {rb['dv_kms']:.1f} · {rb['tof_days']:.0f} d · "
             f"−{ret['total_return_propellant_kg']:.0f} kg prop · delivers "
             f"{ret['delivered_mass_kg']:.0f} kg{note}").style(f"color:{color};font-size:.72rem")


def _porkchop_controls(rc) -> None:
    """The transfer-grid controls: run it, choose what the field shows, choose how dense it is.

    An explicit Run, not an inline compute. Every cell is a real Sims-Flanagan trajectory, which
    costs tens of seconds for a grid. The instant surface it replaces could only ever have been an
    approximation, and measured against real solves it moved the wrong way against them.
    """
    section("Transfer options", "grid_on")
    target = state.S.focus
    grid = _grid_result()
    status = jobs.read_grid_status(S.grid_run_id) if S.grid_run_id else {}
    running = bool(S.grid_run_id) and not jobs.is_terminal(status)

    if running:
        done, total = status.get("cells_done") or 0, status.get("cells_total") or 0
        ui.label(status.get("message") or "solving the grid...").style(
            f"color:{ACCENT};font-size:.74rem")
        ui.linear_progress(value=float(status.get("progress") or 0.0), show_value=False).props(
            "rounded").classes("w-full")
        if total:
            ui.label(f"{done}/{total} cells").style(f"color:{MUTED};font-size:.68rem")
        return

    if not (isinstance(target, dict) and _has_full_elements(target)):
        ui.label("No target in focus - pick one in Find targets.").style(
            f"color:{MUTED};font-size:.74rem")
        return

    if grid is not None:
        n_ok, n = grid.get("n_feasible") or 0, grid.get("n_cells") or 0
        ui.label(f"✓ {n_ok}/{n} cells solved for {grid.get('target_name')}").style(
            f"color:{GREEN};font-size:.74rem")
        if grid.get("restarts") and int(grid["restarts"]) <= 1:
            # A single-restart grid finds which cells fly but reports its neighbour's answer, not
            # cell's, so its numbers are not ΔV claims. Saying so beats letting them be read as
            # one.
            ui.label("Feasibility pass only - re-run without 'fast' for ΔV").style(
                f"color:{AMBER};font-size:.68rem")
    else:
        name = target.get("full_name") or target.get("pdes")
        ui.label(f"Focus: {name}").style(f"color:{MUTED};font-size:.74rem")
        if status.get("state") == jobs.ERROR:
            ui.label("The last grid failed - see the log below.").style(
                f"color:{AMBER};font-size:.7rem")

    with ui.row().classes("items-center no-wrap w-full gap-2").style("margin-top:.3rem"):
        ui.select({"margin": "propellant margin", "dv": "ΔV", "mass": "delivered mass"},
                  value=S.grid_colour,
                  on_change=lambda e: _set_grid(colour=e.value)).props(
            "dense outlined").style("flex:1 1 0;min-width:0").tooltip(
            "What the grid colours by. Propellant margin is the kilograms left in the tank after "
            "the cruise; delivered mass and ΔV are the same trajectories read differently. In "
            "every view the same colours mean the same thing: hot is good, and grey is a real "
            "trajectory this vehicle cannot fly, short of propellant or over the ΔV the tank "
            "holds after escape.")
        ui.switch("fast", value=S.grid_fast,
                  on_change=lambda e: _set_grid(fast=e.value)).props("dense").tooltip(
            "One restart per cell: shows which cells fly, but does not quote ΔV.")
    with ui.row().classes("items-center no-wrap w-full gap-2").style("margin-top:.2rem"):
        ui.number("departures", value=int(S.grid_n_dep), min=2, max=24, step=1,
                  on_change=lambda e: _set_grid(n_dep=e.value)).props(
            "dense outlined").style("flex:1 1 0;min-width:0").tooltip(
            "Cells across the departure window. Cheap to widen, these run in parallel.")
        ui.number("flight times", value=int(S.grid_n_tof), min=2, max=24, step=1,
                  on_change=lambda e: _set_grid(n_tof=e.value)).props(
            "dense outlined").style("flex:1 1 0;min-width:0").tooltip(
            "Steps along flight time. These run one after another, so this is what costs wall "
            "clock.")
    ui.button("Re-run grid" if grid is not None else "Run transfer grid",
              icon="grid_on", on_click=lambda: _run_grid(rc)).props(
        "unelevated no-caps").classes("w-full").style("margin-top:.4rem")


def _set_grid(**kw) -> None:
    """Record a grid control. Only the colour changes anything already on screen.

    The rail is never rebuilt from here. These handlers fire on every keystroke, and rebuilding
    the rail replaces the very input being typed into, so the box loses focus after one character.
    The axis counts and the fast switch are read when Run is pressed and are drawn nowhere, so
    they need no redraw at all; the colour is what the field is drawn by, so it redraws the canvas.
    """
    if "colour" in kw and kw["colour"]:
        S.grid_colour = str(kw["colour"])
        _canvas.refresh()
    if "fast" in kw:
        S.grid_fast = bool(kw["fast"])
    # Clamped to the range the inputs offer. A part-typed number arrives here as its own value, so
    # the clamp holds the state sane while the box keeps whatever is being typed.
    if kw.get("n_dep"):
        S.grid_n_dep = int(min(24, max(2, int(kw["n_dep"]))))
    if kw.get("n_tof"):
        S.grid_n_tof = int(min(24, max(2, int(kw["n_tof"]))))


def _grid_result() -> dict | None:
    """The transfer grid in force: finished, readable, and solved for the config now live.

    A grid is a picture of one vehicle flying to one body, so a grid whose config no longer matches
    describes a mission that no longer exists, so it clears rather than lingering behind a warning,
    the same discipline a stale cruise gets.
    """
    rc = _resolve()
    if rc is None:
        return None
    return products.grid_block(S.grid_run_id, rc, _available_cruise_power_W(rc),
                               target_pdes=str((S.focus or {}).get("pdes") or ""))


def _run_grid(rc_arg) -> None:
    """Submit the transfer grid as a detached job. Tens of seconds, so it streams progress rather
    than blocking the workspace."""
    rc = _resolve()
    if rc is None:
        ui.notify("The configuration does not resolve - fix it in Project.", type="warning")
        return
    target = state.S.focus
    if not (isinstance(target, dict) and _has_full_elements(target)):
        ui.notify("No target in focus - pick one in Find targets.", type="warning")
        return
    from prospector.trades.pipeline import grid as G
    # The grid's cells have to run under the same solver terms the cruise does: the departure and
    # arrival v-infinity, the duty cap, the window slack. Leaving them to the solver's own defaults
    # made every cell solve at a 1 km/s departure excess regardless of the knob, so the surface was
    # a picture of a different mission and its stored vectors could not be rebuilt under the
    # config.
    options = {k: v for k, v in _solver_options(rc, 300.0).items()
               if k not in ("n_starts", "starts_workers", "max_tof_days", "nseg", "x0",
                            "return_options")}
    options.update({
        "n_dep": int(S.grid_n_dep), "n_tof": int(S.grid_n_tof),
        "nseg": int(S.solve_nseg),
        "restarts": G.FAST_RESTARTS if S.grid_fast else G.GRID_RESTARTS,
        # No polish on a feasibility pass: it sharpens a ΔV the fast grid does not claim.
        "polish": not S.grid_fast,
    })
    S.grid_run_id = jobs.submit_grid(rc.model_dump(mode="json"), dict(target), options=options)
    S.grid_sel = None
    _refresh_all()


def _cruise_controls(rc) -> None:
    section("Cruise to the target", "rocket")
    grid = _grid_result()
    if grid is None:
        ui.label("Run the transfer grid first.").style(f"color:{MUTED};font-size:.74rem")
        return
    run_id = S.solve_run_id
    if run_id and not jobs.is_terminal(jobs.read_status(run_id)):
        _cruise_progress(run_id)
        return

    dep_m, tof = _seed_cell(grid)
    _selected_cell_line(grid, dep_m, tof)
    duty = labeled_slider("Max duty cycle", 10, 100, float(S.solve_duty_pct), 5, "{:.0f}", " %",
                          on_change=lambda e: _set_solve(duty=e.value))
    duty.tooltip("The most of any one segment the engine may fire for, so it caps on-time "
                 "rather than thrust.")
    power = _available_cruise_power_W(rc)
    if power is not None:
        thrust_N, isp_s = rc.thrust_isp_at_power(power)
        rated_mN, rated_isp = rc.total_thrust_mN, rc.effective_isp
        tfrac = (thrust_N * 1e3) / rated_mN if rated_mN else 1.0
        if tfrac < 0.999 or isp_s < rated_isp - 1.0:
            isp_clause = (f", Isp {isp_s:.0f} s (vs {rated_isp:.0f} rated)"
                          if isp_s < rated_isp - 1.0 else "")
            ui.label(f"⚡ Power-limited at 1 AU: {thrust_N * 1e3:.0f} mN "
                     f"({tfrac * 100:.0f}% of rated){isp_clause}; the array buys more nearer "
                     f"the Sun, up to rated, and less further out.").style(
                f"color:{AMBER};font-size:.72rem;font-weight:600")
    window_days = max(200, (rc.mission.arrive_by - rc.departure_window[0]).days)
    default_tof = int(min(window_days, max(150, round(tof) + 100)))
    if S.solve_max_tof_days is None:
        S.solve_max_tof_days = float(default_tof)
    labeled_slider("Max flight time", 150, int(window_days), float(S.solve_max_tof_days), 30,
                   "{:.0f}", " d", on_change=lambda e: _set_solve(max_tof=e.value))
    labeled_slider("Segments (fidelity)", 10, 40, int(S.solve_nseg), 1, "{:.0f}", "",
                   on_change=lambda e: _set_solve(nseg=e.value))
    labeled_slider("Search starts", 1, 16, int(S.solve_starts), 1, "{:.0f}", "",
                   on_change=lambda e: _set_solve(starts=e.value)).tooltip(
        "Try this many starting guesses in parallel and keep the best.")
    label = "Find best trajectory" if run_id is None else "Re-run from this cell"
    ui.button(label, icon="play_arrow", on_click=lambda: _run_cruise(rc)).props(
        "unelevated no-caps").classes("w-full").style("margin-top:.4rem")

    # The status line for a finished cruise.
    if run_id and jobs.is_terminal(jobs.read_status(run_id)):
        res = jobs.read_result(run_id)
        if jobs.read_status(run_id).get("state") == jobs.ERROR:
            ui.label("⚠ The solve failed.").style(f"color:{RED};font-size:.72rem")
        elif res and _converged(res["sf"]["feasible"], res["sf"]["mismatch"]):
            sf = res["sf"]
            ui.label(f"✓ ΔV {sf['dv_kms']:.1f} · {sf['tof_days']:.0f} d · "
                     f"{sf['propellant_kg']:.0f} kg prop").style(f"color:{GREEN};font-size:.72rem")
            if sf.get("refresh_settled") is False:
                ui.label("⚠ The leg's Isp did not settle: the path implies a different Isp from "
                         "the one it was priced at, so this propellant figure is approximate. "
                         "Re-run.").style(f"color:{AMBER};font-size:.7rem")
        elif res:
            ui.label("⚠ Did not converge - retune and re-run.").style(f"color:{AMBER};font-size:.72rem")
        ui.button("New target / clear", icon="restart_alt", on_click=_clear_solve).props(
            "flat dense no-caps").style(f"color:{MUTED};font-size:.7rem")


def _cruise_progress(run_id: str) -> None:
    status = jobs.read_status(run_id)
    stage = status.get("stage") or status.get("state", "")
    ui.linear_progress(value=_clip(status.get("progress", 0.0)), show_value=False).props(
        "rounded").style("margin-top:.3rem")
    ui.label(f"{stage} - {status.get('message', '')}").style(f"color:{MUTED};font-size:.72rem")
    ui.button("Cancel / dismiss", on_click=_cancel_cruise).props("flat dense no-caps").style(
        f"color:{MUTED};font-size:.7rem")


# ======================================================================================
# bottom dock: the mission timeline (escape vs cruise phases over real dates)
# ======================================================================================

@ui.refreshable
def _timeline() -> None:
    """The mission timeline: an escape→cruise phase track over real dates with a play/pause
    scrubber. A phase that hasn't been solved yet shows GRAY (escape until the spiral is flown,
    cruise until the solve converges); a solved phase shows its color (amber escape, accent
    cruise). The legend dots track that live, so gray reads as 'not solved yet'."""
    global _tl_slider, _tl_label, _tl_span
    rc = _resolve()
    span = _timeline_span(rc) if rc is not None else dict(_tl_span)
    _tl_span = span
    total = span["total_days"]
    esc_pct = (span["esc_days"] / total * 100.0) if total else 0.0
    cruise_pct = ((span["esc_days"] + span["cruise_days"]) / total * 100.0) if total else esc_pct
    stay_pct = ((span["esc_days"] + span["cruise_days"] + span["stay_days"]) / total * 100.0
                if total else cruise_pct)
    escape_solved = rc is not None and (rc.launch.escape_provided
                                        or state.session_spiral_run(rc) is not None)
    cruise_solved = _cruise_result(S.solve_run_id) is not None
    return_planned = rc is not None and rc.mission.return_trip
    return_solved = span.get("return_days", 0.0) > 0.0
    esc_color = AMBER if escape_solved else BORDER
    cruise_color = ACCENT if cruise_solved else BORDER
    stay_color = STAY if return_solved else BORDER
    return_color = RETURN if return_solved else BORDER
    text, color = _tl_text(span, S.timeline_t)
    with ui.element("div").style(
            f"flex:0 0 64px;height:64px;width:100%;display:flex;align-items:center;gap:.6rem;"
            f"background:{PANEL};border:1px solid {BORDER};border-radius:8px;padding:0 14px;"
            f"box-sizing:border-box"):
        ui.button(icon="pause" if S.timeline_playing else "play_arrow",
                  on_click=_toggle_play).props("flat dense round").style(f"color:{ACCENT}")
        ui.button(icon="replay", on_click=lambda: _set_t(0.0)).props("flat dense round").style(
            f"color:{MUTED}")
        with ui.element("div").style(
                "position:relative;flex:1 1 0;height:38px;display:flex;align-items:center"):
            ui.element("div").style(
                f"position:absolute;left:6px;right:6px;top:50%;transform:translateY(-50%);"
                f"height:6px;border-radius:3px;background:linear-gradient(to right,"
                f"{esc_color} 0%,{esc_color} {esc_pct:.1f}%,"
                f"{cruise_color} {esc_pct:.1f}%,{cruise_color} {cruise_pct:.1f}%,"
                f"{stay_color} {cruise_pct:.1f}%,{stay_color} {stay_pct:.1f}%,"
                f"{return_color} {stay_pct:.1f}%,{return_color} 100%)")
            _tl_slider = ui.slider(min=0, max=1, step=0.001, value=S.timeline_t,
                                   on_change=lambda e: _set_t(e.value)).props(
                "thumb-size=16px color=white").classes("pf-timeline absolute").style("left:0;right:0")
        _tl_label = ui.label(text).style(
            f"color:{color};font-size:.78rem;font-family:monospace;min-width:150px;text-align:right")
        with ui.row().classes("items-center no-wrap").style("gap:.5rem"):
            ui.html(f'<span style="color:{esc_color};font-size:.7rem">● escape</span>')
            ui.html(f'<span style="color:{cruise_color};font-size:.7rem">● cruise</span>')
            if return_planned:
                ui.html(f'<span style="color:{stay_color};font-size:.7rem">● stay</span>')
                ui.html(f'<span style="color:{return_color};font-size:.7rem">● return</span>')


def _timeline_span(rc) -> dict:
    """The mission timeline span: liftoff date and per-phase day counts (escape, cruise, asteroid
    stay, return). Uses the converged legs' real durations when they exist, else the planned escape
    + mission window. Stay/return are zero until the return solve produces its block.

    The clock starts at liftoff and runs escape straight into cruise, with nothing in between.
    There is nothing in between because liftoff follows from the solved departure. The launch
    date and the cruise departure are one choice offset by the spiral's duration, so back-dating
    the departure by that duration always lands liftoff inside the launch window (the departure
    window is the launch window shifted). Pinning liftoff to the window open instead would
    manufacture a gap that no vehicle flies: past escape the spacecraft is on its own heliocentric
    orbit and drifting, so it cannot hold position waiting for a later departure."""
    esc_days = float(rc.escape_tof_days)
    stay_days = return_days = 0.0
    res = _cruise_result(S.solve_run_id)
    if res is not None:
        leg = res["sf"]
        dep_date = figures.mjd2000_to_datetime(float(leg["dep_mjd2000"])).date()
        cruise_days = float(leg["tof_days"])
        sched = rc.departure_schedule(dep_date)
        liftoff = sched["liftoff_if_no_coast"] or sched["launch_open"]
        if rc.mission.return_trip and _return_plottable(res):
            stay_days = float(rc.mission.time_at_asteroid or 0.0)
            return_days = float(res["return"]["sf"]["tof_days"])
        return {"liftoff": liftoff, "esc_days": esc_days,
                "cruise_days": cruise_days, "stay_days": stay_days, "return_days": return_days,
                "total_days": esc_days + cruise_days + stay_days + return_days}
    # No cruise yet: liftoff = the mission window open; cruise = the planned span to arrive-by.
    sched = rc.departure_schedule()
    cruise_days = max(0.0, float((rc.mission.arrive_by - sched["cruise_open"]).days))
    return {"liftoff": sched["launch_open"], "esc_days": esc_days,
            "cruise_days": cruise_days, "stay_days": 0.0, "return_days": 0.0,
            "total_days": esc_days + cruise_days}


def _phase_fracs(span: dict) -> dict:
    """The mission clock's phase boundaries, as fractions of the whole span.

    One source for the three scrubbers, which each map the global timeline position onto their own
    leg's clock. The cruise begins the moment the escape hands over, so the escape's end and the
    cruise's start are one boundary.

    A zero-length span (nothing solved) yields the degenerate boundaries that make each scrub park
    at its own leg's end rather than divide by zero.
    """
    total = float(span.get("total_days") or 0.0)
    if total <= 0.0:
        return {"escape_end": 1.0, "cruise_start": 0.0, "cruise_end": 1.0, "return_start": 1.0}
    esc = float(span["esc_days"])
    cruise_start = esc
    cruise_end = cruise_start + float(span.get("cruise_days", 0.0))
    return {"escape_end": esc / total, "cruise_start": cruise_start / total,
            "cruise_end": cruise_end / total,
            "return_start": (cruise_end + float(span.get("stay_days", 0.0))) / total}


def _tl_text(span: dict, t: float) -> tuple[str, str]:
    from datetime import timedelta
    total = span["total_days"]
    if not total or span["liftoff"] is None:
        return "no mission timeline yet", MUTED
    days = round(t * total)
    when = span["liftoff"] + timedelta(days=days)
    # Color the readout by which phase the playhead sits in (cumulative day boundaries).
    esc = span["esc_days"]
    cruise_end = esc + span.get("cruise_days", 0.0)
    stay_end = cruise_end + span.get("stay_days", 0.0)
    if days < esc:
        color, phase = AMBER, "escape"
    elif days < cruise_end:
        color, phase = ACCENT, "cruise"
    elif days < stay_end:
        color, phase = STAY, "at the target"
    else:
        color, phase = RETURN, "return"
    return f"T+ {days} d · {when:%b %d, %Y} · {phase}", color


# Playback runs at a fixed MISSION-DAYS-PER-SECOND, so the visual speed is the same whether the
# timeline is a one-way cruise or a full round trip (a fixed steps-per-range instead made a longer
# round trip whip by ~2x faster). The thumb and the marker advance on this one clock (the 0.1 s
# tick), so they can never drift apart.
_PLAY_DAYS_PER_SEC = 100.0
_TICK_SECONDS = 0.1


def _set_t(value: float) -> None:
    """A manual scrub (user dragging the slider): move the position, the date readout, and the
    visible plot's marker to it.

    Ignored entirely during play: the tick is the single clock that advances the thumb, readout,
    and marker together, and the slider's own change events must not fight it. The tick sets
    ``_tl_slider.value`` every frame, and the client can echo that programmatic set back as a
    change event a frame or two out of date, and letting it write ``S.timeline_t`` would drag the
    playhead backward against the tick, so the thumb jumps around and the marker / readout appear
    frozen on the bouncing value. During play, only :func:`tick` moves things.
    """
    if S.timeline_playing:
        return
    S.timeline_t = float(value)
    _update_readout()
    _render_marker(S.timeline_t)


def _update_readout() -> None:
    if _tl_label is not None:
        text, color = _tl_text(_tl_span, S.timeline_t)
        _tl_label.text = text
        _tl_label.style(f"color:{color};font-size:.78rem;font-family:monospace;"
                        f"min-width:150px;text-align:right")


def _trace_index(fig, name: str):
    """The index of the trace named ``name`` in a figure, or None."""
    for i, trace in enumerate(fig.data):
        if getattr(trace, "name", None) == name:
            return i
    return None


def _clampfrac(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def _render_marker(t: float) -> None:
    """Move the visible plot's marker(s) to the timeline position by restyling only those 1-point
    traces (the heavy arc is never recomputed). Drives whichever plot is on screen.

    Each leg's scrub is isolated. A failure in one must not abort the others, and above all must
    not propagate to the slider's on_change or the play tick, which would otherwise
    swallow the whole UI update (the slider would freeze and the plots stop moving)."""
    for scrub in (_scrub_spiral, _scrub_traj, _scrub_return):
        try:
            scrub(t)
        except Exception:  # noqa: BLE001  (one leg must never block the rest or the tick)
            pass


def _scrub_spiral(t: float) -> None:
    """The escape phase [0, esc_frac] of the timeline scrubs the spacecraft up the spiral."""
    if _spiral_plot is None or _spiral_marker is None or _spiral_pos is None:
        return
    frac = _clampfrac(t / _spiral_esc_frac) if _spiral_esc_frac > 0 else 1.0
    p = _spiral_pos[int(round(frac * (len(_spiral_pos) - 1)))]
    _run(_spiral_plot, "restyle",
         {"x": [[float(p[0])]], "y": [[float(p[1])]], "z": [[float(p[2])]]}, [_spiral_marker])


def _scrub_traj(t: float) -> None:
    """The cruise phase [esc_frac, cruise_end_frac] scrubs the spacecraft (+ Earth, asteroid,
    thrust cursor) along the outbound transfer (cruise_end_frac is 1.0 when there is no return)."""
    if _traj_plot is None or _traj_sc is None:
        return
    span = _traj_cruise_end_frac - _traj_esc_frac
    frac = _clampfrac((t - _traj_esc_frac) / span) if span > 0 else 1.0
    i = int(round(frac * (len(_traj_sc) - 1)))
    idxs, xs, ys, zs = [], [], [], []
    for name, arr in (("scrub-sc", _traj_sc), ("scrub-earth", _traj_earth), ("scrub-ast", _traj_ast),
                      ("scrub-fb", _traj_fb)):
        mi = _traj_marker.get(name)
        if mi is None or arr is None or len(arr) != len(_traj_sc):
            continue
        p = arr[i]
        idxs.append(mi)
        xs.append([float(p[0])])
        ys.append([float(p[1])])
        zs.append([float(p[2])])
    if idxs:
        _run(_traj_plot, "restyle", {"x": xs, "y": ys, "z": zs}, idxs)
    ci = _traj_marker.get("scrub-cursor")
    if ci is not None and _traj_days is not None:
        d = float(_traj_days[i])
        _run(_traj_plot, "restyle", {"x": [[d, d]], "y": [[0, 105]]}, [ci])


def _scrub_return(t: float) -> None:
    """The return phase [return_start_frac, 1] scrubs the laden return arc the same way."""
    if _ret_plot is None or _ret_sc is None:
        return
    span = 1.0 - _ret_start_frac
    frac = _clampfrac((t - _ret_start_frac) / span) if span > 0 else 1.0
    i = int(round(frac * (len(_ret_sc) - 1)))
    idxs, xs, ys, zs = [], [], [], []
    for name, arr in (("scrub-sc", _ret_sc), ("scrub-earth", _ret_earth), ("scrub-ast", _ret_ast)):
        mi = _ret_marker.get(name)
        if mi is None or arr is None or len(arr) != len(_ret_sc):
            continue
        p = arr[i]
        idxs.append(mi)
        xs.append([float(p[0])])
        ys.append([float(p[1])])
        zs.append([float(p[2])])
    if idxs:
        _run(_ret_plot, "restyle", {"x": xs, "y": ys, "z": zs}, idxs)
    ci = _ret_marker.get("scrub-cursor")
    if ci is not None and _ret_days is not None:
        d = float(_ret_days[i])
        _run(_ret_plot, "restyle", {"x": [[d, d]], "y": [[0, 105]]}, [ci])


def _run(plot, method: str, *args) -> None:
    """Fire a Plotly method on a plotly element, tolerating it having been replaced/removed."""
    try:
        plot.run_method("run_plot_method", method, *args)
    except Exception:  # noqa: BLE001  (element gone between refreshes; the next render re-syncs)
        pass


def _play_range() -> tuple[float, float]:
    """The timeline sub-range the active plot's phase spans: escape [0, esc_frac] for the spiral,
    cruise [esc_frac, 1] for the trajectory, the whole bar when neither is on screen."""
    if _spiral_plot is not None and _spiral_marker is not None:
        return 0.0, _spiral_esc_frac
    if _traj_plot is not None and _traj_sc is not None:
        return _traj_esc_frac, 1.0
    return 0.0, 1.0


def _toggle_play() -> None:
    """Play/pause. Both the thumb and the plot marker are advanced by the tick on one clock, so
    they stay locked together; play just sets the flag (restarting from the phase start if at the
    end) and the tick does the work."""
    S.timeline_playing = not S.timeline_playing
    if S.timeline_playing:
        lo, hi = _play_range()
        if S.timeline_t >= hi - 1e-6:
            S.timeline_t = lo
            _render_marker(S.timeline_t)
    _timeline.refresh()                             # flip the play/pause icon


def tick() -> None:
    """Page timer (0.1 s): while playing, advance the thumb, date readout, and plot marker one
    step together (single clock). Stops at the end of the active phase."""
    if S.workspace != "flight" or not S.timeline_playing:
        return
    lo, hi = _play_range()
    # Advance a fixed number of mission days per tick: convert days to a fraction of the whole
    # timeline so the spacecraft moves at the same speed across escape, cruise, stay, and return.
    total = max(1.0, float(_tl_span.get("total_days") or 0.0))
    step = (_PLAY_DAYS_PER_SEC * _TICK_SECONDS) / total
    t = min(hi, S.timeline_t + step)
    S.timeline_t = t
    if _tl_slider is not None:
        _tl_slider.value = t
    _update_readout()
    _render_marker(t)
    if t >= hi - 1e-9:
        S.timeline_playing = False
        _timeline.refresh()


def _set_porkchop_min(value: bool) -> None:
    S.porkchop_min = value
    _canvas.refresh()


# ======================================================================================
# target elements, and running the escape spiral
# ======================================================================================

def _has_full_elements(target: dict) -> bool:
    try:
        return all(np.isfinite(float(target[k])) for k in _ELEMENTS)
    except (KeyError, TypeError, ValueError):
        return False




def _run_spiral() -> None:
    rc = _resolve()
    if rc is None:
        return
    S.launch_run_id = jobs.submit_spiral(
        rc.model_dump(mode="json"), options=state.spiral_options(),
        fingerprint=state.spiral_fingerprint(rc), label=S.study_name)
    _refresh_all()



def _run_cruise(rc) -> None:
    """Refine the picked grid cell into the flown trajectory.

    The grid already solved this cell, so the cruise starts from its converged decision vector
    rather than from a conic at the same dates. That is the whole point of keeping the vector cube:
    measured, a warm start from a converged neighbour matched or beat a nine-start multi-start at a
    fraction of the solver time.

    The multi-start is kept rather than cut to one. A grid-seeded refine regressed against it in 2
    of 9 measured trials, by under 1%, and only because one start covers less ground than nine. One
    of those starting points is a cold Lambert guess, which was the only thing that produced a
    valid trajectory in the one case where every warm start failed. Cheap insurance, now that it is
    cheap.
    """
    grid = _grid_result()
    if grid is None:
        return
    dep_m, tof = _seed_cell(grid)
    arr_m = dep_m + tof
    options = _solver_options(rc, tof)
    options["nseg"] = int(grid.get("nseg") or options["nseg"])
    seed_x = _grid_seed(grid, dep_m, tof, int(options["nseg"]))
    if seed_x is not None:
        options["x0"] = seed_x
    if rc.mission.return_trip:
        # The chained return leg solves in the same job; its knobs ride alongside (the
        # destination/deadline/stay are mission definition, read from the config).
        options["return_options"] = state.return_options()
    target_row = (grid.get("target_row")
                  or jobs.read_grid_job(S.grid_run_id).get("target") or dict(S.focus or {}))
    S.solve_run_id = jobs.submit(
        rc.model_dump(mode="json"), target_row, dep_m, arr_m,
        label=S.study_name, porkchop=None, options=options)
    _refresh_all()


def _grid_cell_index(grid: dict, dep_m: float, tof: float) -> tuple[int, int] | None:
    """The grid indices nearest ``(dep_m, tof)``.

    Nearest rather than exact because a click reports the cell's own coordinates while a curve
    point may carry the polished departure, which sits between two grid lines.
    """
    dep = np.asarray(grid["dep_mjd2000"], float)
    tofs = np.asarray(grid["tof_days"], float)
    if not dep.size or not tofs.size:
        return None
    return (int(np.argmin(np.abs(dep - float(dep_m)))),
            int(np.argmin(np.abs(tofs - float(tof)))))


def _polished_point(grid: dict, dep_m: float, tof: float):
    """The refined solve a curve point came from, or None if the click was on a cell.

    A refined point's departure sits between two grid lines, so it is not any cell and its delta-v
    is not any cell's. Its own converged vector is stored, so a click resolves to the trajectory
    the curve quoted rather than to the nearest cell, which reads up to 0.03 km/s worse.
    """
    for p in (grid.get("polished") or []):
        if not p.get("feasible") or p.get("decision_vector") is None:
            continue
        if (abs(float(p["tof_days"]) - float(tof)) < 1e-3
                and abs(float(p["dep_mjd2000"]) - float(dep_m)) < 1e-6):
            return p
    return None


def _grid_vector(grid: dict, dep_m: float, tof: float):
    """The converged decision vector at the grid cell nearest ``(dep_m, tof)``, or None.

    It has to come from a cell that converged. One that did not is not a solution, and starting
    from it would begin at a point satisfying nothing.
    """
    idx = _grid_cell_index(grid, dep_m, tof)
    if idx is None:
        return None
    i, j = idx
    if not np.asarray(grid["feasible"], bool)[i, j]:
        return None
    return grid["decision_vectors"][i][j]


def _grid_leg_terms(grid: dict, dep_m: float, tof: float) -> dict:
    """The thrust terms the grid cell nearest ``(dep_m, tof)`` was solved under (see
    ``grid.leg_terms``), or an empty dict for a grid that did not record them."""
    idx = _grid_cell_index(grid, dep_m, tof)
    cube = grid.get("leg_terms")
    if idx is None or not cube:
        return {}
    i, j = idx
    try:
        return dict(cube[i][j] or {})
    except (IndexError, TypeError):
        return {}


def _grid_seed(grid: dict, dep_m: float, tof: float, nseg: int):
    """The warm start for a refine, or None when the segment counts do not agree.

    A control history may not be resampled between segment counts: the resampled point satisfies
    the coarse discretization's constraints but not the finer one's, and the optimizer settles
    there and reports a plausible ΔV with a failed matchpoint. So a mismatch costs the seed rather
    than silently interpolating, and the rail says so, because a fidelity slider that quietly
    disables the warm start would be worse than one that refuses.
    """
    if int(nseg) != int(grid.get("nseg") or 0):
        return None
    refined = _polished_point(grid, dep_m, tof)
    if refined is not None:
        return refined["decision_vector"]
    return _grid_vector(grid, dep_m, tof)


def _available_cruise_power_W(rc):
    """The array power the cruise runs on: frozen at the post-escape (degraded) value. None when
    no matching spiral flew this session, in which case the cruise runs at rated power."""
    return products.spiral_power_available_W(state.session_spiral_run(rc))


def _available_thrust_fraction(rc) -> float:
    """Fraction of rated thrust the cruise can produce at its frozen post-escape power. Used to
    draw the thrust ceiling and gate the porkchop's reachability."""
    return products.available_thrust_fraction(rc, _available_cruise_power_W(rc))


def _solver_options(rc, seed_tof_days: float) -> dict:
    window_days = max(200, (rc.mission.arrive_by - rc.departure_window[0]).days)
    default_tof = int(min(window_days, max(150, round(seed_tof_days) + 100)))
    max_tof = float(S.solve_max_tof_days) if S.solve_max_tof_days is not None else float(default_tof)
    # Power-limiting is handled by ``available_power_W``: the cruise runs at the frozen post-escape
    # array power, and the solver reads the reduced thrust and Isp off the engine throttle curve at
    # that power. ``thrust_limit`` is then only the user's Max-thrust setting, a fraction of the
    # thrust available at that power.
    return {"window_slack_days": float(S.solve_slack_days),
            "max_tof_days": max_tof, "nseg": int(S.solve_nseg),
            "max_duty_cycle": float(S.solve_duty_pct) / 100.0,
            # Seed-diverse parallel multi-start: > 1 fans that many trajectory-family seeds across
            # cores and keeps the best, so a short clicked cell can't trap the solve in a local
            # optimum. starts_workers left auto (worker sizes it to the cores).
            "n_starts": int(S.solve_starts),
            "available_power_W": _available_cruise_power_W(rc),
            # The SF transcription scales its v∞ inequality by the cap, so an exact zero gets the
            # sweep's 1 m/s floor.
            "vinf_dep_kms": max(float(rc.departure_vinf_kms), sweep.VINF_CAP_FLOOR_KMS),
            "max_dep_decl_deg": float(state.departure_declination_cap(rc))}


def _converged(feasible, mismatch) -> bool:
    return products.converged(feasible, mismatch)


def _return_plottable(res) -> bool:
    """True when the return leg produced a flyable (converged) arc worth plotting/charting.

    An unconverged optimization is not a real trajectory, so its arc, diagnostics, and
    timeline phase are all suppressed, and only its status is reported."""
    ret = res.get("return") if res else None
    return bool(ret and "error" not in ret and ret.get("sf")
                and _converged(ret["sf"]["feasible"], ret["sf"]["mismatch"]))


async def _export_trajectory() -> None:
    """Download the open solve as a trajectory bundle: the escape spiral and the heliocentric
    cruise on one launch-relative clock, the target orbit, and the vehicle properties. The
    spiral used is the one matching the live config this session, or none when the launch
    vehicle provides escape."""
    rc = _resolve()
    if rc is None:
        ui.notify("The configuration does not resolve - fix it in Project.", type="warning")
        return
    if _cruise_result(S.solve_run_id) is None:
        ui.notify("Run a converged cruise first.", type="warning")
        return
    spiral_id = state.session_spiral_run(rc)
    if spiral_id is None and not rc.launch.escape_provided:
        ui.notify("No escape spiral in force, so the export will be cruise-only.",
                  type="warning")

    note = ui.notification("Building the trajectory bundle...", spinner=True, timeout=None)
    try:
        path = await run.io_bound(_build_trajectory_bundle, rc, S.solve_run_id, spiral_id)
    except Exception as exc:  # noqa: BLE001  (report the failure; never crash the app)
        note.dismiss()
        ui.notify(f"Export failed: {str(exc).splitlines()[0]}", type="negative")
        return
    note.dismiss()
    ui.download(str(path))
    ui.notify("Trajectory bundle ready.", type="positive")


def _build_trajectory_bundle(rc, cruise_run_id: str, spiral_run_id: str | None) -> Path:
    """Assemble the bundle for one solve and write it to a temporary zip; returns its path."""
    files = trajectory_export.build_bundle_from_runs(rc, cruise_run_id, spiral_run_id)
    slug = "".join(c if c.isalnum() else "_" for c in str(rc.vehicle.name)).strip("_") or "mission"
    out = Path(tempfile.gettempdir()) / f"trajectory_bundle_{slug}.zip"
    out.write_bytes(trajectory_export.bundle_zip_bytes(files))
    return out



def _run_matches_config(run_id, rc) -> bool:
    """Whether a finished cruise run was solved for the config in force now."""
    return products.cruise_run_matches(run_id, rc, _available_cruise_power_W(rc))


def _cruise_result(run_id) -> dict | None:
    if not run_id or not jobs.is_terminal(jobs.read_status(run_id)):
        return None
    # A finished run is shown only while it still matches the live config; a vehicle/mission edit
    # or a re-flown spiral clears the trajectory window rather than leaving a stale arc (and stale
    # total-mission numbers) up with a "needs re-run" warning.
    if not _run_matches_config(run_id, _resolve()):
        return None
    res = jobs.read_result(run_id)
    if res and _converged(res["sf"]["feasible"], res["sf"]["mismatch"]):
        return res
    return None



def _escape_cost_at(rc, vinf_kms: float) -> tuple[float, str]:
    """Price the launch phase at the v∞ a solve departed with: ``(dv_kms, source_label)``."""
    if rc.launch.escape_provided:
        return 0.0, "launch vehicle"
    run_id = state.session_spiral_run(rc)
    curve = jobs.read_spiral_summary(run_id) if run_id is not None else None
    dv, _tof = sweep.escape_cost(curve, float(vinf_kms), rc.launch)
    curve_v = (curve or {}).get("curve_vinf_kms") or []
    if curve_v and float(vinf_kms) <= float(curve_v[-1]) + 1e-9:
        source = "spiral run"
    elif curve_v:
        source = f"spiral run to v∞ {float(curve_v[-1]):.2f}, extended analytically"
    else:
        source = "estimate"
    return float(dv), source


# ======================================================================================
# small helpers + state setters
# ======================================================================================

def _stat_cards(cards: list[tuple[str, str]]) -> None:
    with ui.row().classes("w-full no-wrap gap-2").style("flex:0 0 auto"):
        for label, value in cards:
            with ui.column().classes("gap-0 p-1 rounded-lg flex-grow items-center").style(
                    f"background:{PANEL2};border:1px solid {BORDER}"):
                ui.label(value).style(f"color:{TEXT};font-family:monospace;font-size:.82rem")
                ui.label(label).style(f"color:{MUTED};font-size:.62rem")


def _selected_cell_line(grid: dict, dep_m: float, tof: float) -> None:
    """What the picked cell already is, from the grid's own solve of it.

    No bracket: the cell is a converged trajectory, not a range between an optimistic floor and a
    pessimistic estimate. What it carries is its ΔV, its delivered mass, and whether the search
    closed it at all.
    """
    dep = np.asarray(grid["dep_mjd2000"], float)
    tofs = np.asarray(grid["tof_days"], float)
    ok = np.asarray(grid["feasible"], bool)
    i = int(np.argmin(np.abs(dep - dep_m)))
    j = int(np.argmin(np.abs(tofs - tof)))
    tag = "Selected" if S.grid_sel is not None else "Best found (click to change)"
    if not ok[i, j]:
        ui.label(f"{tag}: {_fmt(dep_m)} → {_fmt(dep_m + tof)} · {tof:.0f} d · "
                 "no trajectory found here yet").style(f"color:{AMBER};font-size:.72rem")
        return
    dv = float(np.asarray(grid["dv_kms"], float)[i, j])
    mass = float(np.asarray(grid["final_mass_kg"], float)[i, j])
    ui.label(f"{tag}: {_fmt(dep_m)} → {_fmt(dep_m + tof)} · {tof:.0f} d · "
             f"{dv:.2f} km/s · {mass:.0f} kg delivered").style(
        f"color:{MUTED};font-size:.72rem")


def _seed_cell(grid: dict) -> tuple[float, float]:
    """The cell to refine, as ``(departure epoch, flight time)``: the picked one, else the best on
    the curve. Falling back to the cheapest point on it means Run does the obvious thing before
    anything has been clicked."""
    if S.grid_sel:
        return float(S.grid_sel[0]), float(S.grid_sel[1])
    pts = [p for p in (grid.get("frontier") or []) if p.get("dv_kms") is not None]
    if pts:
        best = min(pts, key=lambda p: p["dv_kms"])
        return float(best["dep_mjd2000"]), float(best["tof_days"])
    dep = np.asarray(grid["dep_mjd2000"], float)
    tofs = np.asarray(grid["tof_days"], float)
    return float(dep[0]), float(tofs[0])


def _to_mjd2000(value) -> float:
    ts = np.datetime64(value)
    return (ts - np.datetime64("2000-01-01")) / np.timedelta64(1, "D")


def _fmt(mjd2000: float) -> str:
    return figures.mjd2000_to_datetime(float(mjd2000)).date().isoformat()


def _clip(value) -> float:
    return min(max(float(value), 0.0), 1.0)


def _set_launch(*, vinf=None, duty=None, years=None, target_inc=None,
                rad_model=None, coverglass=None, coverglass_density=None) -> None:
    rail_dirty = False
    if vinf is not None:
        S.departure_vinf = float(vinf)
    if duty is not None:
        S.launch_duty = float(duty)
    if years is not None:
        S.launch_years = float(years)
    if target_inc is not None:
        S.launch_target_inc = float(target_inc)
    if rad_model is not None:
        S.launch_radiation_model = str(rad_model)
        # Snap the cover thickness + density to the chosen scenario's own values as a sensible
        # start; the inputs then override them. Rebuild the rail so the inputs show the new values.
        models = load_radiation_models()
        if rad_model in models:
            S.launch_coverglass_um = float(models[rad_model].coverglass_um)
            S.launch_coverglass_density = float(models[rad_model].coverglass_density_g_cm3)
        rail_dirty = True
    if coverglass is not None:
        S.launch_coverglass_um = float(coverglass)
    if coverglass_density is not None:
        S.launch_coverglass_density = float(coverglass_density)
    state.mark_dirty()
    # The v∞ knob reprices the whole budget (cards, porkchop credit, departure date); refresh the
    # budget strip + the top bar. The rail's own value labels update live without a rebuild.
    _canvas.refresh()
    _timeline.refresh()
    from ui import topbar
    topbar.header.refresh()
    if rail_dirty:
        _rail.refresh()


def _set_steer(value: bool) -> None:
    S.launch_steer = bool(value)
    if value and S.launch_target_inc is None:
        rc = _resolve()
        S.launch_target_inc = float(rc.launch.inclination_deg or 0.0) if rc else 0.0
    _rail.refresh()


def _set_solve(*, duty=None, max_tof=None, nseg=None, starts=None) -> None:
    if duty is not None:
        S.solve_duty_pct = float(duty)
    if max_tof is not None:
        S.solve_max_tof_days = float(max_tof)
    if nseg is not None:
        S.solve_nseg = int(nseg)
    if starts is not None:
        S.solve_starts = max(1, int(starts))


def _set_return(*, duty=None, max_tof=None, nseg=None) -> None:
    if duty is not None:
        S.return_duty_pct = float(duty)
    if nseg is not None:
        S.return_nseg = int(nseg)
    if max_tof is not None:
        # The return deadline and duration are part of the mission, so update the working mission
        # the budget strip + a saved study reflect it.
        S.mission = S.mission.model_copy(update={"return_max_tof_days": float(max_tof)})
        state.mark_dirty()


def _set_return_dest(key: str) -> None:
    """Pick the return destination (mission definition); reprices the round-trip budget."""
    S.mission = S.mission.model_copy(update={"return_destination": str(key)})
    state.mark_dirty()
    _canvas.refresh()
    _timeline.refresh()


def _clear_solve() -> None:
    S.solve_run_id = None
    S.grid_run_id = None
    S.grid_sel = None
    _refresh_all()


def _cancel_cruise() -> None:
    S.solve_run_id = None
    _refresh_all()


def _dismiss_spiral() -> None:
    S.launch_run_id = None
    _refresh_all()


def _delete_spiral() -> None:
    if S.launch_run_id:
        jobs.delete_spiral_run(S.launch_run_id)
    S.launch_run_id = None
    _refresh_all()


# ======================================================================================
# page-level polling (registered in app.py): stream job progress, finalize on terminal edge
# ======================================================================================

def poll() -> None:
    """Page timer hook: advance the active job's progress bar, and trigger a single full
    re-render when a run finishes, so the converged result renders once."""
    if S.workspace != "flight":
        return
    edge = False
    finished: set[str] = set()
    for channel, run_id, reader in (
            ("spiral", S.launch_run_id, jobs.read_spiral_status),
            ("grid", S.grid_run_id, jobs.read_grid_status),
            ("cruise", S.solve_run_id, jobs.read_status)):
        if not run_id:
            _running.discard(channel)
            continue
        terminal = jobs.is_terminal(reader(run_id))
        if not terminal:
            _running.add(channel)
        elif channel in _running:
            _running.discard(channel)
            finished.add(channel)
            edge = True
    if edge:
        # A grid that just finished has a best cell already solved, so show it instead of leaving
        # the panel empty for a click that only has one sensible target.
        if "grid" in finished and _cruise_result(S.solve_run_id) is None:
            grid = _grid_result()
            if grid is not None and S.grid_sel is None:
                cell = _seed_cell(grid)
                if (_polished_point(grid, cell[0], cell[1]) is not None
                        or _grid_vector(grid, cell[0], cell[1]) is not None):
                    # Set BEFORE the rebuild, so a rebuild that fails is not retried on every tick.
                    S.grid_sel = cell
                    _show_grid_cell(cell)
        _refresh_all()
    elif _running:
        _rail.refresh()         # advance the in-flight progress bar (no heavy plot redraw)
