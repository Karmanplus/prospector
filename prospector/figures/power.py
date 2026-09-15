"""Power and radiation figures: what the arrays deliver, and what the belts take.

Array output over the escape (with the power-limited zone shaded), the dose-vs-altitude profile,
and the whole-mission power and engine-performance timelines.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from prospector.constants import EARTH_EQUATORIAL_RADIUS_KM
from prospector.figures.theme import (
    DANGER,
    EARTH,
    EARTH_BLUE,
    ECCEN,
    GRID,
    MUTED,
    PLAY_GREEN,
    TEXT,
    _layout,
)
from prospector.figures.trajectory import _phase_transition_lines
from prospector.solvers import spiral as _spiral


def degradation_profile(times_days, power_fraction, *,
                        end_label="end of escape", power_floor_pct=None) -> go.Figure:
    """Permanent solar-array degradation accumulated over the climb, as the percentage of
    beginning-of-life power lost so far.

    The mirror image of the array-power curve: damage is ``(1 - power_fraction) x 100``, and
    because it is permanent the curve only ever climbs, stepping up while the spacecraft is in the
    belts and holding flat once it is clear. Where the curve steepens is where the dose is being
    taken, which makes two climb strategies directly comparable: one that gets out of the belts
    sooner has a flatter tail and less damage at the end. There are no markers for where the belts
    start and stop, because they have no real boundary; the curve's own slope shows it. The final
    value is the permanent loss the arrays have to be sized to live with.

    ``power_floor_pct``, the loss at which the array can no longer run the engines at full thrust,
    draws a dashed line with a shaded zone above it. It is shown whenever power is being tracked,
    not only when it gets crossed, and the y-axis keeps it in view either way: a design with
    headroom reads as a curve well below the line, and one that crosses it reads as power-limited.
    The margin is visible in both cases.
    """
    t = np.asarray(times_days, float)
    deg = (1.0 - np.asarray(power_fraction, float)) * 100.0
    fig = go.Figure()
    # Power-limited zone: above the floor the array can't hold full thrust. Shade it and line it so
    # both the headroom case (curve below) and the limited case (curve crosses in) read at a
    # glance.
    limited = power_floor_pct is not None and 0.0 <= float(power_floor_pct) < 100.0
    y_top = None
    if limited and len(deg):
        floor = float(power_floor_pct)
        crossed = float(deg[-1]) >= floor
        # Frame the axis to always include the floor (+headroom), so a high floor isn't off-screen.
        y_top = max(float(deg.max()) * 1.1, floor * 1.18, floor + 6.0)
        fig.add_hrect(y0=floor, y1=y_top, fillcolor="rgba(231,76,60,0.10)", line_width=0,
                      annotation_text="power-limited (full thrust can't be held)",
                      annotation_position="top left", annotation_font=dict(color=DANGER, size=10))
        fig.add_hline(y=floor, line=dict(color=DANGER, width=1.4, dash="dash"),
                      annotation_text=f"full-thrust floor ({floor:.0f}% lost)",
                      annotation_position="bottom left", annotation_font=dict(color=DANGER, size=10))
        if not crossed:                                  # name the margin so headroom is explicit
            fig.add_annotation(x=0.02, xref="paper", y=floor, yanchor="bottom",
                               text=f"{floor - float(deg[-1]):.0f}% margin to full-thrust floor",
                               showarrow=False, xanchor="left",
                               font=dict(color=PLAY_GREEN, size=10))
    fig.add_trace(go.Scatter(x=t, y=deg, mode="lines", fill="tozeroy",
                             line=dict(color=TEXT, width=2.6),
                             fillcolor="rgba(225,230,240,0.07)",
                             hovertemplate="day %{x:.0f}<br>%{y:.1f}% lost<extra></extra>"))
    if len(deg):
        end_txt = f"{deg[-1]:.0f}% lost - {end_label}"
        if limited and deg[-1] >= float(power_floor_pct):
            end_txt += " · power-limited"
        fig.add_annotation(x=float(t[-1]), y=float(deg[-1]),
                           text=end_txt, showarrow=True,
                           arrowcolor=MUTED, font=dict(color=TEXT, size=11), ax=-52, ay=-26)
    _layout(fig, "Permanent array degradation through the belts")
    fig.update_layout(width=1000, height=440, showlegend=False)
    fig.update_xaxes(title="days from launch", color=MUTED, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title="array power lost (% of beginning-of-life)", color=MUTED,
                     gridcolor=GRID, zeroline=False, rangemode="tozero",
                     range=([0, y_top] if y_top is not None else None))
    return fig


def altitude_radiation_profile(times_days, positions_km, *, n_bins: int = 90) -> go.Figure:
    """The range of altitudes the climb covers over time, with the radiation dose painted on it.

    The orbit swings between its low and high points hundreds of times while growing over months,
    so plotting every sample gives an unreadable smear. Instead the climb is split into time bins
    and each bin reduced to its lowest and highest altitude. The shaded band between them shows how
    far the spacecraft ranges at each stage, widening as the orbit stretches and rising as it
    escapes. Markers along the top are coloured by the worst dose rate in each bin, using the same
    model the damage calculation integrates, so where the band glows shows both when and at what
    altitude the dose is taken. The altitude axis is logarithmic so both the low belt crossings and
    the climb to escape are readable.
    """
    t = np.asarray(times_days, float)
    pos = np.asarray(positions_km, float)
    if pos.ndim != 2 or len(pos) < 2 or len(t) != len(pos):
        return go.Figure()
    alt = np.maximum(np.linalg.norm(pos, axis=1) - EARTH_EQUATORIAL_RADIUS_KM, 1.0)
    dose = _spiral.belt_ddd_rate(pos)

    # Bin in time; reduce each bin to perigee (min alt), apogee (max alt), peak dose.
    nb = int(max(8, min(n_bins, len(t))))
    edges = np.linspace(float(t[0]), float(t[-1]) + 1e-9, nb + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, nb - 1)
    lo = np.full(nb, np.inf)
    np.minimum.at(lo, idx, alt)
    hi = np.full(nb, -np.inf)
    np.maximum.at(hi, idx, alt)
    peak = np.zeros(nb)
    np.maximum.at(peak, idx, dose)
    tc = 0.5 * (edges[:-1] + edges[1:])
    keep = np.isfinite(lo) & np.isfinite(hi)
    tc, lo, hi, peak = tc[keep], lo[keep], np.maximum(hi[keep], lo[keep] * 1.001), peak[keep]
    col = np.log10(peak + 1e6)
    cutoff = float(np.log10(0.02 * _spiral.PROTON_DDD_CORE + 1e6))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tc, y=lo, mode="lines", line=dict(color=EARTH_BLUE, width=1),
                             name="perigee", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=tc, y=hi, mode="lines", line=dict(color=EARTH_BLUE, width=1),
                             fill="tonexty", fillcolor="rgba(74,163,255,0.12)", name="apogee",
                             hovertemplate="day %{x:.0f}<br>apogee %{y:,.0f} km<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=tc, y=hi, mode="markers", name="dose",
        marker=dict(size=6, color=col, colorscale="Inferno", cmin=cutoff,
                    cmax=max(float(np.nanmax(col)), cutoff + 1.0), showscale=True,
                    colorbar=dict(title="peak dose<br>MeV/g/day", x=1.0, len=0.75,
                                  tickvals=[6, 7, 8, 9], ticktext=["~0", "1e7", "1e8", "1e9"])),
        customdata=peak,
        hovertemplate="day %{x:.0f}<br>apogee %{y:,.0f} km<br>peak dose "
                      "%{customdata:.1e} MeV/g/day<extra></extra>"))
    _layout(fig, "Altitude band and radiation dose over the climb")
    fig.update_layout(width=1000, height=440, showlegend=False)
    fig.update_xaxes(title="days from launch", color=MUTED, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title="altitude (km)", type="log", color=MUTED, gridcolor=GRID, zeroline=False)
    return fig


# Whole-mission phase colors, matching propellant_timeline's convention: escape amber, cruise blue,
# return violet: one palette across every mission-profile chart.
PHASE_COLORS = (EARTH, EARTH_BLUE, ECCEN)


def mission_power_timeline(phases, *, transitions=(), nameplate_W=None,
                           title="Array power over the mission") -> go.Figure:
    """Solar-array power across the whole mission, and the three things that shape it.

    Three panels, separating what a single number hides.

    The top panel, in watts, shows for each phase what the panels generate (solid) and what reaches
    the thrusters (dashed) after the electronics in between. The constant gap between the two is
    the conversion loss, around 20-25% depending on which bus the thruster electronics are wired
    to. What varies is the solid line moving away from the ``nameplate_W`` reference.

    The bottom panel, as a percentage of the value at 1 AU, shows the three separate effects whose
    product gives that variation:

      * Radiation damage, a genuine loss of efficiency, permanent and never above 100%. It steps
        down through the escape spiral and then holds flat, because the cells never recover.
      * Distance from the Sun, which is not an efficiency but simply how much sunlight there is:
        1/r^2, so it goes above 100% closer in than 1 AU and below further out. This is what lets
        the panels beat their rating near the Sun.
      * Temperature, a genuine efficiency again, relative to 1 AU: cells convert worse when hot
        near the Sun and better when cold further out, so it moves the opposite way to sunlight
        and partly cancels it.

    The third panel is the temperature itself, in degrees Celsius, with the absolute cell
    efficiency (percent of sunlight converted) on its right-hand axis: the panel runs hot near
    Earth (albedo and Earth infrared on its back) and near the Sun, and cools going out, and the
    efficiency mirrors it. A phase that carries no thermal series leaves the panel empty.

    ``phases`` is ``[(label, times_days, array_output_W, thruster_W, degradation_pct,
    irradiance_pct, cell_eff_pct[, panel_temp_C, cell_eff_abs_pct]), ...]`` on the shared mission
    clock; ``nameplate_W`` is the 1-AU BOL array output; ``transitions`` marks the phase seams.
    """
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        row_heights=[0.42, 0.30, 0.28],
                        specs=[[{}], [{}], [{"secondary_y": True}]])
    series = {"belt degradation": ([], [], DANGER),
              "irradiance (sun distance)": ([], [], EARTH),
              "cell efficiency (temperature)": ([], [], ECCEN)}
    thermal = {"array temperature": ([], [], DANGER),
               "cell efficiency": ([], [], ECCEN)}
    for k, phase in enumerate(phases):
        label, t, array_W, thruster_W, deg_pct, irr_pct, cell_pct, *therm = phase
        t = np.asarray(t, float)
        if not len(t):
            continue
        color = PHASE_COLORS[k % len(PHASE_COLORS)]
        fig.add_trace(go.Scatter(
            x=t, y=np.asarray(array_W, float), mode="lines", name=f"{label} · array output",
            line=dict(color=color, width=2.5),
            hovertemplate="day %{x:.0f}<br>%{y:,.0f} W generated<extra>" + str(label)
                          + "</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=t, y=np.asarray(thruster_W, float), mode="lines", name=f"{label} · at thrusters",
            line=dict(color=color, width=1.4, dash="dot"),
            hovertemplate="day %{x:.0f}<br>%{y:,.0f} W to thrusters<extra>" + str(label)
                          + "</extra>"), row=1, col=1)
        # Gather each effect across every phase into one continuous line (a None break at each
        # phase seam keeps stay-period gaps from drawing a false connecting segment).
        for pct, (xs, ys, _c) in zip((deg_pct, irr_pct, cell_pct), series.values()):
            xs.extend([*t.tolist(), None])
            ys.extend([*np.asarray(pct, float).tolist(), None])
        if len(therm) == 2:
            for vals, (xs, ys, _c) in zip(therm, thermal.values()):
                xs.extend([*t.tolist(), None])
                ys.extend([*np.asarray(vals, float).tolist(), None])
    if any(xs for xs, _y, _c in series.values()):
        for name, (xs, ys, color) in series.items():
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name=name, line=dict(color=color, width=2.5),
                hovertemplate="day %{x:.0f}<br>%{y:.1f}% (" + name + ")<extra></extra>"),
                row=2, col=1)
        fig.add_hline(y=100.0, row=2, col=1, line=dict(color=MUTED, width=1, dash="dash"),
                      annotation_text="1 AU value", annotation_position="bottom right",
                      annotation_font=dict(color=MUTED, size=10))
    if any(xs for xs, _y, _c in thermal.values()):
        (tx, ty, tcolor), (ex, ey, ecolor) = thermal.values()
        fig.add_trace(go.Scatter(
            x=tx, y=ty, mode="lines", name="array temperature",
            line=dict(color=tcolor, width=2.5),
            hovertemplate="day %{x:.0f}<br>%{y:.0f} °C panel<extra></extra>"),
            row=3, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(
            x=ex, y=ey, mode="lines", name="cell efficiency",
            line=dict(color=ecolor, width=2.0, dash="dot"),
            hovertemplate="day %{x:.0f}<br>%{y:.1f}% of sunlight converted<extra></extra>"),
            row=3, col=1, secondary_y=True)
    if nameplate_W:
        fig.add_hline(y=float(nameplate_W), row=1, col=1,
                      line=dict(color=MUTED, width=1, dash="dash"),
                      annotation_text=f"nameplate BOL at 1 AU · {float(nameplate_W):,.0f} W",
                      annotation_position="bottom right",
                      annotation_font=dict(color=MUTED, size=11))
    _phase_transition_lines(fig, transitions, rows=(1, 2, 3))
    _layout(fig, title)
    # Eleven legend entries wrap to two rows, so the legend grows upward from the top panel's
    # edge into a taller top margin rather than over the traces.
    fig.update_layout(legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0,
                                  font=dict(color=MUTED)),
                      margin=dict(t=104), title_y=0.985, title_yanchor="top")
    fig.update_xaxes(color=MUTED, gridcolor=GRID, zeroline=False)
    fig.update_xaxes(title="days from launch", row=3, col=1)
    fig.update_yaxes(title="power (W)", color=MUTED, gridcolor=GRID, zeroline=False,
                     rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title="effect vs 1-AU value (%)", color=MUTED,
                     gridcolor=GRID, zeroline=False, rangemode="tozero", row=2, col=1)
    fig.update_yaxes(title="panel °C", color=MUTED, gridcolor=GRID, zeroline=False,
                     row=3, col=1, secondary_y=False)
    fig.update_yaxes(title="cell eff. (%)", color=MUTED, showgrid=False, zeroline=False,
                     row=3, col=1, secondary_y=True)
    return fig


def engine_performance_timeline(phases, *, transitions=(), rated_thrust_mN=None,
                                title="Engine performance over the mission") -> go.Figure:
    """What the thruster stack can do, and what it is doing, across the whole mission.

    Two panels on one mission clock. The top shows the thrust available, meaning what the delivered
    power buys along the engines' throttle curve, which is the ceiling power sets. Where a solved
    trajectory provides it, the thrust being used is drawn too. The gap between them is margin. The
    bottom shows the Isp at that operating point: with a throttle curve, running short of power
    costs fuel efficiency as well, so a cruise far from the Sun burns more than the drop in thrust
    alone would suggest. ``phases`` is ``[(label, times_days, avail_mN, demand_mN, isp_s), ...]``
    (``demand_mN``/``isp_s`` may be None); ``rated_thrust_mN`` draws the stack's rated ceiling for
    reference.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.62, 0.38])
    for k, (label, t, avail, demand, isp) in enumerate(phases):
        t = np.asarray(t, float)
        if not len(t):
            continue
        color = PHASE_COLORS[k % len(PHASE_COLORS)]
        fig.add_trace(go.Scatter(
            x=t, y=np.asarray(avail, float), mode="lines", name=f"{label} · available",
            line=dict(color=color, width=2.5),
            hovertemplate="day %{x:.0f}<br>%{y:,.0f} mN available<extra>" + str(label)
                          + "</extra>"), row=1, col=1)
        if demand is not None:
            fig.add_trace(go.Scatter(
                x=t, y=np.asarray(demand, float), mode="lines", name=f"{label} · flown",
                line=dict(color=color, width=1.4, dash="dot"),
                hovertemplate="day %{x:.0f}<br>%{y:,.0f} mN flown<extra>" + str(label)
                              + "</extra>"), row=1, col=1)
        if isp is not None:
            fig.add_trace(go.Scatter(
                x=t, y=np.asarray(isp, float), mode="lines", name=f"{label} · Isp",
                showlegend=False, line=dict(color=color, width=2),
                hovertemplate="day %{x:.0f}<br>Isp %{y:,.0f} s<extra>" + str(label)
                              + "</extra>"), row=2, col=1)
    if rated_thrust_mN:
        fig.add_hline(y=float(rated_thrust_mN), row=1, col=1,
                      line=dict(color=MUTED, width=1, dash="dash"),
                      annotation_text=f"rated {float(rated_thrust_mN):,.0f} mN",
                      annotation_position="bottom right",
                      annotation_font=dict(color=MUTED, size=11))
    _phase_transition_lines(fig, transitions, rows=(1, 2))
    _layout(fig, title)
    fig.update_layout(legend=dict(orientation="h", y=1.05, x=0, font=dict(color=MUTED)))
    fig.update_xaxes(color=MUTED, gridcolor=GRID, zeroline=False)
    fig.update_xaxes(title="days from launch", row=2, col=1)
    fig.update_yaxes(title="thrust (mN)", color=MUTED, gridcolor=GRID, zeroline=False,
                     rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title="Isp (s)", color=MUTED, gridcolor=GRID, zeroline=False, row=2, col=1)
    return fig
