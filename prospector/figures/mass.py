"""Mass and propellant figures: the budget as a picture.

Cumulative propellant across escape and cruise against the usable load, and where the dry mass goes
once the arrays, tank, thrusters, and bus are charged against it.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from prospector.figures.theme import (
    ASTEROID_ORANGE,
    BG,
    DEPARTURE,
    DOME,
    EARTH,
    EARTH_BLUE,
    ECCEN,
    GRID,
    MUTED,
    PLAY_GREEN,
    REFERENCE,
    TEXT,
    _layout,
)

# One stable color per dry-mass slice, distinct enough to read side by side.
_ALLOC_COLORS = (DOME, EARTH_BLUE, EARTH, ECCEN, PLAY_GREEN, ASTEROID_ORANGE,
                 MUTED, "#f0e68c", REFERENCE)


def propellant_timeline(escape_days, escape_prop_cum, cruise_days, cruise_prop_cum,
                        *, usable_kg=None) -> go.Figure:
    """Cumulative propellant spent over the whole mission, escape then cruise.

    Two phases on one clock: the Earth-escape spiral and the heliocentric cruise, joined at Earth
    departure. The right-hand end is the total propellant the mission consumes; an optional dashed
    line marks the usable load aboard so the headroom is visible at a glance.
    """
    ed = np.asarray(escape_days, float)
    ep = np.asarray(escape_prop_cum, float)
    have_escape = len(ed) > 0
    offset = float(ed[-1]) if have_escape else 0.0
    base = float(ep[-1]) if have_escape else 0.0
    fig = go.Figure()
    if have_escape:
        fig.add_trace(go.Scatter(x=ed, y=ep, mode="lines", name="Earth escape",
                                 line=dict(color=EARTH, width=2.5),
                                 hovertemplate="day %{x:.0f}<br>%{y:.1f} kg<extra>escape</extra>"))
    cd = np.asarray(cruise_days, float) + offset
    cp = np.asarray(cruise_prop_cum, float) + base
    if len(cd):
        fig.add_trace(go.Scatter(x=cd, y=cp, mode="lines", name="heliocentric cruise",
                                 line=dict(color=EARTH_BLUE, width=2.5),
                                 hovertemplate="day %{x:.0f}<br>%{y:.1f} kg<extra>cruise</extra>"))
        if have_escape:
            fig.add_trace(go.Scatter(x=[offset], y=[base], mode="markers", hoverinfo="skip",
                                     name="Earth departure",
                                     marker=dict(color=DEPARTURE, size=9,
                                                 line=dict(color=BG, width=1))))
    if usable_kg:
        fig.add_hline(y=float(usable_kg), line=dict(color=MUTED, width=1, dash="dash"),
                      annotation_text=f"usable load {float(usable_kg):.0f} kg",
                      annotation_position="bottom right",
                      annotation_font=dict(color=MUTED, size=11))
    _layout(fig, "Propellant used over the mission")
    fig.update_layout(width=1000, height=440,
                      legend=dict(orientation="h", y=1.04, x=0, font=dict(color=MUTED)))
    fig.update_xaxes(title="days from launch", color=MUTED, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title="cumulative propellant (kg)", color=MUTED, gridcolor=GRID,
                     zeroline=False, rangemode="tozero")
    return fig


def mass_allocation(components, *, dry_mass_kg, title="Dry-mass allocation") -> go.Figure:
    """How the dry-mass budget divides across the bus, as a horizontal bar per item.

    ``components`` is a list of ``(label, kg)`` covering the sized hardware (arrays, power
    distribution, thrusters, tank), the fixed-fraction subsystems, and the leftover reserve for
    payload and margin. Reading the bars shows what a solar-electric bus spends its mass on: arrays
    and tank dominate alongside structure and the fixed subsystems.
    """
    comps = [(str(label), float(kg)) for label, kg in components if float(kg) != 0.0]
    # Keep every other slice sorted by size, but always seat 'Payload + margin' at the bottom of
    # the chart (the first entry of a horizontal bar) regardless of its size.
    payload = [c for c in comps if "payload" in c[0].lower()]
    others = sorted((c for c in comps if "payload" not in c[0].lower()), key=lambda lk: lk[1])
    comps = payload + others
    labels = [label for label, _ in comps]
    vals = [kg for _, kg in comps]
    pcts = [100.0 * kg / dry_mass_kg if dry_mass_kg else 0.0 for kg in vals]
    colors = [_ALLOC_COLORS[i % len(_ALLOC_COLORS)] for i in range(len(comps))]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(color=BG, width=0.5)),
        text=[f"{kg:.0f} kg · {p:.0f}%" for kg, p in zip(vals, pcts)],
        textposition="outside", textfont=dict(color=TEXT, size=11),
        hovertemplate="%{y}: %{x:.1f} kg<extra></extra>"))
    _layout(fig, title)
    fig.update_layout(width=1000, height=max(260, 34 * len(comps) + 90), showlegend=False)
    fig.update_xaxes(title=f"kilograms (of {dry_mass_kg:.0f} kg dry mass)", color=MUTED,
                     gridcolor=GRID, zeroline=False, rangemode="tozero")
    fig.update_yaxes(color=MUTED, showgrid=False, zeroline=False, automargin=True)
    return fig
