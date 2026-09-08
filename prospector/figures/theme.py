"""The dark space-navy palette and the one place a figure's layout is applied.

Every figure module styles itself through :func:`_layout` and the colours here, so the whole set
looks like one thing, and so the report's conversion for print
(:func:`prospector.reporting.document.print_figure`) has one dark theme to invert.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go

_MJD2000_EPOCH = datetime(2000, 1, 1)


def mjd2000_to_datetime(mjd2000):
    """MJD2000 (days since 2000-01-01) -> datetime, vectorized over arrays."""
    arr = np.atleast_1d(np.asarray(mjd2000, float))
    out = [_MJD2000_EPOCH + timedelta(days=float(m)) for m in arr]
    return out if np.ndim(mjd2000) else out[0]


# Palette mirrors ui/theme.py so a screenshot drops cleanly into the app or a proposal. Kept as
# plain constants here (not imported) to honor the pure-core / no-UI-import rule.
BG = "#080a0f"


SURFACE = "#12151e"


TEXT = "#e1e6f0"


MUTED = "#6a728a"


GRID = "#252b3d"


DOME = "#5294ff"        # blue -- the reachable region / budget surface; selected cell


EARTH = "#fbaf2a"       # amber -- Earth's orbit on the reachability charts


REFERENCE = "#FFD700"   # gold -- a named calibration target


FOCUS = "#ff6a3c"       # orange -- the focus target, mirroring the app's focus accent


PICKED = "#e6e9f2"      # near-white -- the row picked in a table, a ring around its point


DV_SCALE = "Viridis"    # dV colormap for reachable points


PORKCHOP_SCALE = "thermal"    # porkchop colours, reversed so the cheap region runs hot


EARTH_BLUE = "#4aa3ff"  # blue -- Earth's orbit + Earth sphere on the trajectory plot






ASTEROID_ORANGE = "#ff9f43"   # orange -- the target's orbit + asteroid sphere


DEPARTURE = "#e6e9f2"   # near-white -- the departure point (kept off-blue so it isn't "Earth")


SPACECRAFT = "#ffffff"  # white -- the moving spacecraft marker


THRUST_SCALE = "RdYlGn"       # throttle colormap: red (coast) -> green (full thrust)


PLAY_GREEN = "#2ecc71"  # animation Play button (dark glyph on a bright fill, readable)


PAUSE_AMBER = "#fbaf2a"  # animation Pause button


ECCEN = "#c58cf0"       # light violet -- eccentricity track + radial thrust on the diagnostics


DANGER = "#e74c3c"      # red -- the weakest tier / structurally marginal targets


def _layout(fig: go.Figure, title: str) -> go.Figure:
    """Apply the dark space-navy theme to a figure in one place."""
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, size=15)),
        paper_bgcolor=BG,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT, size=12),
        margin=dict(l=10, r=10, t=46, b=10),
        legend=dict(bgcolor="rgba(22,28,51,0.7)", bordercolor=GRID, borderwidth=1),
    )
    return fig
