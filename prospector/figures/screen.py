"""Population-screening figures: the reachable volume and what lives inside it.

The reachability dome is the delta-v filter drawn as a surface, covering everywhere that
``lowthrust_dv(a, e, i)`` equals the budget, so the picture and the screen cannot disagree.
Alongside it: how the population's orbits are distributed, and, over the reachable targets that
have been described, their grades and how they compare.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from prospector.constants import EARTH_ORBIT_ECC, EARTH_ORBIT_INC_DEG, EARTH_ORBIT_SMA_AU
from prospector.enrichment.schema import TIER_ORDER
from prospector.figures.theme import (
    BG,
    DANGER,
    DOME,
    DV_SCALE,
    EARTH,
    FOCUS,
    GRID,
    MUTED,
    PICKED,
    PLAY_GREEN,
    REFERENCE,
    SURFACE,
    TEXT,
    _layout,
)
from prospector.solvers.edelbaum import lowthrust_dv

# A color per mission-target tier, running best to worst: blue/teal, green, amber, grey, red. The
# order itself comes from the contract rather than being restated here, so a tier added there
# cannot go missing from the legend.
_TIER_ORDER = list(TIER_ORDER)


TIER_COLORS = {"S": DOME, "A": PLAY_GREEN, "B": EARTH, "C": MUTED, "D": DANGER}


UNKNOWN_TIER = "-"      # bucket/label for a reachable target not yet (or unsuccessfully) enriched


SIZE_LABEL = "orbit size (Earth = 1)"


SHAPE_LABEL = "orbit shape (eccentricity)"


TILT_LABEL = "orbit tilt (deg)"


def _grid_extent(df: pd.DataFrame | None, dv_budget: float,
                 reference: dict | None) -> tuple[float, float, float]:
    """Pick (size, eccentricity, tilt) plot bounds wide enough to hold the reachable
    region, any supplied population, and a named reference target.

    The dome itself sets a sensible floor: at e=0 the budget reaches some maximum orbit size, so
    the box always frames the surface even before any population is loaded.
    """
    # Largest circular orbit the budget can reach, as a size floor for the box.
    a_axis = np.linspace(0.3, 6.0, 400)
    reachable_circular = lowthrust_dv(a_axis, 0.0, 0.0) <= dv_budget
    a_hi = float(a_axis[reachable_circular].max() / EARTH_ORBIT_SMA_AU) * 1.25 if reachable_circular.any() else 3.0
    e_hi, i_hi = 0.4, 30.0
    if df is not None and len(df):
        reach = df["lowthrust_dv"].to_numpy(float) <= dv_budget
        if reach.any():
            a_hi = max(a_hi, float(np.nanmax(df["a"].to_numpy(float)[reach]) / EARTH_ORBIT_SMA_AU) * 1.2)
            e_hi = max(e_hi, float(np.nanmax(df["e"].to_numpy(float)[reach])) * 1.3)
            i_hi = max(i_hi, float(np.nanmax(df["i"].to_numpy(float)[reach])) * 1.4)
    if reference:
        a_hi = max(a_hi, reference["a"] / EARTH_ORBIT_SMA_AU * 1.15)
        e_hi = max(e_hi, reference["e"] * 1.2)
        i_hi = max(i_hi, reference["i"] * 1.2)
    return min(a_hi, 5.0), min(e_hi, 0.95), min(i_hi, 60.0)


def _split_reachable(df: pd.DataFrame, dv_budget: float):
    """Return (reachable_df, unreachable_df) by the dV budget; drops non-finite dV."""
    dv = df["lowthrust_dv"].to_numpy(float)
    finite = np.isfinite(dv)
    reach = finite & (dv <= dv_budget)
    return df[reach], df[finite & ~reach]


def reachability_dome_3d(dv_budget: float, df: pd.DataFrame | None = None,
                         reference: dict | None = None, config_name: str = "",
                         grid: int = 48, max_points: int = 4000,
                         highlights=None) -> go.Figure:
    """What is reachable, drawn as a surface in orbit size, shape and tilt.

    The surface is everywhere that ``lowthrust_dv(a, e, i)`` equals the budget, and every orbit
    inside it, toward Earth, is reachable. This is the screen's own filter drawn directly. No
    population is needed for it, since it depends only on the vehicle's budget.

    Pass ``df`` (with columns a, e, i, lowthrust_dv) to scatter a population on top: reachable
    points colored by dV, over-budget points faint for context. ``reference`` (dict with a, e, i,
    label) stars a named calibration target. ``grid`` sets the isosurface resolution;
    ``max_points`` caps the scatter so the browser stays smooth.

    ``highlights`` marks individual bodies by name, being the focus target and the row picked in
    the table, so a chosen target is findable in a cloud of thousands (see
    :func:`_add_highlights_3d`).
    """
    a_hi, e_hi, i_hi = _grid_extent(df, dv_budget, reference)

    # Evaluate the dV field on a regular (a, e, i) lattice and let Plotly extract the budget level
    # set. indexing="ij" keeps x/y/z/value ravel order consistent.
    a_axis = np.linspace(0.3 * EARTH_ORBIT_SMA_AU, a_hi * EARTH_ORBIT_SMA_AU, grid)
    e_axis = np.linspace(0.0, e_hi, grid)
    i_axis = np.linspace(0.0, i_hi, grid)
    A, E, Inc = np.meshgrid(a_axis, e_axis, i_axis, indexing="ij")
    DV = lowthrust_dv(A, E, Inc)

    fig = go.Figure()
    fig.add_trace(go.Isosurface(
        x=(A / EARTH_ORBIT_SMA_AU).ravel(), y=E.ravel(), z=Inc.ravel(), value=DV.ravel(),
        isomin=dv_budget, isomax=dv_budget, surface_count=1,
        colorscale=[[0, DOME], [1, DOME]], showscale=False, opacity=0.28,
        caps=dict(x_show=False, y_show=False, z_show=False),
        name="reach limit", showlegend=True,
        hovertemplate="budget surface<br>%{value:.2f} km/s<extra></extra>",
    ))

    if df is not None and len(df):
        reach, miss = _split_reachable(df, dv_budget)
        if len(miss):
            miss = _subsample(miss, max_points // 2)
            fig.add_trace(go.Scatter3d(
                x=miss["a"].to_numpy(float) / EARTH_ORBIT_SMA_AU, y=miss["e"], z=miss["i"],
                mode="markers", name="over budget",
                marker=dict(size=2, color=MUTED, opacity=0.18),
                hoverinfo="skip",
            ))
        if len(reach):
            # When the reachable set has been characterized and a desirability filter applied,
            # split it: the selected (kept) orbits stay dV-colored; the filtered-out ones drop to
            # faint markers, showing the filter inside the same reachable volume.
            if "selected" in reach.columns:
                kept = _subsample(reach[reach["selected"].fillna(False).astype(bool)], max_points)
                filtered = _subsample(reach[~reach["selected"].fillna(False).astype(bool)], max_points // 2)
            else:
                kept, filtered = _subsample(reach, max_points), reach.iloc[0:0]
            if len(filtered):
                fig.add_trace(go.Scatter3d(
                    x=filtered["a"].to_numpy(float) / EARTH_ORBIT_SMA_AU, y=filtered["e"], z=filtered["i"],
                    mode="markers", name="filtered (desirability)",
                    marker=dict(size=2, color=MUTED, opacity=0.3),
                    customdata=_designations(filtered),
                    hovertemplate="%{customdata}<br>filtered by desirability<extra></extra>",
                ))
            if len(kept):
                fig.add_trace(go.Scatter3d(
                    x=kept["a"].to_numpy(float) / EARTH_ORBIT_SMA_AU, y=kept["e"], z=kept["i"],
                    mode="markers", name=("selected" if len(filtered) else "reachable"),
                    marker=dict(size=3, color=kept["lowthrust_dv"], colorscale=DV_SCALE,
                                cmin=0, cmax=dv_budget, line=dict(width=0),
                                colorbar=dict(title="dV (km/s)", x=1.02, len=0.6)),
                    customdata=_designations(kept),
                    hovertemplate=("%{customdata}<br>size %{x:.2f}· e %{y:.3f}· "
                                   "i %{z:.1f}°<extra></extra>"),
                ))

    _add_origin_3d(fig)
    if reference:
        _add_reference_3d(fig, reference)
    # Last, so the highlighted bodies sit on top of the population scatter rather than under it.
    _add_highlights_3d(fig, highlights)

    title = (f"{config_name} - reachable volume" if config_name else "Reachable volume")
    _layout(fig, f"{title}  ·  drag to rotate  ·  budget {dv_budget:.2f} km/s")
    fig.update_layout(
        height=620,
        scene=dict(
            xaxis=dict(title=SIZE_LABEL, backgroundcolor=SURFACE, gridcolor=GRID,
                       color=MUTED, range=[0.3, a_hi]),
            yaxis=dict(title=SHAPE_LABEL, backgroundcolor=SURFACE, gridcolor=GRID,
                       color=MUTED, range=[0.0, e_hi]),
            zaxis=dict(title=TILT_LABEL, backgroundcolor=SURFACE, gridcolor=GRID,
                       color=MUTED, range=[0.0, i_hi]),
            camera=dict(eye=dict(x=1.7, y=-1.7, z=0.9)),
        ),
    )
    return fig


_ELEMENTS = (
    ("a", "semimajor axis a (AU)", EARTH_ORBIT_SMA_AU),
    ("e", "eccentricity e", EARTH_ORBIT_ECC),
    ("i", "inclination i (deg)", EARTH_ORBIT_INC_DEG),
)


def element_distributions(df: pd.DataFrame, dv_budget: float | None = None,
                          config_name: str = "", bins: int = 50,
                          clip_percentile: float = 99.0,
                          selected_mask=None) -> go.Figure:
    """Side-by-side histograms of the population's a, e, and i, layered by screen stage.

    Shows the shape of the whole set the screen runs over in a muted colour, the reachable part the
    vehicle's delta-v buys in teal, and, once those targets have been described and a filter
    applied, the selected ones on top in gold. Each panel therefore nests: everything, then what is
    reachable, then what is also worth reaching, in one window.

    The population has a long tail of a few extreme orbits which, left in, squashes the bulk of
    every histogram against the left edge. So each panel is framed to the middle
    ``clip_percentile`` of its own values, with a count of what falls outside noted in the axis
    title. The extremes are acknowledged rather than quietly dropped.
    """
    titles = [label for _, label, _ in _ELEMENTS]
    fig = make_subplots(rows=1, cols=3, subplot_titles=titles)

    reach_mask = None
    if dv_budget is not None and "lowthrust_dv" in df.columns:
        dv = df["lowthrust_dv"].to_numpy(float)
        reach_mask = np.isfinite(dv) & (dv <= dv_budget)
    sel_mask = np.asarray(selected_mask, bool) if selected_mask is not None else None
    # Only draw the selected layer when it genuinely narrows the reachable set; otherwise it would
    # sit on the reachable layer and read as visual noise.
    show_selected = sel_mask is not None and reach_mask is not None and sel_mask.sum() < reach_mask.sum()

    for col, (key, _, earth_val) in enumerate(_ELEMENTS, start=1):
        all_vals = pd.to_numeric(df[key], errors="coerce").to_numpy(float)
        finite = np.isfinite(all_vals)
        lo, hi = _robust_bounds(all_vals[finite], earth_val, clip_percentile)
        xbins = dict(start=lo, end=hi, size=(hi - lo) / bins)
        n_clipped = int(np.sum(finite & (all_vals > hi)))

        in_frame = finite & (all_vals >= lo) & (all_vals <= hi)
        fig.add_trace(go.Histogram(
            x=all_vals[in_frame], xbins=xbins, autobinx=False, name="all",
            marker_color=MUTED, opacity=0.55, legendgroup="all", showlegend=(col == 1),
        ), row=1, col=col)
        if reach_mask is not None:
            fig.add_trace(go.Histogram(
                x=all_vals[reach_mask & in_frame], xbins=xbins, autobinx=False, name="reachable",
                marker_color=DOME, opacity=0.75, legendgroup="reachable",
                showlegend=(col == 1),
            ), row=1, col=col)
        if show_selected:
            fig.add_trace(go.Histogram(
                x=all_vals[sel_mask & in_frame], xbins=xbins, autobinx=False, name="selected",
                marker_color=REFERENCE, opacity=0.9, legendgroup="selected",
                showlegend=(col == 1),
            ), row=1, col=col)
        # Mark Earth's own element as the origin every transfer reshapes away from.
        if lo <= earth_val <= hi:
            fig.add_vline(x=earth_val, line=dict(color=EARTH, width=1.5, dash="dot"),
                          row=1, col=col)
        axis_title = titles[col - 1] + (f"  (+{n_clipped} beyond)" if n_clipped else "")
        fig.update_xaxes(title_text=axis_title, range=[lo, hi], gridcolor=GRID,
                         color=MUTED, zeroline=False, row=1, col=col)

    n = len(df)
    suffix = f" · {int(reach_mask.sum())} reachable" if reach_mask is not None else ""
    if show_selected:
        suffix += f" · {int(sel_mask.sum())} selected"
    title = (f"{config_name} - population structure" if config_name
             else "Population structure")
    _layout(fig, f"{title}  ·  {n} objects{suffix}")
    fig.update_layout(height=380, barmode="overlay", bargap=0.02)
    fig.update_yaxes(gridcolor=GRID, color=MUTED, zeroline=False)
    fig.update_yaxes(title_text="count", row=1, col=1)
    fig.update_annotations(font=dict(color=TEXT, size=12))  # subplot titles -> readable
    return fig


def _robust_bounds(values: np.ndarray, earth_val: float, hi_pct: float) -> tuple[float, float]:
    """Frame an element to its central distribution: 0 (or the data min) up to the
    ``hi_pct`` percentile, always wide enough to include Earth's own value."""
    if values.size == 0:
        return 0.0, 1.0
    lo = min(float(values.min()), earth_val)
    lo = max(lo, 0.0) if earth_val >= 0 else lo
    hi = max(float(np.nanpercentile(values, hi_pct)), earth_val)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _tier_series(df: pd.DataFrame) -> pd.Series:
    """The ``tier`` column as plain strings, or an all-unknown series if absent."""
    if "tier" in df.columns:
        return df["tier"]
    return pd.Series([None] * len(df), index=df.index, dtype=object)


def tier_breakdown(df: pd.DataFrame, config_name: str = "") -> go.Figure:
    """Bar chart of how many reachable targets fall in each mission-target tier.

    Grades run S at the best down to D, with a final 'unknown' bar counting reachable targets
    nothing is known about, either because the lookup was unavailable or because it failed. That
    bar is kept visible rather than dropped, so the gap between "reachable" and "described" is
    never hidden.
    """
    tier = _tier_series(df)
    counts = [int((tier == t).sum()) for t in _TIER_ORDER]
    unknown = int(len(tier) - sum(counts)) if len(tier) else 0
    labels = _TIER_ORDER + [UNKNOWN_TIER]
    counts.append(unknown)
    colors = [TIER_COLORS[t] for t in _TIER_ORDER] + [GRID]

    fig = go.Figure(go.Bar(x=labels, y=counts, marker_color=colors,
                           text=[c or "" for c in counts], textposition="outside"))
    title = f"{config_name} - mission-target tiers" if config_name else "Mission-target tiers"
    _layout(fig, f"{title}  ·  {sum(counts)} reachable")
    fig.update_layout(height=300, bargap=0.3, showlegend=False)
    fig.update_xaxes(title_text="tier (S best → D · - = not characterized)",
                     color=MUTED, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title_text="targets", color=MUTED, gridcolor=GRID, zeroline=False)
    return fig


def desirability_scatter(df: pd.DataFrame, config_name: str = "") -> go.Figure:
    """Scatter of diameter vs rotation period, one series per tier.

    Size against spin: bigger means more material, on a log axis, and faster spin means a body less
    likely to hold together while something works alongside it. Targets missing either property are
    left off the scatter and counted in the title, so nothing goes missing quietly.
    """
    fig = go.Figure()
    name_col = next((c for c in ("full_name", "name") if c in df.columns), None)
    have = {"diameter_m", "period_h"}.issubset(df.columns)
    plotted = 0
    if have:
        diameter = pd.to_numeric(df["diameter_m"], errors="coerce")
        period = pd.to_numeric(df["period_h"], errors="coerce")
        tier = _tier_series(df)
        for t in _TIER_ORDER + [UNKNOWN_TIER]:
            in_tier = (tier == t) if t != UNKNOWN_TIER else ~tier.isin(_TIER_ORDER)
            mask = in_tier & diameter.notna() & period.notna()
            if not mask.any():
                continue
            plotted += int(mask.sum())
            names = df.loc[mask, name_col].astype(str) if name_col else pd.Series([""] * int(mask.sum()))
            tax = df.loc[mask, "taxonomy"].astype(str) if "taxonomy" in df.columns else pd.Series(["-"] * int(mask.sum()))
            dv = (pd.to_numeric(df.loc[mask, "lowthrust_dv"], errors="coerce")
                  if "lowthrust_dv" in df.columns else pd.Series([float("nan")] * int(mask.sum())))
            customdata = np.column_stack([names.to_numpy(str), tax.to_numpy(str), dv.to_numpy(float)])
            fig.add_trace(go.Scatter(
                x=diameter[mask], y=period[mask], mode="markers",
                name=("unknown" if t == UNKNOWN_TIER else t),
                marker=dict(size=9, color=TIER_COLORS.get(t, GRID), line=dict(width=0.5, color=BG)),
                customdata=customdata,
                hovertemplate=("<b>%{customdata[0]}</b><br>tier " + ("unknown" if t == UNKNOWN_TIER else t)
                               + "<br>taxonomy %{customdata[1]}<br>diameter %{x:.0f} m"
                               + "<br>period %{y:.2f} h<br>dV %{customdata[2]:.2f} km/s<extra></extra>"),
            ))
    title = f"{config_name} - size vs spin by tier" if config_name else "Size vs spin by tier"
    omitted = len(df) - plotted
    suffix = f"  ·  {plotted} shown" + (f", {omitted} missing size/spin" if omitted > 0 else "")
    _layout(fig, title + suffix)
    fig.update_layout(height=420, legend=dict(title="tier"))
    fig.update_xaxes(title_text="diameter (m, log)", type="log", color=MUTED, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title_text="rotation period (h)", color=MUTED, gridcolor=GRID, zeroline=False)
    return fig


def _add_origin_3d(fig: go.Figure) -> None:
    """Star Earth's orbit, the origin of every transfer."""
    fig.add_trace(go.Scatter3d(
        x=[EARTH_ORBIT_SMA_AU], y=[EARTH_ORBIT_ECC], z=[EARTH_ORBIT_INC_DEG], mode="markers+text", name="Earth's orbit",
        marker=dict(symbol="diamond", size=6, color=EARTH, line=dict(color=BG, width=1)),
        text=["Earth"], textposition="top center", textfont=dict(color=EARTH),
        hovertemplate="Earth's orbit (origin)<extra></extra>",
    ))


def _add_reference_3d(fig: go.Figure, reference: dict) -> None:
    fig.add_trace(go.Scatter3d(
        x=[reference["a"] / EARTH_ORBIT_SMA_AU], y=[reference["e"]], z=[reference["i"]],
        mode="markers+text", name=reference["label"],
        marker=dict(symbol="diamond", size=6, color=REFERENCE, line=dict(color=BG, width=1)),
        text=[reference["label"]], textposition="top center",
        textfont=dict(color=REFERENCE),
        hovertemplate=f"{reference['label']}<extra></extra>",
    ))


# How each highlighted body is drawn. The two roles differ in shape as well as colour, so the
# picture survives greyscale:
#
#   focus   a filled diamond in the focus accent, matching Earth and the reference target
#   picked  an open ring in near-white, the row currently picked in the table
#
# A body that is both gets the ring around the diamond.
_HIGHLIGHT_STYLES = {
    "focus": {"symbol": "diamond", "size": 9, "color": FOCUS, "name": "focus target"},
    "picked": {"symbol": "circle-open", "size": 14, "color": PICKED, "name": "picked in table"},
}


def _add_highlights_3d(fig: go.Figure, highlights) -> None:
    """Mark named bodies in the reachable volume: the focus target and the picked table row.

    Each highlight is ``{a, e, i, role}`` plus an optional ``label`` (drawn beside the marker) --
    ``role`` selects the styling above. Anything with a non-finite element is skipped rather than
    collapsing to the origin, where it would read as a body at Earth's own orbit.

    They are drawn as their own traces rather than by recolouring a point in the population
    scatter, for two reasons. The scatter is thinned out to keep the browser responsive, so the
    very body someone picked may not be in it. And a target can sit outside the reachable set, or
    outside the filter, entirely, where there is no point to recolour.
    """
    for hl in highlights or ():
        style = _HIGHLIGHT_STYLES.get(hl.get("role"))
        if style is None:
            continue
        try:
            a, e, i = float(hl["a"]), float(hl["e"]), float(hl["i"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(np.isfinite(v) for v in (a, e, i)):
            continue
        label = str(hl.get("label") or "").strip()
        fig.add_trace(go.Scatter3d(
            x=[a / EARTH_ORBIT_SMA_AU], y=[e], z=[i],
            mode="markers+text" if label else "markers",
            name=style["name"],
            marker=dict(symbol=style["symbol"], size=style["size"], color=style["color"],
                        line=dict(color=style["color"], width=2)),
            text=[label] if label else None,
            textposition="bottom center",
            textfont=dict(color=style["color"], size=11),
            hovertemplate=(f"{label or style['name']}<br>{style['name']}<br>"
                           "size %{x:.2f}· e %{y:.3f}· i %{z:.1f}°<extra></extra>"),
        ))


def _designations(df: pd.DataFrame) -> np.ndarray:
    """A label per row for hover, from a name/designation column if present."""
    for col in ("full_name", "name", "designation", "pdes"):
        if col in df.columns:
            return df[col].astype(str).to_numpy()
    return np.array([f"orbit {k}" for k in range(len(df))])


def _subsample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Evenly thin a frame to at most n rows (order-preserving, deterministic)."""
    if n <= 0 or len(df) <= n:
        return df
    keep = np.linspace(0, len(df) - 1, n).astype(int)
    return df.iloc[keep]
