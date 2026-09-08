"""Design-space figures: the vehicle trade scatter and its supporting analysis.

Draws what :mod:`prospector.trades.design_search` writes out: any numeric column against any other,
coloured by a third, with whether a design flies and whether it can be built shown in the marker.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from prospector.figures.theme import (
    ASTEROID_ORANGE,
    BG,
    DANGER,
    DOME,
    DV_SCALE,
    EARTH,
    EARTH_BLUE,
    ECCEN,
    GRID,
    MUTED,
    PLAY_GREEN,
    _layout,
)

# One stable color per engine family across every chart in the set.
ENGINE_CYCLE = (DOME, EARTH, PLAY_GREEN, ECCEN, EARTH_BLUE, DANGER,
                 ASTEROID_ORANGE, "#f0e68c")


def trade_scatter(df: pd.DataFrame, x: str, y: str, color: str, *,
                  x_label: str, y_label: str) -> go.Figure:
    """A configurable scatter over vehicle-search evaluations: any column on each axis.

    ``x`` and ``y`` name numeric columns and ``color`` names the column to colour by. A column of
    categories, such as the engine fitted, draws one series per value using the fixed engine
    colours; a numeric one uses a continuous scale. ``x_label`` and ``y_label`` title the axes.
    Rows missing either plotted value are dropped without comment, and each point's dry mass,
    propellant and engine appear on hover.
    """
    d = df.copy()
    for c in (x, y, color):
        if c in d.columns and c not in ("engine", "propellant"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    xv = pd.to_numeric(d.get(x, pd.Series(dtype=float, index=d.index)), errors="coerce")
    yv = pd.to_numeric(d.get(y, pd.Series(dtype=float, index=d.index)), errors="coerce")
    d = d[xv.notna() & yv.notna()]

    has_slug = "slug" in d.columns and d["slug"].notna().any()

    def _ident(g):
        # Object array so the memorable name (a string) rides alongside the numerics; the numeric
        # hover fields still format cleanly (%{...:.0f}) off their float entries.
        return np.column_stack([
            pd.to_numeric(g.get("dry_kg", pd.Series(np.nan, index=g.index)),
                          errors="coerce"),
            pd.to_numeric(g.get("prop_kg", pd.Series(np.nan, index=g.index)),
                          errors="coerce"),
            pd.to_numeric(g.get("n_engines", pd.Series(np.nan, index=g.index)),
                          errors="coerce"),
            g.get("slug", pd.Series("", index=g.index)).astype(str),
        ]).astype(object)

    # The memorable design name leads the hover when the rows carry one (older sessions don't).
    name_line = "<b>%{customdata[3]}</b><br>" if has_slug else ""
    hover = (f"{name_line}{x_label} %{{x}}<br>{y_label} %{{y}}<br>"
             "dry %{customdata[0]:.0f} + prop %{customdata[1]:.0f} kg · "
             "×%{customdata[2]:.0f}")
    fig = go.Figure()
    categorical = color in d.columns and (
        d[color].dtype == object or color == "engine"
        or d[color].map(lambda v: isinstance(v, bool)).any())
    if categorical and color in d.columns:
        values = sorted(d[color].astype(str).fillna("-").unique())
        palette = {v: ENGINE_CYCLE[i % len(ENGINE_CYCLE)]
                   for i, v in enumerate(values)}
        for val, g in d.groupby(d[color].astype(str).fillna("-")):
            fig.add_trace(go.Scatter(
                x=pd.to_numeric(g[x], errors="coerce"),
                y=pd.to_numeric(g[y], errors="coerce"),
                mode="markers", name=str(val),
                marker=dict(color=palette[str(val)], size=8, opacity=0.85,
                            line=dict(width=0.5, color=BG)),
                customdata=_ident(g),
                hovertemplate=hover + "<extra>%{fullData.name}</extra>"))
        fig.update_layout(legend=dict(title=color))
    else:
        cv = (pd.to_numeric(d.get(color), errors="coerce")
              if color in d.columns else pd.Series(np.nan, index=d.index))
        fig.add_trace(go.Scatter(
            x=pd.to_numeric(d[x], errors="coerce"),
            y=pd.to_numeric(d[y], errors="coerce"),
            mode="markers",
            marker=dict(color=cv, colorscale=DV_SCALE, size=8, opacity=0.85,
                        line=dict(width=0.5, color=BG),
                        colorbar=dict(title=dict(text=color, font=dict(color=MUTED)),
                                      tickfont=dict(color=MUTED))),
            customdata=_ident(d),
            hovertemplate=hover + "<extra></extra>"))
    _layout(fig, f"{y_label} vs {x_label}")
    fig.update_layout(height=440)
    fig.update_xaxes(title_text=x_label, color=MUTED, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title_text=y_label, color=MUTED, gridcolor=GRID, zeroline=False)
    return fig


def _truthy(series: pd.Series) -> pd.Series:
    """Robust bool view of a verdict column (bools, CSV strings, or NaN)."""
    return series.map(lambda v: v is True or str(v).strip().lower() == "true")


def vehicle_search_analysis(df: pd.DataFrame,
                            engine_specs: dict) -> list[tuple[go.Figure, str]]:
    """Trend charts for a set of vehicle-search evaluations: ``(figure, takeaway)``
    pairs, each isolating one driver of the design space.

    ``df`` is one row per vehicle evaluated, and ``engine_specs`` maps each engine name to its
    ``{"thrust_mN", "power_W", "isp_s"}`` per unit. A chart whose columns are missing, as in an
    older session, is simply left out rather than raising.
    """
    d = df.copy()
    for c in ("dry_kg", "prop_kg", "wet_kg", "n_engines", "margin_kms", "belt_days",
              "array_kg", "min_dry_kg", "prop_dry_ratio", "total_tof_days",
              "payload_capacity_kg", "build_cost_musd", "eol_factor"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if "engine" not in d.columns or not len(d):
        return []
    d = d[d["engine"].astype(str).isin(engine_specs)]
    if not len(d):
        return []
    spec = d["engine"].astype(str).map(engine_specs.get)
    d["n_engines"] = d["n_engines"].fillna(1).astype(int)
    if "wet_kg" not in d.columns or d["wet_kg"].isna().all():
        d["wet_kg"] = d["dry_kg"] + d["prop_kg"]
    if "prop_dry_ratio" not in d.columns:
        d["prop_dry_ratio"] = d["prop_kg"] / d["dry_kg"]
    d["accel_mm_s2"] = [s["thrust_mN"] * n / w if w else np.nan
                        for s, n, w in zip(spec, d["n_engines"], d["wet_kg"])]
    d["mn_per_w"] = [s["thrust_mN"] / s["power_W"] for s in spec]
    d["isp_s"] = [s["isp_s"] for s in spec]
    d["closes"] = (_truthy(d["success"]) if "success" in d.columns
                   else pd.Series(False, index=d.index))
    d["builds"] = (_truthy(d["buildable"]) if "buildable" in d.columns
                   else pd.Series(False, index=d.index))
    d["viable"] = d["closes"] & d["builds"]
    colors = {eng: ENGINE_CYCLE[i % len(ENGINE_CYCLE)]
              for i, eng in enumerate(sorted(d["engine"].astype(str).unique()))}
    charts: list[tuple[go.Figure, str]] = []

    def _axes(fig, x_title, y_title, height=340):
        fig.update_xaxes(title_text=x_title, color=MUTED, gridcolor=GRID,
                         zeroline=False)
        fig.update_yaxes(title_text=y_title, color=MUTED, gridcolor=GRID,
                         zeroline=False)
        fig.update_layout(height=height)
        return fig

    hover = ("dry %{customdata[0]:.0f} + prop %{customdata[1]:.0f} kg · "
             "×%{customdata[2]}<extra>%{fullData.name}</extra>")

    def _scatter_by_engine(frame, x, y):
        fig = go.Figure()
        for eng, g in frame.groupby(frame["engine"].astype(str)):
            fig.add_trace(go.Scatter(
                x=g[x], y=g[y], mode="markers", name=eng,
                marker=dict(color=colors[eng], size=7, opacity=0.8,
                            line=dict(width=0.5, color=BG)),
                customdata=np.stack([g["dry_kg"], g["prop_kg"],
                                     g["n_engines"]], axis=-1),
                hovertemplate=hover))
        return fig

    # 1 - belt residence vs launch acceleration (the spiral's master variable)
    sub = d.dropna(subset=["belt_days", "accel_mm_s2"])
    if len(sub) >= 3:
        fig = _scatter_by_engine(sub, "accel_mm_s2", "belt_days")
        charts.append((
            _axes(_layout(fig, "Radiation-belt time vs launch acceleration"),
                  "initial acceleration (mN/kg ≡ mm/s²)", "days in the belts"),
            "Belt residence collapses onto one curve set by thrust per kg at "
            "launch - which engine provides the thrust barely matters. Roughly, "
            "doubling acceleration halves the time (and the array damage) in "
            "the belts."))

    # 2 - array mass vs engine count, one line per family
    sub = d.dropna(subset=["array_kg"])
    if len(sub) >= 3 and sub["n_engines"].nunique() > 1:
        med = (sub.groupby([sub["engine"].astype(str), "n_engines"])["array_kg"]
               .median().reset_index())
        fig = go.Figure()
        for eng, g in med.groupby("engine"):
            g = g.sort_values("n_engines")
            fig.add_trace(go.Scatter(
                x=g["n_engines"], y=g["array_kg"], mode="lines+markers", name=eng,
                line=dict(color=colors[eng], width=2), marker=dict(size=8),
                hovertemplate="×%{x}: %{y:.0f} kg array"
                              "<extra>%{fullData.name}</extra>"))
        fig.update_xaxes(dtick=1)
        charts.append((
            _axes(_layout(fig, "Solar array mass vs engine count (median)"),
                  "engines on the stack", "array mass (kg)"),
            "Adding engines always grows the array: each unit adds 33–50% more "
            "power demand, while the faster belt transit it buys only improves "
            "end-of-life output by 10–15%. Acceleration bought with unit count "
            "is paid for in array (and the buildable dry floor rises with it)."))

    # 3 - the buildable floor vs thruster efficiency (thrust per watt)
    sub = d.dropna(subset=["min_dry_kg"])
    if len(sub) >= 3:
        med = (sub.groupby([sub["engine"].astype(str), "n_engines"])
               .agg(mn_per_w=("mn_per_w", "first"), min_dry=("min_dry_kg", "median"))
               .reset_index())
        fig = go.Figure()
        for eng, g in med.groupby("engine"):
            g = g.sort_values("n_engines")
            fig.add_trace(go.Scatter(
                x=g["mn_per_w"], y=g["min_dry"], mode="markers+text", name=eng,
                text=[f"×{n}" for n in g["n_engines"]], textposition="middle right",
                textfont=dict(color=MUTED, size=10),
                marker=dict(color=colors[eng], size=9,
                            line=dict(width=0.5, color=BG)),
                hovertemplate="%{text}: floor %{y:.0f} kg dry at %{x:.3f} mN/W"
                              "<extra>%{fullData.name}</extra>"))
        charts.append((
            _axes(_layout(fig, "Minimum buildable dry mass vs thrust-per-watt"),
                  "engine efficiency (mN per W)", "min buildable dry (kg)"),
            "The thruster-efficiency paradigm: each engine family is one column "
            "(thrust/W is a property of the engine, not the stack), climbing as "
            "units are added. Efficient engines (right) keep the buildable floor "
            "low at any count; inefficient ones (left) price themselves out of "
            "light vehicles before trajectory physics even enters."))

    # 4 - Isp's role: the prop fraction needed to close
    sub = d.dropna(subset=["margin_kms", "prop_dry_ratio"])
    if len(sub) >= 3:
        fig = go.Figure(go.Scatter(
            x=sub["prop_dry_ratio"], y=sub["margin_kms"], mode="markers",
            marker=dict(color=sub["isp_s"], colorscale=DV_SCALE, size=7,
                        opacity=0.85, line=dict(width=0.5, color=BG),
                        colorbar=dict(title=dict(text="Isp (s)",
                                                 font=dict(color=MUTED)),
                                      tickfont=dict(color=MUTED))),
            customdata=np.stack([sub["dry_kg"], sub["prop_kg"], sub["n_engines"],
                                 sub["isp_s"]], axis=-1),
            hovertemplate=("dry %{customdata[0]:.0f} + prop %{customdata[1]:.0f} kg "
                           "· ×%{customdata[2]} · Isp %{customdata[3]:.0f} s<br>"
                           "prop/dry %{x:.2f} → margin %{y:+.2f} km/s"
                           "<extra></extra>")))
        fig.add_hline(y=0.0, line=dict(color=MUTED, width=1, dash="dot"))
        charts.append((
            _axes(_layout(fig, "ΔV margin vs propellant fraction, colored by Isp"),
                  "prop / dry mass", "margin (km/s)"),
            "Isp sets where a design crosses the zero-margin line: higher-Isp "
            "stacks (brighter) close at lower prop/dry, because capability is "
            "Isp × ln(wet/dry) while the mission cost is nearly fixed. Below "
            "the dotted line nothing flies, however buildable it is."))

    # 5 - the viable frontier: the only designs worth building
    sub = d[d["viable"]].dropna(subset=["total_tof_days", "wet_kg"]) \
        if "total_tof_days" in d.columns else d.iloc[0:0]
    if len(sub) >= 2:
        fig = go.Figure()
        for eng, g in sub.groupby(sub["engine"].astype(str)):
            fig.add_trace(go.Scatter(
                x=g["total_tof_days"], y=g["wet_kg"], mode="markers", name=eng,
                marker=dict(color=colors[eng], size=9, opacity=0.85,
                            line=dict(width=0.5, color=BG)),
                customdata=np.stack(
                    [g["dry_kg"], g["prop_kg"], g["n_engines"],
                     g.get("payload_capacity_kg", pd.Series(np.nan, index=g.index)),
                     g.get("build_cost_musd", pd.Series(np.nan, index=g.index))],
                    axis=-1),
                hovertemplate=("dry %{customdata[0]:.0f} + prop "
                               "%{customdata[1]:.0f} kg · ×%{customdata[2]}<br>"
                               "payload %{customdata[3]:.0f} kg · "
                               "$%{customdata[4]:.0f}M<br>mission %{x:.0f} d, "
                               "wet %{y:.0f} kg<extra>%{fullData.name}</extra>")))
        charts.append((
            _axes(_layout(fig, "The viable frontier - closes AND builds"),
                  "total mission time (days)", "wet mass (kg)"),
            "Every point here survives both checks; down is lighter, left is "
            "faster. The lower-left edge is the set worth picking from - "
            "anything up-right of another point is dominated. Hover for "
            "payload and cost."))

    # 6 - what a kg of payload costs, among viable designs
    sub = (d[d["viable"]].dropna(subset=["payload_capacity_kg", "build_cost_musd"])
           if "payload_capacity_kg" in d.columns else d.iloc[0:0])
    if len(sub) >= 2:
        fig = _scatter_by_engine(sub, "payload_capacity_kg", "build_cost_musd")
        fig.add_vline(x=25.0, line=dict(color=MUTED, width=1, dash="dot"),
                      annotation_text="25 kg", annotation_font_color=MUTED)
        charts.append((
            _axes(_layout(fig, "Build cost vs payload + margin (viable designs)"),
                  "payload + margin (kg)", "build cost ($M)"),
            "The procurement view: right of the dotted line carries a 25 kg "
            "payload. Cost rises with array size and unit count, so the cheap "
            "high-payload corner (bottom-right) is where efficient single- and "
            "twin-engine stacks live."))

    return charts
