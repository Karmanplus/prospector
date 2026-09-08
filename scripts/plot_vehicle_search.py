"""
Plot a vehicle-search session: how the margin varies, and the heaviest vehicle that still works.

Reads the CSVs a prospector.trades.design_search run wrote and renders two interactive HTML figures
into the same directory:

  margin_landscape.html   margin against dry mass, one line per engine and propellant setup.
                          Fully solved designs are markers, the quick analytic bound is a faint
                          dashed reference, and the zero line is where a design stops working.
  frontier.html           the heaviest working dry mass for each engine and propellant setup.

Usage:
    pixi run python scripts/plot_vehicle_search.py runs/vehicle_search/<stamp>/
    pixi run python scripts/plot_vehicle_search.py            # newest session
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "runs" / "vehicle_search"

COLORS = {1: "#9b7fe8", 2: "#4cd0e8", 3: "#4c9be8", 4: "#58c49a", 5: "#e8b54c",
          6: "#e8654c"}


def newest_session(root: Path) -> Path:
    sessions = sorted((d for d in root.iterdir()
                       if d.is_dir() and (d / "combos.csv").is_file()),
                      key=lambda d: d.stat().st_mtime)
    if not sessions:
        raise SystemExit(f"no search sessions with combos.csv under {root}")
    return sessions[-1]


def margin_landscape(out: Path, combos: pd.DataFrame,
                     analytic: pd.DataFrame | None) -> None:
    fig = go.Figure()
    combos = combos.assign(prop_offset=combos["prop_kg"] - combos["dry_kg"])
    offsets = sorted(combos["prop_offset"].unique())
    dashes = {off: d for off, d in zip(offsets, ("solid", "dot", "dash", "longdash"))}

    if analytic is not None:
        for (n_eng, off), g in analytic.groupby(["n_engines", "prop_offset"]):
            g = g.sort_values("dry_kg")
            fig.add_trace(go.Scatter(
                x=g["dry_kg"], y=g["bound_margin_kms"], mode="lines",
                line=dict(color=COLORS.get(int(n_eng), "#999"), width=1,
                          dash=dashes.get(off, "dot")),
                opacity=0.35, showlegend=False, hoverinfo="skip"))

    for (n_eng, off), g in combos.groupby(["n_engines", "prop_offset"]):
        g = g.sort_values("dry_kg")
        fig.add_trace(go.Scatter(
            x=g["dry_kg"], y=g["margin_kms"], mode="lines+markers",
            name=f"{int(n_eng)} eng, prop = dry+{off:.0f}",
            line=dict(color=COLORS.get(int(n_eng), "#999"),
                      dash=dashes.get(off, "solid")),
            marker=dict(size=9, symbol=["circle" if s else "x" for s in g["success"]]),
            customdata=g[["prop_kg", "dv_short_kms", "why"]],
            hovertemplate=("dry %{x:.0f} kg · prop %{customdata[0]:.0f} kg<br>"
                           "margin %{y:+.2f} km/s · %{customdata[2]}<extra></extra>")))

    fig.add_hline(y=0.0, line_color="#888", line_dash="dash",
                  annotation_text="closes")
    fig.update_layout(
        title="Vehicle search - dV margin vs dry mass "
              "(markers = SF-evaluated; faint lines = analytic upper bound on margin; "
              "x = did not close)",
        xaxis_title="dry mass (kg)", yaxis_title="dV margin (km/s)",
        template="plotly_dark", legend=dict(orientation="h", y=-0.18))
    fig.write_html(out / "margin_landscape.html", include_plotlyjs="cdn")


def frontier(out: Path, combos: pd.DataFrame) -> None:
    combos = combos.assign(prop_offset=combos["prop_kg"] - combos["dry_kg"])
    ok = combos[combos["success"].astype(bool)]
    rows = (ok.groupby(["n_engines", "prop_offset"])["dry_kg"].max().reset_index()
            if not ok.empty else pd.DataFrame(columns=["n_engines", "prop_offset",
                                                       "dry_kg"]))
    fig = go.Figure()
    for off, g in rows.groupby("prop_offset"):
        fig.add_trace(go.Bar(
            x=g["n_engines"].astype(int).astype(str), y=g["dry_kg"],
            name=f"prop = dry + {off:.0f} kg",
            text=[f"{v:.0f}" for v in g["dry_kg"]], textposition="outside"))
    for y, label in ((200, "200 kg minimum"), (250, "250 kg goal")):
        fig.add_hline(y=y, line_color="#888", line_dash="dot", annotation_text=label)
    fig.update_layout(
        title="Heaviest closing dry mass per engine count",
        xaxis_title="engine count", yaxis_title="max closing dry mass (kg)",
        barmode="group", template="plotly_dark")
    fig.write_html(out / "frontier.html", include_plotlyjs="cdn")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    session = Path(argv[0]) if argv else newest_session(DEFAULT_ROOT)
    combos = pd.read_csv(session / "combos.csv")
    analytic_path = session / "analytic_map.csv"
    analytic = pd.read_csv(analytic_path) if analytic_path.is_file() else None
    margin_landscape(session, combos, analytic)
    frontier(session, combos)
    print(f"wrote {session}/margin_landscape.html and {session}/frontier.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
