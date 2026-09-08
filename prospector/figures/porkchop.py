"""The porkchop: transfer cost over a date grid.

Two builders, for two different surfaces.

:func:`converged_grid` draws the solved grid, where every cell is a real trajectory over departure
date and flight time. That is the one to show. It colours by how much mass gets delivered, since
that is what the solve maximises and what a payload decision turns on, and it draws a cell the
search could not solve as "not found" and not as impossible. Only the solver converging supports
either claim, and a picture that showed the two the same way would reject reachable targets.

:func:`porkchop` draws the older two-burn surface over departure and arrival dates. It is kept for
the Lambert grid that still gives a cold solve somewhere to start, and it must not be used to rank,
budget or grey out a cell. Measured over three targets, its numbers move the wrong way against real
cost as flight time changes, and on one target it read over budget on 58 of the 64 cells the solver
went on to solve. See ``docs/physics.md``.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from prospector.figures.theme import (
    BG,
    DOME,
    GRID,
    MUTED,
    PLAY_GREEN,
    PORKCHOP_SCALE,
    REFERENCE,
    _layout,
    mjd2000_to_datetime,
)

# Cells that were found but cannot be flown by this vehicle: a grey gradient, light for just-over
# (or just-short) to dark for hopeless, so the structure of the region stays legible instead of
# flattening into one slab. "20 days later this closes" is the useful reading.
UNAFFORDABLE_SCALE = [[0.0, "#9aa2b6"], [0.5, "#5b6377"], [1.0, "#2b3142"]]


def converged_grid(dep_mjd2000, tof_days, *, dv_kms, final_mass_kg, feasible,
                   frontier=None, sel_dep=None, sel_tof=None, target_name="",
                   liftoff_offset_days: float = 0.0, colour_by: str = "mass",
                   dv_budget=None, burnout_mass_kg=None, past_deadline=None) -> go.Figure:
    """The converged low-thrust grid: propellant margin, delivered mass or ΔV over departure epoch x
    flight time.

    Every cell drawn is a trajectory the solver found, so there is no estimate here to disagree
    with and nothing is a bound. What it can claim is delta-v good to about 1% typically and 15% at
    worst against a converged reference. That is close enough to rank on and to show with a stated
    tolerance, and anything committed to gets solved again properly.

    ``feasible`` is the only thing that greys out a cell, and it means the search found no
    trajectory at this effort, not that the cell cannot be flown. Those are drawn flat, and say
    "not found" on hover. There is no verdict that a cell is impossible: the surface this replaced
    had one, and over three targets it fired on 16, 21 and 18 cells that fly.

    ``colour_by`` picks what is shown: ``"margin"``, the kilograms of propellant left in the tank
    after the cruise (the cell's final mass less ``burnout_mass_kg``, the dry mass plus the unusable
    propellant; the escape's spend is already gone from the cruise's starting mass), where more is
    better and a cell that comes up short is greyed; ``"mass"``, the kilograms delivered, where
    more is better; or ``"dv"`` in km/s, where less is better, with cells over ``dv_budget``
    greyed. Every view uses the same colours, hot for good, and the same grey for a trajectory
    this vehicle cannot fly, so flipping the field never changes what a colour means. Margin needs
    ``burnout_mass_kg``; without it, mass is shown.

    The y axis is the date the spacecraft leaves Earth, meaning when the cruise starts, which with
    a spiral escape is the liftoff date plus however long the spiral takes. Pass
    ``liftoff_offset_days`` and every hover carries the liftoff date too, since that is the date
    that gets scheduled.
    """
    dep = np.asarray(dep_mjd2000, float)
    tof = np.asarray(tof_days, float)
    dv = np.asarray(dv_kms, float)
    mass = np.asarray(final_mass_kg, float)
    ok = np.asarray(feasible, bool) & np.isfinite(dv)
    # Cells no trajectory was sought for, because that departure cannot reach the deadline at that
    # flight time. Left blank: they are not a failed search, there was nothing to search.
    blocked = (np.asarray(past_deadline, bool) if past_deadline is not None
               else np.zeros_like(ok, dtype=bool))

    by_margin = str(colour_by) == "margin" and burnout_mass_kg is not None
    by_mass = str(colour_by) == "mass" or (str(colour_by) == "margin" and not by_margin)
    margin = (mass - float(burnout_mass_kg)) if burnout_mass_kg is not None else None
    field = margin if by_margin else (mass if by_mass else dv)
    label = ("propellant margin (kg)" if by_margin
             else "delivered mass (kg)" if by_mass else "ΔV (km/s)")

    # Colour means AFFORDABLE. A cell whose cruise ΔV exceeds what the tank holds after escape is
    # not an option, so it recedes to grey while staying drawn and clickable, because this ΔV is
    # a converged number rather than an estimate and the operator may want to see how far over it is.
    #
    # This is a claim the surface it replaces could not make. There, greying keyed on an impulsive
    # number that moved the wrong way against real cost, so it hid 90% of the flyable window on one
    # target. Here the number is the arbiter's own, so comparing it to capability is meaningful.
    budget = float(dv_budget) if (dv_budget is not None and dv_budget > 0) else None
    # In the margin view the tank itself is the budget: a cell that lands with propellant to spare
    # is affordable, one that comes up short is not, and recedes to grey exactly as an over-budget
    # cell does in the ΔV view.
    if by_margin:
        afford = ok & (margin >= 0.0)
    else:
        afford = ok & ((dv <= budget) if budget is not None else True)

    # The ramp spans the AFFORDABLE cells, clipped near their top. Both matter. Spanning every
    # solved cell lets a handful at the edge stretch the scale over a range the bulk of the field
    # does not occupy: measured on Apophis, 88% of cells fell in the bottom 20% of that ramp, which
    # is why three quarters of the field read as one flat colour. Clipping to the affordable cells'
    # 90th percentile puts the contrast where the cells are. The same measurement gives 35%,
    # against 20% for a perfectly even spread. Cells above the clip saturate at the cold end, which
    # is honest: they are the ones about to price themselves out.
    shown = field[afford]
    if by_margin and shown.size:
        # Zero margin is the cold end, since that is where the tank runs dry; the clip sits at the
        # low end as for mass, because here too more is better.
        hi = float(shown.max())
        zmin = float(np.percentile(shown, 10.0)) if shown.size >= 5 else 0.0
        zmin = max(0.0, min(zmin, hi))
        zmax = max(hi, zmin + 1.0)
    elif shown.size:
        lo, hi = float(shown.min()), float(shown.max())
        clip = float(np.percentile(shown, 90.0)) if shown.size >= 5 else hi
        zmin, zmax = (lo, max(clip, lo * 1.01 + 1e-6)) if not by_mass else \
                     (min(clip, hi), hi)
        if by_mass:
            # More mass is better, so the clip belongs at the LOW end for this field.
            zmin = float(np.percentile(shown, 10.0)) if shown.size >= 5 else lo
            zmax = max(hi, zmin * 1.01 + 1e-6)
    else:
        zmin, zmax = 0.0, 1.0

    dep_dates = mjd2000_to_datetime(dep)
    fig = go.Figure(go.Heatmap(
        x=tof, y=dep_dates, z=np.where(afford, field, np.nan), zmin=zmin, zmax=zmax,
        colorscale=PORKCHOP_SCALE,
        # Hot means GOOD either way: more margin, more delivered mass, or less ΔV. Reversing only
        # for ΔV keeps the reading identical when the operator flips the field.
        reversescale=not (by_mass or by_margin),
        hoverongaps=False, hoverinfo="skip", name="field",
        colorbar=dict(title=label)))

    # Converged but unaffordable: the grey gradient (UNAFFORDABLE_SCALE), light for just-over to
    # dark for hopeless. In the ΔV view that is the ΔV past the budget, up to twice the budget; in
    # the margin view it is the kilograms the tank is short, up to the largest shortfall drawn.
    over = ok & ~afford
    if over.any() and by_margin:
        short = -margin
        worst = max(float(np.nanmax(short[over])), 1.0)
        fig.add_trace(go.Heatmap(
            x=tof, y=dep_dates, z=np.where(over, np.clip(short, 0.0, worst), np.nan),
            zmin=0.0, zmax=worst, colorscale=UNAFFORDABLE_SCALE,
            showscale=False, hoverongaps=False, hoverinfo="skip", name="short of propellant"))
    elif over.any() and budget is not None:
        grey = np.where(over, np.clip(dv, budget, 2.0 * budget), np.nan)
        fig.add_trace(go.Heatmap(
            x=tof, y=dep_dates, z=grey, zmin=budget, zmax=2.0 * budget,
            colorscale=UNAFFORDABLE_SCALE,
            showscale=False, hoverongaps=False, hoverinfo="skip", name="over budget"))

    # Cells the search did not close: one flat slab, not a gradient. There is nothing to rank here
    # since a non-converged cell has no cost, only an absence, and a gradient would imply one.
    missing = ~ok & ~blocked
    if missing.any():
        fig.add_trace(go.Heatmap(
            x=tof, y=dep_dates, z=np.where(missing, 1.0, np.nan), zmin=0.0, zmax=1.0,
            colorscale=[[0.0, "#1e2733"], [1.0, "#1e2733"]],
            showscale=False, hoverongaps=False, hoverinfo="skip", name="not found"))

    # Clickable overlay: one transparent marker per cell, carrying its exact coordinates. Plotly
    # hit-tests these by position rather than by rendered pixels, so they select and hover without
    # being drawn, since any tint accumulates into pale bands along a dense row.
    DEP, TOF = np.meshgrid(dep, tof, indexing="ij")
    flat_dep, flat_tof = DEP.ravel(), TOF.ravel()
    flat_dv, flat_mass, flat_ok = dv.ravel(), mass.ravel(), ok.ravel()
    flat_blocked = blocked.ravel()
    dep_txt = [d.strftime("%Y-%m-%d") for d in mjd2000_to_datetime(flat_dep)]
    arr_txt = [d.strftime("%Y-%m-%d") for d in mjd2000_to_datetime(flat_dep + flat_tof)]
    if liftoff_offset_days > 0:
        lift_txt = [f"<br>liftoff ~{d.strftime('%Y-%m-%d')}"
                    for d in mjd2000_to_datetime(flat_dep - liftoff_offset_days)]
    else:
        lift_txt = [""] * len(dep_txt)
    flat_afford = afford.ravel()
    flat_margin = (margin.ravel() if margin is not None else np.full(flat_dv.shape, np.nan))
    hover = []
    for d, lo, a, t, v, m, mg, good, aff, blk in zip(dep_txt, lift_txt, arr_txt, flat_tof,
                                                     flat_dv, flat_mass, flat_margin, flat_ok,
                                                     flat_afford, flat_blocked):
        head = f"leaves Earth {d}{lo}<br>{t:.0f} d flight, arrives {a}"
        margin_txt = f" · {mg:+.0f} kg margin" if np.isfinite(mg) else ""
        if blk:
            hover.append(f"{head}<br>arrives after the deadline")
        elif not good:
            hover.append(f"{head}<br>no trajectory found at this effort")
        elif aff:
            hover.append(f"{head}<br>{v:.2f} km/s · {m:.0f} kg delivered{margin_txt}")
        elif by_margin:
            # Named as a shortfall against the tank, not as a verdict on the transfer: the
            # trajectory is real and flyable, this vehicle just cannot carry the propellant.
            hover.append(f"{head}<br>{v:.2f} km/s · {-mg:.0f} kg short of propellant")
        else:
            # Likewise, against the ΔV the tank holds after escape.
            over = v - float(budget)
            hover.append(f"{head}<br>{v:.2f} km/s · {over:+.2f} km/s over the cruise budget"
                         f" ({float(budget):.2f})")
    fig.add_trace(go.Scattergl(
        x=flat_tof, y=mjd2000_to_datetime(flat_dep), mode="markers", name="cells",
        marker=dict(size=16, color="rgba(0,0,0,0)"),
        customdata=np.column_stack([flat_dep, flat_tof, flat_dv, flat_mass,
                                    flat_ok.astype(float)]),
        text=hover, hovertemplate="%{text}<extra></extra>"))

    # The frontier: the best departure at each flight time. This is the flight-time trade, drawn on
    # the same axes as the field it came from rather than as a separate study.
    pts = [p for p in (frontier or []) if p.get("dv_kms") is not None
           and p.get("dep_mjd2000") is not None]
    if pts:
        fx = [float(p["tof_days"]) for p in pts]
        fy = mjd2000_to_datetime(np.array([float(p["dep_mjd2000"]) for p in pts]))
        # Quoting the same pair the cells do, so hovering the curve and hovering the field under
        # it can be compared whichever way the field is coloured. A refined point says so: its
        # departure falls between two grid dates, so its numbers are nobody's cell.
        ftxt = []
        for p in pts:
            m = p.get("final_mass_kg")
            mass_txt = f" · {float(m):.0f} kg delivered" if m is not None else ""
            if m is not None and burnout_mass_kg is not None:
                mass_txt += f" · {float(m) - float(burnout_mass_kg):+.0f} kg margin"
            tail = " · refined, between grid dates" if p.get("source") == "polish" else ""
            ftxt.append(f"best at {p['tof_days']:.0f} d<br>{p['dv_kms']:.2f} km/s"
                        f"{mass_txt}{tail}")
        # The frontier and the star sit ON TOP of the transparent cell overlay, so a click there
        # hits them rather than the cells. They therefore carry the same customdata shape, and a
        # click on that curve resolves to that flight time's best cell, which is what someone
        # aiming at the line wants anyway. Without this the most interesting cells on the plot were
        # the only ones that could not be selected.
        fcd = np.column_stack([[float(p["dep_mjd2000"]) for p in pts], fx])
        fig.add_trace(go.Scatter(
            x=fx, y=fy, mode="lines+markers", name="best per flight time",
            line=dict(color=REFERENCE, width=1.5),
            marker=dict(symbol="diamond", size=7, color=REFERENCE,
                        line=dict(color=BG, width=1)),
            customdata=fcd, text=ftxt, hovertemplate="%{text}<extra></extra>"))
        # The star marks the best cell for the field on show: the most propellant left (which is
        # also the most delivered mass) when the surface is about kilograms, the least ΔV when it
        # is about ΔV. The two usually coincide; when they do not, the star should agree with the
        # colours around it.
        massed = [p for p in pts if p.get("final_mass_kg") is not None]
        if (by_margin or by_mass) and massed:
            best = max(massed, key=lambda p: float(p["final_mass_kg"]))
            what = (f"most margin: {float(best['final_mass_kg']) - float(burnout_mass_kg):+.0f} kg"
                    if by_margin else f"most delivered: {float(best['final_mass_kg']):.0f} kg")
        else:
            best = min(pts, key=lambda p: p["dv_kms"])
            what = f"cheapest {best['dv_kms']:.2f} km/s"
        fig.add_trace(go.Scatter(
            x=[float(best["tof_days"])],
            y=mjd2000_to_datetime(np.array([float(best["dep_mjd2000"])])),
            mode="markers", name="best",
            marker=dict(symbol="star", size=16, color=PLAY_GREEN, line=dict(color=BG, width=1)),
            customdata=np.array([[float(best["dep_mjd2000"]), float(best["tof_days"])]]),
            hovertemplate=f"{what} at {best['tof_days']:.0f} d"
                          + (" (refined)" if best.get("source") == "polish" else "")
                          + "<extra></extra>"))

    if sel_dep is not None and sel_tof is not None:
        fig.add_trace(go.Scatter(
            x=[float(sel_tof)], y=mjd2000_to_datetime(np.array([float(sel_dep)])),
            mode="markers", name="selected",
            marker=dict(symbol="x", size=16, color=DOME, line=dict(color=BG, width=2)),
            customdata=np.array([[float(sel_dep), float(sel_tof)]]),
            hovertemplate="selected cell<extra></extra>"))

    title = f"{target_name} - transfer options" if target_name else "Transfer options"
    _layout(fig, title)
    fig.update_layout(height=440, showlegend=False)
    y_title = ("Earth departure (cruise start)"
               + (f" = liftoff + {liftoff_offset_days:.0f} d" if liftoff_offset_days > 0 else ""))
    fig.update_xaxes(title="flight time (days)", gridcolor=GRID, color=MUTED,
                     tickfont=dict(size=10))
    fig.update_yaxes(title=y_title, gridcolor=GRID, color=MUTED, tickformat="%Y-%m-%d",
                     tickfont=dict(size=10))
    return fig
