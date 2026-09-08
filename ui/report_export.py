"""Assemble and download the standalone feasibility PDF (the top bar's Export PDF button).

A thin shell over :mod:`prospector.reporting`. It resolves the open study, works out which escape
and cruise results count this session, and hands them to :mod:`prospector.figures.products`, the
same layer the Plan trajectory workspace uses, so the document and the app cannot end up showing
different figures or different numbers. The wording, layout and rendering are all in
:mod:`prospector.reporting`; this module only decides which sections have anything to say.

The document is meant to read as an analysis of a vehicle and a trajectory, so its title and
content name the methods rather than the software. Both the escape and the cruise are checked
against the live config: the escape has to come from a spiral run that still matches
(:func:`ui.state.session_spiral_run`), and the cruise from
:func:`~prospector.figures.products.cruise_block`, which refuses a run the config has moved past.
An out-of-date trajectory must never reach paper, where nothing marks it as such. The image export
runs off the event loop so the app never freezes.
"""
from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

from nicegui import run, ui

from prospector import reporting as report
from prospector.figures import products
from prospector.spacecraft import buildability
from prospector.spacecraft.propellants import load_propellants
from ui import state
from ui.state import S


async def export_pdf() -> None:
    """Build the feasibility document for the open study and download it."""
    if S.project is None:
        ui.notify("Open a project first.", type="warning")
        return
    note = ui.notification("Generating report…", spinner=True, timeout=None)
    try:
        path = await run.io_bound(_build)        # kaleido export is slow; keep the UI responsive
    except Exception as exc:  # noqa: BLE001  (report a build failure; never crash the app)
        note.dismiss()
        ui.notify(f"Report failed: {str(exc).splitlines()[0]}", type="negative")
        return
    note.dismiss()
    ui.download(str(path))
    ui.notify("Report ready.", type="positive")


def _build() -> Path:
    """Resolve the study, gather results, build the figures, assemble the sections, render."""
    rc = state.resolved()
    spiral_id = None if rc.launch.escape_provided else state.session_spiral_run(rc)
    spiral = products.spiral_block(spiral_id)
    avail_power = products.spiral_power_available_W(spiral_id)
    res = products.cruise_block(S.solve_run_id, rc, avail_power)
    sf = res["sf"] if res else None
    target_name = _target_name(res)
    build = _assess_build(rc, spiral)

    # The cruise propellant integral feeds both the mission-propellant chart and the cruise
    # section's thrust-history cross-check.
    cruise_days, cruise_prop_cum, verification = products.cruise_propellant_profile(res)

    # Static figures: the same ones the workspace shows, with the scrub chrome off.
    escape_figs = products.escape_figures(spiral, animate=False)
    closure_figs = _present(products.propellant_timeline_figure(
        rc, spiral, cruise_days, cruise_prop_cum))
    # The dry-mass breakdown leads the buildability section; the panel-power chart sits with the
    # array-sizing text there (the radiation loss the sized panels fly through).
    build_figs = _present(
        products.mass_allocation_figure(
            buildability.mass_allocation_components(
                build, dry_mass_kg=float(rc.vehicle.dry_mass),
                n_engines=rc.vehicle.engine_count) if build else None,
            dry_mass_kg=float(rc.vehicle.dry_mass)),
        products.array_power_figure(spiral))
    cruise_figs = products.cruise_figures(
        res, max_throttle_pct=products.cruise_thrust_ceiling_pct(rc, sf),
        animate=False, controls=False)
    # The flight-time trade is the curve across the transfer grid, so the grid is what carries it
    # here. Gated through the same products call the workspace uses, so a surface the app has
    # already retired as stale cannot reach paper, where nothing marks a figure out of date.
    grid = products.grid_block(S.grid_run_id, rc, avail_power,
                               target_pdes=str((S.focus or {}).get("pdes") or ""))
    sel = S.grid_sel or (None, None)
    cruise_figs = cruise_figs + _present(
        products.grid_figure(grid, rc, sel_dep=sel[0], sel_tof=sel[1]))

    sections = [
        report.summary_section(rc, spiral=spiral, sf=sf, build=build, target_name=target_name),
        report.spacecraft_section(rc),
        report.escape_section(rc, spiral, figures=escape_figs),
        report.cruise_section(rc, sf, figures=cruise_figs, target_name=target_name,
                              verification=verification),
        report.closure_section(rc, spiral=spiral, sf=sf, figures=closure_figs),
        report.buildability_section(rc, build, figures=build_figs),
        # None when no cruise converged, so the table states a segment count only when a solve
        # produced one.
        report.assumptions_section(nseg=(sf or {}).get("nseg"),
                                   departure_vinf_kms=rc.departure_vinf_kms),
        report.parameters_section(rc),
    ]
    out = Path(tempfile.gettempdir()) / f"feasibility_{_slug(target_name)}.pdf"
    return report.build_report(
        out,
        title=f"Vehicle and trajectory feasibility - {target_name}",
        subtitle=f"{rc.vehicle.name} · solar-electric propulsion",
        sections=sections, generated=datetime.now())


def _present(*figures) -> list:
    """Drop the figures that could not be built, keeping the section order intact."""
    return [f for f in figures if f]


def _target_name(res: dict | None) -> str:
    if res and res.get("target", {}).get("name"):
        return str(res["target"]["name"]).strip()
    if S.focus:
        return str(S.focus.get("full_name") or S.focus.get("pdes") or "target").strip()
    return "target"


def _assess_build(rc, spiral: dict | None) -> dict | None:
    """The bus-level mass + lifetime assessment for the vehicle's assembly, using the spiral's
    belt residence and revolution count when a matching run exists."""
    v = rc.vehicle
    if not v.mounts:
        return None
    try:
        # Use the resolved engines (rc.engines), not the raw library: when the vehicle loads a
        # non-native gas, the resolved engines carry that gas, so the tank is sized and the
        # propellant labeled for what the spacecraft flies on (krypton, not xenon).
        return buildability.assess_assembly(
            dry_kg=float(v.dry_mass), prop_kg=float(v.fuel_mass), mounts=v.mounts,
            catalog=rc.engines, propellants=load_propellants(),
            belt_days=(spiral or {}).get("belt_days"),
            eol_power_fraction=(spiral or {}).get("power_fraction_end"),
            ignitions=(spiral or {}).get("revolutions"),
            margin_pct=v.array_margin_pct)      # size at the vehicle's own array margin
    except Exception:  # noqa: BLE001  (a sizing gap drops one block, never the report)
        return None


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(name)).strip("_") or "study"
