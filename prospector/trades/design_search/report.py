"""The session artifacts: the CSVs, the structured record, and the human summary.

Everything a finished session leaves behind. REPORT.md is written to be read on its own, so it
states the assumptions and the caveats next to the winner, in particular that a solve failing to
converge is evidence rather than proof, and that the quick analytic pass can only ever rule a
design out.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from prospector.trades.design_search.evaluate import spiral_curve
from prospector.trades.design_search.session import SPIRAL_CACHE_DIR

log = logging.getLogger(__name__)

if TYPE_CHECKING:                  # annotation only; search imports this module
    from prospector.trades.design_search.search import Search


def _fmt(value, spec=".2f", missing="-"):
    return missing if value is None else format(value, spec)


def write_outputs(search: Search, lines: list[dict], analytic: list[dict],
                  anchor: dict | None) -> None:
    import pandas as pd

    out = search.out
    if analytic:
        pd.DataFrame(analytic).to_csv(out / "analytic_map.csv", index=False)

    combo_rows, point_rows = [], []
    for v in search.evaluations.values():
        row = {k: v.get(k) for k in (
            "dry_kg", "prop_kg", "wet_kg", "prop_dry_ratio", "n_engines", "engine",
            "propellant", "propellant_key", "propellant_estimated", "anchor",
            "thrust_mN_per_kg", "power_W_per_kg", "thrust_mN_per_W",
            "buildable", "build_why", "lifetime_ok", "lifetime_why",
            "prop_per_thruster_kg", "throughput_margin_kg", "mission_ignitions",
            "payload_capacity_kg", "min_dry_kg", "eol_factor", "belt_days",
            "bol_power_W", "array_kg", "pcdu_kg", "thruster_sys_kg", "tank_kg",
            "fixed_bus_kg", "build_cost_musd", "success", "why", "margin_kms",
            "dv_short_kms", "prop_margin_kg", "capability_kms", "escape_dv_kms",
            "cruise_dv_kms", "total_dv_kms",
            "spiral_tof_days", "cruise_tof_days", "total_tof_days",
            "power_limited", "power_required_W", "power_available_end_W",
            "n_converged", "elapsed_s")}
        # Carry the swept model-setting columns through to combos.csv so they're plottable.
        row.update({k: val for k, val in v.items() if k.startswith("param_")})
        combo_rows.append(row)
        for p in v.get("points", []):
            point_rows.append({
                "dry_kg": v["dry_kg"], "prop_kg": v["prop_kg"],
                "n_engines": v["n_engines"],
                **{k: p.get(k) for k in (
                    "vinf_kms", "converged", "success", "mismatch", "escape_dv_kms",
                    "cruise_dv_kms", "total_dv_kms", "margin_kms", "prop_needed_kg",
                    "prop_margin_kg", "tof_days", "dep_date", "arr_date", "error")},
            })
    combos = pd.DataFrame(combo_rows).sort_values(
        ["engine", "propellant_key", "n_engines", "prop_kg", "dry_kg"]
    ) if combo_rows else pd.DataFrame()
    combos.to_csv(out / "combos.csv", index=False)
    pd.DataFrame(point_rows).to_csv(out / "evaluations.csv", index=False)

    # The reference vehicle (anchor) is reported on its own line, never as the winner or among the
    # near misses, which list the requested combinations only.
    successes = [v for v in search.evaluations.values()
                 if v.get("success") and not v.get("anchor")]
    winner = max(successes, key=lambda v: (v["dry_kg"], v.get("prop_margin_kg") or 0.0),
                 default=None)
    near = sorted((v for v in search.evaluations.values()
                   if not v.get("success") and not v.get("anchor")
                   and v.get("dv_short_kms") is not None),
                  key=lambda v: v["dv_short_kms"])[:8]

    result = {
        "target": search.target.get("name"),
        "mission": search.args.mission,
        "args": vars(search.args),
        "winner": winner,
        "lines": lines,
        "anchor": anchor,
        "evaluations": list(search.evaluations.values()),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, default=_json_default))

    md = [f"# Vehicle design-space search - {datetime.now():%Y-%m-%d %H:%M}", ""]
    md += [f"Target **{search.target.get('name')}** on mission "
           f"`{search.args.mission}` (launch {search.mission.launch_window[0]} … "
           f"{search.mission.launch_window[1]}, arrive by {search.mission.arrive_by}, "
           f"{search.mission.launch_orbit} drop-off). Engine `{search.args.engine}`, "
           f"duty/thrust limit {search.args.duty:.0%}, wet-mass cap "
           f"{search.args.max_wet:.0f} kg.", ""]
    if anchor is not None:
        md += [f"Reference vehicle (150/300 kg, 3 engines - the fixed cross-session "
               f"comparison point): "
               f"{'closes' if anchor['success'] else 'does not close'} "
               f"({_fmt(anchor.get('prop_margin_kg'), '+.0f')} kg propellant margin; "
               f"{_fmt(anchor.get('margin_kms'), '+.2f')} km/s at rated Isp).", ""]
    md += ["## Winner", ""]
    if winner:
        bp = winner.get("best_point") or {}
        md += [
            f"**Dry {winner['dry_kg']:.0f} kg + prop {winner['prop_kg']:.0f} kg "
            f"({winner['wet_kg']:.0f} kg wet, prop/dry "
            f"{winner['prop_kg'] / winner['dry_kg']:.2f}), {winner['n_engines']}x "
            f"{winner.get('engine', '?')} on {winner.get('propellant', 'Xenon')}"
            f"{' (estimated)' if winner.get('propellant_estimated') else ''}** - "
            f"{_fmt(winner['prop_margin_kg'], '+.0f')} kg propellant margin "
            f"({_fmt(winner['margin_kms'], '+.2f')} km/s at rated Isp).",
            "",
            f"Best split: departure v-inf {_fmt(bp.get('vinf_kms'))} km/s - escape "
            f"{_fmt(bp.get('escape_dv_kms'))} + cruise {_fmt(bp.get('cruise_dv_kms'))} "
            f"= {_fmt(bp.get('total_dv_kms'))} km/s of "
            f"{_fmt(winner.get('capability_kms'))} km/s capability. Spiral "
            f"{_fmt(winner.get('spiral_tof_days'), '.0f')} d; cruise departs "
            f"{bp.get('dep_date', '-')}, arrives {bp.get('arr_date', '-')} "
            f"({_fmt(bp.get('tof_days'), '.0f')} d).", ""]
        if winner.get("buildable") is not None:
            md += [
                f"Buildability: {'fits' if winner['buildable'] else 'DOES NOT FIT'} "
                f"the dry budget - payload + margin "
                f"{_fmt(winner.get('payload_capacity_kg'), '.0f')} kg after arrays "
                f"({_fmt(winner.get('array_kg'), '.1f')} kg for "
                f"{_fmt(winner.get('bol_power_W'), '.0f')} W BOL), tank, thrusters, "
                f"and a standard bus. Rough build cost "
                f"~${_fmt(winner.get('build_cost_musd'), '.1f')}M (hardware + wraps, "
                f"launch excluded).", ""]
    else:
        md += ["No evaluated combo closed. See the near-miss table.", ""]

    md += ["## Frontier per line", "",
           "Every dry mass at or below a line's frontier ALSO closes (margin only grows "
           "as dry shrinks at fixed propellant) - the highlighted row is each line's "
           "heaviest, i.e. its **minimum prop/dry** closing point.", "",
           "| engine | gas | count | propellant (kg) | max closing dry (kg) | "
           "min prop/dry | margin (km/s) | limit |",
           "|---|---|---|---|---|---|---|---|"]
    for ln in lines:
        f = ln.get("frontier") or {}
        d = ln.get("frontier_dry_kg")
        prop = ln["prop_kg"]
        ratio = None if d is None else prop / d
        gas = ln.get("gas", "xenon")
        gas += "*" if (f.get("propellant_estimated")) else ""
        md += [f"| {ln.get('engine', '?')} | {gas} | {ln['n_engines']} | "
               f"{prop:.0f} | "
               f"{_fmt(d, '.0f')} | {_fmt(ratio)} | "
               f"{_fmt(f.get('margin_kms'), '+.2f')} | {ln.get('why', '')} |"]

    md += ["", "## Near misses (closest first)", "",
           "| dry | prop | prop/dry | engines | gas | short by (km/s) | "
           "short by (kg prop) | why |",
           "|---|---|---|---|---|---|---|---|"]
    for v in near:
        gas = v.get("propellant_key", "xenon") + ("*" if v.get("propellant_estimated") else "")
        md += [f"| {v['dry_kg']:.0f} | {v['prop_kg']:.0f} | "
               f"{v['prop_kg'] / v['dry_kg']:.2f} | {v['n_engines']}x {v.get('engine', '?')} | "
               f"{gas} | {_fmt(v['dv_short_kms'])} | "
               f"{_fmt(None if v['prop_margin_kg'] is None else -v['prop_margin_kg'], '.0f')} | "
               f"{v['why']} |"]

    md += ["", "## Assumptions & caveats", "",
           f"- Duty cycle {search.args.duty:.0%} applied as the spiral duty cycle AND "
           f"the cruise thrust limit.",
           "- Engine system mass (thruster + PPU + feed, per the engine library) is "
           "not added on top of dry mass for the trajectory - budget it inside dry; "
           "the buildability check charges it explicitly.",
           "- Success = SF matchpoint mismatch <= 1e-3 and escape+cruise propellant "
           "(rocket equation at the solved total dV) fits the usable load. The SF "
           "solve itself is tank-unconstrained, so shortfalls are measured, not "
           "guessed.",
           "- A failed SF convergence is evidence, not proof (MBH is stochastic); a blanket "
           "failure is retried once with a new seed.",
           "- Target elements are re-osculated at the arrival era via Horizons, so the "
           "two-body propagation is locally accurate there. An arrival on the far side "
           "of a planetary close approach still carries growing error - MONTE (or a real "
           "ephemeris) must arbitrate any finalist.",
           "- The escape estimate and the element-based cruise floor in analytic_map.csv are both "
           "optimistic bounds: that map can prove infeasibility, never feasibility.",
           f"- Propellants evaluated: {', '.join(search.gases)}. A gas marked `*` (and "
           "flagged `propellant_estimated` in combos.csv) is estimated - the engine has "
           "no authored numbers for it, so its thrust/Isp were scaled from the xenon "
           "values by the propellant library's baseline factors. Treat those as "
           "first-order; a thruster with measured multi-gas data should be added as its "
           "own engine entry.",
           "- Buildability columns use the prospector/buildability.py sizing model: "
           "arrays sized for the thruster's power at "
           "EOL, PCDU, per-gas propellant tank, standard deep-space bus, structure/thermal/"
           "harness ratios, 10% system margin. Negative payload capacity = the bus "
           "alone busts the dry budget. Costs are comparison-grade only (array $/W "
           "and propellant $/kg from the sizing model; thruster unit and bus $/kg are "
           "documented placeholders; launch not included)."]
    (out / "REPORT.md").write_text("\n".join(md) + "\n")
    log.info(f"\nwrote {out}/REPORT.md (+ combos.csv, evaluations.csv, "
          f"analytic_map.csv, result.json)")


def _json_default(value):
    """The ``json.dumps(default=...)`` hook: render one value json refused to serialize.

    Not the same job as :func:`prospector.jobs.jsonable_row`, which walks a whole mapping. This is
    called once per value json could not handle and must always return something, so anything
    unrecognised falls through to ``str``.
    """
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return str(value)


def persist_winner(search: Search, winner: dict) -> str | None:
    """Submit the winning combo as a normal sweep job so it lands in the app.

    The sweep worker solves every point again in fresh processes, records the winner as a full
    ``runs/<id>/`` solve from its solution vector, and the UI's run history picks both up. Those
    are the same files an interactive session would leave.
    """
    from prospector import jobs

    gas = winner.get("propellant_key", "xenon")
    rc = search.rc(winner["dry_kg"], winner["prop_kg"], winner["n_engines"],
                   winner.get("engine"), gas)
    curve = spiral_curve(rc, duty=search.args.duty,
                         vinf_max=max(search.vinf_values) + 0.3,
                         cache_dir=SPIRAL_CACHE_DIR)
    options = {**search.sf_options, "vinf_values": search.vinf_values, "curve": curve}
    run_id = jobs.submit_sweep(rc.model_dump(mode="json"), search.target,
                               options=options,
                               label=f"vehicle search winner "
                                     f"{winner['dry_kg']:.0f}/{winner['prop_kg']:.0f} "
                                     f"x{winner['n_engines']} {gas}")
    log.info(f"winner re-submitted through the sweep channel: runs/sweep/{run_id}")
    return run_id
