"""
Standalone solve worker: run one solver job in a detached process.

Invoked by ``prospector.jobs`` as one of:
    python worker.py <run_id>          -- the Lambert -> Sims-Flanagan design (default)
    python worker.py <run_id> spiral   -- the launch-phase escape spiral (runs/spiral/<id>/)
    python worker.py <run_id> sweep    -- the departure-v-infinity sweep (runs/sweep/<id>/)
    python worker.py <run_id> enrich   -- the physical-value characterization job

The default mode reads the job spec from ``runs/<id>/job.json``, resolves the study into a config,
runs the pipeline for the chosen target, and writes ``status.json``, ``summary.json`` and
``result.pkl``. All the real work is in the library; this file is only the process the UI polls.

Run directly to debug a job:  python worker.py <run_id> [spiral|sweep|enrich]
"""
import os
import signal
import sys
import traceback
from pathlib import Path

# Running `python worker.py` puts the repo root on sys.path[0], so `prospector` imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prospector import jobs  # noqa: E402
from prospector.config import ResolvedConfig  # noqa: E402
from prospector.figures import products  # noqa: E402
from prospector.trades import pipeline  # noqa: E402


def _run_sf(run_id: str) -> int:
    spec = jobs.read_job(run_id)
    try:
        rc = ResolvedConfig.model_validate(spec["config"])
        target_row = spec["target"]

        def on_progress(stage: str, fraction: float, message: str) -> None:
            jobs.write_status(run_id, state=jobs.RUNNING, stage=stage,
                              progress=fraction, message=message)

        jobs.write_status(run_id, state=jobs.RUNNING, stage="start", progress=0.0,
                          message=f"solving {target_row.get('full_name', '')}")
        result = pipeline.solve_from_cell(
            rc, target_row, spec["dep_mjd2000"], spec["arr_mjd2000"], on_progress=on_progress,
            **spec.get("options", {}))
        jobs.write_result(run_id, result)
        jobs.write_summary(run_id, products.run_summary(result))
        jobs.write_status(run_id, state=jobs.DONE, stage="done", progress=1.0,
                          message="complete")
        return 0
    except Exception:
        jobs.write_status(run_id, state=jobs.ERROR, stage="error", progress=0.0,
                          message="solve failed", error=traceback.format_exc())
        return 1


def _spiral_summary(sol, duty_cycle=None) -> dict:
    """A compact, JSON-safe digest of a finished spiral run, including the thinned
    dv-vs-v-infinity tradeoff curve, so budget refinement and the Solve tab's escape
    lookup never need the heavy pickle."""
    curve_idx = _decimate_indices(len(sol.curve_vinf_kms), 200)
    return {
        "status": sol.status,
        # The velocity change flown, and the propellant burned as rated-Isp delta-v (the budget's
        # currency); see SpiralSolution.
        "dv_kms": float(sol.dv_kms),
        "dv_at_escape_kms": None if sol.dv_at_escape_kms is None else float(sol.dv_at_escape_kms),
        "dv_equiv_kms": None if sol.dv_equiv_kms is None else float(sol.dv_equiv_kms),
        "dv_at_escape_equiv_kms": (None if sol.dv_at_escape_equiv_kms is None
                                   else float(sol.dv_at_escape_equiv_kms)),
        "tof_days": float(sol.tof_days),
        "tof_at_escape_days": (None if sol.tof_at_escape_days is None
                               else float(sol.tof_at_escape_days)),
        "vinf_kms": float(sol.vinf_kms),
        "target_vinf_kms": float(sol.target_vinf_kms),
        "propellant_kg": float(sol.propellant_kg),
        "final_mass_kg": float(sol.final_mass_kg),
        "eclipse_days": float(sol.eclipse_days),
        "belt_days": float(sol.belt_days),
        "revolutions": float(sol.revolutions),
        "power_fraction_end": float(sol.power_fraction_end),
        # Power adequacy: did the degrading array stay above the thruster's full-power demand?
        "power_limited": bool(sol.power_limited),
        "power_required_W": float(sol.power_required_W),
        "power_available_end_W": float(sol.power_available_end_W),
        # The thruster duty cycle the spiral was flown at, so the report quotes what ran.
        "duty_cycle": None if duty_cycle is None else float(duty_cycle),
        "inc_deg_end": float(sol.inc_deg_end),
        "curve_vinf_kms": sol.curve_vinf_kms[curve_idx].tolist(),
        "curve_dv_kms": sol.curve_dv_kms[curve_idx].tolist(),
        "curve_tof_days": sol.curve_tof_days[curve_idx].tolist(),
        # The orientation flown (launch targeting) and the achieved asymptote, so the solve tab can
        # check the hand-off without the heavy pickle.
        "raan_deg": float(getattr(sol, "raan_deg", 0.0)),
        "argp_deg": float(getattr(sol, "argp_deg", 0.0)),
        "exit_lat_deg": (None if getattr(sol, "exit_lat_deg", None) is None
                         else float(sol.exit_lat_deg)),
        "exit_lon_deg": (None if getattr(sol, "exit_lon_deg", None) is None
                         else float(sol.exit_lon_deg)),
    }


def _decimate_indices(n: int, keep: int):
    """Indices that keep at most ``keep`` evenly spaced samples (always the last one)."""
    import numpy as np
    if n == 0:
        return np.empty(0, int)
    return np.unique(np.linspace(0, n - 1, min(keep, n)).astype(int))


def _run_spiral(run_id: str) -> int:
    """Propagate the launch-phase escape spiral for a config snapshot (runs/spiral/<id>/)."""
    # Imported here so the other modes needn't load scipy.
    from dataclasses import asdict

    from prospector.solvers import spiral

    try:
        spec = jobs.read_spiral_job(run_id)
        rc = ResolvedConfig.model_validate(spec["config"])

        def on_progress(fraction: float, message: str) -> None:
            jobs.write_spiral_status(run_id, state=jobs.RUNNING, stage="spiral",
                                     progress=fraction, message=message)

        jobs.write_spiral_status(run_id, state=jobs.RUNNING, stage="start", progress=0.0,
                                 message=f"spiraling out of {rc.launch.name}")
        # The array powers the thrusters through the conversion chain the engines' PPU wiring
        # implies, so the escape flies on that fraction of the degraded array output, the same
        # chain the arrays were sized against. Defaulted here (not in the study config) so an
        # unchanged spiral spec still reflects the current build model; a caller may override it in
        # the options.
        options = dict(spec.get("options", {}))
        options.setdefault("array_to_thruster_eff", rc.thruster_chain_eff())
        sol = spiral.solve_for_config(rc, progress=on_progress, **options)
        jobs.write_spiral_result(run_id, {"spiral": asdict(sol)})
        jobs.write_spiral_summary(run_id, _spiral_summary(sol, duty_cycle=options.get("duty_cycle")))
        jobs.write_spiral_status(run_id, state=jobs.DONE, stage="done", progress=1.0,
                                 message=f"complete - {sol.status}")
        return 0
    except Exception:
        jobs.write_spiral_status(run_id, state=jobs.ERROR, stage="error", progress=0.0,
                                 message="spiral failed", error=traceback.format_exc())
        return 1


def _sweep_summary(spec: dict, points: list[dict]) -> dict:
    """A compact, JSON-safe digest of a finished departure-speed sweep, including the full
    points table (small scalars only), so the tradeoff plot never needs the pickle."""
    target = spec.get("target") or {}
    feasible = [p for p in points
                if p.get("feasible") and p.get("total_dv_kms") is not None]
    best = min(feasible, key=lambda p: p["total_dv_kms"]) if feasible else None
    return {
        "target": target.get("full_name") or target.get("pdes") or "target",
        "n_points": len(points),
        "n_feasible": len(feasible),
        "best_vinf_kms": None if best is None else float(best["vinf_kms"]),
        "best_total_dv_kms": None if best is None else float(best["total_dv_kms"]),
        "points": points,
    }


def _record_best_sweep_run(spec: dict, sf_options: dict, points: list):
    """Persist the sweep winner as a normal ``runs/<id>/`` solve; return its id or None.

    Every sweep point already ran a full Sims-Flanagan solve; the winner's stored decision vector
    rebuilds the trajectory deterministically (milliseconds, no re-optimization), so its complete
    result lands, with the cards, trajectory and diagnostics, in the same channel as an interactive
    solve and the UI can simply open it.
    """
    from prospector.trades import sweep

    candidates = [p for p in points
                  if p.get("feasible") and p.get("decision_vector")
                  and p.get("total_dv_kms") is not None]
    if not candidates:
        return None
    best = min(candidates, key=lambda p: p["total_dv_kms"])

    # The winner's physics, as its sweep point ran them.
    rc = ResolvedConfig.model_validate(spec["config"])
    rc_point = sweep.with_escape_terms(
        rc.model_copy(update={"departure_vinf_kms": float(best["vinf_kms"])}),
        dv_kms=float(best["escape_dv_kms"]), tof_days=float(best["escape_tof_days"]),
        propellant_kg=float(best["escape_propellant_kg"]))
    sf_kwargs = dict(sf_options)
    sf_kwargs["vinf_dep_kms"] = max(float(best["vinf_kms"]), sweep.VINF_CAP_FLOOR_KMS)
    # The point's converged per-segment thrust ceilings (sun-distance array power) travel with its
    # decision vector so the rebuilt problem carries the exact capped constraints.
    if best.get("seg_caps"):
        sf_kwargs["seg_caps"] = best["seg_caps"]
    for key in ("thrust_N", "isp_s", "seg_isp_s"):
        if best.get(f"cruise_{key}") is not None:
            sf_kwargs[key] = best[f"cruise_{key}"]
    # If this point flew a return leg, hand its stored decision vector through so the rebuilt run
    # carries the same return trajectory (no re-optimization).
    if best.get("return_decision_vector"):
        sf_kwargs["return_decision_vector"] = best["return_decision_vector"]
        for key in ("seg_caps", "thrust_N", "isp_s", "seg_isp_s"):
            if best.get(f"return_{key}") is not None:
                sf_kwargs[f"return_{key}"] = best[f"return_{key}"]
    result = pipeline.result_from_decision(rc_point, spec["target"],
                                           best["decision_vector"], **sf_kwargs)

    solve_id = jobs.record_run(rc_point.model_dump(mode="json"), spec["target"],
                               float(best["dep_mjd2000"]), float(best["arr_mjd2000"]),
                               label=f"{spec.get('label') or 'sweep'} · sweep best",
                               options=sf_kwargs)
    jobs.write_result(solve_id, result)
    jobs.write_summary(solve_id, products.run_summary(result))
    jobs.write_status(solve_id, state=jobs.DONE, stage="done", progress=1.0,
                      message=f"sweep best (v∞ {best['vinf_kms']:.2f} km/s)")
    return solve_id


def _run_sweep(run_id: str) -> int:
    """Sweep the cruise solve over departure-v-infinity caps for one target (runs/sweep/<id>/)."""
    # Imported here so the other modes needn't load the sweep (and it the solver stack).
    from prospector.trades import sweep

    try:
        spec = jobs.read_sweep_job(run_id)
        options = dict(spec.get("options") or {})
        curve = options.pop("curve", None)
        vinf_values = options.pop("vinf_values", None)
        cap = options.pop("vinf_cap_kms", None)
        if vinf_values is None:
            vinf_values = sweep.default_vinf_grid(float(cap) if cap is not None else 2.0)
        target = spec["target"]
        # The points are independent solves, so fan them out across processes by default; leave one
        # core for the UI/other work. "workers": 1 forces serial.
        workers = int(options.pop("workers", 0) or 0)
        if workers <= 0:
            workers = min(len(vinf_values), max(1, (os.cpu_count() or 2) - 1))

        def on_progress(i: int, n: int, point: dict) -> None:
            jobs.write_sweep_status(
                run_id, state=jobs.RUNNING, stage="sweep", progress=i / n,
                message=f"point {i}/{n} - v∞ {point['vinf_kms']:.2f} km/s", point=point)

        jobs.write_sweep_status(run_id, state=jobs.RUNNING, stage="start", progress=0.0,
                                message=f"sweeping {len(vinf_values)} departure v∞ caps "
                                        f"for {target.get('full_name', '')} "
                                        f"({workers} in parallel)")
        points = sweep.run_sweep(spec["config"], target, vinf_values=vinf_values,
                                 curve=curve, options=options, progress=on_progress,
                                 workers=workers)
        jobs.write_sweep_result(run_id, {"points": points})
        summary = _sweep_summary(spec, points)
        try:
            summary["best_run_id"] = _record_best_sweep_run(spec, options, points)
        except Exception:
            summary["best_run_id"] = None   # the sweep stands on its own either way
        jobs.write_sweep_summary(run_id, summary)
        jobs.write_sweep_status(run_id, state=jobs.DONE, stage="done", progress=1.0,
                                message="sweep complete")
        return 0
    except Exception:
        jobs.write_sweep_status(run_id, state=jobs.ERROR, stage="error", progress=0.0,
                                message="sweep failed", error=traceback.format_exc())
        return 1


def _run_grid(run_id: str) -> int:
    """Solve the converged date grid for one target (runs/grid/<id>/).

    Detached because it costs tens of seconds. The impulsive surface it replaces was instant and
    ran inline, which is why that surface had to be an approximation. Measured against converged
    solves, it was anti-correlated with the truth and hid most of the flyable window.
    """
    # Imported here so the other modes needn't load the solver stack.
    from prospector.trades.pipeline import grid as G

    try:
        spec = jobs.read_grid_job(run_id)
        options = dict(spec.get("options") or {})
        target = spec["target"]
        rc = ResolvedConfig.model_validate(spec["config"])
        polish = bool(options.pop("polish", True))
        # Cells within one flight-time step are independent, so fan them across processes; the
        # march itself is serial in flight time by construction. Leave a core for the UI.
        workers = int(options.pop("workers", 0) or 0)
        if workers <= 0:
            workers = max(1, (os.cpu_count() or 2) - 1)
        name = target.get("full_name") or target.get("pdes") or "target"

        def on_progress(done: int, total: int) -> None:
            jobs.write_grid_status(
                run_id, state=jobs.RUNNING, stage="grid", progress=0.9 * done / max(total, 1),
                message=f"{done}/{total} cells solved", cells_done=done, cells_total=total)

        jobs.write_grid_status(run_id, state=jobs.RUNNING, stage="start", progress=0.0,
                               message=f"solving the date grid for {name} "
                                       f"({workers} cells in parallel)")
        surface = G.lowthrust_grid(rc, target, workers=workers, progress=on_progress, **options)

        # The polish is a refinement of the grid, so a failure here leaves the grid standing rather
        # than losing the whole surface it was meant to sharpen.
        polished: list = []
        if polish and surface["n_feasible"]:
            jobs.write_grid_status(run_id, state=jobs.RUNNING, stage="polish", progress=0.92,
                                   message="sharpening the best-per-flight-time curve",
                                   cells_done=surface["n_cells"], cells_total=surface["n_cells"])
            try:
                polished = G.polish_best_per_flight_time(
                    rc, target, surface, workers=workers,
                    available_power_W=options.get("available_power_W"))
            except Exception:  # noqa: BLE001
                polished = []
        surface["polished"] = polished
        surface["frontier"] = G.best_per_flight_time(surface, polished)
        surface["departure_spread"] = G.departure_spread(surface)

        jobs.write_grid_result(run_id, surface)
        jobs.write_grid_summary(run_id, _grid_summary(spec, surface))
        jobs.write_grid_status(run_id, state=jobs.DONE, stage="done", progress=1.0,
                               message=f"{surface['n_feasible']}/{surface['n_cells']} cells solved",
                               cells_done=surface["n_cells"], cells_total=surface["n_cells"])
        return 0
    except Exception:
        jobs.write_grid_status(run_id, state=jobs.ERROR, stage="error", progress=0.0,
                               message="grid failed", error=traceback.format_exc())
        return 1


def _grid_summary(spec: dict, surface: dict) -> dict:
    """The flat row a grid run is identified and reloaded by, never the surface itself, which is
    large enough that a status poll must not carry it."""
    frontier = [p for p in (surface.get("frontier") or []) if p.get("dv_kms") is not None]
    best = min(frontier, key=lambda p: p["dv_kms"], default=None)
    target = spec.get("target") or {}
    return {
        "target": target.get("full_name") or target.get("pdes") or "target",
        "target_pdes": str(target.get("pdes") or ""),
        "n_cells": surface.get("n_cells"),
        "n_feasible": surface.get("n_feasible"),
        "nseg": surface.get("nseg"),
        "restarts": surface.get("restarts"),
        "best_dv_kms": None if best is None else float(best["dv_kms"]),
        "best_tof_days": None if best is None else float(best["tof_days"]),
        "best_dep_mjd2000": None if best is None else float(best["dep_mjd2000"]),
        "n_polished": len(surface.get("polished") or []),
    }


def _raise_keyboard_interrupt(_signum, _frame):
    raise KeyboardInterrupt


def _run_enrich(run_id: str) -> int:
    """Characterize this job's reachable set in the background, streaming per-target progress.

    Results go to the global per-object cache. Reports per-target counts to the job's status as it
    goes, stops early if a newer screen has replaced this job or it was cancelled, and on a
    terminating signal, whether Ctrl-C or a kill, unwinds cleanly, keeping every target described
    so far rather than discarding the batch.
    """
    # Imported here so the solve paths needn't load the enrichment deps, which are optional.
    from prospector import enrichment

    try:
        ids = jobs.read_enrich_ids(run_id)
        options = jobs.read_enrich_options(run_id)
        total = len(ids)
        jobs.write_enrich_status(run_id, state=jobs.RUNNING, done=0, total=total, message="starting")

        def on_progress(done: int, total_targets: int, message: str) -> None:
            jobs.write_enrich_status(run_id, state=jobs.RUNNING, done=done, total=total_targets,
                                     message=message)

        enrichment.enrich(ids, on_progress=on_progress, n_workers=options.get("n_workers"),
                          should_continue=lambda: jobs.enrich_should_continue(run_id))
        done = jobs.read_enrich_status(run_id).get("done", total)
        if jobs.enrich_cancelled(run_id):
            jobs.write_enrich_status(run_id, state=jobs.DONE, done=done, total=total,
                                     message="cancelled - kept what was characterized")
        elif not jobs.enrich_is_current(run_id):
            jobs.write_enrich_status(run_id, state=jobs.DONE, done=done, total=total,
                                     message="superseded by a newer screen")
        else:
            jobs.write_enrich_status(run_id, state=jobs.DONE, done=total, total=total,
                                     message="complete")
        return 0
    except enrichment.EnrichmentUnavailable as exc:
        jobs.write_enrich_status(run_id, state=jobs.ERROR, message="enrichment unavailable",
                                 error=str(exc))
        return 1
    except KeyboardInterrupt:
        # Interrupted (Ctrl-C / kill / app shutdown). Every target that finished is already in the
        # cache; record a terminal status so the UI doesn't poll a dead job forever.
        done = jobs.read_enrich_status(run_id).get("done", 0)
        jobs.write_enrich_status(run_id, state=jobs.DONE, done=done, total=done,
                                 message="stopped")
        return 1
    except Exception as exc:
        debug = getattr(exc, "debug", "")
        jobs.write_enrich_status(run_id, state=jobs.ERROR, message="enrichment failed",
                                 error=(traceback.format_exc() + ("\n\n" + debug if debug else "")))
        return 1


def main(run_id: str, mode: str = "sf") -> int:
    if mode == "spiral":
        return _run_spiral(run_id)
    if mode == "sweep":
        return _run_sweep(run_id)
    if mode == "grid":
        return _run_grid(run_id)
    if mode == "enrich":
        # Turn SIGTERM into KeyboardInterrupt so a `kill` (or app shutdown) unwinds through the
        # normal exit path, where the worker pool drains and the final status is written, rather
        # than dying mid-batch with the job stuck at "running" (SIGINT already does).
        try:
            signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
        except (ValueError, OSError):
            pass
        return _run_enrich(run_id)
    return _run_sf(run_id)


if __name__ == "__main__":
    if not 2 <= len(sys.argv) <= 3:
        print("usage: python worker.py <run_id> [spiral|sweep|grid|enrich]", file=sys.stderr)
        sys.exit(2)
    run_mode = sys.argv[2] if len(sys.argv) == 3 else "sf"
    if run_mode not in ("sf", "spiral", "sweep", "grid", "enrich"):
        print(f"unknown mode {run_mode!r}; expected 'sf', 'spiral', 'sweep', 'grid', or 'enrich'",
              file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], run_mode))
