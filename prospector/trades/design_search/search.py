"""The search session and its driver: evaluate every vehicle the axes describe.

:class:`Search` holds what one session shares: the mission, the resolved target, the engine and
propellant catalogs, the record of combinations already evaluated, and the stream of results the
app follows. :func:`run_search` runs the two stages. First the instant analytic pass, which can
rule a combination out but never rule one in, so it orders and narrows without deciding. Then the
real solve, over every combination of dry mass, propellant load, engine pairing, gas and swept
model setting that fits under the wet-mass limit.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

from prospector.config import ResolvedConfig, load_mission
from prospector.launch import load_launch_orbits
from prospector.population import get_target, load_population
from prospector.spacecraft import buildability
from prospector.spacecraft.propellants import load_propellants
from prospector.spacecraft.propulsion import load_engines
from prospector.trades.design_search.evaluate import (
    _combo_job,
    analytic_row,
    build_config,
    evaluate_one,
)
from prospector.trades.design_search.grid import model_combinations, parse_model_axes, parse_pairs
from prospector.trades.design_search.report import _json_default, persist_winner, write_outputs
from prospector.trades.design_search.session import (  # noqa: F401
    DEFAULT_OUT_ROOT,
    SPIRAL_CACHE_DIR,
    Cancelled,
    write_session_status,
)

log = logging.getLogger(__name__)


class Search:
    """Shared state for one search session: fixed mission/target, caches, results.

    Combinations spread across ``args.parallel`` worker processes, since PyKEP, pygmo and the
    spiral's integration all hold the GIL and threads would only coordinate. Every grid cell goes
    through one shared pool, and a lock guards the record of what has been evaluated.
    """

    def __init__(self, args):
        self.args = args
        self.mission = load_mission(args.mission)
        self.catalog = load_engines()
        self.launches = load_launch_orbits()
        # Resolve the target against the population at this study's screening cutoff, not the
        # lookup default (H<=25): faint bodies live only in a wider fetch, so a study that screens
        # deeper must resolve its sweep target just as deep. Against the whole candidate set,
        # planets included, since resolving against the asteroid query alone let "Mars" fall
        # through to the substring branch and land on the asteroid 343158 Marsyas.
        self.target = get_target(args.target, population=load_population(h_max=args.h_max))
        if self.target is None:
            raise SystemExit(
                f"target {args.target!r} not found in the population at H<={args.h_max:.1f} "
                f"- it may be fainter than the cutoff; raise --h-max to pull the fainter tail")
        # Re-osculate the target at the mission's arrival era (one cached Horizons fetch) so every
        # combo's porkchop seed and SF solve target the orbit the body flies then: a planetary
        # close approach inside the mission span can move a target's orbit substantially (Apophis's
        # April 2029 Earth flyby shifts it 0.92 -> 1.10 AU), so arrivals past one would otherwise
        # chase a stale orbit.
        from prospector import population
        self.target = population.refresh_target_elements(
            dict(self.target), self.mission.arrive_by)
        log.info(f"target elements: {self.target.get('elements_source', '?')} @ "
              f"{self.target.get('elements_epoch', 'SBDB epoch')} - "
              f"a {float(self.target['a']):.4f} AU, i {float(self.target['i']):.2f}°")
        self.pairs = (parse_pairs(args.pairs, self.catalog) if args.pairs
                      else parse_pairs([f"{args.engine}:" +
                                        ",".join(str(n) for n in args.engines)],
                                       self.catalog))
        launch = self.launches[self.mission.launch_orbit]
        self.vinf_values = ([float(self.mission.departure_vinf_kms or 0.0)]
                            if launch.escape_provided else [float(v) for v in args.vinf])
        # A stated array (the study vehicle's hardware number) is kept on every design, as the
        # project page keeps it; without one each design gets an array sized to its thrusters.
        self.array = ((float(args.array_W), float(args.array_m2 or 0.0))
                      if getattr(args, "array_W", None) else None)
        # The working gases every pairing is evaluated under; validated against the propellant
        # library so an unknown gas fails fast rather than silently estimating.
        prop_catalog = load_propellants()
        unknown = [g for g in args.propellants if g not in prop_catalog]
        if unknown:
            raise SystemExit(f"unknown propellant(s) {unknown}; available: "
                             f"{', '.join(sorted(prop_catalog))}")
        self.gases = list(dict.fromkeys(args.propellants)) or ["xenon"]
        # Model-setting axes swept on TOP of the vehicle grid: one override dict per combination,
        # multiplied into every (engine, count, gas, propellant) line. The power/mass/tank knobs
        # only bite when buildability runs (it sizes the arrays and tank the overrides move), so
        # warn rather than silently no-op.
        self.model_axes = parse_model_axes(args.sweep_param)
        self.model_combos = model_combinations(self.model_axes)
        model_kind = [k for k in self.model_axes
                      if buildability.SWEEPABLE[k][2] == "model"]
        if model_kind and not args.buildability:
            log.warning(f"sweeping {', '.join(model_kind)} has no effect without "
                  f"buildability (it sizes the arrays/tank these knobs move); "
                  f"re-run without --no-buildability.")
        self.sf_options = {
            "nseg": args.nseg, "maxeval": args.maxeval, "restarts": args.restarts,
            "max_duty_cycle": float(getattr(args, "cruise_duty", 1.0)), "rng_seed": 42,
        }
        self.spiral_options = {
            "radiation_model": getattr(args, "radiation_model", None),
            "coverglass_um": getattr(args, "coverglass_um", None),
            "coverglass_density": getattr(args, "coverglass_density", None),
        }
        self.parallel = args.parallel or max(1, (os.cpu_count() or 2) - 1)
        # With combo-level processes each combo solves its sweep points serially; only a serial
        # session fans the points themselves out.
        self.sweep_workers = 1 if self.parallel > 1 else (
            args.workers or min(len(self.vinf_values),
                                max(1, (os.cpu_count() or 2) - 1)))
        self.pool = (cf.ProcessPoolExecutor(max_workers=self.parallel)
                     if self.parallel > 1 else None)
        # Acquired here, so released here. A Search built outside ``main`` would otherwise leave
        # its workers running, and interpreter shutdown blocks joining them.
        self.lock = threading.Lock()
        self.evaluations: dict[tuple, dict] = {}  # (engine, gas, dry, prop, n) -> verdict
        # The denominator for the app's status bar, filled in once the combo count is known.
        self.progress: dict = {}
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        # What this session is searching, in one small file, which the app's tab labels sessions
        # from it without touching the heavy artifacts.
        (self.out / "session.json").write_text(json.dumps({
            "target": self.target.get("name"),
            "pdes": str(self.target.get("pdes", "")),
            "mission": args.mission, "study": getattr(args, "study", None),
            "engine": self.pairs[0][0],
            "pairs": [[k, n] for k, n in self.pairs],
            "engines": sorted({n for _, n in self.pairs}),
            "propellants": list(self.gases),
            "prop_range_kg": [args.prop_min, args.prop_max, args.prop_step],
            "dry_range_kg": [args.dry_min, args.dry_max, args.dry_step],
            "max_wet_kg": args.max_wet, "duty": args.duty,
            "vinf_kms": list(self.vinf_values),
            "array_W": (self.array[0] if self.array else None),
            # The swept model axes (key -> values), so the app can label the param_* columns and
            # recover what this session varied without re-parsing the flag.
            "sweep_axes": self.model_axes,
        }, indent=2))
        write_session_status(self.out, state="running",
                             message=f"starting - {self.target.get('name')}")


    def close(self) -> None:
        """Shut the worker pool down, cancelling anything not started."""
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None

    def __enter__(self) -> Search:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def rc(self, dry: float, prop: float, n_eng: int,
           engine_key: str | None = None, propellant_key: str = "xenon") -> ResolvedConfig:
        rc, _est = build_config(self.mission, self.catalog, self.launches,
                                dry_kg=dry, prop_kg=prop, n_engines=n_eng,
                                engine_key=engine_key or self.pairs[0][0],
                                propellant_key=propellant_key, array=self.array)
        return rc

    def evaluate(self, dry: float, prop: float, n_eng: int, engine_key: str,
                 propellant_key: str = "xenon", anchor: bool = False,
                 overrides: dict | None = None) -> dict:
        """Evaluate one combination, remembering the answer and sending it to the process pool
        when there is one. ``anchor`` marks the reference vehicle rather than a combination
        anybody asked for. ``overrides`` are the swept settings for this point: ``duty`` replaces
        the duty cycle and the rest override build-model coefficients. They are part of what
        identifies the point, and each is recorded as a ``param_<key>`` column."""
        if (self.out / "cancel").exists():
            raise Cancelled()
        overrides = overrides or {}
        # The override is part of a combo's identity: the same vehicle at two array margins is two
        # distinct points, so two cache entries.
        key = (engine_key, propellant_key, round(dry, 1), round(prop, 1), int(n_eng),
               tuple(sorted(overrides.items())))
        with self.lock:
            if key in self.evaluations:
                return self.evaluations[key]
        t0 = time.time()
        eff_duty = float(overrides.get("duty", self.args.duty))
        model_overrides = {k: v for k, v in overrides.items()
                           if buildability.SWEEPABLE[k][2] == "model"}
        kw = dict(engine_key=engine_key, propellant_key=propellant_key,
                  dry=dry, prop=prop, n_eng=n_eng,
                  duty=eff_duty, vinf_values=self.vinf_values,
                  # A swept duty axis moves the escape's duty; the cruise keeps its own limit.
                  sf_options=dict(self.sf_options),
                  spiral_options=dict(self.spiral_options),
                  retries=self.args.retries,
                  run_buildability=self.args.buildability,
                  sweep_workers=self.sweep_workers, model_overrides=model_overrides,
                  array=self.array)
        if self.pool is not None:
            verdict = self.pool.submit(
                _combo_job, {"mission": self.args.mission, "target": self.target,
                             "kw": kw}).result()
        else:
            verdict = evaluate_one(self.mission, self.catalog, self.launches,
                                   self.target, **kw)
        verdict["anchor"] = bool(anchor)
        verdict["elapsed_s"] = round(time.time() - t0, 1)
        # The swept knobs become plottable columns (the unswept axis is just absent).
        for k, v in overrides.items():
            verdict[f"param_{k}"] = v
        with self.lock:
            if key in self.evaluations:      # two lines raced to the same combo
                return self.evaluations[key]
            self.evaluations[key] = verdict
            self._log_verdict(verdict)
            # Stream the verdict so the app's tab can grow its table live.
            with (self.out / "evaluations.jsonl").open("a") as fh:
                fh.write(json.dumps(verdict, default=_json_default) + "\n")
            margin = verdict.get("margin_kms")
            write_session_status(
                self.out, state="running", n_evaluated=len(self.evaluations),
                progress=dict(self.progress),
                message=(f"dry {dry:.0f} + prop {prop:.0f} kg, "
                         f"{n_eng}x {engine_key} ({propellant_key}) - "
                         + (f"margin {margin:+.2f} km/s" if margin is not None
                            else verdict.get("why", ""))))
        return verdict

    @staticmethod
    def _log_verdict(v: dict) -> None:
        tag = "CLOSES" if v.get("success") else "no"
        margin = v.get("margin_kms")
        kg = v.get("prop_margin_kg")
        detail = (f"margin {margin:+.2f} km/s" + (f", {kg:+.0f} kg" if kg is not None else "")
                  if margin is not None else v.get("why", ""))
        ref = " [reference]" if v.get("anchor") else ""
        gas = v.get("propellant_key", "xenon")
        est = " est" if v.get("propellant_estimated") else ""
        log.info(f"  [{tag:>6}] dry {v['dry_kg']:.0f} + prop {v['prop_kg']:.0f} kg, "
              f"{v['n_engines']}x {v.get('engine', '?')} ({gas}{est}) - {detail} "
              f"({v['elapsed_s']:.0f}s){ref}")


def run_search(search: Search, args) -> int:
    dry_grid = [round(d, 1) for d in
                np.arange(args.dry_min, args.dry_max + args.dry_step / 2, args.dry_step)]
    prop_grid = [round(p, 1) for p in
                 np.arange(args.prop_min, args.prop_max + args.prop_step / 2,
                           args.prop_step)]

    log.info(f"target: {search.target.get('name')}  |  mission: {args.mission}  |  "
          f"out: {search.out}")

    # Each line fixes (engine, count, gas, propellant-load, model-overrides) and sweeps dry mass;
    # the propellant LOAD is a direct kg axis, the gas and the swept model settings separate
    # categorical axes (one override dict per model-setting combination).
    line_specs = [(ek, n, gas, prop, ov) for (ek, n) in search.pairs
                  for gas in search.gases for prop in prop_grid
                  for ov in search.model_combos]

    # Phase A: the analytic relationship map over the whole grid (instant, no SF). Only the duty
    # override touches the mass-and-schedule bound here; the build-model knobs move buildability
    # (Phase B), so they ride along as columns without changing the bound.
    log.info("\nPhase A: analytic map (capability vs the element-based estimate) ...")
    analytic = []
    for ek, n_eng, gas, prop, ov in line_specs:
        eff_duty = float(ov.get("duty", args.duty))
        for dry in dry_grid:
            if dry + prop > args.max_wet:
                continue
            rc = search.rc(dry, prop, n_eng, ek, gas)
            analytic.append({"dry_kg": dry, "prop_kg": prop,
                             "wet_kg": dry + prop,
                             "prop_dry_ratio": round(prop / dry, 4),
                             "n_engines": n_eng, "engine": ek, "propellant": gas,
                             "prop_offset": prop - dry,
                             **{f"param_{k}": v for k, v in ov.items()},
                             **analytic_row(rc, search.target,
                                            search.vinf_values, eff_duty)})
    n_dead = sum(1 for a in analytic if a["provably_infeasible"])
    log.info(f"  {len(analytic)} combos mapped; {n_dead} provably infeasible "
          f"(bound only - SF arbitrates the rest)")

    # Phase B: the SF-arbitrated solve. Grid cells are independent, so they dispatch through the
    # shared process pool via coordinating threads: the threads only wait on futures, the physics
    # runs in worker processes. A Stop request (the session's `cancel` file) lands between
    # evaluations; everything finished so far is still reported.
    lines = []
    cancelled = False
    anchor = None
    fan_out = max(1, search.parallel)

    def _gather(tasks):
        """Run thunks across coordinating threads; collect results, honor Stop."""
        nonlocal cancelled
        results = []
        if fan_out == 1 or len(tasks) == 1:
            for t in tasks:
                try:
                    results.append(t())
                except Cancelled:
                    cancelled = True
                    break
            return results
        with cf.ThreadPoolExecutor(max_workers=min(len(tasks), fan_out)) as tp:
            futures = [tp.submit(t) for t in tasks]
            for fut in futures:
                try:
                    results.append(fut.result())
                except Cancelled:
                    cancelled = True
        return results

    try:
        if args.anchor:
            # The reference vehicle, which is not one of the requested combinations but a fixed
            # point evaluated for cross-session comparison, and labeled as such everywhere. The
            # first requested pairing, so the reference is drawn from what this session is sweeping
            # rather than from an engine key baked into source.
            ref_engine = search.pairs[0][0]
            log.info(f"\nReference vehicle: dry 150 + prop 300, 3x {ref_engine} "
                  f"(the fixed cross-session reference)")
            anchor = search.evaluate(150.0, 300.0, 3, ref_engine, anchor=True)
            if not anchor["success"]:
                log.info("  note: the reference did not close for this target/mission - "
                      "expected for harder targets.")
    except Cancelled:
        cancelled = True
    if not cancelled:
        combos = [(ek, n, gas, dry, prop, ov) for ek, n, gas, prop, ov in line_specs
                  for dry in dry_grid if dry + prop <= args.max_wet]
        search.progress["n_planned"] = len(combos) + (1 if args.anchor else 0)
        log.info(f"\nPhase B: {len(combos)} combos, {fan_out} in parallel ...")
        _gather([lambda ek=ek, n=n, gas=gas, d=d, pr=pr, ov=ov:
                 search.evaluate(d, pr, n, ek, gas, overrides=ov)
                 for ek, n, gas, d, pr, ov in combos])
    if cancelled:
        log.info("\nstop requested - writing what finished")
    # One frontier record per line, read off the (possibly partial) grid results: the heaviest dry
    # mass that closed. The swept model overrides are part of a line's identity, so match them too.
    for ek, n_eng, gas, prop, ov in line_specs:
        ok = [v for v in search.evaluations.values()
              if v["n_engines"] == n_eng and v.get("engine") == ek
              and v.get("propellant_key", "xenon") == gas and v["prop_kg"] == prop
              and all(v.get(f"param_{k}") == val for k, val in ov.items())
              and v.get("success") and not v.get("anchor")]
        best = max(ok, key=lambda v: v["dry_kg"], default=None)
        lines.append({"engine": ek, "gas": gas, "n_engines": n_eng, "prop_kg": prop,
                      **{f"param_{k}": v for k, v in ov.items()},
                      "frontier_dry_kg": None if best is None else best["dry_kg"],
                      "frontier": best, "why": "grid"})

    write_outputs(search, lines, analytic, anchor)

    # The reference vehicle never competes for the win, since nobody asked for it. The win is the
    # LIGHTEST wet mass that closes AND builds (the design goal); when nothing passes both checks,
    # the lightest closer stands in.
    successes = [v for v in search.evaluations.values()
                 if v.get("success") and not v.get("anchor")]
    viable = [v for v in successes if v.get("buildable")]
    winner = min(viable or successes,
                 key=lambda v: (v["wet_kg"], -(v.get("prop_margin_kg") or 0.0)),
                 default=None)
    if winner and args.persist_winner and not cancelled:
        persist_winner(search, winner)
    if winner:
        tag = "closes+builds" if winner.get("buildable") else "closes (nothing builds)"
        gas = winner.get("propellant_key", "xenon")
        gas += "*" if winner.get("propellant_estimated") else ""
        log.info(f"\nWINNER ({tag}, lightest): dry {winner['dry_kg']:.0f} kg + prop "
              f"{winner['prop_kg']:.0f} kg, "
              f"{winner['n_engines']}x {winner.get('engine', '?')} ({gas}) "
              f"({winner['prop_margin_kg']:+.0f} kg propellant margin, "
              f"{winner['margin_kms']:+.2f} km/s at rated Isp)")
        message = (f"winner ({tag}): dry {winner['dry_kg']:.0f} + prop "
                   f"{winner['prop_kg']:.0f} kg, "
                   f"{winner['n_engines']}x {winner.get('engine', '?')} ({gas}) "
                   f"({winner['prop_margin_kg']:+.0f} kg)")
    else:
        log.info("\nNo requested combo closed; see the near-miss table in REPORT.md.")
        message = "no combo closed - see the near-miss table"
    if cancelled:
        message = f"stopped by request - {message}"
    write_session_status(search.out, state="stopped" if cancelled else "done",
                         message=message, n_evaluated=len(search.evaluations))
    return 0
