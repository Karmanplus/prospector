"""Evaluating one candidate vehicle: assemble it, price its escape, fly its cruise.

This is the work the search spreads across processes. :func:`evaluate_one` is the whole answer for
one (dry mass, propellant, engine count, gas) point and holds no session state, so it can run in a
child process. Around it: :func:`build_config` puts the candidate onto the mission,
:func:`spiral_curve` flies and caches its escape, :func:`analytic_row` works out the instant map
that orders the search, and :func:`evaluate_combo` runs the cruise sweep against the priced escape.

The escape spiral is cached per distinct combination of wet mass, engines and power terms, because
its physics does not depend on how that wet mass splits between dry mass and propellant. It is
flown past the point the tank runs dry, so a vehicle whose tank is too small still produces the
whole curve, which is what a near miss needs in order to say how near it was.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import timedelta
from pathlib import Path

import numpy as np

from prospector.config import EngineMount, ResolvedConfig, Screening, Vehicle, load_mission
from prospector.constants import G0_KM_S2
from prospector.launch import escape_dv_estimate, load_launch_orbits, spiral_time_estimate_days
from prospector.solvers.simsflanagan import MISMATCH_TOL
from prospector.spacecraft import buildability
from prospector.spacecraft.propellants import load_propellants
from prospector.spacecraft.propulsion import load_engines
from prospector.trades.design_search.session import SPIRAL_CACHE_DIR


def build_config(mission, catalog, launches, *, dry_kg: float, prop_kg: float,
                 n_engines: int, engine_key: str, propellant_key: str = "xenon",
                 propellants: dict | None = None) -> tuple[ResolvedConfig, bool]:
    """One candidate vehicle on the fixed mission, fully resolved, running a chosen gas.

    Returns ``(config, estimated)``. The engine is moved onto ``propellant_key`` by
    :func:`prospector.spacecraft.propellants.engine_on_propellant`, which uses measured numbers
    when the engine already names that gas and otherwise estimates them by scaling its xenon
    figures. The scaled engine goes into the catalog, so the resolved config's Isp, thrust and
    power all reflect the gas, and so do the spiral, sweep and buildability built on them.
    ``estimated`` comes back so the result can say which it was.
    """
    from prospector.spacecraft.propellants import engine_on_propellant, load_propellants

    if propellants is None:
        propellants = load_propellants()
    scaled, estimated = engine_on_propellant(catalog[engine_key], propellant_key, propellants)
    catalog = {**catalog, engine_key: scaled}
    vehicle = Vehicle(
        name=f"search {dry_kg:.0f}/{prop_kg:.0f} x{n_engines} {propellant_key}",
        dry_mass=float(dry_kg), fuel_mass=float(prop_kg), unusable_prop=0.0,
        engines=[EngineMount(type=engine_key, count=int(n_engines))],
    )
    rc = ResolvedConfig.build(mission, vehicle, Screening(dv_margin_factor=1.0),
                              catalog, launches=launches)
    return rc, estimated


def analytic_row(rc: ResolvedConfig, target_row: dict, vinf_grid: list[float],
                 duty: float) -> dict:
    """What one combination can do, against the least it could possibly need. No real solve.

    The bound is the smallest, over every departure speed, of the escape estimate plus a floor on
    what the cruise can cost. Both terms have to undercharge for the answer to mean anything: a
    negative `bound_margin` then proves the combination cannot work, while a positive one proves
    nothing and the real solver decides. `window_days` is how much time is left for the cruise
    after the estimated spiral, which is the same story told in schedule instead of delta-v.

    The cruise floor is the element-based estimate, held below itself by
    :data:`impulse_margin.CERTIFIED_MARGIN`. A two-burn Lambert cost is not a floor on a low-thrust
    transfer, whatever it looks like: the best unlimited-thrust answer may use many burns, so that
    cost is an upper limit on the best impulsive answer instead. Measured over three targets it
    runs 2.9-5.2x the converged cost and moves the wrong way against it as flight time changes,
    whereas the element estimate runs 1.02-1.14 and the margin below it produced no false
    rejections at all.
    """
    from prospector.solvers import impulse_margin as im
    from prospector.solvers.edelbaum import lowthrust_dv

    capability = rc.total_dv_capability
    # It does not depend on dates, so it is worked out once rather than per departure speed.
    cruise_floor = im.CERTIFIED_MARGIN * float(
        lowthrust_dv(target_row["a"], target_row["e"], target_row["i"]))
    best_bound, best_vinf, best_window = math.inf, None, None
    for vinf in vinf_grid:
        esc = escape_dv_estimate(rc.launch, vinf)
        tof = spiral_time_estimate_days(
            rc.launch, wet_mass_kg=rc.vehicle.wet_mass,
            thrust_N=rc.total_thrust_mN * 1e-3, isp_s=rc.effective_isp,
            vinf_kms=vinf, duty_cycle=duty)
        start, end = rc.mission.launch_window
        shift = timedelta(days=round(tof))
        window_days = (rc.mission.arrive_by - (end + shift)).days
        if window_days < 60:           # not even the shortest possible trip fits
            continue
        # The departure v-infinity is credited back: the escape charge above already paid for it,
        # and a free departure excess offsets up to that much of any transfer's delta-v.
        # Optimistic, as a bound whose only verdict is rejection must be.
        floor = max(0.0, cruise_floor - vinf)
        bound = esc + (floor if math.isfinite(floor) else 0.0)
        if bound < best_bound:
            best_bound, best_vinf, best_window = bound, vinf, window_days
    return {
        "capability_kms": capability,
        "bound_kms": None if math.isinf(best_bound) else best_bound,
        "bound_margin_kms": None if math.isinf(best_bound) else capability - best_bound,
        "bound_vinf_kms": best_vinf,
        "window_days": best_window,
        "provably_infeasible": (math.isinf(best_bound)          # no v-inf leaves a window
                                or capability < best_bound),
    }


def _no_spiral_curve(vinf_max: float) -> dict:
    """The escape record for a launch type whose vehicle delivers escape: there is no spiral.

    ``spiral_curve`` reaches the propagator through the low-level entry point, which flies the
    drop-off orbit directly with a 1 kg mass floor and does not carry the ``escape_provided`` check
    that :func:`spiral.solve_for_config` has. Without this shortcut, a mission whose launch vehicle
    already escapes would fly a full climb out of the drop-off orbit for every distinct wet mass
    and then charge for it, an escape somebody else already paid for, costing both run time and
    accuracy.

    So every term reported here is zero or neutral: no delta-v, no flight time, no time in the
    belts, and an undamaged array. The trade curve is flat at zero across the whole speed range,
    which is right: the launch vehicle hands over the speed for free, so asking for more of it
    costs the spacecraft nothing.
    """
    v = [0.0, float(vinf_max)]
    return {"status": "escaped", "dv_at_escape_kms": 0.0, "tof_at_escape_days": 0.0,
            "belt_days": 0.0, "eclipse_days": 0.0, "revolutions": 0.0,
            "power_fraction_end": 1.0, "power_limited": False,
            "power_required_W": 0.0, "power_available_end_W": 0.0,
            "curve_vinf_kms": v, "curve_dv_kms": [0.0, 0.0], "curve_tof_days": [0.0, 0.0],
            "key": {"escape": "launch-vehicle"}}


def spiral_curve(rc: ResolvedConfig, *, duty: float, vinf_max: float,
                 cache_dir: Path, bol_power_W: float = 0.0,
                 power_req_W: float | None = None, array_to_thruster_eff: float | None = None,
                 array_reserve_W: float | None = None,
                 radiation_model: str | None = None, coverglass_um: float | None = None,
                 coverglass_density: float | None = None, max_years: float = 8.0) -> dict:
    """The propagated escape tradeoff curve for this wet mass + engine stack (cached).

    Flown through :func:`spiral.solve_for_config`, the same entry the app's escape runs use, so a
    sweep's escape is the app's escape: the drop-off orbit, the drag area, the array's chain
    efficiency and housekeeping reserve, the radiation scenario and coverglass, all read the same
    way. Two deliberate differences, both about what a curve is for. The vehicle is probed with a
    one-kilogram dry mass, so the run never ends for want of propellant and the curve extends over
    the whole departure-speed range; how much propellant the real vehicle has is the question the
    sweep answers afterwards, per point, off this curve. And ``bol_power_W`` is the array to fly
    (zero = power unlimited, the first pass), not the vehicle's saved array.

    The spiral depends only on wet mass, thrust, Isp, the drop-off orbit, the duty cycle and, when
    power is being tracked, the array and electronics terms, so the cache key is those; combos
    that share them share a curve. ``power_req_W`` is accepted for older callers and ignored: the
    stack's rated draw comes from the config.
    """
    if rc.launch.escape_provided:
        return _no_spiral_curve(vinf_max)
    from prospector.solvers import spiral as sp
    from prospector.spacecraft.radiation import DEFAULT_RADIATION_MODEL

    model_name = radiation_model or DEFAULT_RADIATION_MODEL
    eff = rc.thruster_chain_eff() if array_to_thruster_eff is None else float(array_to_thruster_eff)
    reserve = rc.array_reserve_W() if array_reserve_W is None else float(array_reserve_W)
    # The stack's power->(thrust, Isp) throttle curve, so the escape reads the same operating point
    # the interactive spiral does as the array degrades. Fingerprinted into the cache key so a
    # curve edit (even at unchanged rated thrust/Isp) busts the cache.
    power_grid = rc.power_thrust_isp_grid()
    grid_fp = (None if power_grid is None else
               hashlib.sha1(np.round(np.concatenate(power_grid), 4).tobytes()).hexdigest()[:8])
    array_model = rc.bus_model().array_model()
    array_fp = hashlib.sha1(
        json.dumps(array_model.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()[:8]
    key = {
        "v": 10,                           # cache schema: flown through spiral.solve_for_config
        "perigee": rc.launch.perigee_alt_km, "apogee": rc.launch.apogee_alt_km,
        "inc": rc.launch.inclination_deg, "min_alt": rc.launch.min_thrust_alt_km,
        "wet": round(rc.vehicle.wet_mass, 3), "thrust": round(rc.total_thrust_mN, 6),
        "isp": round(rc.effective_isp, 3), "duty": duty, "vinf_max": vinf_max,
        "bol_power_W": round(float(bol_power_W), 1),
        "power_req_W": round(float(rc.total_power_W), 1),
        "thruster_eff": round(float(eff), 6), "reserve_W": round(float(reserve), 3),
        "area_m2": round(float(rc.vehicle.area_m2), 4), "max_years": float(max_years),
        "radiation": model_name, "coverglass_um": coverglass_um,
        "coverglass_density": coverglass_density, "pgrid": grid_fp, "array": array_fp,
    }
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"spiral_{digest}.json"
    if path.is_file():
        return json.loads(path.read_text())

    # The probe vehicle: the same wet mass, thrust and drag area, a 1 kg dry mass so the climb
    # never runs dry, and the array this pass flies.
    probe = rc.model_copy(update={"vehicle": rc.vehicle.model_copy(update={
        "dry_mass": 1.0, "unusable_prop": 0.0, "fuel_mass": float(rc.vehicle.wet_mass) - 1.0,
        "solar_power_W": float(bol_power_W)})})
    opts = dict(duty_cycle=duty, target_vinf_kms=vinf_max, max_years=float(max_years),
                radiation_model=model_name, array_to_thruster_eff=eff, array_reserve_W=reserve)
    if coverglass_um is not None:
        opts["coverglass_um"] = float(coverglass_um)
    if coverglass_density is not None:
        opts["coverglass_density"] = float(coverglass_density)
    sol = sp.solve_for_config(probe, **opts)
    idx = np.unique(np.linspace(0, max(len(sol.curve_vinf_kms) - 1, 0),
                                min(200, max(len(sol.curve_vinf_kms), 1))).astype(int))
    curve = {
        "status": sol.status,
        # The budget term: propellant burned as rated-Isp delta-v, which is what the point's
        # total_dv is read against. The velocity change flown rides along for the record.
        "dv_at_escape_kms": (sol.dv_at_escape_equiv_kms if sol.dv_at_escape_equiv_kms is not None
                             else sol.dv_at_escape_kms),
        "dv_at_escape_flown_kms": sol.dv_at_escape_kms,
        "tof_at_escape_days": sol.tof_at_escape_days,
        "belt_days": float(sol.belt_days),
        "eclipse_days": float(sol.eclipse_days),
        "revolutions": float(sol.revolutions),
        "power_fraction_end": float(sol.power_fraction_end),
        # Power adequacy of the sized array against the thruster's full-power demand at the
        # most-degraded point of the escape (zero/False on the power-rich first pass).
        "power_limited": bool(sol.power_limited),
        "power_required_W": float(sol.power_required_W),
        "power_available_end_W": float(sol.power_available_end_W),
        "curve_vinf_kms": np.asarray(sol.curve_vinf_kms)[idx].tolist(),
        "curve_dv_kms": np.asarray(sol.curve_dv_kms)[idx].tolist(),
        "curve_tof_days": np.asarray(sol.curve_tof_days)[idx].tolist(),
        "key": key,
    }
    path.write_text(json.dumps(curve))
    return curve


def evaluate_combo(rc: ResolvedConfig, target_row: dict, curve: dict,
                   vinf_values: list[float], sf_options: dict,
                   workers: int) -> dict:
    """Run the escape-priced SF sweep for one vehicle and render the verdict.

    It works if at least one sweep point both converged and fits the tank, meaning the rocket
    equation at the solved escape-plus-cruise delta-v needs no more propellant than was loaded. The
    headline margin is taken at the best point that converged, so a near miss reports a shortfall
    worth trusting.
    """
    from prospector.trades import sweep

    # The cruise flies at the array's frozen post-escape power: after the belt transit the array is
    # permanently degraded, so a power-limited escape hands the cruise (and the laden return) a
    # reduced operating point - lower thrust and Isp off the engine curve. None (the power-rich
    # first pass, or buildability off) leaves the cruise at full rated thrust.
    opts = dict(sf_options)
    avail = curve.get("power_available_end_W")
    if avail is not None and float(avail) > 0.0:
        opts["available_power_W"] = float(avail)
        opts["return_options"] = {**(opts.get("return_options") or {}),
                                  "available_power_W": float(avail)}
    points = sweep.run_sweep(rc.model_dump(mode="json"), target_row,
                             vinf_values=vinf_values, curve=curve,
                             options=opts, workers=workers)

    capability = rc.total_dv_capability
    veff = rc.effective_isp * G0_KM_S2
    usable = rc.vehicle.fuel_mass - rc.vehicle.unusable_prop
    return_trip = rc.mission.return_trip
    rows = []
    for p in points:
        converged = bool(p.get("feasible")) or (
            p.get("mismatch") is not None and float(p["mismatch"]) <= MISMATCH_TOL)
        total_dv = p.get("total_dv_kms")
        margin = prop_needed = prop_margin = None
        # The laden return leg's propellant (cruise + insertion), zero when one-way.
        return_prop = float(p.get("return_propellant_kg") or 0.0)
        if total_dv is not None:
            margin = capability - float(total_dv)
            # The whole mission's propellant against the tank. The point records what each leg
            # spent at the operating points it flew (``propellant_kg``, which already carries the
            # return when there is one); the rocket equation at rated Isp is the fallback for a
            # point without it, and understates a power-limited cruise.
            if p.get("propellant_kg") is not None:
                prop_needed = float(p["propellant_kg"])
            else:
                prop_needed = (rc.vehicle.wet_mass * (1.0 - math.exp(-float(total_dv) / veff))
                               + return_prop)
            prop_margin = usable - prop_needed
        # Closing is a question of kilograms: the legs run at different Isps (and the escape at a
        # changing one), so a single-Isp delta-v margin can read positive while the tank comes up
        # short. ``margin_kms`` stays as the budget-basis figure; the verdict is the propellant.
        # A round trip must also have its return converge.
        return_ok = (not return_trip) or bool(p.get("return_feasible"))
        # And the answer's leg Isp must agree with the Isp its own path implies. The thrust
        # ceilings are part of the problem, so they hold by construction; the Isp is settled after
        # the solve, and one that did not settle is priced at a slightly wrong Isp. It is not a
        # closed design until a re-solve settles it.
        settled = bool(p.get("settled", True))
        success = bool(converged and settled and prop_margin is not None and prop_margin >= 0.0
                       and return_ok)
        rows.append({**p, "converged": converged, "settled": settled, "margin_kms": margin,
                     "prop_needed_kg": prop_needed, "prop_margin_kg": prop_margin,
                     "success": success})

    converged_rows = [r for r in rows if r["converged"] and r["margin_kms"] is not None]
    # The headline point is the one that leaves the most propellant, the quantity the verdict is
    # judged on; the delta-v margin decides only between points without a propellant figure.
    best = (max(converged_rows, key=lambda r: (r["prop_margin_kg"] if r["prop_margin_kg"] is not None
                                               else r["margin_kms"]))
            if converged_rows else None)
    window_dead = all("window too short" in (r.get("error") or "") for r in rows)
    return {
        "points": rows,
        "n_converged": len(converged_rows),
        "best_point": best,
        "success": bool(best and best["success"]),
        "margin_kms": None if best is None else best["margin_kms"],
        "dv_short_kms": None if best is None else max(0.0, -best["margin_kms"]),
        "prop_margin_kg": None if best is None else best["prop_margin_kg"],
        # Mission clock at the best point: heliocentric cruise alone, and the whole mission (the
        # point's own refined escape time + cruise) - the per-point escape time corrects the raw
        # spiral for that point's mass and window.
        "cruise_tof_days": None if best is None else best.get("tof_days"),
        "total_tof_days": (None if best is None or best.get("tof_days") is None
                           else float(best.get("escape_tof_days") or 0.0)
                           + float(best["tof_days"])),
        # The ΔV ledger at the best point: what the cruise spent, what the whole mission spent
        # (escape + cruise), against the capability the tank holds.
        "cruise_dv_kms": None if best is None else best.get("cruise_dv_kms"),
        "total_dv_kms": None if best is None else best.get("total_dv_kms"),
        "capability_kms": capability,
        # The return-leg ledger at the best point (None for one-way missions).
        "return_dv_kms": None if best is None else best.get("return_dv_kms"),
        "return_prop_kg": None if best is None else best.get("return_propellant_kg"),
        "return_tof_days": None if best is None else best.get("return_tof_days"),
        "return_insertion_dv_kms": (None if best is None else
                                    (rc.insertion_dv_kms(_best_destination(rc))
                                     if return_trip and best.get("return_dv_kms") is not None else None)),
        "insertion_prop_kg": None if best is None else best.get("insertion_prop_kg"),
        "delivered_payload_kg": None if best is None else best.get("delivered_mass_kg"),
        "total_prop_kg": None if best is None else best.get("prop_needed_kg"),
        "round_trip_days": None if best is None else best.get("round_trip_days"),
        "why": ("ok" if best and best["success"]
                else "Isp unsettled" if best and not best["settled"]
                and best["prop_margin_kg"] is not None and best["prop_margin_kg"] >= 0.0
                else "return won't close" if best and return_trip
                and best["prop_margin_kg"] is not None and best["prop_margin_kg"] >= 0.0
                and not best["success"]
                else "tank short" if best and best["margin_kms"] is not None
                and best["margin_kms"] >= 0.0
                else "over budget" if best
                else "window too short" if window_dead
                else "SF did not converge"),
    }


def _best_destination(rc: ResolvedConfig):
    """The return destination object for this config (for the insertion-dV display)."""
    from prospector.launch import load_return_destinations
    return load_return_destinations()[rc.mission.return_destination]


def evaluate_one(mission, catalog, launches, target: dict, *, engine_key: str,
                 dry: float, prop: float, n_eng: int, duty: float,
                 vinf_values: list[float], sf_options: dict, retries: int,
                 run_buildability: bool, sweep_workers: int,
                 propellant_key: str = "xenon",
                 model_overrides: dict | None = None,
                 spiral_options: dict | None = None) -> dict:
    """One combination through the whole pipeline. Holds no session state, so it can run in a
    child process when combinations are spread across cores.

    The engine runs on ``propellant_key``, using measured numbers if it is characterised on that
    gas and estimated ones otherwise, and which it was travels through to the result. The first
    spiral runs with power treated as unlimited, which prices the escape and measures the time
    spent in the belts. The buildability check then sizes the array for the heaviest operating mode
    plus margin at start of life, since radiation damage is flown rather than sized against. The
    second spiral brings power in, so the sized array throttles the thrust back as it degrades
    through the belts and an undersized array escapes power-limited. Finally the sweep flies the
    cruise against that escape curve, at whatever power the array has left. A power-limited escape
    is a slower flight and not a rejection, and whether it still works is the trade an array-margin
    sweep exists to find.
    """
    propellants = load_propellants()
    rc, estimated = build_config(mission, catalog, launches, dry_kg=dry, prop_kg=prop,
                                 n_engines=n_eng, engine_key=engine_key,
                                 propellant_key=propellant_key, propellants=propellants)
    bus_model = buildability.apply_model_overrides(rc.bus_model(),
                                                   model_overrides or {})
    prop_name = (propellants[propellant_key].name if propellant_key in propellants
                 else propellant_key)
    vinf_max = max(vinf_values) + 0.3
    # The escape's radiation scenario and coverglass, as the app's escape runs set them.
    spiral_kw = {k: v for k, v in (spiral_options or {}).items()
                 if k in ("radiation_model", "coverglass_um", "coverglass_density", "max_years")
                 and v is not None}
    curve = spiral_curve(rc, duty=duty, vinf_max=vinf_max, cache_dir=SPIRAL_CACHE_DIR, **spiral_kw)
    build = None
    if run_buildability:
        engine = rc.engines[engine_key]
        build = buildability.assess_engine(
            dry_kg=dry, prop_kg=prop, n_engines=n_eng, engine=engine,
            belt_days=curve.get("belt_days"), eol_power_fraction=curve.get("power_fraction_end"),
            ignitions=curve.get("revolutions"), propellants=propellants, model=bus_model)
        if curve.get("status") == "escaped" and build["bol_power_W"] > 0:
            # Pass 2 closes the power loop: the sized array degrades through the belts and powers
            # the thrusters through the conversion chain their PPU wiring implies, so the escape
            # flies on that fraction, the same chain the array was sized against.
            curve = spiral_curve(rc, duty=duty, vinf_max=vinf_max,
                                 cache_dir=SPIRAL_CACHE_DIR,
                                 bol_power_W=build["bol_power_W"],
                                 array_to_thruster_eff=rc.thruster_chain_eff(bus_model),
                                 array_reserve_W=rc.array_reserve_W(bus_model), **spiral_kw)
            build = buildability.assess_engine(
                dry_kg=dry, prop_kg=prop, n_engines=n_eng, engine=engine,
                belt_days=curve.get("belt_days"),
                eol_power_fraction=curve.get("power_fraction_end"),
                ignitions=curve.get("revolutions"), propellants=propellants, model=bus_model)
    verdict = {"dry_kg": dry, "prop_kg": prop, "wet_kg": dry + prop,
               "prop_dry_ratio": round(prop / dry, 4),
               "n_engines": n_eng, "engine": engine_key,
               # The working gas this combo flies, and whether its thrust/Isp are measured
               # (native) or estimated by scaling xenon numbers (a baseline assumption).
               # propellant = display name (table chip); propellant_key = library key
               # (grouping, rebuild, gas-aware re-assessment).
               "propellant": prop_name, "propellant_key": propellant_key,
               "propellant_estimated": bool(estimated),
               # What the engines can do per kg of launch mass, which is the efficiency the vehicle
               # brings vs what it weighs. mN/kg is numerically the initial acceleration in mm/s^2.
               "thrust_mN_per_kg": round(rc.total_thrust_mN / (dry + prop), 4),
               "power_W_per_kg": round(rc.total_power_W / (dry + prop), 4),
               # Thrust per watt drawn, an engine property (count cancels), the efficiency axis the
               # engine choice itself sets.
               "thrust_mN_per_W": round(rc.total_thrust_mN / rc.total_power_W, 4),
               "escape_dv_kms": curve.get("dv_at_escape_kms"),
               "spiral_tof_days": curve.get("tof_at_escape_days"),
               "spiral_status": curve.get("status"),
               # Escape power adequacy: did the degrading array stay above the thruster's
               # full-power demand? (False/None until the power loop is closed in pass 2.)
               "power_limited": curve.get("power_limited"),
               "power_required_W": curve.get("power_required_W"),
               "power_available_end_W": curve.get("power_available_end_W")}
    if build is not None:
        verdict.update({k: build[k] for k in (
            "buildable", "build_why", "propellant", "lifetime_ok", "lifetime_why",
            "prop_per_thruster_kg", "throughput_margin_kg", "mission_ignitions",
            "payload_capacity_kg", "min_dry_kg",
            "eol_factor", "belt_days", "bol_power_W", "array_kg", "pcdu_kg",
            "thruster_sys_kg", "tank_kg", "fixed_bus_kg", "build_cost_musd")})
    # Only a spiral that never escaped counts as a failed flight. Skip the cruise and say so. A
    # power-limited escape is not a failure: the array is sized to the worst-mode load at beginning
    # of life, so belt degradation can drop it below full thrust, and the vehicle then flies
    # THROTTLED (the cruise inherits the reduced post-escape power, threaded above). Whether that
    # throttled mission still closes is the trade the array-margin sweep surfaces.
    fail_why = None
    if curve.get("status") != "escaped":
        fail_why = f"spiral {curve.get('status')}"
    if fail_why is not None:
        verdict.update({"success": False, "margin_kms": None, "dv_short_kms": None,
                        "prop_margin_kg": None, "points": [], "n_converged": 0,
                        "best_point": None, "cruise_tof_days": None,
                        "total_tof_days": None, "cruise_dv_kms": None,
                        "total_dv_kms": None,
                        "capability_kms": rc.total_dv_capability,
                        "return_dv_kms": None, "return_prop_kg": None,
                        "return_tof_days": None, "return_insertion_dv_kms": None,
                        "insertion_prop_kg": None, "delivered_payload_kg": None,
                        "total_prop_kg": None, "round_trip_days": None,
                        "why": fail_why})
    else:
        result = evaluate_combo(rc, target, curve, vinf_values, sf_options,
                                sweep_workers)
        if result["n_converged"] == 0 and retries > 0:
            # MBH is stochastic; one blanket non-convergence warrants a reseeded try.
            result = evaluate_combo(rc, target, curve, vinf_values,
                                    dict(sf_options, rng_seed=1337), sweep_workers)
        verdict.update(result)
    return verdict


def _combo_job(payload: dict) -> dict:
    """Child-process entry: rebuild the libraries (cheap YAML reads) and evaluate."""
    return evaluate_one(load_mission(payload["mission"]), load_engines(),
                        load_launch_orbits(), payload["target"], **payload["kw"])
