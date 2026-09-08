"""Export a converged mission solve as a self-contained trajectory bundle.

The bundle is a small folder describing one flown mission, meant for 3D viewers and further
simulation, and it does not favour any one viewer: plain JSON and YAML in SI units, with each phase
left in its own frame and labelled as such, rather than flattened into some particular tool's
conventions.

  trajectory.json  the whole trajectory: the escape spiral around Earth followed by the cruise
                   around the Sun, put on one clock that starts at t = 0 when the spacecraft is
                   dropped off. Each phase stays in its own frame, the spiral relative to
                   Earth's equator and the cruise relative to the Sun and Earth's orbit, both
                   in metres, so a viewer places them knowingly rather than guessing.
  asteroid.json    the target body's heliocentric orbit (a closed polyline) plus a time-track
                   sampled on the cruise clock, for a moving target entity.
  vehicle.yaml     the physical properties prospector knows about the vehicle: its mass
                   breakdown, its engines, and what they can do. Anything to do with pointing the
                   spacecraft, such as inertia and thruster geometry, is missing, because
                   prospector does not model it and a viewer supplies its own.

The escape spiral and the cruise are separate solves, in ``runs/spiral/<id>`` and ``runs/<id>``, in
different frames. :func:`build_bundle` is the one place that puts them on a single clock.

Units in every emitted file are SI (metres, seconds, kg); angles in degrees; the launch epoch is
also given as an absolute UTC ISO-8601 string so a consumer can drive real mission dates.
"""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pykep as pk
import yaml

from prospector import jobs
from prospector.constants import EARTH_OBLIQUITY_DEG, G0_M_S2, SECONDS_PER_DAY
from prospector.solvers import lambert as lb

# AU in metres, taken from pykep rather than from prospector.constants. The stored ``*_au`` arrays
# were made by dividing SI positions by ``pk.AU``, so multiplying back by that same constant
# recovers what the solve produced. pykep 3 matches ``constants.AU_M``, but a library free to
# redefine its AU would otherwise offset the exported arc from the solved one. A test pins the
# agreement rather than trusting it.
AU_M = pk.AU
_THRUST_EPS = 1e-3              # throttle above this counts as "thrusting" (EP plume on)


# ---------------------------------------------------------------------------
# small array helpers
# ---------------------------------------------------------------------------

def _f(a) -> np.ndarray:
    return np.asarray(a, dtype=float)


def _unit_rows(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise unit vectors of an (N, 3) array; zero rows stay zero."""
    v = _f(v)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(n < eps, 0.0, v / np.maximum(n, eps))


def _dir_of_motion(pos: np.ndarray) -> np.ndarray:
    """Per-sample prograde direction (unit) from a finite difference of an (N, 3) path."""
    pos = _f(pos)
    if len(pos) < 2:
        return np.zeros_like(pos)
    return _unit_rows(np.gradient(pos, axis=0))


def _round(a: np.ndarray, nd: int = 3) -> list:
    """JSON-friendly nested list, rounded to keep the file small (mm precision is plenty)."""
    return np.round(_f(a), nd).tolist()


# ---------------------------------------------------------------------------
# bundle assembly
# ---------------------------------------------------------------------------

def build_bundle(rc, cruise_result: dict, spiral_result: dict | None) -> dict[str, str]:
    """Assemble the bundle from already-read result dicts.

    ``cruise_result`` is ``jobs.read_result(run_id)``, holding ``sf``, ``orbits`` and ``target``.
    ``spiral_result`` is the ``spiral`` part of ``jobs.read_spiral_result(...)``, or None when the
    launch vehicle did the escaping, in which case the trajectory is the cruise alone. ``rc`` is
    the ResolvedConfig, for the vehicle properties.

    Returns a ``{filename: text}`` dict ready to write to a config folder or zip.
    """
    sf = cruise_result["sf"]
    orbits = cruise_result.get("orbits", {})
    target = cruise_result.get("target", {})

    dep_mjd2000 = float(sf["dep_mjd2000"])
    cruise_tof_days = float(sf["tof_days"])
    spiral_tof_days = float(spiral_result["tof_days"]) if spiral_result else 0.0

    # One launch-relative clock: t = 0 at injection, escape at spiral_tof, arrival at the end. The
    # cruise departs at dep_mjd2000, which IS the escape epoch, so launch precedes it by the
    # spiral's flight time. When the escape is LV-provided there is no spiral and launch == dep.
    launch_mjd2000 = dep_mjd2000 - spiral_tof_days
    spiral_end_t_s = spiral_tof_days * SECONDS_PER_DAY
    arrival_t_s = (spiral_tof_days + cruise_tof_days) * SECONDS_PER_DAY

    meta = {
        "name": str(rc.vehicle.name),
        "target_name": str(target.get("name", "")).strip(),
        "launch_mjd2000": launch_mjd2000,
        "dep_mjd2000": dep_mjd2000,
        "launch_utc": lb.date_from_mjd2000(launch_mjd2000).isoformat() + "Z",
        "dep_utc": lb.date_from_mjd2000(dep_mjd2000).isoformat() + "Z",
        "arrival_utc": lb.date_from_mjd2000(dep_mjd2000 + cruise_tof_days).isoformat() + "Z",
        "obliquity_deg": EARTH_OBLIQUITY_DEG,
        "spiral_end_t_s": spiral_end_t_s,
        "arrival_t_s": arrival_t_s,
        "au_m": AU_M,
        "frames": {
            "spiral": "geocentric_equatorial_m",
            "cruise": "heliocentric_ecliptic_m",
        },
        "phases": (["spiral"] if spiral_result else []) + ["cruise"],
    }

    trajectory = {
        "meta": meta,
        "spiral": _spiral_block(spiral_result,
                                thrust_N=rc.total_thrust_mN * 1e-3,
                                isp_s=rc.effective_isp),
        "cruise": _cruise_block(sf, spiral_end_t_s),
        "earth_orbit": _body_block(orbits.get("earth_au"), orbits.get("earth_track_au"),
                                   sf.get("fine_times_days"), spiral_end_t_s),
    }

    asteroid = {
        "name": str(target.get("name", "")).strip() or "target",
        "frame": "heliocentric_ecliptic_m",
        "au_m": AU_M,
        **_body_block(orbits.get("target_au"), orbits.get("target_track_au"),
                      sf.get("fine_times_days"), spiral_end_t_s),
    }

    return {
        "trajectory.json": json.dumps(trajectory, separators=(",", ":")),
        "asteroid.json": json.dumps(asteroid, separators=(",", ":")),
        "vehicle.yaml": _vehicle_yaml(rc, sf),
    }


def _spiral_block(spiral: dict | None, max_points: int = 25000,
                  thrust_N: float = 0.0, isp_s: float = 0.0) -> dict | None:
    """The escape phase: positions relative to Earth's equator in metres, seconds since launch,
    pointing along the velocity.

    ``mass_kg`` is the mass at each point as flown, from the propagator, which switches the engine
    off in eclipse, applies the duty cycle and throttles back with array power. Where that mass
    history and the engine's rated ``thrust_N`` and ``isp_s`` are available, ``throttle`` and
    ``thrust_on`` come from how fast mass is being used (throttle_i = mdot_i / mdot_full), so
    eclipses and duty cycling show up point by point; without a mass history they fall back to
    assuming the engine ran flat out.

    Keep the stored path at full resolution, since it is only a few samples per orbit as it is. A
    viewer curves between these points to draw a smooth coil, so it needs them less than 180
    degrees apart. Only thin it out if it is absurdly large.
    """
    if not spiral or not len(spiral.get("positions_km", [])):
        return None
    t_days = _f(spiral["times_days"])
    pos_m = _f(spiral["positions_km"]) * 1000.0
    mass = spiral.get("mass_kg")
    mass = _f(mass) if mass is not None and len(mass) == len(t_days) else None
    if len(t_days) > max_points:
        stride = int(np.ceil(len(t_days) / max_points))
        keep = np.unique(np.r_[np.arange(0, len(t_days), stride), len(t_days) - 1])
        t_days, pos_m = t_days[keep], pos_m[keep]
        if mass is not None:
            mass = mass[keep]

    if mass is not None and thrust_N > 0.0 and isp_s > 0.0:
        # Real thrust profile from the flown mass flow (full precision, before rounding).
        mdot_full = thrust_N / (isp_s * G0_M_S2)
        dt = np.maximum(np.diff(t_days * SECONDS_PER_DAY), 1e-6)
        thr = np.zeros(len(t_days))
        thr[:-1] = np.clip((-np.diff(mass) / dt) / mdot_full, 0.0, 1.0)
        thr[-1] = thr[-2] if len(thr) > 1 else 0.0
        throttle = [round(float(x), 3) for x in thr]
        thrust_on = [bool(x > _THRUST_EPS) for x in thr]
    else:
        throttle = [1.0] * len(t_days)        # no mass history: assume full thrust throughout
        thrust_on = [True] * len(t_days)

    block = {
        "t_s": np.round(t_days * SECONDS_PER_DAY, 1).tolist(),
        "pos_m": _round(pos_m, 1),
        "vel_dir": _round(_dir_of_motion(pos_m), 5),
        "thrust_on": thrust_on,
        "throttle": throttle,
    }
    if mass is not None:
        block["mass_kg"] = _round(mass, 3)
    return block


def _cruise_block(sf: dict, spiral_end_t_s: float) -> dict:
    """Cruise phase: heliocentric ecliptic positions (m) on the dense fine arc for smooth
    motion, plus the coarse (~20-segment) node pointing directions for slerped attitude. Times
    continue after the spiral so the whole mission shares one clock."""
    fine_t = _f(sf["fine_times_days"]) * SECONDS_PER_DAY + spiral_end_t_s
    fine_pos = _f(sf["fine_positions_au"]) * AU_M
    fine_thr = _f(sf["fine_throttle"])

    node_t = _f(sf["node_times_days"]) * SECONDS_PER_DAY + spiral_end_t_s
    node_thr = _f(sf["throttle"])
    # node_thrust_vec is (thrust direction * throttle); normalize to a pure pointing direction.
    node_dir = _unit_rows(sf["node_thrust_vec"])

    nodes = [
        {"t_s": round(float(t), 1),
         "dir": [round(float(d[0]), 5), round(float(d[1]), 5), round(float(d[2]), 5)],
         "on": bool(thr > _THRUST_EPS)}
        for t, d, thr in zip(node_t, node_dir, node_thr)
    ]
    return {
        "t_s": np.round(fine_t, 1).tolist(),
        "pos_m": _round(fine_pos, 1),
        "thrust_on": [bool(x > _THRUST_EPS) for x in fine_thr],
        "throttle": [round(float(x), 3) for x in fine_thr],   # EP thrust level, fraction of max
        "nodes": nodes,
    }


def _body_block(orbit_au, track_au, fine_times_days, spiral_end_t_s: float) -> dict:
    """A celestial body for the cruise (heliocentric ecliptic) scene: a closed orbit polyline
    (m) plus a time-track (m) sampled on the same clock as the cruise fine arc."""
    out: dict = {}
    if orbit_au is not None and len(orbit_au):
        out["path_m"] = _round(_f(orbit_au) * AU_M, 0)
    if track_au is not None and len(track_au) and fine_times_days is not None:
        t_s = _f(fine_times_days) * SECONDS_PER_DAY + spiral_end_t_s
        out["track"] = {
            "t_s": np.round(t_s, 1).tolist(),
            "pos_m": _round(_f(track_au) * AU_M, 0),
        }
    return out


def _vehicle_yaml(rc, sf: dict) -> str:
    """The prospector-known vehicle physical properties. A consumer fills in the attitude-only
    fields (inertia, thruster geometry) from its own model when they are absent here."""
    v = rc.vehicle
    perf = rc.performance  # blended assembly (isp_s, thrust_mN, power_W, mass_kg)
    doc = {
        "name": str(v.name),
        "source": "prospector",
        "dry_mass_kg": float(v.dry_mass),
        "fuel_mass_kg": float(v.fuel_mass),
        "unusable_prop_kg": float(v.unusable_prop),
        "wet_mass_kg": float(v.wet_mass),
        "propellant": v.propellant,
        "solar_power_W": float(v.solar_power_W),
        "engines": [
            {"type": m.type, "count": m.count,
             "thrust_mN": float(getattr(rc.engines.get(m.type), "thrust_mN", 0.0)),
             "isp_s": float(getattr(rc.engines.get(m.type), "isp_s", 0.0)),
             "power_W": float(getattr(rc.engines.get(m.type), "power_W", 0.0))}
            for m in v.engines
        ],
        # The resolved engine set, and the operating point the cruise flew at.
        "propulsion": {
            "total_thrust_mN": float(perf.get("thrust_mN", 0.0)),
            "effective_isp_s": float(perf.get("isp_s", 0.0)),
            "total_power_W": float(perf.get("power_W", 0.0)),
            "cruise_operating_thrust_N": float(sf.get("thrust_N", 0.0)),
            "cruise_operating_isp_s": float(sf.get("isp_s", 0.0)),
        },
    }
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# convenience: read by run id + zip for the UI download
# ---------------------------------------------------------------------------

def build_bundle_from_runs(rc, cruise_run_id: str, spiral_run_id: str | None) -> dict[str, str]:
    """Read the on-disk results for the given runs and assemble the bundle."""
    cruise_result = jobs.read_result(cruise_run_id)
    if not cruise_result:
        raise ValueError(f"no cruise result for run {cruise_run_id}")
    spiral_result = None
    if spiral_run_id:
        sr = jobs.read_spiral_result(spiral_run_id)
        spiral_result = sr.get("spiral") if sr else None
    return build_bundle(rc, cruise_result, spiral_result)


def bundle_zip_bytes(files: dict[str, str]) -> bytes:
    """Zip a ``{filename: text}`` bundle into an in-memory archive for ``ui.download``."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return buf.getvalue()
