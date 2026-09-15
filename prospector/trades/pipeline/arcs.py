"""Turning a solved trajectory into something you can draw and check.

Sims-Flanagan treats thrust as one small kick per segment, so joining its nodes with coasting arcs
gives a path that zig-zags. The real trajectory is smooth, and :func:`_densify` gets it back by
re-flying each segment with its thrust applied continuously, the way the engine runs.

:func:`_assemble_leg` turns a solution into the block the UI and the report read: the smooth path,
per-node orbit and thrust-direction numbers, the two bodies' orbits, and the speeds at departure
and arrival. Used for both the outbound trip and the loaded return.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pykep as pk

ProgressFn = Callable[[str, float, str], None]


def _noop(stage: str, fraction: float, message: str) -> None:
    pass


def _sample_orbit(planet, t0_mjd2000: float, span_days: float, n: int = 240) -> np.ndarray:
    """Sample a body's heliocentric position over ``span_days`` as an (n, 3) array in AU."""
    epochs = np.linspace(t0_mjd2000, t0_mjd2000 + span_days, n)
    return np.asarray(planet.eph_v(epochs), float)[:, 0:3] / pk.AU


# Scales that strip the units off a segment for the integrator below: lengths in AU, speeds in the
# circular speed at 1 AU, times in the time unit those two imply.
_L = pk.AU
_V = math.sqrt(pk.MU_SUN / pk.AU)
_T = _L / _V
_ZOH: object | None = None


def _zoh_integrator():
    """The integrator used to re-fly one segment, built once.

    ``pk.ta.get_zoh_kep`` handles two-body motion plus a thrust that stays fixed in direction and
    size for the whole step, with mass draining at the matching rate, which is what one
    Sims-Flanagan segment amounts to. It works in unitless quantities (mu = 1), so states and
    parameters get scaled by :data:`_L` / :data:`_V` / :data:`_T` going in.
    """
    global _ZOH
    if _ZOH is None:
        _ZOH = pk.ta.get_zoh_kep(1e-16)
    return _ZOH


def _propagate_constant_thrust(r0, v0, m0, thrust_vec_N, dt_s, veff_ms):
    """Fly one segment at fixed thrust: SI state in, SI position out.

    ``thrust_vec_N`` is the segment's thrust in newtons, already scaled by the throttle, and
    ``veff_ms = Isp*g0``. A segment with the engine off comes out right as well, since the
    direction does not matter once the magnitude is zero, so coasting needs no separate path.
    """
    ta = _zoh_integrator()
    thrust = np.asarray(thrust_vec_N, float)
    mag = float(np.linalg.norm(thrust))
    direction = thrust / mag if mag > 0.0 else np.zeros(3)
    ta.time = 0.0
    ta.state[:] = np.concatenate([np.asarray(r0, float) / _L,
                                  np.asarray(v0, float) / _V, [1.0]])
    # p0 puts the thrust in the integrator's acceleration unit at unit mass; p1..p3 are its
    # direction; p4 turns that thrust into the matching mass flow (1/veff).
    ta.pars[:] = [mag * _L / (float(m0) * _V * _V), direction[0], direction[1], direction[2],
                  _V / float(veff_ms)]
    ta.propagate_until(float(dt_s) / _T)
    return np.asarray(ta.state[0:3], float) * _L


def _densify(trajectory: np.ndarray, thrust_N: float, veff_ms: float, n_points: int = 400):
    """Rebuild the flown path as a smooth curve, for plotting.

    Sims-Flanagan treats thrust as one small kick per segment, so joining its nodes with coasting
    arcs zig-zags. The real trajectory is smooth, because the engine bends the orbit gradually.
    Each segment holds its throttle fixed, so re-flying it with the fixed-thrust integrator
    reproduces that bend.

    ``thrust_N`` is the stack's maximum thrust and ``veff_ms = Isp*g0`` its exhaust speed; a
    segment's thrust is ``thrust_N * throttle_vector``. Returns ``(positions_au [~n_points,3],
    days, throttle)``.

    The 2*nseg+1 nodes sit at evenly spaced half-segments; segment ``i`` covers nodes ``2i..2i+2``
    with its throttle at node ``2i``. If the halves did not quite meet, that shows as a small step
    where they join, and the solve reports the size of the gap.
    """
    node_days = trajectory[:, 0] - float(trajectory[0, 0])
    nseg = (len(trajectory) - 1) // 2
    total = float(node_days[-1])
    if nseg < 1 or total <= 0:                  # nothing to fly
        return trajectory[:, 1:4] / pk.AU, node_days, trajectory[:, 8]

    seg_dt = total / nseg                        # segments split the flight time evenly
    per_seg = max(4, n_points // nseg)
    pos, days, thr = [], [], []
    for i in range(nseg):
        n0 = 2 * i
        r0 = [float(x) for x in trajectory[n0, 1:4]]
        v0 = [float(x) for x in trajectory[n0, 4:7]]
        m0 = float(trajectory[n0, 7])
        u_vec = trajectory[n0, 9:12]             # throttle vector, |u| <= 1
        thrust_vec = [float(thrust_N * c) for c in u_vec]
        throttle = float(np.linalg.norm(u_vec))
        for f in np.linspace(0.0, 1.0, per_seg, endpoint=False):
            r = _propagate_constant_thrust(r0, v0, m0, thrust_vec,
                                           float(f) * seg_dt * pk.DAY2SEC, veff_ms)
            pos.append(r / pk.AU)
            days.append(i * seg_dt + float(f) * seg_dt)
            thr.append(throttle)
    pos.append(np.asarray(trajectory[-1, 1:4], float) / pk.AU)
    days.append(total)
    thr.append(float(np.linalg.norm(trajectory[-1, 9:12])))
    return np.asarray(pos), np.asarray(days), np.asarray(thr)


def _track(planet, abs_mjd2000: np.ndarray) -> np.ndarray:
    """Where a body is (AU) at each of the given times. Drives the moving sphere."""
    return np.asarray(planet.eph_v(np.asarray(abs_mjd2000, float)), float)[:, 0:3] / pk.AU


def _node_diagnostics(trajectory: np.ndarray):
    """The orbit at each node, and which way the thrust points relative to it.

    Splitting the thrust into radial, transverse and normal components is useful because each
    direction changes a different thing:

        radial     (out from the Sun)            -> the shape (eccentricity)
        transverse (along the velocity)          -> the size (semi-major axis)
        normal     (perpendicular to the orbit)  -> the tilt (inclination)

    Returns ``(a_au, e, i_deg, thrust_radial, thrust_transverse, thrust_normal)``. Each thrust
    component is a signed fraction of maximum thrust, so coasting is 0 and the three add up to the
    throttle. Plotted next to a/e/i, they show what each burn buys.
    """
    n = trajectory.shape[0]
    a = np.empty(n)
    e = np.empty(n)
    inc = np.empty(n)
    radial = np.zeros(n)
    transverse = np.zeros(n)
    normal = np.zeros(n)
    for k in range(n):
        r = np.asarray([float(x) for x in trajectory[k, 1:4]])
        v = np.asarray([float(x) for x in trajectory[k, 4:7]])
        elements = pk.ic2par([list(r), list(v)], pk.MU_SUN)  # a (m), e, i (rad), Om, om, E
        a[k] = elements[0] / pk.AU
        e[k] = elements[1]
        inc[k] = np.degrees(elements[2])
        u = np.asarray([float(c) for c in trajectory[k, 9:12]])   # throttle vector, |u| <= 1
        r_mag = float(np.linalg.norm(r))
        h = np.cross(r, v)
        h_mag = float(np.linalg.norm(h))
        if r_mag > 0 and h_mag > 0:
            r_hat = r / r_mag
            n_hat = h / h_mag                    # perpendicular to the orbit
            t_hat = np.cross(n_hat, r_hat)       # in the orbit, roughly along the velocity
            radial[k] = float(np.dot(u, r_hat))
            transverse[k] = float(np.dot(u, t_hat))
            normal[k] = float(np.dot(u, n_hat))
    return a, e, inc, radial, transverse, normal


def _mission_elements(rc, target_row) -> dict:
    """The target's orbit, refreshed for the mission's arrival date (``arrive_by``).

    The cached SBDB orbit is quoted for one date, and propagating it forward breaks across a
    planetary close approach: Apophis's April 2029 Earth flyby moves it from 0.92 to 1.10 AU.
    Horizons integrates the real dynamics, so asking it for the orbit at the arrival date keeps the
    simple model accurate where the rendezvous happens. Cached per body and date, so only the first
    call for a mission goes to the network; offline it returns the row unchanged with a warning
    rather than failing. An arrival on the far side of a close approach from ``arrive_by`` still
    carries some error, which the Sims-Flanagan solve sorts out.
    """
    from prospector import population
    return population.refresh_target_elements(dict(target_row), rc.mission.arrive_by)


def _assemble_flyby(fsol, earth, flyby_body, target, target_a_au: float,
                    scaled: ProgressFn, *, light: bool = False) -> tuple[dict, dict]:
    """``sf`` and ``orbits`` blocks for a :class:`prospector.solvers.flyby.FlybySolution`, in a
    direct leg's shape plus a ``flyby`` sub-block. Each leg is assembled on its own and joined
    (leg two's times offset by leg one's flight time); ``orbits`` gains ``flyby_au`` /
    ``flyby_track_au``."""
    from types import SimpleNamespace
    n1, n2 = fsol.leg_trajectories
    caps = fsol.seg_caps or (None, None)
    isps = fsol.seg_isp_s or (None, None)

    def leg(nodes, isp_s, ms, mf, seg_caps, seg_isp_s):
        tof = float(nodes[-1, 0] - nodes[0, 0])
        dv = isp_s * 9.80665 * np.log(ms / mf) / 1000.0 if mf > 0 else float("nan")
        return SimpleNamespace(
            feasible=fsol.feasible, mismatch=fsol.mismatch, dep_mjd2000=float(nodes[0, 0]),
            tof_days=tof, dv_kms=dv, initial_mass_kg=ms, final_mass_kg=mf, propellant_kg=ms - mf,
            nseg=(nodes.shape[0] - 1) // 2, positions_au=nodes[:, 1:4] / pk.AU,
            throttle=nodes[:, 8], trajectory=nodes, decision_vector=fsol.decision_vector,
            isp_s=isp_s, thrust_N=fsol.thrust_N, max_duty_cycle=fsol.max_duty_cycle,
            seg_caps=seg_caps, seg_isp_s=seg_isp_s, refresh_settled=fsol.refresh_settled)

    fb_a_au = float(np.linalg.norm(flyby_body.eph(float(fsol.flyby_mjd2000))[0]) / pk.AU)
    leg1 = leg(n1, fsol.isp_s[0], fsol.initial_mass_kg, fsol.flyby_mass_kg, caps[0], isps[0])
    leg2 = leg(n2, fsol.isp_s[1], fsol.flyby_mass_kg, fsol.final_mass_kg, caps[1], isps[1])
    b1, o1 = _assemble_leg(leg1, earth, flyby_body, 1.0, fb_a_au, scaled, light=light,
                           blue_body=earth, blue_a_au=1.0, orange_body=target,
                           orange_a_au=target_a_au)
    b2, o2 = _assemble_leg(leg2, flyby_body, target, fb_a_au, target_a_au, scaled, light=light,
                           blue_body=earth, blue_a_au=1.0, orange_body=target,
                           orange_a_au=target_a_au)

    def cat(key, offset=0.0):
        a, b = np.asarray(b1[key]), np.asarray(b2[key])
        if a.size == 0 and b.size == 0:
            return a
        return np.concatenate([a, b + offset]) if offset else np.concatenate([a, b])

    tof1 = float(fsol.tof1_days)
    block = dict(b1)
    block.update({
        "feasible": fsol.feasible, "mismatch": fsol.mismatch,
        "dep_mjd2000": float(fsol.dep_mjd2000), "tof_days": float(fsol.tof_days),
        "dv_kms": float(fsol.dv_kms), "initial_mass_kg": float(fsol.initial_mass_kg),
        "final_mass_kg": float(fsol.final_mass_kg), "propellant_kg": float(fsol.propellant_kg),
        "nseg": int(sum(fsol.nseg)),
        "positions_au": cat("positions_au"), "throttle": cat("throttle"),
        "node_thrust_vec": cat("node_thrust_vec"), "node_times_days": cat("node_times_days", tof1),
        "fine_positions_au": cat("fine_positions_au"), "fine_times_days": cat("fine_times_days", tof1),
        "fine_throttle": cat("fine_throttle"),
        "node_a_au": cat("node_a_au"), "node_e": cat("node_e"), "node_i_deg": cat("node_i_deg"),
        "node_thrust_radial": cat("node_thrust_radial"),
        "node_thrust_transverse": cat("node_thrust_transverse"),
        "node_thrust_normal": cat("node_thrust_normal"), "node_speed_kms": cat("node_speed_kms"),
        "vinf_arr_kms": b2["vinf_arr_kms"], "target_speed_arr_kms": b2["target_speed_arr_kms"],
        "seg_caps": (None if caps[0] is None else np.concatenate([np.asarray(c, float) for c in caps])),
        "seg_isp_s": (None if isps[0] is None else np.concatenate([np.asarray(c, float) for c in isps])),
        # The one Isp that reproduces the two legs' propellant total from the total mass ratio.
        "isp_s": float(fsol.dv_kms * 1000.0 / (9.80665 * np.log(fsol.initial_mass_kg / fsol.final_mass_kg)))
        if fsol.final_mass_kg > 0 and fsol.dv_kms == fsol.dv_kms else float(fsol.isp_s[0]),
        "trajectory": fsol.trajectory, "decision_vector": fsol.decision_vector,
        "flyby": {
            "body": fsol.flyby_name, "mjd2000": float(fsol.flyby_mjd2000),
            "tof1_days": tof1, "tof2_days": float(fsol.tof2_days),
            "mass_kg": float(fsol.flyby_mass_kg),
            "vinf_kms": float(fsol.flyby["vinf_kms"]), "turn_deg": float(fsol.flyby["turn_deg"]),
            "periapsis_alt_km": float(fsol.flyby["periapsis_alt_km"]),
            "min_periapsis_alt_km": float(fsol.flyby["min_periapsis_alt_km"]),
            "leg_isp_s": [float(v) for v in fsol.isp_s],
            "leg_propellant_kg": [float(fsol.initial_mass_kg - fsol.flyby_mass_kg),
                                  float(fsol.flyby_mass_kg - fsol.final_mass_kg)],
        },
    })
    orbits = {}
    if not light:
        abs_times = fsol.dep_mjd2000 + block["fine_times_days"]
        period = 365.25 * fb_a_au ** 1.5
        orbits = {"earth_au": o1["earth_au"],
                  "target_au": _sample_orbit(target, fsol.dep_mjd2000,
                                             365.25 * float(target_a_au) ** 1.5),
                  "earth_track_au": _track(earth, abs_times),
                  "target_track_au": _track(target, abs_times),
                  "flyby_au": _sample_orbit(flyby_body, fsol.dep_mjd2000, period),
                  "flyby_track_au": _track(flyby_body, abs_times),
                  "flyby_name": fsol.flyby_name}
    return block, orbits


def _assemble_leg(sol, depart_body, arrive_body, depart_a_au: float, arrive_a_au: float,
                  scaled: ProgressFn, *, blue_body=None, blue_a_au=None,
                  orange_body=None, orange_a_au=None, light: bool = False) -> tuple[dict, dict]:
    """Build the plot-ready ``sf`` and ``orbits`` blocks for one solved leg.

    ``light=True`` skips the parts that exist only to be drawn: the densified path, the body orbits
    and tracks. The design sweep assembles hundreds of legs and reads only the numbers, so it pays
    for none of that; the block keeps every key, with empty arrays where the plots would read, and
    ``orbits`` comes back empty.

    Used for the outbound trip and the return. ``depart_body``/``arrive_body`` are where the leg
    starts and ends, with ``depart_a_au``/``arrive_a_au`` their orbit sizes. They set the speeds:
    departure v-infinity is measured against the body being left, arrival against the one met.

    Colours stay tied to the bodies, not the direction of travel, through a separate blue/orange
    mapping: ``earth_au``/``earth_track_au`` are drawn blue and hold ``blue_body``, and
    ``target_au``/``target_track_au`` orange and hold ``orange_body``. They default to the
    departure and arrival bodies, which is right outbound. The return passes blue=Earth and
    orange=asteroid explicitly, so neither swaps colour between legs.
    """
    blue_body = blue_body if blue_body is not None else depart_body
    blue_a_au = blue_a_au if blue_a_au is not None else depart_a_au
    orange_body = orange_body if orange_body is not None else arrive_body
    orange_a_au = orange_a_au if orange_a_au is not None else arrive_a_au

    scaled("orbits", 0.95, "sampling orbits")
    veff_ms = sol.isp_s * 9.80665                  # exhaust speed (m/s), for re-flying segments
    # sol.thrust_N is the maximum the engines can do; the throttle stored along the trajectory is
    # a fraction of it, so rebuilding the path uses maximum thrust times that throttle.
    if light:
        fine_pos, fine_days, fine_thr = np.empty((0, 3)), np.empty(0), np.empty(0)
        blue_orbit = orange_orbit = blue_track = orange_track = np.empty((0, 3))
    else:
        fine_pos, fine_days, fine_thr = _densify(sol.trajectory, sol.thrust_N, veff_ms)
        blue_period = 365.25 * float(blue_a_au) ** 1.5
        orange_period = 365.25 * float(orange_a_au) ** 1.5
        blue_orbit = _sample_orbit(blue_body, sol.dep_mjd2000, blue_period)
        orange_orbit = _sample_orbit(orange_body, sol.dep_mjd2000, orange_period)
        # Where each body is over the course of the trip, sampled at the same times as the smooth
        # path, so the animation moves the two spheres in step with the spacecraft.
        abs_times = sol.dep_mjd2000 + fine_days
        blue_track = _track(blue_body, abs_times)
        orange_track = _track(orange_body, abs_times)
    node_a, node_e, node_i, node_radial, node_transverse, node_normal = \
        _node_diagnostics(sol.trajectory)
    node_days = sol.trajectory[:, 0] - sol.trajectory[0, 0]

    # Speed at each node, together with the speeds of the two bodies it has to match. Leaving, the
    # spacecraft is going faster than the body it departs from by the departure v-infinity;
    # arriving, it has to match the destination's velocity. Together these say how fast it left and
    # whether it caught what it was aiming at.
    arr_mjd2000 = sol.dep_mjd2000 + sol.tof_days
    v_dep_body = np.asarray(depart_body.eph(float(sol.dep_mjd2000))[1], float)
    v_arr_body = np.asarray(arrive_body.eph(float(arr_mjd2000))[1], float)
    node_speed = np.linalg.norm(sol.trajectory[:, 4:7], axis=1) / 1000.0
    vinf_dep_vec = sol.trajectory[0, 4:7] - v_dep_body
    vinf_dep = float(np.linalg.norm(vinf_dep_vec) / 1000.0)
    # How far out of the ecliptic the departure v-infinity points. On the outbound leg this is what
    # the launch has to deliver: it sets the plane the escape needs to reach, whether that comes
    # from the spiral or from the launch vehicle.
    vinf_dep_lat = (float(np.degrees(np.arcsin(np.clip(
        vinf_dep_vec[2] / np.linalg.norm(vinf_dep_vec), -1.0, 1.0))))
        if vinf_dep > 1e-9 else 0.0)
    # ...and which way round, so the pair give the full exit direction and the spiral plot can draw
    # where this solution leaves Earth.
    vinf_dep_lon = (float(np.degrees(np.arctan2(vinf_dep_vec[1], vinf_dep_vec[0])))
                    if vinf_dep > 1e-9 else 0.0)
    vinf_arr = float(np.linalg.norm(sol.trajectory[-1, 4:7] - v_arr_body) / 1000.0)
    dep_speed = float(np.linalg.norm(v_dep_body) / 1000.0)
    arr_speed = float(np.linalg.norm(v_arr_body) / 1000.0)

    sf_block = {
        "feasible": sol.feasible, "mismatch": sol.mismatch,
        "dep_mjd2000": sol.dep_mjd2000, "tof_days": sol.tof_days, "dv_kms": sol.dv_kms,
        "initial_mass_kg": sol.initial_mass_kg, "final_mass_kg": sol.final_mass_kg,
        "propellant_kg": sol.propellant_kg, "nseg": sol.nseg,
        "positions_au": sol.positions_au, "throttle": sol.throttle,
        "node_thrust_vec": sol.trajectory[:, 9:12],   # thrust direction x throttle, for the cones
        "node_times_days": node_days, "fine_positions_au": fine_pos,
        "fine_times_days": fine_days, "fine_throttle": fine_thr,
        "node_a_au": node_a, "node_e": node_e, "node_i_deg": node_i,
        "node_thrust_radial": node_radial, "node_thrust_transverse": node_transverse,
        "node_thrust_normal": node_normal,
        "node_speed_kms": node_speed, "vinf_dep_kms": vinf_dep, "vinf_arr_kms": vinf_arr,
        "vinf_dep_lat_deg": vinf_dep_lat, "vinf_dep_lon_deg": vinf_dep_lon,
        # The limit on how much of each segment the engine may fire for. The cap is what was asked
        # for; the average throttle is what the answer used.
        "max_duty_cycle": float(sol.max_duty_cycle),
        # Per-segment thrust limits set by distance from the Sun, as fractions of maximum thrust
        # (the same basis as throttle; None means a flat limit). The mission-profile plots draw
        # this as the varying ceiling.
        "seg_caps": (None if sol.seg_caps is None else np.asarray(sol.seg_caps, float)),
        # The Isp each segment ran at (its array power on the throttle curve); ``isp_s`` below is
        # the one leg-wide value the optimizer spent propellant at, their throttle-weighted mean.
        "seg_isp_s": (None if getattr(sol, "seg_isp_s", None) is None
                      else np.asarray(sol.seg_isp_s, float)),
        # What the engines ran at. On a power-limited cruise these come out lower than the
        # vehicle's rated thrust and Isp, so propellant figures and the throttle ceiling reflect
        # what the mission gets rather than the numbers on the datasheet.
        "thrust_N": float(sol.thrust_N), "isp_s": float(sol.isp_s),
        # Whether the thrust ceilings the leg flew under settled on its own path (see
        # SimsFlanaganSolution.refresh_settled); False reads as an optimistic cost.
        "refresh_settled": bool(getattr(sol, "refresh_settled", True)),

        "earth_speed_dep_kms": dep_speed, "target_speed_arr_kms": arr_speed,
        "trajectory": sol.trajectory, "decision_vector": sol.decision_vector,
    }
    orbits = ({} if light else
              {"earth_au": blue_orbit, "target_au": orange_orbit,
               "earth_track_au": blue_track, "target_track_au": orange_track})
    return sf_block, orbits
