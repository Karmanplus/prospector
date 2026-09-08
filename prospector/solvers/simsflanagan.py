"""
The real low-thrust solve: PyKEP's Sims-Flanagan model driven by pygmo.

Given a target orbit, a spacecraft (mass, thrust, Isp) and a launch window, this finds a trajectory
you could fly: departure date, flight time, throttle history, final mass, and from those the real
delta-v. It respects the limit on thrust acceleration that Lambert ignores, so it decides what is
reachable and what it costs.

``sf_pl2pl`` models the trip as ``nseg`` segments, each with a small bounded kick, that have to
meet in the middle. The variables are

    z = [t0, mf, Vinf_dep(3), Vinf_arr(3), throttles(3*nseg), tof]

It maximises final mass and supplies its own derivatives. There are many local answers, so pygmo's
Monotonic Basin Hopping drives repeated SLSQP descents from perturbed starts. A
``lambert.LambertSolution`` gives the first guess; one guess covers one part of the space, which is
why several run in parallel. See ``docs/physics.md``.

Delta-v in km/s, masses in kg, thrust in newtons, Isp in seconds, flight time in days, dates as
MJD2000. PyKEP works in SI internally.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pygmo as pg
import pykep as pk

from prospector.constants import EARTH_OBLIQUITY_DEG
from prospector.launch import max_departure_declination_deg
from prospector.solvers import lambert as lb
from prospector.solvers.sunpower import CHORD_WEIGHTS, SmoothCap, effective_isp, sun_power_model

G0 = pk.G0   # 9.80665 m/s^2

# The decision vector PyKEP's sf_pl2pl optimizes over:
#
#     z = [t0, mf, Vinf_dep(3), Vinf_arr(3), throttles(3*nseg), tof]
#
# Named rather than written inline because the flight time sits at the end, past a throttle
# block whose length varies: getting the arithmetic wrong would not raise, it would quietly
# constrain the wrong quantity.
_I_T0 = 0
_I_MF = 1
_I_VINF_DEP = slice(2, 5)
_I_VINF_ARR = slice(5, 8)
_I_THROTTLE0 = 8            # throttle i occupies _I_THROTTLE0 + 3i .. +3i+2

# Where the trip splits into its forward and backward halves. They meet in the middle, so each half
# covers the same number of segments and neither builds up more error than the other.
_MATCHPOINT_CUT = 0.5

# Lowest allowed final mass, as a fraction of the starting mass. It is there to scale the problem,
# not to express a design limit: at 3000 s it works out to a cruise delta-v of 68 km/s, ten times
# anything solved here, so it can never bite and can never reject a trip that works.
_MF_FLOOR_FRACTION = 0.1
# A floor under the mass the midpoint chain divides by, so an absurd Isp or a runaway throttle
# during a search produces a nonsense (rejected) point rather than a division by zero.
_MASS_FLOOR_KG = 1e-6

# How far apart the two halves may end up and still count as having met. The solver's own
# feasibility flag sits at a tighter 1e-4 that threaded gradients can jitter across, so the gap
# itself is the reliable signal and this is the threshold everything else uses. The gap is unitless
# (position over 1 AU, velocity over Earth's orbital speed, mass over the starting mass), so this
# is a loose test, around 1.5e5 km of position. A converged solve lands two or three orders of
# magnitude inside it.
MISMATCH_TOL = 1e-3
# How far a segment's throttle may sit above the ceiling its own path implies and still count as
# respecting it (a fraction of the leg's maximum thrust). The ceiling is a constraint inside the
# solve, so a converged answer sits within the solver's own tolerance of it; this is the looser
# test a rebuilt or stored answer is judged by. Two percent is well inside what a constant-thrust
# segment approximates anyway.
CAP_TOL = 0.02
# How far the leg's Isp may move between the value it was optimized at and the value its own path
# implies before the Isp pass re-solves for it (a fraction).
ISP_TOL = 0.002
# Earth's tilt. PyKEP works relative to Earth's orbit, but the departure directions launch can
# reach are limited relative to Earth's equator, since the spiral's plane has a fixed tilt to it.
# So the constraint rotates the departure velocity into the equatorial frame first.
_OBLIQUITY_RAD = math.radians(EARTH_OBLIQUITY_DEG)
_SIN_EPS, _COS_EPS = math.sin(_OBLIQUITY_RAD), math.cos(_OBLIQUITY_RAD)


# ---------------------------------------------------------------------------
# the problem: sf_pl2pl plus the constraints a real launch adds
# ---------------------------------------------------------------------------

class _LowThrustUDP:
    """A pygmo problem wrapping PyKEP's ``sf_pl2pl``.

    ``sf_pl2pl`` is the complete model between two :class:`pykep.planet` bodies: objective, bounds,
    the constraints that make the halves meet, and its own derivatives. This adds the three things
    a real launch brings: an arrival deadline, per-segment thrust limits from the array model, and
    a limit on which departure directions launch can reach.

    The thrust limit comes in two forms. ``seg_caps`` is one fixed ceiling per segment, used to
    rebuild a stored answer under exactly the ceilings it was found under. ``cap_fn`` is the
    ceiling as a function of distance from the Sun (a :class:`SmoothCap`), applied to each
    segment as the MEAN of the ceiling over six points along the chords between the segment's
    start, midpoint and end (``sunpower.CHORD_WEIGHTS``, the same points the exact ceiling is
    judged at afterwards): a segment's kick stands for the thrust integrated over the segment, so
    the impulse it may carry is what the array averages over that arc, not what it makes at one
    instant. That makes the ceiling part of the problem: as the optimizer moves the
    path sunward the ceiling rises with it, and the answer respects the power its own path has by
    construction, in one solve. Its derivative follows the nodes through the propagation chain
    (:meth:`_segment_nodes`), using the state-transition matrices the propagator returns, so
    nothing here is a finite difference.
    """

    def __init__(self, target, mass_kg, thrust_N, isp_s, nseg, t0_bounds, tof_bounds,
                 vinf_dep_kms, vinf_arr_kms, arrive_mjd2000=None, max_dep_decl_deg=None,
                 depart_body=None, seg_caps=None, cap_fn=None):
        # The departure body is Earth going out and the asteroid coming back. Both are passed in as
        # bodies, so there is never a placeholder to get wrong.
        self.base = pk.trajopt.sf_pl2pl(
            pls=lb.earth_planet() if depart_body is None else depart_body,
            plf=target,
            ms=float(mass_kg),
            mu=pk.MU_SUN,
            max_thrust=float(thrust_N),
            veff=float(isp_s) * G0,
            t0_bounds=[float(t0_bounds[0]), float(t0_bounds[1])],
            tof_bounds=[float(tof_bounds[0]), float(tof_bounds[1])],
            mf_bounds=[float(mass_kg) * _MF_FLOOR_FRACTION, float(mass_kg)],
            vinfs=float(vinf_dep_kms),
            vinff=float(vinf_arr_kms),
            nseg=int(nseg),
            cut=_MATCHPOINT_CUT,
            # The mass gap is reported as a fraction of the mass this leg starts with, so
            # MISMATCH_TOL means the same thing on a 200 kg smallsat and a 10 t stack.
            mass_scaling=float(mass_kg),
            with_gradient=True)
        # A hard deadline: arrival (t0 + tof) must not run past arrive_by. This lets the flight
        # time reach all the way to (arrive_by - earliest departure), so a long trip is possible by
        # leaving early, while still guaranteeing arrival inside the mission window.
        self.arrive_mjd2000 = arrive_mjd2000
        # Which departure directions launch can reach. The escape leaves in the plane of the
        # spiral, and a plane tilted by i contains no direction further than i from the equator
        # whatever the time of day. So how far out of the equator the departure points is launch's
        # to dictate, not the optimizer's to tune; where round the plane it points is
        # launch-targetable, so nothing more is needed. Switched off at 90 degrees or more, which
        # covers a launch-vehicle escape and near-polar planes.
        decl = (None if max_dep_decl_deg is None or float(max_dep_decl_deg) >= 90.0
                else float(max_dep_decl_deg))
        self.max_dep_decl_deg = decl
        self._sin2_decl = None if decl is None else math.sin(math.radians(decl)) ** 2
        self._vinf2_ms2 = max((float(vinf_dep_kms) * 1000.0) ** 2, 1.0)
        self._nseg = int(nseg)
        self._i_tof = 8 + 3 * int(nseg)
        # Per-segment throttle limits from the array model: further from the Sun the array makes
        # less power, so the engines cannot run at full thrust there. One limit per segment, as a
        # fraction of the thrust this problem flies, each becoming a smooth inequality
        # |u_i|^2 <= cap_i^2 alongside sf_pl2pl's own |u_i| <= 1. They are held fixed for a
        # round, with :func:`solve` refreshing them from the last converged trajectory, so the
        # objective stays cheap and smooth.
        self.seg_caps = None if seg_caps is None else np.clip(np.asarray(seg_caps, float),
                                                              0.0, 1.0)
        if self.seg_caps is not None and self.seg_caps.size != self._nseg:
            raise ValueError(f"seg_caps must have one entry per segment "
                             f"({self.seg_caps.size} != nseg {self._nseg})")
        if seg_caps is not None and cap_fn is not None:
            raise ValueError("pass fixed seg_caps or a cap_fn, not both")
        self.cap_fn = cap_fn
        self._nseg_fwd = int(self._nseg * _MATCHPOINT_CUT)
        self._cap_cols = ([np.asarray(self._cap_fn_columns(i), int) for i in range(self._nseg)]
                          if cap_fn is not None else [])
        self._cache_key = None
        self._cache = None
        # Whether the last evaluation found the ceiling flat (zero slope) at every sample, in which
        # case its gradient is the throttle term alone and the next evaluation can skip the
        # state-transition matrices. A leg starts on the full chain; a power-rich leg moves to the
        # light path at its first flat evaluation and stays there; one that ever meets a sloped
        # sample gives the light path up for good, since a leg that flips between the two would
        # walk the chain twice at every such point.
        self._flat = False
        self._light_ok = True

    # -- the segment midpoints, and how they move with the decision vector ----------------

    def _segment_nodes(self, x, jacobian: bool = True):
        """Each segment's start, midpoint and end positions (m) and their Jacobians against the
        decision vector.

        Returns ``(P, J)``: ``P`` is ``(nseg, 3, 3)`` (segment, node, xyz) and ``J`` is ``(nseg, 3,
        3, dim)``, or None when ``jacobian`` is False, which walks the same chain without the
        state-transition matrices at a fraction of the cost (see :meth:`_cap_fn_rows`). A forward
        segment's end and a backward segment's start are the states its own
        half of the chain reaches, so at the cut the two halves each carry their own. The chain is
        the one :func:`_leg_nodes` walks, forward from departure to the cut and backward from arrival,
        carrying alongside each state its sensitivity ``D = ds/dx`` (6 x dim) and the mass's
        ``dm/dx``. A Lagrangian propagation over a duration contributes its state-transition
        matrix on ``D`` plus the end state's time derivative times how the duration depends on the
        flight time; a kick adds ``c u / m`` to the velocity, whose derivative runs through the
        throttle, the flight time (the kick lasts a segment) and the mass; the mass update follows
        the rocket equation. Departure and arrival states move with ``t0`` and ``tof`` at the
        bodies' velocity and two-body acceleration, as PyKEP's own gradient does.

        The propagator itself costs a microsecond a call; what this loop spends is Python, so it
        is written for few array operations rather than for elegance.
        """
        nseg, base = self._nseg, self.base
        dim = len(x)
        i_tof = self._i_tof
        t0 = float(x[_I_T0])
        tof_days = float(x[i_tof])
        dt = tof_days * pk.DAY2SEC / nseg           # one segment, seconds
        d_dt = pk.DAY2SEC / nseg                    # d(dt) / d(tof_days)
        thrust = float(base.leg.max_thrust)
        c = thrust * dt                             # full-throttle kick x mass
        veff, mu = float(base.leg.veff), float(base.leg.mu)
        thr = np.asarray(x[_I_THROTTLE0:_I_THROTTLE0 + 3 * nseg], float).reshape(nseg, 3)
        P = np.zeros((nseg, 3, 3))
        J = np.zeros((nseg, 3, 3, dim)) if jacobian else None
        f = np.empty(6)                             # the state's time derivative, reused
        prop = pk.propagate_lagrangian

        if not jacobian:
            return self._segment_positions(x, P, thr, dt, c, veff, mu), None

        def step(s, D, tau, dtau_dtof):
            (r, v), stm = prop([s[0:3], s[3:6]], tof=tau, mu=mu, stm=True)
            s2 = np.array([*r, *v])
            D2 = np.asarray(stm, float) @ D
            rn = math.sqrt(s2[0] * s2[0] + s2[1] * s2[1] + s2[2] * s2[2])
            f[0:3] = s2[3:6]
            f[3:6] = s2[0:3] * (-mu / (rn * rn * rn))
            D2[:, i_tof] += f * dtau_dtof
            return s2, D2

        def kick(D, Dm, u, m, j, sign):
            """Apply a segment's kick (sign +1 forward, -1 backward) to ``D`` and return the
            rocket-equation exponent ``k`` and its gradient."""
            un = math.sqrt(float(u @ u) + 1e-18)
            cm = sign * c / m
            D[3, j] += cm
            D[4, j + 1] += cm
            D[5, j + 2] += cm
            D[3:6, i_tof] += (sign * thrust * d_dt / m) * u
            D[3:6] -= (sign * c / (m * m)) * u[:, None] * Dm[None, :]
            k = un * c / (m * veff)
            Dk = (-(un * c / (m * m * veff))) * Dm
            Dk[j:j + 3] += (c / (m * veff * un)) * u
            Dk[i_tof] += un * thrust * d_dt / (m * veff)
            return k, Dk

        # Forward from departure to the cut.
        rs, vs = base.pls.eph(t0)
        rs, vs = np.asarray(rs, float), np.asarray(vs, float)
        s = np.concatenate([rs, vs + np.asarray(x[_I_VINF_DEP], float)])
        D = np.zeros((6, dim))
        # The epoch is in days; the bodies move at their velocity and two-body acceleration.
        rn = float(np.linalg.norm(rs))
        D[0:3, _I_T0] = vs * pk.DAY2SEC
        D[3:6, _I_T0] = rs * (-mu / rn ** 3 * pk.DAY2SEC)
        D[3, 2] = D[4, 3] = D[5, 4] = 1.0
        m = max(float(base.leg.ms), _MASS_FLOOR_KG)
        Dm = np.zeros(dim)
        for i in range(self._nseg_fwd):
            P[i, 0], J[i, 0] = s[0:3], D[0:3]
            s, D = step(s, D, dt / 2, d_dt / 2)
            P[i, 1], J[i, 1] = s[0:3], D[0:3]
            u = thr[i]
            j = _I_THROTTLE0 + 3 * i
            k, Dk = kick(D, Dm, u, m, j, 1.0)
            s[3:6] += u * (c / m)
            ek = math.exp(-min(k, 700.0))
            Dm = ek * Dm - (m * ek) * Dk
            m = max(m * ek, _MASS_FLOOR_KG)       # an absurd Isp must not divide by zero
            s, D = step(s, D, dt / 2, d_dt / 2)
            P[i, 2], J[i, 2] = s[0:3], D[0:3]

        # Backward from arrival to the cut.
        rf, vf = base.plf.eph(t0 + tof_days)
        rf, vf = np.asarray(rf, float), np.asarray(vf, float)
        s = np.concatenate([rf, vf + np.asarray(x[_I_VINF_ARR], float)])
        D = np.zeros((6, dim))
        rn = float(np.linalg.norm(rf))
        f_arr = np.concatenate([vf, rf * (-mu / rn ** 3)]) * pk.DAY2SEC   # arrival epoch = t0 + tof
        D[:, _I_T0] = f_arr
        D[:, i_tof] = f_arr
        D[3, 5] = D[4, 6] = D[5, 7] = 1.0
        m = max(float(x[_I_MF]), _MASS_FLOOR_KG)
        Dm = np.zeros(dim)
        Dm[_I_MF] = 1.0
        for jj in range(nseg - self._nseg_fwd):
            seg = nseg - 1 - jj
            P[seg, 2], J[seg, 2] = s[0:3], D[0:3]
            s, D = step(s, D, -dt / 2, -d_dt / 2)
            P[seg, 1], J[seg, 1] = s[0:3], D[0:3]
            u = thr[seg]
            j = _I_THROTTLE0 + 3 * seg
            # Walking backward the kick is removed, sized on the mass after it (as _leg_nodes).
            k, Dk = kick(D, Dm, u, m, j, -1.0)
            s[3:6] -= u * (c / m)
            ek = math.exp(min(k, 700.0))
            Dm = ek * Dm + (m * ek) * Dk
            m = max(m * ek, _MASS_FLOOR_KG)
            s, D = step(s, D, -dt / 2, -d_dt / 2)
            P[seg, 0], J[seg, 0] = s[0:3], D[0:3]
        return P, J

    def _segment_positions(self, x, P, thr, dt, c, veff, mu):
        """The positions half of :meth:`_segment_nodes`: the same chain, no sensitivities."""
        nseg, base = self._nseg, self.base
        t0 = float(x[_I_T0])
        tof_days = float(x[self._i_tof])
        prop = pk.propagate_lagrangian

        def step(rv, tau):
            r, v = prop(rv, tof=tau, mu=mu, stm=False)
            return [list(r), list(v)]

        rs, vs = base.pls.eph(t0)
        rv = [list(rs), [a + b for a, b in zip(vs, x[_I_VINF_DEP])]]
        m = max(float(base.leg.ms), _MASS_FLOOR_KG)
        for i in range(self._nseg_fwd):
            P[i, 0] = rv[0]
            rv = step(rv, dt / 2)
            P[i, 1] = rv[0]
            u = thr[i]
            un = math.sqrt(float(u @ u) + 1e-18)
            rv[1] = [a + b * c / m for a, b in zip(rv[1], u)]
            m = max(m * math.exp(-min(un * c / (m * veff), 700.0)), _MASS_FLOOR_KG)
            rv = step(rv, dt / 2)
            P[i, 2] = rv[0]
        rf, vf = base.plf.eph(t0 + tof_days)
        rv = [list(rf), [a + b for a, b in zip(vf, x[_I_VINF_ARR])]]
        m = max(float(x[_I_MF]), _MASS_FLOOR_KG)
        for jj in range(nseg - self._nseg_fwd):
            seg = nseg - 1 - jj
            P[seg, 2] = rv[0]
            rv = step(rv, -dt / 2)
            P[seg, 1] = rv[0]
            u = thr[seg]
            un = math.sqrt(float(u @ u) + 1e-18)
            rv[1] = [a - b * c / m for a, b in zip(rv[1], u)]
            m = max(m * math.exp(min(un * c / (m * veff), 700.0)), _MASS_FLOOR_KG)
            rv = step(rv, -dt / 2)
            P[seg, 0] = rv[0]
        return P

    def segment_caps(self, x) -> np.ndarray | None:
        """The ceiling each segment is under at ``x``, as a fraction of the thrust this problem
        flies: the fixed ``seg_caps``, or ``cap_fn`` read at the midpoints. None without either."""
        if self.cap_fn is not None:
            P, _J = self._segment_nodes(x, jacobian=False)
            pts = CHORD_WEIGHTS @ P                                       # (n, S, 3)
            caps = np.asarray(self.cap_fn(np.linalg.norm(pts, axis=2) / pk.AU), float)
            return caps.mean(axis=1)
        return self.seg_caps

    def _cap_fn_rows(self, x, gradient: bool = True):
        """``|u_i|^2 - capbar_i(x)^2 <= 0`` per segment, ``capbar`` the mean ceiling over the
        segment's chord samples, and (with ``gradient``) the rows' gradient entries in the order
        :meth:`_cap_fn_columns` declares them, flat, else None. Cached on ``x``: pygmo asks for
        the fitness and the gradient several times at one point, and a fitness-only point (a line
        search, a perturbed retry) walks the chain without its sensitivities."""
        xa = np.asarray(x, float)
        key = xa.tobytes()
        if key == self._cache_key and (self._cache[1] is not None or not gradient):
            return self._cache
        # Where the ceiling is flat at every sample its gradient is the throttle term alone, and
        # the chain need only supply positions. Try the light path when the last point was flat;
        # fall back to the full chain the moment a sample has slope.
        light = not gradient or (self._light_ok and self._flat)
        P, J = self._segment_nodes(xa, jacobian=not light)               # (n,3,3), (n,3,3,dim)
        S = CHORD_WEIGHTS @ P                                           # (n,S,3) chord samples
        r_s = np.sqrt((S * S).sum(axis=2))                              # (n,S) sample distances
        cap_s, slope_s = self.cap_fn.with_derivative(r_s / pk.AU)
        flat = not np.any(slope_s)
        if not flat:
            self._light_ok = False
            if J is None and gradient:
                P, J = self._segment_nodes(xa, jacobian=True)
        self._flat = flat
        cap = cap_s.mean(axis=1)
        u = xa[_I_THROTTLE0:_I_THROTTLE0 + 3 * self._nseg].reshape(self._nseg, 3)
        vals = ((u * u).sum(axis=1) - cap * cap).tolist()
        if not gradient and not flat:
            self._cache_key, self._cache = key, (vals, None)
            return self._cache
        if flat:
            G = np.zeros((self._nseg, xa.size))
        else:
            # The chord samples are linear in the nodes, so their Jacobians are the same
            # combinations. d(-capbar^2)/dx = -2 capbar (1/N) sum_s cap'(r_s) dr_s/dx, with
            # dr_s/dx = (p_s . dp_s/dx)/r_s.
            n, ns = self._nseg, CHORD_WEIGHTS.shape[0]
            # (n,S,3,dim): the chord samples' Jacobians, then p_s . dp_s/dx summed over xyz, then
            # the weighted sum over the samples, all as matmuls.
            JS = (CHORD_WEIGHTS @ J.reshape(n, 3, 3 * xa.size)).reshape(n, ns, 3, xa.size)
            w = slope_s / (np.maximum(r_s, 1.0) * pk.AU)                # (n,S)
            pj = (S[:, :, :, None] * JS).sum(axis=2)                    # (n,S,dim): p_s . dp_s/dx
            G = (w[:, :, None] * pj).sum(axis=1) * (-2.0 * cap / ns)[:, None]
        idx = np.arange(self._nseg)[:, None] * 3 + _I_THROTTLE0 + np.arange(3)[None, :]
        G[np.arange(self._nseg)[:, None], idx] += 2.0 * u
        grad = np.concatenate([G[i, cols] for i, cols in enumerate(self._cap_cols)])
        self._cache_key, self._cache = key, (vals, grad)
        return vals, grad

    def _cap_fn_columns(self, i) -> list[int]:
        """Which variables segment ``i``'s midpoint can depend on, in increasing order: the
        departure epoch, the flight time, the departure (forward) or arrival (backward) excess
        velocity and final mass, and the throttles of the segments walked through to reach it
        plus its own (for the ``|u_i|^2`` term)."""
        if i < self._nseg_fwd:
            cols = [_I_T0, 2, 3, 4]
            cols += list(range(_I_THROTTLE0, _I_THROTTLE0 + 3 * (i + 1)))
        else:
            cols = [_I_T0, _I_MF, 5, 6, 7]
            cols += list(range(_I_THROTTLE0 + 3 * i, _I_THROTTLE0 + 3 * self._nseg))
        cols.append(self._i_tof)
        return cols

    # -- the added constraints, each with its own closed-form gradient ------------------

    def _deadline(self, x):
        """How far past the deadline the arrival is, in days: ``t0 + tof - arrive_by <= 0``."""
        return float(x[_I_T0] + x[self._i_tof] - self.arrive_mjd2000)

    def _seg_cap_violations(self, x):
        """``|u_i|^2 - cap_i^2 <= 0``, one per segment."""
        out = []
        for i, cap in enumerate(self.seg_caps):
            j = _I_THROTTLE0 + 3 * i
            ux, uy, uz = x[j], x[j + 1], x[j + 2]
            out.append(float(ux * ux + uy * uy + uz * uz - cap * cap))
        return out

    def _declination_violation(self, x):
        """The limit on departure direction, written as a smooth inequality.

        PyKEP works relative to Earth's orbit, but the limit is relative to Earth's equator, so
        rotate the departure velocity by Earth's tilt first and then cap it: ``|vz_eq| <=
        sin(decl_max)|v|``, written as ``vz_eq^2 - sin^2 |v|^2 <= 0`` and divided through by the
        departure speed so it scales like the other constraints.
        """
        vx, vy, vz = x[2], x[3], x[4]
        vz_eq = vy * _SIN_EPS + vz * _COS_EPS
        return float((vz_eq * vz_eq - self._sin2_decl * (vx * vx + vy * vy + vz * vz))
                     / self._vinf2_ms2)

    # -- the pygmo problem interface ----------------------------------------------------

    def fitness(self, x):
        f = list(self.base.fitness(x))
        if self.arrive_mjd2000 is not None:
            f.append(self._deadline(x))
        if self.seg_caps is not None:
            f.extend(self._seg_cap_violations(x))
        elif self.cap_fn is not None:
            f.extend(self._cap_fn_rows(x, gradient=False)[0])
        if self._sin2_decl is not None:
            f.append(self._declination_violation(x))
        return f

    def get_bounds(self):
        return self.base.get_bounds()

    def get_nobj(self):
        return 1

    def get_nec(self):
        return self.base.get_nec()

    def get_nic(self):
        return (self.base.get_nic() + (1 if self.arrive_mjd2000 is not None else 0)
                + (self._nseg if (self.seg_caps is not None or self.cap_fn is not None) else 0)
                + (1 if self._sin2_decl is not None else 0))

    def has_gradient(self):
        return True

    def _added_rows(self):
        """``(row, columns)`` for each added constraint, in the order pygmo expects.

        The added rows come after everything PyKEP produces, in increasing (row, column) order,
        which is the order a sparsity pattern and its values have to arrive in.
        """
        row = 1 + self.base.get_nec() + self.base.get_nic()
        rows = []
        if self.arrive_mjd2000 is not None:
            rows.append((row, [_I_T0, self._i_tof]))
            row += 1
        if self.seg_caps is not None:
            for i in range(self._nseg):
                j = _I_THROTTLE0 + 3 * i
                rows.append((row, [j, j + 1, j + 2]))
                row += 1
        elif self.cap_fn is not None:
            for cols in self._cap_cols:
                rows.append((row, cols.tolist()))
                row += 1
        if self._sin2_decl is not None:
            rows.append((row, [2, 3, 4]))
        return rows

    def gradient_sparsity(self):
        sp = [(int(i), int(j)) for i, j in self.base.gradient_sparsity()]
        for row, cols in self._added_rows():
            sp.extend((row, c) for c in cols)
        return sp

    def gradient(self, x):
        g = list(self.base.gradient(x))
        if self.arrive_mjd2000 is not None:
            g.extend((1.0, 1.0))                      # d(t0 + tof - deadline) / d(t0), d(tof)
        if self.seg_caps is not None:
            for i in range(self._nseg):
                j = _I_THROTTLE0 + 3 * i
                g.extend((2.0 * x[j], 2.0 * x[j + 1], 2.0 * x[j + 2]))
        elif self.cap_fn is not None:
            g.extend(self._cap_fn_rows(x)[1].tolist())
        if self._sin2_decl is not None:
            vx, vy, vz = x[2], x[3], x[4]
            vz_eq = vy * _SIN_EPS + vz * _COS_EPS
            k, s2 = 2.0 / self._vinf2_ms2, self._sin2_decl
            g.extend((k * (-s2 * vx),
                      k * (vz_eq * _SIN_EPS - s2 * vy),
                      k * (vz_eq * _COS_EPS - s2 * vz)))
        return g


def _leg_nodes(udp, x) -> np.ndarray:
    """The flown trajectory as one array of nodes, which is what everything downstream reads.

    Returns ``2*nseg + 1`` rows of ``[t (mjd2000), x, y, z (m), vx, vy, vz (m/s), m (kg), |u|, ux,
    uy, uz]``. Even rows are segment boundaries, odd rows the midpoints where the model puts each
    kick, so a midpoint carries the mass going into that kick. Both rows of a segment carry its
    throttle and the last row repeats the one before, giving the step function the model means.

    Built forward from the departure and backward from the arrival. Where they meet, any leftover
    gap shows as a small step; the solve reports its size.

    Worked out from the variables and the two bodies rather than read back out of PyKEP's own
    object, so nothing depends on state left by an earlier call.
    """
    nseg = udp._nseg
    base = udp.base
    t0 = float(x[_I_T0])
    tof_days = float(x[udp._i_tof])
    dt = tof_days * pk.DAY2SEC / nseg              # one segment, seconds
    c = base.leg.max_thrust * dt                   # full-throttle kick, before dividing by mass
    veff = base.leg.veff
    mu = base.leg.mu
    throttles = np.asarray(x[_I_THROTTLE0:_I_THROTTLE0 + 3 * nseg], float).reshape(nseg, 3)

    nseg_fwd = int(nseg * _MATCHPOINT_CUT)
    nseg_bck = nseg - nseg_fwd

    nodes = np.zeros((2 * nseg + 1, 12))
    nodes[:, 0] = t0 + np.arange(2 * nseg + 1) * (tof_days / (2 * nseg))

    rs, vs = base.pls.eph(t0)
    rv = [list(rs), [a + b for a, b in zip(vs, x[_I_VINF_DEP])]]
    mass = float(base.leg.ms)
    for i in range(nseg_fwd):
        u = throttles[i]
        nodes[2 * i, 1:4], nodes[2 * i, 4:7], nodes[2 * i, 7] = rv[0], rv[1], mass
        rv = list(pk.propagate_lagrangian(rv, tof=dt / 2, mu=mu, stm=False))
        nodes[2 * i + 1, 1:4], nodes[2 * i + 1, 4:7], nodes[2 * i + 1, 7] = rv[0], rv[1], mass
        dv = float(np.linalg.norm(u)) * c / mass
        rv[1] = [a + b * c / mass for a, b in zip(rv[1], u)]
        mass = max(mass * math.exp(-min(dv / veff, 700.0)), _MASS_FLOOR_KG)
        rv = list(pk.propagate_lagrangian(rv, tof=dt / 2, mu=mu, stm=False))

    rf, vf = base.plf.eph(t0 + tof_days)
    rv = [list(rf), [a + b for a, b in zip(vf, x[_I_VINF_ARR])]]
    mass = float(x[_I_MF])
    for j in range(nseg_bck):
        k = 2 * nseg - 2 * j                       # this segment's far node
        u = throttles[nseg - 1 - j]
        nodes[k, 1:4], nodes[k, 4:7], nodes[k, 7] = rv[0], rv[1], mass
        rv = list(pk.propagate_lagrangian(rv, tof=-dt / 2, mu=mu, stm=False))
        nodes[k - 1, 1:4], nodes[k - 1, 4:7], nodes[k - 1, 7] = rv[0], rv[1], mass
        dv = float(np.linalg.norm(u)) * c / mass
        rv[1] = [a - b * c / mass for a, b in zip(rv[1], u)]
        mass = max(mass * math.exp(min(dv / veff, 700.0)), _MASS_FLOOR_KG)
        rv = list(pk.propagate_lagrangian(rv, tof=-dt / 2, mu=mu, stm=False))
    # The backward pass finishes holding the state at the meeting point. Recording that one, rather
    # than the forward pass's, is what makes the gap visible as a step between the last forward
    # node and this one, instead of averaging the two halves into a smooth fiction.
    match = 2 * nseg_fwd
    nodes[match, 1:4], nodes[match, 4:7], nodes[match, 7] = rv[0], rv[1], mass

    for i in range(nseg):
        nodes[2 * i, 9:12] = throttles[i]
        nodes[2 * i + 1, 9:12] = throttles[i]
    nodes[2 * nseg, 9:12] = throttles[-1]
    nodes[:, 8] = np.linalg.norm(nodes[:, 9:12], axis=1)
    return nodes


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------

@dataclass
class SimsFlanaganSolution:
    """A solved trajectory, or the best attempt at one, with everything needed to plot it."""
    target_name: str
    feasible: bool
    mismatch: float                # how far apart the two halves ended up, unitless
    dep_mjd2000: float
    tof_days: float
    initial_mass_kg: float
    final_mass_kg: float
    dv_kms: float
    nseg: int
    isp_s: float
    thrust_N: float                # the engines' maximum thrust; throttle is a fraction of it
    max_duty_cycle: float          # how much of each segment the engine could fire for (0-1);
    #                                the throttle never exceeds it.
    # :func:`_leg_nodes` output: 2*nseg+1 rows of [t(mjd2000), x,y,z (m), vx,vy,vz (m/s), m (kg),
    # |u|, ux,uy,uz]. The throttle is a fraction of maximum thrust, so a thrust limit shows up as
    # a throttle that never goes above it.
    trajectory: np.ndarray
    decision_vector: np.ndarray = field(repr=False)
    # Per-segment thrust limits from the array model, as fractions of maximum thrust. That is the
    # same basis as ``throttle``, so the two can be drawn on top of each other and the throttle
    # never crosses its segment's limit. None means no array model was applied, leaving only the
    # flat ``max_duty_cycle`` limit.
    seg_caps: np.ndarray | None = None
    # The Isp each segment's operating point implies (the array power there, on the throttle
    # curve). The leg itself is optimized at one Isp, ``isp_s``, the throttle-weighted mean of
    # these, so the propellant total is right; these say where along the way it was spent at
    # what efficiency. None when no array model was applied.
    seg_isp_s: np.ndarray | None = None
    # One entry per Isp pass: the leg Isp before and after, and the final mass before and after
    # the pass. Diagnostics; None for a solve without an array model.
    refresh_log: list | None = None
    # Whether the answer's leg Isp agrees with the Isp its own path implies (within ISP_TOL). The
    # thrust ceilings are part of the problem, so a converged answer respects them by
    # construction; the Isp is the one term settled after the fact. False means the passes ran
    # out first, so the propellant total is priced at a slightly wrong Isp.
    refresh_settled: bool = True

    @property
    def propellant_kg(self) -> float:
        return self.initial_mass_kg - self.final_mass_kg

    @property
    def dep_date(self) -> datetime:
        return lb.date_from_mjd2000(self.dep_mjd2000)

    @property
    def arr_date(self) -> datetime:
        return lb.date_from_mjd2000(self.dep_mjd2000 + self.tof_days)

    @property
    def positions_au(self) -> np.ndarray:
        """Node positions as an (N, 3) array in AU, for the trajectory plots."""
        return self.trajectory[:, 1:4] / pk.AU

    @property
    def throttle(self) -> np.ndarray:
        """Throttle at each node, as a fraction of maximum thrust and never above the limit."""
        return self.trajectory[:, 8]


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------

def _make_udp(target, *, mass_kg, thrust_N, isp_s, nseg, launch_window, arrive_by,
              vinf_dep_kms, vinf_arr_kms, window_slack_days, min_tof_days,
              max_tof_days, max_duty_cycle, max_dep_decl_deg, depart_body=None,
              seg_caps=None, cap_fn=None) -> _LowThrustUDP:
    """The fully bounded problem for one solve. Shared by :func:`solve` and
    :func:`rebuild_solution`, so a stored solution rebuilds against the same bounds and
    constraints it was optimized under.
    """
    start, end = launch_window
    t0_lo = lb.mjd2000_from_date(start) - window_slack_days
    t0_hi = lb.mjd2000_from_date(end) + window_slack_days
    arrive_mjd = lb.mjd2000_from_date(arrive_by)
    # The flight time may run up to (arrive_by - earliest departure), since the longest trip is the
    # one that leaves as the window opens. The deadline constraint (t0 + tof <= arrive_by) then
    # keeps every trip inside the window, so a late departure simply cannot fly that long. If even
    # the earliest departure leaves less than min_tof_days before the deadline, the window is too
    # short. Say so, instead of quietly reporting a trip that arrives late.
    tof_ceiling = arrive_mjd - t0_lo
    tof_hi = tof_ceiling if max_tof_days is None else min(float(max_tof_days), tof_ceiling)
    if tof_hi <= min_tof_days:
        raise ValueError(
            f"mission window too short for a low-thrust transfer: the earliest departure leaves "
            f"only {tof_ceiling:.0f} days to arrive_by ({arrive_by}), below the "
            f"{min_tof_days:.0f}-day minimum flight time. Widen arrive_by or the launch window.")
    tof_bounds = (min_tof_days, tof_hi)

    # Limiting how much of a segment the engine may fire for limits the velocity change that
    # segment can produce, which at fixed Isp is the same as scaling the thrust the optimizer may
    # plan with, so it is applied that way. _build_solution scales the throttle back to a fraction
    # of maximum thrust, so the limit shows up as a ceiling on the chart.
    thrust_used = thrust_N * max_duty_cycle
    return _LowThrustUDP(target, mass_kg, thrust_used, isp_s, nseg,
                         (t0_lo, t0_hi), tof_bounds, vinf_dep_kms, vinf_arr_kms,
                         arrive_mjd2000=arrive_mjd, max_dep_decl_deg=max_dep_decl_deg,
                         depart_body=depart_body, seg_caps=seg_caps, cap_fn=cap_fn)


def solve(
    target,
    *,
    mass_kg: float,
    thrust_N: float,
    isp_s: float,
    launch_window: tuple[date, date],
    arrive_by: date,
    seed: lb.LambertSolution | None = None,
    x0=None,
    nseg: int = 15,
    maxeval: int = 1500,
    restarts: int = 3,
    vinf_dep_kms: float = 1.0,
    vinf_arr_kms: float = 0.1,
    window_slack_days: float = 0.0,
    min_tof_days: float = 120.0,
    max_tof_days: float | None = None,
    max_duty_cycle: float = 1.0,
    max_dep_decl_deg: float | None = None,
    depart_body=None,
    thrust_cap_fn=None,
    seg_isp_fn: Callable | None = None,
    isp_iters: int = 2,
    c_tol: float = 1e-4,
    rng_seed: int = 42,
    progress: Callable[[int, int], None] | None = None,
) -> SimsFlanaganSolution:
    """Optimize a low-thrust rendezvous from Earth to a target, and return the best found.

    It optimizes for final mass, so the most propellant-efficient rendezvous inside the time
    limits. It does not try to be quick, and given room it will spend more time to save propellant.
    Lowering ``max_tof_days`` makes it hurry at the cost of a thirstier trip.

    ``max_duty_cycle`` (0-1) is the most of any one segment the engine may fire for. Isp is fixed
    here, so full thrust for 80% of a segment and 80% thrust for all of it are the same velocity
    change and the same propellant; the operational reading is the useful one. It applies per
    segment, not to the mission average, so the engine can still fire hard where the work is. The
    default of 1.0 leaves it unrestricted, since restricting it rejects trajectories that work
    while enforcing a limit the answer was nowhere near: measured mission averages run 13-27%.

    The launch window bounds the departure date. By default the departure stays inside it and the
    flight time is capped so even the latest departure arrives by ``arrive_by``.
    ``window_slack_days`` widens that bound, since the real low-thrust window is broader than the
    two-burn one, at the cost of departing outside the nominal window.

    ``seed`` starts the first attempt at the cheapest porkchop cell. ``restarts`` is how many
    rounds to run, each an SLSQP descent plus a perturbed retry, keeping the best trajectory whose
    halves met. ``progress(done, total)`` is called after each round.

    ``vinf_arr_kms`` stays small for a rendezvous, but a little slack converges far better than a
    hard zero. The result carries a ``feasible`` flag and the ``mismatch`` between its halves, so a
    caller never quietly accepts a trajectory that did not close.

    ``max_dep_decl_deg`` limits how far from Earth's equator the departure may point. The escape
    leaves in the plane of the spiral, so this belongs to launch; without it the optimizer aims out
    of a plane the escape cannot reach. None, or 90 or more, leaves it free.

    ``thrust_cap_fn`` maps distance from the Sun (AU) to the thrust the array can power there, as a
    fraction of ``thrust_N``. It is applied INSIDE the solve, at each segment's midpoint, as a
    constraint the optimizer differentiates (a :class:`SmoothCap`; any other callable is smoothed
    into one), so the answer respects the power its own path has, in one solve, and the optimizer
    is free to move the path sunward to buy thrust. It multiplies with ``max_duty_cycle``. None
    leaves a single flat limit.

    ``seg_isp_fn`` maps a trajectory's node positions (metres, ``2*nseg+1`` rows) to the Isp each
    segment runs at. The leg is optimized at one Isp, so after the solve the throttle-weighted mean
    of those is compared with the Isp the leg used; if they differ by more than ``ISP_TOL`` the
    leg is re-solved from its own answer at the new Isp, up to ``isp_iters`` times. The Isp only
    moves a percent or two once the path is known, so one pass is the norm.
    """
    cap = None
    if thrust_cap_fn is not None:
        cap = (thrust_cap_fn if isinstance(thrust_cap_fn, SmoothCap)
               else SmoothCap.from_function(thrust_cap_fn))
    leg_isp = float(isp_s)
    seg_isps = None

    def make(isp):
        return _make_udp(target, mass_kg=mass_kg, thrust_N=thrust_N, isp_s=isp, nseg=nseg,
                         launch_window=launch_window, arrive_by=arrive_by,
                         vinf_dep_kms=vinf_dep_kms, vinf_arr_kms=vinf_arr_kms,
                         window_slack_days=window_slack_days, min_tof_days=min_tof_days,
                         max_tof_days=max_tof_days, max_duty_cycle=max_duty_cycle,
                         max_dep_decl_deg=max_dep_decl_deg, depart_body=depart_body, cap_fn=cap)

    udp = make(leg_isp)
    x_start = None
    if x0 is not None:
        # A solution from a neighbouring solve. It carries the expensive part, the throttle
        # history, which transfers across because a throttle is a fraction of its own segment
        # rather than an absolute number. Only the date and flight time are moved. It also brings
        # its own path, so the Isp it needs is read off that rather than off the 1 AU point.
        x_start = _adapt_decision_vector(x0, udp)
        if seg_isp_fn is not None:
            traj0 = _leg_nodes(udp, x_start)
            warm_isps = seg_isp_fn(traj0[:, 1:4])
            if warm_isps is not None:
                seg_isps = np.asarray(warm_isps, float)
                leg_isp = float(effective_isp(traj0[0:2 * int(nseg):2, 8], seg_isps, leg_isp))
                udp = make(leg_isp)
                x_start = _adapt_decision_vector(x0, udp)
    name = getattr(target, "name", "target")
    prob = pg.problem(udp)
    prob.c_tol = [c_tol] * prob.get_nc()

    local = pg.nlopt("slsqp")
    local.maxeval = int(maxeval)
    local.xtol_rel = 1e-8
    # Each round is one descent plus a perturbed retry. Looping here rather than inside the
    # algorithm is what lets progress be reported between rounds.
    #
    # The seed is fixed: pygmo seeds the perturbation randomly otherwise, so the same vehicle
    # solved twice would return a different trajectory. A re-run should reproduce. The design
    # search reseeds to 1337 to retry a blanket non-convergence, which only means anything if the
    # default is fixed.
    algo = pg.algorithm(pg.mbh(pg.algorithm(local), stop=1, perturb=0.05, seed=rng_seed))

    pop = pg.population(prob, size=1, seed=rng_seed)
    if x_start is not None:
        pop.set_x(0, x_start)
    elif seed is not None:
        pop.set_x(0, _seed_decision_vector(seed, udp, mass_kg, nseg))

    best_x = pop.champion_x
    best = _evaluate(prob, best_x)
    rounds = max(1, int(restarts))
    isp_rounds = int(isp_iters) if seg_isp_fn is not None else 0
    total_rounds = rounds + isp_rounds
    for k in range(rounds):
        pop = algo.evolve(pop)
        cand = _evaluate(prob, pop.champion_x)
        if _better(cand, best):
            best, best_x = cand, pop.champion_x
        if progress is not None:
            progress(k + 1, total_rounds)

    # The Isp pass. The leg was optimized at one Isp; its own path says what Isp each segment
    # really runs at. If the throttle-weighted mean disagrees with the Isp used, re-solve from the
    # answer at the new Isp: a plain local descent, since the answer is already feasible under
    # the ceilings and only the mass bookkeeping moved. The best converged answer is kept in
    # case a pass runs out of evaluations short of convergence.
    kept = (udp, best_x, best, leg_isp, seg_isps) if best[0] else None
    refresh_log: list = []
    settled = True
    for k in range(isp_rounds):
        traj = _leg_nodes(udp, best_x)
        new_isps = seg_isp_fn(traj[:, 1:4])
        if new_isps is None:
            break
        new_isps = np.asarray(new_isps, float)
        used = np.abs(traj[0:2 * int(nseg):2, 8])
        new_isp = float(effective_isp(used, new_isps, leg_isp))
        refresh_log.append({"round": k, "excess": 0.0, "isp_before": float(leg_isp),
                            "isp_after": new_isp, "mf_before": float(best[2])})
        seg_isps = new_isps
        if abs(new_isp - leg_isp) <= ISP_TOL * leg_isp:
            settled = True
            break
        settled = False
        leg_isp = new_isp
        udp = make(leg_isp)
        prob = pg.problem(udp)
        prob.c_tol = [c_tol] * prob.get_nc()
        pop = pg.population(prob, size=1, seed=rng_seed + 1 + k)
        pop.set_x(0, best_x)
        best = _evaluate(prob, best_x)
        pop = pg.algorithm(local).evolve(pop)
        cand = _evaluate(prob, pop.champion_x)
        if _better(cand, best):
            best, best_x = cand, pop.champion_x
        refresh_log[-1]["mf_after"] = float(best[2])
        if best[0]:
            kept = (udp, best_x, best, leg_isp, seg_isps)
        if progress is not None:
            progress(rounds + k + 1, total_rounds)
    if progress is not None and total_rounds > rounds:
        progress(total_rounds, total_rounds)

    restored = False
    if not best[0] and kept is not None:
        udp, best_x, best, leg_isp, seg_isps = kept
        restored = True
    if isp_rounds and (restored or not (settled and best[0])):
        # The passes ran out, or the answer returned is an earlier round's: judge THAT answer's
        # Isp against its own path.
        traj = _leg_nodes(udp, best_x)
        end_isps = seg_isp_fn(traj[:, 1:4])
        if end_isps is not None:
            end_isp = float(effective_isp(np.abs(traj[0:2 * int(nseg):2, 8]),
                                          np.asarray(end_isps, float), leg_isp))
            settled = abs(end_isp - leg_isp) <= ISP_TOL * leg_isp
    sol = _build_solution(udp, best_x, name, mass_kg, leg_isp, nseg, thrust_N, max_duty_cycle,
                          seg_isp_s=seg_isps)
    sol.refresh_log = refresh_log or None
    sol.refresh_settled = bool(settled)
    return sol


def _perihelion_au(body, mjd2000: float) -> float | None:
    """A body's perihelion distance (AU) from its state at ``mjd2000``; None if it cannot be
    read (a body with no ephemeris there, or an unbound orbit)."""
    try:
        r, v = (np.asarray(x, float) for x in body.eph(pk.epoch(float(mjd2000))))
    except Exception:  # noqa: BLE001  (any ephemeris failure: the default closest approach stands)
        return None
    # Perihelion from the state directly: q = a (1 - e), with a from the orbital energy and e from
    # the eccentricity vector. Written out rather than through the library's element conversion,
    # whose calling convention has moved between versions.
    mu = float(pk.MU_SUN)
    rn = float(np.linalg.norm(r))
    if rn <= 0.0:
        return None
    energy = float(v @ v) / 2.0 - mu / rn
    if energy >= 0.0:
        return None                                  # unbound: no perihelion to speak of
    a = -mu / (2.0 * energy)
    e_vec = np.cross(v, np.cross(r, v)) / mu - r / rn
    e = float(np.linalg.norm(e_vec))
    if not (a > 0.0 and 0.0 <= e < 1.0):
        return None
    return float(a * (1.0 - e) / pk.AU)


def _leg_power_terms(rc, bodies, available_power_W=None, array_model=None, *, when_mjd2000=None):
    """The thrust, Isp and per-segment limits a config's array implies for one leg.

    Returns ``{thrust_N, isp_s, thrust_cap_fn, seg_isp_fn}`` for :func:`solve` (see
    :mod:`prospector.solvers.sunpower` for the model). ``thrust_N`` is the most the leg can ever
    use: the operating point at the closest approach to the Sun the leg can make, taken as the
    smaller perihelion of the two ``bodies`` it flies between (never further in than 1 AU's worth
    of margin, never inside the model's floor). ``isp_s`` is the 1 AU operating point, the Isp the
    solve starts at before its own path says better. ``thrust_cap_fn`` is the ceiling against
    distance the solve applies inside the problem, and ``seg_isp_fn`` the Isp each segment of a
    path runs at. With no array model to apply, the rated point and no limits.
    """
    rated = {"thrust_N": rc.total_thrust_mN * 1e-3, "isp_s": rc.effective_isp,
             "thrust_cap_fn": None, "seg_isp_fn": None}
    model = sun_power_model(rc, available_power_W, array_model)
    if model is None:
        return rated
    epoch = float(when_mjd2000) if when_mjd2000 is not None else 0.0
    q = [_perihelion_au(b, epoch) for b in bodies]
    r_min = min((v for v in q if v is not None), default=None)
    thrust_N = model.leg_thrust_N(r_min)
    if thrust_N <= 0.0:
        # The array cannot run the stack even at the leg's closest approach. The leg keeps the
        # rated numbers as its basis and the ceiling is zero everywhere, so the solve fails to
        # converge and says so, rather than quietly flying a cruise on power it does not have.
        thrust_N = model.rated_thrust_N
        return {"thrust_N": thrust_N, "isp_s": model.rated_isp_s,
                "thrust_cap_fn": model.smooth_cap(0.0),
                "seg_isp_fn": lambda nodes: model.segment_isps(nodes, thrust_N)}
    _t1, isp1 = model.at_one_au()
    return {"thrust_N": thrust_N, "isp_s": isp1, "thrust_cap_fn": model.smooth_cap(thrust_N),
            "seg_isp_fn": lambda nodes: model.segment_isps(nodes, thrust_N)}


def _pop_power_kwargs(kwargs):
    """Take the power-model terms out of a solver kwargs dict: the at-thruster power at 1 AU, the
    array physics, and any stored leg terms (``thrust_N``, ``isp_s``, ``seg_isp_s``) a rebuild
    has to reproduce rather than derive."""
    return (kwargs.pop("available_power_W", None), kwargs.pop("array_model", None),
            {k: kwargs.pop(k) for k in ("thrust_N", "isp_s", "seg_isp_s") if k in kwargs})


def solve_for_config(rc, target, *, progress=None, **kwargs) -> SimsFlanaganSolution:
    """Solve using the vehicle and mission a :class:`ResolvedConfig` describes.

    Takes thrust and Isp from the resolved engine assembly. This solve picks up where the escape
    leaves off, so it starts from the mass the cruise begins with, which is the liftoff mass minus
    the spiral's propellant, and its departure bounds are the launch window shifted by however long
    the spiral takes. Both reduce to the plain wet mass and launch window when the launch vehicle
    provides the escape. The departure direction is limited to what the launch type can reach by
    default; pass ``max_dep_decl_deg`` to widen it when the spiral is allowed to steer its plane.

    In cruise the array is clear of the radiation belts, so its degradation holds at whatever the
    escape left it with (``available_power_W``, the at-thruster power at 1 AU; None means the
    start-of-life array), and only the distance from the Sun still moves its output. The engines
    follow that power along their throttle curve, up as well as down: more thrust and a higher
    Isp inside 1 AU, up to the rated point, less outside (:func:`_leg_power_terms`), and the
    ceiling that implies is part of the problem the optimizer solves. Pass ``thrust_cap_fn=None``
    to switch the power model off, or a function of sun distance to use one of your own at the 1 AU
    operating point; ``array_model`` selects particular array physics.
    """
    kwargs.setdefault("max_dep_decl_deg", max_departure_declination_deg(rc.launch))
    # The window and deadline come from the config but can be overridden, because a grid cell holds
    # the departure to one date (:func:`solve_cell_for_config`) and still has to reach the same
    # vehicle numbers through this function rather than working them out again.
    kwargs.setdefault("launch_window", rc.departure_window)
    kwargs.setdefault("arrive_by", rc.mission.arrive_by)
    available_power_W, array_model, _stored = _pop_power_kwargs(kwargs)
    terms = _leg_terms_for_call(rc, (lb.earth_planet(), target), available_power_W, array_model,
                                kwargs)
    return solve(
        target,
        mass_kg=rc.cruise_start_mass_kg,
        progress=progress,
        **terms,
        **kwargs,
    )


def _leg_terms_for_call(rc, bodies, available_power_W, array_model, kwargs) -> dict:
    """The ``thrust_N / isp_s / thrust_cap_fn / seg_isp_fn`` for a solve, honouring an explicit
    ``thrust_cap_fn`` in ``kwargs``: None switches the power model off (rated point, no limits);
    a function keeps the frozen 1 AU operating point and applies that function alone."""
    when = lb.mjd2000_from_date(kwargs["launch_window"][0])
    if "thrust_cap_fn" in kwargs:
        cap_fn = kwargs.pop("thrust_cap_fn")
        if cap_fn is None:
            return {"thrust_N": rc.total_thrust_mN * 1e-3, "isp_s": rc.effective_isp}
        model = sun_power_model(rc, available_power_W, array_model)
        t1, isp1 = (model.at_one_au() if model is not None
                    else (rc.total_thrust_mN * 1e-3, rc.effective_isp))
        return {"thrust_N": t1, "isp_s": isp1, "thrust_cap_fn": cap_fn}
    return _leg_power_terms(rc, bodies, available_power_W, array_model, when_mjd2000=when)


def solve_cell_for_config(rc, target, *, dep_mjd2000: float, tof_days: float,
                          x0=None, tol_days: float = 0.75, restarts: int = 1,
                          nseg: int = 12, **kwargs) -> SimsFlanaganSolution:
    """One grid cell: the same cruise solve as :func:`solve_for_config`, but with the departure
    date and the flight time held fixed instead of searched.

    Holding them fixed is what makes a grid affordable. Those two variables are where the free
    solve struggles most, because a trip can work at many departure timings and many flight times
    and the optimizer spends most of its restarts choosing between them. Fixing both leaves only
    the throttle history to find, which one restart usually manages. It is also what makes the
    answer a cell at all: the caller wants this departure at this duration, not the best one
    nearby.

    ``tol_days`` is how much slack the fixed values get. It is not zero, because a bound pinned to
    a single value is worse conditioned than a narrow one, and 0.75 days sits well inside a grid's
    own spacing.

    ``x0`` starts from a neighbouring cell's solution, which is what turns a grid from a set of
    independent solves into a sweep: the throttle history carries across and only the date and
    duration change. It has to come from a solve at the same ``nseg``; see
    :func:`_adapt_decision_vector` for why re-sampling one is a hazard.

    Starting from a neighbour inherits where the neighbour ended up as well as its throttles, and
    one restart is not always enough to get away from it. Measured on Apophis at a 400-day flight
    time, starting a cell from a neighbour ten days away: 3.2445 km/s at ``restarts=1`` against
    3.0194 from scratch, then 3.2059 at 2, 3.0187 at 4 and 2.9925 at 8. So four restarts recovers
    the from-scratch answer and eight beats it, for about 1.3 s. That is why ``restarts`` defaults
    low here, for a caller that just wants a quick does-it-work sweep; a grid quoting delta-v
    should pass 4 or more, since below that a cell reports its neighbour's answer rather than its
    own.

    A cell that does not converge reports ``feasible=False``, which means the search found no
    trajectory at this effort, not that the cell cannot be flown. Only the optimizer converging
    supports either claim, and a grid that showed the two the same way would reject reachable
    targets, which is the failure this solver exists to prevent.
    """
    half = max(0.05, float(tol_days))
    # The solver only takes bounds as a date plus a slack, so the requested date is rounded to a
    # whole day. A date with a fraction of a day on it cannot be expressed, and widening the slack
    # to cover the rounding pushes the bounds outside the launch window, which produces cells
    # departing before the pad is available. Callers that care, such as a grid, put their dates on
    # whole days a step inside each window edge, so the rounding changes nothing for them.
    dep = lb.date_from_mjd2000(round(float(dep_mjd2000))).date()
    # This function owns the fixed departure and duration, so the terms that define them come from
    # ``tol_days`` rather than from the caller. A caller passing on a shared set of solver settings
    # will be carrying its own window slack and flight-time bounds; letting those through would
    # either collide, raising a TypeError that a pooled caller is apt to swallow as "this cell did
    # not converge", or quietly unfix the cell it was asked to solve.
    for owned in ("launch_window", "window_slack_days", "min_tof_days", "max_tof_days"):
        kwargs.pop(owned, None)
    return solve_for_config(
        rc, target,
        launch_window=(dep, dep),
        window_slack_days=half,
        min_tof_days=max(1.0, float(tof_days) - half),
        max_tof_days=float(tof_days) + half,
        x0=x0, restarts=restarts, nseg=nseg, **kwargs)


def solve_return_for_config(rc, asteroid, *, start_mass_kg: float,
                            depart_window: tuple[date, date], arrive_by: date,
                            destination, progress=None, **kwargs) -> SimsFlanaganSolution:
    """Solve the loaded return leg: leave the asteroid and rendezvous near Earth.

    The mirror of :func:`solve_for_config` for the way home. The spacecraft leaves the asteroid
    travelling with it, so the departure speed relative to it is almost zero, and the asteroid is
    the departure body with Earth the arrival body. There is no launch escape to respect, since the
    spacecraft is already out in open space, so the departure direction is free. ``destination`` is
    a :class:`launch.ReturnDestination`; its ``arrival_vinf_kms`` is how much speed relative to
    Earth the arrival may still carry, with whatever it captures into absorbing the rest. The
    insertion burn is charged separately, not here.

    ``start_mass_kg`` is the mass after loading up: the outbound arrival mass plus what was
    collected. The propellant left for this leg is whatever the escape and the outbound cruise did
    not spend, which the caller checks against the solved figure.
    """
    earth = lb.earth_planet()
    kwargs.setdefault("max_dep_decl_deg", None)     # no launch direction limit on the way home
    kwargs.setdefault("vinf_dep_kms", 0.1)          # leave travelling with the asteroid
    kwargs["vinf_arr_kms"] = float(destination.arrival_vinf_kms)
    # The way home is also clear of the radiation belts, so it runs on the same array, following
    # the Sun the same way (:func:`_leg_power_terms`).
    available_power_W, array_model, _stored = _pop_power_kwargs(kwargs)
    kwargs["launch_window"] = depart_window
    terms = _leg_terms_for_call(rc, (asteroid, earth), available_power_W, array_model, kwargs)
    kwargs.pop("launch_window")
    return solve(
        earth,                                      # arriving at Earth
        mass_kg=float(start_mass_kg),
        launch_window=depart_window,
        arrive_by=arrive_by,
        depart_body=asteroid,                       # leaving from the asteroid
        progress=progress,
        **terms,
        **kwargs,
    )


def rebuild_solution(
    target,
    *,
    x,
    mass_kg: float,
    thrust_N: float,
    isp_s: float,
    launch_window: tuple[date, date],
    arrive_by: date,
    nseg: int = 15,
    vinf_dep_kms: float = 1.0,
    vinf_arr_kms: float = 0.1,
    window_slack_days: float = 0.0,
    min_tof_days: float = 120.0,
    max_tof_days: float | None = None,
    max_duty_cycle: float = 1.0,
    max_dep_decl_deg: float | None = None,
    depart_body=None,
    seg_caps=None,
    seg_isp_s=None,
    **_ignored,
) -> SimsFlanaganSolution:
    """Rebuild a trajectory from a stored solution vector, with no optimizing.

    Given the same problem, the stored vector completely determines the trajectory, so a solve
    found elsewhere, such as a sweep point or a saved run, rebuilds in milliseconds. The terms have
    to match the original solve's, which the shared :func:`_make_udp` takes care of. ``seg_caps``
    are the original solve's per-segment thrust limits (:attr:`SimsFlanaganSolution.seg_caps`, as
    fractions of maximum thrust); None rebuilds a problem with no such limits. Solver-only settings
    like ``maxeval`` and ``restarts`` are accepted and ignored, so a stored options dict can be
    passed straight through.
    """
    # Stored limits are fractions of maximum thrust, the same basis as the throttle; the problem
    # wants them as fractions of the thrust it flies, which is the inverse of the scaling
    # _build_solution applies.
    udp_caps = (None if seg_caps is None
                else np.asarray(seg_caps, float) / max(float(max_duty_cycle), 1e-12))
    udp = _make_udp(target, mass_kg=mass_kg, thrust_N=thrust_N, isp_s=isp_s, nseg=nseg,
                    launch_window=launch_window, arrive_by=arrive_by,
                    vinf_dep_kms=vinf_dep_kms, vinf_arr_kms=vinf_arr_kms,
                    window_slack_days=window_slack_days, min_tof_days=min_tof_days,
                    max_tof_days=max_tof_days, max_duty_cycle=max_duty_cycle,
                    max_dep_decl_deg=max_dep_decl_deg, depart_body=depart_body,
                    seg_caps=udp_caps)
    name = getattr(target, "name", "target")
    return _build_solution(udp, _checked_decision_vector(x, udp), name, mass_kg, isp_s,
                           int(nseg), thrust_N, max_duty_cycle,
                           seg_isp_s=None if seg_isp_s is None else np.asarray(seg_isp_s, float))


def _checked_decision_vector(x, udp) -> np.ndarray:
    """A stored solution vector, checked against this problem before anything is built from it.

    A rebuild takes the vector on trust, so one written with the variables in a different order
    builds a trajectory rather than refusing: the flight time gets read as a final mass, the final
    mass as a velocity, and out comes a plausible-looking path describing nothing. Checking against
    the bounds catches it, because a reordered vector puts values in slots that cannot hold them.
    """
    z = np.asarray(x, float)
    lo, hi = (np.asarray(b, float) for b in udp.get_bounds())
    if z.shape != lo.shape:
        raise ValueError(
            f"stored decision vector has {z.size} entries, but this problem has "
            f"{lo.size} ({udp._nseg} segments). It was written for a different problem.")
    outside = np.flatnonzero((z < lo - 1e-6) | (z > hi + 1e-6))
    if outside.size:
        i = int(outside[0])
        raise ValueError(
            f"stored decision vector is outside this problem's bounds at index {i} "
            f"({z[i]:.6g} not in [{lo[i]:.6g}, {hi[i]:.6g}]); {outside.size} of {z.size} "
            f"entries are. Either the terms differ from the solve that produced it, or it "
            f"was written against the older variable order "
            f"[t0, tof, mf, ...] rather than [t0, mf, ..., tof].")
    return z


def rebuild_for_config(rc, target, x, **kwargs) -> SimsFlanaganSolution:
    """:func:`rebuild_solution` with the vehicle and mission numbers a config implies.

    The exact mirror of :func:`solve_for_config`, including the default limit on departure
    direction and the post-escape power (``available_power_W``), so a stored options dict rebuilds
    against the same thrust and Isp the point was solved with."""
    kwargs.setdefault("max_dep_decl_deg", max_departure_declination_deg(rc.launch))
    # Overridable, as in solve_for_config: a vector stored from a fixed-date cell was optimized
    # inside that narrow range, and rebuilding it against the whole mission window rejects a
    # perfectly good vector whose date sits inside the range but outside a window edge.
    kwargs.setdefault("launch_window", rc.departure_window)
    kwargs.setdefault("arrive_by", rc.mission.arrive_by)
    thrust_N, isp_s, seg_isp_s = _rebuild_leg_terms(rc, (lb.earth_planet(), target), kwargs)
    return rebuild_solution(
        target,
        x=x,
        mass_kg=rc.cruise_start_mass_kg,
        thrust_N=thrust_N,
        isp_s=isp_s,
        seg_isp_s=seg_isp_s,
        **kwargs,
    )


def _rebuild_leg_terms(rc, bodies, kwargs) -> tuple[float, float, np.ndarray | None]:
    """The thrust and Isp a stored vector is rebuilt against: the stored leg terms when the point
    recorded them (``thrust_N``, ``isp_s``, ``seg_isp_s`` in ``kwargs``; the Isp a solve settles
    on depends on its trajectory, so it has to travel with the vector), else the terms a fresh
    solve would start from."""
    available_power_W, array_model, stored = _pop_power_kwargs(kwargs)
    kwargs.pop("thrust_cap_fn", None)
    kwargs.pop("seg_isp_fn", None)
    when = lb.mjd2000_from_date(kwargs["launch_window"][0])
    terms = _leg_power_terms(rc, bodies, available_power_W, array_model, when_mjd2000=when)
    thrust_N = float(stored.get("thrust_N", terms["thrust_N"]))
    isp_s = float(stored.get("isp_s", terms["isp_s"]))
    seg_isp_s = stored.get("seg_isp_s")
    return thrust_N, isp_s, (None if seg_isp_s is None else np.asarray(seg_isp_s, float))


def rebuild_return_for_config(rc, asteroid, x, *, start_mass_kg: float,
                              depart_window: tuple[date, date], arrive_by: date,
                              destination, **kwargs) -> SimsFlanaganSolution:
    """:func:`rebuild_solution` for the return leg: the mirror of
    :func:`solve_return_for_config`, rebuilding a swept return point in milliseconds."""
    earth = lb.earth_planet()
    kwargs.setdefault("max_dep_decl_deg", None)
    kwargs.setdefault("vinf_dep_kms", 0.1)
    kwargs["vinf_arr_kms"] = float(destination.arrival_vinf_kms)
    kwargs["launch_window"] = depart_window
    thrust_N, isp_s, seg_isp_s = _rebuild_leg_terms(rc, (asteroid, earth), kwargs)
    kwargs.pop("launch_window")
    return rebuild_solution(
        earth,
        x=x,
        mass_kg=float(start_mass_kg),
        thrust_N=thrust_N,
        isp_s=isp_s,
        seg_isp_s=seg_isp_s,
        launch_window=depart_window,
        arrive_by=arrive_by,
        depart_body=asteroid,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _seed_decision_vector(seed, udp, mass_kg, nseg) -> np.ndarray:
    """Build a starting vector from a Lambert solution: coasting, with the engine off."""
    lo, hi = udp.get_bounds()
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    vinf_dep = seed.v_transfer_dep_ms - seed.v_dep_body_ms      # m/s
    z = np.zeros(len(lo))
    z[_I_T0] = np.clip(seed.dep_mjd2000, lo[_I_T0], hi[_I_T0])
    z[udp._i_tof] = np.clip(seed.tof_days, lo[udp._i_tof], hi[udp._i_tof])
    z[_I_MF] = mass_kg * 0.9                                    # a guess at the final mass
    z[_I_VINF_DEP] = np.clip(vinf_dep, lo[_I_VINF_DEP], hi[_I_VINF_DEP])
    if udp.max_dep_decl_deg is not None:
        # A two-burn guess may point somewhere launch cannot reach, so bring it back to the edge of
        # what it can. The descent then starts from a legal point while the guess's dates and
        # in-plane energy survive.
        vx, vy_eq, vz_eq = (z[2], z[3] * _COS_EPS - z[4] * _SIN_EPS,
                            z[3] * _SIN_EPS + z[4] * _COS_EPS)
        z_max = math.tan(math.radians(udp.max_dep_decl_deg)) * math.hypot(vx, vy_eq)
        vz_eq = float(np.clip(vz_eq, -z_max, z_max))
        z[3] = vy_eq * _COS_EPS + vz_eq * _SIN_EPS      # equatorial -> ecliptic
        z[4] = -vy_eq * _SIN_EPS + vz_eq * _COS_EPS
    # Arrival speed and throttles start at zero, which makes it a pure coasting guess.
    return np.clip(z, lo, hi)


def _adapt_decision_vector(x, udp) -> np.ndarray:
    """One solve's answer, made into a legal starting point for another.

    Only the bounds are applied: the date and flight time are clipped into this problem's window,
    and everything else carries across untouched, the throttles above all. That is the value of
    starting from a neighbour: the throttle history is what the optimizer spent its time finding,
    and a throttle is a fraction of its own segment rather than an absolute number.

    The segment count has to match. Re-sampling a throttle history to a different number of
    segments is what turns a warm start from an advantage into a hazard: the re-sampled point
    satisfies the coarser problem's constraints but not the finer one's, so the optimizer settles
    there and reports a plausible delta-v whose two halves never met. Measured on a 2008 EV5 grid,
    starting a 12-segment solve from an 8-segment one came out worse than starting from scratch in
    every trial; starting it from another 12-segment solve came out worse in none. So a mismatch
    raises rather than interpolating.
    """
    z = np.asarray(x, float)
    lo, hi = (np.asarray(b, float) for b in udp.get_bounds())
    if z.shape != lo.shape:
        raise ValueError(
            f"cannot warm-start from a {z.size}-entry decision vector: this problem has "
            f"{lo.size} ({udp._nseg} segments). Solve the grid at the segment count the refine "
            f"will use rather than resampling the control history between them.")
    return np.clip(z, lo, hi)


def _fold_throttles_to_caps(x, caps) -> np.ndarray:
    """A copy of ``x`` with each segment's throttle scaled back inside its limit, so a start
    under new per-segment limits begins legal while keeping the previous answer's dates, speeds
    and thrust directions."""
    z = np.asarray(x, float).copy()
    for i, cap in enumerate(np.asarray(caps, float)):
        lo = _I_THROTTLE0 + 3 * i
        u = z[lo:lo + 3]
        n = float(np.linalg.norm(u))
        if n > cap:
            z[lo:lo + 3] = u * (cap * (1.0 - 1e-9) / n) if n > 0 else 0.0
    return z


def _evaluate(prob, x):
    """Return (feasible, worst gap between the halves, final mass) for a solution vector."""
    f = np.asarray(prob.fitness(x), float)
    nec = prob.get_nec()
    mismatch = float(np.max(np.abs(f[1:1 + nec]))) if nec else 0.0
    return (bool(prob.feasibility_x(x)), mismatch, float(x[_I_MF]))


def _better(cand, best) -> bool:
    """A solution that converged wins; otherwise more final mass, then a smaller gap."""
    cand_feas, cand_mis, cand_mf = cand
    best_feas, best_mis, best_mf = best
    if cand_feas != best_feas:
        return cand_feas
    if cand_feas:
        return cand_mf > best_mf
    return cand_mis < best_mis


def _build_solution(udp, x, name, mass_kg, isp_s, nseg, thrust_N, max_duty_cycle,
                    seg_isp_s=None) -> SimsFlanaganSolution:
    prob = pg.problem(udp)
    prob.c_tol = [1e-4] * prob.get_nc()
    feasible, mismatch, mf = _evaluate(prob, x)
    traj = _leg_nodes(udp, x)
    # Inside the problem, throttle is a fraction of the limited thrust. Scale the reported throttle
    # back to a fraction of maximum thrust, so a thrust limit reads as a ceiling on the chart, and
    # maximum thrust times reported throttle comes back out right.
    traj[:, 8:12] *= max_duty_cycle
    dv = isp_s * G0 * np.log(mass_kg / mf) / 1000.0 if mf > 0 else float("nan")
    # The problem's limits are fractions of the thrust it flies; report them on the same basis as
    # the throttle, fractions of maximum thrust, so the two can be drawn together. With a ceiling
    # against distance, these are the ceilings the answer's own midpoints sit under.
    caps = udp.segment_caps(x)
    seg_caps = None if caps is None else np.asarray(caps, float) * float(max_duty_cycle)
    return SimsFlanaganSolution(
        target_name=name, feasible=feasible, mismatch=mismatch,
        dep_mjd2000=float(x[_I_T0]), tof_days=float(x[udp._i_tof]),
        initial_mass_kg=float(mass_kg),
        final_mass_kg=mf, dv_kms=dv, nseg=int(nseg), isp_s=float(isp_s),
        thrust_N=float(thrust_N), max_duty_cycle=float(max_duty_cycle),
        trajectory=traj, decision_vector=np.asarray(x, float), seg_caps=seg_caps,
        seg_isp_s=seg_isp_s)
