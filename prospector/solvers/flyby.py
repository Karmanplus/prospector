"""A low-thrust transfer with one gravity assist: two Sims-Flanagan legs joined at a planet.

Both legs, the flyby date and mass, and the in/out velocities relative to the flyby body are
optimized together for final mass. The flyby is patched-conic (``pykep.fb_con``): equal relative
speed in and out, and a turn no larger than the minimum periapsis allows at that speed. Each leg
is a :class:`pykep.leg.sims_flanagan`; the ephemeris chain, flyby derivatives and the
:class:`SmoothCap` ceiling (both legs, each with its own Isp) are written out here. Dense gradient.

Decision vector (S.I. except the three times, in days)::

    z = [t0, tof1, tof2, m_fb, mf, v_dep(3), v_in(3), v_out(3), v_arr(3), u1(3*nseg1), u2(3*nseg2)]
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pygmo as pg
import pykep as pk

from prospector.solvers import lambert as lb
from prospector.solvers.simsflanagan import (
    _COS_EPS,
    _MASS_FLOOR_KG,
    _MATCHPOINT_CUT,
    _MF_FLOOR_FRACTION,
    _SIN_EPS,
    G0,
    ISP_TOL,
)
from prospector.solvers.sunpower import CHORD_WEIGHTS, SmoothCap, effective_isp

_T0, _TOF1, _TOF2, _MFB, _MF = 0, 1, 2, 3, 4
_VDEP, _VIN, _VOUT, _VARR = slice(5, 8), slice(8, 11), slice(11, 14), slice(14, 17)
_HEAD = 17

# The speed relative to the flyby body is bounded only so the problem is boxed.
_VINF_FB_MAX_KMS = 25.0
# Match-point mismatch (scaled) above which a start with its flyby split held is not worth freeing.
_DEAD_MISMATCH = 0.05


def _body_term(mcg_x: np.ndarray, r, v, mu: float) -> np.ndarray:
    """d(mismatch)/d(epoch in seconds) through one body's state, the body moving at (v, a) with
    a = -mu r / |r|^3."""
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    acc = -mu * r / float(np.linalg.norm(r)) ** 3
    return mcg_x[:, :3] @ v + mcg_x[:, 3:6] @ acc


@dataclass(frozen=True)
class _LegSpec:
    """Where one leg's variables sit in the joint vector."""
    start_cols: tuple[int, ...]     # the start epoch is the sum of these (days)
    end_cols: tuple[int, ...]       # the end epoch is the sum of these
    tof_col: int
    ms_col: int | None              # None: the start mass is the fixed launch mass
    mf_col: int
    vs: slice                       # relative velocity at the start
    vf: slice                       # relative velocity at the end
    u: slice
    nseg: int


class _FlybyUDP:
    """The pygmo problem: two legs, one flyby, objective final mass."""

    def __init__(self, depart_body, flyby_body, target, mass_kg, thrust_N, isp_s, nseg1, nseg2,
                 t0_bounds, tof1_bounds, tof2_bounds, vinf_dep_kms, vinf_arr_kms,
                 arrive_mjd2000=None, max_dep_decl_deg=None, min_flyby_alt_km=None,
                 flyby_mu=None, cap_fn: SmoothCap | None = None, tof_total_bounds=None,
                 vinf_dep_exact=False):
        self.bodies = (depart_body, flyby_body, target)
        self.mass_kg = float(mass_kg)
        self.n1, self.n2 = int(nseg1), int(nseg2)
        self.isp_s = tuple(float(v) for v in (isp_s if isinstance(isp_s, (tuple, list))
                                              else (isp_s, isp_s)))
        self.legs = []
        for n, isp in zip((self.n1, self.n2), self.isp_s):
            leg = pk.leg.sims_flanagan()
            leg.max_thrust = float(thrust_N)
            leg.veff = isp * G0
            leg.mu = pk.MU_SUN
            leg.cut = _MATCHPOINT_CUT
            leg.throttles = [0.0] * (3 * n)
            self.legs.append(leg)
        self.t0_bounds = tuple(float(b) for b in t0_bounds)
        self.tof1_bounds = tuple(float(b) for b in tof1_bounds)
        self.tof2_bounds = tuple(float(b) for b in tof2_bounds)
        self.vinf_dep = float(vinf_dep_kms) * 1000.0
        self.vinf_arr = float(vinf_arr_kms) * 1000.0
        self.vinf_dep_exact = bool(vinf_dep_exact)     # launcher speed held, not a ceiling
        self.arrive_mjd2000 = None if arrive_mjd2000 is None else float(arrive_mjd2000)
        # Bounds on tof1 + tof2 (days): a grid cell holds the whole trip to its flight time.
        self.tof_total_bounds = (None if tof_total_bounds is None
                                 else tuple(float(b) for b in tof_total_bounds))
        decl = (None if max_dep_decl_deg is None or float(max_dep_decl_deg) >= 90.0
                else float(max_dep_decl_deg))
        self.max_dep_decl_deg = decl
        self._sin2_decl = None if decl is None else math.sin(math.radians(decl)) ** 2
        self._vinf2 = max(self.vinf_dep ** 2, 1.0)
        # Periapsis floor: safe radius unless an altitude is given. mu is overridable (a test sets
        # it near zero, which forbids any turn).
        self.fb_mu = float(flyby_body.get_mu_self() if flyby_mu is None else flyby_mu)
        self.fb_rp = float(flyby_body.get_safe_radius() if min_flyby_alt_km is None
                           else flyby_body.get_radius() + float(min_flyby_alt_km) * 1000.0)
        self.r_scaling = pk.AU
        self.v_scaling = pk.EARTH_VELOCITY
        self.m_scaling = self.mass_kg
        self.dim = _HEAD + 3 * (self.n1 + self.n2)
        u1 = slice(_HEAD, _HEAD + 3 * self.n1)
        u2 = slice(_HEAD + 3 * self.n1, self.dim)
        self._u1, self._u2 = u1, u2
        self.specs = (
            _LegSpec((_T0,), (_T0, _TOF1), _TOF1, None, _MFB, _VDEP, _VIN, u1, self.n1),
            _LegSpec((_T0, _TOF1), (_T0, _TOF1, _TOF2), _TOF2, _MFB, _MF, _VOUT, _VARR, u2, self.n2),
        )
        self.cap_fn = cap_fn
        self._cache_key = None
        self._cache = None
        self._flat = False
        self._light_ok = True

    # -- leg states from the decision vector --------------------------------------------

    def _epochs(self, x):
        t0 = float(x[_T0])
        t_fb = t0 + float(x[_TOF1])
        return t0, t_fb, t_fb + float(x[_TOF2])

    def _states(self, x):
        t0, t_fb, t_arr = self._epochs(x)
        dep, fb, tgt = self.bodies
        return tuple(tuple(np.asarray(a, float) for a in body.eph(t))
                     for body, t in ((dep, t0), (fb, t_fb), (tgt, t_arr)))

    def _set_legs(self, x):
        """Load both legs from ``x``; returns the three body states (r, v), start to end."""
        (r0, v0), (r1, v1), (r2, v2) = states = self._states(x)
        l1, l2 = self.legs
        l1.ms = self.mass_kg
        l1.mf = float(x[_MFB])
        l1.rvs = [list(r0), list(v0 + x[_VDEP])]
        l1.rvf = [list(r1), list(v1 + x[_VIN])]
        l1.tof = float(x[_TOF1]) * pk.DAY2SEC
        l1.throttles = list(x[self._u1])
        l2.ms = float(x[_MFB])
        l2.mf = float(x[_MF])
        l2.rvs = [list(r1), list(v1 + x[_VOUT])]
        l2.rvf = [list(r2), list(v2 + x[_VARR])]
        l2.tof = float(x[_TOF2]) * pk.DAY2SEC
        l2.throttles = list(x[self._u2])
        return states

    def _scale_rows(self, rows: np.ndarray) -> np.ndarray:
        s = np.array([self.r_scaling] * 3 + [self.v_scaling] * 3 + [self.m_scaling])
        return rows / s

    # -- the segment chain of one leg, with sensitivities ------------------------------------

    def _chain(self, x, k: int, jacobian: bool):
        """Leg ``k``'s segment node positions ``P`` (nseg, 3, 3) and Jacobians ``J``
        (nseg, 3, 3, dim) against the joint vector, or None without ``jacobian``. The walk is
        :func:`simsflanagan._LowThrustUDP._segment_nodes` on the leg's own columns; an epoch is a
        sum of day columns, so the body's velocity and acceleration enter each of them."""
        spec, leg = self.specs[k], self.legs[k]
        (r0, v0), (r1, v1), (r2, v2) = self._states(x)
        rs, vs = ((r0, v0), (r1, v1))[k]
        rf, vf = ((r1, v1), (r2, v2))[k]
        nseg, dim = spec.nseg, self.dim
        tof_days = float(x[spec.tof_col])
        dt = tof_days * pk.DAY2SEC / nseg
        d_dt = pk.DAY2SEC / nseg
        thrust = float(leg.max_thrust)
        c = thrust * dt
        veff, mu = float(leg.veff), float(leg.mu)
        thr = np.asarray(x[spec.u], float).reshape(nseg, 3)
        nseg_fwd = int(nseg * _MATCHPOINT_CUT)
        i_tof = spec.tof_col
        u0 = spec.u.start
        P = np.zeros((nseg, 3, 3))
        prop = pk.propagate_lagrangian
        ms = self.mass_kg if spec.ms_col is None else float(x[spec.ms_col])
        mf = float(x[spec.mf_col])

        if not jacobian:
            rv = [list(rs), list(vs + x[spec.vs])]
            m = max(ms, _MASS_FLOOR_KG)
            for i in range(nseg_fwd):
                P[i, 0] = rv[0]
                r, v = prop(rv, tof=dt / 2, mu=mu, stm=False)
                rv = [list(r), list(v)]
                P[i, 1] = rv[0]
                u = thr[i]
                un = math.sqrt(float(u @ u) + 1e-18)
                rv[1] = [a + b * c / m for a, b in zip(rv[1], u)]
                m = max(m * math.exp(-min(un * c / (m * veff), 700.0)), _MASS_FLOOR_KG)
                r, v = prop(rv, tof=dt / 2, mu=mu, stm=False)
                rv = [list(r), list(v)]
                P[i, 2] = rv[0]
            rv = [list(rf), list(vf + x[spec.vf])]
            m = max(mf, _MASS_FLOOR_KG)
            for jj in range(nseg - nseg_fwd):
                seg = nseg - 1 - jj
                P[seg, 2] = rv[0]
                r, v = prop(rv, tof=-dt / 2, mu=mu, stm=False)
                rv = [list(r), list(v)]
                P[seg, 1] = rv[0]
                u = thr[seg]
                un = math.sqrt(float(u @ u) + 1e-18)
                rv[1] = [a - b * c / m for a, b in zip(rv[1], u)]
                m = max(m * math.exp(min(un * c / (m * veff), 700.0)), _MASS_FLOOR_KG)
                r, v = prop(rv, tof=-dt / 2, mu=mu, stm=False)
                rv = [list(r), list(v)]
                P[seg, 0] = rv[0]
            return P, None

        J = np.zeros((nseg, 3, 3, dim))
        f = np.empty(6)

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
            un = math.sqrt(float(u @ u) + 1e-18)
            cm = sign * c / m
            D[3, j] += cm
            D[4, j + 1] += cm
            D[5, j + 2] += cm
            D[3:6, i_tof] += (sign * thrust * d_dt / m) * u
            D[3:6] -= (sign * c / (m * m)) * u[:, None] * Dm[None, :]
            kk = un * c / (m * veff)
            Dk = (-(un * c / (m * m * veff))) * Dm
            Dk[j:j + 3] += (c / (m * veff * un)) * u
            Dk[i_tof] += un * thrust * d_dt / (m * veff)
            return kk, Dk

        def epoch_rows(D, r, v, cols):
            rn = float(np.linalg.norm(r))
            f_t = np.concatenate([v, r * (-mu / rn ** 3)]) * pk.DAY2SEC
            for col in cols:
                D[:, col] += f_t

        s = np.concatenate([rs, vs + np.asarray(x[spec.vs], float)])
        D = np.zeros((6, dim))
        epoch_rows(D, rs, vs, spec.start_cols)
        D[3, spec.vs.start] = D[4, spec.vs.start + 1] = D[5, spec.vs.start + 2] = 1.0
        m = max(ms, _MASS_FLOOR_KG)
        Dm = np.zeros(dim)
        if spec.ms_col is not None:
            Dm[spec.ms_col] = 1.0
        for i in range(nseg_fwd):
            P[i, 0], J[i, 0] = s[0:3], D[0:3]
            s, D = step(s, D, dt / 2, d_dt / 2)
            P[i, 1], J[i, 1] = s[0:3], D[0:3]
            u = thr[i]
            j = u0 + 3 * i
            kk, Dk = kick(D, Dm, u, m, j, 1.0)
            s[3:6] += u * (c / m)
            ek = math.exp(-min(kk, 700.0))
            Dm = ek * Dm - (m * ek) * Dk
            m = max(m * ek, _MASS_FLOOR_KG)
            s, D = step(s, D, dt / 2, d_dt / 2)
            P[i, 2], J[i, 2] = s[0:3], D[0:3]

        s = np.concatenate([rf, vf + np.asarray(x[spec.vf], float)])
        D = np.zeros((6, dim))
        epoch_rows(D, rf, vf, spec.end_cols)
        D[3, spec.vf.start] = D[4, spec.vf.start + 1] = D[5, spec.vf.start + 2] = 1.0
        m = max(mf, _MASS_FLOOR_KG)
        Dm = np.zeros(dim)
        Dm[spec.mf_col] = 1.0
        for jj in range(nseg - nseg_fwd):
            seg = nseg - 1 - jj
            P[seg, 2], J[seg, 2] = s[0:3], D[0:3]
            s, D = step(s, D, -dt / 2, -d_dt / 2)
            P[seg, 1], J[seg, 1] = s[0:3], D[0:3]
            u = thr[seg]
            j = u0 + 3 * seg
            kk, Dk = kick(D, Dm, u, m, j, -1.0)
            s[3:6] -= u * (c / m)
            ek = math.exp(min(kk, 700.0))
            Dm = ek * Dm + (m * ek) * Dk
            m = max(m * ek, _MASS_FLOOR_KG)
            s, D = step(s, D, -dt / 2, -d_dt / 2)
            P[seg, 0], J[seg, 0] = s[0:3], D[0:3]
        return P, J

    def segment_caps(self, x) -> tuple[np.ndarray, np.ndarray] | None:
        """Each leg's per-segment ceiling at ``x`` (fraction of the thrust flown), or None."""
        if self.cap_fn is None:
            return None
        out = []
        for k in range(2):
            P, _ = self._chain(x, k, jacobian=False)
            pts = CHORD_WEIGHTS @ P
            out.append(np.asarray(self.cap_fn(np.linalg.norm(pts, axis=2) / pk.AU), float).mean(axis=1))
        return tuple(out)

    def _cap_rows(self, x, gradient: bool = True):
        """``|u_i|^2 - capbar_i(x)^2 <= 0`` for every segment of both legs and, with ``gradient``,
        the dense rows ``(n1 + n2, dim)``. Cached on ``x``; light chain where the ceiling was flat."""
        xa = np.asarray(x, float)
        key = xa.tobytes()
        if key == self._cache_key and (self._cache[1] is not None or not gradient):
            return self._cache
        light = not gradient or (self._light_ok and self._flat)
        vals, rows, flat_all = [], [], True
        for k, spec in enumerate(self.specs):
            P, J = self._chain(xa, k, jacobian=not light)
            S = CHORD_WEIGHTS @ P
            r_s = np.sqrt((S * S).sum(axis=2))
            cap_s, slope_s = self.cap_fn.with_derivative(r_s / pk.AU)
            flat = not np.any(slope_s)
            if not flat:
                self._light_ok = False
                flat_all = False
                if J is None and gradient:
                    P, J = self._chain(xa, k, jacobian=True)
            cap = cap_s.mean(axis=1)
            u = xa[spec.u].reshape(spec.nseg, 3)
            vals.extend(((u * u).sum(axis=1) - cap * cap).tolist())
            if not gradient:
                continue
            n, ns = spec.nseg, CHORD_WEIGHTS.shape[0]
            if flat:
                G = np.zeros((n, xa.size))
            else:
                JS = (CHORD_WEIGHTS @ J.reshape(n, 3, 3 * xa.size)).reshape(n, ns, 3, xa.size)
                w = slope_s / (np.maximum(r_s, 1.0) * pk.AU)
                pj = (S[:, :, :, None] * JS).sum(axis=2)
                G = (w[:, :, None] * pj).sum(axis=1) * (-2.0 * cap / ns)[:, None]
            idx = np.arange(n)[:, None] * 3 + spec.u.start + np.arange(3)[None, :]
            G[np.arange(n)[:, None], idx] += 2.0 * u
            rows.append(G)
        self._flat = flat_all
        grad = np.vstack(rows) if gradient else None
        self._cache_key, self._cache = key, (vals, grad)
        return self._cache

    # -- the flyby ------------------------------------------------------------------------

    def _flyby(self, vin, vout):
        """(equality, inequality) of the unpowered flyby and their gradients w.r.t. (vin, vout).

        eq = (|vin|^2 - |vout|^2) / v_scaling^2;
        ineq = (1 - 2/e^2) - cos(alpha), e = 1 + |vin|^2 rp/mu, cos(alpha) = vin.vout/(|vin||vout|).
        """
        vin = np.asarray(vin, float)
        vout = np.asarray(vout, float)
        a2, b2 = float(vin @ vin), float(vout @ vout)
        eq = (a2 - b2) / self.v_scaling ** 2
        d_eq = np.concatenate((2.0 * vin, -2.0 * vout)) / self.v_scaling ** 2
        a, b = math.sqrt(max(a2, 1e-12)), math.sqrt(max(b2, 1e-12))
        cos_alpha = float(vin @ vout) / (a * b)
        e = 1.0 + a2 * self.fb_rp / self.fb_mu
        ineq = (1.0 - 2.0 / e ** 2) - cos_alpha
        dcos_dvin = vout / (a * b) - cos_alpha * vin / a2
        dcos_dvout = vin / (a * b) - cos_alpha * vout / b2
        d_ineq = np.concatenate(((4.0 / e ** 3) * (2.0 * self.fb_rp / self.fb_mu) * vin - dcos_dvin,
                                 -dcos_dvout))
        return eq, ineq, d_eq, d_ineq

    def flyby_geometry(self, x) -> dict:
        """Speed relative to the flyby body (km/s), turn angle (deg) and the periapsis altitude
        (km) that turn implies, for reporting."""
        vin, vout = np.asarray(x[_VIN], float), np.asarray(x[_VOUT], float)
        a, b = float(np.linalg.norm(vin)), float(np.linalg.norm(vout))
        cos_alpha = float(np.clip(vin @ vout / max(a * b, 1e-12), -1.0, 1.0))
        alpha = math.acos(cos_alpha)
        s = math.sin(alpha / 2.0)
        rp = float("inf") if s <= 1e-12 or a <= 0 else (1.0 / s - 1.0) * self.fb_mu / a ** 2
        radius = self.bodies[1].get_radius()
        return {"vinf_kms": a / 1000.0, "vinf_out_kms": b / 1000.0, "turn_deg": math.degrees(alpha),
                "periapsis_alt_km": (rp - radius) / 1000.0 if math.isfinite(rp) else float("inf"),
                "min_periapsis_alt_km": (self.fb_rp - radius) / 1000.0}

    # -- pygmo interface --------------------------------------------------------------------

    def fitness(self, x):
        x = np.asarray(x, float)
        self._set_legs(x)
        l1, l2 = self.legs
        eq1 = self._scale_rows(np.asarray(l1.compute_mismatch_constraints(), float))
        eq2 = self._scale_rows(np.asarray(l2.compute_mismatch_constraints(), float))
        fb_eq, fb_ineq, _, _ = self._flyby(x[_VIN], x[_VOUT])
        f = [-float(x[_MF]) / self.m_scaling]
        f.extend(eq1.tolist())
        f.extend(eq2.tolist())
        f.append(fb_eq)
        f.extend(l1.compute_throttle_constraints())
        f.extend(l2.compute_throttle_constraints())
        f.append((float(x[_VDEP] @ x[_VDEP]) - self.vinf_dep ** 2) / self.v_scaling ** 2)
        if self.vinf_dep_exact:
            f.append((self.vinf_dep ** 2 - float(x[_VDEP] @ x[_VDEP])) / self.v_scaling ** 2)
        f.append((float(x[_VARR] @ x[_VARR]) - self.vinf_arr ** 2) / self.v_scaling ** 2)
        f.append(fb_ineq)
        if self.arrive_mjd2000 is not None:
            f.append(float(x[_T0] + x[_TOF1] + x[_TOF2]) - self.arrive_mjd2000)
        if self.tof_total_bounds is not None:
            total = float(x[_TOF1] + x[_TOF2])
            f.append(total - self.tof_total_bounds[1])
            f.append(self.tof_total_bounds[0] - total)
        if self._sin2_decl is not None:
            f.append(self._declination(x))
        if self.cap_fn is not None:
            f.extend(self._cap_rows(x, gradient=False)[0])
        return f

    def _declination(self, x) -> float:
        vx, vy, vz = x[_VDEP]
        vz_eq = vy * _SIN_EPS + vz * _COS_EPS
        return float((vz_eq * vz_eq - self._sin2_decl * (vx * vx + vy * vy + vz * vz)) / self._vinf2)

    def get_bounds(self):
        m0 = self.mass_kg
        vfb = _VINF_FB_MAX_KMS * 1000.0
        lo = ([self.t0_bounds[0], self.tof1_bounds[0], self.tof2_bounds[0],
               m0 * _MF_FLOOR_FRACTION, m0 * _MF_FLOOR_FRACTION]
              + [-self.vinf_dep] * 3 + [-vfb] * 6 + [-self.vinf_arr] * 3
              + [-1.0] * (3 * (self.n1 + self.n2)))
        hi = ([self.t0_bounds[1], self.tof1_bounds[1], self.tof2_bounds[1], m0, m0]
              + [self.vinf_dep] * 3 + [vfb] * 6 + [self.vinf_arr] * 3
              + [1.0] * (3 * (self.n1 + self.n2)))
        return lo, hi

    def get_nobj(self):
        return 1

    def get_nec(self):
        return 15

    def get_nic(self):
        return (self.n1 + self.n2 + 3 + (1 if self.arrive_mjd2000 is not None else 0)
                + (1 if self.vinf_dep_exact else 0)
                + (2 if self.tof_total_bounds is not None else 0)
                + (1 if self._sin2_decl is not None else 0)
                + (self.n1 + self.n2 if self.cap_fn is not None else 0))

    def has_gradient(self):
        return True

    def gradient(self, x):
        x = np.asarray(x, float)
        (r0, v0), (r1, v1), (r2, v2) = self._set_legs(x)
        l1, l2 = self.legs
        mu = pk.MU_SUN
        nrows = 1 + self.get_nec() + self.get_nic()
        G = np.zeros((nrows, self.dim))
        row = 0
        G[row, _MF] = -1.0 / self.m_scaling
        row += 1

        xs, xf, th = (np.asarray(a, float) for a in l1.compute_mc_grad())
        t_dep = _body_term(xs, r0, v0, mu)
        t_fb1 = _body_term(xf, r1, v1, mu)
        block = np.zeros((7, self.dim))
        block[:, _T0] = (t_dep + t_fb1) * pk.DAY2SEC
        block[:, _TOF1] = (th[:, -1] + t_fb1) * pk.DAY2SEC
        block[:, _MFB] = xf[:, 6]
        block[:, _VDEP] = xs[:, 3:6]
        block[:, _VIN] = xf[:, 3:6]
        block[:, self._u1] = th[:, :-1]
        G[row:row + 7] = self._scale_rows(block.T).T
        row += 7

        xs, xf, th = (np.asarray(a, float) for a in l2.compute_mc_grad())
        t_fb2 = _body_term(xs, r1, v1, mu)
        t_arr = _body_term(xf, r2, v2, mu)
        block = np.zeros((7, self.dim))
        block[:, _T0] = (t_fb2 + t_arr) * pk.DAY2SEC
        block[:, _TOF1] = (t_fb2 + t_arr) * pk.DAY2SEC
        block[:, _TOF2] = (th[:, -1] + t_arr) * pk.DAY2SEC
        block[:, _MFB] = xs[:, 6]
        block[:, _MF] = xf[:, 6]
        block[:, _VOUT] = xs[:, 3:6]
        block[:, _VARR] = xf[:, 3:6]
        block[:, self._u2] = th[:, :-1]
        G[row:row + 7] = self._scale_rows(block.T).T
        row += 7

        _, _, d_eq, d_ineq = self._flyby(x[_VIN], x[_VOUT])
        G[row, _VIN], G[row, _VOUT] = d_eq[:3], d_eq[3:]
        row += 1
        G[row:row + self.n1, self._u1] = np.asarray(l1.compute_tc_grad(), float)
        row += self.n1
        G[row:row + self.n2, self._u2] = np.asarray(l2.compute_tc_grad(), float)
        row += self.n2
        G[row, _VDEP] = 2.0 * x[_VDEP] / self.v_scaling ** 2
        row += 1
        if self.vinf_dep_exact:
            G[row, _VDEP] = -2.0 * x[_VDEP] / self.v_scaling ** 2
            row += 1
        G[row, _VARR] = 2.0 * x[_VARR] / self.v_scaling ** 2
        row += 1
        G[row, _VIN], G[row, _VOUT] = d_ineq[:3], d_ineq[3:]
        row += 1
        if self.arrive_mjd2000 is not None:
            G[row, [_T0, _TOF1, _TOF2]] = 1.0
            row += 1
        if self.tof_total_bounds is not None:
            G[row, [_TOF1, _TOF2]] = 1.0
            G[row + 1, [_TOF1, _TOF2]] = -1.0
            row += 2
        if self._sin2_decl is not None:
            vx, vy, vz = x[_VDEP]
            vz_eq = vy * _SIN_EPS + vz * _COS_EPS
            k, s2 = 2.0 / self._vinf2, self._sin2_decl
            G[row, _VDEP] = (k * (-s2 * vx), k * (vz_eq * _SIN_EPS - s2 * vy),
                             k * (vz_eq * _COS_EPS - s2 * vz))
            row += 1
        if self.cap_fn is not None:
            n = self.n1 + self.n2
            G[row:row + n] = self._cap_rows(x)[1]
            row += n
        return G.ravel().tolist()

    # -- the flown nodes ----------------------------------------------------------------------

    def leg_nodes(self, x) -> tuple[np.ndarray, np.ndarray]:
        """Both legs as node arrays in :func:`simsflanagan._leg_nodes` layout: ``2*nseg + 1`` rows
        of ``[t (mjd2000), x, y, z (m), vx, vy, vz (m/s), m (kg), |u|, ux, uy, uz]``."""
        x = np.asarray(x, float)
        (r0, v0), (r1, v1), (r2, v2) = self._set_legs(x)
        t0, t_fb, _ = self._epochs(x)
        l1, l2 = self.legs
        n1 = _nodes(l1, t0, float(x[_TOF1]), (r0, v0 + x[_VDEP]), (r1, v1 + x[_VIN]),
                    self.mass_kg, float(x[_MFB]), x[self._u1].reshape(self.n1, 3))
        n2 = _nodes(l2, t_fb, float(x[_TOF2]), (r1, v1 + x[_VOUT]), (r2, v2 + x[_VARR]),
                    float(x[_MFB]), float(x[_MF]), x[self._u2].reshape(self.n2, 3))
        return n1, n2


def _nodes(leg, t_start, tof_days, rv_start, rv_end, ms, mf, throttles) -> np.ndarray:
    """One leg's nodes, forward from its start and backward from its end, meeting at the cut."""
    nseg = throttles.shape[0]
    dt = tof_days * pk.DAY2SEC / nseg
    c = leg.max_thrust * dt
    veff, mu = leg.veff, leg.mu
    nseg_fwd = int(nseg * _MATCHPOINT_CUT)
    nodes = np.zeros((2 * nseg + 1, 12))
    nodes[:, 0] = t_start + np.arange(2 * nseg + 1) * (tof_days / (2 * nseg))

    rv = [list(rv_start[0]), list(rv_start[1])]
    mass = float(ms)
    for i in range(nseg_fwd):
        u = throttles[i]
        nodes[2 * i, 1:4], nodes[2 * i, 4:7], nodes[2 * i, 7] = rv[0], rv[1], mass
        rv = list(pk.propagate_lagrangian(rv, tof=dt / 2, mu=mu, stm=False))
        nodes[2 * i + 1, 1:4], nodes[2 * i + 1, 4:7], nodes[2 * i + 1, 7] = rv[0], rv[1], mass
        dv = float(np.linalg.norm(u)) * c / mass
        rv[1] = [a + b * c / mass for a, b in zip(rv[1], u)]
        mass = max(mass * math.exp(-min(dv / veff, 700.0)), _MASS_FLOOR_KG)
        rv = list(pk.propagate_lagrangian(rv, tof=dt / 2, mu=mu, stm=False))

    rv = [list(rv_end[0]), list(rv_end[1])]
    mass = float(mf)
    for j in range(nseg - nseg_fwd):
        k = 2 * nseg - 2 * j
        u = throttles[nseg - 1 - j]
        nodes[k, 1:4], nodes[k, 4:7], nodes[k, 7] = rv[0], rv[1], mass
        rv = list(pk.propagate_lagrangian(rv, tof=-dt / 2, mu=mu, stm=False))
        nodes[k - 1, 1:4], nodes[k - 1, 4:7], nodes[k - 1, 7] = rv[0], rv[1], mass
        dv = float(np.linalg.norm(u)) * c / mass
        rv[1] = [a - b * c / mass for a, b in zip(rv[1], u)]
        mass = max(mass * math.exp(min(dv / veff, 700.0)), _MASS_FLOOR_KG)
        rv = list(pk.propagate_lagrangian(rv, tof=-dt / 2, mu=mu, stm=False))
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
class FlybySolution:
    """A solved two-leg transfer, or the best attempt at one."""
    target_name: str
    flyby_name: str
    feasible: bool
    mismatch: float
    dep_mjd2000: float
    tof1_days: float
    tof2_days: float
    initial_mass_kg: float
    flyby_mass_kg: float
    final_mass_kg: float
    dv_kms: float
    nseg: tuple[int, int]
    isp_s: tuple[float, float]                   # each leg's Isp
    thrust_N: float
    max_duty_cycle: float
    flyby: dict                                  # :meth:`_FlybyUDP.flyby_geometry`
    # Both legs' nodes in sequence; leg one's last and leg two's first both sit at the flyby body.
    trajectory: np.ndarray
    leg_trajectories: tuple[np.ndarray, np.ndarray] = field(repr=False)
    decision_vector: np.ndarray = field(repr=False)
    seg_caps: tuple[np.ndarray, np.ndarray] | None = None
    seg_isp_s: tuple[np.ndarray, np.ndarray] | None = None
    refresh_log: list | None = None
    refresh_settled: bool = True

    @property
    def tof_days(self) -> float:
        return self.tof1_days + self.tof2_days

    @property
    def flyby_mjd2000(self) -> float:
        return self.dep_mjd2000 + self.tof1_days

    @property
    def propellant_kg(self) -> float:
        return self.initial_mass_kg - self.final_mass_kg

    @property
    def dep_date(self) -> datetime:
        return lb.date_from_mjd2000(self.dep_mjd2000)

    @property
    def flyby_date(self) -> datetime:
        return lb.date_from_mjd2000(self.flyby_mjd2000)

    @property
    def arr_date(self) -> datetime:
        return lb.date_from_mjd2000(self.dep_mjd2000 + self.tof_days)

    @property
    def positions_au(self) -> np.ndarray:
        return self.trajectory[:, 1:4] / pk.AU

    @property
    def throttle(self) -> np.ndarray:
        return self.trajectory[:, 8]


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------

def _make_udp(target, flyby_body, *, mass_kg, thrust_N, isp_s, nseg, launch_window, arrive_by,
              vinf_dep_kms, vinf_arr_kms, window_slack_days, min_tof_days, max_tof_days,
              max_duty_cycle, max_dep_decl_deg, depart_body, min_flyby_alt_km, flyby_mu,
              cap_fn=None, tof_total_bounds=None, vinf_dep_exact=False):
    start, end = launch_window
    t0_lo = lb.mjd2000_from_date(start) - window_slack_days
    t0_hi = lb.mjd2000_from_date(end) + window_slack_days
    arrive_mjd = lb.mjd2000_from_date(arrive_by)
    tof_ceiling = arrive_mjd - t0_lo
    tof_hi = tof_ceiling if max_tof_days is None else min(float(max_tof_days), tof_ceiling)
    if tof_total_bounds is not None:
        tof_hi = min(tof_hi, float(tof_total_bounds[1]))
    min1, min2 = min_tof_days
    if tof_hi <= min1 + min2:
        raise ValueError(
            f"window too short for a two-leg transfer: at most {tof_hi:.0f} days are allowed "
            f"(the earliest departure leaves {tof_ceiling:.0f} to arrive_by {arrive_by}"
            + (f", max_tof_days caps it at {float(max_tof_days):.0f}" if max_tof_days is not None else "")
            + f"), below the {min1 + min2:.0f}-day minimum for the two legs.")
    return _FlybyUDP(lb.earth_planet() if depart_body is None else depart_body, flyby_body, target,
                     mass_kg, thrust_N * max_duty_cycle, isp_s, nseg[0], nseg[1],
                     (t0_lo, t0_hi), (min1, tof_hi - min2), (min2, tof_hi - min1),
                     vinf_dep_kms, vinf_arr_kms, arrive_mjd2000=arrive_mjd,
                     max_dep_decl_deg=max_dep_decl_deg, min_flyby_alt_km=min_flyby_alt_km,
                     flyby_mu=flyby_mu, cap_fn=cap_fn, tof_total_bounds=tof_total_bounds,
                     vinf_dep_exact=vinf_dep_exact)


def _impulsive_start(udp: _FlybyUDP, t_dep: float, t_arr: float, rng_seed: int = 42):
    """The start for a departure day and an arrival day: pykep's ``mga_1dsm`` optimum between
    them (days pinned, flyby split and DSM free, fixed seed) decoded into the two-leg vector with
    the engine off. A start needs a consistent geometry: bare Lambert arcs through the planet
    closed 2 of 64 cells (14-17 km/s mismatch), this closes 36 of 36, under a second each.
    Returns the vector and the impulsive dv (km/s, departure excess excluded), or None."""
    dep, fbody, tgt = udp.bodies
    lo, hi = (np.asarray(b, float) for b in udp.get_bounds())
    tof = float(t_arr - t_dep)
    try:
        mga = pk.trajopt.mga_1dsm(
            seq=[dep, fbody, tgt], t0=[pk.epoch(t_dep - 0.5), pk.epoch(t_dep + 0.5)],
            tof=[max(tof - 1.0, lo[_TOF1] + lo[_TOF2]), tof + 1.0],
            vinf=[0.0, max(udp.vinf_dep / 1000.0, 0.1)], add_vinf_dep=False, add_vinf_arr=True,
            tof_encoding="alpha", multi_objective=False)
        prob = pg.problem(mga)
        algo = pg.algorithm(pg.sade(gen=300, seed=rng_seed))
        best = None
        for k in range(2):
            pop = algo.evolve(pg.population(prob, size=40, seed=rng_seed + k))
            if best is None or pop.champion_f[0] < best[0]:
                best = (pop.champion_f[0], pop.champion_x)
        x = best[1]
        _dv, lamberts, T, legs, _eps = mga._compute_dvs(x)
    except Exception:  # noqa: BLE001  (an impulsive search that fails is a start not taken)
        return None
    t0, T1, T2 = float(x[0]), float(T[0]), float(T[1])
    _, v_e = (np.asarray(a, float) for a in dep.eph(t0))
    _, v_m = (np.asarray(a, float) for a in fbody.eph(t0 + T1))
    _, v_p = (np.asarray(a, float) for a in tgt.eph(t0 + T1 + T2))
    z = np.zeros(udp.dim)
    z[_T0], z[_TOF1], z[_TOF2] = t0, T1, T2
    z[_MFB], z[_MF] = 0.9 * udp.mass_kg, 0.75 * udp.mass_kg
    z[_VDEP] = np.asarray(legs[0][1], float) - v_e
    z[_VIN] = np.asarray(lamberts[0].v1[0], float) - v_m
    z[_VOUT] = np.asarray(legs[2][1], float) - v_m
    z[_VARR] = np.asarray(lamberts[1].v1[0], float) - v_p
    return np.clip(z, lo, hi), float(best[0]) / 1000.0


def _problem(udp: _FlybyUDP, c_tol: float):
    prob = pg.problem(udp)
    prob.c_tol = [c_tol] * prob.get_nc()
    return prob


def _descent(maxeval: int):
    local = pg.nlopt("slsqp")
    local.maxeval = int(maxeval)
    local.xtol_rel = 1e-8
    return pg.algorithm(local)


def _evaluate(prob, x):
    f = np.asarray(prob.fitness(x), float)
    nec = prob.get_nec()
    mismatch = float(np.max(np.abs(f[1:1 + nec]))) if nec else 0.0
    return bool(prob.feasibility_x(x)), mismatch, float(x[_MF])


def _better(cand, best) -> bool:
    if cand[0] != best[0]:
        return cand[0]
    return cand[2] > best[2] if cand[0] else cand[1] < best[1]


def _descend(udp: _FlybyUDP, x, *, rounds: int, c_tol: float, maxeval: int, rng_seed: int,
             tick=None, hop: bool = True):
    """``rounds`` rounds of an SLSQP descent plus one perturbed retry from ``x``, fixed seed;
    returns the best (feasible, mismatch, final mass) and its vector, evaluated on ``udp``."""
    prob = _problem(udp, c_tol)
    lo, hi = (np.asarray(b, float) for b in udp.get_bounds())
    pop = pg.population(prob, size=1, seed=rng_seed)
    pop.set_x(0, np.clip(np.asarray(x, float), lo, hi))
    best_x = pop.champion_x
    best = _evaluate(prob, best_x)
    # A plain descent for a small correction (the Isp pass); the perturbed retry that helps a
    # cold start can carry a nearly-right answer into a far worse basin (measured on Psyche).
    algo = (pg.algorithm(pg.mbh(_descent(maxeval), stop=1, perturb=0.05, seed=rng_seed))
            if hop else _descent(maxeval))
    for _ in range(max(1, int(rounds))):
        pop = algo.evolve(pop)
        cand = _evaluate(prob, pop.champion_x)
        if _better(cand, best):
            best, best_x = cand, pop.champion_x
        if tick is not None:
            tick()
    return best, best_x


def _leg_isps(udp: _FlybyUDP, x, seg_isp_fn) -> tuple[tuple[float, float], tuple | None]:
    """Each leg's throttle-weighted Isp off its own path, and the per-segment Isps."""
    n1, n2 = udp.leg_nodes(x)
    isps, segs = [], []
    for nodes, isp0 in zip((n1, n2), udp.isp_s):
        seg = seg_isp_fn(nodes[:, 1:4])
        if seg is None:
            return udp.isp_s, None
        seg = np.asarray(seg, float)
        used = np.abs(nodes[0:2 * ((nodes.shape[0] - 1) // 2):2, 8])
        isps.append(float(effective_isp(used, seg, isp0)))
        segs.append(seg)
    return (isps[0], isps[1]), (segs[0], segs[1])


def solve(
    target,
    flyby_body,
    *,
    mass_kg: float,
    thrust_N: float,
    isp_s: float | tuple[float, float],
    launch_window: tuple[date, date],
    arrive_by: date,
    seed: tuple[float, float] | None = None,
    x0=None,
    nseg: tuple[int, int] = (12, 12),
    maxeval: int = 2000,
    restarts: int = 8,
    vinf_dep_kms: float = 1.0,
    vinf_arr_kms: float = 0.1,
    window_slack_days: float = 0.0,
    min_tof_days: tuple[float, float] = (60.0, 120.0),
    max_tof_days: float | None = None,
    tof_total_bounds: tuple[float, float] | None = None,
    vinf_dep_exact: bool = False,
    max_duty_cycle: float = 1.0,
    max_dep_decl_deg: float | None = None,
    depart_body=None,
    min_flyby_alt_km: float | None = None,
    flyby_mu: float | None = None,
    thrust_cap_fn=None,
    seg_isp_fn: Callable | None = None,
    isp_iters: int = 3,
    dv_budget_kms: float | None = None,
    c_tol: float = 1e-4,
    rng_seed: int = 42,
    progress: Callable[[int, int], None] | None = None,
) -> FlybySolution:
    """Optimize a low-thrust transfer Earth -> ``flyby_body`` -> ``target`` for final mass; the
    two-leg counterpart of :func:`simsflanagan.solve`, same conventions, fixed seeds throughout.

    ``seed`` is the ``(departure, arrival)`` MJD2000 pair the impulsive start is built at
    (:func:`_impulsive_start`; None: window start and deadline); ``x0`` is a two-leg vector
    instead. Each of ``restarts`` rounds is its own impulsive geometry (several exist at one pair
    of days: eight on Hayabusa2's best cell descended to 22 to 59 kg), descended with the flyby
    split held and then free; the best is polished.
    ``min_tof_days`` is per leg; ``max_tof_days`` caps the trip; ``tof_total_bounds`` pins it (a
    grid cell). ``min_flyby_alt_km`` floors the periapsis (None: safe radius). ``thrust_cap_fn``
    and ``seg_isp_fn`` apply to both legs, each with its own Isp pass. With ``dv_budget_kms``, a
    start whose impulsive dv exceeds the tank or the engine's delivery is returned unsolved (most
    of a grid, a second each instead of a minute).
    """
    cap = None
    if thrust_cap_fn is not None:
        cap = (thrust_cap_fn if isinstance(thrust_cap_fn, SmoothCap)
               else SmoothCap.from_function(thrust_cap_fn))
    isps = tuple(float(v) for v in (isp_s if isinstance(isp_s, (tuple, list)) else (isp_s, isp_s)))

    def make(isp_pair, tof1_fixed=None, tof2_fixed=None):
        udp = _make_udp(target, flyby_body, mass_kg=mass_kg, thrust_N=thrust_N, isp_s=isp_pair,
                        nseg=nseg, launch_window=launch_window, arrive_by=arrive_by,
                        vinf_dep_kms=vinf_dep_kms, vinf_arr_kms=vinf_arr_kms,
                        window_slack_days=window_slack_days, min_tof_days=min_tof_days,
                        max_tof_days=max_tof_days, max_duty_cycle=max_duty_cycle,
                        max_dep_decl_deg=max_dep_decl_deg, depart_body=depart_body,
                        min_flyby_alt_km=min_flyby_alt_km, flyby_mu=flyby_mu, cap_fn=cap,
                        tof_total_bounds=tof_total_bounds, vinf_dep_exact=vinf_dep_exact)
        if tof1_fixed is not None:
            lo1, hi1 = udp.tof1_bounds
            udp.tof1_bounds = (max(lo1, tof1_fixed - 1.0), min(hi1, tof1_fixed + 1.0))
            lo2, hi2 = udp.tof2_bounds
            udp.tof2_bounds = (max(lo2, tof2_fixed - 1.0), min(hi2, tof2_fixed + 1.0))
        return udp

    udp = make(isps)
    isp_rounds = int(isp_iters) if seg_isp_fn is not None else 0
    rounds = max(1, int(restarts))
    total = (rounds if x0 is not None and seed is None
             else 2 * rounds + (1 if x0 is not None else 0)) + 1 + isp_rounds
    done = [0]

    def tick(n: int = 1) -> None:
        done[0] += n
        if progress is not None:
            progress(done[0], total)

    name = getattr(target, "name", "target")
    if x0 is not None and seed is None:
        best, best_x = _descend(udp, x0, rounds=rounds, c_tol=c_tol, maxeval=maxeval,
                                rng_seed=rng_seed, tick=tick)
    else:
        if seed is None:
            seed = (udp.t0_bounds[0], float(udp.arrive_mjd2000))
        # The impulsive problem is multi-modal too, so each restart is its own geometry (its own
        # seed), descended with the flyby split held first and then free; the best is kept. A
        # warm start is one more candidate, not a substitute: the best cell's own geometries are
        # where the better answers were found.
        best, best_x, skipped = None, None, None
        if x0 is not None:
            best, best_x = _descend(udp, x0, rounds=1, c_tol=c_tol, maxeval=maxeval,
                                    rng_seed=rng_seed, tick=tick)
        for r in range(rounds):
            start = _impulsive_start(udp, float(seed[0]), float(seed[1]), rng_seed + r)
            if start is None:
                tick(2)
                continue
            z, dv_impulsive = start
            if dv_budget_kms is not None and dv_impulsive > 1.1 * _dv_deliverable(
                    dv_budget_kms, mass_kg, thrust_N * max_duty_cycle, isps[0],
                    float(z[_TOF1] + z[_TOF2])):
                skipped = z
                tick(2)
                continue
            held, z = _descend(make(isps, float(z[_TOF1]), float(z[_TOF2])), z, rounds=1,
                               c_tol=c_tol, maxeval=maxeval, rng_seed=rng_seed + r, tick=tick)
            if not held[0] and held[1] > _DEAD_MISMATCH:
                # A geometry that is still far off with the split held never closes when freed
                # (measured 0.35 to 0.99 against 1e-5 for one that does); its second descent is
                # what makes a dead cell cost a live one.
                cand, x_r = held, z
                tick()
            else:
                cand, x_r = _descend(udp, z, rounds=1, c_tol=c_tol, maxeval=maxeval,
                                     rng_seed=rng_seed + r, tick=tick)
            if best is None or _better(cand, best):
                best, best_x = cand, x_r
        if best is None:
            if skipped is None:
                raise ValueError("no impulsive gravity-assist geometry to start from between "
                                 f"{lb.date_from_mjd2000(seed[0]):%Y-%m-%d} and "
                                 f"{lb.date_from_mjd2000(seed[1]):%Y-%m-%d}")
            # Every route needs more than the engine can deliver in this flight time: unsolved.
            if progress is not None:
                progress(total, total)
            return _build_solution(udp, skipped, name, _body_name(flyby_body), thrust_N,
                                   max_duty_cycle)
    if best[0]:
        cand, x_p = _descend(udp, best_x, rounds=1, c_tol=c_tol, maxeval=maxeval,
                             rng_seed=rng_seed + 50, tick=tick)
        if _better(cand, best):
            best, best_x = cand, x_p
    else:
        tick()

    # Isp pass, per leg: re-solve from the answer at the Isps its own path implies until both agree.
    kept = (udp, best_x, best, isps, None) if best[0] else None
    refresh_log: list = []
    settled = True
    seg_isps = None
    for k in range(isp_rounds):
        new_isps, seg_isps = _leg_isps(udp, best_x, seg_isp_fn)
        if seg_isps is None:
            break
        refresh_log.append({"round": k, "isp_before": isps, "isp_after": new_isps,
                            "mf_before": float(best[2])})
        if all(abs(n - o) <= ISP_TOL * o for n, o in zip(new_isps, isps)):
            settled = True
            break
        settled = False
        isps = new_isps
        udp = make(isps)
        cand, x_k = _descend(udp, best_x, rounds=1, c_tol=c_tol, maxeval=maxeval,
                             rng_seed=rng_seed + 100 + k, tick=tick, hop=False)
        best, best_x = cand, x_k
        refresh_log[-1]["mf_after"] = float(best[2])
        if best[0]:
            kept = (udp, best_x, best, isps, seg_isps)
    if progress is not None and done[0] < total:
        progress(total, total)

    restored = False
    if not best[0] and kept is not None:
        udp, best_x, best, isps, seg_isps = kept
        restored = True
    if isp_rounds and (restored or not (settled and best[0])):
        end_isps, seg_isps = _leg_isps(udp, best_x, seg_isp_fn)
        if seg_isps is not None:
            settled = all(abs(n - o) <= ISP_TOL * o for n, o in zip(end_isps, udp.isp_s))
    sol = _build_solution(udp, best_x, name, _body_name(flyby_body), thrust_N, max_duty_cycle,
                          seg_isp_s=seg_isps)
    sol.refresh_log = refresh_log or None
    sol.refresh_settled = bool(settled)
    return sol


def _dv_deliverable(dv_tank_kms: float, mass_kg: float, thrust_N: float, isp_s: float,
                    tof_days: float) -> float:
    """Upper bound on the cruise dv (km/s): the tank's, or the engine firing the whole flight time
    at the lightest mass the tank allows, whichever is less."""
    m_min = mass_kg * math.exp(-dv_tank_kms * 1000.0 / (isp_s * G0))
    dv_time = thrust_N * tof_days * 86400.0 / max(m_min, 1.0) / 1000.0
    return min(float(dv_tank_kms), dv_time)


def _body_name(flyby_body) -> str:
    # pykep names its analytic planets "mars(jpl_lp)"; the body's plain name is what is reported.
    return str(flyby_body.get_name()).split("(")[0].strip() or "flyby body"


def _build_solution(udp: _FlybyUDP, x, target_name, flyby_name, thrust_N, max_duty_cycle,
                    seg_isp_s=None) -> FlybySolution:
    prob = _problem(udp, 1e-4)
    feasible, mismatch, mf = _evaluate(prob, x)
    n1, n2 = udp.leg_nodes(x)
    for n in (n1, n2):
        n[:, 8:12] *= max_duty_cycle
    m0, mfb = udp.mass_kg, float(x[_MFB])
    isp1, isp2 = udp.isp_s
    dv = ((isp1 * math.log(m0 / mfb) + isp2 * math.log(mfb / mf)) * G0 / 1000.0
          if mf > 0 and mfb > 0 else float("nan"))
    caps = udp.segment_caps(x)
    seg_caps = None if caps is None else tuple(np.asarray(c, float) * float(max_duty_cycle)
                                               for c in caps)
    return FlybySolution(
        target_name=target_name, flyby_name=flyby_name, feasible=feasible, mismatch=mismatch,
        dep_mjd2000=float(x[_T0]), tof1_days=float(x[_TOF1]), tof2_days=float(x[_TOF2]),
        initial_mass_kg=m0, flyby_mass_kg=mfb, final_mass_kg=mf, dv_kms=dv,
        nseg=(udp.n1, udp.n2), isp_s=udp.isp_s, thrust_N=float(thrust_N),
        max_duty_cycle=float(max_duty_cycle), flyby=udp.flyby_geometry(x),
        trajectory=np.vstack((n1, n2)), leg_trajectories=(n1, n2),
        decision_vector=np.asarray(x, float), seg_caps=seg_caps, seg_isp_s=seg_isp_s)


# ---------------------------------------------------------------------------
# from a config: the same three entry points as simsflanagan
# ---------------------------------------------------------------------------

def flyby_body(key: str):
    """The pykep planet for a ``Mission.gravity_assist`` key."""
    key = str(key).strip().lower()
    if key == "earth":
        return lb.earth_planet()
    return pk.planet(pk.udpla.jpl_lp(key))


def _config_kwargs(rc, kwargs: dict) -> None:
    """Config defaults, in place: window, deadline, declination cone, arrival speed, flyby floor."""
    from prospector.solvers import simsflanagan as sf
    kwargs.setdefault("max_dep_decl_deg", sf.max_departure_declination_deg(rc.launch))
    kwargs.setdefault("launch_window", rc.departure_window)
    kwargs.setdefault("arrive_by", rc.mission.arrive_by)
    kwargs.setdefault("vinf_arr_kms", rc.arrival_vinf_kms)
    kwargs.setdefault("min_flyby_alt_km", float(rc.bus_model().flyby_min_altitude_km))
    kwargs.setdefault("dv_budget_kms", float(rc.cruise_dv_limit))
    kwargs.setdefault("vinf_dep_exact", bool(rc.launch.escape_provided))
    # Single-leg callers pass one flight-time range: here the total; legs keep their own minimums.
    lo = kwargs.pop("min_tof_days", None)
    hi = kwargs.pop("max_tof_days", None)
    if "tof_total_bounds" not in kwargs and (lo is not None or hi is not None):
        ceiling = (lb.mjd2000_from_date(kwargs["arrive_by"])
                   - lb.mjd2000_from_date(kwargs["launch_window"][0]))
        kwargs["tof_total_bounds"] = (float(lo) if lo is not None else 0.0,
                                      float(hi) if hi is not None else ceiling)
    nseg = kwargs.pop("nseg", 12)
    kwargs["nseg"] = tuple(nseg) if isinstance(nseg, (tuple, list)) else (int(nseg), int(nseg))


def solve_for_config(rc, target, *, seed=None, progress=None, **kwargs) -> FlybySolution:
    """:func:`solve` from a :class:`ResolvedConfig`, the mirror of
    :func:`simsflanagan.solve_for_config`. ``seed`` is a Lambert transfer or a
    ``(departure, arrival)`` MJD2000 pair."""
    from prospector.solvers import simsflanagan as sf
    _config_kwargs(rc, kwargs)
    body = flyby_body(rc.mission.gravity_assist)
    available_power_W, array_model, _stored = sf._pop_power_kwargs(kwargs)
    terms = sf._leg_terms_for_call(rc, (lb.earth_planet(), target), available_power_W,
                                   array_model, kwargs)
    if seed is not None and not isinstance(seed, (tuple, list)):
        seed = (float(seed.dep_mjd2000), float(seed.arr_mjd2000))
    return solve(target, body, mass_kg=rc.cruise_start_mass_kg, seed=seed, progress=progress,
                 **terms, **kwargs)


def solve_cell_for_config(rc, target, *, dep_mjd2000: float, tof_days: float, x0=None,
                          tol_days: float = 0.75, restarts: int = 1, nseg: int = 12,
                          **kwargs) -> FlybySolution:
    """One grid cell, the mirror of :func:`simsflanagan.solve_cell_for_config`: departure date and
    total flight time held, flyby split free, start from the impulsive optimum between the cell's
    days. A neighbour's ``x0`` is used only when its total flight time sits inside this cell's
    pin (a two-leg vector does not re-sample across flight times)."""
    half = max(0.05, float(tol_days))
    dep = lb.date_from_mjd2000(round(float(dep_mjd2000))).date()
    # The cell owns its window and flight time; a one-leg Lambert seed has no two-leg meaning.
    for owned in ("launch_window", "window_slack_days", "min_tof_days", "max_tof_days",
                  "tof_total_bounds", "seed"):
        kwargs.pop(owned, None)
    if x0 is not None:
        x0 = np.asarray(x0, float)
        if abs(float(x0[_TOF1] + x0[_TOF2]) - float(tof_days)) > half:
            x0 = None
    return solve_for_config(
        rc, target,
        launch_window=(dep, dep),
        window_slack_days=half,
        tof_total_bounds=(max(1.0, float(tof_days) - half), float(tof_days) + half),
        seed=(float(dep_mjd2000), float(dep_mjd2000) + float(tof_days)),
        x0=x0, restarts=restarts, nseg=nseg, **kwargs)


def rebuild_for_config(rc, target, x, **kwargs) -> FlybySolution:
    """Rebuild a stored two-leg vector without solving, the mirror of
    :func:`simsflanagan.rebuild_for_config`. The settled leg Isps travel as an ``isp_s`` pair
    (:func:`grid.leg_terms`); without it the starting Isp is used for both. ``seg_caps`` has no
    two-leg form and is ignored; the ceiling is the power model's."""
    from prospector.solvers import simsflanagan as sf
    _config_kwargs(rc, kwargs)
    body = flyby_body(rc.mission.gravity_assist)
    available_power_W, array_model, stored = sf._pop_power_kwargs(kwargs)
    kwargs.pop("seg_caps", None)
    for key in ("thrust_cap_fn", "seg_isp_fn", "restarts", "maxeval", "isp_iters", "x0",
                "seed", "rng_seed", "c_tol", "progress"):
        kwargs.pop(key, None)
    when = lb.mjd2000_from_date(kwargs["launch_window"][0])
    terms = sf._leg_power_terms(rc, (lb.earth_planet(), target), available_power_W, array_model,
                                when_mjd2000=when)
    cap = terms.get("thrust_cap_fn")
    if cap is not None and not isinstance(cap, SmoothCap):
        cap = SmoothCap.from_function(cap)
    thrust_N = float(stored.get("thrust_N", terms["thrust_N"]))
    isp0 = float(terms["isp_s"])
    x = np.asarray(x, float)
    # The problem's own settings with the solver's defaults; other stored options are not the problem's.
    shape = {"vinf_dep_kms": 1.0, "vinf_arr_kms": 0.1, "window_slack_days": 0.0,
             "min_tof_days": (60.0, 120.0), "max_tof_days": None, "max_duty_cycle": 1.0,
             "max_dep_decl_deg": None, "depart_body": None, "min_flyby_alt_km": None,
             "flyby_mu": None, "tof_total_bounds": None, "vinf_dep_exact": False}
    shape.update({k: kwargs[k] for k in shape if k in kwargs})
    shape.update(launch_window=kwargs["launch_window"], arrive_by=kwargs["arrive_by"],
                 nseg=kwargs["nseg"])
    max_duty_cycle = float(shape["max_duty_cycle"])

    def make(isp_pair):
        return _make_udp(target, body, mass_kg=rc.cruise_start_mass_kg, thrust_N=thrust_N,
                         isp_s=isp_pair, cap_fn=cap, **shape)

    stored_isp = stored.get("isp_s")
    isps = ((float(stored_isp[0]), float(stored_isp[1]))
            if isinstance(stored_isp, (tuple, list)) and len(stored_isp) == 2 else (isp0, isp0))
    seg = stored.get("seg_isp_s")
    seg_pair = None
    if seg is not None:
        flat = np.concatenate([np.asarray(part, float).ravel() for part in
                               (seg if isinstance(seg[0], (list, tuple, np.ndarray)) else [seg])])
        n1 = int(shape["nseg"][0])
        seg_pair = (flat[:n1], flat[n1:])
    udp = make(isps)
    lo, hi = (np.asarray(b, float) for b in udp.get_bounds())
    return _build_solution(udp, np.clip(x, lo, hi), getattr(target, "name", "target"),
                           _body_name(body), thrust_N, max_duty_cycle, seg_isp_s=seg_pair)
