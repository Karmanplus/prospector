"""Thrust and Isp the array buys the thrusters against distance from the Sun, for one cruise leg.

Cruise is clear of the belts, so degradation is fixed at the post-escape value and only the
sun-distance factor moves (:class:`prospector.spacecraft.arrays`): at-thruster power is
``base_power x power_fraction(r)`` and the stack runs the operating point that power supports
(:meth:`prospector.config.ResolvedConfig.thrust_isp_at_power`), capped at the rated draw.
Supplies the leg's maximum thrust (at closest approach), the :class:`SmoothCap` ceiling the
optimizer differentiates inside the solve, and the per-segment Isps settled after the fact.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d, minimum_filter1d

# Closest approach assumed when neither body's orbit says otherwise (just inside Earth's).
DEFAULT_R_MIN_AU = 0.98
# Floor on the closest approach, so a perihelion the transfer never visits cannot size the leg.
R_MIN_FLOOR_AU = 0.3
# Ceiling sample points as weights on a segment's (start, midpoint, end) nodes, spread evenly along
# the two chords; the segment's ceiling is the mean over them, in-solve and afterwards alike.
# 12 per half: six let the optimizer park samples on a step's dim side (measured 0.04 of full
# thrust over the true arc mean).
_CHORD_N = 12                                       # samples per half-segment
CHORD_WEIGHTS = np.vstack([
    np.column_stack([1.0 - t, t, np.zeros_like(t)]) for t in [np.linspace(0.0, 1.0, _CHORD_N, endpoint=False)]
] + [
    np.column_stack([np.zeros_like(t), 1.0 - t, t]) for t in [np.linspace(0.0, 1.0, _CHORD_N + 1)]
])


class SmoothCap:
    """A differentiable throttle ceiling against distance from the Sun (fraction of leg thrust).

    The physical ceiling is a staircase (discrete modes) or has corners, so it is sampled finely,
    eroded to its running minimum over ``erode_sigmas`` sigma, Gaussian-smoothed with ``sigma_au``
    and fitted with a clamped cubic spline; outside the sampled range it holds its end value with
    zero slope. Erosion matters: a plain smoothing sits half a step above the true ceiling on the
    dim side and the optimizer parks segments there (measured 0.14 of full thrust over-credit on a
    grid); 3 sigma never over-credits but lost a third of a grid's cells on the bright side; 1
    sigma at 0.01 AU kept every cell within 0.02 of the true arc ceiling.
    """

    def __init__(self, r_au, cap):
        r = np.asarray(r_au, float)
        c = np.clip(np.asarray(cap, float), 0.0, 1.0)
        self.r_lo, self.r_hi = float(r[0]), float(r[-1])
        self._spline = CubicSpline(r, c, bc_type="clamped")
        self._slope = self._spline.derivative()
        # Spline pieces for a direct Horner read: several times cheaper than the spline object's
        # call at hundreds of evaluations per iteration, same value and slope.
        self._knots = np.asarray(self._spline.x, float)
        self._coef = np.asarray(self._spline.c, float)             # (4, n-1), highest power first

    @classmethod
    def from_function(cls, fn, *, r_lo_au: float = 0.2, r_hi_au: float = 8.0, n: int = 4000,
                      sigma_au: float = 0.02, erode_sigmas: float = 3.0) -> SmoothCap:
        """Smooth any ``cap(r_au)`` (scalar or vectorized) into a differentiable ceiling, eroded
        to its running minimum over ``erode_sigmas`` sigma first (3: never over-credits; 0: plain
        smoothing, half a step over on the dim side; 1: a sixth of a step)."""
        r = np.linspace(float(r_lo_au), float(r_hi_au), int(n))
        try:
            raw = np.asarray(fn(r), float)
            if raw.shape != r.shape:
                raise ValueError
        except Exception:  # noqa: BLE001  (a scalar-only function)
            raw = np.asarray([float(fn(float(v))) for v in r], float)
        raw = np.clip(np.nan_to_num(raw, nan=0.0), 0.0, 1.0)
        step = float(r[1] - r[0])
        sig = float(sigma_au) / step
        if sig <= 0:
            return cls(r, raw)
        base = raw
        if erode_sigmas > 0:
            base = minimum_filter1d(raw, size=2 * int(np.ceil(erode_sigmas * sig)) + 1, mode="nearest")
        return cls(r, gaussian_filter1d(base, sig, mode="nearest"))

    @classmethod
    def constant(cls, value: float) -> SmoothCap:
        v = float(np.clip(value, 0.0, 1.0))
        r = np.linspace(0.2, 8.0, 8)
        return cls(r, np.full(r.shape, v))

    def with_derivative(self, r_au):
        """``(cap, dcap/dr)`` at ``r_au`` (elementwise), the slope in 1/AU."""
        r = np.asarray(r_au, float)
        rc = np.clip(r, self.r_lo, self.r_hi)
        idx = np.clip(np.searchsorted(self._knots, rc, side="right") - 1, 0, self._knots.size - 2)
        dx = rc - self._knots[idx]
        c0, c1, c2, c3 = self._coef[:, idx]
        raw = ((c0 * dx + c1) * dx + c2) * dx + c3
        cap = np.clip(raw, 0.0, 1.0)
        inside = (r > self.r_lo) & (r < self.r_hi) & (raw > 0.0) & (raw < 1.0)
        slope = np.where(inside, (3.0 * c0 * dx + 2.0 * c1) * dx + c2, 0.0)
        return cap, slope

    def __call__(self, r_au):
        return self.with_derivative(r_au)[0]


class SunPowerModel:
    """Thrust and Isp against distance from the Sun for a resolved config's thruster stack.

    ``base_power_W`` is the at-thruster power at 1 AU. Construct via :func:`sun_power_model`,
    which returns None when there is nothing to limit.
    """

    def __init__(self, rc, base_power_W: float, array_model, *, chain_eff: float = 1.0,
                 reserve_W: float = 0.0):
        self.rc = rc
        self.base_power_W = float(base_power_W)
        self.array_model = array_model
        # The housekeeping reserve is a fixed draw: taken off the array output at each distance,
        # before the chain, not off ``base_power``.
        self.chain_eff = max(1e-9, float(chain_eff))
        self.reserve_W = max(0.0, float(reserve_W))
        self.grid = rc.power_thrust_isp_grid()
        self.rated_thrust_N = float(rc.total_thrust_mN) * 1e-3
        self.rated_isp_s = float(rc.effective_isp)

    def power_W(self, r_au) -> np.ndarray:
        """Power reaching the thrusters at ``r_au``, uncapped."""
        r = np.asarray(r_au, float)
        array_1au = self.base_power_W / self.chain_eff + self.reserve_W
        array_r = array_1au * np.asarray(self.array_model.power_fraction(r), float)
        return np.maximum(array_r - self.reserve_W, 0.0) * self.chain_eff

    def thrust_isp(self, r_au) -> tuple[np.ndarray, np.ndarray]:
        """The stack's (thrust N, Isp s) at ``r_au``, elementwise; power is capped at the rated draw."""
        p = self.power_W(r_au)
        scalar = np.ndim(p) == 0
        p = np.atleast_1d(p)
        if self.grid is not None:
            pw, thrust, isp = self.grid
            capped = np.minimum(p, pw[-1])
            t, i = np.interp(capped, pw, thrust), np.interp(capped, pw, isp)
        else:
            pairs = [self.rc.thrust_isp_at_power(float(v)) for v in p]
            t = np.asarray([a for a, _ in pairs], float)
            i = np.asarray([b for _, b in pairs], float)
        return (t[0], i[0]) if scalar else (t, i)

    def thrust_N(self, r_au) -> np.ndarray:
        return self.thrust_isp(r_au)[0]

    def isp_s(self, r_au) -> np.ndarray:
        return self.thrust_isp(r_au)[1]

    def leg_thrust_N(self, r_min_au: float | None) -> float:
        """Leg maximum thrust: the operating point at its closest approach."""
        r = DEFAULT_R_MIN_AU if r_min_au is None else float(r_min_au)
        r = min(DEFAULT_R_MIN_AU, max(R_MIN_FLOOR_AU, r))
        return float(self.thrust_N(r))

    def at_one_au(self) -> tuple[float, float]:
        """The operating point at 1 AU, the first round's starting point."""
        t, i = self.thrust_isp(1.0)
        return float(t), float(i)

    def smooth_cap(self, leg_thrust_N: float) -> SmoothCap:
        """The :class:`SmoothCap` ceiling as a fraction of ``leg_thrust_N`` (1 sigma of 0.01 AU);
        zero everywhere when the leg thrust is zero."""
        if leg_thrust_N <= 0.0:
            return SmoothCap.constant(0.0)
        return SmoothCap.from_function(
            lambda r: np.minimum(np.asarray(self.thrust_N(r), float) / float(leg_thrust_N), 1.0),
            sigma_au=0.01, erode_sigmas=1.0)

    def segment_isps(self, node_positions_m: np.ndarray, leg_thrust_N: float) -> np.ndarray:
        """The Isp each segment runs at, from :meth:`segment_terms`."""
        return self.segment_terms(node_positions_m, leg_thrust_N)[1]

    def segment_terms(self, node_positions_m: np.ndarray, leg_thrust_N: float
                      ) -> tuple[np.ndarray, np.ndarray]:
        """Per-segment throttle ceilings and Isps for ``node_positions_m`` ((2*nseg+1, 3), metres).

        Each segment is sampled at the :data:`CHORD_WEIGHTS` points: the ceiling is the mean
        available thrust over them as a fraction of ``leg_thrust_N``, the Isp the thrust-weighted
        mean of the samples' Isps. The solver constrains the same mean at the same points.
        """
        pos = np.asarray(node_positions_m, float)
        nseg = (len(pos) - 1) // 2
        caps = np.zeros(nseg)
        isps = np.zeros(nseg)
        au = 1.495978707e11
        for i in range(nseg):
            pts = CHORD_WEIGHTS @ pos[2 * i:2 * i + 3]
            r_au = np.linalg.norm(pts, axis=1) / au
            t, isp = self.thrust_isp(r_au)
            caps[i] = float(np.mean(t)) / leg_thrust_N if leg_thrust_N > 0 else 0.0
            fired = t > 0.0
            if np.any(fired):
                isps[i] = float(np.sum(t[fired]) / np.sum(t[fired] / isp[fired]))
            else:
                isps[i] = float(self.thrust_isp(float(r_au[len(r_au) // 2]))[1])
        return np.clip(caps, 0.0, 1.0), isps


def effective_isp(throttles: np.ndarray, seg_isp_s: np.ndarray, fallback: float) -> float:
    """The one Isp that spends the propellant the segments really would: the throttle-weighted
    harmonic mean (each segment's propellant is proportional to ``|u_i| / Isp_i``). ``fallback``
    when nothing fires."""
    u = np.clip(np.asarray(throttles, float), 0.0, None)
    isp = np.asarray(seg_isp_s, float)
    ok = (u > 1e-9) & (isp > 0.0)
    if not np.any(ok):
        return float(fallback)
    return float(np.sum(u[ok]) / np.sum(u[ok] / isp[ok]))


def sun_power_model(rc, available_power_W: float | None = None, array_model=None
                    ) -> SunPowerModel | None:
    """The :class:`SunPowerModel` for a config, or None when there is nothing to limit (no power
    draw, or no array modelled)."""
    if rc.total_power_W <= 0.0:
        return None
    if array_model is None:
        array_model = rc.bus_model().array_model()
    chain_eff = rc.thruster_chain_eff()
    reserve = rc.array_reserve_W()
    base_power = (float(available_power_W) if available_power_W is not None
                  else max(0.0, rc.vehicle.solar_power_W - reserve) * chain_eff)
    if base_power <= 0.0 and rc.vehicle.solar_power_W <= 0.0:
        return None                                  # no array modelled: power-rich, nothing to limit
    return SunPowerModel(rc, base_power, array_model, chain_eff=chain_eff, reserve_W=reserve)
