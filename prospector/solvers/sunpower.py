"""What the array buys the thrusters at each distance from the Sun, for one cruise leg.

The cruise is clear of the radiation belts, so the array's permanent degradation is whatever the
escape left it with, and only the sun-distance factor still moves: sunlight falls off as 1/r^2,
partly offset by colder cells running more efficiently (:class:`prospector.spacecraft.arrays`).
So the power reaching the thrusters at a distance ``r`` is ``base_power x power_fraction(r)``, where
``base_power`` is the at-thruster power at 1 AU (the post-escape figure when a spiral flew, else the
start-of-life array output through the power chain), and the thrusters then run the operating point
that power supports on their throttle curve (:meth:`prospector.config.ResolvedConfig.thrust_isp_at_power`).

Inside 1 AU the array makes MORE than at the escape, and this model credits it, capped at the
stack's rated draw where extra sunlight buys nothing. Before, the cruise was frozen at the
post-escape point everywhere and only ever throttled down from it. That was mildly pessimistic with
a smooth throttle curve and a whole mode pessimistic with a discrete one: a thruster left in its
3 kW mode on a 0.9 AU arc where the array would run its 4.5 kW mode.

The leg's optimizer (:mod:`prospector.solvers.simsflanagan`) carries one maximum thrust and one Isp
per leg, and a throttle ceiling per segment that follows the segment's distance from the Sun. This
model supplies all three: the leg thrust is what the array buys at the closest approach the leg can
make; the ceiling is :class:`SmoothCap`, the available thrust against distance as a smooth curve
the optimizer can differentiate, applied inside the solve at each segment's midpoint; and the Isp
is the thrust-weighted mean of the segments' Isps, settled once the trajectory is known.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d, minimum_filter1d

# The closest a cruise is assumed able to get to the Sun when neither body's orbit says otherwise:
# just inside Earth's, since a transfer to a near-Earth target rarely dips much further in.
DEFAULT_R_MIN_AU = 0.98
# A floor under the closest approach, so a target on an extreme orbit cannot size the leg's thrust
# off a perihelion the transfer never visits.
R_MIN_FLOOR_AU = 0.3
# Where along a segment the ceiling is sampled, as weights on its (start, midpoint, end) nodes:
# points spread evenly along the two straight chords. The segment's ceiling is the mean over them,
# both inside the solve and when a path is judged afterwards, so the two agree. Dense on purpose:
# the smooth ceiling sits above a mode step on its dim side and below it on the bright side, and
# only a mean over the whole arc lets those cancel, as they do in the true integral. Six samples
# left an optimizer room to park samples on the dim side (measured 0.04 of full thrust over the
# true arc mean); a few dozen leave only the arc's end effects.
_CHORD_N = 12                                       # samples per half-segment
CHORD_WEIGHTS = np.vstack([
    np.column_stack([1.0 - t, t, np.zeros_like(t)]) for t in [np.linspace(0.0, 1.0, _CHORD_N, endpoint=False)]
] + [
    np.column_stack([np.zeros_like(t), 1.0 - t, t]) for t in [np.linspace(0.0, 1.0, _CHORD_N + 1)]
])


class SmoothCap:
    """A throttle ceiling against distance from the Sun that the optimizer can differentiate.

    The physical ceiling is what the array buys the thrusters at a distance, as a fraction of the
    leg's thrust. On a discrete-mode stack that is a staircase in ``r``, and even a continuous
    throttle curve has corners, neither of which a gradient method can follow when the constraint
    sits inside the solve. So the ceiling is sampled finely over the distances a cruise can reach,
    smoothed with a Gaussian of ``sigma_au`` and fitted with a cubic spline. Calls return the value
    and, from :meth:`with_derivative`, the slope in ``1/AU``. Outside the sampled range the ceiling
    holds its end value with zero slope.

    Before the Gaussian, the ceiling is eroded to its running minimum over ``erode_sigmas`` sigma
    either side. A plain smoothing of a step sits half a step above it on the dim side, and an
    optimizer given that curve finds it: measured on a grid, every cell parked segments just past a
    mode step and drew power the array would not make there, by 0.14 of full thrust on average,
    even averaged over a segment's arc. Eroding by three sigma makes a curve that never exceeds the
    true one but costs a whole step within three sigma on the bright side, and a grid lost a third
    of its cells to that. The model uses one sigma of 0.01 AU: a sixth of a step of over-credit on
    the dim side, a fraction of a step lost on the bright side, both within 0.01 AU of a step. On
    the same grid the answers then sat within 0.02 of the true arc ceiling everywhere (two cells at
    0.023, none above), which is the tolerance a constant-thrust segment was always judged by.
    """

    def __init__(self, r_au, cap):
        r = np.asarray(r_au, float)
        c = np.clip(np.asarray(cap, float), 0.0, 1.0)
        self.r_lo, self.r_hi = float(r[0]), float(r[-1])
        self._spline = CubicSpline(r, c, bc_type="clamped")
        self._slope = self._spline.derivative()
        # The spline's pieces, kept for a direct read: a solve evaluates the ceiling at hundreds
        # of points per iteration, and Horner on the coefficients is several times cheaper than
        # the spline object's own call while giving exactly its value and slope.
        self._knots = np.asarray(self._spline.x, float)
        self._coef = np.asarray(self._spline.c, float)             # (4, n-1), highest power first

    @classmethod
    def from_function(cls, fn, *, r_lo_au: float = 0.2, r_hi_au: float = 8.0, n: int = 4000,
                      sigma_au: float = 0.02, erode_sigmas: float = 3.0) -> SmoothCap:
        """Smooth any ``cap(r_au)`` (scalar or vectorized) into a differentiable ceiling. It is
        eroded to its running minimum over ``erode_sigmas`` sigma either side first: 3 makes a
        curve that never exceeds the true one, 0 a plain smoothing that sits half a step above
        it on the dim side of each step, 1 leaves a sixth of a step there."""
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

    ``available_power_W`` is the at-thruster power at 1 AU; None means the start-of-life array
    output through the power chain. ``array_model=None`` loads the application's build model.
    Use :func:`sun_power_model` to construct, which returns None when there is nothing to limit.
    """

    def __init__(self, rc, base_power_W: float, array_model, *, chain_eff: float = 1.0,
                 reserve_W: float = 0.0):
        self.rc = rc
        self.base_power_W = float(base_power_W)
        self.array_model = array_model
        # The chain between array and thrusters, and the array output the bus keeps for its own
        # housekeeping. The reserve is a fixed draw: it does not shrink or grow with the Sun, so it
        # is taken off the array's output at each distance before the chain, not off ``base_power``.
        self.chain_eff = max(1e-9, float(chain_eff))
        self.reserve_W = max(0.0, float(reserve_W))
        self.grid = rc.power_thrust_isp_grid()
        self.rated_thrust_N = float(rc.total_thrust_mN) * 1e-3
        self.rated_isp_s = float(rc.effective_isp)

    def power_W(self, r_au) -> np.ndarray:
        """The power reaching the thrusters at ``r_au`` (uncapped: above the 1 AU figure inside
        1 AU, below it outside)."""
        r = np.asarray(r_au, float)
        array_1au = self.base_power_W / self.chain_eff + self.reserve_W
        array_r = array_1au * np.asarray(self.array_model.power_fraction(r), float)
        return np.maximum(array_r - self.reserve_W, 0.0) * self.chain_eff

    def thrust_isp(self, r_au) -> tuple[np.ndarray, np.ndarray]:
        """The stack's (thrust N, Isp s) operating point at ``r_au``, elementwise. Power above the
        rated draw buys nothing, so it is capped at the top of the throttle curve."""
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
        """The most thrust the leg can ever use: the operating point at its closest approach."""
        r = DEFAULT_R_MIN_AU if r_min_au is None else float(r_min_au)
        r = min(DEFAULT_R_MIN_AU, max(R_MIN_FLOOR_AU, r))
        return float(self.thrust_N(r))

    def at_one_au(self) -> tuple[float, float]:
        """The operating point at 1 AU: the frozen post-escape point the cruise used to fly at,
        and still the first round's starting point."""
        t, i = self.thrust_isp(1.0)
        return float(t), float(i)

    def smooth_cap(self, leg_thrust_N: float) -> SmoothCap:
        """The throttle ceiling against distance, as a fraction of ``leg_thrust_N``, smoothed for
        the optimizer (:class:`SmoothCap`, one sigma of 0.01 AU, see there for why). Zero
        everywhere when the leg thrust is zero."""
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
        """Per-segment throttle ceilings and Isps for a trajectory's node positions.

        ``node_positions_m`` is the (2*nseg+1, 3) array of segment start / midpoint / end
        positions (metres). Each segment is sampled at the :data:`CHORD_WEIGHTS` points along the
        chords between its nodes; its ceiling is the mean available thrust over those samples as a
        fraction of ``leg_thrust_N``, so a segment that crosses a mode boundary (or the floor)
        part-way gets the impulse it really has rather than the value at one instant. Its Isp is
        the thrust-weighted mean of the samples' Isps: the mass flow that thrust implies. The
        solver constrains the same mean, of its smooth ceiling, at the same points.
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
    """The one Isp a fixed-Isp leg needs to spend the propellant its segments really would.

    Each segment fires at throttle ``|u_i|`` for the same duration, so its propellant is
    proportional to ``|u_i| / Isp_i``; the leg Isp that reproduces the total is the throttle-
    weighted harmonic mean. ``fallback`` is returned when nothing fires (a coast, or a first
    guess with zero throttle), where the choice cannot matter.
    """
    u = np.clip(np.asarray(throttles, float), 0.0, None)
    isp = np.asarray(seg_isp_s, float)
    ok = (u > 1e-9) & (isp > 0.0)
    if not np.any(ok):
        return float(fallback)
    return float(np.sum(u[ok]) / np.sum(u[ok] / isp[ok]))


def sun_power_model(rc, available_power_W: float | None = None, array_model=None
                    ) -> SunPowerModel | None:
    """The :class:`SunPowerModel` for a config, or None when the stack draws no power or the
    array delivers none, where there is nothing to limit and the rated point stands."""
    if rc.total_power_W <= 0.0:
        return None
    if array_model is None:
        from prospector.spacecraft.buildability import load_bus_model
        array_model = load_bus_model().array_model()
    chain_eff = rc.thruster_chain_eff()
    reserve = rc.array_reserve_W()
    base_power = (float(available_power_W) if available_power_W is not None
                  else max(0.0, rc.vehicle.solar_power_W - reserve) * chain_eff)
    if base_power <= 0.0 and rc.vehicle.solar_power_W <= 0.0:
        return None                                  # no array modelled: power-rich, nothing to limit
    return SunPowerModel(rc, base_power, array_model, chain_eff=chain_eff, reserve_W=reserve)
