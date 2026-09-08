"""Rotational-disruption strength: how much cohesion a spinning rubble pile needs to hold.

Holsapple (2007) treats an asteroid as a self-gravitating triaxial ellipsoid of granular material
obeying a Drucker-Prager yield criterion. Spin pulls it apart, self-gravity holds it together, and
any shortfall must be made up by cohesion between grains. Two questions fall out, and enrichment
asks both:

``min_cohesion_pa``
    Given the measured spin, how much cohesion does this body *require* to exist? A body that
    needs essentially none (a fraction of a pascal) is a coherent, mechanically simple
    rubble pile. One that needs several pascals is either a monolith or bound by something
    other than gravity, which makes it harder to anchor to and harder to mine, so the grade
    drops.

``critical_spin_rad_s``
    Inverting the same relation: at what spin would this body need a given cohesion? Compared
    against the *distribution* of spins for bodies of its size (:mod:`.spin`), this gives the
    probability the body is safely below the disruption limit rather than a single yes/no
    from one uncertain period measurement.

Units throughout: lengths in metres, spin in rad/s, density in kg/m^3, cohesion in pascals,
friction angle in degrees.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate

from prospector.constants import GRAVITATIONAL_CONSTANT_M3_KG_S2, SECONDS_PER_HOUR

# Gravitational constant, N m^2 / kg^2.
_G = GRAVITATIONAL_CONSTANT_M3_KG_S2

# Bulk density assumed for a rubble pile when none is measured (kg/m^3), and the internal friction
# angle of the granular material (degrees). Both are the standard values for this model; a measured
# density would be better but is available for only a handful of bodies.
NOMINAL_DENSITY_KGM3 = 2000.0
NOMINAL_FRICTION_ANGLE_DEG = 32.0


def friction_coefficient(friction_angle_deg: float) -> float:
    """The Drucker-Prager slope ``s`` for a granular material of the given friction angle."""
    phi = np.radians(friction_angle_deg)
    return (2 * np.sin(phi)) / (np.sqrt(3) * (3 - np.sin(phi)))


def _shape_integral(exponent_index: int, alpha: float, beta: float) -> float:
    """One of the three ellipsoid gravity-shape integrals A_x, A_y, A_z.

    ``exponent_index`` selects which semi-axis carries the 3/2 power in the integrand: 0 for x (the
    a-axis), 1 for y (b), 2 for z (c). For a sphere all three come out equal.
    """
    powers = [(1.5, 0.5, 1.5), (0.5, 1.5, 0.5), (0.5, 0.5, 1.5)][exponent_index]

    def integrand(u):
        p1, p2, p3 = powers
        return 1.0 / ((u + 1) ** p1 * (u + beta**2) ** p2 * (u + alpha**2) ** p3)

    value, _error = integrate.quad(integrand, 0, np.inf)
    return alpha * beta * value


def _principal_stresses(a: float, b: float, c: float, omega: float, density: float):
    """The three principal stresses at the centre of a spinning self-gravitating ellipsoid.

    Only the x and y stresses carry the centrifugal term; the spin axis (z) sees gravity alone,
    which is what makes fast rotation a *shear* problem rather than a uniform one.
    """
    alpha, beta = c / a, b / a
    ax = _shape_integral(0, alpha, beta)
    ay = _shape_integral(1, alpha, beta)
    az = _shape_integral(2, alpha, beta)

    gravity = 2 * np.pi * density**2 * _G
    sigma_x = (density * omega**2 - gravity * ax) * a**2 / 5
    sigma_y = (density * omega**2 - gravity * ay) * b**2 / 5
    sigma_z = (-gravity * az) * c**2 / 5
    return sigma_x, sigma_y, sigma_z


def min_cohesion_pa(radius_m: float, period_h: float,
                    density: float = NOMINAL_DENSITY_KGM3,
                    friction_angle_deg: float = NOMINAL_FRICTION_ANGLE_DEG) -> float:
    """Minimum cohesion (Pa) a spherical body of this size must have to survive its spin.

    Never negative: a body whose self-gravity already exceeds the centrifugal demand needs no
    cohesion at all, and the model's negative values there are not a physical "anti-cohesion".
    """
    s = friction_coefficient(friction_angle_deg)
    omega = 2 * np.pi / (period_h * SECONDS_PER_HOUR)
    sigma_x, sigma_y, sigma_z = _principal_stresses(radius_m, radius_m, radius_m,
                                                    omega, density)
    shear = (1 / 6) * ((sigma_x - sigma_y) ** 2 + (sigma_y - sigma_z) ** 2
                       + (sigma_z - sigma_x) ** 2)
    cohesion = np.sqrt(shear) + s * (sigma_x + sigma_y + sigma_z)
    return float(max(cohesion, 0.0))


def critical_spin_rad_s(radius_m: float, cohesion_pa: float = 1.0,
                        density: float = NOMINAL_DENSITY_KGM3,
                        friction_angle_deg: float = NOMINAL_FRICTION_ANGLE_DEG) -> float:
    """The spin (rad/s) at which a body of this size and cohesion reaches the yield limit.

    This inverts :func:`min_cohesion_pa`. Both principal stresses that carry spin do so through
    ``omega**2``, so writing ``u = omega**2`` makes the yield condition a quadratic in ``u`` with a
    closed-form solution, so there is no root-finding and no run-to-run variation.

    Squaring the yield condition to get that quadratic introduces roots of the mirrored equation
    ``sqrt(shear) = -(k - s*trace)``, which are not solutions of the original. They are rejected by
    requiring ``k - s*trace >= 0``. Of the survivors the smallest positive spin is returned:
    required cohesion rises with spin, so that is the first limit a body spinning up would meet.

    Returns 0.0 when the body cannot reach the limit at any spin (already yielding at rest).
    """
    s = friction_coefficient(friction_angle_deg)
    r = radius_m
    alpha = beta = 1.0
    ax = _shape_integral(0, alpha, beta)
    ay = _shape_integral(1, alpha, beta)
    az = _shape_integral(2, alpha, beta)

    # Each principal stress is linear in u = omega**2: sigma = slope * u + offset.
    gravity = 2 * np.pi * density**2 * _G
    x_slope, x_offset = density * r**2 / 5, -gravity * ax * r**2 / 5
    y_slope, y_offset = density * r**2 / 5, -gravity * ay * r**2 / 5
    z_slope, z_offset = 0.0, -gravity * az * r**2 / 5

    # ...so each stress difference and the trace are linear in u too.
    diffs = [(x_slope - y_slope, x_offset - y_offset),
             (y_slope - z_slope, y_offset - z_offset),
             (z_slope - x_slope, z_offset - x_offset)]
    trace_slope = x_slope + y_slope + z_slope
    trace_offset = x_offset + y_offset + z_offset

    # Yield: (1/6) * sum(diff^2) = (k - s*trace)^2, both sides quadratic in u.
    yield_slope, yield_offset = -s * trace_slope, cohesion_pa - s * trace_offset
    quad = (1 / 6) * sum(m * m for m, _ in diffs) - yield_slope**2
    lin = (1 / 3) * sum(m * n for m, n in diffs) - 2 * yield_slope * yield_offset
    const = (1 / 6) * sum(n * n for _, n in diffs) - yield_offset**2

    candidates = np.roots([quad, lin, const]) if quad else (
        [-const / lin] if lin else [])
    spins = []
    for root in candidates:
        u = np.real(root)
        if abs(np.imag(root)) > 1e-9 * max(abs(u), 1.0) or u <= 0:
            continue
        if yield_slope * u + yield_offset < 0:      # root of the mirrored equation
            continue
        spins.append(np.sqrt(u))
    return float(min(spins)) if spins else 0.0


def critical_period_h(radius_m: float, cohesion_pa: float = 1.0,
                      density: float = NOMINAL_DENSITY_KGM3,
                      friction_angle_deg: float = NOMINAL_FRICTION_ANGLE_DEG) -> float:
    """:func:`critical_spin_rad_s` expressed as a rotation period in hours.

    Infinite when the body has no reachable limit, which reads correctly downstream: no finite
    period is short enough to disrupt it.
    """
    omega = critical_spin_rad_s(radius_m, cohesion_pa, density, friction_angle_deg)
    return float((2 * np.pi / omega) / SECONDS_PER_HOUR) if omega > 0 else float("inf")
