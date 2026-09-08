"""Tests for the Holsapple (2007) rotational-strength model.

The critical-spin inversion is the delicate part: it solves the yield condition in closed form
after squaring it, which introduces roots of the mirrored equation. The tests below pin the
numbers, check the branch selection directly, and, most usefully, close the loop by running the
forward and inverse models against each other, which no algebra slip survives.

The pinned values come from an independent symbolic solve of the same equation, so they check the
algebra rather than merely freezing whatever this implementation happens to produce.
"""
import numpy as np
import pytest

from prospector.enrichment.models import cohesion

# Reference critical spins (rad/s) from a symbolic solve of the yield condition, for a spherical
# body of the given radius (m) at the given cohesion (Pa) and density (kg/m^3).
_CRITICAL_SPIN_REFERENCE = [
    # radius, cohesion, density, omega
    (1.0, 0.0, 2000.0, 0.0004993135068862611),
    (100.0, 0.0, 2000.0, 0.0004993135068862615),
    (100.0, 1.0, 2000.0, 0.000701594374851089),
    (100.0, 5.0, 2000.0, 0.0011975703795682178),
    (500.0, 1.0, 2000.0, 0.0005091724324931322),
    (500.0, 100.0, 3000.0, 0.0010078541430409728),
    (5000.0, 5.0, 1200.0, 0.00038783762122150087),
]


@pytest.mark.parametrize("radius,cohesion_pa,density,expected", _CRITICAL_SPIN_REFERENCE)
def test_critical_spin_matches_the_symbolic_solution(radius, cohesion_pa, density, expected):
    got = cohesion.critical_spin_rad_s(radius, cohesion_pa, density)
    assert got == pytest.approx(expected, rel=1e-9)


def test_forward_and_inverse_models_agree():
    """The spin the inverse model returns for a cohesion must require exactly that cohesion.

    This is the check that catches a wrong root: a solution of the mirrored equation would satisfy
    the squared form and fail here.
    """
    for radius in (25.0, 250.0, 2500.0):
        for cohesion_pa in (0.5, 1.0, 4.0, 20.0):
            omega = cohesion.critical_spin_rad_s(radius, cohesion_pa)
            period_h = (2 * np.pi / omega) / 3600.0
            assert cohesion.min_cohesion_pa(radius, period_h) == pytest.approx(cohesion_pa,
                                                                              rel=1e-8)


def test_required_cohesion_rises_as_a_body_spins_faster():
    radius = 200.0
    demands = [cohesion.min_cohesion_pa(radius, period) for period in (24.0, 8.0, 4.0, 2.0, 1.0)]
    assert demands == sorted(demands), "a faster spin cannot need less cohesion"


def test_a_slow_rotator_needs_no_cohesion():
    # At a long period self-gravity dominates: the model's negative "requirement" is not an
    # anti-cohesion, and reporting it as one would put the body in the best tier for the wrong
    # reason.
    assert cohesion.min_cohesion_pa(200.0, 100.0) == 0.0


def test_a_bigger_body_reaches_its_limit_at_a_slower_spin():
    spins = [cohesion.critical_spin_rad_s(radius, 1.0) for radius in (50.0, 500.0, 5000.0)]
    assert spins == sorted(spins, reverse=True)


def test_critical_period_is_the_spin_expressed_in_hours():
    radius = 300.0
    omega = cohesion.critical_spin_rad_s(radius, 1.0)
    assert cohesion.critical_period_h(radius, 1.0) == pytest.approx((2 * np.pi / omega) / 3600.0)


def test_an_unreachable_limit_reads_as_an_infinite_period():
    # Cohesion cannot be negative, so no spin satisfies the yield condition and the model has no
    # limit to report. "inf" reads correctly downstream; no finite period is short enough to
    # disrupt the body, where a NaN would poison the stability margin instead.
    assert cohesion.critical_spin_rad_s(100.0, -1.0) == 0.0
    assert cohesion.critical_period_h(100.0, -1.0) == float("inf")


def test_friction_coefficient_is_zero_for_a_frictionless_material():
    assert cohesion.friction_coefficient(0.0) == pytest.approx(0.0)
    # The nominal 32-degree granular material, the value every call here defaults to.
    assert cohesion.friction_coefficient(32.0) == pytest.approx(0.24772, abs=1e-5)
