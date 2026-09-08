"""
Delta-v estimates from orbital elements alone, with no dates and no positions involved.

A cheap estimate (km/s) of what it costs a low-thrust spacecraft to get from Earth's orbit onto a
target orbit (a, e, i). Elements only, so the whole small-body population can be worked out at
once. Good for ranking and for a rough yes/no; never a trajectory anyone could fly.

:func:`lowthrust_dv` takes the smaller of two closed-form answers that bracket the real manoeuvre
from either side: :func:`spiral_dv`, a continuous arc the vehicle could more or less fly, which
runs high, and :func:`intercept_dv`, a two-burn transfer it cannot fly, which runs low. The
realistic answer is somewhere between, and taking the smaller errs toward keeping targets. The
two-burn term is what rescues eccentric, Earth-approaching orbits that the spiral charges far too
much for. ``docs/physics.md`` covers why they disagree and where each comes from.

Speeds in km/s, a in AU, i in degrees. Earth's orbit (a=1, e=0.0167, i=0) is where every transfer
starts. Everything here is vectorized: pass scalars or arrays.
"""
import numpy as np
import pandas as pd

from prospector.constants import (
    AU_KM,
    EARTH_ORBIT_ECC,
    EARTH_ORBIT_INC_DEG,
    EARTH_ORBIT_SMA_AU,
    MU_SUN_KM3_S2,
)


def circular_speed(a_au):
    """Circular orbital speed (km/s) at semimajor axis a (AU)."""
    return np.sqrt(MU_SUN_KM3_S2 / (np.asarray(a_au, dtype=float) * AU_KM))


def edelbaum_dv(a0, i0, af, i_f):
    """Edelbaum's combined size-change and tilt cost (km/s) between two circular orbits.

    dV = sqrt(v0^2 + vf^2 - 2 v0 vf cos(pi/2 * di)), where v = sqrt(mu/a) is the circular speed
    and di is the change in inclination. Exact for circular orbits, and the main term in
    :func:`spiral_dv`. The pi/2 factor is what makes it a continuous tilt rather than a single
    burn: spreading the tilt around the whole orbit costs about 1.57x more per degree than one
    well-placed burn would.
    """
    v0 = circular_speed(a0)
    vf = circular_speed(af)
    di = np.radians(np.abs(np.asarray(i_f, float) - np.asarray(i0, float)))
    return np.sqrt(v0 ** 2 + vf ** 2 - 2 * v0 * vf * np.cos(np.pi / 2 * di))


def eccentricity_dv(af, e0, ef):
    """Rough continuous-thrust cost (km/s) of changing eccentricity by |ef - e0|.

    The spread between an orbit's fastest and slowest speeds goes roughly as v_c * e, so stretching
    eccentricity by |de| costs about v_c(af) * |de|. This is what the spiral pays to build
    eccentricity it does not already have, and the reason it charges too much for eccentric orbits.
    It is an approximation: doing it properly couples with a and needs a numerical solve.
    """
    return circular_speed(af) * np.abs(np.asarray(ef, float) - np.asarray(e0, float))


def spiral_dv(a, e, i, a0=EARTH_ORBIT_SMA_AU, e0=EARTH_ORBIT_ECC, i0=EARTH_ORBIT_INC_DEG):
    """Cost (km/s) of spiralling Earth's orbit into the target's.

    Thrusting continuously over many revolutions to change the orbit's size and tilt
    (:func:`edelbaum_dv`) while stretching its eccentricity (:func:`eccentricity_dv`), the two
    added in quadrature. This is close to something the vehicle could really fly, so it runs
    high. It charges far too much for eccentric orbits, though, which is what
    :func:`intercept_dv` is there to fix. Vectorized."""
    dv_shape = edelbaum_dv(a0, i0, a, i)
    dv_ecc = eccentricity_dv(a, e0, e)
    return np.sqrt(dv_shape ** 2 + dv_ecc ** 2)


def intercept_dv(a, e, i, a0=EARTH_ORBIT_SMA_AU, i0=EARTH_ORBIT_INC_DEG):
    """Cost (km/s) of coasting out to the cheapest meeting point on the target orbit and
    burning once to match it there.

    The burns are instant, so a low-thrust vehicle cannot fly this. It is kept as an optimistic low
    end that rescues the eccentric targets :func:`spiral_dv` charges too much for. Timing is
    assumed free: the launch date is whatever gets the spacecraft and the target to the meeting
    point together.

    Three meeting points are tried and the cheapest kept: the target's closest approach to the
    Sun, its furthest, and, where the orbit crosses it, the 1 AU radius. Each leg out to a
    turning point R is a half-ellipse (a_t = (a0 + R)/2), and the tilt is paid for on the match
    burn, where the spacecraft is often moving slowly. ``docs/physics.md`` covers why the
    turning points usually win. Vectorized."""
    a = np.asarray(a, float)
    e = np.asarray(e, float)
    cdi = np.cos(np.radians(np.abs(np.asarray(i, float) - i0)))
    V1 = circular_speed(1.0)                        # vis-viva scale; ~29.78 km/s
    vc0 = circular_speed(a0)                         # departure circular speed at a0
    q = a * (1 - e)                                  # perihelion (AU)
    Q = a * (1 + e)                                  # aphelion (AU)

    def _speed_at(r, sma):
        """Vis-viva speed (km/s) at radius r (AU) on an orbit of semimajor axis sma."""
        return V1 * np.sqrt(np.clip(2.0 / r - 1.0 / sma, 0.0, None))

    def _meet_at_apse(R):
        """Cost of coasting from a0 out to a turning point R on the target orbit and matching
        the target there. At a turning point the target is moving straight across the line to
        the Sun, so the match is just a difference in speed plus a cheap, slow-speed tilt."""
        a_t = 0.5 * (a0 + R)                         # Hohmann transfer semimajor axis
        dv_depart = np.abs(_speed_at(a0, a_t) - vc0)  # burn 1: raise/lower an apse to R
        v_transfer = _speed_at(R, a_t)               # arrival speed at R (tangential)
        v_target = _speed_at(R, a)                   # target speed at R (tangential)
        dv_match = np.sqrt(np.clip(                  # burn 2: match speed + plane at R
            v_transfer ** 2 + v_target ** 2 - 2 * v_transfer * v_target * cdi, 0.0, None))
        return dv_depart + dv_match

    cost_perihelion = _meet_at_apse(q)
    cost_aphelion = _meet_at_apse(Q)

    # Match at the 1 AU crossing, which only means anything where the orbit reaches a0.
    crosses = (q <= a0) & (a0 <= Q)
    v_target_at_a0 = _speed_at(a0, a)                # target speed where it crosses a0
    v_target_tangential = V1 * np.sqrt(np.clip(a * (1 - e ** 2), 0.0, None)) / a0
    dv_crossing = np.sqrt(np.clip(
        v_target_at_a0 ** 2 + vc0 ** 2 - 2 * vc0 * v_target_tangential * cdi, 0.0, None))
    cost_crossing = np.where(crosses, dv_crossing, np.inf)

    return np.minimum(np.minimum(cost_perihelion, cost_aphelion), cost_crossing)


def lowthrust_dv(a, e, i, a0=EARTH_ORBIT_SMA_AU, e0=EARTH_ORBIT_ECC, i0=EARTH_ORBIT_INC_DEG):
    """Estimated low-thrust delta-v (km/s) from Earth's orbit to a target orbit (a, e, i).

    The smaller of :func:`spiral_dv`, which runs high, and :func:`intercept_dv`, which runs low.
    The realistic cost is between them, and taking the smaller errs toward calling a target
    reachable, which is the safe direction for a screen because the real solver has the final
    say. Vectorized."""
    return np.minimum(spiral_dv(a, e, i, a0=a0, e0=e0, i0=i0),
                      intercept_dv(a, e, i, a0=a0, i0=i0))


def evaluate_dataframe(df, dv_budget=None, a_col="a", e_col="e", i_col="i"):
    """Add a 'lowthrust_dv' column to a DataFrame of candidates, from its (a, e, i) columns.

    Given ``dv_budget`` (km/s), also adds a boolean 'reachable' column. Rows whose elements do not
    parse get NaN and reachable=False.
    """
    a = pd.to_numeric(df[a_col], errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(df[e_col], errors="coerce").to_numpy(dtype=float)
    i = pd.to_numeric(df[i_col], errors="coerce").to_numpy(dtype=float)
    dv = lowthrust_dv(a, e, i)
    out = df.copy()
    out["lowthrust_dv"] = dv
    if dv_budget is not None:
        out["reachable"] = np.isfinite(dv) & (dv <= dv_budget)
    return out
