"""
Two-burn transfers between real dates: launch-window timing, and a starting point for the solver.

Solves Izzo's Lambert problem on every (departure, arrival) cell of a date grid and records what
the transfer costs, which is a porkchop plot. Unlike the date-free estimate in ``edelbaum``, this
puts the real positions of the bodies back in, so it can say *when* the alignment is good.

It may rank and it may supply a starting point; it may never reject. The burns are instant, so the
cost comes out low, and the window shape is wrong too: a continuously steered many-revolution trip
cares far less about exact timing than a short coasting one, so the real low-thrust window is wider
and sits elsewhere. The ``dv`` here is not comparable to the vehicle's budget. See
``docs/physics.md``.

Earth comes from PyKEP's fitted ``jpl_lp`` series, good to about an arcminute over 1800-2050; a
small body is propagated forward from its elements with ``pk.udpla.keplerian``.

Delta-v in km/s, a in AU, angles in degrees, flight time in days. Dates are MJD2000, with helpers
to convert to and from calendar dates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pykep as pk

MU_SUN = pk.MU_SUN          # m^3/s^2 (heliocentric)
_DEG2RAD = pk.DEG2RAD
_AU_M = pk.AU               # m
_DAY2SEC = pk.DAY2SEC       # s
_MJD2000_EPOCH = datetime(2000, 1, 1)   # MJD2000 = 0


# ---------------------------------------------------------------------------
# date <-> MJD2000
# ---------------------------------------------------------------------------

def mjd2000_from_date(d: date) -> float:
    """Calendar date -> MJD2000 (days since 2000-01-01 00:00)."""
    dt = datetime(d.year, d.month, d.day)
    return (dt - _MJD2000_EPOCH).total_seconds() / _DAY2SEC


def date_from_mjd2000(mjd2000: float) -> datetime:
    """MJD2000 -> datetime."""
    return _MJD2000_EPOCH + timedelta(days=float(mjd2000))


# ---------------------------------------------------------------------------
# bodies
# ---------------------------------------------------------------------------

_EARTH: object | None = None


def earth_planet():
    """Earth as PyKEP's low-precision analytic ephemeris (the departure body)."""
    global _EARTH
    if _EARTH is None:
        _EARTH = pk.planet(pk.udpla.jpl_lp("earth"))
    return _EARTH


def target_planet(a_au, e, i_deg, om_deg, w_deg, ma_deg, epoch_jd, name="target"):
    """Build a PyKEP body from SBDB orbital elements.

    The elements (a, e, i, Omega, omega, M) are quoted for ``epoch_jd``, a Julian date in TDB.

    The element type is spelled out because SBDB gives the last angle as a mean anomaly while
    ``pk.udpla.keplerian`` reads its sixth element as a true anomaly unless told otherwise, and it
    accepts either without complaint. Getting it wrong does not raise; it puts the body elsewhere
    on the same orbit, which for a rendezvous is the whole problem (a measured 17-107% position
    error on one target).
    """
    elements = (
        float(a_au) * _AU_M,
        float(e),
        float(i_deg) * _DEG2RAD,
        float(om_deg) * _DEG2RAD,
        float(w_deg) * _DEG2RAD,
        float(ma_deg) * _DEG2RAD,
    )
    ref_epoch = pk.epoch(float(epoch_jd), pk.epoch.julian_type.JD)
    # mu_self, radius and safe_radius do not affect where a massless body is; -1 marks each of the
    # three as unknown.
    udpla = pk.udpla.keplerian(ref_epoch, list(elements), MU_SUN, str(name),
                               [-1.0, -1.0, -1.0], pk.el_type.KEP_M)
    return pk.planet(udpla)


def planet_from_row(row) -> object:
    """Build the body a population row describes.

    A planet uses PyKEP's ``jpl_lp`` fitted series, the same one Earth uses. A small body is
    propagated forward from its elements (:func:`target_planet`), which is all SBDB provides.

    The difference matters: propagating a planet forward over a trip years long drifts it along its
    orbit, and the perturbations behind that drift are what the fitted series already accounts for.
    Meeting a planet is a timing problem, so the drift lands on the arrival date. Small bodies have
    no such series, so their elements are re-fetched for the mission date
    (:func:`prospector.population.refresh_target_elements`).
    """
    from prospector.population.planets import jpl_lp_key

    key = jpl_lp_key(row)
    if key:
        return pk.planet(pk.udpla.jpl_lp(key))
    name = row.get("full_name") or row.get("pdes") or "target"
    return target_planet(row["a"], row["e"], row["i"], row["om"], row["w"], row["ma"],
                          row["epoch"], name=str(name))


# ---------------------------------------------------------------------------
# a single transfer, and the porkchop over a date grid
# ---------------------------------------------------------------------------

@dataclass
class LambertSolution:
    """One two-burn transfer: the cheapest cell of a porkchop, and the solver's starting point.

    Carries the cost, for ranking, and the geometry at each end, for starting the low-thrust
    solve: where each body is and how fast it is going at departure and arrival, plus the
    transfer's own velocities. Vectors are SI (m, m/s); scalars are km/s, km^2/s^2, days and
    MJD2000.
    """
    target_name: str
    dep_mjd2000: float
    arr_mjd2000: float
    tof_days: float
    dv_kms: float
    vinf_dep_kms: float
    vinf_arr_kms: float
    c3_km2s2: float
    revs: int
    r_dep_m: np.ndarray            # heliocentric position at departure (Earth)
    v_dep_body_ms: np.ndarray      # Earth velocity at departure
    r_arr_m: np.ndarray            # heliocentric position at arrival (target)
    v_arr_body_ms: np.ndarray      # target velocity at arrival
    v_transfer_dep_ms: np.ndarray  # transfer-orbit velocity at departure
    v_transfer_arr_ms: np.ndarray  # transfer-orbit velocity at arrival

    @property
    def dep_date(self) -> datetime:
        return date_from_mjd2000(self.dep_mjd2000)

    @property
    def arr_date(self) -> datetime:
        return date_from_mjd2000(self.arr_mjd2000)


def lambert_transfer(dep_body, arr_body, dep_mjd2000, arr_mjd2000, target_name="target",
                     max_revs=0) -> LambertSolution | None:
    """Solve one transfer and return the cheapest answer across the possible revolution counts.

    Returns None if there is nothing to solve, such as an arrival at or before the departure. The
    cost is the total for a rendezvous: how much faster than Earth the spacecraft has to leave,
    plus what it takes to match the target's velocity on arrival.
    """
    tof_days = float(arr_mjd2000) - float(dep_mjd2000)
    if tof_days <= 0:
        return None
    r1, v_dep = dep_body.eph(float(dep_mjd2000))
    r2, v_arr = arr_body.eph(float(arr_mjd2000))
    r1 = np.asarray(r1, float)
    v_dep = np.asarray(v_dep, float)
    r2 = np.asarray(r2, float)
    v_arr = np.asarray(v_arr, float)
    try:
        lp = pk.lambert_problem(r1, r2, tof_days * _DAY2SEC, MU_SUN, False, int(max_revs))
    except RuntimeError:
        return None

    # v0 and v1 are the transfer's velocities at the first and second position. (PyKEP 2 called
    # these get_v1() and get_v2(), numbered from one: same quantities, shifted names.)
    v1_solutions = lp.v0
    v2_solutions = lp.v1
    best: LambertSolution | None = None
    for k in range(len(v1_solutions)):
        v1 = np.asarray(v1_solutions[k], float)
        v2 = np.asarray(v2_solutions[k], float)
        vinf_dep = np.linalg.norm(v1 - v_dep) / 1000.0   # km/s
        vinf_arr = np.linalg.norm(v2 - v_arr) / 1000.0   # km/s
        dv = vinf_dep + vinf_arr
        if best is None or dv < best.dv_kms:
            best = LambertSolution(
                target_name=target_name, dep_mjd2000=float(dep_mjd2000),
                arr_mjd2000=float(arr_mjd2000), tof_days=tof_days, dv_kms=dv,
                vinf_dep_kms=vinf_dep, vinf_arr_kms=vinf_arr, c3_km2s2=vinf_dep ** 2,
                revs=(k + 1) // 2, r_dep_m=r1, v_dep_body_ms=v_dep, r_arr_m=r2,
                v_arr_body_ms=v_arr, v_transfer_dep_ms=v1, v_transfer_arr_ms=v2)
    return best


def credited_dv(vinf_dep_kms, vinf_arr_kms, vinf_cap_kms: float) -> np.ndarray:
    """What the cruise has to supply per cell, once the free departure speed is taken off.

    The escape delivers up to ``vinf_cap_kms`` of departure speed pointing wherever the cruise
    asks, the same allowance the low-thrust solve assumes, so charging the cruise for it twice
    would push the cheap region toward cells a low-thrust vehicle does not want.
    """
    vd = np.asarray(vinf_dep_kms, float)
    va = np.asarray(vinf_arr_kms, float)
    return np.maximum(vd - float(vinf_cap_kms), 0.0) + va


def thrust_limited_mask(dv_kms, tof_days, *, thrust_N: float, mass_kg: float,
                        thrust_limit: float = 1.0) -> np.ndarray:
    """Cells asking for more delta-v than the engine can produce in the time available.

    A low-thrust engine can build up at most ``a x t`` over a trip, so a cell wanting more than
    that cannot be flown however it is steered. A two-burn map cannot see this, which is how it
    misleads on timing: it likes short trips an electric vehicle cannot fly. Using the starting
    mass errs toward keeping cells, since the real acceleration only rises as propellant burns off.
    """
    a_kms_per_day = float(thrust_limit) * float(thrust_N) / float(mass_kg) * _DAY2SEC / 1000.0
    dv = np.asarray(dv_kms, float)
    tof = np.asarray(tof_days, float)
    return np.isfinite(dv) & (dv > a_kms_per_day * tof)


@dataclass
class Porkchop:
    """The two-burn delta-v over a launch/arrival grid for one target, plus its cheapest cell."""
    target_name: str
    dep_mjd2000: np.ndarray          # 1D, departure dates
    arr_mjd2000: np.ndarray          # 1D, arrival dates
    dv_kms: np.ndarray               # 2D [n_dep, n_arr], NaN where there is no transfer
    vinf_dep_kms: np.ndarray         # 2D
    vinf_arr_kms: np.ndarray         # 2D
    best: LambertSolution | None = field(default=None)

    @property
    def best_dv_kms(self) -> float:
        return float(self.best.dv_kms) if self.best else float("nan")

    @property
    def n_feasible(self) -> int:
        return int(np.isfinite(self.dv_kms).sum())


def porkchop(target, dep_mjd2000, arr_mjd2000, dep_body=None, max_revs=0) -> Porkchop:
    """Solve every (departure, arrival) cell of the grid for one target.

    Returns a :class:`Porkchop` holding the delta-v surface, with NaN where arrival comes before
    departure, and the cheapest cell as a :class:`LambertSolution`. Each solve takes well under a
    millisecond, so a whole grid for one target is effectively instant.
    """
    dep_body = dep_body or earth_planet()
    dep = np.asarray(dep_mjd2000, float)
    arr = np.asarray(arr_mjd2000, float)
    name = getattr(target, "name", "target")

    dv = np.full((dep.size, arr.size), np.nan)
    vinf_dep = np.full_like(dv, np.nan)
    vinf_arr = np.full_like(dv, np.nan)
    best: LambertSolution | None = None
    for di, d0 in enumerate(dep):
        for ai, a0 in enumerate(arr):
            sol = lambert_transfer(dep_body, target, d0, a0, target_name=name,
                                   max_revs=max_revs)
            if sol is None:
                continue
            dv[di, ai] = sol.dv_kms
            vinf_dep[di, ai] = sol.vinf_dep_kms
            vinf_arr[di, ai] = sol.vinf_arr_kms
            if best is None or sol.dv_kms < best.dv_kms:
                best = sol
    return Porkchop(target_name=name, dep_mjd2000=dep, arr_mjd2000=arr, dv_kms=dv,
                    vinf_dep_kms=vinf_dep, vinf_arr_kms=vinf_arr, best=best)


# ---------------------------------------------------------------------------
# date grid from a mission window
# ---------------------------------------------------------------------------

def launch_arrival_grids(launch_window, arrive_by, step_days=10.0, escape_days=0.0,
                         dep_step_days=None, arr_step_days=None):
    """Build (departure, arrival) MJD2000 grids from a mission window.

    Departures are the launch window pushed back by however long the Earth escape takes; arrivals
    run from the earliest departure to ``arrive_by``. The two axes can be sampled at different
    spacings (``dep_step_days`` and ``arr_step_days``, each defaulting to ``step_days``). A short
    launch window usually wants a fine departure step so there is something to click on. Returns
    ``(dep_grid, arr_grid)`` as 1D MJD2000 arrays.
    """
    dep_step = dep_step_days or step_days
    arr_step = arr_step_days or step_days
    start, end = launch_window
    dep_start = mjd2000_from_date(start) + escape_days
    dep_end = mjd2000_from_date(end) + escape_days
    arr_end = mjd2000_from_date(arrive_by)
    dep = np.arange(dep_start, dep_end + dep_step, dep_step)
    arr = np.arange(dep_start, arr_end + arr_step, arr_step)
    return dep, arr


# ---------------------------------------------------------------------------
# ranking a whole population; ranks only, and drops no rows
# ---------------------------------------------------------------------------
