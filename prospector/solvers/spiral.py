"""
The low-thrust escape spiral out of Earth orbit, flown properly.

What it costs to leave Earth on electric propulsion. Given the orbit the launch vehicle drops the
spacecraft into and the vehicle itself, this integrates the climb from that orbit out to escape,
and past it up to a requested departure speed, including everything that shapes a months-long
spiral: Earth's equatorial bulge, the Sun and Moon's pull, sunlight pressure, drag in the upper
atmosphere, the engine switching off in eclipse, solar arrays degrading from time in the radiation
belts, and how the arrays perform with distance and temperature (`prospector.spacecraft.arrays`,
which barely varies this close to Earth apart from panel heating early in the climb). It is what
settles the quick estimate in `prospector.launch` (`escape_dv_estimate`).

Where the escape meets the cruise. The heliocentric solvers (`lambert`, `simsflanagan`) start out
on Earth's orbit with a departure speed they simply assume. This is what prices that speed: a run
out to ``target_vinf_kms`` reports the total spiral delta-v, and since the trajectory passes
through every lower speed on the way, it records the whole curve of delta-v against departure speed
past the escape point. One run therefore answers the trade in both directions -- spiral to a chosen
speed and hand it to the cruise solve, or let the cruise solve pick a speed and read off what the
spiral pays to deliver it. ``dv_at_escape_kms``, the point where the spacecraft is just barely free
of Earth, is what the screening budget uses.

Steering. Thrust points along the velocity, which is the most efficient direction for a slow climb,
optionally blended with an out-of-plane component that walks the tilt toward ``target_inc_deg``,
weighted toward the points where changing the plane is cheapest. Thrust is switched off in eclipse,
below ``min_thrust_alt_km`` to stay out of the worst radiation, and when the propellant runs out;
available power throttles it back as the arrays degrade. A ``duty_cycle`` scales thrust down for
operational downtime.

Frames, models and units. The state is Cartesian, relative to Earth's equator. The Sun and Moon
move on simple tilted circular orbits, which is accurate enough for eclipse timing and their
gravitational pull at this level of detail. Inputs and outputs use the repo's units (km, km/s,
days, degrees, kg); the integration runs in SI. A months-long spiral takes minutes to compute, so
it runs as a background job (`worker.py spiral`) and never inline in the UI.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from prospector.constants import (
    AU_M,
    DAYS_PER_JULIAN_YEAR,
    EARTH_EQUATORIAL_RADIUS_M,
    EARTH_J2,
    EARTH_OBLIQUITY_RAD,
    EARTH_ROTATION_RAD_S,
    G0_M_S2,
    MOON_ORBIT_PERIOD_DAYS,
    MOON_ORBIT_TILT_RAD,
    MU_EARTH_M3_S2,
    MU_MOON_M3_S2,
    MU_SUN_M3_S2,
    SECONDS_PER_DAY,
    SRP_PRESSURE_N_M2,
)
from prospector.spacecraft.arrays import ArrayModel
from prospector.spacecraft.radiation import (
    RadiationModel,
    default_radiation_model,
    load_radiation_model,
)

MU = MU_EARTH_M3_S2            # Earth GM (m^3/s^2), short name to keep the maths readable

OMEGA_EARTH_ORBIT = 2 * np.pi / (DAYS_PER_JULIAN_YEAR * SECONDS_PER_DAY)   # Sun's rate (rad/s)
OMEGA_MOON = 2 * np.pi / (MOON_ORBIT_PERIOD_DAYS * SECONDS_PER_DAY)        # Moon's rate (rad/s)

C_R = 1.5                      # radiation-pressure coefficient
C_D = 2.2                      # drag coefficient

# A rough band of altitudes, used only for counting time spent in the belts, which is what
# ``belt_days`` reports: a sample counts as "in the belts" while its distance from Earth falls
# between these two. It is bookkeeping only; the damage model itself is the ``RadiationModel``
# below.
VA_INNER = EARTH_EQUATORIAL_RADIUS_M + 1000e3
VA_OUTER = EARTH_EQUATORIAL_RADIUS_M + 60000e3

# How much the arrays degrade is a selectable model (``prospector.spacecraft.radiation``): the
# radiation environment, the glass shielding over the cells, and how the cells respond to a dose
# are one RadiationModel loaded from ``configs/radiation/``. A spiral uses whatever model
# :func:`solve` is handed. The module-level names and functions below mirror the default scenario,
# so the plotting and report layers have something stable to import.
_DEFAULT_RAD = default_radiation_model()
RE_KM = _DEFAULT_RAD.re_km                       # mean Earth radius for the dipole L-shell
PROTON_PLATEAU_L = _DEFAULT_RAD.proton_plateau_l
PROTON_DDD_CORE = _DEFAULT_RAD.proton_ddd_core
ELECTRON_DDD_CORE = _DEFAULT_RAD.electron_ddd_core

# Out-of-plane steering: weight of the normal component at the nodes, and the inclination tolerance
# inside which plane steering switches off.
YAW_WEIGHT = 0.85
INC_TOL_RAD = np.radians(0.5)

# State: [x, y, z, vx, vy, vz, m, t_belt, t_eclipse, ddd]. Position and velocity (m, m/s), mass
# (kg), two running totals in seconds (time in the belts, time in eclipse), and the radiation dose
# the arrays have taken (MeV/g). The dose is integrated along with everything else, so the power
# throttle follows the same model the result reports.


@dataclass
class SpiralSolution:
    """A flown escape spiral: what it cost, the trade curve, and the path it took.

    ``status`` is ``"escaped"`` (reached the requested departure speed), ``"out_of_fuel"``, or
    ``"timed_out"``.

    Two delta-v figures, because propellant and velocity change part company when the engine runs
    below its rated Isp. ``dv_kms`` and ``dv_at_escape_kms`` are the velocity change the thrust
    actually produced, integrated along the flight: what the spacecraft got. ``dv_equiv_kms`` and
    ``dv_at_escape_equiv_kms`` are the propellant burned expressed as the velocity change it would
    have bought at the RATED Isp (the rocket equation on the mass history). The two agree when the
    engine ran at rated Isp throughout and the equivalent is the larger whenever the array left it
    in a lower mode. The budget, the screening cutoff and the ``curve_*`` trade curve are in the
    equivalent, because everything they are compared against (the vehicle's capability, the
    cruise budget) is the propellant load priced at rated Isp; so converting either back to
    kilograms at the rated Isp gives the true propellant. Anything shown to a reader as the
    velocity change flown is the integrated figure. The path arrays are thinned out for plotting
    rather than being the integrator's own output.
    """

    status: str
    dv_kms: float
    dv_at_escape_kms: float | None
    tof_days: float
    tof_at_escape_days: float | None
    vinf_kms: float                      # hyperbolic excess achieved at the end (0 if bound)
    target_vinf_kms: float
    initial_mass_kg: float
    final_mass_kg: float
    eclipse_days: float
    belt_days: float
    revolutions: float                   # orbits flown to escape; also the engine restart count
    power_fraction_end: float            # array power remaining at the end (1.0 = no loss)
    inc_deg_end: float
    # dv-vs-v-infinity tradeoff curve (empty until the spiral escapes)
    curve_vinf_kms: np.ndarray
    curve_dv_kms: np.ndarray
    curve_tof_days: np.ndarray
    # decimated path for plotting/diagnostics
    times_days: np.ndarray
    positions_km: np.ndarray             # (N, 3)
    mass_kg: np.ndarray
    energy_km2_s2: np.ndarray            # specific orbital energy; >= 0 past escape
    inc_deg: np.ndarray
    # Array power remaining at each point, from the dose integrated during the flight with the
    # chosen radiation model, thinned to match the path. The UI and report read this directly, so
    # what they show follows the model that was flown rather than a recomputed default.
    power_fraction: np.ndarray
    # How the starting orbit was oriented. Both of these belong to launch, one set by time of day
    # and the other by ascent targeting. Also the direction the spacecraft left in, which is None
    # while it was still going round Earth. That direction is what the cruise solve's departure has
    # to agree with.
    raan_deg: float = 0.0
    argp_deg: float = 0.0
    exit_lat_deg: float | None = None
    exit_lon_deg: float | None = None
    # Which radiation scenario was flown, so the result explains itself: the report names it, and
    # array sizing knows what thickness of cover glass the end-of-life power assumed.
    radiation_model: str = ""
    coverglass_um: float = 0.0
    coverglass_density_g_cm3: float = 0.0
    # Whether a degrading array kept up. ``power_available_end_W`` is its output at the end, the
    # worst point, since the dose only grows. ``power_limited`` flags that it fell below
    # ``power_required_W``. The spiral flies that case at reduced thrust rather than giving up, so
    # the flag is informational. ``available_thrust_fraction`` is the share of full thrust the
    # cruise can hold once clear of the belts. All zero or False when ``bol_power_W = 0``.
    power_required_W: float = 0.0
    power_available_end_W: float = 0.0
    # The propellant-equivalent delta-v at rated Isp (see the class docstring): the budget's
    # currency. None on an older result that predates the split.
    dv_equiv_kms: float | None = None
    dv_at_escape_equiv_kms: float | None = None
    power_limited: bool = False
    # How array output varies with distance from the Sun and panel temperature
    # (prospector.spacecraft.arrays; 1.0 is the value at 1 AU), thinned to match the path. This
    # close to Earth it stays near 1.0, the only real excursion being panel heating from the Earth
    # itself early in the climb. The mission-profile power plot multiplies this by the degradation
    # above. All ones when no array model was flown.
    power_fraction_sun: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def propellant_kg(self) -> float:
        return self.initial_mass_kg - self.final_mass_kg

    @property
    def available_thrust_fraction(self) -> float:
        """The share of full thrust the aged array can still support, at fixed Isp.

        The cruise runs clear of the belts, so array power holds at whatever it was left with
        after the escape, and thrust scales with the power ratio, capped at 1.0. Returns 1.0 when
        there is no power budget to track, in which case the optimizer sees full thrust."""
        if self.power_required_W <= 0.0 or self.power_available_end_W <= 0.0:
            return 1.0
        return float(min(1.0, self.power_available_end_W / self.power_required_W))

    def dv_at_vinf(self, vinf_kms: float) -> float | None:
        """The spiral delta-v (km/s) needed to reach ``vinf_kms``, read off the curve. None if
        the run never spiralled that far."""
        if self.curve_vinf_kms.size == 0 or vinf_kms > float(self.curve_vinf_kms[-1]) + 1e-9:
            return None
        return float(np.interp(vinf_kms, self.curve_vinf_kms, self.curve_dv_kms))


# ---------------------------------------------------------------------------
# dynamics
# ---------------------------------------------------------------------------

def _atmos_density(altitude_m: float) -> float:
    """Air density in the upper atmosphere (kg/m^3), thinning exponentially; zero above 1000 km."""
    if altitude_m > 1000e3:
        return 0.0
    rho_ref, h_ref, scale_h = 2.0e-11, 250e3, 40e3
    return rho_ref * np.exp(-(altitude_m - h_ref) / scale_h)


def _sun_position(t: float) -> np.ndarray:
    """Where the Sun is (m), on a circular orbit, relative to Earth's equator."""
    theta = OMEGA_EARTH_ORBIT * t
    return AU_M * np.array([np.cos(theta),
                          np.sin(theta) * np.cos(EARTH_OBLIQUITY_RAD),
                          np.sin(theta) * np.sin(EARTH_OBLIQUITY_RAD)])


def _moon_position(t: float) -> np.ndarray:
    """Where the Moon is (m), on a circular tilted orbit."""
    theta = OMEGA_MOON * t
    return 384400e3 * np.array([np.cos(theta),
                                np.sin(theta) * np.cos(MOON_ORBIT_TILT_RAD),
                                np.sin(theta) * np.sin(MOON_ORBIT_TILT_RAD)])


def _in_eclipse(r_vec: np.ndarray, r_sun: np.ndarray, r_mag: float) -> bool:
    """Is it in Earth's shadow? True when it is behind Earth and inside its outline."""
    p = float(np.dot(r_vec, r_sun)) / float(np.linalg.norm(r_sun))
    return p < 0 and (r_mag ** 2 - p ** 2) < EARTH_EQUATORIAL_RADIUS_M ** 2


def _eom(t, state, p):
    """How the state changes each moment: gravity, the perturbations, and steered thrust.

    ``p`` is the parameter dict :func:`solve` puts together. The two clock slots integrate a 1
    while inside the belts or in eclipse and a 0 otherwise, and the dose slot integrates the
    radiation model's dose rate, so time in the belts, time in eclipse, and the array degradation
    that throttles the engine all come out of the same integration.
    """
    x, y, z, vx, vy, vz, m, _t_belt, _t_ecl, ddd, _dv = state
    r_vec = np.array([x, y, z])
    v_vec = np.array([vx, vy, vz])
    r = float(np.linalg.norm(r_vec))
    v = float(np.linalg.norm(v_vec))
    alt = r - EARTH_EQUATORIAL_RADIUS_M

    # Earth's gravity, including the bulge at its equator.
    j2_factor = 1.5 * EARTH_J2 * MU * (EARTH_EQUATORIAL_RADIUS_M ** 2) / (r ** 5)
    z2r2 = 5 * (z ** 2) / (r ** 2)
    a_earth = np.array([
        -MU * x / r ** 3 + j2_factor * x * (z2r2 - 1),
        -MU * y / r ** 3 + j2_factor * y * (z2r2 - 1),
        -MU * z / r ** 3 + j2_factor * z * (z2r2 - 3),
    ])

    # The Sun and Moon's pull, as the difference between their effect here and at Earth.
    r_sun = _sun_position(t)
    r_moon = _moon_position(t)
    d_sun, d_moon = r_sun - r_vec, r_moon - r_vec
    a_sun = MU_SUN_M3_S2 * (d_sun / np.linalg.norm(d_sun) ** 3 - r_sun / np.linalg.norm(r_sun) ** 3)
    a_moon = MU_MOON_M3_S2 * (d_moon / np.linalg.norm(d_moon) ** 3
                        - r_moon / np.linalg.norm(r_moon) ** 3)

    in_eclipse = _in_eclipse(r_vec, r_sun, r)

    # Sunlight pressure, in sunlight only, and drag below 1000 km against air that turns with the
    # Earth.
    a_srp = np.zeros(3)
    a_drag = np.zeros(3)
    if p["area"] > 0.0:
        if not in_eclipse:
            a_srp = (-d_sun / np.linalg.norm(d_sun)) * (SRP_PRESSURE_N_M2 * C_R * p["area"] / m)
        if alt < 1000e3:
            v_rel = v_vec - np.cross([0.0, 0.0, EARTH_ROTATION_RAD_S], r_vec)
            v_rel_mag = float(np.linalg.norm(v_rel))
            if v_rel_mag > 0:
                drag = 0.5 * _atmos_density(alt) * v_rel_mag ** 2 * C_D * p["area"]
                a_drag = -(drag / m) * (v_rel / v_rel_mag)

    # What the engines can do right now. The array makes less power as the accumulated dose
    # degrades the cells, and both thrust and Isp follow the available power along the engines'
    # curve. Given that curve, read both off it at the available power, capped at the rated point;
    # without one, throttle thrust in proportion and hold Isp fixed. Either way the spiral carries
    # on at reduced thrust rather than stopping, so an under-powered design can be judged rather
    # than rejected.
    ddd_rate = float(p["radiation"].ddd_rate(r_vec / 1e3))      # dose in MeV/g per day here
    # How array output varies with distance and temperature (prospector.spacecraft.arrays):
    # sunlight falling off as 1/r^2 times how the cells respond to temperature, relative to 1 AU.
    # This close to Earth it is near 1, apart from panel heating early in the climb. None means
    # assume full output everywhere.
    sun_frac = (float(p["array_model"].power_fraction(np.linalg.norm(d_sun) / AU_M, r / 1e3))
                if p["array_model"] is not None else 1.0)
    # Power reaching the thrusters: the degraded array output, less what the bus keeps for its own
    # housekeeping, times the efficiency of the electronics between the two (``thruster_eff``). 1.0
    # means that chain is not modelled, so the raw array output is compared straight against what
    # the thrusters want.
    avail_power = (max(0.0, p["bol_power"] * float(p["radiation"].cell_power_fraction(ddd))
                       * sun_frac - p["array_reserve"]) * p["thruster_eff"]
                   if p["bol_power"] > 0.0 else float("inf"))
    if p["pw_grid"] is not None:
        stack_power = min(avail_power, float(p["pw_grid"][-1]))     # never above the rated point
        thrust_now = float(np.interp(stack_power, p["pw_grid"], p["thrust_grid"]))
        isp_now = float(np.interp(stack_power, p["pw_grid"], p["isp_grid"]))
    else:
        thrust_now, isp_now = p["thrust"], p["isp"]
        if p["bol_power"] > 0.0 and p["power_req"] > 0.0:
            thrust_now *= min(1.0, avail_power / p["power_req"])   # no curve: throttle in proportion

    a_thrust = np.zeros(3)
    mdot = 0.0
    if thrust_now > 0.0 and r > p["min_thrust_radius"] and m > p["dry_mass"] and not in_eclipse and v > 0:
        eff_thrust = thrust_now * p["duty_cycle"]
        u_tangent = v_vec / v

        # Steering out of the plane toward the target tilt, weighted toward the two points in each
        # orbit where pushing sideways changes the plane most.
        h_vec = np.cross(r_vec, v_vec)
        h_mag = float(np.linalg.norm(h_vec))
        u_normal = np.zeros(3)
        yaw_factor = 0.0
        if p["target_inc"] is not None and h_mag > 0:
            current_inc = np.arccos(np.clip(h_vec[2] / h_mag, -1.0, 1.0))
            inc_error = current_inc - p["target_inc"]
            n_vec = np.cross([0.0, 0.0, 1.0], h_vec)        # where the orbit crosses the equator
            n_mag = float(np.linalg.norm(n_vec))
            if n_mag > 1e-6 and abs(inc_error) > INC_TOL_RAD:
                cos_u = float(np.dot(n_vec, r_vec)) / (n_mag * r)
                u_normal = -np.sign(inc_error) * np.sign(cos_u) * (h_vec / h_mag)
                yaw_factor = YAW_WEIGHT * abs(cos_u)

        thrust_dir = (1.0 - yaw_factor) * u_tangent + yaw_factor * u_normal
        norm = float(np.linalg.norm(thrust_dir))
        if norm > 0:
            a_thrust = thrust_dir / norm * (eff_thrust / m)
            mdot = -eff_thrust / (isp_now * G0_M_S2)

    a_total = a_earth + a_sun + a_moon + a_srp + a_drag + a_thrust
    dt_belt = 1.0 if (VA_INNER <= r <= VA_OUTER) else 0.0
    dt_ecl = 1.0 if in_eclipse else 0.0
    return [vx, vy, vz, a_total[0], a_total[1], a_total[2], mdot, dt_belt, dt_ecl,
            ddd_rate / SECONDS_PER_DAY,                                # dose rate per second
            float(np.linalg.norm(a_thrust))]                           # velocity change produced


def _specific_energy(y: np.ndarray) -> np.ndarray:
    """Orbital energy per kilogram (m^2/s^2) for the state columns in ``y``."""
    r = np.linalg.norm(y[0:3], axis=0)
    v = np.linalg.norm(y[3:6], axis=0)
    return v ** 2 / 2 - MU / r


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------

def solve(
    *,
    perigee_alt_km: float,
    apogee_alt_km: float,
    inclination_deg: float,
    mass_kg: float,
    dry_mass_kg: float,
    thrust_N: float,
    isp_s: float,
    power_req_W: float = 0.0,
    bol_power_W: float = 0.0,
    array_to_thruster_eff: float = 1.0,
    array_reserve_W: float = 0.0,
    area_m2: float = 0.0,
    duty_cycle: float = 1.0,
    target_vinf_kms: float = 0.0,
    target_inc_deg: float | None = None,
    raan_deg: float = 0.0,
    argp_deg: float = 0.0,
    min_thrust_alt_km: float = 0.0,
    radiation_model: RadiationModel | None = None,
    array_model: ArrayModel | None = None,
    power_grid: tuple | None = None,
    max_years: float = 8.0,
    chunk_days: float = 20.0,
    # A months-long spiral is thousands of revolutions; keep enough samples that the plotted path
    # doesn't alias into noise (index decimation keeps the dense early revs).
    sample_points: int = 20000,
    progress: Callable[[float, str], None] | None = None,
) -> SpiralSolution:
    """Fly the escape spiral from the starting orbit out to ``target_vinf_kms``.

    Starts at the low point of the (``perigee_alt_km`` x ``apogee_alt_km``, ``inclination_deg``)
    drop-off orbit and integrates until the orbit reaches the requested departure speed, the
    propellant runs out (``mass <= dry_mass_kg``), or ``max_years`` go by. ``target_vinf_kms = 0``
    means stop as soon as it is free of Earth. ``target_inc_deg``, given relative to Earth's
    equator, steers the plane during the climb; None keeps the plane it started in. ``bol_power_W =
    0`` assumes power is not a constraint, so the engine is never throttled back, though the dose
    is still tracked. ``area_m2 = 0`` skips drag and sunlight pressure.

    ``radiation_model`` picks the scenario, meaning the environment, the cover glass and the cell
    response, whose dose degrades the arrays and sets their end-of-life power; None uses the
    default (``prospector.spacecraft.radiation``). ``array_model`` adds how output varies with
    distance and temperature (`prospector.spacecraft.arrays`), whose only real effect this close to
    Earth is panel heating early in the climb; None assumes full output everywhere. ``power_grid``
    is an optional ``(power_W, thrust_N, isp_s)`` lookup of what the engines do at a given input
    power, taken from their throttle curve; given one, reduced power sets both thrust and Isp along
    it, and without it thrust simply scales with power at fixed ``isp_s``.

    ``raan_deg`` and ``argp_deg`` orient the starting orbit. Both belong to launch, one set by time
    of day and the other by ascent targeting, while the drop-off tilt comes from the launch type
    and is not adjustable here. Together they can point the departure anywhere the plane reaches
    (see :func:`solve_targeted`). The defaults give the standard orientation.

    The integration runs in chunks of ``chunk_days`` so progress can be reported between them:
    ``progress(fraction, message)``, where the fraction is how much of the climb is done.
    """
    duty = min(max(float(duty_cycle), 0.0), 1.0)
    model = radiation_model if radiation_model is not None else default_radiation_model()
    p = {
        "thrust": float(thrust_N), "isp": float(isp_s), "dry_mass": float(dry_mass_kg),
        "duty_cycle": duty, "area": float(area_m2),
        "bol_power": float(bol_power_W), "power_req": float(power_req_W),
        "thruster_eff": max(1e-9, float(array_to_thruster_eff)),
        # Array output the bus keeps for housekeeping (W at the array), taken off before the rest
        # goes through the chain to the thrusters. 0 hands the thrusters everything.
        "array_reserve": max(0.0, float(array_reserve_W)),
        "radiation": model,
        "array_model": array_model,
        "min_thrust_radius": EARTH_EQUATORIAL_RADIUS_M + float(min_thrust_alt_km) * 1e3,
        "target_inc": None if target_inc_deg is None else np.radians(float(target_inc_deg)),
        "pw_grid": None if power_grid is None else np.asarray(power_grid[0], float),
        "thrust_grid": None if power_grid is None else np.asarray(power_grid[1], float),
        "isp_grid": None if power_grid is None else np.asarray(power_grid[2], float),
    }

    # Starting state: the low point of the drop-off orbit, oriented by (raan, inc, argp). With raan
    # and argp both zero, that low point sits on the +x axis with the velocity tilted by the
    # inclination.
    r_p = EARTH_EQUATORIAL_RADIUS_M + perigee_alt_km * 1e3
    r_a = EARTH_EQUATORIAL_RADIUS_M + apogee_alt_km * 1e3
    sma = 0.5 * (r_p + r_a)
    v_p = float(np.sqrt(MU * (2.0 / r_p - 1.0 / sma)))
    rot = _orientation_matrix(np.radians(raan_deg), np.radians(inclination_deg),
                              np.radians(argp_deg))
    r0 = rot @ np.array([r_p, 0.0, 0.0])
    v0 = rot @ np.array([0.0, v_p, 0.0])
    y = np.array([*r0, *v0, float(mass_kg), 0.0, 0.0, 0.0, 0.0])  # ..., t_belt, t_ecl, ddd, dv

    e_target = (float(target_vinf_kms) * 1e3) ** 2 / 2.0
    e0 = float(_specific_energy(y.reshape(-1, 1))[0])
    if e0 >= e_target:
        raise ValueError("the drop-off orbit already meets the target C3; nothing to spiral")

    def target_c3_event(t, state, _p):
        return float(_specific_energy(np.asarray(state).reshape(-1, 1))[0]) - e_target
    target_c3_event.terminal = True
    target_c3_event.direction = 1

    def out_of_fuel_event(t, state, _p):
        return state[6] - p["dry_mass"] - 0.1
    out_of_fuel_event.terminal = True

    t_end = max_years * DAYS_PER_JULIAN_YEAR * SECONDS_PER_DAY
    chunk = max(chunk_days, 1.0) * SECONDS_PER_DAY
    times = [0.0]
    states = [y.copy()]
    status = "timed_out"
    t = 0.0
    # Give up when nothing is happening. A spiral that is escaping gains energy chunk after
    # chunk; one whose array has degraded below what the thrusters need makes no headway at all.
    # Rather than integrate a starved coast all the way out to max_years, which is what made some
    # runs take forever, stop once the climb has stalled for a while. The answer is the same --
    # timed_out, did not escape, but it arrives in seconds.
    e_best = e0                                    # highest energy reached so far
    stalled = 0                                    # chunks in a row with no real gain
    stall_gain = 1e-3 * (e_target - e0)            # "real" means 0.1% of the whole climb
    stall_limit = max(4, int(90.0 / max(chunk_days, 1.0)))   # give up after ~90 days of nothing
    while t < t_end:
        sol = solve_ivp(
            _eom, (t, min(t + chunk, t_end)), y, args=(p,),
            events=(target_c3_event, out_of_fuel_event),
            method="RK45", max_step=5000, rtol=1e-5, atol=1e-5)
        if sol.t.size > 1:
            times.extend(sol.t[1:].tolist())
            states.extend(sol.y[:, 1:].T)
        y = sol.y[:, -1].copy()
        t = float(sol.t[-1])
        if sol.t_events[0].size:
            status = "escaped"
            break
        if sol.t_events[1].size:
            status = "out_of_fuel"
            break
        if not sol.success:
            break
        e_now = float(_specific_energy(y.reshape(-1, 1))[0])
        if e_now > e_best + stall_gain:
            e_best = e_now
            stalled = 0
        else:
            stalled += 1
            if stalled >= stall_limit:             # out of power: it is not going to escape
                break
        if progress is not None:
            frac = min(max((e_now - e0) / (e_target - e0), 0.0), 1.0)
            progress(frac, f"day {t / SECONDS_PER_DAY:.0f} · "
                           f"v∞² climb {frac * 100.0:.0f}%")

    return _build_solution(np.asarray(times), np.column_stack(states), status,
                           float(mass_kg), float(isp_s), float(target_vinf_kms),
                           p, sample_points, raan_deg=float(raan_deg),
                           argp_deg=float(argp_deg))


def solve_for_config(rc, *, progress=None, **kwargs) -> SpiralSolution:
    """Solve the escape spiral for the launch type and vehicle a :class:`ResolvedConfig` names.

    Takes the starting orbit from the launch type and the masses, thrust, Isp and power model from
    the vehicle. Raises ``ValueError`` for a launch type whose vehicle provides the escape itself,
    since then there is no spiral to fly. Passing ``exit_lat_deg`` and ``exit_lon_deg``, the
    direction the cruise solve wants to depart in, routes through :func:`solve_targeted` so the
    spiral aims there.
    """
    if rc.launch.escape_provided:
        raise ValueError(
            f"launch type '{rc.launch.name}' provides escape, so there is no spiral to solve")
    opts = dict(kwargs)
    # The array powers the thrusters through the conversion chain the engines' PPU wiring implies,
    # after the bus has taken its housekeeping, the same terms the array was sized against. Defaulted
    # here so every caller flies the same power model; either may be overridden in the options.
    opts.setdefault("array_to_thruster_eff", rc.thruster_chain_eff())
    opts.setdefault("array_reserve_W", rc.array_reserve_W())
    exit_lat = opts.pop("exit_lat_deg", None)
    exit_lon = opts.pop("exit_lon_deg", None)
    # Work out the radiation scenario from the run options: a model name, keyed into
    # configs/radiation/, plus optional overrides for cover glass thickness and density. The same
    # model then both degrades the escape and sizes the arrays, so the two cannot disagree. A
    # ready-made RadiationModel may also be passed straight in, which is what tests do.
    if not isinstance(opts.get("radiation_model"), RadiationModel):
        model = load_radiation_model(opts.pop("radiation_model", None))
        coverglass_um = opts.pop("coverglass_um", None)
        coverglass_density = opts.pop("coverglass_density", None)
        if coverglass_um is not None or coverglass_density is not None:
            model = model.with_coverglass(float(coverglass_um) if coverglass_um is not None
                                          else model.coverglass_um, coverglass_density)
        opts["radiation_model"] = model
    # The array physics default to the application's build model, the same way the worker supplies
    # its power-conversion chain, so a spiral run from a config uses the numbers the settings
    # window edits. Tests may pass a ready-made ArrayModel, or None to assume full output.
    if "array_model" not in opts:
        opts["array_model"] = rc.bus_model().array_model()
    common = dict(
        perigee_alt_km=rc.launch.perigee_alt_km,
        apogee_alt_km=rc.launch.apogee_alt_km,
        inclination_deg=rc.launch.inclination_deg,
        min_thrust_alt_km=rc.launch.min_thrust_alt_km,
        mass_kg=rc.vehicle.wet_mass,
        dry_mass_kg=rc.vehicle.burnout_mass,
        thrust_N=rc.total_thrust_mN * 1e-3,
        isp_s=rc.effective_isp,
        power_req_W=rc.total_power_W,
        bol_power_W=rc.vehicle.solar_power_W,
        power_grid=rc.power_thrust_isp_grid(),
        area_m2=rc.vehicle.area_m2,
        progress=progress,
        **opts,
    )
    if exit_lat is not None and exit_lon is not None:
        return solve_targeted(exit_lat_deg=float(exit_lat), exit_lon_deg=float(exit_lon),
                              **common)
    return solve(**common)


# ---------------------------------------------------------------------------
# aiming the departure: what launch can target
# ---------------------------------------------------------------------------


def _orientation_matrix(raan_rad: float, inc_rad: float, argp_rad: float) -> np.ndarray:
    """The standard rotation from the orbit's own frame into Earth's, Rz(raan) Rx(inc) Rz(argp)."""
    def rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    c, s = np.cos(inc_rad), np.sin(inc_rad)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return rz(raan_rad) @ rx @ rz(argp_rad)


def _ecliptic_to_equatorial(v_ecl: np.ndarray) -> np.ndarray:
    """Rotate a vector from Earth's orbital frame into Earth's equatorial one."""
    c, s = np.cos(EARTH_OBLIQUITY_RAD), np.sin(EARTH_OBLIQUITY_RAD)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]) @ np.asarray(v_ecl, float)


def exit_direction_deg(positions_km) -> tuple | None:
    """The direction the spacecraft left in, as ``(latitude, longitude)`` degrees relative to
    Earth's orbit.

    Read off the direction of motion at the end of the path, which settles down to the final
    direction once the spacecraft is clear of Earth. ``positions_km`` are relative to Earth's
    equator, the frame this module works in. Returns None when the path is too short.
    """
    pos = np.asarray(positions_km, float)
    if pos.ndim != 2 or len(pos) < 2:
        return None
    v = pos[-1] - pos[-2]
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        return None
    v_eq = v / norm
    c, s = np.cos(EARTH_OBLIQUITY_RAD), np.sin(EARTH_OBLIQUITY_RAD)
    v_ecl = np.array([v_eq[0], v_eq[1] * c + v_eq[2] * s, -v_eq[1] * s + v_eq[2] * c])
    return (float(np.degrees(np.arcsin(np.clip(v_ecl[2], -1.0, 1.0)))),
            float(np.degrees(np.arctan2(v_ecl[1], v_ecl[0]))))


def belt_ddd_rate(positions_km):
    """Total dose rate (MeV/g per day) under the default scenario."""
    return _DEFAULT_RAD.ddd_rate(positions_km)


def cell_power_fraction(ddd):
    """How much of a cell's output survives a cumulative dose (MeV/g), under the default cell."""
    return _DEFAULT_RAD.cell_power_fraction(ddd)


def belt_power_profile(positions_km, times_days):
    """Array power along a flown path under the default scenario. See
    :meth:`prospector.spacecraft.radiation.RadiationModel.power_profile`. Returns
    ``(in_belt, ddd_cumulative, power_fraction)``."""
    return _DEFAULT_RAD.power_profile(positions_km, times_days)


def _plane_and_exit(positions_km) -> tuple | None:
    """Which way the orbit plane faces, and which way the spacecraft left, from the path's end."""
    pos = np.asarray(positions_km, float)
    if pos.ndim != 2 or len(pos) < 2:
        return None
    v_end = pos[-1] - pos[-2]
    n = np.cross(pos[-1], v_end)
    if np.linalg.norm(n) < 1e-9 or np.linalg.norm(v_end) < 1e-9:
        return None
    return n / np.linalg.norm(n), v_end / np.linalg.norm(v_end)


def _target_unit_eq(exit_lat_deg: float, exit_lon_deg: float) -> np.ndarray:
    lat, lon = np.radians(float(exit_lat_deg)), np.radians(float(exit_lon_deg))
    return _ecliptic_to_equatorial([np.cos(lat) * np.cos(lon),
                                    np.cos(lat) * np.sin(lon), np.sin(lat)])


def plane_alignment_error_deg(positions_km, exit_lat_deg: float,
                              exit_lon_deg: float) -> float | None:
    """How far out of the spiral's own plane a direction lies, in degrees.

    Zero means the plane contains that direction, and launch geometry requires that of the cruise
    solve's departure. Where along the plane the spiral leaves is a separate question of launch
    date: eclipse gating anchors it to the Sun and the mission window owns the date. So plane
    containment is the test worth running. Returns None when the path is too short.
    """
    pe = _plane_and_exit(positions_km)
    if pe is None:
        return None
    n, _ = pe
    e_t = _target_unit_eq(exit_lat_deg, exit_lon_deg)
    return float(np.degrees(np.arcsin(np.clip(abs(float(np.dot(n, e_t))), 0.0, 1.0))))


def _node_step_deg(positions_km, exit_lat_deg: float,
                   exit_lon_deg: float) -> float | None:
    """How far to rotate the starting orbit, in degrees of launch time of day, so that its plane
    contains the target direction.

    Solves ``(Rz(d) n) . e_t = 0``, that is ``a cos d + b sin d = -c``, treating the flown path as
    if it rotated rigidly, and takes whichever of the two answers is the smaller turn. This is only
    the correction step for :func:`solve_targeted`: the perturbations depend on orientation, so the
    corrected starting conditions are always re-flown rather than assumed. Returns None when the
    target sits too far from the equator for this plane to reach.
    """
    pe = _plane_and_exit(positions_km)
    if pe is None:
        return None
    n, _ = pe
    e_t = _target_unit_eq(exit_lat_deg, exit_lon_deg)

    a = e_t[0] * n[0] + e_t[1] * n[1]
    b = e_t[1] * n[0] - e_t[0] * n[1]
    c = e_t[2] * n[2]
    r = float(np.hypot(a, b))
    if r < abs(c):
        return None
    phi = float(np.arctan2(b, a))
    half = float(np.arccos(np.clip(-c / r, -1.0, 1.0)))
    candidates = [np.arctan2(np.sin(d), np.cos(d)) for d in (phi + half, phi - half)]
    return float(np.degrees(min(candidates, key=abs)))


def solve_targeted(*, exit_lat_deg: float, exit_lon_deg: float, tol_deg: float = 1.5,
                   max_aim_iter: int = 3, raan_deg: float = 0.0, argp_deg: float = 0.0,
                   progress: Callable[[float, str], None] | None = None,
                   **kwargs) -> SpiralSolution:
    """Fly the spiral so that its orbital plane contains a requested departure direction.

    Nothing is rotated after the fact. Each pass re-flies the whole spiral with a corrected launch
    time of day, keeping the launch type's drop-off tilt, using the previous pass's geometry to
    work out the correction. That way the perturbations, which depend on orientation, are priced at
    the orientation flown. The test is whether the plane contains the direction. Where along the
    plane the spiral leaves is anchored to the Sun by eclipse gating, which makes it a matter of
    launch date that the mission window owns, so it is not chased here and not faked by rotating
    the picture afterwards. One correction is usually enough. If the requested direction is out of
    the plane's reach, which the cruise solve is not allowed to ask for, the best attempt comes
    back and :func:`plane_alignment_error_deg` says how far off it is.
    """
    raan = float(raan_deg)
    sol = None
    for k in range(max(0, int(max_aim_iter)) + 1):
        def prog(frac, msg, _k=k):
            if progress is not None:
                progress(frac, msg if _k == 0 else f"{msg} · aim pass {_k}")
        sol = solve(raan_deg=raan, argp_deg=float(argp_deg), progress=prog, **kwargs)
        if sol.status != "escaped" or sol.exit_lat_deg is None:
            return sol
        err = plane_alignment_error_deg(sol.positions_km, exit_lat_deg, exit_lon_deg)
        if err is None or err <= tol_deg:
            return sol
        step = _node_step_deg(sol.positions_km, exit_lat_deg, exit_lon_deg)
        if step is None:
            return sol             # out of reach: report the run as it was flown
        raan += step
    return sol


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def _build_solution(times: np.ndarray, ys: np.ndarray, status: str, m0: float, isp_s: float,
                    target_vinf_kms: float, p: dict, sample_points: int,
                    raan_deg: float = 0.0, argp_deg: float = 0.0) -> SpiralSolution:
    """Turn the raw integration history into the result: costs, trade curve, thinned path."""
    energy = _specific_energy(ys)                       # m^2/s^2 per sample
    mass = ys[6]
    # Two delta-v histories (see SpiralSolution): the velocity change the thrust produced, which
    # was integrated as state slot 10, and the propellant burned priced at the rated Isp.
    dv_kms_t = ys[10] / 1e3
    dv_equiv_kms_t = isp_s * G0_M_S2 * np.log(m0 / mass) / 1e3
    days = times / SECONDS_PER_DAY

    # The moment it gets free of Earth: interpolate cost and time at the first sign change.
    dv_at_escape = dv_at_escape_equiv = tof_at_escape = None
    crossings = np.flatnonzero((energy[:-1] < 0) & (energy[1:] >= 0))
    if energy[0] >= 0:
        dv_at_escape, dv_at_escape_equiv, tof_at_escape = 0.0, 0.0, 0.0
    elif crossings.size:
        k = int(crossings[0])
        frac = -energy[k] / (energy[k + 1] - energy[k])
        dv_at_escape = float(dv_kms_t[k] + frac * (dv_kms_t[k + 1] - dv_kms_t[k]))
        dv_at_escape_equiv = float(dv_equiv_kms_t[k]
                                   + frac * (dv_equiv_kms_t[k + 1] - dv_equiv_kms_t[k]))
        tof_at_escape = float(days[k] + frac * (days[k + 1] - days[k]))
    elif status == "escaped":
        # A run that stops as soon as it is free ends at that moment, and the root finder can land
        # a hair below zero, so the sign-change scan misses it. The last point is the crossing.
        dv_at_escape, tof_at_escape = float(dv_kms_t[-1]), float(days[-1])
        dv_at_escape_equiv = float(dv_equiv_kms_t[-1])

    # The trade curve: delta-v and elapsed time against departure speed, from the escape point
    # onward. Only points where it is already free of Earth count; the crossing anchors the curve.
    # In the propellant-equivalent, because it is read against the budget (`sweep.escape_cost`).
    hyper = energy >= 0
    if dv_at_escape is not None:
        vinf_h = np.sqrt(2.0 * energy[hyper]) / 1e3
        curve_vinf = np.concatenate([[0.0], vinf_h])
        curve_dv = np.concatenate([[dv_at_escape_equiv], dv_equiv_kms_t[hyper]])
        curve_tof = np.concatenate([[tof_at_escape], days[hyper]])
        # Force the speed to increase along the curve, since perturbations can wobble it, so that
        # interpolating on it behaves.
        keep = np.maximum.accumulate(curve_vinf) <= curve_vinf
        curve_vinf, curve_dv, curve_tof = curve_vinf[keep], curve_dv[keep], curve_tof[keep]
    else:
        curve_vinf = curve_dv = curve_tof = np.empty(0)

    # Thin the path out for plotting.
    idx = np.unique(np.linspace(0, len(times) - 1, min(sample_points, len(times))).astype(int))
    pos_km = ys[0:3, idx].T / 1e3
    h = np.cross(ys[0:3, idx].T, ys[3:6, idx].T)
    h_mag = np.linalg.norm(h, axis=1)
    inc_deg = np.degrees(np.arccos(np.clip(
        np.divide(h[:, 2], h_mag, out=np.zeros_like(h_mag), where=h_mag > 0), -1.0, 1.0)))

    # Orbits flown before escaping: add up dt / period over the bound phase, taking the period from
    # each point's energy (a = -MU/2E, P = 2*pi*sqrt(a^3/MU)). Counting time rather than laps stays
    # right however coarsely the path is sampled. The build-stage check uses this as the engine
    # restart count, since thrust switches off in eclipse and back on in sunlight, so the engine
    # restarts at most once per orbit. An upper bound: a plane that never enters Earth's shadow
    # restarts less often.
    dt = np.diff(times)
    e_bound = energy[:-1]
    bound = e_bound < 0.0
    sma = np.where(bound, -MU / (2.0 * np.where(bound, e_bound, -1.0)), np.nan)
    period = 2.0 * np.pi * np.sqrt(sma ** 3 / MU)
    revolutions = float(np.nansum(np.where(bound, dt / period, 0.0)))

    belt_days = float(ys[7, -1] / SECONDS_PER_DAY)
    # Permanent array damage: the dose the spiral took, integrated along with everything else in
    # state slot 9, which is the same dose that throttled the engine, converted to remaining power
    # by the model's cell curve. So the end-of-life power, the throttle that was flown and the
    # plotted profile all come from one model. This happens regardless of bol_power: the arrays
    # degrade whether or not power was the limiting factor.
    power_frac = float(p["radiation"].cell_power_fraction(float(ys[9, -1])))
    # Remaining power at each point along the thinned path, from that same dose and the same cell
    # curve, so the plotted curve and the headline number both come from the model that was flown
    # rather than from a default recomputed later.
    power_fraction_path = np.asarray(p["radiation"].cell_power_fraction(ys[9, idx]), float)
    # The distance-and-temperature factor at the same points, from the array model that was flown,
    # using the same simple Sun geometry as the eclipse and sunlight-pressure terms. All ones when
    # no model was flown.
    if p["array_model"] is not None:
        theta = OMEGA_EARTH_ORBIT * times[idx]
        sun_m = AU_M * np.column_stack([np.cos(theta),
                                      np.sin(theta) * np.cos(EARTH_OBLIQUITY_RAD),
                                      np.sin(theta) * np.sin(EARTH_OBLIQUITY_RAD)])
        sun_au = np.linalg.norm(sun_m - ys[0:3, idx].T, axis=1) / AU_M
        sun_fraction_path = np.asarray(
            p["array_model"].power_fraction(sun_au, np.linalg.norm(ys[0:3, idx].T, axis=1) / 1e3),
            float)
    else:
        sun_fraction_path = np.ones(idx.size)
    sun_frac_end = float(sun_fraction_path[-1]) if sun_fraction_path.size else 1.0

    # Whether power kept up, checked at the worst point of the flight, which is the end, since the
    # dose only grows. ``power_limited`` flags that the aged array can no longer meet what the
    # thrusters want at full power. The spiral is then flown at reduced thrust, taking longer,
    # rather than being rejected, so the flag is there to be read rather than acted on. The 0.1%
    # tolerance ignores rounding noise on a design sized right at its end-of-life load.
    power_required_W = float(p["power_req"])
    power_available_end_W = (max(0.0, float(p["bol_power"]) * power_frac * sun_frac_end
                                 - float(p["array_reserve"])) * float(p["thruster_eff"]))
    power_limited = bool(p["bol_power"] > 0.0 and power_required_W > 0.0
                         and power_available_end_W < power_required_W * 0.999)

    final_energy = float(energy[-1])
    # The direction it left in, read off the end of the path, which only means anything once it is
    # free of Earth.
    exit_dir = exit_direction_deg(pos_km) if status == "escaped" else None
    return SpiralSolution(
        status=status,
        dv_kms=float(dv_kms_t[-1]),
        dv_at_escape_kms=dv_at_escape,
        tof_days=float(days[-1]),
        tof_at_escape_days=tof_at_escape,
        vinf_kms=float(np.sqrt(2.0 * final_energy) / 1e3) if final_energy > 0 else 0.0,
        target_vinf_kms=float(target_vinf_kms),
        initial_mass_kg=float(m0),
        final_mass_kg=float(mass[-1]),
        eclipse_days=float(ys[8, -1] / SECONDS_PER_DAY),
        belt_days=belt_days,
        revolutions=revolutions,
        power_fraction_end=power_frac,
        inc_deg_end=float(inc_deg[-1]),
        curve_vinf_kms=np.asarray(curve_vinf, float),
        curve_dv_kms=np.asarray(curve_dv, float),
        curve_tof_days=np.asarray(curve_tof, float),
        times_days=days[idx],
        positions_km=pos_km,
        mass_kg=mass[idx],
        energy_km2_s2=energy[idx] / 1e6,
        inc_deg=inc_deg,
        power_fraction=power_fraction_path,
        raan_deg=raan_deg,
        argp_deg=argp_deg,
        exit_lat_deg=exit_dir[0] if exit_dir is not None else None,
        exit_lon_deg=exit_dir[1] if exit_dir is not None else None,
        radiation_model=str(p["radiation"].name),
        coverglass_um=float(p["radiation"].coverglass_um),
        coverglass_density_g_cm3=float(p["radiation"].coverglass_density_g_cm3),
        power_required_W=power_required_W,
        power_available_end_W=power_available_end_W,
        power_limited=power_limited,
        power_fraction_sun=sun_fraction_path,
        dv_equiv_kms=float(dv_equiv_kms_t[-1]),
        dv_at_escape_equiv_kms=dv_at_escape_equiv,
    )
