"""
Launch: the launch-orbit library and the Earth-escape delta-v it implies.

A mission names a launch type: either the orbit the launch vehicle drops the spacecraft into, or an
escape the launch vehicle provides itself. If it escapes, the electric propulsion pays nothing; if
it drops the spacecraft into an orbit, the spacecraft spirals out on its own and that comes off the
cruise budget.

This is the quick, analytic half of working that out:

  * :class:`LaunchOrbit`, the drop-off orbit spec, one YAML per type under ``configs/launches/``.
  * :func:`escape_dv_estimate`, the instant estimate the budget uses. Optimistic, so the screen
    keeps borderline targets.
  * :func:`escape_fingerprint`, the inputs that decide what a flown spiral costs, used to match a
    finished `solvers.spiral` run to the config it was run for.

The flown spiral lives in ``solvers/spiral.py`` and settles the estimate here.

The estimate is ``dv = sqrt(v_c(a0)^2 + v_inf^2)``: exact for a bare escape at low thrust, and
optimistic when leaving with speed to spare, since it credits the gain from adding energy deep in
Earth's gravity. Optimistic is the safe direction, since a low escape charge leaves a bigger cruise
budget. An elliptical drop-off uses the circular speed at its average radius, and nothing is
charged for changing the plane. See ``docs/physics.md``.

Units: km, km/s, degrees; YAML altitudes in km above the surface.
"""
from __future__ import annotations

import math
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from prospector import paths
from prospector.constants import (
    EARTH_EQUATORIAL_RADIUS_KM,
    MU_EARTH_KM3_S2,
)
from prospector.yamlio import dump_preserving_comments

# Earth's orbital speed and axial tilt are what couple the geocentric launch frame to the
# heliocentric cruise frame.


# Both libraries are file-based: one YAML per entry, keyed by file stem.
def default_launch_dir() -> Path:
    """The launch-type library in the active config library, resolved per call."""
    return paths.config_dir() / "launches"


def default_return_dir() -> Path:
    """The return-destination library in the active config library, resolved per call."""
    return paths.config_dir() / "returns"


# The three terms that describe the drop-off orbit, and mean nothing without one.
_ORBIT_FIELDS = ("perigee_alt_km", "apogee_alt_km", "inclination_deg")


class LaunchOrbit(BaseModel):
    """The orbit a launch type injects into, and whether escape comes with it.

    ``escape_provided`` marks the types where the launch vehicle gets the spacecraft free of Earth
    by itself, such as a lunar-assisted escape or a direct injection. There is no drop-off orbit
    then: the injection is hyperbolic, or the Moon opens it out before the spacecraft comes back
    round, so the three orbit terms are left empty and nothing reads them. Otherwise the spacecraft
    starts in this closed orbit and has to spiral out under its own power, and all three are
    required.
    """

    name: str
    perigee_alt_km: float | None = Field(
        default=None, ge=0, description="Perigee altitude above the surface (km).")
    apogee_alt_km: float | None = Field(
        default=None, ge=0, description="Apogee altitude above the surface (km).")
    inclination_deg: float | None = Field(
        default=None, ge=0, le=180, description="Geocentric inclination (deg).")
    escape_provided: bool = Field(
        default=False, description="True if the launch vehicle itself delivers escape (C3 >= 0).")
    min_thrust_alt_km: float = Field(
        default=0.0, ge=0,
        description="Coast below this altitude during the spiral (radiation-belt avoidance).")

    @model_validator(mode="before")
    @classmethod
    def _drop_unused_orbit(cls, data):
        """Clear the orbit terms of a launch type that escapes by itself, whatever the file says:
        the spacecraft is never on that orbit, and a stored number reads as if it were."""
        if isinstance(data, dict) and data.get("escape_provided"):
            data = {k: v for k, v in data.items() if k not in _ORBIT_FIELDS}
        return data

    @model_validator(mode="after")
    def _check_orbit(self) -> LaunchOrbit:
        if self.escape_provided:
            return self
        orbit = (self.perigee_alt_km, self.apogee_alt_km, self.inclination_deg)
        missing = [f for f, v in zip(_ORBIT_FIELDS, orbit) if v is None]
        if missing:
            raise ValueError(
                f"{', '.join(missing)} required unless escape_provided is set: the spacecraft has "
                f"to spiral out of a drop-off orbit")
        if self.apogee_alt_km < self.perigee_alt_km:
            raise ValueError("apogee_alt_km is below perigee_alt_km")
        return self

    def _drop_off(self, term: str) -> float:
        value = getattr(self, term)
        if value is None:
            raise ValueError(f"{self.name} has no drop-off orbit: the launch vehicle provides "
                             f"escape, so there is no {term} to work from")
        return float(value)

    @property
    def perigee_radius_km(self) -> float:
        return EARTH_EQUATORIAL_RADIUS_KM + self._drop_off("perigee_alt_km")

    @property
    def apogee_radius_km(self) -> float:
        return EARTH_EQUATORIAL_RADIUS_KM + self._drop_off("apogee_alt_km")

    @property
    def sma_km(self) -> float:
        """Semi-major axis (km)."""
        return 0.5 * (self.perigee_radius_km + self.apogee_radius_km)

    @property
    def v_circ_kms(self) -> float:
        """Circular speed at the semi-major axis (km/s), which sets the escape scale."""
        return math.sqrt(MU_EARTH_KM3_S2 / self.sma_km)


class ReturnDestination(BaseModel):
    """Where the loaded spacecraft delivers its payload at the end of a return trip.

    The homebound counterpart of :class:`LaunchOrbit`: somewhere near Earth or the Moon. At the
    detail the return cruise is solved with, all of those sit essentially where Earth is, so the
    cruise aims at Earth and the destination enters as two numbers:

      * ``arrival_vinf_kms``: how much speed relative to Earth the arrival may still carry and
        still be captured, so the rendezvous need not be exact.
      * ``insertion_dv_kms``: the capture burn, charged as extra propellant against the loaded
        arrival mass rather than flown in the cruise.

    Both are rough, screening-level figures.
    """

    name: str
    arrival_vinf_kms: float = Field(
        ge=0, description="Allowed Earth-relative arrival v-infinity (km/s).")
    insertion_dv_kms: float = Field(
        ge=0, description="Post-arrival insertion/capture dV (km/s), charged as propellant.")
    description: str = ""


def escape_dv_estimate(orbit: LaunchOrbit, vinf_kms: float = 0.0) -> float:
    """The analytic SEP escape delta-v (km/s) from ``orbit`` to hyperbolic excess ``vinf_kms``.

    Zero when the launch vehicle does the escaping. Otherwise the optimistic estimate ``sqrt(v_c^2
    + v_inf^2)`` described at the top of this module: exact for a bare escape when the thrust is
    low, and optimistic when leaving with speed to spare. The flown spiral (`solvers.spiral`)
    settles it.
    """
    if orbit.escape_provided:
        return 0.0
    return math.sqrt(orbit.v_circ_kms ** 2 + float(vinf_kms) ** 2)


def spiral_time_estimate_days(orbit: LaunchOrbit, *, wet_mass_kg: float, thrust_N: float,
                              isp_s: float, vinf_kms: float = 0.0,
                              duty_cycle: float = 1.0) -> float:
    """A coarse spiral duration (days): propellant for the estimated dv at constant mass flow.

    ``t = m_prop / mdot``, with ``m_prop`` from the rocket equation at the estimated escape delta-v
    and ``mdot = F / (Isp g0)``, stretched by the duty cycle. Eclipses and fading array power make
    the real spiral longer. This is for the launch-type comparison table rather than for planning;
    the flown spiral reports the real figure.
    """
    dv_kms = escape_dv_estimate(orbit, vinf_kms)
    if dv_kms <= 0 or thrust_N <= 0 or isp_s <= 0:
        return 0.0
    veff_ms = isp_s * 9.80665
    prop_kg = wet_mass_kg * (1.0 - math.exp(-dv_kms * 1000.0 / veff_ms))
    mdot = thrust_N / veff_ms
    duty = min(max(float(duty_cycle), 1e-6), 1.0)
    return prop_kg / mdot / duty / 86400.0


# ---------------------------------------------------------------------------
# Plane-change geometry: where the tilt gets bought, on departure, in the spiral, or in the
# cruise. The three differ by an order of magnitude in cost, and these helpers quantify each leg
# so the UI can lay the trade out per target. Rough in both accuracy and frame; the flown spiral
# and the low-thrust solve decide. ``docs/physics.md`` works through all three.
# ---------------------------------------------------------------------------

def plane_max_declination_deg(inclination_deg: float) -> float:
    """The largest |declination| of any direction lying in an orbit plane of this
    (equatorial) inclination: i for prograde, 180 - i for retrograde."""
    i = abs(float(inclination_deg)) % 360.0
    return min(i, 180.0 - i) if i <= 180.0 else min(i - 180.0, 360.0 - i)


def max_departure_declination_deg(orbit: LaunchOrbit,
                                  steer_inc_deg: float | None = None) -> float:
    """The largest |equatorial declination| this launch can point the departure at.

    A spiral leaves in its own orbital plane, and a plane tilted by ``i`` contains no direction
    further than ``min(i, 180-i)`` from the equator, whatever the launch time of day. Launch
    chooses the time of day and where along the plane the spacecraft leaves, so everything inside
    that limit is reachable and nothing beyond it is, short of rotating the plane on purpose
    (``steer_inc_deg``). This is the cone the cruise solver's departure has to stay inside.
    Measured against Earth's equator, so the band the UI quotes relative to Earth's orbit
    (:func:`asymptote_latitude_band_deg`) is this cone seen from there. A launch vehicle that
    escapes by itself is unconstrained at 90.
    """
    if orbit.escape_provided:
        return 90.0
    inc = float(steer_inc_deg) if steer_inc_deg is not None else float(orbit.inclination_deg)
    return plane_max_declination_deg(inc)


# ---------------------------------------------------------------------------
# Launch library: one YAML per launch type in configs/launches/, keyed by file stem.
# ---------------------------------------------------------------------------

def load_launch_orbits(launch_dir: str | Path | None = None) -> dict[str, LaunchOrbit]:
    """The launch catalog: every ``*.yaml`` in ``launch_dir``, keyed by file stem."""
    launch_dir = Path(launch_dir) if launch_dir is not None else default_launch_dir()
    if not launch_dir.is_dir():
        raise FileNotFoundError(
            f"no launch-type library at {launch_dir}; the catalog is file-based and has no "
            f"built-in fallback")
    return {
        path.stem: LaunchOrbit.model_validate(yaml.safe_load(path.read_text()))
        for path in sorted(launch_dir.glob("*.yaml"))
    }


def list_launch_orbits(launch_dir: str | Path | None = None) -> list[str]:
    """All launch-type keys, sorted."""
    return sorted(load_launch_orbits(launch_dir))


def save_launch_orbit(orbit: LaunchOrbit, key: str,
                      launch_dir: str | Path | None = None) -> Path:
    """Persist ``orbit`` to ``configs/launches/<key>.yaml`` and return the path."""
    launch_dir = Path(launch_dir) if launch_dir is not None else default_launch_dir()
    launch_dir.mkdir(parents=True, exist_ok=True)
    return dump_preserving_comments(orbit.model_dump(mode="json"),
                                    launch_dir / f"{paths.config_name(key)}.yaml")


def delete_launch_orbit(key: str, launch_dir: str | Path | None = None) -> None:
    """Remove ``configs/launches/<key>.yaml``. A key with no file is not an error."""
    launch_dir = Path(launch_dir) if launch_dir is not None else default_launch_dir()
    (launch_dir / f"{paths.config_name(key)}.yaml").unlink(missing_ok=True)


def load_return_destinations(
        return_dir: str | Path | None = None) -> dict[str, ReturnDestination]:
    """The return-destination catalog: every ``*.yaml`` in ``return_dir``, keyed by file stem."""
    return_dir = Path(return_dir) if return_dir is not None else default_return_dir()
    if not return_dir.is_dir():
        raise FileNotFoundError(
            f"no return-destination library at {return_dir}; the catalog is file-based and has no "
            f"built-in fallback")
    return {
        path.stem: ReturnDestination.model_validate(yaml.safe_load(path.read_text()))
        for path in sorted(return_dir.glob("*.yaml"))
    }


def list_return_destinations(return_dir: str | Path | None = None) -> list[str]:
    """All return-destination keys, sorted."""
    return sorted(load_return_destinations(return_dir))


def save_return_destination(destination: ReturnDestination, key: str,
                            return_dir: str | Path | None = None) -> Path:
    """Persist ``destination`` to ``configs/returns/<key>.yaml`` and return the path."""
    return_dir = Path(return_dir) if return_dir is not None else default_return_dir()
    return_dir.mkdir(parents=True, exist_ok=True)
    return dump_preserving_comments(destination.model_dump(mode="json"),
                                    return_dir / f"{paths.config_name(key)}.yaml")


def delete_return_destination(key: str, return_dir: str | Path | None = None) -> None:
    """Remove ``configs/returns/<key>.yaml``. A key with no file is not an error."""
    return_dir = Path(return_dir) if return_dir is not None else default_return_dir()
    (return_dir / f"{paths.config_name(key)}.yaml").unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Matching a propagated spiral run back to the config it was computed for.
# ---------------------------------------------------------------------------

def escape_fingerprint(rc, options: dict | None = None) -> dict:
    """The inputs that determine a spiral's cost *up to escape*, as a comparable dict.

    A finished spiral can replace the estimated escape budget, but only for the exact setup it was
    flown with, or the refinement would quietly go out of date as soon as the vehicle is edited.
    This collects those inputs, being the drop-off orbit, masses, thrust, Isp, power, drag area and
    the physical spiral settings, rounded so the comparison survives a round trip through JSON.

    Three things are left out. The target departure speed, because the spiral is the same all the
    way to escape however far past it the run continued, so any finished run can refine the
    bare-escape figure. The runtime settings, because a longer time limit or finer sampling cannot
    change what an escaped run cost. And the orientation of the drop-off, meaning the time of day,
    where along the plane, and what it was aimed at, because that is launch targeting rather than a
    different vehicle. It shifts the eclipse and third-body timing only slightly, and the newest
    matching run, usually the re-fly aimed at a chosen exit, supplies the refinement at what it
    cost.
    """
    opts = dict(options or {})
    for key in ("target_vinf_kms", "max_years", "chunk_days", "sample_points",
                "raan_deg", "argp_deg", "exit_lat_deg", "exit_lon_deg"):
        opts.pop(key, None)
    fp = {
        "launch": rc.mission.launch_orbit,
        "orbit": [rc.launch.perigee_alt_km, rc.launch.apogee_alt_km,
                  rc.launch.inclination_deg, rc.launch.min_thrust_alt_km],
        "wet_mass_kg": rc.vehicle.wet_mass,
        "burnout_mass_kg": rc.vehicle.burnout_mass,
        "thrust_N": rc.total_thrust_mN * 1e-3,
        "isp_s": rc.effective_isp,
        "power_req_W": rc.total_power_W,
        "bol_power_W": rc.vehicle.solar_power_W,
        "area_m2": rc.vehicle.area_m2,
        "options": opts,
        # The array output the bus keeps for itself while thrusting; the spiral flies on the rest.
        "array_reserve_W": rc.array_reserve_W(),
    }
    # A thruster that switches between fixed modes flies a different spiral from one that throttles
    # smoothly on the same rated numbers. Recorded only when it applies, so every fingerprint of a
    # continuous stack, and every run already on disk, reads exactly as before.
    modes = discrete_modes(rc)
    if modes:
        fp["discrete_modes"] = modes
    return _rounded(fp)


def discrete_modes(rc) -> dict:
    """The mode tables of the config's discrete thrusters, keyed by engine: ``{key: [[power_W,
    thrust_mN, isp_s], ...]}``. Empty when every thruster throttles continuously."""
    out = {}
    for key, eng in sorted(rc.engines.items()):
        if getattr(eng, "discrete", False):
            out[key] = [[pp.power_W, pp.thrust_mN, pp.isp_s] for pp in eng.modes()]
    return out


def _rounded(value):
    """Round every float to 9 significant digits so fingerprints compare stably via JSON."""
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_rounded(v) for v in value]
    if isinstance(value, float):
        return float(f"{value:.9g}")
    return value
