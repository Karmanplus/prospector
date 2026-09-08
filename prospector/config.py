"""
Mission configuration: layered, reproducible, and validated.

Separate reusable libraries, plus one wrapper that picks one of each. A launch profile can be held
fixed while vehicles are swapped, and a whole analysis reloads from a single named file:

    Engine       (propulsion.py)  a thruster's performance
    LaunchOrbit  (launch.py)      a launch type: the drop-off orbit, or an escape the LV provides
    Vehicle                       the physical bus: masses plus an engine assembly
    Mission                       launch window, arrival, launch type, return trip, payload
    Screening                     the reachability cutoff and delta-v margin
    Desirability                  which reachable targets are worth reaching
    Study                         the wrapper: one mission, one vehicle, both screens

A :class:`Study` names a mission and a vehicle and holds the screening terms. :func:`resolve_study`
loads those plus the engine and launch libraries into one :class:`ResolvedConfig`, which works out
the delta-v budget. Budgets always follow from the masses, the engine and the launch type rather
than being stored, so a config cannot carry an out-of-date one. The escape charge is never typed
in: it is the quick estimate for the launch type (`launch.escape_dv_estimate`), replaced by a flown
spiral (`solvers.spiral`) when one matches.

Plain and importable: no UI framework. YAML under ``configs/{engines,vehicles,missions,studies}/``
is the editable source, and each kind round-trips through the helpers below.

Units: masses in kg, speeds in km/s, dates as ISO calendar dates, Isp in seconds.
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, Field, model_validator

from prospector import paths
from prospector.constants import G0_KM_S2, G0_M_S2, SECONDS_PER_DAY
from prospector.launch import (
    LaunchOrbit,
    escape_dv_estimate,
    load_launch_orbits,
    load_return_destinations,
)
from prospector.spacecraft.propulsion import (
    Engine,
    assembly_performance,
    load_engines,
    tsiolkovsky_dv,
)
from prospector.yamlio import dump_preserving_comments

# ===========================================================================
# Building blocks
# ===========================================================================

class EngineMount(BaseModel):
    """One block of identical engines on a vehicle: an engine type and how many."""

    type: str = Field(description="Engine key into the engine library.")
    count: int = Field(default=1, ge=1)


class Vehicle(BaseModel):
    """The physical spacecraft bus: its mass breakdown and its engine assembly.

    Physical only, with no dates and no mission profile, so the same vehicle can fly any mission.
    What delta-v it can produce is worked out in ResolvedConfig, once an engine catalog is
    available.
    """

    name: str = "Untitled vehicle"
    dry_mass: float = Field(gt=0, description="Dry mass (kg), excluding propellant.")
    fuel_mass: float = Field(gt=0, description="Usable + unusable propellant loaded (kg).")
    unusable_prop: float = Field(default=0.0, ge=0, description="Trapped/residual propellant (kg).")
    engines: list[EngineMount] = Field(min_length=1, description="Engine assembly (one or more blocks).")
    # The array, as flown: what it makes when new and how much of it the atmosphere and sunlight
    # push on. Left at zero, :meth:`ResolvedConfig.build` sizes both from the engines this vehicle
    # mounts and the build model, so a vehicle nobody has sized still flies a real array rather
    # than an unlimited one. They don't enter the dV budget.
    solar_power_W: float = Field(
        default=0.0, ge=0,
        description="Beginning-of-life array power (W); 0 = size it from the engine load.")
    area_m2: float = Field(
        default=0.0, ge=0,
        description="Effective area for drag/SRP (m^2); 0 = size it with the array.")
    # The working gas loaded. None = run each engine on the gas it was measured with. When set to a
    # different gas, the engines' thrust/Isp are estimated by scaling the native numbers, using the
    # propellant library's per-gas factors. It is the same conversion the design sweep applies, so
    # the trajectory and budget reflect that gas rather than xenon.
    propellant: str | None = Field(
        default=None, description="Working gas loaded; None = each engine's native gas.")
    # The array-sizing reserve this vehicle was built to, over the worst-mode load AFTER the
    # power-conversion chain (see buildability.required_bol_W). None = use the build model's
    # default margin. 0 = the array delivers full power when new; negative = knowingly undersized
    # (flies power-limited). The materialized ``solar_power_W`` above is sized from it.
    array_margin_pct: float | None = Field(
        default=None, description="Solar-array sizing reserve (%); None = build-model default.")

    @model_validator(mode="after")
    def _check_usable_prop(self) -> Vehicle:
        if self.unusable_prop >= self.fuel_mass:
            raise ValueError("unusable_prop must be less than fuel_mass (no usable propellant)")
        return self

    @property
    def wet_mass(self) -> float:
        """Fueled mass at departure (kg), engines included."""
        return self.dry_mass + self.fuel_mass

    @property
    def burnout_mass(self) -> float:
        """Mass once all usable propellant is spent (kg)."""
        return self.dry_mass + self.unusable_prop

    @property
    def mounts(self) -> list[tuple[str, int]]:
        """The assembly as ``(engine_key, count)`` pairs for performance blending."""
        return [(m.type, m.count) for m in self.engines]

    @property
    def engine_count(self) -> int:
        return sum(m.count for m in self.engines)


class Mission(BaseModel):
    """The flight profile: when the spacecraft launches, when it must arrive, and the trip shape."""

    name: str = "Untitled mission"
    launch_window: tuple[date, date] = (date(2028, 1, 1), date(2028, 3, 31))
    # The launch type only, as a key into the launch library (configs/launches/). What escape costs
    # is derived from it (estimate, then propagated spiral), never stored here.
    launch_orbit: str = Field(default="TLI", description="Launch type (launch-library key).")
    arrive_by: date = date(2029, 3, 1)

    return_trip: bool = False
    return_by: date | None = None
    # Where the loaded spacecraft delivers its payload, as a key into the return-destination
    # library (configs/returns/). Its insertion dV is derived from that catalog unless
    # return_orbit_insert_dv overrides it below.
    return_destination: str = Field(default="EML2", description="Return destination (return-library key).")
    return_orbit_insert_dv: float | None = Field(
        default=None, ge=0,
        description="Override the destination's insertion dV (km/s); None uses the catalog value.")
    # Optional speed knob for the return leg, mirroring the outbound max flight time: a shorter cap
    # forces a faster, hungrier return. None = bounded only by return_by.
    return_max_tof_days: float | None = Field(default=None, ge=0, description="Max return flight time (days).")
    time_at_asteroid: float | None = Field(default=None, ge=0, description="Stay (days).")
    asteroid_payload_mass: float = Field(
        default=0.0, ge=0, description="Mass collected at the asteroid (kg); loads the return leg only."
    )

    @model_validator(mode="after")
    def _check_dates(self) -> Mission:
        start, end = self.launch_window
        if end < start:
            raise ValueError("launch_window end is before its start")
        if self.arrive_by < end:
            raise ValueError("arrive_by is before the launch window closes "
                             "(a launch at the end of the window could never arrive in time)")
        if self.return_trip:
            if self.return_by is None:
                raise ValueError("return_trip is set but return_by is missing")
            if self.return_by < self.arrive_by:
                raise ValueError("return_by is before arrive_by")
        return self


class Screening(BaseModel):
    """The terms that define and bound a reachability screen (set on the Screen step)."""

    h_max: float = Field(default=25.0, description="Absolute-magnitude cutoff (fainter = smaller).")
    dv_margin_factor: float = Field(
        default=1.2, ge=1.0, le=3.0,
        description="Loosens the budget so coarse estimates don't drop borderline targets.",
    )
    reference: str | None = Field(
        default=None, description="Named calibration target to highlight (e.g. '2008 EV5')."
    )


class Desirability(BaseModel):
    """The desirability downselect: of the reachable targets, which are worth reaching.

    The second screen, independent of reachability: it narrows the targets that survived by how
    valuable they would be to mine. Every term starts permissive, either None or off, so an unset
    filter keeps everything and a target whose property is unknown is always kept -- missing data
    can never quietly drop a reachable target. Tightening is something the user chooses, and the
    real solver still decides what can be flown.

    Periods are rotation periods in hours, diameters in meters, tiers ordered S (best) -> D.
    """

    min_tier: Literal["S", "A", "B", "C", "D"] | None = Field(
        default=None, description="Lowest mission-target tier to keep (S best); None keeps all tiers."
    )
    taxonomy_include: list[str] | None = Field(
        default=None,
        description="Spectral major classes to keep (e.g. ['C','B','D']); None keeps all classes.",
    )
    min_period_h: float | None = Field(
        default=None, ge=0, description="Minimum rotation period (h); excludes fast rotators."
    )
    max_period_h: float | None = Field(
        default=None, ge=0, description="Maximum rotation period (h); excludes slow rotators."
    )
    min_diameter_m: float | None = Field(
        default=None, ge=0, description="Minimum diameter (m); excludes sub-value bodies."
    )
    max_diameter_m: float | None = Field(
        default=None, ge=0, description="Maximum diameter (m); excludes oversized bodies."
    )


class Study(BaseModel):
    """The umbrella artifact: one mission + one vehicle + screening terms, by reference.

    This is what gets saved and reloaded to reproduce an analysis: it names the mission and the
    vehicle, which come from their own libraries, and carries both sets of screening terms
    itself: how reachable a target is, and how valuable.
    """

    name: str = "Untitled study"
    mission: str = Field(description="Mission name (library reference).")
    vehicle: str = Field(description="Vehicle name (library reference).")
    target: str | None = Field(
        default=None,
        description="Default focus target (SBDB designation or name); resolved at open. "
                    "None means no project default; the user picks one in Find targets.")
    screening: Screening = Field(default_factory=Screening)
    desirability: Desirability = Field(default_factory=Desirability)


# ===========================================================================
# Resolved runtime config (parts inlined + budgets derived)
# ===========================================================================

class ResolvedConfig(BaseModel):
    """A fully-resolved study: parts inlined and the delta-v budget derived.

    Built from a study (or directly from its parts) once the libraries are loaded. This is the
    object the screen consumes and a run snapshots for reproducibility.
    """

    mission: Mission
    vehicle: Vehicle
    screening: Screening = Field(default_factory=Screening)
    desirability: Desirability = Field(default_factory=Desirability)
    engines: dict[str, Engine] = Field(description="The engines the vehicle references, inlined.")
    launch: LaunchOrbit = Field(description="The mission's launch type, inlined from the library.")
    # The planned departure hyperbolic excess (km/s). This is the user's design knob, not mission
    # config: the spiral spends propellant to deliver it, the cruise solvers treat it as free up to
    # this value, and every derived escape term below prices at it, so one number moves the escape
    # charge, the cruise budget, the departure mass, and the departure window together (the
    # escape/cruise trade made explicit).
    departure_vinf_kms: float = Field(default=0.0, ge=0)
    # A converged, numerically-propagated spiral's escape cost (km/s) and duration (days) AT the
    # departure v-infinity above, set at runtime by whoever verified the run matches this exact
    # config (see launch.escape_fingerprint) and priced it off the run's dV-vs-v-infinity curve.
    # None -> the analytic estimates stand. Serialized into run snapshots, so a run records the
    # escape terms it was screened and solved under.
    # In the budget's currency: the propellant the spiral burned, expressed as the velocity change
    # it buys at the RATED Isp. That is what the capability below is in too, so the two subtract;
    # it is above the velocity change actually flown whenever the array left the engine in a
    # lower mode (see solvers.spiral.SpiralSolution).
    escape_dv_refined: float | None = Field(default=None, ge=0)
    escape_tof_refined: float | None = Field(default=None, ge=0)
    escape_prop_refined: float | None = Field(default=None, ge=0)

    @classmethod
    def build(cls, mission: Mission, vehicle: Vehicle, screening: Screening,
              catalog: dict[str, Engine],
              desirability: Desirability | None = None,
              launches: dict[str, LaunchOrbit] | None = None) -> ResolvedConfig:
        """Resolve the vehicle's engine references and the mission's launch type; assemble.

        ``launches`` defaults to the shipped launch library so existing callers resolve without
        threading a second catalog through.
        """
        missing = [m.type for m in vehicle.engines if m.type not in catalog]
        if missing:
            raise ValueError(
                f"vehicle '{vehicle.name}' references unknown engine(s): {', '.join(missing)}; "
                f"available: {', '.join(sorted(catalog))}"
            )
        engines = {m.type: catalog[m.type] for m in vehicle.engines}
        # Run the engines on the vehicle's loaded gas: when it differs from an engine's native gas,
        # its thrust/Isp are estimated by scaling (the propellant library's factors), so the
        # resolved performance matches that gas rather than xenon, and so does everything derived
        # from it: the budget, the spiral, the cruise solve, buildability. None leaves every engine
        # on its authored gas.
        if vehicle.propellant:
            from prospector.spacecraft.propellants import engine_on_propellant, load_propellants
            pcat = load_propellants()
            engines = {k: engine_on_propellant(e, vehicle.propellant, pcat)[0]
                       for k, e in engines.items()}
        if launches is None:
            launches = load_launch_orbits()
        if mission.launch_orbit not in launches:
            raise ValueError(
                f"mission '{mission.name}' references unknown launch type "
                f"'{mission.launch_orbit}'; available: {', '.join(sorted(launches)) or '(none)'}"
            )
        if mission.return_trip:
            destinations = load_return_destinations()
            if mission.return_destination not in destinations:
                raise ValueError(
                    f"mission '{mission.name}' references unknown return destination "
                    f"'{mission.return_destination}'; available: "
                    f"{', '.join(sorted(destinations)) or '(none)'}"
                )
        # Size the array from the engines it has to run, unless the vehicle already carries a
        # figure. Everything downstream reads solar_power_W, so leaving it at zero would mean the
        # spiral, the cruise throttle and the mass budget each deciding for themselves what an
        # unsized array can do.
        from prospector.spacecraft.buildability import with_sized_array
        vehicle = with_sized_array(vehicle, engines)
        return cls(mission=mission, vehicle=vehicle, screening=screening,
                   desirability=desirability or Desirability(), engines=engines,
                   launch=launches[mission.launch_orbit])

    @property
    def performance(self) -> dict:
        """Blended assembly performance (isp_s, thrust_mN, power_W, mass_kg)."""
        return assembly_performance(self.vehicle.mounts, self.engines)

    @property
    def effective_isp(self) -> float:
        return self.performance["isp_s"]

    @property
    def total_thrust_mN(self) -> float:
        return self.performance["thrust_mN"]

    @property
    def total_power_W(self) -> float:
        return self.performance["power_W"]

    def thruster_chain_eff(self, model=None) -> float:
        """Fraction of array output that reaches this vehicle's thrusters.

        The loss depends on which bus each engine's electronics are wired to, so it belongs to the
        set of engines rather than the bus model alone (see
        `prospector.spacecraft.buildability.assembly_thruster_chain_eff`). Everything that needs
        the delivered fraction reads it here, so one vehicle cannot be flown two ways. ``model``
        defaults to the application's build model.

        Imported inside the function to keep the config layer free of the sizing model.
        """
        from prospector.spacecraft.buildability import assembly_thruster_chain_eff
        return assembly_thruster_chain_eff(self.vehicle.mounts, self.engines, model)

    def array_reserve_W(self, model=None) -> float:
        """The array output the bus keeps for itself while thrusting: the housekeeping load, grossed
        up through the avionics power chain, in watts at the array.

        The array is sized to carry this on top of the thrusters (see
        `prospector.spacecraft.buildability.bol_power_terms`), so the escape and the cruise have
        to take it off the top before handing the rest to the thrusters, or they fly on power the
        bus is already using. ``model`` defaults to the application's build model.
        """
        from prospector.spacecraft.buildability import load_bus_model
        bm = load_bus_model() if model is None else model
        return float(bm.housekeeping_W) / float(bm.avionics_power_eff())

    def thrust_isp_at_power(self, available_power_W: float) -> tuple[float, float]:
        """The stack's (thrust_N, Isp_s) when the bus supplies ``available_power_W``: the operating
        point on the engines' throttle curve (thrust AND Isp fall when power-limited). At or above
        rated power this is the full ``total_thrust_mN`` / ``effective_isp``. Used to price the
        cruise at its frozen post-escape array power."""
        from prospector.spacecraft.propulsion import assembly_performance_at_power
        r = assembly_performance_at_power(self.vehicle.mounts, self.engines, float(available_power_W))
        return r["thrust_mN"] * 1e-3, r["isp_s"]

    def power_thrust_isp_grid(self):
        """Dense ``(power_W, thrust_N, Isp_s)`` lookup of the stack over 0..rated power, for the
        escape propagator to read the live operating point as the array degrades. None if no power
        model (see :func:`propulsion.assembly_power_grid`)."""
        from prospector.spacecraft.propulsion import assembly_power_grid
        return assembly_power_grid(self.vehicle.mounts, self.engines)

    @property
    def total_dv_capability(self) -> float:
        """Full payload-free delta-v the stack can produce (km/s)."""
        return tsiolkovsky_dv(self.effective_isp, self.vehicle.wet_mass, self.vehicle.burnout_mass)

    @property
    def escape_dv_estimate(self) -> float:
        """The analytic escape charge at the planned departure v-infinity (km/s).

        Zero when the launch vehicle does the escaping; otherwise the optimistic spiral estimate
        (see `launch.escape_dv_estimate`) priced at :attr:`departure_vinf_kms`. Charging for that
        departure speed here is not double-counting, because the screen budget credits the same
        speed back (see :attr:`dv_budget`) and the cruise solvers get it free up to that value.
        """
        return escape_dv_estimate(self.launch, self.departure_vinf_kms)

    @property
    def escape_dv(self) -> float:
        """The Earth-escape charge the budget uses (km/s): spiral-refined when available.

        ``escape_dv_refined`` is only ever set from a flown spiral whose fingerprint matches this
        config, so a refined number cannot outlive the vehicle or launch type it was worked out
        for.
        """
        return self.escape_dv_refined if self.escape_dv_refined is not None \
            else self.escape_dv_estimate

    @property
    def escape_source(self) -> str:
        """Where the escape charge came from: 'launch vehicle', 'spiral', or 'estimate'."""
        if self.launch.escape_provided:
            return "launch vehicle"
        return "spiral" if self.escape_dv_refined is not None else "estimate"

    @property
    def escape_tof_days(self) -> float:
        """How long the escape takes (days): 0 for an LV escape, else the matching spiral
        run's duration, else an idealized constant-burn estimate. This is the lag between
        lifting off and leaving Earth, so the cruise departs this much later.

        The estimate is the time to expend :attr:`escape_propellant_kg` at full mass flow,
        so it is priced at the same escape dV the budget uses (:attr:`escape_dv`): a
        spiral-refined escape dV can never pair with a stale, estimate-priced duration."""
        if self.launch.escape_provided:
            return 0.0
        if self.escape_tof_refined is not None:
            return self.escape_tof_refined
        thrust_N = self.total_thrust_mN * 1e-3
        veff_ms = self.effective_isp * G0_M_S2
        if thrust_N <= 0 or veff_ms <= 0:
            return 0.0
        mdot_kg_s = thrust_N / veff_ms
        return self.escape_propellant_kg / mdot_kg_s / SECONDS_PER_DAY

    @property
    def escape_propellant_kg(self) -> float:
        """Propellant the escape burns (kg): what the spiral run reports, else the rocket
        equation at the escape dV in force. Zero for an LV-provided escape. This is the mass
        the cruise never sees, since the spacecraft hands the cruise
        :attr:`cruise_start_mass_kg`, not its liftoff mass.

        The fallback is priced at :attr:`escape_dv` (spiral-refined when available, else the
        analytic estimate) rather than the bare estimate, so the escape dV, its propellant, and
        its duration always describe one consistent escape, never a mix of refined and
        estimated terms."""
        if self.launch.escape_provided:
            return 0.0
        if self.escape_prop_refined is not None:
            return self.escape_prop_refined
        veff_kms = self.effective_isp * G0_KM_S2
        if veff_kms <= 0:
            return 0.0
        return self.vehicle.wet_mass * (1.0 - math.exp(-self.escape_dv / veff_kms))

    @property
    def cruise_start_mass_kg(self) -> float:
        """Mass at Earth departure (kg): liftoff wet mass minus the spiral's propellant.

        The cruise solvers have to start from this mass: a spacecraft that has already spiralled
        out is lighter, so it accelerates better and has less propellant left. The delta-v budget
        needs no such correction, since delta-vs simply add, but the mass matters for the solver's
        acceleration limit and its propellant figures.
        """
        return max(self.vehicle.burnout_mass, self.vehicle.wet_mass - self.escape_propellant_kg)

    @property
    def departure_window(self) -> tuple[date, date]:
        """When the spacecraft can leave Earth: the mission's liftoff window shifted by the
        escape spiral's duration. The trajectory solvers plan against this window, not the raw
        launch window."""
        shift = timedelta(days=round(self.escape_tof_days))
        start, end = self.mission.launch_window
        return (start + shift, end + shift)

    def departure_schedule(self, cruise_start: date | None = None) -> dict:
        """The launch date a cruise departure implies, and whether the pad can deliver it.

        The escape and the cruise are one schedule with no gap between them. The cruise leaves
        :attr:`escape_tof_days` after liftoff, and :attr:`departure_window` is
        ``mission.launch_window`` shifted by that long, so a departure date and a launch date are
        the same choice read from either end. Once escaped, the spacecraft is drifting away at
        roughly its departure speed and cannot hold station, so the only ways to move the handover
        are a different launch date or a slower spiral.

        ``coast_days`` measures the departure against the earliest launch opportunity rather than
        the implied one, so it describes that convention and is zero whenever liftoff is derived. A
        sweep records it per point, where the earliest launch is the fixed reference.

        ``liftoff_in_window`` is the thing worth checking: false means the departure bounds were
        widened past the launch window by a solve's slack, and the pad cannot deliver it.

        Without a ``cruise_start`` the cruise-dependent terms are None and only the windows come
        back.
        """
        escape = timedelta(days=round(self.escape_tof_days))
        launch_open, launch_close = self.mission.launch_window
        cruise_open, cruise_close = self.departure_window
        out = {
            "escape_days": float(self.escape_tof_days),
            "launch_open": launch_open, "launch_close": launch_close,
            "cruise_open": cruise_open, "cruise_close": cruise_close,
            "cruise_start": cruise_start, "coast_days": None,
            "liftoff_if_no_coast": None, "liftoff_in_window": None,
        }
        if cruise_start is None:
            return out
        out["coast_days"] = max(0.0, float((cruise_start - cruise_open).days))
        liftoff = cruise_start - escape
        out["liftoff_if_no_coast"] = liftoff
        out["liftoff_in_window"] = bool(launch_open <= liftoff <= launch_close)
        return out

    @property
    def cruise_dv_limit(self) -> float:
        """Delta-v for the outbound cruise after escape (km/s), payload-free.

        Deliberately ignores the payload: it is collected at the asteroid, so a target must not be
        screened out on the way there because the loaded return leg looks tight. Whether both legs
        work together is the real solver's call.
        """
        return max(0.0, self.total_dv_capability - self.escape_dv)

    @property
    def dv_budget(self) -> float:
        """The reachability budget the screen compares targets against (km/s).

        The cruise limit, plus the departure speed credited back, loosened by the screening margin.
        Crediting it keeps the trade straight: the escape charge above already paid for that speed,
        and getting it for free covers up to that much of any transfer's delta-v. That is
        optimistic, as a screening bound should be, since the element-based estimate is coarse and
        the real solver decides. Because each extra km/s of departure speed costs the spiral well
        under a km/s, planning a faster departure can only make this budget bigger, never quietly
        shrink the screen.
        """
        return (self.cruise_dv_limit + self.departure_vinf_kms) * self.screening.dv_margin_factor

    @property
    def return_dv_capability(self) -> float | None:
        """Theoretical return-leg delta-v with the payload aboard (km/s), or None.

        An upper bound assuming a full tank: the loaded spacecraft burning all its usable
        propellant. How the fuel really splits between the two legs is the trajectory solver's
        business, not modelled here.
        """
        if not self.mission.return_trip:
            return None
        laden_dry = self.vehicle.burnout_mass + self.mission.asteroid_payload_mass
        laden_wet = laden_dry + (self.vehicle.fuel_mass - self.vehicle.unusable_prop)
        return tsiolkovsky_dv(self.effective_isp, laden_wet, laden_dry)

    @property
    def usable_propellant_kg(self) -> float:
        """Total usable propellant the stack ever has (kg): loaded fuel minus the residual."""
        return self.vehicle.fuel_mass - self.vehicle.unusable_prop

    def return_start_mass_kg(self, outbound_final_mass_kg: float) -> float:
        """Laden mass at the start of the return (kg): the outbound arrival mass plus the
        payload collected at the asteroid. The return leg flies heavier than it arrived."""
        return float(outbound_final_mass_kg) + self.mission.asteroid_payload_mass

    def return_available_propellant_kg(self, outbound_cruise_prop_kg: float) -> float:
        """Propellant left for the return (kg): the usable load minus what the escape and the
        outbound cruise already spent. Floored at zero, since a dry tank cannot fly home.

        The escape charge is measured against the whole usable tank rather than the mass left
        after escaping, so subtracting it here alongside the outbound cruise gives the true
        remainder."""
        return max(0.0, self.usable_propellant_kg
                   - self.escape_propellant_kg - float(outbound_cruise_prop_kg))

    def insertion_dv_kms(self, destination) -> float:
        """The return insertion dV (km/s): the mission's explicit override when set, else the
        destination catalog's value. The override lets a user hand-tune the capture cost."""
        if self.mission.return_orbit_insert_dv is not None:
            return float(self.mission.return_orbit_insert_dv)
        return float(destination.insertion_dv_kms)

    def insertion_propellant_kg(self, laden_final_mass_kg: float, destination) -> float:
        """Propellant for the post-arrival insertion burn (kg): the rocket equation at the
        insertion delta-v applied to the loaded arrival mass, since the burn comes after the
        cruise."""
        veff_kms = self.effective_isp * G0_KM_S2
        dv = self.insertion_dv_kms(destination)
        if veff_kms <= 0 or dv <= 0:
            return 0.0
        return float(laden_final_mass_kg) * (1.0 - math.exp(-dv / veff_kms))


# ===========================================================================
# Config libraries: YAML per kind, addressed by file stem.
# ===========================================================================

_T = TypeVar("_T", bound=BaseModel)
_KIND = {Vehicle: "vehicles", Mission: "missions", Study: "studies"}


def _kind_dir(model_cls: type[BaseModel], config_dir: str | Path | None) -> Path:
    """Where one kind of config lives. The single place ``config_dir=None`` resolves to the active
    library, so every loader below inherits the override without repeating the check."""
    root = Path(config_dir) if config_dir is not None else paths.config_dir()
    return root / _KIND[model_cls]


def _list(model_cls: type[BaseModel], config_dir: str | Path | None) -> list[str]:
    d = _kind_dir(model_cls, config_dir)
    return sorted(p.stem for p in d.glob("*.yaml")) if d.is_dir() else []


def _load(model_cls: type[_T], name: str, config_dir: str | Path | None) -> _T:
    path = _kind_dir(model_cls, config_dir) / f"{paths.config_name(name)}.yaml"
    if not path.is_file():
        available = ", ".join(_list(model_cls, config_dir)) or "(none)"
        raise FileNotFoundError(
            f"no {_KIND[model_cls][:-1]} '{name}' in {path.parent}; available: {available}"
        )
    return model_cls.model_validate(yaml.safe_load(path.read_text()))


def _save(obj: BaseModel, name: str, config_dir: str | Path | None) -> Path:
    d = _kind_dir(type(obj), config_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{paths.config_name(name)}.yaml"
    return dump_preserving_comments(obj.model_dump(mode="json"), path)


def load_vehicle(name: str, config_dir: str | Path | None = None) -> Vehicle:
    return _load(Vehicle, name, config_dir)


def save_vehicle(vehicle: Vehicle, name: str, config_dir: str | Path | None = None) -> Path:
    return _save(vehicle, name, config_dir)


def list_missions(config_dir: str | Path | None = None) -> list[str]:
    return _list(Mission, config_dir)


def load_mission(name: str, config_dir: str | Path | None = None) -> Mission:
    return _load(Mission, name, config_dir)


def save_mission(mission: Mission, name: str, config_dir: str | Path | None = None) -> Path:
    return _save(mission, name, config_dir)


def list_studies(config_dir: str | Path | None = None) -> list[str]:
    return _list(Study, config_dir)


def load_study(name: str, config_dir: str | Path | None = None) -> Study:
    return _load(Study, name, config_dir)


def save_study(study: Study, name: str, config_dir: str | Path | None = None) -> Path:
    return _save(study, name, config_dir)


def studies_referencing(part: str, key: str, config_dir: str | Path | None = None) -> list[str]:
    """The study keys whose ``mission`` or ``vehicle`` reference points at ``key``.

    ``part`` is ``"mission"`` or ``"vehicle"``. A study file that will not load is skipped rather
    than raised on: this answers "who else shares this part", and one broken study elsewhere in the
    library must not stop another from saving. Sorted, so the answer is stable.
    """
    if part not in ("mission", "vehicle"):
        raise ValueError(f"part must be 'mission' or 'vehicle', got {part!r}")
    hits = []
    for name in list_studies(config_dir):
        try:
            study = load_study(name, config_dir)
        except (OSError, ValueError, yaml.YAMLError):
            continue
        if getattr(study, part) == key:
            hits.append(name)
    return hits


def resolve_study(study: Study, config_dir: str | Path | None = None,
                  catalog: dict[str, Engine] | None = None) -> ResolvedConfig:
    """Load a study's mission, vehicle, engines, and launch type; assemble a ResolvedConfig.

    Engines and launch types are shared rather than belonging to any one study, so they are read
    from ``config_dir`` when it has them and from the shipped library otherwise. That search order
    is the point rather than a fallback: it lets a study directory hold only the parts a study
    owns, meaning its mission, its vehicle and itself. Either way, whichever catalog is found has
    to contain what the mission asks for; :meth:`ResolvedConfig.build` raises when it does not,
    listing what is available.
    """
    mission = load_mission(study.mission, config_dir)
    vehicle = load_vehicle(study.vehicle, config_dir)
    if catalog is None:
        catalog = load_engines(_catalog_dir(config_dir, "engines"))
    launches = load_launch_orbits(_catalog_dir(config_dir, "launches"))
    return ResolvedConfig.build(mission, vehicle, study.screening, catalog, study.desirability,
                                launches=launches)


def _catalog_dir(config_dir: str | Path | None, kind: str) -> Path:
    """A shared catalog's directory: under ``config_dir`` when present, else the shipped one.

    Written out explicitly rather than as an ``or {}`` on the load: an empty catalog would let a
    caller fall through to built-in defaults, whereas a missing directory here just means this
    config tree does not override that catalog.
    """
    root = Path(config_dir) if config_dir is not None else paths.config_dir()
    local = root / kind
    return local if local.is_dir() else paths.config_dir() / kind


# ---- sensible blank-slate parts for "New ..." ----

def default_vehicle(config_dir: str | Path | None = None) -> Vehicle:
    """A blank-slate vehicle mounting whatever the active engine library offers first.

    The engine is looked up rather than named, because hardcoding a name ties the source to one
    library's contents and a different or trimmed-down library would leave the blank vehicle
    pointing at an engine that does not exist. Raises if the library is empty: a vehicle with no
    engine is not a vehicle, and Vehicle rejects one anyway.
    """
    catalog = load_engines(_catalog_dir(config_dir, "engines"))
    if not catalog:
        raise ValueError(
            f"no engines in the library at {_catalog_dir(config_dir, 'engines')}; a new vehicle "
            f"needs at least one engine to mount")
    return Vehicle(name="New vehicle", dry_mass=301.2, fuel_mass=350.0, unusable_prop=10.0,
                   engines=[EngineMount(type=sorted(catalog)[0], count=3)])


def default_mission() -> Mission:
    return Mission(name="New mission")


def default_resolved(config_dir: str | Path | None = None) -> ResolvedConfig:
    """A blank-slate resolved config for a fresh session, against the active engine library."""
    return ResolvedConfig.build(default_mission(), default_vehicle(), Screening(),
                                load_engines(_catalog_dir(config_dir, "engines")))
