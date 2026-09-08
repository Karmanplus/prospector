"""
Buildability and rough build cost: can this vehicle be built?

The solvers answer "does it fly?". This answers the next question: do the parts a real spacecraft
needs fit inside the dry mass the trajectory assumed, with room left for a payload, and roughly
what would building it cost? A power-hungry engine needing an array too heavy for its own dry
budget fails here even though its trajectory worked out.

It sizes the power chain, the solar array (off the curve in
:class:`prospector.spacecraft.arrays.ArrayModel`), the power distribution unit, the thruster
systems, a pressure-sized tank (:func:`tank_dry_mass_kg`), and a fixed share for the bus. Every
coefficient is one editable :class:`BusModel` value, and the three modelling choices behind them
are documented there.

Radiation damage is not sized against; it is reported per design as ``eol_factor``
(:func:`mission_eol_factor`). An array that turns out too small runs the vehicle at reduced power,
which is a slower flight and not a rejection.

The relation ``dry = [payload + auto] / (1 - ratios)`` inverts directly, so :func:`assess` reports
the dry mass a budget has left over, which the app calls "payload + margin". A negative figure
means the vehicle cannot be built at that dry mass.

Costs compare designs against each other and nothing more. Array cost per watt and propellant cost
per kilogram come from the sizing tool; the per-engine cost (``Engine.cost_musd``) and the bus cost
per kilogram are placeholders. Launch is not included.

Everything is a plain function of (dry mass, propellant, engine count, engine spec), so it runs
instantly for any vehicle. ``docs/physics.md`` has the full breakdown and the wear check.

Units: kg, W, years; costs in $M.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from prospector import paths
from prospector.constants import SOLAR_CONSTANT_W_M2
from prospector.spacecraft.arrays import EXAMPLE_MASS_CURVE, ArrayMassSegment, ArrayModel
from prospector.spacecraft.propulsion import PPU_BUS_SIDES, assembly_power_by_ppu_side
from prospector.yamlio import dump_preserving_comments, warn_unknown_keys

# Storage density (kg/m^3) used to size the propellant tank when no working gas is given (a
# gas-free assessment); equals xenon's, the reference gas, so a bare assess() prices the tank like
# a xenon one. A real gas passes its own density from the propellant library.
REFERENCE_DENSITY_KG_M3 = 1990.5

# The sizing/cost model is a file-based, application-level config: one YAML, not a hardcoded source
# default, so the coefficients are tunable in the settings window. The field defaults below remain
# the seed used when the file is absent.
def default_bus_model_path() -> Path:
    """The sizing/cost model file in the active config library.

    Resolved per call, never captured at import, so an override cannot be silently ignored (see
    :func:`prospector.paths.config_dir`)."""
    return paths.config_dir() / "build-model.yaml"


class BusModel(BaseModel):
    """Every coefficient of the sizing and cost model.

    Three modelling choices:

      * arrays are sized for the heaviest single operating mode, not every load at once: either
        thrusting, with comms and payload off, or operating, with the thrust off;
      * engine masses in the library are complete systems, thruster and electronics and feed, so
        nothing is added for those, and a thruster's cost travels with the engine;
      * no growth factor is applied on top of a sized component, because the ratios and the tank
        and array coefficients already carry growth inside them: roughly 10% on the tank, 15% on
        the arrays, 10-15% across the subsystems. Applying an allowance again would count it
        twice, so whatever is left over is reported as "payload + margin".

    The one margin applied as its own multiplier is ``power_margin_pct``, a reserve on power rather
    than on mass.

    Radiation damage is a flown result, not a sizing input: the array is built to the worst-mode
    load when new (:func:`required_bol_W`) and the spiral flies whatever the degrading array
    delivers, so an undersized array runs power-limited instead of being grossed up. The
    end-of-life fraction is measured per design (:func:`mission_eol_factor`). There is no generic
    %/yr aging term.
    """

    # -- power chain --
    housekeeping_W: float = 165.0      # GNC 25 + C&DH 35 + wheels 10 + cameras 15 + thermal 80
    ops_load_W: float = 600.0          # payload 300 + comms TX 300 (thrust off)
    power_margin_pct: float = 20.0     # array-sizing reserve on the worst-mode power load
    array_specific_power_W_m2: float = 300.0   # array power per m^2 when new, at 1 AU; sizes
    #                                            the spiral's drag/SRP area from the array power
    # -- solar-array performance physics (prospector.spacecraft.arrays.ArrayModel) --
    # Output vs sun distance (1/r^2 irradiance x temperature-dependent cell efficiency,
    # normalized to 1 AU) and the piecewise vendor power->mass curve that replaced the old
    # blanket W/kg. Flat coefficients here so the settings window edits them like any other;
    # :meth:`array_model` assembles them into the physics object the solvers consume.
    array_alpha_front: float = 0.90            # front-face solar absorptivity
    array_epsilon_front: float = 0.85          # front-face IR emissivity
    array_alpha_rear: float = 0.30             # rear-face solar absorptivity (Earth albedo)
    array_epsilon_rear: float = 0.85           # rear-face IR emissivity
    array_earth_albedo_W_m2: float = 0.3 * SOLAR_CONSTANT_W_M2  # Earth-reflected sunlight at the surface
    array_earth_ir_W_m2: float = 250.0         # Earth infrared at low altitude
    array_cell_eff_slope_pct_per_C: float = -0.06      # cell efficiency slope (%/degC)
    array_cell_eff_intercept_pct: float = 30.0         # cell efficiency at 0 degC (%)
    # The cell temperature the array's datasheet wattage is quoted at. A sunlit panel at 1 AU runs
    # hotter than the 28 degC standard, so a vehicle's operating power (solar_power_W, what the
    # solvers fly) needs a LARGER rated array: mass, area and cost are charged at
    # operating / ArrayModel.operating_over_rated().
    array_rating_temp_C: float = 28.0          # datasheet rating temperature (degC)
    array_mass_scale: float = 1.0              # multiplier on the power->mass curve (sweepable)
    array_mass_curve: list[ArrayMassSegment] = Field(
        default_factory=lambda: list(EXAMPLE_MASS_CURVE),
        description="Piecewise vendor power->mass curve; edited in the YAML, not the UI.")
    # Power-conversion chain, array output to loads. Three stages, and which a load sits behind
    # depends on where it is wired: transmission (the harness from the array to the high-voltage
    # bus) reaches everything; the HV->LV converter reaches the avionics and a low-side PPU; the
    # PPU reaches the thrusters. So a 1000 W thruster draws ~1195 W off the main bus wired
    # low-side and ~1111 W wired high-side. The engine's ``ppu_bus_side`` says which.
    # Transmission is WIRING (ohmic) loss only: the array's temperature and sun-distance response
    # are modelled by ArrayModel, so they must not be folded in here a second time.
    transmission_eff_pct: float = 90.0         # solar array -> main (high-voltage) bus, wiring loss
    hvlv_converter_eff_pct: float = 93.0       # HV->LV converter (avionics + a low-side PPU)
    ppu_eff_pct: float = 90.0                  # power processing unit (thrusters only)
    pcdu_kg_per_kW: float = 1.0

    # -- mass budget (ratios of dry mass; growth allowance already inside them) --
    fixed_bus_ratio_pct: float = 13.0  # GNC, C&DH, comms, batteries, RCS, secondary
    #                                    tank - everything not auto-sized, as % of dry
    structure_ratio_pct: float = 20.0
    thermal_ratio_pct: float = 1.0
    harness_ratio_pct: float = 4.0
    # One propellant tank, sized as a pressure vessel of fixed shape (:func:`tank_dry_mass_kg`).
    # Volume sets the diameter at the length:diameter ratio below; burst pressure sizes the walls.
    # A sphere is the lightest vessel for a given volume; an elongated tank packs into a slender
    # bus at a mass penalty. Defaults are a Type IV COPV at 2:1, about 91 kg/m^3. Use the Type III
    # strength and density (750e6 Pa, 1800 kg/m^3) to recover the earlier calibration.
    tank_base_mass_kg: float = 6.77       # fixed tank/feed structure (valves, mounts, lines)
    tank_length_diameter_ratio: float = Field(
        default=2.0, ge=1.0, description="Tank length / diameter (1.0 = sphere, the lightest).")
    tank_pressure_bar: float = 150.0      # operating pressure (MEOP); deep-space Xe storage
    tank_temperature_C: float = 20.0      # storage temperature; sets real-fluid density (CoolProp)
    tank_safety_factor: float = 1.5       # burst / operating pressure (uncrewed standard)
    tank_wall_strength_pa: float = 850e6  # effective wall strength (Type IV COPV)
    tank_wall_density_kg_m3: float = 1600.0   # wall material density (Type IV COPV)
    tank_wall_factor: float = 1.2         # wall-thickness margin: weld lands, knockdown, bosses

    # -- rough cost ($M unless noted) --
    cell_cost_per_W: float = 300.0     # IMM 3J, $/W of BOL
    propellant_cost_per_kg: float = 2500.0     # xenon, $/kg (per-gas library overrides this)
    bus_cost_per_kg_kusd: float = 150.0        # placeholder: smallsat-class bus $k/kg
    iat_pct: float = 15.0              # integration & test wrap
    pm_pct: float = 10.0               # program management wrap

    def avionics_power_eff(self) -> float:
        """The share of start-of-life array output reaching the avionics, meaning housekeeping,
        payload and comms, after transmission losses and the step-down converter."""
        return (self.transmission_eff_pct / 100.0) * (self.hvlv_converter_eff_pct / 100.0)

    def thruster_power_eff(self, ppu_bus_side: str = "low") -> float:
        """The share of start-of-life array output reaching the thrusters, per bus side.

        On the low-voltage bus the thruster electronics pay for transmission, the step-down
        converter and their own conversion, which is the longest chain aboard. On the high-voltage
        bus they skip the step-down converter. Which applies is the engine's
        `prospector.spacecraft.propulsion.Engine.ppu_bus_side`; for a mixed assembly use
        :func:`assembly_thruster_chain_eff` rather than picking one.
        """
        if ppu_bus_side not in PPU_BUS_SIDES:
            raise ValueError(f"unknown PPU bus side {ppu_bus_side!r}; "
                             f"expected one of {', '.join(PPU_BUS_SIDES)}")
        transmission = self.transmission_eff_pct / 100.0
        upstream = transmission if ppu_bus_side == "high" else self.avionics_power_eff()
        return upstream * (self.ppu_eff_pct / 100.0)

    def array_model(self) -> ArrayModel:
        """The solar-array performance physics assembled from the flat ``array_*`` coefficients
        for output against sun distance and the power-to-mass curve (see
        `prospector.spacecraft.arrays`)."""
        return ArrayModel(
            alpha_front=self.array_alpha_front,
            epsilon_front=self.array_epsilon_front,
            alpha_rear=self.array_alpha_rear,
            epsilon_rear=self.array_epsilon_rear,
            earth_albedo_W_m2=self.array_earth_albedo_W_m2,
            earth_ir_W_m2=self.array_earth_ir_W_m2,
            cell_eff_slope_pct_per_C=self.array_cell_eff_slope_pct_per_C,
            cell_eff_intercept_pct=self.array_cell_eff_intercept_pct,
            rating_temp_C=self.array_rating_temp_C,
            mass_curve=self.array_mass_curve,
            mass_scale=self.array_mass_scale,
        )


def load_bus_model(path: str | Path | None = None) -> BusModel:
    """The configured sizing and cost model.

    Raises :class:`FileNotFoundError` when the file is absent, since every mass and cost here is
    scaled by these coefficients. Within the file, unknown keys are ignored and missing ones fall
    back to the field default, which lets a partial file survive a new coefficient being added.
    """
    path = Path(path) if path is not None else default_bus_model_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"no build model at {path}; the sizing and cost coefficients are file-based and "
            f"have no built-in fallback")
    data = yaml.safe_load(path.read_text()) or {}
    warn_unknown_keys(data, BusModel.model_fields, path)
    if not data.get("array_mass_curve"):
        # The array mass sets the payload capacity, so a file with no curve quietly sizes the whole
        # spacecraft off the illustrative round numbers in ``arrays.EXAMPLE_MASS_CURVE``. That
        # reads as a real answer -- it produced a report quoting 100 W/kg to somebody who had a
        # measured 153 W/kg table -- and halving the array's specific power roughly halves what is
        # left for payload.
        warnings.warn(
            f"{path}: no array_mass_curve, so the array is sized off the illustrative curve in "
            f"prospector.spacecraft.arrays (about 100 W/kg). Add the measured power-to-mass curve "
            f"before relying on the array mass or the payload capacity.",
            stacklevel=2,
        )
    known = {k: v for k, v in data.items() if k in BusModel.model_fields}
    return BusModel.model_validate(known)


def save_bus_model(model: BusModel, path: str | Path | None = None) -> Path:
    """Persist the sizing/cost model to ``configs/build-model.yaml`` and return the path."""
    path = Path(path) if path is not None else default_bus_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return dump_preserving_comments(model.model_dump(mode="json"), path)


# Settings the design search (:mod:`prospector.trades.design_search`) can sweep alongside the
# vehicle parameters. Each moves a real trade: a bigger array margin means a heavier array and less
# payload, but also a faster escape spiral. Driving the search's ``--sweep-param`` flag, its
# validation and the UI's control off this one list keeps all three in agreement. ``kind`` says how
# a value applies: ``"model"`` overrides a ``BusModel`` field, ``"duty"`` replaces the duty cycle.
# Each entry maps a key to (label, unit, kind).
SWEEPABLE: dict[str, tuple[str, str, str]] = {
    "power_margin_pct": ("Solar array margin", "%", "model"),
    "array_mass_scale": ("Array mass scale", "x", "model"),
    "transmission_eff_pct": ("Power transmission eff.", "%", "model"),
    "ops_load_W": ("Ops power load", "W", "model"),
    "fixed_bus_ratio_pct": ("Fixed bus fraction", "%", "model"),
    "tank_wall_factor": ("Tank wall factor", "x", "model"),
    "duty": ("Duty cycle", "", "duty"),
}

# Every "model" key has to name a real coefficient, or an override would quietly do nothing.
assert all(key in BusModel.model_fields
           for key, (_, _, kind) in SWEEPABLE.items() if kind == "model")


def apply_model_overrides(model: BusModel, overrides: dict) -> BusModel:
    """A copy of ``model`` with the swept ``"model"``-kind fields replaced.

    Non-model keys (e.g. ``duty``, applied by the caller as a duty cycle) are ignored, so the
    caller can pass a whole sweep-combination dict without sieving it first.
    """
    fields = {k: v for k, v in overrides.items()
              if SWEEPABLE.get(k, (None, None, None))[2] == "model"}
    return model.model_copy(update=fields) if fields else model


def array_rated_W(bol_power_W: float, model: BusModel | None = None) -> float:
    """The datasheet (rated) wattage behind a beginning-of-life OPERATING power at 1 AU.

    A vehicle's ``solar_power_W`` is what the panel makes facing the Sun at 1 AU, at the
    temperature it runs there. Datasheets quote cells at ``array_rating_temp_C`` (28 degC), where
    they convert better, so the array that delivers the operating figure must be rated higher by
    ``1 / ArrayModel.operating_over_rated()``. Mass, area and cost are charged at this rated
    figure; the flight physics never see it.
    """
    m = model or load_bus_model()
    return m.array_model().rated_power_W(bol_power_W)


def array_area_m2(bol_power_W: float, model: BusModel | None = None) -> float:
    """The solar array's physical area (m^2) for a beginning-of-life operating power.

    From the bus model's areal power (``array_specific_power_W_m2``), which like any catalogue
    figure is quoted at the rating temperature, so the area is read at the RATED power
    (:func:`array_rated_W`). For a SEP bus the array dominates the escape spiral's drag and
    solar-radiation-pressure cross-section, so this is the natural estimate for a vehicle's
    ``area_m2`` when a design was sized by power rather than drawn geometrically. Zero power (or no
    areal coefficient) -> zero area.
    """
    m = model or load_bus_model()
    if float(bol_power_W) <= 0.0 or m.array_specific_power_W_m2 <= 0.0:
        return 0.0
    return array_rated_W(bol_power_W, m) / m.array_specific_power_W_m2


def tank_dry_mass_kg(volume_m3: float, model: BusModel) -> float:
    """Dry mass (kg) of one spherocylindrical propellant tank enclosing ``volume_m3``.

    The tank is a capsule: two hemispherical caps joined by a barrel, total length ``L = ratio x
    D``. ``tank_length_diameter_ratio`` fixes the shape (1.0 is a sphere, the lightest for a given
    volume) and the diameter follows from ``V = pi (1/6 + (ratio - 1)/4) D^3``.

    Walls are pressure-sized by membrane stress at burst pressure: the barrel carries hoop stress
    (``t = P r / sigma``), the caps half that, so an all-cap sphere is lightest. Dry mass is wall
    area x thickness x density, scaled by ``tank_wall_factor`` for weld lands and knockdown, plus
    the fixed ``tank_base_mass_kg``. Linear in volume at fixed shape and pressure.
    """
    if volume_m3 <= 0:
        return model.tank_base_mass_kg
    k = model.tank_length_diameter_ratio
    # V = pi * shape_factor * D^3 for a capsule of length k*D; invert for the diameter.
    shape_factor = 1.0 / 6.0 + (k - 1.0) / 4.0
    diameter = (volume_m3 / (math.pi * shape_factor)) ** (1.0 / 3.0)
    radius = diameter / 2.0
    burst_pa = model.tank_pressure_bar * 1e5 * model.tank_safety_factor   # bar -> Pa
    sigma = model.tank_wall_strength_pa
    t_barrel = burst_pa * radius / sigma            # hoop stress (cylinder)
    t_cap = burst_pa * radius / (2.0 * sigma)       # biaxial stress (sphere)
    cap_area = math.pi * diameter**2                # two hemispheres = one full sphere
    barrel_area = math.pi * diameter * (k - 1.0) * diameter
    wall_mass = (model.tank_wall_density_kg_m3 * model.tank_wall_factor
                 * (cap_area * t_cap + barrel_area * t_barrel))
    return model.tank_base_mass_kg + wall_mass


def mission_eol_factor(model: BusModel, eol_power_fraction: float | None = None) -> float:
    """The end-of-life power fraction: how much of the array's start-of-life power still reaches
    the loads after radiation damage and transmission losses.

    A reported result (``eol_factor``) rather than a sizing gross-up. The array is built to the
    worst-mode load when new (:func:`required_bol_W`), and this says how much survives, which is
    what makes an undersized array read as power-limited. Damage is measured along the flown
    escape, not from a %/yr rate. With no spiral yet, only the transmission loss applies, which is
    optimistic and resolves as soon as one flies.
    """
    eol = model.transmission_eff_pct / 100.0
    if eol_power_fraction is not None:
        eol *= max(0.0, min(1.0, float(eol_power_fraction)))
    return eol


def assembly_thruster_chain_eff(mounts, catalog, model: BusModel | None = None) -> float:
    """The single array-to-thruster efficiency an engine assembly implies.

    A mixed assembly draws through both bus-side chains at once, so the blend is the power-weighted
    harmonic mean, ``eff = total_power / sum(power_side / eff_side)``. That is the only blend where
    dividing total rated power by one number reproduces the real per-side demand. Same reasoning as
    the effective-Isp blend in `prospector.spacecraft.propulsion.assembly_performance`. Reduces to
    :meth:`BusModel.thruster_power_eff` for a stack wired entirely one way.

    A stack drawing no power has no chain to average, so the deeper low-side chain comes back.
    """
    m = model or load_bus_model()
    by_side = assembly_power_by_ppu_side(list(mounts), catalog)
    total = sum(by_side.values())
    if total <= 0.0:
        return m.thruster_power_eff("low")
    demand = sum(power / m.thruster_power_eff(side) for side, power in by_side.items())
    return total / demand


def bol_power_terms(n_engines: int, engine_power_W: float, model: BusModel,
                    *, margin_pct: float | None = None,
                    thruster_chain_eff: float | None = None) -> dict:
    """The array-sizing derivation, term by term: the worst operating mode plus the margin,
    grossed up for the conversion chain each load sits behind.

    Avionics run on transmission x HV->LV; thrusters run on ``thruster_chain_eff``, which depends
    on where their PPUs are wired (None charges the deeper low-side chain). Thrusting and full ops
    never run together, since an electric cruise gates comms and payload around thrust arcs, so the
    two modes priced are

        thrust mode:  housekeeping / avionics_eff  +  thrusters / thruster_eff
        ops mode:     (housekeeping + ops) / avionics_eff

    and the array is sized to the larger, times (1 + margin).

    ``margin_pct`` is the sizing reserve; None uses the model default. A negative margin undersizes
    the array on purpose, to explore a power-limited design. Radiation damage is not sized against
    here.

    Returns the intermediate terms as well as ``bol``, so an explainer can show the derivation
    without recomputing it. :func:`required_bol_W` is this function's ``bol``.
    """
    margin = model.power_margin_pct if margin_pct is None else float(margin_pct)
    avionics_eff = model.avionics_power_eff()
    thruster_eff = (model.thruster_power_eff("low") if thruster_chain_eff is None
                    else float(thruster_chain_eff))
    thrusters = n_engines * float(engine_power_W)
    thrust_mode = model.housekeeping_W / avionics_eff + thrusters / thruster_eff
    ops_mode = (model.housekeeping_W + model.ops_load_W) / avionics_eff
    return {
        "thrusters_W": thrusters,
        "housekeeping_W": model.housekeeping_W,
        "ops_load_W": model.ops_load_W,
        "avionics_eff": avionics_eff,
        "thruster_eff": thruster_eff,
        "thrust_mode": thrust_mode,
        "ops_mode": ops_mode,
        "thrust_limited": thrust_mode >= ops_mode,
        "margin": margin,
        "bol": max(thrust_mode, ops_mode) * (1.0 + margin / 100.0),
    }


def required_bol_W(n_engines: int, engine_power_W: float, model: BusModel,
                   *, margin_pct: float | None = None,
                   thruster_chain_eff: float | None = None) -> float:
    """Beginning-of-life array power (W) for a vehicle: the ``bol`` term of
    :func:`bol_power_terms`, which documents the sizing model."""
    return bol_power_terms(n_engines, engine_power_W, model, margin_pct=margin_pct,
                           thruster_chain_eff=thruster_chain_eff)["bol"]


def array_sizing_terms(vehicle, catalog: dict, model: BusModel | None = None) -> dict | None:
    """The array a vehicle's engine assembly needs, term by term, plus the mass and area it
    implies. None when an engine key does not resolve against ``catalog``.

    The derivation is :func:`bol_power_terms`, the same function that sizes the array everywhere
    else, so an explainer can only ever show the number the vehicle was built to.
    """
    model = model or load_bus_model()
    try:
        thrusters_W = sum(catalog[m.type].power_W * m.count for m in vehicle.engines)
        chain_eff = assembly_thruster_chain_eff(vehicle.mounts, catalog, model)
    except KeyError:
        return None
    margin = vehicle.array_margin_pct
    terms = bol_power_terms(1, thrusters_W, model,
                            margin_pct=(None if margin is None else float(margin)),
                            thruster_chain_eff=chain_eff)
    bol = terms["bol"]
    # Which side of the bus the PPUs are wired to, for the explainer's chain sentence: one side
    # when every mounted type agrees, "mixed" when the blend above is averaging two chains.
    sides = {catalog[m.type].ppu_bus_side for m in vehicle.engines}
    am = model.array_model()
    rated = am.rated_power_W(bol)
    return {**terms, "ppu_bus_side": (sides.pop() if len(sides) == 1 else "mixed"),
            # The operating figure is what the vehicle flies; the rated one is what gets bought,
            # weighed and given area (see array_rated_W).
            "rated_W": rated, "operating_over_rated": am.operating_over_rated(),
            "operating_temp_C": am.operating_temp_C(), "rating_temp_C": am.rating_temp_C,
            "array_kg": am.mass_kg(rated), "area_m2": array_area_m2(bol, model)}


def with_sized_array(vehicle, catalog: dict, model: BusModel | None = None):
    """The vehicle with its array power and drag area filled in from its engines and margin.

    A vehicle that carries neither is not a vehicle with unlimited power; it is one nobody has
    sized yet. Sizing it here means the spiral, the cruise throttle and the mass budget all read
    one array rather than each assuming its own. A vehicle that already carries a power figure is
    left alone, since that is somebody's deliberate number, and so is one whose engines do not
    resolve, which has bigger problems than its array.
    """
    if float(vehicle.solar_power_W) > 0.0:
        return vehicle
    terms = array_sizing_terms(vehicle, catalog, model)
    if terms is None:
        return vehicle
    return vehicle.model_copy(update={"solar_power_W": round(terms["bol"], 1),
                                      "area_m2": round(terms["area_m2"], 3)})


def _lifetime_check(prop_kg: float, n_engines: int,
                    throughput_kg: float | None, lifetime_ignitions: int | None,
                    ignitions: float | None) -> dict:
    """Whether each thruster stays inside its qualified wear limits on this mission.

    Two limits, both per thruster and both optional; an unset limit is not checked. Throughput: the
    propellant load is shared evenly, so each thruster processes ``prop_kg / n_engines`` against
    ``throughput_kg``. Ignitions: every thruster restarts together each time the spiral leaves
    eclipse, so the mission count applies per thruster against ``lifetime_ignitions``.

    A violation is a caution (``lifetime_ok``), not a failure, and stays separate from the mass-fit
    ``buildable`` verdict. None when neither limit can be checked.
    """
    per_thruster = float(prop_kg) / max(int(n_engines), 1)
    checks, cautions, statuses = [], [], []
    if throughput_kg is not None:
        ok = per_thruster <= float(throughput_kg)
        checks.append(ok)
        statuses.append(f"throughput {per_thruster:.0f}/{float(throughput_kg):.0f} kg per thruster")
        if not ok:
            cautions.append(f"propellant throughput {per_thruster:.0f} kg/thruster exceeds the "
                            f"{float(throughput_kg):.0f} kg qualified ({per_thruster - float(throughput_kg):.0f} kg over)")
    if lifetime_ignitions is not None and ignitions is not None:
        ok = float(ignitions) <= float(lifetime_ignitions)
        checks.append(ok)
        statuses.append(f"ignitions {float(ignitions):.0f}/{int(lifetime_ignitions)}")
        if not ok:
            cautions.append(f"{float(ignitions):.0f} ignitions exceeds the {int(lifetime_ignitions)} "
                            f"qualified cycles")
    return {
        "lifetime_ok": (all(checks) if checks else None),
        "prop_per_thruster_kg": round(per_thruster, 2),
        "throughput_margin_kg": (None if throughput_kg is None
                                 else round(float(throughput_kg) - per_thruster, 2)),
        "mission_ignitions": (None if ignitions is None else round(float(ignitions), 1)),
        "lifetime_why": ("; ".join(cautions) if cautions
                         else ("within limits - " + ", ".join(statuses) if statuses
                               else "no qualified lifetime on file")),
    }


def assess(*, dry_kg: float, prop_kg: float, n_engines: int,
           engine_power_W: float, engine_mass_kg: float,
           thruster_cost_musd: float = 1.5,
           belt_days: float | None = None,
           eol_power_fraction: float | None = None,
           throughput_kg: float | None = None,
           lifetime_ignitions: int | None = None,
           ignitions: float | None = None,
           propellant_name: str | None = None,
           propellant_cost_per_kg: float | None = None,
           propellant_density_kg_m3: float | None = None,
           margin_pct: float | None = None,
           thruster_chain_eff: float | None = None,
           model: BusModel | None = None) -> dict:
    """Size the vehicle the configurator's way and report whether it fits ``dry_kg``.

    Returns a JSON-safe dict:

      ``buildable``            everything fits with payload capacity at or above zero. Mass
                               only; going over a wear limit does not change it
      ``payload_capacity_kg``  dry mass left after every component. Negative means the bus alone
                               busts the budget, and reads as how overweight it is. With no growth
                               margin in the model, this leftover is the reserve
      ``min_dry_kg``           the smallest dry mass this design closes at
      ``bol_power_W``, ``array_kg``, ``pcdu_kg``, ``thruster_sys_kg``, ``tank_kg``
                               the items sized automatically
      ``build_cost_musd``      rough hardware and wraps cost ($M, launch excluded)
      ``cost_breakdown_musd``  arrays / thrusters / bus / propellant / wraps
      ``lifetime_ok``          whether each thruster stays inside its wear limits (None if no
                               limit on file); a caution, separate from ``buildable``
      ``prop_per_thruster_kg``, ``throughput_margin_kg``, ``mission_ignitions``,
      ``lifetime_why``         the lifetime breakdown

    The sized items are added up as they are; the fixed bus, structure, thermal and wiring are
    shares of total dry mass, so ``dry = (payload + auto) / (1 - ratios)``. Solved forward with no
    payload it gives ``min_dry_kg``; inverted it gives the payload at a given dry mass.

    ``thruster_cost_musd`` is per engine, so the cost line is ``n_engines x thruster_cost_musd``.
    ``throughput_kg`` and ``lifetime_ignitions`` are the engine's qualified per-thruster limits and
    ``ignitions`` the mission count; each is optional and simply not checked when absent.

    ``propellant_cost_per_kg`` and ``propellant_density_kg_m3`` come from the working gas and
    override the xenon defaults, so a cheaper, less-dense gas reprices the propellant line and
    needs a bigger tank. ``propellant_name`` is display only. ``thruster_chain_eff`` is the
    array-to-thruster efficiency the PPU wiring implies; None charges the low-side chain.
    """
    m = model or load_bus_model()
    density = (REFERENCE_DENSITY_KG_M3 if propellant_density_kg_m3 is None
               else float(propellant_density_kg_m3))
    propellant_cost_per_kg = (m.propellant_cost_per_kg if propellant_cost_per_kg is None
                              else float(propellant_cost_per_kg))

    # The automatically sized items, added straight into the fixed mass with no growth margin. The
    # array is sized for the heaviest operating mode plus the margin at start of life, counting
    # transmission loss only; radiation damage is something the flight produces rather than
    # something sized against (see required_bol_W). The spiral's time in the belts still feeds in
    # below, but only to report eol_factor, not to make the array bigger.
    bol = required_bol_W(n_engines, engine_power_W, m, margin_pct=margin_pct,
                         thruster_chain_eff=thruster_chain_eff)
    # The mass curve is a catalogue curve, so it is read at the rated (datasheet) wattage the
    # operating power implies (array_rated_W), not at the operating power itself.
    array_kg = m.array_model().mass_kg(array_rated_W(bol, m))
    pcdu_kg = bol / 1000.0 * m.pcdu_kg_per_kW
    thruster_kg = n_engines * float(engine_mass_kg)
    # Single propellant tank, sized as a pressure vessel from the volume the load occupies at this
    # gas's storage density (a less-dense gas -> more volume -> a bigger, heavier tank). Geometry,
    # pressure, and material live on the BusModel (see tank_dry_mass_kg).
    tank_volume_m3 = float(prop_kg) / density
    tank_kg = tank_dry_mass_kg(tank_volume_m3, m)

    fixed_sum = array_kg + pcdu_kg + thruster_kg + tank_kg

    # The fixed bus grows with the vehicle, so it goes on the share side of the relation alongside
    # structure, thermal and wiring. The 13% is the allocation.
    ratios = (m.fixed_bus_ratio_pct + m.structure_ratio_pct
              + m.thermal_ratio_pct + m.harness_ratio_pct) / 100.0

    # Invert the relation to get the payload capacity at the given budget...
    capacity = float(dry_kg) * (1.0 - ratios) - fixed_sum
    # ...and run it forward at zero payload for the smallest closing dry mass.
    min_dry = fixed_sum / (1.0 - ratios)

    # Rough cost: the configurator's own unit costs (arrays, propellant) plus the
    # per-engine thruster cost and the bus $/kg placeholder, wrapped with IAT + PM.
    ratio_kg = float(dry_kg) * ratios           # fixed bus + structure/thermal/harness
    bus_kg = pcdu_kg + tank_kg + max(ratio_kg, 0.0)
    cost_arrays = bol * m.cell_cost_per_W / 1e6
    cost_thrusters = n_engines * float(thruster_cost_musd)
    cost_bus = bus_kg * m.bus_cost_per_kg_kusd / 1e3
    cost_prop = float(prop_kg) * propellant_cost_per_kg / 1e6
    hardware = cost_arrays + cost_thrusters + cost_bus
    wraps = hardware * (m.iat_pct + m.pm_pct) / 100.0
    total = hardware + wraps + cost_prop

    life = _lifetime_check(prop_kg, n_engines, throughput_kg, lifetime_ignitions, ignitions)

    buildable = bool(capacity >= 0.0)
    fixed_bus_kg = float(dry_kg) * m.fixed_bus_ratio_pct / 100.0
    if buildable:
        why = f"fits - {capacity:.0f} kg for payload + margin"
        if life["lifetime_ok"] is False:
            # Mass fits, but a thruster is run past its qualified life: a caution on an
            # otherwise-buildable design, not a disqualification.
            why += f"; lifetime caution - {life['lifetime_why']}"
    else:
        # Name the bill so a failure reads as a diagnosis, not a verdict: the auto-sized items
        # (CBE) that must fit before any payload does, and how far over the budget the whole stack
        # lands.
        why = (f"needs ≥{min_dry:.0f} kg dry ({min_dry - float(dry_kg):.0f} kg over): "
               f"arrays {array_kg:.0f} + thrusters {thruster_kg:.0f} + tank "
               f"{tank_kg:.0f} kg CBE, plus bus ({m.fixed_bus_ratio_pct:.0f}% of dry) "
               f"and structure/thermal/harness ratios")
    return {
        "buildable": buildable,
        "build_why": why,
        "propellant": propellant_name,
        "lifetime_ok": life["lifetime_ok"],
        "lifetime_why": life["lifetime_why"],
        "prop_per_thruster_kg": life["prop_per_thruster_kg"],
        "throughput_margin_kg": life["throughput_margin_kg"],
        "mission_ignitions": life["mission_ignitions"],
        "payload_capacity_kg": round(capacity, 2),
        "min_dry_kg": round(min_dry, 2),
        "eol_factor": round(mission_eol_factor(m, eol_power_fraction), 4),
        "belt_days": None if belt_days is None else round(float(belt_days), 1),
        "bol_power_W": round(bol, 1),
        "array_kg": round(array_kg, 2),
        "pcdu_kg": round(pcdu_kg, 2),
        "thruster_sys_kg": round(thruster_kg, 2),
        "tank_kg": round(tank_kg, 2),
        "fixed_bus_kg": round(fixed_bus_kg, 2),
        "build_cost_musd": round(total, 2),
        "cost_breakdown_musd": {
            "solar_arrays": round(cost_arrays, 2),
            "thrusters": round(cost_thrusters, 2),
            "bus_hardware": round(cost_bus, 2),
            "propellant": round(cost_prop, 2),
            "iat_pm_wraps": round(wraps, 2),
        },
    }


def assess_engine(*, dry_kg: float, prop_kg: float, n_engines: int, engine,
                  belt_days: float | None = None,
                  eol_power_fraction: float | None = None,
                  ignitions: float | None = None,
                  propellants: dict | None = None,
                  margin_pct: float | None = None,
                  model: BusModel | None = None) -> dict:
    """:func:`assess` fed from an engine-library :class:`~prospector.spacecraft.propulsion.Engine`.

    By convention the library's ``mass_kg`` is the full propulsion system per unit and
    ``cost_musd`` that unit's rough cost. The engine's qualified ``throughput_kg`` and
    ``lifetime_ignitions`` drive the wear check; either may be None.

    Given ``propellants``, the engine's gas is resolved from it to reprice the propellant and the
    tank, with storage density taken at the model's tank conditions. Omitted, it prices as xenon.
    The array is grossed up for the chain this engine's PPU sits behind.
    """
    m = model or load_bus_model()
    prop = _resolve_propellant(getattr(engine, "propellant", None), propellants)
    return assess(dry_kg=dry_kg, prop_kg=prop_kg, n_engines=int(n_engines),
                  engine_power_W=engine.power_W, engine_mass_kg=engine.mass_kg,
                  thruster_chain_eff=m.thruster_power_eff(
                      getattr(engine, "ppu_bus_side", "low")),
                  thruster_cost_musd=getattr(engine, "cost_musd", 1.5),
                  belt_days=belt_days, eol_power_fraction=eol_power_fraction,
                  throughput_kg=getattr(engine, "throughput_kg", None),
                  lifetime_ignitions=getattr(engine, "lifetime_ignitions", None),
                  ignitions=ignitions,
                  propellant_name=(prop.name if prop else None),
                  propellant_cost_per_kg=(prop.cost_per_kg if prop else None),
                  propellant_density_kg_m3=_storage_density_kg_m3(prop, m),
                  margin_pct=margin_pct, model=m)


def _resolve_propellant(key, propellants: dict | None):
    """The :class:`~prospector.spacecraft.propellants.Propellant` an engine names, or None.

    Imported lazily so :mod:`prospector.spacecraft.buildability` stays a leaf module (no import of
    the config libraries) for callers that price as xenon and never touch a gas catalog.
    """
    if propellants is None:
        return None
    from prospector.spacecraft.propellants import resolve_propellant
    return resolve_propellant(key, propellants)


def _storage_density_kg_m3(prop, model: BusModel) -> float | None:
    """The working gas's storage density (kg/m^3) at the model's tank pressure/temperature.

    Delegates to :func:`prospector.spacecraft.propellants.storage_density`: real-fluid (CoolProp)
    at the configured pressure and temperature when the gas names a CoolProp fluid, else the gas's
    authored density. None when there is no gas, so the caller keeps the xenon reference. Imported
    lazily to keep this module a config-free leaf.
    """
    if prop is None:
        return None
    from prospector.spacecraft.propellants import storage_density
    return storage_density(prop, model.tank_pressure_bar, model.tank_temperature_C)


def assess_assembly(*, dry_kg: float, prop_kg: float, mounts, catalog,
                    belt_days: float | None = None,
                    eol_power_fraction: float | None = None,
                    ignitions: float | None = None,
                    propellants: dict | None = None,
                    margin_pct: float | None = None,
                    model: BusModel | None = None) -> dict:
    """:func:`assess` for a vehicle's engine assembly of ``(engine_key, count)`` pairs.

    Totals add across mounts, and the per-unit figures :func:`assess` scales by count are their
    averages, so a mixed assembly prices the same totals as the equivalent uniform one. The
    lifetime limits are the most limiting across the mounted types, so a mixed stack is gated by
    its weakest thruster. The propellant is the most-mounted engine's gas, since one tank holds one
    gas.

    The array-to-thruster chain is the one thing that cannot be averaged per unit, because the two
    bus sides are different chains, so it is blended power-weighted.
    """
    m = model or load_bus_model()
    n = sum(count for _, count in mounts)
    power_W = sum(count * catalog[key].power_W for key, count in mounts)
    mass_kg = sum(count * catalog[key].mass_kg for key, count in mounts)
    cost_musd = sum(count * getattr(catalog[key], "cost_musd", 1.5) for key, count in mounts)
    throughputs = [catalog[key].throughput_kg for key, _ in mounts
                   if catalog[key].throughput_kg is not None]
    ignition_caps = [catalog[key].lifetime_ignitions for key, _ in mounts
                     if catalog[key].lifetime_ignitions is not None]
    dominant_key = max(mounts, key=lambda mt: mt[1])[0] if mounts else None
    prop = _resolve_propellant(
        getattr(catalog.get(dominant_key), "propellant", None) if dominant_key else None,
        propellants)
    return assess(dry_kg=dry_kg, prop_kg=prop_kg, n_engines=n,
                  engine_power_W=power_W / n, engine_mass_kg=mass_kg / n,
                  thruster_cost_musd=cost_musd / n,
                  thruster_chain_eff=assembly_thruster_chain_eff(mounts, catalog, m),
                  belt_days=belt_days, eol_power_fraction=eol_power_fraction,
                  throughput_kg=(min(throughputs) if throughputs else None),
                  lifetime_ignitions=(min(ignition_caps) if ignition_caps else None),
                  ignitions=ignitions,
                  propellant_name=(prop.name if prop else None),
                  propellant_cost_per_kg=(prop.cost_per_kg if prop else None),
                  propellant_density_kg_m3=_storage_density_kg_m3(prop, m),
                  margin_pct=margin_pct, model=m)


def mass_allocation_components(build: dict, *, dry_mass_kg: float,
                               n_engines: int) -> list[tuple[str, float]]:
    """The ``(label, kg)`` dry-mass slices for the allocation chart; they sum to the dry mass.

    The auto-sized hardware comes straight from the build assessment; the fixed-fraction subsystems
    and the leftover payload-and-margin reserve close the budget to the dry mass.
    """
    m = load_bus_model()
    dry = float(dry_mass_kg)
    n = int(n_engines)
    comps = [
        ("Solar array", build.get("array_kg", 0.0)),
        ("Power distribution", build.get("pcdu_kg", 0.0)),
        (f"Thrusters (×{n})", build.get("thruster_sys_kg", 0.0)),
        ("Propellant tank", build.get("tank_kg", 0.0)),
        ("Avionics & bus", dry * m.fixed_bus_ratio_pct / 100.0),
        ("Structure", dry * m.structure_ratio_pct / 100.0),
        ("Thermal", dry * m.thermal_ratio_pct / 100.0),
        ("Harness", dry * m.harness_ratio_pct / 100.0),
        ("Payload + margin", build.get("payload_capacity_kg", 0.0)),
    ]
    return [(label, float(kg or 0.0)) for label, kg in comps]
