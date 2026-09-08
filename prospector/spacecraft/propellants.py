"""
Propellants: the working-gas library, and what depends on which gas is chosen.

An electric thruster's thrust and Isp are quoted for one working gas, and the engine library
records those measured numbers per gas: a krypton entry carries krypton numbers, a xenon entry
xenon numbers (see ``Engine.propellant``). This module holds the properties of the gas rather than
of the thruster: what it costs per kilogram, and how densely it stores, which decides how heavy a
tank has to be to hold a kilogram of it.

Krypton, for instance, is roughly a quarter the price of xenon per kilogram but stores far less
densely, so the same mass of it needs a bigger, heavier tank. Both effects feed the build-stage
cost and mass model (`prospector.spacecraft.buildability`).

Thrust and Isp are measured per engine on its own gas; changing gas does not re-derive them. The
one exception is an estimate: when an engine has no numbers for a chosen gas,
:func:`engine_on_propellant` scales its native performance by that gas's ``thrust_scale`` and
``isp_scale``, single factors relative to xenon. The estimate is flagged where it is used so it is
never mistaken for measured data, and a thruster with real multi-gas data gets its own engine entry
instead.

Like the engines, propellants are a file-based library: one YAML per gas in
``configs/propellants/``, keyed by filename, with none hardcoded here. ``xenon`` is the reference
the build model's defaults describe.

Units: cost in USD per kg, density in kg/m^3, tank fraction dimensionless.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from prospector import paths
from prospector.constants import KELVIN_OFFSET_C
from prospector.yamlio import dump_preserving_comments, validated


# The propellant library is file-based: one YAML per gas, nothing hardcoded.
def default_propellant_dir() -> Path:
    """The working-gas library in the active config library, resolved per call."""
    return paths.config_dir() / "propellants"

# The reference gas the build model's defaults describe. Used when an engine names no propellant,
# or names one the library does not have.
DEFAULT_PROPELLANT = "xenon"


class Propellant(BaseModel):
    """A working gas's cost and storage properties: everything the gas choice changes
    downstream that is not the thruster's measured Isp/thrust.

    ``density_kg_m3`` is the gas's storage density. It sizes the propellant tank in the build
    model: a load of propellant occupies ``mass / density`` of volume, and the tank's dry mass
    grows with that volume (see `prospector.spacecraft.buildability.tank_dry_mass_kg`). A
    less-dense gas (krypton) therefore needs a bigger, heavier tank for the same mass.

    This is the AUTHORED fallback density. When ``coolprop_name`` is set and CoolProp is installed,
    :func:`storage_density` recomputes the density from the equation of state at the tank's actual
    pressure and temperature (the build-model knobs), so those knobs move the stored density; the
    authored value stands only when CoolProp can't model the gas (iodine has no CoolProp fluid) or
    isn't installed.
    """

    name: str
    cost_per_kg: float = Field(ge=0, description="Propellant cost (USD/kg).")
    density_kg_m3: float = Field(
        gt=0, description="Authored storage density (kg/m^3); the fallback when CoolProp is "
                          "unavailable. Sizes the tank volume.")
    coolprop_name: str | None = Field(
        default=None,
        description="CoolProp fluid name (e.g. 'Xenon') for real-fluid density at tank P/T; "
                    "None = always use the authored density (e.g. iodine, unmodeled).")
    thrust_scale: float = Field(
        default=1.0, gt=0,
        description="Estimated thrust vs xenon for an engine run on this gas (xenon = 1.0).")
    isp_scale: float = Field(
        default=1.0, gt=0,
        description="Estimated Isp vs xenon for an engine run on this gas (xenon = 1.0).")


def load_propellants(propellant_dir: str | Path | None = None) -> dict[str, Propellant]:
    """The propellant catalog: every ``*.yaml`` in ``propellant_dir``, keyed by file stem."""
    propellant_dir = Path(propellant_dir) if propellant_dir is not None else default_propellant_dir()
    if not propellant_dir.is_dir():
        raise FileNotFoundError(
            f"no propellant library at {propellant_dir}; the catalog is file-based and has no "
            f"built-in fallback")
    return {path.stem: validated(Propellant, path)
            for path in sorted(propellant_dir.glob("*.yaml"))}


def list_propellants(propellant_dir: str | Path | None = None) -> list[str]:
    """All propellant keys, sorted."""
    return sorted(load_propellants(propellant_dir))


def save_propellant(propellant: Propellant, key: str,
                    propellant_dir: str | Path | None = None) -> Path:
    """Persist ``propellant`` to ``configs/propellants/<key>.yaml`` and return the path."""
    propellant_dir = Path(propellant_dir) if propellant_dir is not None else default_propellant_dir()
    propellant_dir.mkdir(parents=True, exist_ok=True)
    return dump_preserving_comments(propellant.model_dump(mode="json"),
                                    propellant_dir / f"{paths.config_name(key)}.yaml")


def resolve_propellant(key: str | None,
                       catalog: dict[str, Propellant] | None = None) -> Propellant | None:
    """The :class:`Propellant` an engine's ``propellant`` key names, or None.

    Falls back to the catalog's reference gas (``xenon``) when the key is unset or absent, so an
    engine that names no gas still prices like the build model's historical default. None only when
    no catalog is available at all (the caller then keeps BusModel defaults).
    """
    if catalog is None:
        catalog = load_propellants()
    if not catalog:
        return None
    if key and key in catalog:
        return catalog[key]
    return catalog.get(DEFAULT_PROPELLANT)


def storage_density(propellant: Propellant, pressure_bar: float,
                    temperature_c: float) -> float:
    """The gas's storage density (kg/m^3) at the tank's pressure and temperature.

    When the gas names a CoolProp fluid (``coolprop_name``) and CoolProp is installed, this is the
    real-fluid (equation-of-state) density at ``pressure_bar`` / ``temperature_c`` -- so raising
    the pressure or cooling the tank densifies the stored gas as the physics says. Failing that,
    whether because there is no CoolProp fluid for the gas (iodine), CoolProp is not installed, or
    the conditions fall outside the equation of state, the gas's authored ``density_kg_m3`` stands.
    Xenon and krypton are supercritical at typical SEP storage conditions, so the ideal-gas law
    does not apply and the EOS matters.
    """
    if propellant.coolprop_name:
        rho = _coolprop_density(propellant.coolprop_name, pressure_bar, temperature_c)
        if rho is not None:
            return rho
    return propellant.density_kg_m3


def _coolprop_density(coolprop_name: str, pressure_bar: float,
                      temperature_c: float) -> float | None:
    """Real-fluid density (kg/m^3) from CoolProp, or None if it can't be computed.

    None on any failure, whether CoolProp is absent (it is optional), the fluid is unknown, or the
    conditions fall outside the equation of state, so the caller falls back to the authored density
    rather than raising. CoolProp is imported lazily to keep the gas library usable without it (the
    build model still prices tanks, just at the authored density).
    """
    try:
        from CoolProp.CoolProp import PropsSI
    except Exception:                       # CoolProp not installed -> authored density
        return None
    try:
        rho = PropsSI("D", "P", pressure_bar * 1e5, "T", temperature_c + KELVIN_OFFSET_C, coolprop_name)
    except Exception:                       # unknown fluid / out-of-bounds state
        return None
    return float(rho) if rho and rho > 0 else None


def engine_on_propellant(engine, target_key: str | None,
                         catalog: dict[str, Propellant] | None = None):
    """The engine as it would perform on ``target_key``, and whether that is estimated.

    Returns ``(engine, estimated)``:

      * **native**: the engine already names ``target_key`` (or the gas is unset or absent
        from the library, so there is nothing to scale to): the engine is returned
        unchanged with ``estimated=False``.
      * **estimated**: otherwise the thrust and Isp are scaled from the engine's authored
        (native-gas) numbers by the target/source ratio of the library scale factors
        (xenon's are 1.0), the propellant key is set to ``target_key``, and
        ``estimated=True``. The throttle curve's points are scaled by the same two factors, so
        the curve's top still agrees with the rated point and a power-limited operating point
        reads on the same gas as the rated one. Power, mass, and the qualified lifetime limits
        are unchanged -- the spreadsheet baseline scales only performance.

    The estimate is a baseline assumption, not measured engine data; the caller surfaces the flag
    so the two are never confused. A thruster with real multi-gas data should get its own engine
    entry (authored on that gas) and be selected directly instead.
    """
    if catalog is None:
        catalog = load_propellants()
    source_key = getattr(engine, "propellant", None) or DEFAULT_PROPELLANT
    if not target_key or target_key == source_key or target_key not in catalog:
        return engine, False
    target = catalog[target_key]
    source = resolve_propellant(source_key, catalog)
    if source is None:                       # no catalog to scale against -> leave as-is
        return engine, False
    f_thrust = target.thrust_scale / source.thrust_scale
    f_isp = target.isp_scale / source.isp_scale
    curve = getattr(engine, "power_curve", None)
    scaled = engine.model_copy(update={
        "propellant": target_key,
        "thrust_mN": engine.thrust_mN * f_thrust,
        "isp_s": engine.isp_s * f_isp,
        "power_curve": ([pp.model_copy(update={"thrust_mN": pp.thrust_mN * f_thrust,
                                               "isp_s": pp.isp_s * f_isp}) for pp in curve]
                        if curve else curve),
    })
    return scaled, True
