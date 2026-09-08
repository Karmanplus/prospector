"""
Engines: the thruster library and the rocket equation.

Holds the candidate electric thrusters and the physics that turns mass plus engines into delta-v.
No tanks, no gas properties, no power modelling; just the performance numbers the screen and
solvers need.

Engines are a file-based library, one YAML per engine in ``configs/engines/``, with none hardcoded
here. A vehicle refers to them by name and count, and :func:`assembly_performance` blends a mix
into one effective engine. Only Isp enters the delta-v budget, since the rocket equation depends on
exhaust speed, but thrust and power are carried for the low-thrust solvers.

Units: Isp in seconds, thrust in millinewtons, power in watts, masses in kg, delta-v in km/s.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field

from prospector import paths
from prospector.constants import G0_KM_S2
from prospector.yamlio import dump_preserving_comments


def default_engine_dir() -> Path:
    """The engine library directory in the active config library, one YAML per engine.

    A function rather than a constant, so the active library is looked up on each call and an
    override cannot be frozen at import time (see :func:`prospector.paths.config_dir`). Named
    ``default_*`` so the ``engine_dir`` parameter of the loaders below cannot shadow it. A
    parameter shadowing a module-level name is how loading the catalog once turned into an
    attribute lookup on None."""
    return paths.config_dir() / "engines"

# Which of the two power buses a thruster's electronics are wired to. The array feeds a
# high-voltage bus, and a converter steps that down to the low-voltage bus the avionics run on.
#
#   "high"  wired straight to the high-voltage bus, skipping the step-down converter
#   "low"   fed from the low-voltage bus, so it pays for the converter too
#
# High-side is more efficient and usual for high-power thrusters. It belongs to the thruster's
# electrical design, which is why it lives here rather than in the bus model.
PPU_BUS_SIDES = ("high", "low")
PPUBusSide = Literal["high", "low"]

# How a thruster follows its throttle curve when the bus cannot supply rated power. "continuous":
# any power between the measured points, thrust and Isp interpolated (a thruster whose discharge
# can be set anywhere in its range). "discrete": the measured points ARE the operating modes, the
# thruster hard-switches to the highest mode it has the power for and runs it at that mode's
# constant power, thrust and Isp (several flight Hall thrusters work this way). A discrete thruster with no
# curve has one mode, its rated point, so it is either on at rated or off.
THROTTLE_MODES = ("continuous", "discrete")
ThrottleMode = Literal["continuous", "discrete"]


class PowerPoint(BaseModel):
    """One measured (input power, thrust, Isp) operating point on a thruster's throttle curve."""
    power_W: float = Field(gt=0)
    thrust_mN: float = Field(gt=0)
    isp_s: float = Field(gt=0)


class Engine(BaseModel):
    """What an electric thruster does, as far as budgeting and solving need to know.

    The ``*_lifetime`` terms are the wear one thruster is qualified for, checked at build time
    (`prospector.spacecraft.buildability`). Both are optional; ``None`` means nobody said and the
    limit is not checked. Total propellant is what usually wears an electric thruster out, while
    ignitions matter for the escape spiral, which restarts roughly once per orbit as it passes
    through Earth's shadow.
    """

    name: str
    # Where these numbers came from, in the author's own words: a datasheet revision, a published
    # paper, a vendor test report, or "synthetic". Recorded per engine because the source decides
    # what may leave the building: without it nobody can tell a published figure from one supplied
    # under an agreement, and the whole library has to be treated as restricted. Empty means
    # unrecorded, which :func:`shareable_engines` treats as restricted.
    source: str = Field(default="", description="Provenance of these numbers (empty = unrecorded).")
    isp_s: float = Field(gt=0, description="Specific impulse (seconds).")
    thrust_mN: float = Field(gt=0, description="Thrust per engine (millinewtons).")
    power_W: float = Field(default=0.0, ge=0, description="Input power per engine (watts).")
    mass_kg: float = Field(default=0.0, ge=0, description="Dry mass per engine (kg).")
    # Rough cost of one thruster system ($M, including electronics and feed). A placeholder for
    # make designs comparable, not a quote. Lives with the engine (not a blanket build-model
    # constant) so a high-power and a low-power thruster can carry different unit costs.
    cost_musd: float = Field(default=1.5, ge=0, description="Thruster-system unit cost ($M).")
    # The working gas the isp_s and thrust_mN above were measured on; a key into the propellant
    # library (`prospector.spacecraft.propellants`). It does not re-derive performance (those
    # numbers are authored per gas); it sets the propellant's cost and storage density, which the
    # build model uses. Default xenon, the build model's reference gas.
    propellant: str = Field(default="xenon", description="Working-gas key (propellant library).")
    # Which bus the thruster's PPU is fed from (see PPU_BUS_SIDES). "low" is the default because it
    # is the deeper chain of the two: an engine whose wiring nobody recorded is charged the
    # step-down converter rather than credited a saving it may not have.
    ppu_bus_side: PPUBusSide = Field(
        default="low",
        description="Bus side the PPU is fed from: 'high' skips the HV->LV converter, 'low' pays it.")
    throughput_kg: float | None = Field(
        default=None, gt=0,
        description="Qualified propellant throughput per thruster (kg); None = unspecified.")
    lifetime_ignitions: int | None = Field(
        default=None, gt=0,
        description="Qualified ignition (on/off) cycles per thruster; None = unspecified.")
    # How the curve below is followed when power-limited (see THROTTLE_MODES). Continuous is the
    # default because it is what the curve meant before modes existed.
    throttle_mode: ThrottleMode = Field(
        default="continuous",
        description="'continuous': interpolate along the curve; 'discrete': the curve points are "
                    "fixed modes, run the highest one the power allows.")
    # Measured input-power -> (thrust, Isp) throttle curve, ascending in power; the rated ``isp_s``
    # / ``thrust_mN`` / ``power_W`` above are its top point. When the bus cannot supply rated power
    # the thruster runs at reduced power and both thrust and Isp drop along this curve (electric
    # thrusters lose efficiency at lower discharge power). ``None`` = no data; performance is then
    # treated as flat at the rated point (thrust may still be throttled at constant Isp).
    power_curve: list[PowerPoint] | None = Field(default=None)

    @property
    def discrete(self) -> bool:
        """Whether this thruster switches between fixed modes rather than throttling smoothly."""
        return self.throttle_mode == "discrete"

    def modes(self) -> list[PowerPoint]:
        """The operating points, ascending in power: the curve, or for a discrete thruster with no
        curve, its rated point alone. Empty for a continuous thruster without a curve."""
        if self.power_curve:
            return sorted(self.power_curve, key=lambda pp: pp.power_W)
        if self.discrete and self.power_W > 0:
            return [PowerPoint(power_W=self.power_W, thrust_mN=self.thrust_mN, isp_s=self.isp_s)]
        return []

    def performance_at(self, power_W: float) -> tuple[float, float]:
        """(thrust_mN, Isp_s) at an input power (W).

        Continuous with no curve: the flat rated (thrust, Isp) at any power. Continuous with a
        curve: linear interpolation between points; below the lowest / above the highest point it
        clamps to that end (the rated point caps the top; the stack decides how many engines to
        run below the operating floor; see :func:`assembly_performance_at_power`). Discrete: the
        highest mode whose power is at or below ``power_W``, at that mode's own thrust and Isp; the
        power in hand above the mode is unused. Below the lowest mode the thruster is off (zero
        thrust, the lowest mode's Isp for the record)."""
        pts = self.modes()
        if not pts:
            return self.thrust_mN, self.isp_s
        pw = [pp.power_W for pp in pts]
        if self.discrete:
            i = int(np.searchsorted(pw, float(power_W), side="right")) - 1
            if i < 0:
                return 0.0, pts[0].isp_s
            return pts[i].thrust_mN, pts[i].isp_s
        p = min(max(float(power_W), pw[0]), pw[-1])
        thrust = float(np.interp(p, pw, [pp.thrust_mN for pp in pts]))
        isp = float(np.interp(p, pw, [pp.isp_s for pp in pts]))
        return thrust, isp

    @property
    def min_power_W(self) -> float:
        """Lowest input power the thruster is characterized to run at (its operating floor): the
        bottom of the curve, or the rated power without one."""
        pts = self.modes()
        return pts[0].power_W if pts else self.power_W


def tsiolkovsky_dv(isp_s: float, wet_mass_kg: float, dry_mass_kg: float) -> float:
    """Ideal delta-v (km/s) from the rocket equation: g0 * Isp * ln(wet / dry).

    Returns 0 when there is no usable propellant (wet <= dry) rather than a negative number, so an
    impossible mass split can never give a budget that makes no sense.
    """
    if wet_mass_kg <= dry_mass_kg:
        return 0.0
    return G0_KM_S2 * isp_s * math.log(wet_mass_kg / dry_mass_kg)


def assembly_performance(mounts: list[tuple[str, int]], catalog: dict[str, Engine]) -> dict:
    """Blend a set of engines into one equivalent engine.

    ``mounts`` is a list of ``(engine_key, count)``. Thrust, power and engine mass simply add up.
    The equivalent Isp is total thrust over total propellant flow, ``sum(F) / sum(F / Isp)``, and
    that is the only blend that keeps both the total thrust and the total flow right, so the
    equivalent engine pushes as hard as the real set and burns propellant at the same rate. With
    only one type fitted it reduces to that engine's own Isp. Averaging Isp weighted by thrust
    would be wrong. Raises ``KeyError`` for an engine name that is not in the catalog.
    """
    total_thrust = total_flow = total_power = total_mass = 0.0
    for key, count in mounts:
        engine = catalog[key]
        total_thrust += count * engine.thrust_mN
        total_flow += count * engine.thrust_mN / engine.isp_s  # proportional to mass flow (g0 cancels)
        total_power += count * engine.power_W
        total_mass += count * engine.mass_kg
    isp_s = total_thrust / total_flow if total_flow else 0.0
    return {"isp_s": isp_s, "thrust_mN": total_thrust, "power_W": total_power, "mass_kg": total_mass}


def assembly_power_by_ppu_side(mounts: list[tuple[str, int]],
                               catalog: dict[str, Engine]) -> dict[str, float]:
    """The assembly's rated input power (W) split by which bus side each type's PPU taps.

    Returns a ``{side: power_W}`` mapping over :data:`PPU_BUS_SIDES`, one entry per side that draws
    anything, so it is empty for a stack that draws no power. The power-conversion chain differs
    between the two sides, so a stack that mixes them draws off the array through two chains at
    once and has to be priced per side rather than as one number, which is what
    `prospector.spacecraft.buildability.assembly_thruster_chain_eff` does with this. Raises
    ``KeyError`` for a missing engine key, like :func:`assembly_performance`.
    """
    by_side: dict[str, float] = {}
    for key, count in mounts:
        engine = catalog[key]
        power = count * engine.power_W
        if power:
            by_side[engine.ppu_bus_side] = by_side.get(engine.ppu_bus_side, 0.0) + power
    return by_side


def assembly_performance_at_power(mounts: list[tuple[str, int]], catalog: dict[str, Engine],
                                  available_power_W: float) -> dict:
    """What the engines can do when the bus can only supply ``available_power_W``.

    At or above rated power, full performance comes back. Below it they run power-limited, and how
    depends on the engine data. With a throttle curve, a set of identical thrusters runs as many as
    can stay inside their operating range, switches the rest off, and gives each
    ``available/n_active`` capped at rated, so both thrust and Isp drop along the curve; a discrete
    thruster then rounds that share down to the mode it has the power for
    (:meth:`Engine.performance_at`), so the stack steps rather than slides. A mixed set runs every
    thruster at the same fraction of its own rated power, switching none off (a discrete type whose
    share falls below its lowest mode contributes nothing). Without a curve, thrust scales with
    power and Isp stays put.

    Returns ``{thrust_mN, isp_s, n_active}``."""
    rated = assembly_performance(mounts, catalog)
    rated_power = rated["power_W"]
    n_total = sum(c for _, c in mounts)
    if rated_power <= 0.0 or available_power_W >= rated_power:
        return {"thrust_mN": rated["thrust_mN"], "isp_s": rated["isp_s"], "n_active": n_total}

    engines = [(catalog[k], c) for k, c in mounts]
    if not any(e.modes() for e, _ in engines):
        f = max(0.0, available_power_W / rated_power)            # no curve: linear thrust, flat Isp
        return {"thrust_mN": rated["thrust_mN"] * f, "isp_s": rated["isp_s"], "n_active": n_total}

    if len(mounts) == 1:                                         # uniform stack: shed to stay in band
        eng, n = engines[0]
        floor = eng.min_power_W
        n_active = min(n, int(available_power_W // floor)) if floor > 0 else n
        if n_active < 1:
            # Off: no thrust, and the Isp reported is the floor's, the last point the thruster
            # could run at. Not the rated Isp: an engine that cannot fire has not become more
            # efficient, and a rated figure here drew as an Isp step UP on every timeline exactly
            # where the array had fallen too far to run the thruster at all.
            floor_isp = eng.modes()[0].isp_s if eng.modes() else rated["isp_s"]
            return {"thrust_mN": 0.0, "isp_s": floor_isp, "n_active": 0}
        per_engine = min(eng.power_W, available_power_W / n_active)
        thrust_pe, isp = eng.performance_at(per_engine)
        return {"thrust_mN": n_active * thrust_pe, "isp_s": isp, "n_active": n_active}

    f = available_power_W / rated_power                          # mixed: same power fraction per type
    total_thrust = total_flow = 0.0
    for eng, count in engines:
        thrust_pe, isp_pe = eng.performance_at(f * eng.power_W)
        total_thrust += count * thrust_pe
        total_flow += count * thrust_pe / isp_pe
    return {"thrust_mN": total_thrust, "isp_s": (total_thrust / total_flow if total_flow else rated["isp_s"]),
            "n_active": n_total}


def assembly_power_grid(mounts: list[tuple[str, int]], catalog: dict[str, Engine],
                        points: int = 64):
    """A dense lookup of stack ``(input power W -> total thrust N, effective Isp s)`` from 0 to the
    stack's rated power, for the escape propagator to read the operating point at each step as the
    array degrades. Returns ``(power_W, thrust_N, isp_s)`` numpy arrays, or ``None`` when the stack
    draws no power. Built from :func:`assembly_performance_at_power`, so it captures the throttle
    curve (and engine shedding) or the curve-less linear-throttle fallback, whichever applies.

    Consumers read the grid with linear interpolation, which would smear a discrete stack's steps
    across the gap between two evenly spaced samples. So when any thruster switches modes, every
    power at which the stack's operating point can change (a mode reached by some number of active
    engines, or by a type's share of a mixed stack) is added as a node, paired with a node just
    below it, and the step survives the interpolation."""
    rated_power = assembly_performance(mounts, catalog)["power_W"]
    if rated_power <= 0.0:
        return None
    pw = np.linspace(0.0, rated_power, int(points))
    steps = _mode_breakpoints(mounts, catalog, rated_power)
    if steps:
        below = [b * (1.0 - 1e-9) for b in steps]
        pw = np.unique(np.concatenate([pw, np.asarray(steps + below, float)]))
        pw = pw[(pw >= 0.0) & (pw <= rated_power)]
    thrust_N, isp_s = [], []
    for p in pw:
        r = assembly_performance_at_power(mounts, catalog, float(p))
        thrust_N.append(r["thrust_mN"] * 1e-3)
        isp_s.append(r["isp_s"])
    return pw, np.asarray(thrust_N, float), np.asarray(isp_s, float)


def _mode_breakpoints(mounts: list[tuple[str, int]], catalog: dict[str, Engine],
                      rated_power: float) -> list[float]:
    """The stack input powers at which a discrete thruster somewhere in it changes mode.

    For a uniform stack of ``n``, ``k`` active engines reach a mode at ``k`` times its power; in a
    mixed stack a type's share is its rated fraction of the total, so it reaches a mode when the
    stack has ``rated_total * mode / rated_engine``. Both are included regardless of the stack's
    shape: an extra node is harmless, a missing one smears a step. Empty when nothing is discrete.
    """
    n_total = sum(c for _, c in mounts)
    points: set[float] = set()
    for key, _count in mounts:
        eng = catalog[key]
        if not eng.discrete:
            continue
        for mode in eng.modes():
            for k in range(1, n_total + 1):
                points.add(k * mode.power_W)
            if eng.power_W > 0:
                points.add(rated_power * mode.power_W / eng.power_W)
    return sorted(b for b in points if 0.0 < b <= rated_power)


# ---------------------------------------------------------------------------
# Engine library: one YAML per engine in configs/engines/, keyed by file stem.
# ---------------------------------------------------------------------------

def load_engines(engine_dir: str | Path | None = None) -> dict[str, Engine]:
    """The engine catalog: every ``*.yaml`` in ``engine_dir``, keyed by file stem."""
    engine_dir = Path(engine_dir) if engine_dir is not None else default_engine_dir()
    if not engine_dir.is_dir():
        raise FileNotFoundError(
            f"no engine library at {engine_dir}; the catalog is file-based and has no "
            f"built-in fallback")
    return {
        path.stem: Engine.model_validate(yaml.safe_load(path.read_text()))
        for path in sorted(engine_dir.glob("*.yaml"))
    }


def list_engines(engine_dir: str | Path | None = None) -> list[str]:
    """All engine keys, sorted."""
    return sorted(load_engines(engine_dir))


def save_engine(engine: Engine, key: str, engine_dir: str | Path | None = None) -> Path:
    """Persist ``engine`` to ``configs/engines/<key>.yaml`` and return the path."""
    engine_dir = Path(engine_dir) if engine_dir is not None else default_engine_dir()
    engine_dir.mkdir(parents=True, exist_ok=True)
    return dump_preserving_comments(engine.model_dump(mode="json"),
                                    engine_dir / f"{paths.config_name(key)}.yaml")


# Sources that make an engine's numbers safe to pass on. An engine whose `source` starts with one
# of these can be published; anything else, including a blank source, is treated as restricted,
# because after the fact there is no telling "nobody wrote it down" apart from "under NDA".
SHAREABLE_SOURCE_PREFIXES = ("synthetic", "public:")


def is_shareable(engine: Engine) -> bool:
    """Whether an engine's numbers may be passed on, judged only by the source recorded for them.

    Deliberately cautious: an engine with no source recorded is not shareable. The alternative,
    assuming an unlabelled engine is fine, gets a vendor's test data into a release once, and there
    is no taking it back.
    """
    return str(getattr(engine, "source", "") or "").strip().lower().startswith(
        SHAREABLE_SOURCE_PREFIXES)

