"""
How Earth's radiation belts damage the solar arrays. A selectable, file-based model.

The escape spiral climbs out through the belts and the trapped particles permanently damage the
cells. A study picks a scenario the way it picks an engine, and the spiral, the array sizing and
the report all use that one.

Three parts: the belt environment, keyed to the magnetic shell ``L = (rho^2 + z^2)^1.5 / (Re
rho^2)``; the cover glass over the cells, which is the biggest design lever; and the cell's
measured response, ``P/P0 = 1 - coef*log10(1 + Dd/ref)``.

Anchored to an OMERE 5.9 run, which it reproduces. ``docs/physics.md`` has the validation and its
caveats.

Positions are relative to Earth in km, ``(3,)`` or ``(N, 3)``. Vectorized. Dose rates in MeV/g per
day; power fractions 0 to 1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from prospector import paths
from prospector.constants import EARTH_MEAN_RADIUS_KM
from prospector.yamlio import dump_preserving_comments, validated


# The radiation library is file-based, one YAML per scenario, like engines and launch types.
def default_radiation_dir() -> Path:
    """The radiation-scenario library in the active config library, resolved per call."""
    return paths.config_dir() / "radiation"
# The cautious default: worst case for a low sun-synchronous spiral, so arrays size generously.
DEFAULT_RADIATION_MODEL = "ap8min-worstcase"


def _sigmoid(a):
    return 1.0 / (1.0 + np.exp(-np.clip(a, -60.0, 60.0)))


class RadiationModel(BaseModel):
    """One radiation-degradation scenario: belt environment, cover glass, cell response.

    A bare ``RadiationModel()`` reproduces the ``ap8min-worstcase`` preset. A study selects a
    preset and may override the cover thickness live; :meth:`with_coverglass` returns that variant.
    """

    name: str = "AP8-MIN worst case"
    cell: str = "GaAs/Ge single-junction (NRL DDD)"

    # --- belt geometry (dipole L-shell) ---
    re_km: float = Field(default=EARTH_MEAN_RADIUS_KM, gt=0, description="Mean Earth radius for the L-shell.")
    # Damaging-proton band: a plateau across (lo, hi) in L, sigmoid shoulders, sharp outer cliff
    # (Bourdarie 2021; Lozinski 2019: 1-10 MeV protons peak at L~1.8-2.5).
    proton_plateau_l: tuple[float, float] = (1.6, 2.7)
    proton_shoulder_l: float = Field(default=0.12, gt=0, description="Shoulder/cliff width (L units).")
    electron_peak_l: float = Field(default=4.5, gt=0, description="Outer electron belt core (L).")
    electron_sigma_l: float = Field(default=0.9, gt=0, description="Outer belt width (L units).")
    # Flux concentrates near the magnetic equator: flux ~ exp(-(B/B0 - 1)/scale), with B/B0 =
    # sqrt(1 + 3 sin^2 mlat) / cos^6 mlat. B/B0 climbs faster than latitude, which kills the
    # low-altitude field-line feet that L-shell alone would alias into the belt.
    bb0_scale: float = Field(default=2.0, gt=0, description="Magnetic-latitude concentration scale.")
    # No belt in the dense atmosphere: dose ramps from zero below ``floor_lo`` to full by
    # ``floor_hi``. The South Atlantic Anomaly is not modeled.
    belt_floor_lo_km: float = Field(default=1000.0, ge=0)
    belt_floor_hi_km: float = Field(default=2500.0, gt=0)

    # --- core DDD rates (MeV/g per day) at the reference coverglass ---
    # Both anchored to the OMERE reference run at the reference cover. Damage in GaAs is
    # overwhelmingly proton (electrons ~0.12% of the dose), so the electron core sits ~800x below
    # the proton core, not the ~10:1 of the raw belt fluxes.
    proton_ddd_core: float = Field(default=4.6e9, ge=0)
    electron_ddd_core: float = Field(default=5.8e6, ge=0)

    # --- cell response: Pmax fraction vs cumulative DDD ---
    # GaAs/Ge single junction, NRL method: P/P0 = 1 - coef*log10(1 + Dd/ref). The NRL Pmax proton
    # curve (C = 0.2904, Dx = 1.10e9 MeV/g), checked against OMERE 5.9 to 4 figures. Applied to the
    # total dose, since protons dominate.
    cell_ddd_ref: float = Field(default=1.1e9, gt=0)
    cell_ddd_coef: float = Field(default=0.2904, gt=0)

    # --- coverglass shielding (the dominant lever) ---
    # Proton stopping goes by areal density (g/cm^2 = thickness x density), not thickness alone.
    # The core rates above are anchored at the reference areal density
    # (``coverglass_ref_um`` x ``coverglass_ref_density_g_cm3``). Defaults describe a multi-layer
    # FEP/adhesive stack: 212.5 um at a mean 1.64 g/cm3. Both are live knobs.
    coverglass_um: float = Field(default=212.5, gt=0, description="Cover total thickness (um).")
    coverglass_density_g_cm3: float = Field(default=1.640, gt=0,
                                            description="Cover thickness-weighted mean density (g/cm3).")
    coverglass_ref_um: float = Field(default=75.0, gt=0,
                                     description="Reference cover thickness the core rates anchor at (um).")
    coverglass_ref_density_g_cm3: float = Field(default=1.9866, gt=0,
                                                description="Reference cover density (g/cm3).")
    # Shielded DDD ~ (areal density)^(-exponent), from a soft proton spectrum phi(>E) ~ E^-1.5,
    # NIEL ~ E^-0.7 and a range law R ~ E^1.75, giving sigma^-1.25. Belt electrons out-range a thin
    # cover, so their exponent is small.
    proton_shield_exponent: float = Field(default=1.25, ge=0)
    electron_shield_exponent: float = Field(default=0.3, ge=0)

    # ------------------------------------------------------------------ shielding

    def areal_density(self) -> float:
        """Cover areal density (g/cm^2): thickness x mean density."""
        return self.coverglass_um * self.coverglass_density_g_cm3 * 1e-4

    def _ref_areal_density(self) -> float:
        return self.coverglass_ref_um * self.coverglass_ref_density_g_cm3 * 1e-4

    def proton_shield(self) -> float:
        """Scale on the proton core rate from the cover's areal density (1.0 at the reference).

        ``(areal / reference_areal)^(-proton_shield_exponent)``; see the field comment above for
        the derivation."""
        return float((self.areal_density() / self._ref_areal_density()) ** (-self.proton_shield_exponent))

    def electron_shield(self) -> float:
        """Scale on the electron core DDD from the cover's areal density (1.0 at the reference).

        Far weaker than the proton term, since belt electrons out-range a thin cover and it barely
        shields them; the small exponent captures only the soft tail."""
        return float((self.areal_density() / self._ref_areal_density()) ** (-self.electron_shield_exponent))

    def with_coverglass(self, coverglass_um: float, density_g_cm3: float | None = None) -> RadiationModel:
        """A copy of this model at a different cover thickness, and optionally density, for the live
        design knobs. ``density_g_cm3=None`` keeps the current mean density."""
        upd = {"coverglass_um": float(coverglass_um)}
        if density_g_cm3 is not None:
            upd["coverglass_density_g_cm3"] = float(density_g_cm3)
        return self.model_copy(update=upd)

    # ------------------------------------------------------------------ environment

    def lshell(self, positions_km):
        """Dipole McIlwain L-shell of geocentric position(s): ``(rho^2 + z^2)^1.5 / (Re rho^2)``.

        L is the equatorial crossing distance (Earth radii) of the field line through the point:
        at the equator ``L = r/Re``; off the equator L rises, running to infinity over the poles."""
        p = np.asarray(positions_km, float)
        rho = np.maximum(np.hypot(p[..., 0], p[..., 1]), 1.0)   # guard the spin axis (rho -> 0)
        r = np.hypot(rho, p[..., 2])
        return r ** 3 / (self.re_km * rho ** 2)

    def rate_components(self, positions_km):
        """The (proton, electron) dose rates (MeV/g per day) at position(s).

        Each belt is a function of L-shell, scaled by its core rate and shield factor, then by
        magnetic latitude and the atmospheric floor. So the rate peaks at equatorial crossings and
        is near zero over the poles: a near-polar orbit takes its dose only as it punches through
        the equator at belt altitude."""
        p = np.asarray(positions_km, float)
        rho = np.maximum(np.hypot(p[..., 0], p[..., 1]), 1.0)
        r = np.hypot(rho, p[..., 2])
        L = r ** 3 / (self.re_km * rho ** 2)
        lo, hi = self.proton_plateau_l
        proton = _sigmoid((L - lo) / self.proton_shoulder_l) * _sigmoid((hi - L) / self.proton_shoulder_l)
        electron = np.exp(-0.5 * ((L - self.electron_peak_l) / self.electron_sigma_l) ** 2)
        bb0 = np.sqrt(1.0 + 3.0 * (p[..., 2] / r) ** 2) / (rho / r) ** 6   # field strength vs equator
        conc = np.exp(-(bb0 - 1.0) / self.bb0_scale)                       # concentrated near equator
        floor = np.clip((r - self.re_km - self.belt_floor_lo_km)
                        / (self.belt_floor_hi_km - self.belt_floor_lo_km), 0.0, 1.0)
        envelope = conc * floor
        return (self.proton_ddd_core * self.proton_shield() * proton * envelope,
                self.electron_ddd_core * self.electron_shield() * electron * envelope)

    def ddd_rate(self, positions_km):
        """Total dose rate (MeV/g per day), the proton and electron components added."""
        proton, electron = self.rate_components(positions_km)
        return proton + electron

    def proton_core_shielded(self) -> float:
        """The shielded proton core dose rate, the in-belt reference the thresholds use."""
        return float(self.proton_ddd_core * self.proton_shield())

    # ------------------------------------------------------------------ cell response

    def cell_power_fraction(self, ddd):
        """Remaining cell Pmax fraction at cumulative DDD (MeV/g): ``1 - coef*log10(1 + Dd/ref)``,
        clamped to [0, 1]."""
        frac = 1.0 - self.cell_ddd_coef * np.log10(1.0 + np.asarray(ddd, float) / self.cell_ddd_ref)
        return np.clip(frac, 0.0, 1.0)

    def power_profile(self, positions_km, times_days):
        """Solar-array power fraction over a flown path.

        Integrates :meth:`ddd_rate` and converts the cumulative dose with
        :meth:`cell_power_fraction`. The fraction steps down at each equator crossing and holds
        flat between, since the damage is permanent. Returns ``(in_belt, ddd_cumulative,
        power_fraction)``; ``in_belt`` flags samples with a non-negligible rate, for shading.
        Empty when the path is too short."""
        pos = np.asarray(positions_km, float)
        t = np.asarray(times_days, float)
        if pos.ndim != 2 or len(pos) < 2 or len(t) != len(pos):
            empty = np.empty(0, float)
            return empty, empty, empty
        rate = self.ddd_rate(pos)                               # MeV/g per day
        dt = np.diff(t)
        seg = dt * 0.5 * (rate[:-1] + rate[1:])                 # trapezoid dose per interval
        ddd_cum = np.concatenate(([0.0], np.cumsum(seg)))
        power_fraction = self.cell_power_fraction(ddd_cum)
        in_belt = rate > 0.02 * max(self.proton_core_shielded(), 1.0)
        return in_belt, ddd_cum, power_fraction


# ---------------------------------------------------------------------------
# library: one YAML per scenario in configs/radiation/, keyed by file stem.
# ---------------------------------------------------------------------------

def load_radiation_models(radiation_dir: str | Path | None = None) -> dict[str, RadiationModel]:
    """The radiation-scenario catalog: every ``*.yaml`` in ``radiation_dir``, keyed by file stem.

    Raises :class:`FileNotFoundError` when the directory is absent, since an empty catalog would
    let every caller fall through to built-in defaults."""
    radiation_dir = Path(radiation_dir) if radiation_dir is not None else default_radiation_dir()
    if not radiation_dir.is_dir():
        raise FileNotFoundError(
            f"no radiation-model library at {radiation_dir}; the belt-degradation physics is "
            f"file-based and has no built-in fallback")
    return {path.stem: validated(RadiationModel, path)
            for path in sorted(radiation_dir.glob("*.yaml"))}


def list_radiation_models(radiation_dir: str | Path | None = None) -> list[str]:
    """All radiation-scenario keys, sorted."""
    return sorted(load_radiation_models(radiation_dir))


def load_radiation_model(name: str | None = None,
                         radiation_dir: str | Path | None = None) -> RadiationModel:
    """One scenario by key, or the configured default when ``name`` is None.

    Raises :class:`KeyError` for an unknown key. Substituting a different scenario would silently
    change the dose the spiral flies and the array it sizes."""
    key = name or DEFAULT_RADIATION_MODEL
    catalog = load_radiation_models(radiation_dir)
    if key not in catalog:
        raise KeyError(
            f"no radiation model {key!r} in {radiation_dir}; "
            f"available: {', '.join(sorted(catalog)) or '(none)'}")
    return catalog[key]


def default_radiation_model() -> RadiationModel:
    """The default scenario (the worst-case inner-belt physics-of-record)."""
    return load_radiation_model(DEFAULT_RADIATION_MODEL)


def save_radiation_model(model: RadiationModel, key: str,
                         radiation_dir: str | Path | None = None) -> Path:
    """Persist ``model`` to ``configs/radiation/<key>.yaml`` and return the path.

    The belt physics is a tuning knob like the sizing coefficients, so it has to be writable from
    the app rather than only by hand."""
    radiation_dir = Path(radiation_dir) if radiation_dir is not None else default_radiation_dir()
    radiation_dir.mkdir(parents=True, exist_ok=True)
    return dump_preserving_comments(model.model_dump(mode="json"),
                                    radiation_dir / f"{key}.yaml")
