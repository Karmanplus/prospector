"""
Solar arrays: how much power they make at a given distance from the Sun, and what they weigh.

The array is the electric vehicle's engine room, and two things about it need models rather than
flat assumptions.

How output varies with distance. Sunlight falls off as ``1/r^2``, but the panel also runs colder
further out, and cells get more efficient as they cool, so power falls off more slowly than
``1/r^2`` alone. The panel's temperature comes from balancing the sunlight it absorbs on the front
against what both faces radiate away, ``T = (alpha_f I / (sigma (eps_f + eps_r)))^(1/4)``, and cell
efficiency is a straight-line fit against temperature. Close to Earth, sunlight bounced off the
planet and Earth's own infrared warm the back of the panel too, scaled down by how much of the sky
Earth fills, ``(R_E / r_geo)^2``. That warm-panel penalty only matters early in the escape spiral
and fades within a few Earth radii.

What an array weighs. Real arrays do not come at one fixed W/kg: small wings carry fixed overheads
and very large ones are built differently altogether. So mass comes from a piecewise straight-line
curve of power against mass across vendor array families, with a single ``mass_scale`` knob to
sweep technology optimism. The curve and the optical constants belong in config, not here: the
defaults below are illustrative round numbers so the model works without a library, and real
measured values go in the build model.

A vehicle's ``solar_power_W`` always means its start-of-life output at 1 AU in deep space, at the
temperature the panel actually runs there, so everything here is a fraction relative to that:
:meth:`ArrayModel.power_fraction` is 1.0 at 1 AU in deep space, about 0.74 at 1.2 AU and about
0.50 at 1.5 AU. Radiation damage is not modelled here, since `prospector.spacecraft.radiation`
owns it, and the two fractions multiply wherever power is used.

Rated against operating power. A datasheet quotes an array's wattage at a standard cell
temperature (``rating_temp_C``, 28 degC by convention), but a panel facing the Sun at 1 AU settles
well above that (around 60 degC for typical optical constants), where the cells convert several
percent worse. :meth:`ArrayModel.operating_over_rated` is that ratio, so an array which must
DELIVER a given power in flight has to be rated (bought, weighed, and given area) for
``operating / ratio``. Sizing goes through it; the flight physics never see it, since they start
from the operating figure.

Sun distances in AU, distances from Earth in km, temperatures in Celsius, power in W, mass in kg.
Everything is vectorized: scalar in, float out; array in, array out. No UI framework, no ephemeris,
and no prospector imports.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from prospector.constants import (
    EARTH_MEAN_RADIUS_KM,
    KELVIN_OFFSET_C,
    SOLAR_CONSTANT_W_M2,
    STEFAN_BOLTZMANN_W_M2_K4,
)


class ArrayMassSegment(BaseModel):
    """One piece of the power-to-mass curve.

    ``[lo_W, hi_W]`` says which powers this piece covers; the mass is a straight line through the
    two points ``(w0, kg0)`` and ``(w1, kg1)``. Those points need not sit at the edges of the
    range, since a vendor family's fit line can extend past the powers it is used over, and that is
    also what lets neighbouring pieces jump where the construction changes to a different kind of
    array.
    """

    lo_W: float = Field(ge=0)
    hi_W: float = Field(gt=0)
    w0: float
    kg0: float
    w1: float
    kg1: float

    def mass_kg(self, power_W: float) -> float:
        return self.kg0 + (self.kg1 - self.kg0) / (self.w1 - self.w0) * (power_W - self.w0)


# An illustrative curve, not a vendor's: round numbers at roughly 100 W/kg with a floor for small
# wings, so :class:`ArrayModel` can be built without a config library.
#
# The real curve belongs in the config library as ``array_mass_curve``. This one is smooth, which
# a real family curve is not, since it jumps where the construction changes. Inventing
# plausible-looking jumps here would misrepresent hardware nobody has quoted.
EXAMPLE_MASS_CURVE: list[ArrayMassSegment] = [
    ArrayMassSegment(lo_W=50.0, hi_W=500.0, w0=50.0, kg0=1.0, w1=500.0, kg1=5.0),
    ArrayMassSegment(lo_W=500.0, hi_W=5000.0, w0=500.0, kg0=5.0, w1=5000.0, kg1=50.0),
    ArrayMassSegment(lo_W=5000.0, hi_W=100000.0, w0=5000.0, kg0=50.0, w1=100000.0, kg1=1000.0),
]


class ArrayModel(BaseModel):
    """The array's optical, thermal and electrical constants, plus its power-to-mass curve.

    Every default here is illustrative rather than measured: round numbers for a generic thin-film
    panel, chosen so the model can be built and behaves sensibly without a config library. The real
    values live in the build model (`prospector.spacecraft.buildability`), which is a file in the
    config library, so an array's measured optical response and efficiency fit can be entered
    without putting vendor data in source.

    ``mass_scale`` is the one knob to sweep: a technology multiplier on the whole curve.
    """

    # -- optical / thermal (radiative balance) --
    alpha_front: float = Field(default=0.90, gt=0, description="Front-face solar absorptivity.")
    epsilon_front: float = Field(default=0.85, gt=0, description="Front-face IR emissivity.")
    alpha_rear: float = Field(default=0.30, ge=0, description="Rear-face solar absorptivity.")
    epsilon_rear: float = Field(default=0.85, gt=0, description="Rear-face IR emissivity.")
    # Heating of the panel's back face near Earth, scaled down by how much of the sky Earth fills,
    # (R_E / r_geo)^2. Both are properties of Earth rather than of any panel, so they never change.
    earth_albedo_W_m2: float = Field(default=0.3 * SOLAR_CONSTANT_W_M2, ge=0,
                                     description="Earth-reflected sunlight at the surface (~30% of solar).")
    earth_ir_W_m2: float = Field(default=250.0, ge=0, description="Earth infrared at low altitude.")

    # -- cell electrical response: efficiency(T) = (slope * T_C + intercept) / 100 --
    cell_eff_slope_pct_per_C: float = Field(default=-0.06,
                                            description="Cell efficiency slope (%/degC; negative = hot is worse).")
    cell_eff_intercept_pct: float = Field(default=30.0, gt=0,
                                          description="Cell efficiency at 0 degC (%).")
    # The temperature a datasheet wattage is quoted at. Standard test conditions are 28 degC, and
    # a sunlit panel at 1 AU runs hotter, so rated power is derated for flight (see
    # :meth:`operating_over_rated`).
    rating_temp_C: float = Field(default=28.0,
                                 description="Cell temperature the rated (datasheet) power is quoted at (degC).")

    # -- mass --
    mass_curve: list[ArrayMassSegment] = Field(default_factory=lambda: list(EXAMPLE_MASS_CURVE))
    mass_scale: float = Field(default=1.0, gt=0,
                              description="Multiplier on the whole power->mass curve (the sweep knob).")

    # ------------------------------------------------------------------ power physics

    def irradiance_W_m2(self, r_sun_au):
        """Solar irradiance at heliocentric distance(s): ``1361 / r^2`` (W/m^2)."""
        r = np.asarray(r_sun_au, float)
        return SOLAR_CONSTANT_W_M2 / r ** 2

    def panel_temperature_C(self, r_sun_au, r_geo_km=None):
        """What temperature the panel settles at (degC), pointed at the Sun.

        Heat in against heat out, per unit area. The front face absorbs sunlight (``alpha_f I``);
        near Earth the back face also absorbs reflected sunlight and Earth's infrared, scaled by
        how much of the sky Earth fills, ``(R_E / r_geo)^2`` (``r_geo_km=None`` means deep space,
        with no Earth term). Both faces radiate away, so ``T^4 = heat_in / (sigma (eps_f +
        eps_r))``.
        """
        heat_in = self.alpha_front * self.irradiance_W_m2(r_sun_au)
        if r_geo_km is not None:
            view = np.minimum((EARTH_MEAN_RADIUS_KM / np.maximum(np.asarray(r_geo_km, float),
                                                            EARTH_MEAN_RADIUS_KM)) ** 2, 1.0)
            heat_in = heat_in + view * (self.alpha_rear * self.earth_albedo_W_m2
                                        + self.epsilon_rear * self.earth_ir_W_m2)
        temp_k = (heat_in / (STEFAN_BOLTZMANN_W_M2_K4 * (self.epsilon_front + self.epsilon_rear))) ** 0.25
        return temp_k - KELVIN_OFFSET_C

    def cell_efficiency(self, temp_C):
        """How efficiently the cells convert sunlight at a given panel temperature: the measured
        straight-line fit, floored at zero, since a panel hot enough to zero the fit makes
        nothing."""
        eff = (self.cell_eff_slope_pct_per_C * np.asarray(temp_C, float)
               + self.cell_eff_intercept_pct) / 100.0
        return np.clip(eff, 0.0, None)

    def power_fraction(self, r_sun_au, r_geo_km=None):
        """Array output at a given distance from the Sun, as a fraction of its output at 1 AU.

        ``(1/r^2) * eta(T(r)) / eta(T(1 AU))``: sunlight falling off, times the efficiency the
        cells gain from running cold, scaled so the answer is 1.0 at 1 AU in deep space, which is
        where the vehicle's ``solar_power_W`` is quoted. Falls to about 0.74 at 1.2 AU and 0.50 at
        1.5 AU, and rises above 1 closer in than 1 AU, which a caller may want to cap. The optional
        near-Earth term (``r_geo_km``) can only lower it, since it warms the panel, and dies away
        as ``(R_E/r_geo)^2``. Radiation damage is not included here; multiply by
        `prospector.spacecraft.radiation`'s fraction for that.
        """
        r = np.asarray(r_sun_au, float)
        eta_ref = self.cell_efficiency(self.panel_temperature_C(1.0))
        eta = self.cell_efficiency(self.panel_temperature_C(r, r_geo_km))
        return (1.0 / r ** 2) * eta / eta_ref

    # ------------------------------------------------------------------ rated vs operating

    def operating_temp_C(self) -> float:
        """The temperature the panel runs at where its operating power is quoted: facing the Sun at
        1 AU in deep space, no Earth heating."""
        return float(self.panel_temperature_C(1.0))

    def operating_over_rated(self) -> float:
        """How much of its rated (datasheet) power the array delivers at 1 AU in deep space.

        The cell-efficiency fit evaluated at the panel's actual 1 AU temperature over the same fit
        at ``rating_temp_C``. Below 1 when the panel runs hotter than the rating temperature, as it
        does for any ordinary panel rated at 28 degC; exactly 1 when the two temperatures agree, so
        setting ``rating_temp_C`` to the operating temperature makes a rated wattage an operating
        one. Floored just above zero so a pathological fit cannot divide by nothing.
        """
        eta_rated = float(self.cell_efficiency(self.rating_temp_C))
        eta_operating = float(self.cell_efficiency(self.operating_temp_C()))
        return eta_operating / max(eta_rated, 1e-9)

    def rated_power_W(self, operating_power_W: float) -> float:
        """The datasheet wattage an array must carry to deliver ``operating_power_W`` at 1 AU in
        deep space: what sizing (mass, area, cost) is charged against."""
        return float(operating_power_W) / self.operating_over_rated()

    # ------------------------------------------------------------------ mass

    def mass_kg(self, rated_power_W: float) -> float:
        """What an array of this rated (datasheet) power weighs (kg), from the piecewise curve.

        The curve is a vendor's, quoted against catalogue wattage, so it is read at the RATED
        power; callers holding an operating figure convert with :meth:`rated_power_W` first.
        Power below the smallest piece is charged at that piece's floor, which stands for the
        smallest wing anyone would build; power above the largest continues along the last family's
        fit line. Scaled by ``mass_scale``.
        """
        segs = self.mass_curve or EXAMPLE_MASS_CURVE
        p = max(float(rated_power_W), segs[0].lo_W)
        for seg in segs:
            if p <= seg.hi_W:
                return self.mass_scale * seg.mass_kg(p)
        return self.mass_scale * segs[-1].mass_kg(p)

    def specific_power_W_kg(self, rated_power_W: float) -> float:
        """Rated watts per kilogram at a design's rated power, for reporting."""
        mass = self.mass_kg(rated_power_W)
        return float(rated_power_W) / mass if mass > 0 else 0.0


def default_array_model() -> ArrayModel:
    """An array model on the illustrative defaults, for callers with no config library.

    Normal use builds the model from the config library instead
    (``buildability.BusModel.array_model``), so this is a starting point rather than the real
    thing."""
    return ArrayModel()
