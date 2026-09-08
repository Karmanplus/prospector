"""The transfer grid coloured by propellant margin: kilograms left after the cruise, hot for more,
grey for a cell that comes up short, with the star on the cell that leaves the most. The quantity
that decides whether a cell closes once the escape and the cruise run at different Isps.

The three views share one colour language: the same ramp for what this vehicle can fly, the same
grey gradient for a real trajectory it cannot afford, and one dark slab for cells never found.
"""
import numpy as np

from prospector import figures
from prospector.figures.porkchop import UNAFFORDABLE_SCALE

DEP = np.array([10000.0, 10010.0])
TOF = np.array([300.0, 350.0, 400.0])
# Final masses against a 336 kg burnout: -6, +4, +14 / -16, -1, +9 kg of margin.
MASS = np.array([[330.0, 340.0, 350.0], [320.0, 335.0, 345.0]])
DV = np.array([[4.0, 3.6, 3.2], [4.4, 3.9, 3.5]])
OK = np.ones_like(DV, dtype=bool)
FRONTIER = [{"tof_days": 300.0, "dv_kms": 4.0, "dep_mjd2000": 10000.0, "final_mass_kg": 330.0},
            {"tof_days": 350.0, "dv_kms": 3.6, "dep_mjd2000": 10000.0, "final_mass_kg": 340.0},
            {"tof_days": 400.0, "dv_kms": 3.2, "dep_mjd2000": 10000.0, "final_mass_kg": 350.0}]


def _traces(fig):
    return {t.name: t for t in fig.data}


def test_margin_view_colours_closing_cells_and_greys_the_short_ones():
    fig = figures.converged_grid(DEP, TOF, dv_kms=DV, final_mass_kg=MASS, feasible=OK,
                                 frontier=FRONTIER, colour_by="margin", burnout_mass_kg=336.0,
                                 dv_budget=3.3)
    tr = _traces(fig)
    field = tr["field"]
    assert field.colorbar.title.text == "propellant margin (kg)"
    # Same ramp as the other views, hot for good, cold end at (or above) zero margin.
    mass_view = figures.converged_grid(DEP, TOF, dv_kms=DV, final_mass_kg=MASS, feasible=OK,
                                       frontier=FRONTIER, colour_by="mass")
    assert field.colorscale == _traces(mass_view)["field"].colorscale
    assert not field.reversescale
    assert field.zmin >= 0.0 and field.zmax == 14.0
    # Closing cells carry their margin; short cells are blank in the field ...
    z = np.asarray(field.z, float)
    np.testing.assert_allclose(z[MASS >= 336.0], (MASS - 336.0)[MASS >= 336.0])
    assert np.isnan(z[MASS < 336.0]).all()
    # ... and drawn in the grey gradient by how short they are: -6, -16, -1 kg.
    short = tr["short of propellant"]
    assert [list(c) for c in short.colorscale] == [list(c) for c in UNAFFORDABLE_SCALE]
    sz = np.asarray(short.z, float)
    np.testing.assert_allclose(sz[MASS < 336.0], [6.0, 16.0, 1.0])
    assert np.isnan(sz[MASS >= 336.0]).all()
    assert short.zmin == 0.0 and short.zmax == 16.0
    assert "over budget" not in tr                    # the ΔV budget plays no part in this view
    # Hover says how short, not a verdict on the transfer; the star marks the most margin.
    cells = tr["cells"]
    assert any("+14 kg margin" in t for t in cells.text)
    assert any("16 kg short of propellant" in t for t in cells.text)
    assert "most margin: +14 kg" in tr["best"].hovertemplate


def test_dv_view_still_greys_over_budget_and_stars_the_cheapest():
    fig = figures.converged_grid(DEP, TOF, dv_kms=DV, final_mass_kg=MASS, feasible=OK,
                                 frontier=FRONTIER, colour_by="dv", burnout_mass_kg=336.0,
                                 dv_budget=3.7)
    names = _traces(fig)
    assert names["field"].colorbar.title.text == "ΔV (km/s)"
    assert "over budget" in names                                     # 4.0, 4.4, 3.9 recede
    assert [list(c) for c in names["over budget"].colorscale] == [list(c) for c in UNAFFORDABLE_SCALE]
    assert "cheapest 3.20 km/s" in names["best"].hovertemplate
    assert any("kg margin" in t for t in names["cells"].text), "the margin rides in every hover"


def test_margin_view_with_every_cell_closing_greys_nothing():
    fig = figures.converged_grid(DEP, TOF, dv_kms=DV, final_mass_kg=MASS + 20.0, feasible=OK,
                                 frontier=FRONTIER, colour_by="margin", burnout_mass_kg=336.0)
    tr = _traces(fig)
    assert "short of propellant" not in tr
    np.testing.assert_allclose(np.asarray(tr["field"].z, float), MASS + 20.0 - 336.0)


def test_margin_without_a_burnout_mass_falls_back_to_delivered_mass():
    fig = figures.converged_grid(DEP, TOF, dv_kms=DV, final_mass_kg=MASS, feasible=OK,
                                 frontier=FRONTIER, colour_by="margin")
    field = _traces(fig)["field"]
    assert field.colorbar.title.text == "delivered mass (kg)"
    assert "most delivered: 350 kg" in _traces(fig)["best"].hovertemplate
