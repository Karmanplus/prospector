"""The major planets as targets: their elements, the composed population, and their ephemeris.

The planets are an authored catalog rather than a query, so the elements themselves need pinning
against an independent source, PyKEP's ``jpl_lp`` series, which is fitted from the same published
table. That cross-check is the real test here: it proves the derived argument of perihelion and
mean anomaly (computed from published longitudes) are right, which no amount of internal
consistency would.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from prospector.population import catalog, planets
from prospector.population.sbdb import FIELDS


def test_the_catalog_carries_the_population_schema():
    """A planet row has to look exactly like a small-body row to everything above, or every
    solver would need a second code path for it."""
    df = planets.planet_population()
    assert set(FIELDS) <= set(df.columns)
    assert len(df) == 7                                   # the majors, Earth excluded
    assert "Earth" not in set(df["pdes"])                 # the origin of every transfer, not a target
    assert set(df[planets.BODY_CLASS_COL]) == {planets.PLANET}
    assert df[["a", "e", "i", "om", "w", "ma", "epoch", "H"]].notna().all().all()
    assert (df["e"] < 1.0).all() and (df["a"] > 0).all()
    assert planets.list_planets()[:3] == ["Mercury", "Venus", "Mars"]   # outward from the Sun


def test_elements_match_an_independent_ephemeris_at_their_epoch():
    """The cross-check that makes the authored table trustworthy: every element, including the
    ``w``/``ma`` this module DERIVES from published longitudes, must agree with PyKEP's fitted
    series at J2000. A transposed or unwrapped angle would show up here and nowhere else."""
    import pykep as pk

    epoch = pk.epoch(planets.J2000_JD, pk.epoch.julian_type.JD).mjd2000
    for _, row in planets.planet_population().iterrows():
        # KEP_M asks for the sixth element as a MEAN anomaly, which is what the table stores.
        body = pk.planet(pk.udpla.jpl_lp(row["jpl_lp_key"]))
        a, e, i, om, w, ma = body.elements(epoch, pk.el_type.KEP_M)
        assert row["a"] == pytest.approx(a / pk.AU, abs=1e-9)
        assert row["e"] == pytest.approx(e, abs=1e-9)
        assert row["i"] == pytest.approx(np.degrees(i), abs=1e-7)
        for mine, theirs in ((row["om"], om), (row["w"], w), (row["ma"], ma)):
            diff = abs(mine - np.degrees(theirs) % 360.0)
            assert min(diff, 360.0 - diff) < 1e-7, row["pdes"]


def test_mars_is_where_mars_should_be():
    # A blunt sanity anchor independent of the table's internals: Mars orbits at ~1.52 AU with a
    # small eccentricity, tilted ~1.85 deg to the ecliptic.
    mars = planets.planet_population().set_index("pdes").loc["Mars"]
    assert mars["a"] == pytest.approx(1.524, abs=0.001)
    assert mars["e"] == pytest.approx(0.0934, abs=0.001)
    assert mars["i"] == pytest.approx(1.85, abs=0.01)
    assert int(mars["spkid"]) == 499


# ---------------------------------------------------------------------------
# the composed population
# ---------------------------------------------------------------------------

def _small_bodies() -> pd.DataFrame:
    return pd.DataFrame({
        "spkid": [2099942], "pdes": ["99942"], "full_name": ["99942 Apophis"],
        "H": [19.7], "condition_code": [0], "a": [0.9224], "e": [0.1914], "i": [3.331],
        "om": [204.4], "w": [126.7], "ma": [180.0], "epoch": [2460800.5],
    })


def test_both_sources_compose_into_one_frame(monkeypatch):
    monkeypatch.setattr(catalog, "fetch_population", lambda **kw: _small_bodies())
    df = catalog.load_population(h_max=22.0)
    assert len(df) == 1 + 7
    # Small bodies keep their leading order, so an existing caller sees no reordering.
    assert df.iloc[0]["pdes"] == "99942"
    assert set(df[planets.BODY_CLASS_COL]) == {planets.ASTEROID, planets.PLANET}


def test_the_brightness_cutoff_does_not_apply_to_planets(monkeypatch):
    """``h_max`` is a SIZE proxy, which it is for an asteroid and is not for a planet. So the
    cutoff must move only the small-body tail; a planet is present at any cutoff, and present
    because it is a planet, not because it happens to be bright."""
    monkeypatch.setattr(catalog, "fetch_population", lambda **kw: _small_bodies())
    for h_max in (15.0, 25.0, 33.0):
        df = catalog.load_population(h_max=h_max)
        assert "Mars" in set(df["pdes"])
    # Excluding them is explicit, and the body_class column is present either way so no consumer
    # has to handle its absence.
    small_only = catalog.load_population(h_max=25.0, include_planets=False)
    assert set(small_only[planets.BODY_CLASS_COL]) == {planets.ASTEROID}
    assert "Mars" not in set(small_only["pdes"])


def test_planets_survive_an_empty_small_body_result(monkeypatch):
    # Concatenating onto an empty frame would drop dtypes, so the planets stand alone instead.
    monkeypatch.setattr(catalog, "fetch_population", lambda **kw: _small_bodies().iloc[0:0])
    df = catalog.load_population()
    assert len(df) == 7 and set(df[planets.BODY_CLASS_COL]) == {planets.PLANET}


def test_body_class_tagging_is_idempotent():
    once = catalog.with_body_class(_small_bodies())
    assert catalog.with_body_class(once)[planets.BODY_CLASS_COL].tolist() == [planets.ASTEROID]
    # An existing planet tag is preserved, never overwritten with 'asteroid'.
    tagged = planets.planet_population()
    assert set(catalog.with_body_class(tagged)[planets.BODY_CLASS_COL]) == {planets.PLANET}


# ---------------------------------------------------------------------------
# body-class dispatch: what changes downstream because a target is a planet
# ---------------------------------------------------------------------------

def test_a_planet_is_recognisable_from_its_row():
    mars = planets.planet_population().set_index("pdes").loc["Mars"].to_dict()
    assert planets.is_planet(mars) and planets.jpl_lp_key(mars) == "mars"
    # A small-body row is not a planet, with or without the column present.
    rock = _small_bodies().iloc[0].to_dict()
    assert not planets.is_planet(rock) and planets.jpl_lp_key(rock) is None
    assert not planets.is_planet(catalog.with_body_class(_small_bodies()).iloc[0].to_dict())


def test_a_planet_is_propagated_by_the_fitted_series_not_two_body():
    """Two-body propagation of MEAN elements over a years-long transfer misplaces a planet, and a
    rendezvous is a phasing problem, so that error lands on the arrival date. A planet row must
    reach the fitted ephemeris instead."""
    import pykep as pk

    from prospector.solvers import lambert as lb
    mars_row = planets.planet_population().set_index("pdes").loc["Mars"].to_dict()
    body = lb.planet_from_row(mars_row)
    # A pykep planet type-erases the body behind it, so ask which one it is holding.
    assert body.extract(pk.udpla.jpl_lp) is not None
    # Well away from the element epoch the two disagree by a real distance, which is the reason the
    # routing exists, so it is asserted rather than assumed.
    two_body = lb.target_planet(mars_row["a"], mars_row["e"], mars_row["i"], mars_row["om"],
                                mars_row["w"], mars_row["ma"], mars_row["epoch"], name="mars-kep")
    far = 11000.0                                             # ~2030, three decades past J2000
    drift_km = np.linalg.norm(np.asarray(body.eph(far)[0]) - np.asarray(two_body.eph(far)[0])) / 1e3
    assert drift_km > 1e5                                     # over 100,000 km apart
    # A small body still gets the two-body Keplerian propagation (no series exists for it).
    small = lb.planet_from_row(_small_bodies().iloc[0])
    assert small.extract(pk.udpla.jpl_lp) is None
    assert small.extract(pk.udpla.keplerian) is not None


def test_a_planet_needs_no_horizons_re_osculation():
    """Re-osculating a planet would spend a network round trip refining numbers nothing reads --
    the solvers take its position from the fitted series. The row comes back unchanged, with its
    provenance saying where the ephemeris actually comes from."""
    from datetime import date

    from prospector import population as pop
    mars = planets.planet_population().set_index("pdes").loc["Mars"].to_dict()
    out = pop.refresh_target_elements(mars, date(2030, 6, 1))
    assert out["elements_source"] == "jpl_lp"
    for col in ("a", "e", "i", "om", "w", "ma"):
        assert out[col] == mars[col]


def test_a_planet_resolves_by_name_like_any_other_target(monkeypatch):
    """``get_target("Mars")`` works exactly like ``get_target("Apophis")``. Patched on the
    ``targets`` module, not on the package facade: a function resolves names in its OWN module, so
    patching the re-export would leave this reading the real cached population."""
    from prospector.population import targets
    frame = pd.concat([_small_bodies(), planets.planet_population()], ignore_index=True)
    monkeypatch.setattr(targets, "load_population", lambda **kw: frame)
    mars = targets.get_target("Mars")
    assert mars is not None and mars["a"] == pytest.approx(1.5237, abs=1e-3)
    assert planets.is_planet(mars) and mars["name"] == "Mars"
    # Case-insensitive, like every other lookup.
    assert targets.get_target("mars")["pdes"] == "Mars"


# ---------------------------------------------------------------------------
# end to end: a vehicle actually reaching Mars
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_a_solar_electric_vehicle_reaches_mars():
    """The whole point of the feature, proved rather than assumed: run a vehicle through every
    solver and require a converged low-thrust rendezvous with the planet, inside its budget.

    Convergence is the claim, and the matchpoint mismatch is what carries it, a Sims-Flanagan
    solution reports a delta-v whether or not its forward and backward arcs meet, so a small
    mismatch (not a plausible-looking delta-v) is the evidence the trajectory is flyable. Seeding
    from the Lambert grid matters here and is asserted: with no starting guess this problem lands
    somewhere that never closes, which is what the Lambert grid exists to prevent.

    The config is built inline from the active engine library rather than loaded from a named
    study, so this test does not depend on any particular library shipping a Mars mission.
    """
    import numpy as np

    from prospector.config import (
        EngineMount,
        Mission,
        ResolvedConfig,
        Screening,
        Vehicle,
    )
    from prospector.solvers import edelbaum
    from prospector.solvers import lambert as lb
    from prospector.solvers import simsflanagan as sf
    from prospector.spacecraft.propulsion import load_engines

    catalog = load_engines()
    # The most capable engine in the library, four up: Mars is a real orbit change and a low-thrust
    # stack needs the acceleration to fly it inside the window.
    engine = max(catalog, key=lambda k: catalog[k].thrust_mN / max(catalog[k].isp_s, 1.0))
    vehicle = Vehicle(name="mars probe", dry_mass=500.0, fuel_mass=400.0, unusable_prop=0.0,
                      engines=[EngineMount(type=engine, count=4)])
    # A direct-escape launch type, so the launch vehicle provides escape and the whole budget goes
    # to the heliocentric cruise (no spiral to fly).
    rc = ResolvedConfig.build(
        Mission(name="mars", launch_orbit="ESCAPE",
                launch_window=(date(2031, 1, 1), date(2031, 6, 30)),
                arrive_by=date(2033, 6, 1)),
        vehicle, Screening(), catalog)
    mars = planets.planet_population().set_index("pdes").loc["Mars"].to_dict()
    mars["name"] = "Mars"

    # The element screen keeps Mars.
    screen_dv = float(edelbaum.lowthrust_dv(mars["a"], mars["e"], mars["i"]))
    assert screen_dv <= rc.dv_budget, "the element screen must not drop Mars for this vehicle"

    # The porkchop, off the fitted positions, produces a starting guess.
    body = lb.planet_from_row(mars)
    dep, arr = lb.launch_arrival_grids(rc.departure_window, rc.mission.arrive_by,
                                       dep_step_days=10.0, arr_step_days=15.0)
    pc = lb.porkchop(body, dep, arr, dep_body=lb.earth_planet(), max_revs=2)
    assert pc.n_feasible > 0 and pc.best is not None

    # The real low-thrust solve, started from that cell.
    sol = sf.solve_for_config(rc, body, seed=pc.best, nseg=20, max_tof_days=900.0,
                              max_duty_cycle=0.9, rng_seed=42, maxeval=600, restarts=4)
    assert sol.feasible, f"the transfer did not close (mismatch {sol.mismatch:.2e})"
    assert sol.mismatch < 1e-3
    assert sol.dv_kms < rc.cruise_dv_limit, "the transfer must fit the vehicle's budget"
    assert sol.propellant_kg < rc.usable_propellant_kg
    # A rendezvous, not a flyby: the deadline is respected.
    arrival_mjd = sol.dep_mjd2000 + sol.tof_days
    assert arrival_mjd <= lb.mjd2000_from_date(rc.mission.arrive_by) + 1.0
    assert np.isfinite(sol.dv_kms) and sol.tof_days > 0
