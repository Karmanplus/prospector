"""The flown-mission examples resolve, and their headline numbers are the published ones.

Dawn, Psyche and Hayabusa2 ship in the example library so a user can check the tool against
history. What makes them worth trusting is that the masses, engines and dates are the public
figures, so those are pinned here: a well-meant "tidy-up" of a YAML must not quietly turn Dawn into
a different spacecraft. Sources are cited in each file.
"""
from datetime import date

import pytest

from prospector import paths
from prospector.config import load_study, resolve_study
from prospector.spacecraft.propulsion import load_engines

LIB = paths.EXAMPLE_CONFIG_DIR


@pytest.mark.parametrize("study, wet_kg, xenon_kg, engine, count, array_W, arrive", [
    ("dawn", 1217.7, 425.0, "nstar", 1, 10300.0, date(2011, 7, 16)),
    ("psyche", 2747.0, 1085.0, "spt-140", 1, 21000.0, date(2029, 8, 31)),
    ("hayabusa2", 609.0, 66.0, "mu10", 3, 2600.0, date(2018, 6, 27)),
    ("dart", 615.0, 60.0, "nstar", 1, 6600.0, date(2022, 9, 26)),
])
def test_flown_mission_resolves_to_its_published_numbers(study, wet_kg, xenon_kg, engine, count,
                                                         array_W, arrive):
    rc = resolve_study(load_study(study, config_dir=LIB))
    assert rc.vehicle.wet_mass == pytest.approx(wet_kg, abs=0.05)
    assert rc.vehicle.fuel_mass == pytest.approx(xenon_kg)
    assert rc.vehicle.mounts == [(engine, count)]
    assert rc.vehicle.solar_power_W == pytest.approx(array_W)
    assert rc.mission.arrive_by == arrive
    # All four launched straight to escape, so the tool charges nothing for escape and the cruise
    # gets the launch v-infinity, held exactly.
    assert rc.launch.escape_provided
    assert rc.escape_propellant_kg == 0.0


def test_the_flown_routes_are_the_missions():
    """Dawn and Psyche via Mars, Hayabusa2 via Earth from its launch, DART direct into Didymos at
    up to 6.5 km/s; none of them a rendezvous with a return."""
    routes = {k: resolve_study(load_study(k, config_dir=LIB)).mission for k in
              ("dawn", "psyche", "hayabusa2", "dart")}
    assert routes["dawn"].gravity_assist == "mars" and routes["psyche"].gravity_assist == "mars"
    assert routes["hayabusa2"].gravity_assist == "earth"
    assert routes["dart"].gravity_assist is None and routes["dart"].arrival_vinf_kms == 6.5
    assert all(m.arrival_vinf_kms == 0.0 for k, m in routes.items() if k != "dart")
    assert not any(m.return_trip for m in routes.values())


def test_nstar_throttle_table_is_the_published_one():
    """Goebel & Katz Table 9-2: sixteen levels, TH0 518 W / 20.7 mN / 1979 s to TH15 2325 W /
    92.7 mN / 3127 s, monotonic in power."""
    nstar = load_engines(LIB / "engines")["nstar"]
    pts = nstar.modes()
    assert len(pts) == 16
    assert (pts[0].power_W, pts[0].thrust_mN, pts[0].isp_s) == (518.0, 20.7, 1979.0)
    assert (pts[-1].power_W, pts[-1].thrust_mN, pts[-1].isp_s) == (2325.0, 92.7, 3127.0)
    assert all(a.power_W < b.power_W for a, b in zip(pts, pts[1:]))


def test_real_engines_are_marked_with_their_sources():
    engines = load_engines(LIB / "engines")
    for key in ("nstar", "spt-140", "mu10"):
        assert engines[key].source.strip(), f"{key} has no source recorded"
        assert "synthetic" not in engines[key].source.lower()
