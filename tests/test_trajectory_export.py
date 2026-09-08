"""The trajectory bundle: escape and cruise reconciled onto one launch-relative clock.

This is the only place the escape spiral around Earth and the cruise around the Sun get joined up,
and the join is what these tests pin: the clock, the frames, and getting back to SI. The bundle
goes to something outside this repo, so a quiet offset here stays invisible until someone else's
viewer draws the wrong picture.
"""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pykep as pk
import pytest

from prospector import trajectory_export as tx
from prospector.config import EngineMount, ResolvedConfig, Screening, Vehicle, load_mission
from prospector.spacecraft.propulsion import load_engines
from tests import library as lib

SECONDS_PER_DAY = 86400.0


@pytest.fixture
def rc() -> ResolvedConfig:
    return ResolvedConfig.build(
        load_mission(lib.mission_key()),
        Vehicle(name="probe", dry_mass=250.0, fuel_mass=325.0, unusable_prop=2.0,
                engines=[EngineMount(type=lib.engine_key(), count=4)]),
        Screening(), load_engines())


def _cruise(n_fine: int = 8, n_nodes: int = 4, dep_mjd2000: float = 10000.0,
            tof_days: float = 300.0) -> dict:
    """A cruise result shaped like the worker stores one, with positions in AU."""
    fine_days = np.linspace(0.0, tof_days, n_fine)
    node_days = np.linspace(0.0, tof_days, n_nodes)
    # A simple arc from 1 AU outward, so positions are recognizable after the AU round-trip.
    fine_pos_au = np.column_stack([np.linspace(1.0, 1.2, n_fine),
                                   np.zeros(n_fine), np.zeros(n_fine)])
    return {
        "sf": {
            "dep_mjd2000": dep_mjd2000, "tof_days": tof_days,
            "fine_times_days": fine_days, "fine_positions_au": fine_pos_au,
            "fine_throttle": np.linspace(0.0, 1.0, n_fine),
            "node_times_days": node_days, "throttle": np.ones(n_nodes),
            "node_thrust_vec": np.tile([2.0, 0.0, 0.0], (n_nodes, 1)),  # unnormalized on purpose
            "propellant_kg": 120.0, "thrust_N": 0.156, "isp_s": 1400.0,
        },
        "orbits": {
            "earth_au": np.column_stack([np.ones(5), np.zeros(5), np.zeros(5)]),
            "target_au": np.column_stack([np.full(5, 1.2), np.zeros(5), np.zeros(5)]),
            "earth_track_au": np.column_stack([np.ones(n_fine), np.zeros(n_fine),
                                               np.zeros(n_fine)]),
            "target_track_au": np.column_stack([np.full(n_fine, 1.2), np.zeros(n_fine),
                                                np.zeros(n_fine)]),
        },
        "target": {"name": "99942 Apophis", "a": 0.922, "e": 0.191, "i": 3.34},
    }


def _spiral(tof_days: float = 200.0, n: int = 6) -> dict:
    """An escape result shaped like the spiral worker stores one, positions in km."""
    return {
        "tof_days": tof_days,
        "times_days": np.linspace(0.0, tof_days, n),
        "positions_km": np.column_stack([np.linspace(7000.0, 90000.0, n),
                                         np.zeros(n), np.zeros(n)]),
        "mass_kg": np.linspace(575.0, 540.0, n),
    }


# ---------------------------------------------------------------------------
# the clock: one timeline across two separately-solved phases
# ---------------------------------------------------------------------------

def test_the_two_phases_share_one_launch_relative_clock(rc):
    """t = 0 is injection, the escape ends where the cruise begins, and arrival closes it.

    The spiral and the cruise are solved separately in different frames; this is the only place
    they are reconciled, so an offset here silently misplaces the whole mission.
    """
    files = tx.build_bundle(rc, _cruise(tof_days=300.0), _spiral(tof_days=200.0))
    meta = json.loads(files["trajectory.json"])["meta"]

    assert meta["spiral_end_t_s"] == pytest.approx(200.0 * SECONDS_PER_DAY)
    assert meta["arrival_t_s"] == pytest.approx(500.0 * SECONDS_PER_DAY)
    # Launch precedes the cruise departure by the escape duration: the cruise departs at the escape
    # epoch, not at liftoff.
    assert meta["launch_mjd2000"] == pytest.approx(10000.0 - 200.0)
    assert meta["dep_mjd2000"] == pytest.approx(10000.0)


def test_the_cruise_clock_starts_where_the_escape_ends(rc):
    traj = json.loads(tx.build_bundle(rc, _cruise(), _spiral(tof_days=200.0))["trajectory.json"])
    assert traj["cruise"]["t_s"][0] == pytest.approx(200.0 * SECONDS_PER_DAY)
    assert traj["spiral"]["t_s"][-1] == pytest.approx(traj["cruise"]["t_s"][0])


def test_a_launch_vehicle_escape_produces_a_cruise_only_bundle(rc):
    """With no spiral there is no escape phase, and launch coincides with departure."""
    traj = json.loads(tx.build_bundle(rc, _cruise(), None)["trajectory.json"])
    assert traj["spiral"] is None
    assert traj["meta"]["phases"] == ["cruise"]
    assert traj["meta"]["launch_mjd2000"] == pytest.approx(traj["meta"]["dep_mjd2000"])
    assert traj["cruise"]["t_s"][0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# the frames and the units: what a consumer needs to place the geometry
# ---------------------------------------------------------------------------

def test_each_phase_declares_its_own_frame(rc):
    """The phases are NOT pre-flattened into one frame, so the bundle has to say which is which
    or a consumer places the spiral in the heliocentric scene."""
    meta = json.loads(tx.build_bundle(rc, _cruise(), _spiral())["trajectory.json"])["meta"]
    assert meta["frames"] == {"spiral": "geocentric_equatorial_m",
                              "cruise": "heliocentric_ecliptic_m"}


def test_cruise_positions_round_trip_back_to_the_metres_they_came_from(rc):
    """The stored arc is in AU because it was divided by pykep's AU. Multiplying back by a
    DIFFERENT AU silently offsets every position, and the export is a reference trajectory."""
    cruise = _cruise()
    files = tx.build_bundle(rc, cruise, None)
    traj = json.loads(files["trajectory.json"])
    first_au = float(cruise["sf"]["fine_positions_au"][0][0])
    exported_m = traj["cruise"]["pos_m"][0][0]
    assert exported_m == pytest.approx(first_au * pk.AU, abs=1.0)


def test_the_declared_au_matches_the_one_used_to_scale(rc):
    """meta.au_m tells a consumer how to convert; it must be the value actually applied."""
    cruise = _cruise()
    traj = json.loads(tx.build_bundle(rc, cruise, None)["trajectory.json"])
    declared = traj["meta"]["au_m"]
    applied = traj["cruise"]["pos_m"][0][0] / float(cruise["sf"]["fine_positions_au"][0][0])
    assert applied == pytest.approx(declared, rel=1e-12)


def test_spiral_positions_are_metres_from_the_stored_kilometres(rc):
    spiral = _spiral()
    traj = json.loads(tx.build_bundle(rc, _cruise(), spiral)["trajectory.json"])
    assert traj["spiral"]["pos_m"][0][0] == pytest.approx(
        float(spiral["positions_km"][0][0]) * 1000.0)


# ---------------------------------------------------------------------------
# pointing and throttle
# ---------------------------------------------------------------------------

def test_node_directions_are_unit_vectors(rc):
    """node_thrust_vec is direction times throttle; a consumer slerping attitude needs pure
    direction, so the magnitude has to be divided out."""
    traj = json.loads(tx.build_bundle(rc, _cruise(), None)["trajectory.json"])
    for node in traj["cruise"]["nodes"]:
        assert np.linalg.norm(node["dir"]) == pytest.approx(1.0, abs=1e-4)


def test_throttle_and_thrust_on_agree(rc):
    traj = json.loads(tx.build_bundle(rc, _cruise(), None)["trajectory.json"])
    cruise = traj["cruise"]
    for throttle, on in zip(cruise["throttle"], cruise["thrust_on"]):
        assert on == (throttle > 1e-3)


# ---------------------------------------------------------------------------
# the vehicle document
# ---------------------------------------------------------------------------

def test_vehicle_yaml_carries_the_resolved_performance(rc):
    import yaml
    doc = yaml.safe_load(tx.build_bundle(rc, _cruise(), None)["vehicle.yaml"])
    assert doc["name"] == "probe"
    assert doc["dry_mass_kg"] == pytest.approx(250.0)
    assert doc["wet_mass_kg"] == pytest.approx(575.0)
    # Four engines blend into the assembly performance the trajectory was flown with.
    assert doc["engines"][0]["count"] == 4
    assert doc["propulsion"]["total_thrust_mN"] == pytest.approx(rc.total_thrust_mN)
    # The operating point the cruise actually flew, which is below rated when power-limited.
    assert doc["propulsion"]["cruise_operating_thrust_N"] == pytest.approx(0.156)


# ---------------------------------------------------------------------------
# packaging
# ---------------------------------------------------------------------------

def test_bundle_names_the_three_files_a_consumer_expects(rc):
    files = tx.build_bundle(rc, _cruise(), _spiral())
    assert set(files) == {"trajectory.json", "asteroid.json", "vehicle.yaml"}
    for text in files.values():
        assert text.strip()


def test_zip_round_trips_every_file(rc):
    files = tx.build_bundle(rc, _cruise(), _spiral())
    with zipfile.ZipFile(__import__("io").BytesIO(tx.bundle_zip_bytes(files))) as z:
        assert sorted(z.namelist()) == sorted(files)
        for name in files:
            assert z.read(name).decode() == files[name]


def test_asteroid_track_shares_the_cruise_clock(rc):
    """The moving target entity has to advance on the same timeline as the spacecraft, or the
    rendezvous does not line up in the viewer."""
    files = tx.build_bundle(rc, _cruise(), _spiral(tof_days=200.0))
    asteroid = json.loads(files["asteroid.json"])
    cruise = json.loads(files["trajectory.json"])["cruise"]
    assert asteroid["track"]["t_s"] == cruise["t_s"]
