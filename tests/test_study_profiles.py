"""A study names its build-model profile and its planned departure speed, and both reach the
resolved config: the profile through every sizing figure, the speed as the escape's v-infinity.
"""
from datetime import date

import pytest

from prospector import paths
from prospector.config import (
    Mission,
    Study,
    load_study,
    resolve_study,
    save_mission,
    save_study,
)
from prospector.spacecraft import buildability


@pytest.fixture
def two_profiles():
    """The library's default profile plus a 'hungry' copy with a much larger housekeeping load."""
    default = buildability.load_bus_model()
    hungry = default.model_copy(update={"housekeeping_W": default.housekeeping_W + 500.0})
    buildability.save_bus_model(hungry, name="hungry")
    return default, hungry


def test_study_round_trips_its_profile(two_profiles):
    base = load_study("example-study")
    save_study(base.model_copy(update={"build_model": "hungry"}), "profiled")
    assert load_study("profiled").build_model == "hungry"
    assert load_study("example-study").build_model == "default"


def test_the_profile_reaches_the_resolved_config(two_profiles):
    default, hungry = two_profiles
    base = load_study("example-study")
    save_study(base.model_copy(update={"build_model": "hungry"}), "profiled")
    plain = resolve_study(load_study("example-study"))
    profiled = resolve_study(load_study("profiled"))
    assert plain.build_model == "default" and profiled.build_model == "hungry"
    assert profiled.bus_model().housekeeping_W == hungry.housekeeping_W
    # The housekeeping load is grossed up into the array reserve and the sized array.
    assert profiled.array_reserve_W() > plain.array_reserve_W()
    assert profiled.vehicle.solar_power_W > plain.vehicle.solar_power_W


def test_an_unknown_profile_fails_on_resolve_naming_the_known_ones(two_profiles):
    base = load_study("example-study")
    save_study(base.model_copy(update={"build_model": "nope"}), "broken")
    with pytest.raises(FileNotFoundError, match="hungry"):
        resolve_study(load_study("broken"))


def test_a_mission_departure_speed_seeds_the_resolved_config():
    base = load_study("example-study")
    m = Mission(name="Hot launch", launch_orbit="ESCAPE", departure_vinf_kms=4.6,
                launch_window=(date(2029, 1, 1), date(2029, 1, 31)), arrive_by=date(2031, 1, 1))
    save_mission(m, "hot")
    save_study(base.model_copy(update={"mission": "hot"}), "hot-study")
    rc = resolve_study(load_study("hot-study"))
    assert rc.departure_vinf_kms == pytest.approx(4.6)
    # A mission that leaves it unset gets the slider default, zero.
    assert resolve_study(base).departure_vinf_kms == 0.0
    assert Mission().departure_vinf_kms is None


def test_the_example_flown_missions_carry_their_launch_speeds():
    for key, vinf in (("dawn", 3.3), ("psyche", 5.8), ("hayabusa2", 4.6)):
        rc = resolve_study(load_study(key, config_dir=paths.EXAMPLE_CONFIG_DIR),
                           config_dir=paths.EXAMPLE_CONFIG_DIR)
        assert rc.departure_vinf_kms == pytest.approx(vinf), key
        assert rc.build_model == "default"


def test_study_model_defaults():
    s = Study(mission="m", vehicle="v")
    assert s.build_model == "default"
