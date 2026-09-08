"""The config library's location (prospector.paths).

The override exists so an installation can keep its own engines, missions and sizing numbers
outside this tree, and so the tests can show the tool works without the repository's own library.
These pin the two things that make it trustworthy: every loader follows it, and it is looked up on
each call rather than captured at import.
"""
from __future__ import annotations

import pytest

from prospector import paths
from prospector.launch import default_launch_dir, default_return_dir
from prospector.spacecraft.buildability import default_bus_model_path
from prospector.spacecraft.propellants import default_propellant_dir
from prospector.spacecraft.propulsion import default_engine_dir
from prospector.spacecraft.radiation import default_radiation_dir

# Every library root that must move when the config library moves. A loader missing from this list
# would keep reading the bundled library while every other one followed the override, a split state
# where the vehicle comes from one library and its engines from another.
RESOLVERS = {
    "engines": default_engine_dir,
    "build-model": default_bus_model_path,
    "propellants": default_propellant_dir,
    "radiation": default_radiation_dir,
    "launches": default_launch_dir,
    "returns": default_return_dir,
    "config root": paths.config_dir,
}


def test_default_is_the_bundled_library(monkeypatch):
    # The suite runs against the example library (see conftest), so clear the override to describe
    # what an installation with nothing set gets.
    monkeypatch.delenv(paths.CONFIG_DIR_ENV, raising=False)
    assert paths.config_dir() == paths.BUNDLED_CONFIG_DIR
    for name, resolve in RESOLVERS.items():
        assert str(resolve()).startswith(str(paths.BUNDLED_CONFIG_DIR)), name


def test_every_library_follows_the_override(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(tmp_path))
    assert paths.config_dir() == tmp_path
    for name, resolve in RESOLVERS.items():
        assert str(resolve()).startswith(str(tmp_path)), f"{name} did not follow the override"


def test_the_override_is_read_per_call_not_captured_at_import(tmp_path, monkeypatch):
    """The property the whole mechanism rests on.

    A module-level constant would be read once when the first importer loaded, so anything set
    afterwards would be ignored while every path still looked correct, the failure mode that makes
    a setting appear to stop mattering. Flipping the variable mid-process must change the answer
    immediately.
    """
    monkeypatch.delenv(paths.CONFIG_DIR_ENV, raising=False)
    before = paths.config_dir()
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(tmp_path))
    assert paths.config_dir() == tmp_path != before
    monkeypatch.delenv(paths.CONFIG_DIR_ENV)
    assert paths.config_dir() == before


def test_a_user_path_is_expanded(monkeypatch):
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, "~/somewhere/configs")
    assert "~" not in str(paths.config_dir())
    assert paths.config_dir().is_absolute()


def test_an_engine_library_that_is_not_there_raises(tmp_path, monkeypatch):
    """Pointing at a library that has no engines must fail loudly. Returning an empty catalog would
    let callers fall through to built-in defaults and report a vehicle nobody configured."""
    from prospector.spacecraft.propulsion import load_engines
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(tmp_path))
    with pytest.raises(FileNotFoundError):
        load_engines()


def test_a_relocated_library_is_the_only_one_read(tmp_path, monkeypatch):
    """No bleed-through: the bundled engines must not appear alongside the relocated ones."""
    from prospector.spacecraft.propulsion import Engine, load_engines, save_engine
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(tmp_path))
    save_engine(Engine(name="Only One", isp_s=1500, thrust_mN=40, power_W=800), "only-one")
    assert sorted(load_engines()) == ["only-one"]
