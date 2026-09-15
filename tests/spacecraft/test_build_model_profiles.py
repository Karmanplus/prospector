"""The build model is a keyed library of profiles, and a pre-profile library still works.

``configs/build-models/<name>.yaml`` is a profile. A library that only has the older single
``configs/build-model.yaml`` reads and writes that file as the ``default`` profile, so nothing has
to move when the app is upgraded. A profile that does not exist fails naming the ones that do.
"""
import pytest

from prospector import paths
from prospector.spacecraft import buildability as B


@pytest.fixture
def lib(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(tmp_path))
    return tmp_path


def _write(path, housekeeping_W):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"housekeeping_W: {housekeeping_W}\n"
                    "array_mass_curve:\n"
                    "- {lo_W: 50.0, hi_W: 100000.0, w0: 50.0, kg0: 1.0, w1: 100000.0, kg1: 1000.0}\n")


def test_a_lone_pre_profile_file_is_the_default_profile(lib):
    _write(lib / "build-model.yaml", 111.0)
    assert B.bus_model_path() == lib / "build-model.yaml"
    assert B.list_bus_models() == ["default"]
    assert B.load_bus_model().housekeeping_W == 111.0
    assert B.load_bus_model(name="default").housekeeping_W == 111.0
    # Saving the default writes the same file back rather than starting a second copy.
    m = B.load_bus_model().model_copy(update={"housekeeping_W": 112.0})
    assert B.save_bus_model(m) == lib / "build-model.yaml"
    assert not (lib / "build-models").exists()


def test_profiles_are_files_in_the_library_directory(lib):
    _write(lib / "build-models" / "default.yaml", 100.0)
    _write(lib / "build-models" / "dawn.yaml", 300.0)
    assert B.list_bus_models() == ["dawn", "default"]
    assert B.load_bus_model(name="dawn").housekeeping_W == 300.0
    assert B.load_bus_model().housekeeping_W == 100.0
    assert B.save_bus_model(B.load_bus_model(name="dawn"), name="psyche") == lib / "build-models" / "psyche.yaml"
    assert "psyche" in B.list_bus_models()


def test_a_profile_directory_wins_over_a_leftover_single_file(lib):
    _write(lib / "build-model.yaml", 111.0)
    _write(lib / "build-models" / "default.yaml", 222.0)
    assert B.load_bus_model().housekeeping_W == 222.0
    assert B.list_bus_models() == ["default"]


def test_an_unknown_profile_names_the_ones_that_exist(lib):
    _write(lib / "build-models" / "default.yaml", 100.0)
    with pytest.raises(FileNotFoundError, match="Profiles in this library: default"):
        B.load_bus_model(name="hayabusa2")


def test_a_profile_name_is_a_file_name_not_a_path(lib):
    with pytest.raises(ValueError, match="plain file name"):
        B.bus_model_path("../outside")


def test_path_and_name_are_exclusive(lib, tmp_path):
    with pytest.raises(TypeError):
        B.load_bus_model(tmp_path / "x.yaml", name="default")
