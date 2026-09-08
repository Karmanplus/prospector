"""A config name is a file name inside the library, never a path.

The app slugs what it saves, so the only way a path gets in is through a hand-edited library, for
instance a shared study whose ``vehicle:`` reads ``../../something``. Every loader and saver goes
through :func:`paths.config_name`, so that fails as a bad name rather than reading or writing
outside the library.
"""
import pytest

from prospector import paths
from prospector.config import default_mission, load_vehicle, save_mission
from prospector.spacecraft.propulsion import load_engines, save_engine


@pytest.mark.parametrize("bad", ["", ".", "..", "../evil", "a/b", "a\\b", "/etc/passwd"])
def test_paths_are_rejected(bad):
    with pytest.raises(ValueError, match="plain file name"):
        paths.config_name(bad)


def test_plain_names_pass_through():
    assert paths.config_name("example-bus") == "example-bus"
    assert paths.config_name("2008 EV5 rendezvous") == "2008 EV5 rendezvous"


def test_loader_refuses_a_path_before_touching_the_disk(tmp_path):
    (tmp_path / "vehicles").mkdir()
    (tmp_path / "outside.yaml").write_text("name: outside\n")
    with pytest.raises(ValueError, match="plain file name"):
        load_vehicle("../outside", config_dir=tmp_path)


def test_savers_refuse_a_path(tmp_path):
    with pytest.raises(ValueError, match="plain file name"):
        save_mission(default_mission(), "../escaped", config_dir=tmp_path)
    engine = next(iter(load_engines(paths.EXAMPLE_CONFIG_DIR / "engines").values()))
    with pytest.raises(ValueError, match="plain file name"):
        save_engine(engine, "../escaped", engine_dir=tmp_path / "engines")
    assert not (tmp_path / "escaped.yaml").exists()
    assert not list(tmp_path.glob("**/escaped.yaml"))
