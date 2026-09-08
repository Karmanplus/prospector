"""Writing a config file back without losing what it says about itself (prospector.yamlio).

Both halves guard against the same thing: a config library slowly stopping explaining itself. A
plain dump rewrites the file and every comment in it disappears, and an ignored key sits in the
file looking authoritative while nothing reads it. Neither raises, and neither shows up until
somebody wonders why a number they tuned did nothing.
"""
from __future__ import annotations

import warnings

import pytest
import yaml
from pydantic import BaseModel

from prospector.yamlio import dump_preserving_comments, validated, warn_unknown_keys

ANNOTATED = """\
# The conservative default: sized against the worst case on purpose.
# Anchored to an OMERE 5.9 run; see docs/physics.md.
name: AP8-MIN worst case
coverglass_um: 212.5        # the dominant design lever

# Core rates, at the reference coverglass.
proton_ddd_core: 4600000000.0
"""


class _Model(BaseModel):
    name: str = "x"
    coverglass_um: float = 1.0
    proton_ddd_core: float = 1.0


def test_editing_a_value_keeps_every_comment(tmp_path):
    path = tmp_path / "scenario.yaml"
    path.write_text(ANNOTATED)

    dump_preserving_comments({"name": "AP8-MIN worst case", "coverglass_um": 300.0,
                              "proton_ddd_core": 4600000000.0}, path)

    text = path.read_text()
    assert "coverglass_um: 300.0" in text
    assert "The conservative default" in text
    assert "OMERE 5.9" in text
    assert "the dominant design lever" in text, "the trailing comment on the edited line"
    assert "Core rates" in text


def test_key_order_survives(tmp_path):
    """A reordered file is a diff on every line, which buries the one value that changed."""
    path = tmp_path / "scenario.yaml"
    path.write_text(ANNOTATED)
    dump_preserving_comments({"proton_ddd_core": 1.0, "name": "n", "coverglass_um": 2.0}, path)
    keys = [line.split(":")[0] for line in path.read_text().splitlines()
            if line and not line.startswith("#") and ":" in line]
    assert keys == ["name", "coverglass_um", "proton_ddd_core"]


def test_a_new_file_is_written_plainly(tmp_path):
    path = tmp_path / "new.yaml"
    dump_preserving_comments({"name": "fresh", "coverglass_um": 5.0}, path)
    assert yaml.safe_load(path.read_text()) == {"name": "fresh", "coverglass_um": 5.0}


def test_keys_the_model_dropped_leave_the_file(tmp_path):
    path = tmp_path / "scenario.yaml"
    path.write_text(ANNOTATED)
    dump_preserving_comments({"name": "n", "coverglass_um": 1.0}, path)
    assert "proton_ddd_core" not in path.read_text()


def test_keys_the_model_gained_are_added(tmp_path):
    path = tmp_path / "scenario.yaml"
    path.write_text(ANNOTATED)
    dump_preserving_comments({"name": "n", "coverglass_um": 1.0,
                              "proton_ddd_core": 1.0, "brand_new": 7.0}, path)
    assert yaml.safe_load(path.read_text())["brand_new"] == 7.0


def test_an_unparseable_file_is_replaced_rather_than_failing(tmp_path):
    # Saving must not be the thing that breaks on a file somebody hand-edited into invalid YAML.
    path = tmp_path / "broken.yaml"
    path.write_text("name: [unclosed\n")
    dump_preserving_comments({"name": "recovered"}, path)
    assert yaml.safe_load(path.read_text()) == {"name": "recovered"}


def test_a_round_trip_of_no_changes_leaves_the_file_alone(tmp_path):
    path = tmp_path / "scenario.yaml"
    path.write_text(ANNOTATED)
    data = yaml.safe_load(ANNOTATED)
    dump_preserving_comments(data, path)
    assert path.read_text() == ANNOTATED, "saving without editing anything rewrote the file"


# ---- the ignored-key warning ----

def test_a_key_the_model_does_not_read_warns(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text("name: n\ncell_specific_power_W_kg: 153.0\n")
    with pytest.warns(UserWarning, match="cell_specific_power_W_kg"):
        model = validated(_Model, path)
    assert model.name == "n", "the file still loads; the key is ignored, not fatal"


def test_a_file_the_model_fully_reads_is_silent(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text("name: n\ncoverglass_um: 2.0\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert validated(_Model, path).coverglass_um == 2.0


def test_the_warning_names_every_unread_key():
    with pytest.warns(UserWarning) as caught:
        warn_unknown_keys({"a": 1, "b": 2, "name": 3}, _Model.model_fields, "src.yaml")
    message = str(caught[0].message)
    assert "a" in message and "b" in message and "src.yaml" in message
    assert "2 key(s)" in message
