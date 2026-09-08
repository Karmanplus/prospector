"""Tests for the local override file.

Overrides live in the config library, so the path must be resolved per call rather than captured at
import; otherwise pointing the tool at a different library would keep serving the previous one's
corrections while every path still looked right.
"""
import pytest

from prospector import paths
from prospector.enrichment.measurements import preferred, preferred_float
from prospector.enrichment.sources import overrides


@pytest.fixture(autouse=True)
def _fresh_cache():
    """The module caches the parsed file; clear it so tests do not leak into each other."""
    overrides._cache = None
    yield
    overrides._cache = None


@pytest.fixture
def library(tmp_path, monkeypatch):
    """A config library containing only an override file."""
    def _write(text):
        (tmp_path / "enrichment-overrides.yaml").write_text(text)
        monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(tmp_path))
        overrides._cache = None
        return tmp_path
    return _write


def test_no_file_means_no_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(tmp_path))
    assert overrides.load() == {}
    assert overrides.for_target("2008 EV5") == {}
    assert overrides.tier_override("2008 EV5") is None


def test_overrides_are_flagged_preferred_so_they_win(library):
    library('"2008 EV5":\n  albedo: 0.12\n  taxonomy: C\n')
    entry = overrides.for_target("2008 EV5")
    assert preferred_float(entry["albedo"]) == 0.12
    assert all(m.preferred for values in entry.values() for m in values)


def test_only_the_listed_properties_are_read(library):
    library('"X1":\n  albedo: 0.2\n  note: "checked by hand"\n  diameter_m: 500\n')
    # An unrecognised key is ignored rather than rejected, so a note costs nothing, but it also
    # cannot smuggle in a property the model does not know how to use.
    assert set(overrides.for_target("X1")) == {"albedo"}


def test_a_tier_override_is_kept_out_of_the_measurement_lists(library):
    library('"X1":\n  tier: a\n  period_h: 4.0\n')
    assert "tier" not in overrides.for_target("X1")
    assert overrides.tier_override("X1") == "A"


def test_a_non_string_tier_is_ignored(library):
    library('"X1":\n  tier: 3\n')
    assert overrides.tier_override("X1") is None


def test_a_malformed_file_degrades_to_no_overrides(library):
    library("this: is: not: valid: yaml:\n")
    assert overrides.load() == {}


def test_a_non_mapping_entry_is_skipped(library):
    library('"X1": 0.2\n"X2":\n  albedo: 0.3\n')
    assert set(overrides.load()) == {"X2"}


def test_the_active_library_decides_which_overrides_apply(tmp_path, monkeypatch):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    (first / "enrichment-overrides.yaml").write_text('"X1":\n  albedo: 0.11\n')
    (second / "enrichment-overrides.yaml").write_text('"X1":\n  albedo: 0.99\n')

    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(first))
    assert preferred(overrides.for_target("X1")["albedo"]) == 0.11
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(second))
    assert preferred(overrides.for_target("X1")["albedo"]) == 0.99
