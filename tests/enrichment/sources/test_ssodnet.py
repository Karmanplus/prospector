"""Tests for the SsODNet adapter.

The real lookup needs an optional package and the network, so what is tested here is the adapter
around it: the datacloud tables come back as pandas frames with inconsistent columns and NaN-filled
rows, and the conversion has to survive all of it without inventing values. The two behaviours that
matter to the screen are pinned directly; an unresolvable body is reported rather than raised, and
a missing package is a run-wide failure rather than a per-target one.
"""
import pandas as pd
import pytest

from prospector.enrichment.sources import ssodnet


def test_missing_package_is_a_run_wide_failure(monkeypatch):
    real_import = __import__

    def _no_rocks(name, *args, **kwargs):
        if name == "rocks":
            raise ImportError("no rocks here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _no_rocks)
    # Not a Body with resolved=False: nothing can be looked up at all, and characterizing the rest
    # of the batch would produce a screen silently missing its desirability axis.
    with pytest.raises(ssodnet.SourceUnavailable, match="rocks"):
        ssodnet.lookup("341843")


def test_measurements_carry_the_surveys_own_preference_flag():
    frame = pd.DataFrame({"albedo": [0.09, 0.12], "preferred": [False, True]})
    values = ssodnet._measurements(frame, "albedo")
    assert [m.value for m in values] == [0.09, 0.12]
    assert [m.preferred for m in values] == [False, True]
    assert all(m.source == "ssodnet" for m in values)


def test_rows_with_no_value_are_dropped():
    frame = pd.DataFrame({"period": [3.7, None, 5.0], "preferred": [True, True, False]})
    assert [m.value for m in ssodnet._measurements(frame, "period")] == [3.7, 5.0]


def test_a_table_without_a_preference_column_defaults_to_unflagged():
    # Some datacloud tables omit the column entirely; treating that as "preferred" would let an
    # arbitrary survey outrank a curated one.
    frame = pd.DataFrame({"class_": ["C", "Ch"]})
    assert not any(m.preferred for m in ssodnet._measurements(frame, "class_"))


@pytest.mark.parametrize("frame", [None, pd.DataFrame(), pd.DataFrame({"other": [1]})])
def test_an_absent_table_yields_no_measurements(frame):
    assert ssodnet._measurements(frame, "albedo") == []


def test_colours_are_keyed_by_their_filter_pair():
    frame = pd.DataFrame({
        "id_filter_1": ["Johnson.B", "SDSS.g"],
        "id_filter_2": ["Johnson.V", "SDSS.r"],
        "value": [0.85, 0.55],
        "preferred": [True, False],
    })
    colors = ssodnet._colors(frame)
    # The fully-qualified survey prefixes are stripped: the solar-colour and wavelength tables are
    # keyed on the bare filter name.
    assert set(colors) == {"B-V", "g-r"}
    assert colors["B-V"][0].value == 0.85 and colors["B-V"][0].preferred


def test_repeated_colours_accumulate_under_one_key():
    frame = pd.DataFrame({
        "id_filter_1": ["Johnson.B", "Johnson.B"],
        "id_filter_2": ["Johnson.V", "Johnson.V"],
        "value": [0.85, 0.90],
    })
    assert len(ssodnet._colors(frame)["B-V"]) == 2


def test_colour_rows_with_no_value_are_dropped():
    frame = pd.DataFrame({
        "id_filter_1": ["Johnson.B", "Johnson.V"],
        "id_filter_2": ["Johnson.V", "Johnson.R"],
        "value": [0.85, None],
    })
    assert set(ssodnet._colors(frame)) == {"B-V"}


@pytest.mark.parametrize("frame", [None, pd.DataFrame(), pd.DataFrame({"value": [0.5]})])
def test_an_absent_colour_table_yields_nothing(frame):
    assert ssodnet._colors(frame) == {}


class _Rock:
    """Stands in for the package's own return value, with the attributes the adapter reads."""

    def __init__(self, name, id_, number=None):
        self.name = name
        self.id_ = id_
        self.number = number
        self.H = None
        self.orbital_elements = None
        self.albedos = self.taxonomies = self.colors = self.spins = None


def _install(monkeypatch, rock):
    class _Rocks:
        @staticmethod
        def Rock(_identifier, **_kwargs):        # noqa: N802 - mirroring the package's API
            if isinstance(rock, Exception):
                raise rock
            return rock

    monkeypatch.setitem(__import__("sys").modules, "rocks", _Rocks)


def test_a_lookup_that_raises_is_one_targets_problem(monkeypatch):
    _install(monkeypatch, ValueError("not found"))
    body = ssodnet.lookup("NOT-A-BODY")
    # One bad designation must not take the rest of the batch with it.
    assert body.resolved is False and body.input_id == "NOT-A-BODY"


def test_a_failed_identification_is_detected_despite_the_echoed_name(monkeypatch):
    """A failed lookup does not raise and does not come back empty -- the input string is
    echoed straight into `name`, so an unresolvable designation looks exactly like a real body
    with nothing measured. Without keying on the resolved id, a typo would be characterized as
    a genuine target and cached as one."""
    _install(monkeypatch, _Rock(name="NOT-A-BODY", id_=""))
    assert ssodnet.lookup("NOT-A-BODY").resolved is False


def test_a_resolved_body_keeps_the_identifier_it_was_asked_for(monkeypatch):
    _install(monkeypatch, _Rock(name="2008 EV5", id_="2008_EV5", number=341843))
    body = ssodnet.lookup("341843")
    assert body.resolved is True
    assert body.input_id == "341843", "the join key is the input, never the resolved name"
    assert body.name == "2008 EV5" and body.number == 341843


def _fake_rocks(tmp_path, monkeypatch, *, tables_landed=True, calls=None):
    """A stand-in for the library: two known bodies, one unknown, and a cache directory the bulk
    fetch writes table files into (or not, when the tables service is 'down')."""
    import sys
    import types

    calls = calls if calls is not None else []
    cache = tmp_path / "rocks-cache"
    cache.mkdir()

    class _Value:
        def __init__(self, value):
            self.value = value

    class _Rock:
        orbital_elements = None
        albedos = taxonomies = spins = colors = None

        def __init__(self, id_, ssocard=None, skip_id_check=False, datacloud=None, **_k):
            calls.append(("Rock", id_, tuple(datacloud or ())))
            self.id_, self.name, self.number = id_, ssocard["name"], ssocard.get("number")
            self.H = _Value(ssocard["H"])

    def identify(ids, return_id=False, **_k):
        calls.append(("identify", list(ids)))
        known = {"99942": ("Apophis", 99942, "Apophis"), "433": ("Eros", 433, "Eros")}
        return [known.get(i, (None, float("nan"), None)) for i in ids]

    def get_ssocard(ids, **_k):
        calls.append(("cards", list(ids)))
        return [{"name": i, "number": 1, "H": 19.1} for i in ids]

    def get_datacloud_catalogue(ids, table, **_k):
        calls.append(("table", table, list(ids)))
        if tables_landed:
            for i in ids:
                (cache / f"{i}_{table}.json").write_text("{}")
        return {}

    tables = {n: {"ssodnet_name": t, "attr_name": n} for n, t in
              (("albedos", "diamalbedo"), ("taxonomies", "taxonomy"), ("colors", "colors"),
               ("spins", "spin"))}
    fake = types.SimpleNamespace(
        Rock=_Rock, rocks=None,
        resolve=types.SimpleNamespace(identify=identify),
        ssodnet=types.SimpleNamespace(get_ssocard=get_ssocard,
                                      get_datacloud_catalogue=get_datacloud_catalogue),
        config=types.SimpleNamespace(DATACLOUD=tables, PATH_CACHE=cache))
    monkeypatch.setitem(sys.modules, "rocks", fake)
    return calls


def test_lookup_many_fetches_the_list_in_bulk_and_builds_each_body_locally(tmp_path, monkeypatch):
    """One resolution, one card fetch and one fetch per table for the whole list; each body is
    then built from its card with the tables that landed, never fetching per body."""
    calls = _fake_rocks(tmp_path, monkeypatch)
    bodies = ssodnet.lookup_many(["99942", "nonsense", "433", "99942"])
    assert calls[0] == ("identify", ["99942", "nonsense", "433"])
    assert calls[1] == ("cards", ["Apophis", "Eros"])
    assert [c for c in calls if c[0] == "table"] == [
        ("table", "diamalbedo", ["Apophis", "Eros"]), ("table", "taxonomy", ["Apophis", "Eros"]),
        ("table", "colors", ["Apophis", "Eros"]), ("table", "spin", ["Apophis", "Eros"])]
    assert [c for c in calls if c[0] == "Rock"] == [
        ("Rock", "Apophis", ("albedos", "taxonomies", "colors", "spins")),
        ("Rock", "Eros", ("albedos", "taxonomies", "colors", "spins"))]
    assert bodies["99942"].resolved and bodies["99942"].abs_mag[0].value == 19.1
    assert bodies["99942"].input_id == "99942"        # the join key is what was asked for
    assert bodies["433"].resolved and not bodies["nonsense"].resolved


def test_a_dropped_connection_is_retried_once_before_giving_up(tmp_path, monkeypatch):
    calls = _fake_rocks(tmp_path, monkeypatch)
    import sys
    fake = sys.modules["rocks"]
    real = fake.ssodnet.get_ssocard
    state = {"n": 0}

    def _flaky(ids, **k):
        state["n"] += 1
        if state["n"] == 1:
            raise ConnectionResetError("reset by peer")
        return real(ids, **k)

    fake.ssodnet.get_ssocard = _flaky
    monkeypatch.setattr(ssodnet.time, "sleep", lambda s: None)
    bodies = ssodnet.lookup_many(["99942"])
    assert state["n"] == 2 and bodies["99942"].resolved
    assert sum(1 for c in calls if c[0] == "cards") == 1


def test_a_tables_outage_is_reported_not_cached_as_nothing_measured(tmp_path, monkeypatch):
    _fake_rocks(tmp_path, monkeypatch, tables_landed=False)
    with pytest.raises(ssodnet.SourceUnavailable, match="tables"):
        ssodnet.lookup_many(["99942", "433"])


def test_a_failed_bulk_call_returns_nothing_rather_than_guessing(tmp_path, monkeypatch):
    calls = _fake_rocks(tmp_path, monkeypatch)
    import sys

    def _down(ids, **_k):
        raise RuntimeError("quaero down")
    sys.modules["rocks"].resolve.identify = _down
    assert ssodnet.lookup_many(["99942"]) == {} and calls == []
