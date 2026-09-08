"""Tests for the Compare-vehicles data layer (ui.vehicle_search) and its sweep form.

These cover the two things the sweep history has to get right: a row must say what makes ITS sweep
different from the others (or the history is a list of indistinguishable timestamps), and the
settings every sweep shares must not be repeated on every row. Plus the new-sweep form's
persistence, since a dismissed modal used to discard everything typed into it.
"""
import pandas as pd
import pytest

from ui import vehicle_search as vs

BASE = {
    "mission": "HF2", "mode": "search", "propellants": ["xenon"],
    "dry_range_kg": [150, 350, 10], "prop_range_kg": [200, 400, 25],
    "max_wet_kg": 750, "duty": 0.9,
}


def _session(name: str, **overrides) -> dict:
    return {"name": name, "meta": {**BASE, **overrides}}


def test_only_the_differing_settings_land_in_a_row():
    """The whole point: a row carries what distinguishes its sweep, and the settings shared by
    every sweep are stated once instead of on each row."""
    out = vs.session_setup([
        _session("a"),
        _session("b", prop_range_kg=[300, 600, 25]),
        _session("c", propellants=["xenon", "krypton"]),
    ])
    assert out["rows"]["a"] == "gas xenon  ·  prop 200-400/25"
    assert out["rows"]["b"] == "gas xenon  ·  prop 300-600/25"
    assert out["rows"]["c"] == "gas xenon+krypton  ·  prop 200-400/25"
    # The agreed-on settings appear once, and never inside a row.
    assert "mission HF2" in out["common"] and "thrust 90%" in out["common"]
    assert "dry 150-350/10" in out["common"]
    assert all("mission" not in row for row in out["rows"].values())


def test_a_lone_sweep_has_nothing_to_distinguish_it():
    # With one session nothing can vary, so the whole setup is 'common', the honest reading.
    out = vs.session_setup([_session("a")])
    assert out["rows"]["a"] == ""
    for expected in ("mission HF2", "gas xenon", "dry 150-350/10", "prop 200-400/25",
                     "max wet 750 kg", "thrust 90%"):
        assert expected in out["common"]


def test_a_swept_model_axis_always_shows_in_its_row():
    """A swept axis is the question the sweep was run to answer, so it is named in the row even
    when every listed session swept the same one; two sweeps of one axis over different values
    are not the same sweep."""
    out = vs.session_setup([
        _session("a", sweep_axes={"power_margin_pct": [0, 10, 20]}),
        _session("b", sweep_axes={"power_margin_pct": [0, 10, 20]}),
    ])
    assert out["rows"]["a"] == "swept power_margin_pct=0,10,20"
    assert out["rows"]["b"] == out["rows"]["a"]
    assert "swept" not in out["common"]


def test_a_missing_setting_counts_as_a_difference():
    # A session that records no thrust limit differs from one that does, so 'thrust' cannot be
    # filed as shared; that would claim a value for a sweep that never stated one.
    out = vs.session_setup([_session("a"), _session("b", duty=None)])
    assert "thrust 90%" in out["rows"]["a"]
    assert "thrust" not in out["rows"]["b"]
    assert "thrust" not in out["common"]


def test_a_malformed_setting_describes_nothing():
    # A field that won't format is left out rather than rendering a traceback into the table.
    out = vs.session_setup([_session("a"), _session("b", dry_range_kg="nonsense")])
    assert "dry" not in out["rows"]["b"]
    assert "dry 150-350/10" in out["rows"]["a"]


def test_a_single_point_range_drops_its_step():
    # A step over a range of nothing says nothing, so a fixed value reads as the value alone.
    out = vs.session_setup([_session("a", dry_range_kg=[200, 200, 10]),
                            _session("b", dry_range_kg=[150, 350, 10])])
    assert out["rows"]["a"] == "dry 200"


def test_sessions_with_no_metadata_are_survivable():
    out = vs.session_setup([{"name": "a", "meta": {}}, {"name": "b"}])
    assert out["rows"] == {"a": "", "b": ""}
    assert out["common"] == ""
    assert vs.session_setup([]) == {"rows": {}, "common": ""}


# ---------------------------------------------------------------------------
# the new-sweep form's persistence
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_form(monkeypatch):
    """A pristine workspace form, restored afterwards (the state is module-level)."""
    from ui.workspaces import vehicle as w
    monkeypatch.setattr(w, "_form", {})
    return w


def test_form_seeds_from_the_factory_defaults_with_no_sweep_history(clean_form, monkeypatch):
    w = clean_form
    monkeypatch.setattr(w, "_sessions", lambda: [])
    w._seed_form()
    assert w._form["dry_min"] == w._FORM_DEFAULTS["dry_min"]
    assert w._form["gases"] == ["xenon"]
    # A copied list, not the shared default, editing the form must not rewrite the defaults.
    w._form["gases"].append("krypton")
    assert w._FORM_DEFAULTS["gases"] == ["xenon"]


def test_form_seeds_from_the_last_sweep_that_ran(clean_form, monkeypatch):
    """Reopening the form should show what was just run -- that is what a user iterates from --
    rather than resetting to the factory numbers."""
    w = clean_form
    monkeypatch.setattr(w, "_sessions", lambda: [
        {"name": "newest", "meta": {**BASE, "dry_range_kg": [400, 900, 50],
                                    "propellants": ["krypton"], "duty": 0.55,
                                    "max_wet_kg": 1200,
                                    "sweep_axes": {"power_margin_pct": [0, 10]}}},
        {"name": "older", "meta": BASE},
    ])
    w._seed_form()
    assert (w._form["dry_min"], w._form["dry_max"], w._form["dry_step"]) == (400.0, 900.0, 50.0)
    assert w._form["gases"] == ["krypton"]
    assert w._form["thrust_pct"] == pytest.approx(55.0)
    assert w._form["max_wet"] == pytest.approx(1200.0)
    assert w._form["sweep_key"] == "power_margin_pct"
    assert w._form["sweep_vals"] == "0, 10"


def test_form_seeding_is_once_only(clean_form, monkeypatch):
    # Seeding must never overwrite what the user has typed; it fills an empty form only.
    w = clean_form
    monkeypatch.setattr(w, "_sessions", lambda: [])
    w._seed_form()
    w._set_form("dry_min", 999.0)
    w._seed_form()
    assert w._form["dry_min"] == 999.0


def test_a_multi_axis_sweep_seeds_no_axis(clean_form, monkeypatch):
    """The form holds one axis, so a session that swept two cannot be represented. It seeds
    nothing rather than half of itself, which would silently run a different sweep."""
    w = clean_form
    monkeypatch.setattr(w, "_sessions", lambda: [
        {"name": "s", "meta": {**BASE, "sweep_axes": {"power_margin_pct": [0, 10],
                                                      "array_mass_scale": [0.8, 1.0]}}}])
    w._seed_form()
    assert w._form["sweep_key"] == "" and w._form["sweep_vals"] == ""


# ---------------------------------------------------------------------------
# which project a sweep belongs to
# ---------------------------------------------------------------------------

def test_sweeps_are_filed_under_the_study_that_launched_them():
    sessions = [
        {"name": "mine", "meta": {"study": "apophis-2", "mission": "hf-apophis", "pdes": "99942"}},
        {"name": "theirs", "meta": {"study": "exlabs", "mission": "hf-apophis", "pdes": "99942"}},
    ]
    kept = vs.sessions_for_project(sessions, "apophis-2", "hf-apophis", "99942")
    assert [s["name"] for s in kept] == ["mine"]


def test_untagged_sweeps_are_claimed_by_mission_and_target_together():
    """Sweeps from before the study was recorded carry only the mission key and target. The
    mission alone is shared by many studies, so both must match, and a project with no default
    target claims none of them."""
    legacy = [{"name": "old-a", "meta": {"mission": "hf-apophis", "pdes": "99942"}},
              {"name": "old-b", "meta": {"mission": "hf-apophis", "pdes": "341843"}},
              {"name": "old-c", "meta": {"mission": "arm", "pdes": "99942"}},
              {"name": "bare", "meta": {}}]
    kept = vs.sessions_for_project(legacy, "apophis-2", "hf-apophis", "99942")
    assert [s["name"] for s in kept] == ["old-a"]
    assert vs.sessions_for_project(legacy, "apophis-2", "hf-apophis", None) == []
    assert vs.sessions_for_project(legacy, "apophis-2", None, "99942") == []


def test_with_no_project_every_sweep_is_listed():
    sessions = [{"name": "x", "meta": {"study": "a"}}, {"name": "y", "meta": {}}]
    assert vs.sessions_for_project(sessions, None, None, None) == sessions


def test_rows_rank_by_fuel_margin_not_dv_margin():
    """Two designs that both close and build at the same wet mass: the one with more propellant
    left ranks first, even when its rated-Isp dV margin is smaller. The dV margin is on the
    budget's basis and can disagree with the tank; the tank decides."""
    df = pd.DataFrame({
        "slug": ["a", "b"], "success": [True, True], "buildable": [True, True],
        "wet_kg": [650.0, 650.0], "margin_kms": [0.9, 0.3], "prop_margin_kg": [2.0, 9.0],
    })
    assert vs.order(df)["slug"].tolist() == ["b", "a"]
