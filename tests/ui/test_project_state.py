"""Tests for the project lifecycle in ``ui.state``: what a save writes where, and new projects.

Run against the per-test copy of the config library (see ``tests/conftest.py``), so the writes
here land nowhere real. The one rule these pin: a study owns the parts it edits. Six studies
sharing one mission file meant that saving any of them moved the launch and arrival dates of all
six, which read as the dates "resetting" when switching projects.
"""
from datetime import date

import pytest

from prospector.config import (
    Mission,
    Study,
    list_studies,
    load_mission,
    load_study,
    load_vehicle,
    save_mission,
    save_study,
)
from tests import library as lib
from ui import state
from ui.state import S


@pytest.fixture
def shared_mission():
    """Two studies, ``a`` and ``b``, referencing one mission file ``shared``; the vehicle is one
    the active library already has. Closes whatever project is open afterwards."""
    vkey = load_study(lib.study_key()).vehicle
    save_mission(Mission(name="Shared mission", arrive_by=date(2030, 3, 1),
                         launch_window=(date(2029, 1, 1), date(2029, 3, 31))), "shared")
    save_study(Study(name="A", mission="shared", vehicle=vkey), "a")
    save_study(Study(name="B", mission="shared", vehicle=vkey), "b")
    yield vkey
    state.close_project()


def test_editing_a_shared_mission_forks_it_on_save(shared_mission):
    state.open_project("a")
    S.mission = S.mission.model_copy(update={"arrive_by": date(2031, 6, 1)})
    notes = state.persist_study("A", "a")

    assert load_study("a").mission == "a", "the edited mission is saved under the study's own key"
    assert load_mission("a").arrive_by == date(2031, 6, 1)
    assert load_mission("shared").arrive_by == date(2030, 3, 1), "the shared file is untouched"
    assert load_study("b").mission == "shared", "the other study still points at the shared file"
    assert S.mission_key == "a" and not S.dirty
    assert notes and "shared" in notes[0]


def test_an_unchanged_shared_mission_keeps_its_key(shared_mission):
    state.open_project("a")
    S.vehicle = S.vehicle.model_copy(update={"dry_mass": S.vehicle.dry_mass + 1.0})
    notes = state.persist_study("A", "a")
    assert load_study("a").mission == "shared", "nothing changed in the mission, so no fork"
    # The vehicle was edited and is shared with the library's own study, so IT forks.
    assert load_study("a").vehicle == "a"
    assert load_vehicle("a").dry_mass == pytest.approx(S.vehicle.dry_mass)
    assert load_vehicle(shared_mission).dry_mass != pytest.approx(S.vehicle.dry_mass)
    assert any("vehicle" in n for n in notes) and not any("mission" in n for n in notes)


def test_a_mission_only_this_study_uses_is_saved_in_place(shared_mission):
    save_study(Study(name="B", mission="other", vehicle=shared_mission), "b")
    save_mission(Mission(name="Other"), "other")
    state.open_project("a")
    S.mission = S.mission.model_copy(update={"arrive_by": date(2031, 6, 1)})
    notes = state.persist_study("A", "a")
    assert load_study("a").mission == "shared"
    assert load_mission("shared").arrive_by == date(2031, 6, 1)
    assert notes == []


def test_save_as_of_a_study_leaves_the_original_alone(shared_mission):
    """Save-as under a new name with edited dates: the new study owns the edit, the original and
    every study sharing its mission read as before."""
    state.open_project("a")
    S.mission = S.mission.model_copy(update={"arrive_by": date(2032, 1, 1)})
    state.persist_study("A prime", "a-prime")
    assert load_study("a-prime").mission == "a-prime"
    assert load_mission("shared").arrive_by == date(2030, 3, 1)
    assert load_study("a").mission == "shared"
    assert S.project == "a-prime"


def test_a_fork_never_lands_on_a_key_another_study_uses(shared_mission):
    """If a stranger already references the mission key the fork would take, the fork steps
    aside rather than overwriting that study's mission."""
    save_mission(Mission(name="Someone else's"), "a")
    save_study(Study(name="C", mission="a", vehicle=shared_mission), "c")
    state.open_project("a")
    S.mission = S.mission.model_copy(update={"arrive_by": date(2031, 6, 1)})
    state.persist_study("A", "a")
    assert load_study("a").mission == "a-2"
    assert load_mission("a").name == "Someone else's"


def test_new_project_is_saved_at_once_and_owns_its_mission(shared_mission):
    state.new_project("Fresh Start")
    assert S.project == "fresh-start" and S.study_name == "Fresh Start"
    assert "fresh-start" in list_studies()
    study = load_study("fresh-start")
    assert study.mission == "fresh-start", "a new project's mission is its own from the start"
    assert study.vehicle == "fresh-start"
    assert study.target is None
    assert not S.dirty and S.workspace == "project"
    assert S.project_vehicle == S.vehicle, "the saved vehicle is the project default"


def test_new_project_refuses_a_taken_or_empty_name(shared_mission):
    before = list_studies()
    with pytest.raises(ValueError):
        state.new_project("A")            # slugs to 'a', which exists
    with pytest.raises(ValueError):
        state.new_project("   ")
    assert list_studies() == before, "nothing was written"


def test_new_project_can_copy_the_open_one(shared_mission):
    state.open_project("a")
    S.mission = S.mission.model_copy(update={"arrive_by": date(2033, 1, 1)})
    state.new_project("A copy", copy_from_open=True)
    assert S.project == "a-copy"
    copied = load_study("a-copy")
    assert copied.mission == "a-copy", "the copy owns its mission..."
    assert load_mission("a-copy").arrive_by == date(2033, 1, 1), "...with the working edits"
    assert copied.vehicle == shared_mission, "...and keeps the library vehicle reference"
    assert load_mission("shared").arrive_by == date(2030, 3, 1), "the source was never written"
    assert load_study("a").mission == "shared"
