"""Opening a project adopts its build-model profile and its mission's departure speed; saving the
project writes the profile back; a remembered slider position still wins over the mission's value.
"""
from datetime import date

import pytest

from prospector.config import Mission, Study, load_study, save_mission, save_study
from prospector.spacecraft import buildability
from ui import session, state
from ui.state import S


@pytest.fixture
def profiled_project():
    vkey = load_study("example-study").vehicle
    default = buildability.load_bus_model()
    buildability.save_bus_model(default.model_copy(update={"housekeeping_W": 900.0}), name="hungry")
    save_mission(Mission(name="Hot", launch_orbit="ESCAPE", departure_vinf_kms=4.6,
                         launch_window=(date(2029, 1, 1), date(2029, 1, 31)),
                         arrive_by=date(2031, 1, 1)), "hot")
    save_study(Study(name="Profiled", mission="hot", vehicle=vkey, build_model="hungry"), "profiled")
    yield "profiled"
    state.close_project()


def test_open_project_adopts_profile_and_departure_speed(profiled_project):
    state.open_project(profiled_project)
    assert S.build_model == "hungry"
    assert S.departure_vinf == pytest.approx(4.6)
    rc = state.resolved()
    assert rc.build_model == "hungry"
    assert rc.bus_model().housekeeping_W == 900.0
    assert rc.departure_vinf_kms == pytest.approx(4.6)


def test_saving_the_project_keeps_its_profile(profiled_project):
    state.open_project(profiled_project)
    S.build_model = "default"
    state.persist_study("Profiled", profiled_project)
    assert load_study(profiled_project).build_model == "default"


def test_a_remembered_slider_wins_over_the_mission_value(profiled_project):
    state.open_project(profiled_project)
    S.departure_vinf = 2.0
    session.remember()
    state.close_project()
    state.open_project(profiled_project)
    assert S.departure_vinf == pytest.approx(2.0)


def test_a_project_without_a_speed_opens_at_zero():
    state.open_project("example-study")
    try:
        assert S.departure_vinf == 0.0
        assert S.build_model == "default"
    finally:
        state.close_project()
