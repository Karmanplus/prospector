"""The Global-settings dialog edits one build-model profile and says which."""
import pytest
from nicegui import ui
from nicegui.testing import User

from prospector.spacecraft import buildability
from ui import state
from ui.state import S


@pytest.fixture
def opened():
    state.open_project("example-study")
    yield
    state.close_project()


@pytest.mark.anyio
@pytest.mark.nicegui_main_file("ui/app.py")
async def test_settings_dialog_names_the_open_projects_profile(user: User, opened) -> None:
    default = buildability.load_bus_model()
    buildability.save_bus_model(default.model_copy(update={"housekeeping_W": 777.0}), name="hungry")
    await user.open("/")
    user.find("Global settings").click()
    await user.should_see("Build-model profile")
    select = next(s for s in user.find(ui.select).elements
                  if s._props.get("label") == "Build-model profile")
    assert select.value == "default"
    assert set(select.options) >= {"default", "hungry"}


def test_adopting_a_profile_switches_the_project_to_it(opened) -> None:
    from ui import settings
    default = buildability.load_bus_model()
    buildability.save_bus_model(default.model_copy(update={"housekeeping_W": 777.0}), name="hungry")
    assert settings.adopt_profile("hungry")
    assert S.build_model == "hungry" and S.dirty
    assert state.resolved().bus_model().housekeeping_W == 777.0
    # Picking the profile already in use changes nothing.
    assert not settings.adopt_profile("hungry")


def test_no_open_project_adopts_nothing() -> None:
    from ui import settings
    state.close_project()
    assert not settings.adopt_profile("anything")
    assert S.build_model == "default"
