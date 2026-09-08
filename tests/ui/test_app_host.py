"""The app listens on loopback unless told otherwise.

NiceGUI's own default is 0.0.0.0 in browser mode, which is what Linux and ``PROSPECTOR_UI=web``
get. There is no login in front of the app, and it writes the config library and starts solver
processes, so that default would hand the app to anyone on the network. This runs ``ui/app.py`` as
a launch with ``ui.run`` replaced and checks what host it was handed.
"""
import runpy

import pytest
from nicegui import ui

from prospector import paths

APP = paths.REPO_ROOT / "ui" / "app.py"


@pytest.fixture
def launch(monkeypatch):
    calls = []
    monkeypatch.setattr(ui, "run", lambda *a, **kw: calls.append(kw))
    monkeypatch.setenv("PROSPECTOR_UI", "web")

    def _launch():
        runpy.run_path(str(APP), run_name="__main__")
        (kw,) = calls
        return kw
    return _launch


def test_default_is_loopback(monkeypatch, launch):
    monkeypatch.delenv("PROSPECTOR_HOST", raising=False)
    kw = launch()
    assert kw["host"] == "127.0.0.1"
    assert kw["native"] is False


def test_host_can_be_opened_on_purpose(monkeypatch, launch):
    monkeypatch.setenv("PROSPECTOR_HOST", "0.0.0.0")
    assert launch()["host"] == "0.0.0.0"
