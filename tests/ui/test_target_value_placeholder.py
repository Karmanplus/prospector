"""The Value view's placeholder tells the truth about the characterization job.

A default install has no characterization (it needs the optional enrichment environment), and the
job the view submits ends in ERROR within seconds. The first person to install the tool fresh saw
"characterizing reachable targets..." for good and took it for a hang. The placeholder now reads the
job's status: progress while it runs, a plain "not installed here" with the command that enables it
when the extra is missing, the error when it failed, and nothing alarming when there is no job.
"""
import pytest

from prospector import jobs
from ui.state import S
from ui.workspaces import target


@pytest.fixture
def shown(monkeypatch):
    calls = []
    monkeypatch.setattr(target, "_empty", lambda icon, text, detail=None: calls.append((icon, text, detail)))
    monkeypatch.setattr(S, "enrich_reachable", ["99942"])
    return calls


def _with_status(monkeypatch, status):
    monkeypatch.setattr(target, "_enrich_status", lambda: status)


def test_running_shows_progress(monkeypatch, shown):
    _with_status(monkeypatch, {"state": jobs.RUNNING, "done": 12, "total": 262})
    target._value_placeholder()
    (icon, text, detail), = shown
    assert icon == "diamond" and "12/262" in text and "Cancel" in detail


def test_missing_extra_says_so_and_names_the_command(monkeypatch, shown):
    _with_status(monkeypatch, {"state": jobs.ERROR, "message": "enrichment unavailable",
                               "error": "enrichment needs the 'rocks' package"})
    target._value_placeholder()
    (_icon, text, detail), = shown
    assert "not installed" in text
    assert "pixi run -e enrichment app" in detail
    assert "characterizing" not in text


def test_a_real_failure_reads_as_one(monkeypatch, shown):
    _with_status(monkeypatch, {"state": jobs.ERROR, "message": "failed", "error": "SsODNet timed out"})
    target._value_placeholder()
    (_icon, text, detail), = shown
    assert text.startswith("Characterization failed")
    assert "SsODNet timed out" in detail


def test_no_job_is_not_a_spinner(monkeypatch, shown):
    monkeypatch.setattr(S, "enrich_reachable", None)
    target._value_placeholder()
    (_icon, text, detail), = shown
    assert "characterizing" not in text and detail is None
