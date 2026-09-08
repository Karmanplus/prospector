"""The Export-PDF assembly (ui.report_export).

The section builders and the PDF renderer have their own tests in ``tests/reporting``, but those
hand-build the dicts each section takes. This covers the layer above: the one that reads app state,
decides which sections have anything to say, and hands them over. That layer is where a retired
piece of state or a stale gate goes unnoticed -- ``S.tof_trade`` was read here for a workspace tab
that no longer exists, and Export PDF raised ``AttributeError`` on every study.

So the load-bearing assertion is simply that a report builds, from real config, through the real
assembly. Plus: that the numbers on it follow the config library rather than a built-in default.
"""
from __future__ import annotations

import tempfile
import warnings

import pytest

from prospector.spacecraft import buildability
from prospector.spacecraft.arrays import ArrayMassSegment
from tests import library as lib
from ui import report_export, state
from ui.state import S


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A study from the active library, open, with the report writing into ``tmp_path``."""
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    state.open_project(lib.study_key())
    yield
    state.close_project()


def test_a_report_builds_for_an_open_study_with_no_results(project):
    """The floor: a study with nothing solved still exports. Every result-bearing section degrades
    to a line saying so, and the config sections always have something to say, so there is no
    state in which the button is allowed to raise."""
    out = report_export._build()
    data = out.read_bytes()
    assert data[:5] == b"%PDF-"
    assert len(data) > 10_000


def test_the_report_never_reads_state_the_app_no_longer_keeps(project):
    """Every attribute the assembly reads off ``S`` has to still exist. Reading a retired one
    raises, and it raises for every study at once rather than for an odd configuration, so a
    missing name here is a total outage of the export."""
    import ast
    import inspect
    src = inspect.getsource(report_export)
    read = {n.attr for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "S"}
    assert read, "found no state reads to check; the walk is wrong"
    missing = sorted(a for a in read if not hasattr(S, a))
    assert not missing, f"the export reads state that no longer exists: {missing}"


def test_the_reports_array_mass_follows_the_config_not_a_built_in_default(project):
    """The array mass sets what is left for payload, so it has to come from the library's own
    power-to-mass curve. Sizing off the illustrative curve instead reported 100 W/kg to somebody
    holding a measured 153 W/kg table, and roughly halved the payload capacity.
    """
    rc = state.resolved()
    base = report_export._assess_build(rc, None)
    assert base and base.get("array_kg"), "no array mass to check"

    # The same vehicle against a curve at twice the specific power: half the array mass, and the
    # kilograms it frees land in payload capacity.
    light = buildability.load_bus_model().model_copy(update={"array_mass_curve": [
        ArrayMassSegment(lo_W=1.0, hi_W=1e6, w0=0.0, kg0=0.0, w1=2.0e5, kg1=1.0e3)]})
    heavy = light.model_copy(update={"array_mass_curve": [
        ArrayMassSegment(lo_W=1.0, hi_W=1e6, w0=0.0, kg0=0.0, w1=1.0e5, kg1=1.0e3)]})
    v = rc.vehicle
    terms = dict(dry_kg=float(v.dry_mass), prop_kg=float(v.fuel_mass), mounts=v.mounts,
                 catalog=rc.engines, margin_pct=v.array_margin_pct)
    a = buildability.assess_assembly(**terms, model=light)
    b = buildability.assess_assembly(**terms, model=heavy)
    # The assessment rounds its kilograms for display, so compare to that precision.
    assert a["array_kg"] == pytest.approx(b["array_kg"] / 2.0, abs=0.02)
    assert a["payload_capacity_kg"] > b["payload_capacity_kg"]
    freed = b["array_kg"] - a["array_kg"]
    assert a["payload_capacity_kg"] - b["payload_capacity_kg"] == pytest.approx(freed, abs=0.02)


def test_a_build_model_with_no_array_curve_says_so_instead_of_sizing_quietly(tmp_path):
    """A build model with no curve falls back to the illustrative one, which is a round 100 W/kg
    and looks like an answer. It has to announce itself, because the number it produces is
    otherwise indistinguishable from a measured one."""
    path = tmp_path / "build-model.yaml"
    path.write_text("power_margin_pct: 10.0\n")
    with pytest.warns(UserWarning, match="no array_mass_curve"):
        model = buildability.load_bus_model(path)
    assert model.array_mass_curve, "the fallback still has to produce a usable curve"

    # A file that carries one is silent.
    path.write_text("power_margin_pct: 10.0\n"
                    "array_mass_curve:\n"
                    "- {lo_W: 50.0, hi_W: 9000.0, w0: 50.0, kg0: 1.0, w1: 9000.0, kg1: 60.0}\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        buildability.load_bus_model(path)


def test_a_report_carries_the_transfer_grid_when_one_is_in_force(project, tmp_path, monkeypatch):
    """The curve across the transfer grid is the flight-time trade, so the grid is what carries
    that trade into the document. It goes through the same ``products`` gate the workspace uses, so
    a grid the app has already retired as stale cannot reach paper.
    """
    from prospector import jobs, paths
    from prospector.figures import products

    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: None)
    rc = state.resolved()
    focus = dict(S.focus or {"pdes": "X", "full_name": "X"})
    run_id = jobs.submit_grid(rc.model_dump(mode="json"), focus, options={})
    jobs.write_grid_result(run_id, {
        "dep_mjd2000": [9000.0, 9010.0], "tof_days": [400.0, 300.0],
        "dv_kms": [[3.0, 3.4], [3.1, 3.5]], "final_mass_kg": [[500.0, 480.0], [495.0, 470.0]],
        "feasible": [[True, True], [True, True]], "decision_vectors": [[[0.0]] * 2] * 2,
        "n_feasible": 4, "n_cells": 4, "target_name": "test body",
        "frontier": [{"tof_days": 300.0, "dv_kms": 3.4, "dep_mjd2000": 9000.0,
                      "final_mass_kg": 480.0, "source": "grid"},
                     {"tof_days": 400.0, "dv_kms": 3.0, "dep_mjd2000": 9000.0,
                      "final_mass_kg": 500.0, "source": "polish"}]})
    jobs.write_grid_status(run_id, state=jobs.DONE, progress=1.0)
    S.grid_run_id, S.focus = run_id, focus
    try:
        grid = products.grid_block(run_id, rc, None, target_pdes=str(focus.get("pdes") or ""))
        assert grid is not None, "the grid this test just wrote should be in force"
        built = products.grid_figure(grid, rc)
        assert built is not None and built[1].data, "the grid figure came back empty"
        # The caption is editorial, so this checks that one exists and carries the count of cells
        # that solved, not how it is worded. How many of them closed is the figure's own result,
        # and a reader cannot judge the surface without it.
        assert built[0].strip(), "the figure came back with no caption"
        assert "4 of 4" in built[0], f"the caption does not report the cell count: {built[0]!r}"

        data = report_export._build().read_bytes()
        assert data[:5] == b"%PDF-"

        # A grid whose target has moved on is refused rather than printed under the wrong body.
        S.focus = {**focus, "pdes": "something else"}
        assert products.grid_block(run_id, rc, None,
                                  target_pdes="something else") is None
    finally:
        S.grid_run_id, S.grid_sel = None, None
