"""What a finished design-search session leaves behind, and how it reports progress.

REPORT.md and combos.csv are what a person actually reads to pick a vehicle, and status.json is
what the app polls, so these are the outputs a wrong verdict would reach a human through. None of
this ran before: the search's artifact layer was the least-covered code in the library.
"""
from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest

from prospector.trades.design_search import report as rp
from prospector.trades.design_search import session as se
from tests import library as lib


def _verdict(dry: float, prop: float = 300.0, *, success: bool, margin: float | None,
             anchor: bool = False, **extra) -> dict:
    """One evaluated combo, shaped the way evaluate_one returns it."""
    return {
        "dry_kg": dry, "prop_kg": prop, "wet_kg": dry + prop,
        "prop_dry_ratio": round(prop / dry, 4), "n_engines": 3, "engine": "eng",
        "propellant": "Xenon", "propellant_key": "xenon", "propellant_estimated": False,
        "anchor": anchor, "success": success, "margin_kms": margin,
        "dv_short_kms": None if margin is None or margin >= 0 else -margin,
        "prop_margin_kg": 40.0 if success else -12.0,
        "capability_kms": 8.0, "why": "ok" if success else "over budget",
        "n_converged": 1, "elapsed_s": 3.0,
        "best_point": {"vinf_kms": 0.6, "escape_dv_kms": 1.0, "cruise_dv_kms": 4.0,
                       "total_dv_kms": 5.0, "tof_days": 400.0,
                       "dep_date": "2028-07-01", "arr_date": "2029-08-05"},
        "points": [{"vinf_kms": 0.6, "converged": True, "success": success,
                    "total_dv_kms": 5.0, "margin_kms": margin}],
        **extra,
    }


def _search(tmp_path, evaluations: list[dict]) -> SimpleNamespace:
    """A stand-in for the Search session: write_outputs only reads these four attributes."""
    return SimpleNamespace(
        out=tmp_path,
        target={"name": "99942 Apophis"},
        args=SimpleNamespace(mission=lib.mission_key(), engine="eng", duty=0.9,
                             max_wet=750.0),
        mission=SimpleNamespace(launch_window=("2028-07-01", "2028-08-01"),
                                arrive_by="2030-08-01", launch_orbit="SSO"),
        evaluations={i: v for i, v in enumerate(evaluations)},
        gases=["xenon"],
    )


def _line(dry: float | None, **extra) -> dict:
    return {"engine": "eng", "gas": "xenon", "n_engines": 3, "prop_kg": 300.0,
            "frontier_dry_kg": dry, "why": "trajectory-limited", **extra}


# ---------------------------------------------------------------------------
# the winner
# ---------------------------------------------------------------------------

def test_the_winner_is_the_heaviest_closing_design(tmp_path):
    """Heavier dry mass at a fixed propellant load means more spacecraft delivered, so among
    designs that close the heaviest is the win."""
    s = _search(tmp_path, [_verdict(150.0, success=True, margin=1.5),
                           _verdict(190.0, success=True, margin=0.2),
                           _verdict(210.0, success=False, margin=-0.4)])
    rp.write_outputs(s, [_line(190.0)], [], None)
    md = (tmp_path / "REPORT.md").read_text()
    assert "**Dry 190 kg" in md


def test_the_reference_vehicle_never_wins(tmp_path):
    """The anchor is a fixed cross-session comparison point, not a requested combo. If it could
    win, a session would report a vehicle nobody asked about."""
    s = _search(tmp_path, [_verdict(300.0, success=True, margin=2.0, anchor=True),
                           _verdict(180.0, success=True, margin=0.5)])
    rp.write_outputs(s, [_line(180.0)], [], s.evaluations[0])
    md = (tmp_path / "REPORT.md").read_text()
    assert "**Dry 180 kg" in md
    assert "Reference vehicle" in md          # reported on its own line instead


def test_a_session_where_nothing_closes_says_so_and_still_reports(tmp_path):
    """A session that found nothing is a result, not an error: the near misses are the output."""
    s = _search(tmp_path, [_verdict(200.0, success=False, margin=-0.3),
                           _verdict(220.0, success=False, margin=-1.1)])
    rp.write_outputs(s, [_line(None, why="even the lightest dry mass fails")], [], None)
    md = (tmp_path / "REPORT.md").read_text()
    assert "No evaluated combo closed" in md
    assert "Near misses" in md


def test_near_misses_are_ordered_closest_first(tmp_path):
    """The point of the near-miss table is 'what would it take', so the most nearly-closing
    design has to lead."""
    s = _search(tmp_path, [_verdict(200.0, success=False, margin=-1.10),
                           _verdict(210.0, success=False, margin=-0.05),
                           _verdict(220.0, success=False, margin=-0.50)])
    rp.write_outputs(s, [_line(None)], [], None)
    body = (tmp_path / "REPORT.md").read_text().split("## Near misses")[1]
    rows = [ln for ln in body.splitlines() if ln.startswith("| 2")]
    assert [r.split("|")[1].strip() for r in rows] == ["210", "220", "200"]


def test_the_anchor_is_excluded_from_the_near_misses_too(tmp_path):
    s = _search(tmp_path, [_verdict(300.0, success=False, margin=-0.01, anchor=True),
                           _verdict(200.0, success=False, margin=-0.90)])
    rp.write_outputs(s, [_line(None)], [], s.evaluations[0])
    body = (tmp_path / "REPORT.md").read_text().split("## Near misses")[1]
    # Match on the leading dry-mass cell only: 300 also appears as the propellant load.
    dry_cells = [ln.split("|")[1].strip() for ln in body.splitlines() if ln.startswith("| 2")]
    assert dry_cells == ["200"]


# ---------------------------------------------------------------------------
# the machine-readable artifacts
# ---------------------------------------------------------------------------

def test_every_evaluated_combo_reaches_combos_csv(tmp_path):
    s = _search(tmp_path, [_verdict(150.0, success=True, margin=1.0),
                           _verdict(200.0, success=False, margin=-0.5)])
    rp.write_outputs(s, [_line(150.0)], [], None)
    rows = list(csv.DictReader((tmp_path / "combos.csv").open()))
    assert {r["dry_kg"] for r in rows} == {"150.0", "200.0"}
    assert {r["success"] for r in rows} == {"True", "False"}


def test_swept_model_settings_survive_as_plottable_columns(tmp_path):
    """A --sweep-param run varies a build-model knob per combo; if the column is dropped the
    session's whole second axis is invisible in the artifacts."""
    s = _search(tmp_path, [_verdict(150.0, success=True, margin=1.0, param_power_margin_pct=10.0),
                           _verdict(150.0, success=True, margin=0.4, param_power_margin_pct=30.0)])
    rp.write_outputs(s, [_line(150.0)], [], None)
    rows = list(csv.DictReader((tmp_path / "combos.csv").open()))
    assert {r["param_power_margin_pct"] for r in rows} == {"10.0", "30.0"}


def test_result_json_records_the_whole_session(tmp_path):
    s = _search(tmp_path, [_verdict(150.0, success=True, margin=1.0)])
    rp.write_outputs(s, [_line(150.0)], [], None)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["target"] == "99942 Apophis"
    assert result["winner"]["dry_kg"] == 150.0
    assert len(result["evaluations"]) == 1


def test_the_analytic_map_is_written_when_present_and_skipped_when_not(tmp_path):
    s = _search(tmp_path, [_verdict(150.0, success=True, margin=1.0)])
    rp.write_outputs(s, [_line(150.0)], [], None)
    assert not (tmp_path / "analytic_map.csv").exists()
    rp.write_outputs(s, [_line(150.0)],
                     [{"dry_kg": 150.0, "prop_kg": 300.0, "bound_margin_kms": 0.4}], None)
    assert (tmp_path / "analytic_map.csv").exists()


def test_numpy_values_survive_json_serialization(tmp_path):
    """Verdicts carry numpy scalars out of the solvers; result.json must not fail on them."""
    import numpy as np
    v = _verdict(150.0, success=True, margin=np.float64(1.25))
    v["n_converged"] = np.int64(2)
    rp.write_outputs(_search(tmp_path, [v]), [_line(150.0)], [], None)
    assert json.loads((tmp_path / "result.json").read_text())["winner"]["margin_kms"] == 1.25


# ---------------------------------------------------------------------------
# the status the app polls
# ---------------------------------------------------------------------------

def test_status_is_written_atomically(tmp_path):
    """The app polls this file while it is being rewritten; a half-written read would show a
    session as failed. The temp file must not survive the rename either."""
    se.write_session_status(tmp_path, state="running", message="working", n_evaluated=7)
    assert json.loads((tmp_path / "status.json").read_text())["n_evaluated"] == 7
    assert not (tmp_path / "status.json.tmp").exists()


def test_status_carries_the_progress_denominator(tmp_path):
    """The combo count has to reach status.json, or the app's progress bar has nothing to divide
    the finished evaluations by and shows a bare count instead."""
    se.write_session_status(tmp_path, state="running", message="running",
                            progress={"n_planned": 240})
    assert json.loads((tmp_path / "status.json").read_text())["n_planned"] == 240


def test_status_records_a_terminal_state(tmp_path):
    for state in ("done", "error", "stopped"):
        se.write_session_status(tmp_path, state=state, message=f"{state} now", n_evaluated=5)
        assert json.loads((tmp_path / "status.json").read_text())["state"] == state


def test_cancelled_is_the_documented_stop_signal():
    """The app asks a search to wind down by raising this between evaluations, so it must stay
    an exception the driver can catch rather than an error state."""
    assert issubclass(se.Cancelled, Exception)
    with pytest.raises(se.Cancelled):
        raise se.Cancelled()
