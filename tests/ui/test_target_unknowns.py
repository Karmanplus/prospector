"""Find targets: an unmeasured property reads "?" and the dock count separates measured matches
from the rows kept only because a filter had nothing to test."""
import math

import pandas as pd

from prospector.config import Desirability
from ui.state import S
from ui.workspaces import target as w


def test_unmeasured_cells_read_question_mark_not_blank():
    df = pd.DataFrame({"pdes": ["1", "2"], "full_name": ["a", "b"], "tier": ["A", None],
                       "taxonomy": ["C", None], "diameter_m": [340.0, math.nan],
                       "period_h": [5.0, math.nan], "lowthrust_dv": [4.0, math.nan]})
    columns, rows = w._table_data(df, ["full_name", "tier", "taxonomy", "diameter_m", "period_h",
                                       "lowthrust_dv"])
    assert rows[1]["tier"] == rows[1]["taxonomy"] == rows[1]["diameter_m"] == rows[1]["period_h"] == "?"
    assert rows[1]["lowthrust_dv"] is None                     # not a described property
    assert rows[0]["diameter_m"] == 340
    # The numeric columns carry a sort that puts "?" below every number.
    by_name = {c["name"]: c for c in columns}
    assert ":sort" in by_name["diameter_m"] and ":sort" not in by_name["tier"]


def test_dock_count_splits_matches_from_unknowns(monkeypatch):
    df = pd.DataFrame({"selected": [True, True, True, False],
                       "uncharacterized": [False, True, True, False]})
    monkeypatch.setattr(S, "desirability", Desirability(keep_unknown=True))
    assert w._selection_count(df) == "(1 match · 2 uncharacterized)"
    monkeypatch.setattr(S, "desirability", Desirability(keep_unknown=False))
    assert w._selection_count(df) == "(1 match · 2 uncharacterized hidden)"
    none = pd.DataFrame({"selected": [True], "uncharacterized": [False]})
    assert w._selection_count(none) == "(1)"


def test_characterization_looks_up_only_what_is_not_on_disk():
    """The nearest-by-ΔV window is what gets characterized; ids already cached are not sent again,
    so a finished batch leaves nothing to do and a wider window adds only the new ids."""
    pdes = [str(i) for i in range(10)]
    covered, todo = w._enrichment_plan(pdes, cached={"0", "2", "3"}, window=5)
    assert covered == ["0", "1", "2", "3", "4"] and todo == ["1", "4"]
    _, again = w._enrichment_plan(pdes, cached={"0", "1", "2", "3", "4"}, window=5)
    assert again == []
    _, wider = w._enrichment_plan(pdes, cached={"0", "1", "2", "3", "4"}, window=10)
    assert wider == ["5", "6", "7", "8", "9"]


def test_cached_ids_reads_nothing_and_names_only_what_exists(tmp_path):
    from prospector import enrichment
    from prospector.enrichment import _obj_cache_path
    hit = _obj_cache_path("99942", tmp_path)
    hit.parent.mkdir(parents=True)
    hit.write_bytes(b"not even parquet")
    assert enrichment.cached_ids(["99942", "2008 EV5", ""], tmp_path) == {"99942"}


def test_a_failed_job_shows_the_exceptions_own_words():
    status = {"state": "error", "message": "enrichment failed",
              "error": "Traceback (most recent call last):\n  File \"x.py\", line 1\n"
                       "prospector.enrichment.backend.EnrichmentFailed: SsODNet answered for the "
                       "bodies but not for any of their measurement tables; try again later\n"}
    assert w._error_reason(status) == ("SsODNet answered for the bodies but not for any of their "
                                       "measurement tables; try again later")
    assert w._error_reason({"message": "enrichment failed"}) == "enrichment failed"
    assert w._error_reason({}) == ""
