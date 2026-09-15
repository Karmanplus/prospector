"""Tests for the desirability axis (prospector.enrichment).

These lock the parts that must not drift: the normalized column contract, the pure downselect
(which must *keep* unknowns -- never silently drop a reachable target), and the facade's behaviour
around the per-object cache (calm "unavailable" vs hard "failed").

The backend is stubbed at the source modules rather than at the functions that call them.
``backend`` imports each source as a module and calls ``source.lookup(...)``, so replacing the
attribute on the module object is what intercepts the call, patching a name the package re-exports
would leave the real lookup running against the live catalogues.
"""
import pandas as pd
import pytest

from prospector.config import Desirability
from prospector.enrichment import (
    EnrichmentFailed,
    EnrichmentUnavailable,
    apply_selection,
    backend,
    desirability_mask,
    enrich,
    normalize,
)
from prospector.enrichment.sources import ssodnet

# ---- schema.normalize ----

def test_normalize_aliases_sentinels_and_contract():
    raw = pd.DataFrame([
        {"input_id": "341843", "tier": "s", "taxonomy": "C", "albedo": "0.08",
         "period": "3.7", "diameter": "400", "p_crit_cdf": "0.95", "q_min": "0.08"},
        {"input_id": "X1", "tier": "N/A", "taxonomy": "", "albedo": "nan",
         "period": "", "diameter": "N/A", "p_crit_cdf": "none", "q_min": "-"},
    ])
    df = normalize(raw)
    # Unit-free aliases mapped onto the contract; every contract column present.
    assert {"period_h", "diameter_m", "p_gt_pcrit", "q_min_au"} <= set(df.columns)
    assert df.loc[0, "tier"] == "S"          # case-normalized
    assert df.loc[0, "period_h"] == 3.7 and df.loc[0, "diameter_m"] == 400.0
    # every flavor of "missing" becomes NaN/None, never a stray string.
    assert pd.isna(df.loc[1, "tier"]) and pd.isna(df.loc[1, "period_h"])
    assert pd.isna(df.loc[1, "albedo"]) and pd.isna(df.loc[1, "q_min_au"])


def test_normalize_requires_join_key():
    with pytest.raises(ValueError, match="input_id"):
        normalize(pd.DataFrame({"tier": ["S"]}))


# ---- selection.desirability_mask (the pure downselect) ----

def _enriched():
    raw = pd.DataFrame([
        {"input_id": "S1", "tier": "S", "taxonomy": "C", "period": "6", "diameter": "400"},
        {"input_id": "B1", "tier": "b", "taxonomy": "Sq", "period": "30", "diameter": "340"},
        {"input_id": "U1", "tier": "N/A", "taxonomy": "", "period": "", "diameter": "N/A"},
        {"input_id": "M1", "tier": "C", "taxonomy": "M", "period": "1.2", "diameter": "900"},
    ])
    df = normalize(raw)
    df["reachable"] = True
    return df


def _kept(df, des):
    return list(df.loc[apply_selection(df, des)["selected"], "input_id"])


def test_default_lists_the_characterized_rows_only():
    df = _enriched()
    # U1 has no tier: not characterized, so hidden until asked for.
    assert _kept(df, Desirability()) == ["S1", "B1", "M1"]
    assert _kept(df, Desirability(keep_unknown=True)) == ["S1", "B1", "U1", "M1"]


def test_tier_floor_keeps_better_and_only_with_the_switch_the_unknown():
    df = _enriched()
    # min_tier A keeps S (better); drops B and C; the unknown-tier row only when asked.
    assert _kept(df, Desirability(min_tier="A")) == ["S1"]
    assert _kept(df, Desirability(min_tier="A", keep_unknown=True)) == ["S1", "U1"]


def test_taxonomy_include_keeps_listed_and_with_the_switch_the_unknown():
    df = _enriched()
    # carbonaceous keeps the C row; drops S-complex and metallic; the unknown one only when asked.
    assert _kept(df, Desirability(taxonomy_include=["C", "B", "D"])) == ["S1"]
    assert _kept(df, Desirability(taxonomy_include=["C", "B", "D"], keep_unknown=True)) == ["S1", "U1"]


def test_period_and_diameter_floors_never_drop_for_missing_data_when_kept():
    df = _enriched()
    keep = dict(keep_unknown=True)
    assert _kept(df, Desirability(min_period_h=2.0, **keep)) == ["S1", "B1", "U1"]   # fast M1 dropped
    assert _kept(df, Desirability(min_diameter_m=350, **keep)) == ["S1", "U1", "M1"]  # 340 m B1 dropped
    assert _kept(df, Desirability(min_diameter_m=350)) == ["S1", "M1"]


def test_uncharacterized_is_flagged_whether_listed_or_hidden():
    """The ``uncharacterized`` column names the rows that are not characterized or that a filter
    could not test, in both switch positions, so the count of measured matches is always known."""
    df = _enriched()
    for keep in (False, True):
        out = apply_selection(df, Desirability(min_diameter_m=350, keep_unknown=keep))
        assert list(out.loc[out["uncharacterized"], "input_id"]) == ["U1"]
        assert list(out.loc[out["selected"] & ~out["uncharacterized"], "input_id"]) == ["S1", "M1"]
    # A row a known value already ruled out is not "unknown", whatever else is missing.
    out = apply_selection(df, Desirability(min_diameter_m=350, min_period_h=100))
    assert not bool(out.loc[out["input_id"] == "B1", "uncharacterized"].iloc[0])
    # Nothing in force: only the uncharacterized row is unknown.
    assert list(apply_selection(df, Desirability()).query("uncharacterized")["input_id"]) == ["U1"]


def test_unenriched_frame_selected_equals_reachable():
    bare = pd.DataFrame({"input_id": ["a", "b"], "reachable": [True, False]})
    # No enrichment columns -> every filter is inert; selected never widens past reachable.
    assert list(apply_selection(bare, Desirability(min_tier="S"))["selected"]) == [True, False]


def test_mask_is_boolean_series():
    df = _enriched()
    mask = desirability_mask(df, Desirability(min_tier="B"))
    assert mask.dtype == bool and len(mask) == len(df)


# ---- the backend boundary ----

@pytest.fixture
def offline(monkeypatch):
    """Every source stubbed out, so a test drives the backend without touching the network.

    ``characterize`` is replaced wholesale here; the per-source assembly it performs is tested
    against stubbed sources in test_backend.py.
    """
    monkeypatch.setattr(backend, "_preflight", lambda: None)
    monkeypatch.setattr(backend.astorb, "spin_catalog", lambda *a, **k: ([], []))
    monkeypatch.setattr(backend.perihelion, "load_toliou_table", lambda *a, **k: None)
    monkeypatch.setattr(backend.ssodnet, "lookup_many", lambda ids: {})
    monkeypatch.setattr(backend.astorb, "lookup_many", lambda ids: {})

    known: dict[str, dict] = {}

    def _characterize(identifier, **_kwargs):
        return known.get(identifier)

    monkeypatch.setattr(backend, "characterize", _characterize)
    return known


def test_backend_fetches_each_chunk_in_bulk_and_hands_the_results_on(offline, monkeypatch):
    """The two remote lookups are made once per chunk for every id in it, and what they return
    reaches ``characterize`` so it does not look the body up again."""
    asked: dict[str, list] = {"ssodnet": [], "astorb": []}
    monkeypatch.setattr(backend.ssodnet, "lookup_many",
                        lambda ids: asked["ssodnet"].append(list(ids)) or {"A1": "body-A1"})
    monkeypatch.setattr(backend.astorb, "lookup_many",
                        lambda ids: asked["astorb"].append(list(ids)) or {"A1": "cat-A1"})
    handed = {}

    def _characterize(identifier, **kwargs):
        handed[identifier] = (kwargs.get("body"), kwargs.get("catalog_data"))
        return _row(identifier)

    monkeypatch.setattr(backend, "characterize", _characterize)
    monkeypatch.setattr(backend, "PREFETCH_CHUNK", 2)
    backend.run(["A1", "B2", "C3"], n_workers=2)
    assert asked["ssodnet"] == [["A1", "B2"], ["C3"]] and asked["astorb"] == asked["ssodnet"]
    assert handed["A1"] == ("body-A1", "cat-A1") and handed["B2"] == (None, None)


def test_a_source_outage_ends_the_run_as_a_failure_not_as_missing_software(offline, monkeypatch):
    """The package check already passed, so a source saying it is unavailable mid-run is the
    remote service; the run fails with that reason rather than caching thin rows or reading as
    'not installed'."""
    from prospector.enrichment.sources import ssodnet

    def _down(ids):
        raise ssodnet.SourceUnavailable("tables service down; try again later")
    monkeypatch.setattr(backend.ssodnet, "lookup_many", _down)
    offline.update({"A1": _row("A1")})
    with pytest.raises(EnrichmentFailed, match="tables service down"):
        backend.run(["A1"], n_workers=1)


def test_a_failed_bulk_fetch_falls_back_to_per_body_lookups(offline, monkeypatch):
    def _down(ids):
        raise ConnectionError("bulk endpoint down")
    monkeypatch.setattr(backend.ssodnet, "lookup_many", _down)
    offline.update({"A1": _row("A1")})
    assert list(backend.run(["A1"], n_workers=1)["input_id"]) == ["A1"]


def _row(identifier, tier="S"):
    return {"input_id": identifier, "name": identifier, "tier": tier, "taxonomy": "C",
            "albedo": 0.05, "period_h": 6.0, "diameter_m": 400.0}


def test_backend_unavailable_without_optional_dependency(monkeypatch):
    def _no_rocks(name, *args, **kwargs):
        if name == "rocks":
            raise ImportError("no rocks here")
        return __import__(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _no_rocks)
    with pytest.raises(EnrichmentUnavailable, match="rocks"):
        backend.run(["1"])


def test_backend_returns_a_row_per_resolved_target(offline):
    offline.update({"A1": _row("A1"), "B2": _row("B2", tier="B")})
    out = backend.run(["A1", "B2"], n_workers=2)
    assert sorted(out["input_id"]) == ["A1", "B2"]


def test_backend_skips_unresolvable_targets_without_failing_the_run(offline):
    offline["A1"] = _row("A1")
    out = backend.run(["A1", "NOT-A-BODY"], n_workers=2)
    # One bad designation must not take the rest of the batch with it.
    assert list(out["input_id"]) == ["A1"]


def test_backend_fails_only_when_nothing_resolved(offline):
    with pytest.raises(EnrichmentFailed):
        backend.run(["NOT-A-BODY"])


def test_backend_reports_progress_and_streams_results(offline):
    offline.update({"A1": _row("A1"), "B2": _row("B2")})
    seen, streamed = [], []
    backend.run(["A1", "B2"], n_workers=1,
                on_progress=lambda done, total, _msg: seen.append((done, total)),
                on_result=lambda frame: streamed.append(frame["input_id"].iloc[0]))
    assert seen == [(1, 2), (2, 2)]
    # Each row is handed over as it completes, so a cancelled run keeps what it already had.
    assert sorted(streamed) == ["A1", "B2"]


def test_backend_stops_starting_targets_once_superseded(offline):
    offline.update({str(i): _row(str(i)) for i in range(6)})
    backend.run(["0"], n_workers=1)             # warm-up call, so `characterize` is known good
    out = backend.run([str(i) for i in range(6)], n_workers=1, should_continue=lambda: False)
    assert len(out) == 0                        # superseded before any target started
    # Superseded is not a failure: no EnrichmentFailed even though nothing was characterized.


# ---- the facade: per-object caching ----

def test_enrich_caches_per_object(offline, tmp_path):
    offline.update({"341843": _row("341843"), "99942": _row("99942"),
                    "101955": _row("101955")})
    cache = tmp_path / "cache"
    first = enrich(["341843", "99942"], cache_dir=cache)
    assert len(list(cache.rglob("*.parquet"))) == 2, "one cache file per object"

    # Re-screening an OVERLAPPING set: the cached object loads from disk, only the new one is
    # characterized. Emptying the stub proves it: '341843' still resolves from cache.
    second = enrich(["341843", "101955"], cache_dir=cache)
    assert list(second["input_id"]) == ["341843", "101955"]

    offline.clear()
    third = enrich(["341843", "99942"], cache_dir=cache)
    assert list(first["input_id"]) == list(third["input_id"]) == ["341843", "99942"]


def test_enrich_keeps_cached_when_new_id_fails(offline, tmp_path):
    offline["GOOD1"] = _row("GOOD1")
    cache = tmp_path / "cache"
    enrich(["GOOD1"], cache_dir=cache)
    # GOOD1 is cached; BAD1 is a new, unresolvable id whose characterization fails, the cached
    # GOOD1 must survive rather than the whole call raising.
    out = enrich(["GOOD1", "BAD1"], cache_dir=cache)
    assert list(out["input_id"]) == ["GOOD1"]


def test_enrich_raises_when_nothing_cached_and_backend_fails(offline, tmp_path):
    with pytest.raises(EnrichmentFailed):
        enrich(["BAD1"], cache_dir=tmp_path / "c")


def test_enrich_reports_cached_targets_in_progress(offline, tmp_path):
    offline.update({"A1": _row("A1"), "B2": _row("B2")})
    cache = tmp_path / "cache"
    enrich(["A1"], cache_dir=cache)

    seen = []
    enrich(["A1", "B2"], cache_dir=cache, on_progress=lambda d, t, _m: seen.append((d, t)))
    # The cached target counts toward the total immediately, so the bar does not appear stuck.
    assert seen[0] == (1, 2) and seen[-1] == (2, 2)


def test_source_unavailable_propagates_out_of_the_pool(offline, monkeypatch):
    """A missing dependency is a run-wide failure, not a per-target one, and must not be
    swallowed by the per-target error handling that exists to survive bad designations."""
    def _boom(identifier, **_kwargs):
        raise ssodnet.SourceUnavailable("rocks is not installed")

    monkeypatch.setattr(backend, "characterize", _boom)
    with pytest.raises(ssodnet.SourceUnavailable):
        backend.run(["A1"])
