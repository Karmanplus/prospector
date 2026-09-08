"""Tests for the small-body target lookup (prospector.population).

All tests run against a synthetic population frame, ``fetch_population`` is monkeypatched so
nothing touches the network or the on-disk cache. They pin the lookup contract the UI relies on:
case-insensitive matching, exact-designation-first ordering, None/empty on a miss, and a target
dict whose elements come back as plain floats.
"""
import pandas as pd
import pytest

from prospector.population import targets

_COLUMNS = ["spkid", "pdes", "full_name", "H", "condition_code",
            "a", "e", "i", "om", "w", "ma", "epoch"]

_ROWS = [
    # spkid, pdes, full_name (SBDB pads it), H, cc, a, e, i, om, w, ma, epoch
    ["2099942", "99942", "   99942 Apophis (2004 MN4)", 19.7, "0",
     0.9226, 0.1915, 3.34, 203.9, 126.7, 215.5, 2461000.5],
    ["3430291", "2008 EV5", "      (2008 EV5)", 20.0, "0",
     0.9583, 0.0835, 7.44, 93.4, 234.8, 12.3, 2461000.5],
    ["2000433", "433", "    433 Eros (A898 PA)", 10.4, "0",
     1.4580, 0.2228, 10.83, 304.3, 178.9, 271.1, 2461000.5],
    ["2001433", "1433", "   1433 Geramtina (1937 UC)", 11.9, "0",
     2.2150, 0.1230, 5.10, 10.0, 20.0, 30.0, 2461000.5],
]


@pytest.fixture
def population():
    return pd.DataFrame(_ROWS, columns=_COLUMNS)


@pytest.fixture
def patched_fetch(population, monkeypatch):
    """Replace the catalog load with the synthetic frame, recording the call kwargs.

    Patched on the ``targets`` module rather than on the package facade: a function resolves names
    in its own module, so patching the re-export would leave the lookup reading the real cached
    population off disk.
    """
    calls = []

    def _fake_load(h_max=25.0, *args, **kwargs):
        calls.append(h_max)
        return population
    monkeypatch.setattr(targets, "load_population", _fake_load)
    return calls


# ---- search_targets ----

def test_search_exact_pdes_ranks_first(population):
    # "433" is a substring of both pdes "433" and "1433"; the exact hit must lead.
    hits = targets.search_targets("433", population=population)
    assert list(hits["pdes"]) == ["433", "1433"]


def test_search_name_substring_is_case_insensitive(population):
    hits = targets.search_targets("aPoPhIs", population=population)
    assert len(hits) == 1
    assert hits.iloc[0]["pdes"] == "99942"


def test_search_miss_and_blank_return_empty(population):
    assert targets.search_targets("zzgarbage", population=population).empty
    assert targets.search_targets("   ", population=population).empty


def test_search_respects_limit(population):
    # Every synthetic full_name contains a "(", all four rows match.
    assert len(targets.search_targets("(", population=population)) == 4
    assert len(targets.search_targets("(", population=population, limit=2)) == 2


def test_search_loads_population_with_generous_default(patched_fetch):
    hits = targets.search_targets("99942")
    assert len(hits) == 1
    # The default cutoff must keep famous references (Apophis H~19.7, 2008 EV5 H~20.0) in the
    # searchable universe, at least as deep as the Screening default (25.0).
    assert patched_fetch == [targets.DEFAULT_H_MAX]
    assert targets.DEFAULT_H_MAX >= 25.0


# ---- get_target ----

def test_get_target_exact_pdes(population):
    t = targets.get_target("2008 EV5", population=population)
    assert t is not None
    assert t["pdes"] == "2008 EV5"
    assert t["name"] == "(2008 EV5)"           # padded full_name, stripped
    # Solver-facing elements are plain floats.
    for col in ("a", "e", "i", "om", "w", "ma", "epoch", "H"):
        assert isinstance(t[col], float)
    assert t["a"] == pytest.approx(0.9583)
    assert t["i"] == pytest.approx(7.44)
    # The full row rides along, so the dict works as a solver target_row.
    assert t["spkid"] == "3430291"


def test_get_target_unambiguous_name(population):
    t = targets.get_target("apophis", population=population)
    assert t is not None and t["pdes"] == "99942"


def test_get_target_ambiguous_substring_returns_none(population):
    # "433" exactly matches pdes 433, but "43" is a substring of two bodies and an exact
    # designation of none, ambiguous, so the lookup refuses to guess.
    assert targets.get_target("433", population=population)["pdes"] == "433"
    assert targets.get_target("43", population=population) is None


def test_get_target_miss_returns_none(population):
    assert targets.get_target("Bennu", population=population) is None
    assert targets.get_target("", population=population) is None


def test_get_target_uses_default_population(patched_fetch):
    t = targets.get_target("433")
    assert t is not None and t["i"] == pytest.approx(10.83)
    assert patched_fetch == [targets.DEFAULT_H_MAX]


# ---- display_name ----

def test_display_name_prefers_full_name_then_falls_back():
    assert targets.display_name({"full_name": "  99942 Apophis (2004 MN4) ",
                                 "pdes": "99942"}) == "99942 Apophis (2004 MN4)"
    assert targets.display_name({"full_name": "  ", "pdes": "99942"}) == "99942"
    assert targets.display_name({"full_name": float("nan"), "name": "Eros"}) == "Eros"
    assert targets.display_name({}) == "target"


def test_display_name_accepts_series(population):
    assert targets.display_name(population.iloc[2]) == "433 Eros (A898 PA)"
