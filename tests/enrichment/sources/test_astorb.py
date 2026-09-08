"""Tests for the AstorbDB source.

No network: every test replaces the HTTP call with a canned response. What is being checked is the
shape handling; the database returns deeply nested nulls for absent measurements, and a body with
no albedo comes back as a record whose albedo key is present and None rather than missing, plus the
two behaviours the screen depends on: an unreachable database costs properties rather than targets,
and a repeated body is served from cache.
"""
import json

import pytest

from prospector.enrichment.measurements import preferred, preferred_float
from prospector.enrichment.sources import astorb


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Cache to a scratch directory, and reset the process-wide catalogue between tests."""
    monkeypatch.setattr(astorb, "cache_dir", lambda: tmp_path / "astorb")
    astorb._catalog = None
    yield
    astorb._catalog = None


def _responder(payload, calls=None):
    def _post(_url, json=None, timeout=None):        # noqa: A002 - requests' own kwarg name
        if calls is not None:
            calls.append(json)

        class _Response:
            @staticmethod
            def raise_for_status():
                pass

            @staticmethod
            def json():
                return payload

        return _Response()
    return _post


def _body(albedos=(), taxonomies=(), periods=(), tumbling=None, quality=None):
    survey = []
    for albedo in albedos:
        survey.append({"albedo": {"albedo": albedo}, "taxonomy": None, "lightcurve": None})
    for taxonomy in taxonomies:
        survey.append({"albedo": None, "lightcurve": None,
                       "taxonomy": {"taxonomy_sys_type": {"tax_type": taxonomy}}})
    for period in periods:
        survey.append({"albedo": None, "taxonomy": None, "lightcurve": {
            "period": period, "ambiguous_period": None,
            "non_principal_axis_rotator": tumbling,
            "code_lc_quality": {"code": quality} if quality else None}})
    return {"data": {"minorplanet": [{"ast_number": 341843, "surveydata": survey}]}}


def test_survey_rows_are_split_by_property(monkeypatch):
    monkeypatch.setattr(astorb.requests, "post",
                        _responder(_body(albedos=[0.09, 0.11], taxonomies=["C"],
                                         periods=[3.7], tumbling=False, quality="3")))
    out = astorb.lookup("341843")
    assert [m.value for m in out["albedo"]] == [0.09, 0.11]
    assert preferred(out["taxonomy"]) == "C"
    assert preferred_float(out["period_h"]) == 3.7
    assert preferred(out["tumbling"]) is False
    assert preferred(out["lightcurve_quality"]) == "3"


def test_nothing_is_flagged_preferred():
    # AstorbDB reports what each survey found without ranking them, so the choice must fall to the
    # source ordering rather than to a flag this source invents.
    measurements = astorb._survey_measurements(
        _body(albedos=[0.09])["data"]["minorplanet"][0]["surveydata"])
    assert not any(m.preferred for m in measurements["albedo"])


def test_absent_measurements_come_back_empty_not_as_nulls(monkeypatch):
    monkeypatch.setattr(astorb.requests, "post", _responder(_body(periods=[5.0])))
    out = astorb.lookup("341843")
    assert out["albedo"] == [] and out["taxonomy"] == []
    # A null tumbling flag is "not determined", which must not become a False the tier would read
    # as "confirmed not tumbling".
    assert out["tumbling"] == []


def test_an_unknown_body_yields_every_property_empty(monkeypatch):
    monkeypatch.setattr(astorb.requests, "post", _responder({"data": {"minorplanet": []}}))
    assert astorb.lookup("NOT-A-BODY") == {name: [] for name in astorb.PROPERTIES}


def test_an_unreachable_database_costs_properties_not_targets(monkeypatch):
    def _down(*_args, **_kwargs):
        raise ConnectionError("astorb is down")

    monkeypatch.setattr(astorb.requests, "post", _down)
    # Empty, not an exception: the caller assembles the row from whatever did arrive.
    assert astorb.lookup("341843") == {name: [] for name in astorb.PROPERTIES}


def test_numbered_and_designated_bodies_query_on_different_keys(monkeypatch):
    calls = []
    monkeypatch.setattr(astorb.requests, "post", _responder(_body(), calls))
    astorb.lookup("341843")
    astorb.lookup("2008 EV5")
    assert calls[0]["variables"] == {"designations": [], "numbers": [341843]}
    assert calls[1]["variables"] == {"designations": ["2008 EV5"], "numbers": []}


def test_a_repeated_body_is_served_from_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(astorb.requests, "post", _responder(_body(albedos=[0.09]), calls))
    first = astorb.lookup("341843")
    second = astorb.lookup("341843")
    assert len(calls) == 1, "a re-screen of an overlapping population must not re-query"
    assert [m.value for m in first["albedo"]] == [m.value for m in second["albedo"]]


def test_an_unreadable_cache_entry_is_refetched(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(astorb.requests, "post", _responder(_body(albedos=[0.09]), calls))
    astorb.lookup("341843")
    (tmp_path / "astorb" / "341843.json").write_text("{ not json")
    assert [m.value for m in astorb.lookup("341843")["albedo"]] == [0.09]
    assert len(calls) == 2


# ---- the reference spin catalogue ----

def _catalog_payload(rows):
    return {"data": {"minorplanet": [
        {"h": h, "surveydata": [{"lightcurve": {"period": p}} for p in periods]}
        for h, periods in rows]}}


def test_one_period_per_body_enters_the_catalogue(monkeypatch):
    # A heavily-studied body measured five times must contribute one sample, not five, or the
    # distribution would describe the population of observations rather than of bodies.
    monkeypatch.setattr(astorb.requests, "post",
                        _responder(_catalog_payload([(20.0, [5.0, 5.1, 5.2]), (18.0, [9.0])])))
    magnitudes, periods = astorb.spin_catalog()
    assert magnitudes == [20.0, 18.0] and periods == [5.0, 9.0]


def test_bodies_without_a_magnitude_are_skipped(monkeypatch):
    monkeypatch.setattr(astorb.requests, "post",
                        _responder(_catalog_payload([(None, [5.0]), (18.0, [9.0])])))
    assert astorb.spin_catalog() == ([18.0], [9.0])


def test_the_catalogue_is_fetched_once_per_process(monkeypatch):
    calls = []
    monkeypatch.setattr(astorb.requests, "post",
                        _responder(_catalog_payload([(20.0, [5.0])]), calls))
    astorb.spin_catalog()
    astorb.spin_catalog()
    assert len(calls) == 1


def test_the_catalogue_survives_a_restart_through_its_cache(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(astorb.requests, "post",
                        _responder(_catalog_payload([(20.0, [5.0])]), calls))
    astorb.spin_catalog()
    astorb._catalog = None                       # as if the process had restarted
    assert astorb.spin_catalog() == ([20.0], [5.0])
    assert len(calls) == 1, "a multi-megabyte download must not repeat every run"
    assert json.loads((tmp_path / "astorb" / "spin_catalog.json").read_text())["period_h"] == [5.0]


def test_an_unreachable_catalogue_is_empty_not_fatal(monkeypatch):
    def _down(*_args, **_kwargs):
        raise ConnectionError("astorb is down")

    monkeypatch.setattr(astorb.requests, "post", _down)
    # Empty leaves the spin distribution unfitted, which leaves the stability margin unknown
    # -- and unknown only ever withholds a tier promotion.
    assert astorb.spin_catalog() == ([], [])
