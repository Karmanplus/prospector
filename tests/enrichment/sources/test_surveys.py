"""Tests for the targeted-survey source (rotation periods and reflectance spectra).

Both surveys publish text formats that have to be parsed by position rather than by schema, so the
parsing is what these pin. The designation handling matters most: survey tables label the same body
several different ways, and a label that fails to reduce to the screen's designation silently costs
that target its period or its only spectrum.
"""
import json

import pytest

from prospector.enrichment.sources import surveys


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(surveys, "cache_dir", lambda: tmp_path / "surveys")
    surveys._ondrejov_index = None
    surveys._manos_index = None
    yield
    surveys._ondrejov_index = None
    surveys._manos_index = None


def _serve(text, calls=None):
    def _get(url, **_kwargs):
        if calls is not None:
            calls.append(url)

        class _Response:
            @staticmethod
            def raise_for_status():
                pass

        _Response.text = text
        _Response.content = text.encode()
        return _Response
    return _get


# ---- Ondrejov rotation periods ----

def _ondrejov_line(designation, period):
    return f"{designation:<29}{period:<15}rest of the row"


ONDREJOV_TABLE = "\n".join([
    "header row",
    _ondrejov_line("(341843) 2008 EV5", "3.725"),
    _ondrejov_line("2016 CF194", "0.187"),
    _ondrejov_line("(433) Eros", "5.270"),
    _ondrejov_line("(341843) 2008 EV5", "9.999"),      # a later refinement of the same body
    _ondrejov_line("nothing parseable here", "n/a"),
])


def test_periods_are_indexed_by_designation(monkeypatch):
    monkeypatch.setattr(surveys.requests, "get", _serve(ONDREJOV_TABLE))
    index = surveys.ondrejov_periods()
    assert index["2008 EV5"] == 3.725
    assert index["2016 CF194"] == 0.187


def test_a_numbered_body_without_a_provisional_designation_keys_on_its_number(monkeypatch):
    monkeypatch.setattr(surveys.requests, "get", _serve(ONDREJOV_TABLE))
    assert surveys.ondrejov_periods()["433"] == 5.270


def test_the_headline_determination_wins_over_later_rows(monkeypatch):
    monkeypatch.setattr(surveys.requests, "get", _serve(ONDREJOV_TABLE))
    assert surveys.ondrejov_periods()["2008 EV5"] == 3.725


def test_unparseable_rows_are_skipped_not_fatal(monkeypatch):
    monkeypatch.setattr(surveys.requests, "get", _serve(ONDREJOV_TABLE))
    assert len(surveys.ondrejov_periods()) == 3


def test_a_period_lookup_returns_a_measurement_list(monkeypatch):
    monkeypatch.setattr(surveys.requests, "get", _serve(ONDREJOV_TABLE))
    found = surveys.ondrejov_period("2008 EV5")
    assert [m.value for m in found] == [3.725] and found[0].source == "ondrejov"
    assert surveys.ondrejov_period("NOT-A-BODY") == []


def test_the_table_is_fetched_once_and_cached(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(surveys.requests, "get", _serve(ONDREJOV_TABLE, calls))
    surveys.ondrejov_periods()
    surveys._ondrejov_index = None                   # as if the process had restarted
    surveys.ondrejov_periods()
    assert len(calls) == 1
    assert (tmp_path / "surveys" / "ondrejov.txt").is_file()


def test_an_unreachable_survey_leaves_the_period_unknown(monkeypatch):
    def _down(*_args, **_kwargs):
        raise ConnectionError("survey is down")

    monkeypatch.setattr(surveys.requests, "get", _down)
    assert surveys.ondrejov_periods() == {}
    assert surveys.ondrejov_period("2008 EV5") == []


# ---- MANOS spectra ----

MANOS_STATUSES = json.dumps([
    {"primary_designation": "2008 EV5",
     "vis_spectrum": {"exists": True, "products": {"data": "https://example/vis.dat"}},
     "nir_spectrum": {"exists": False, "products": None},
     "color_spectrum": {"exists": True, "products": {"data": "https://example/color.dat"}}},
    {"primary_designation": "2016 CF194",
     "vis_spectrum": {"exists": False, "products": None}},
])

VIS_SPECTRUM = "\n".join(["# metadata", "###",
                          "0.45, 0.90, 0.01", "0.60, 1.00, 0.01",
                          "0.70, 0.86, 0.02", "0.80, 1.02, 0.01"])
COLOR_SPECTRUM = "\n".join(["# metadata", "###", "0.45, 0.9, 0.0", "0.60, 1.0, 0.0"])


def _serve_manos(monkeypatch, spectra=None, calls=None):
    spectra = spectra or {"https://example/vis.dat": VIS_SPECTRUM,
                          "https://example/color.dat": COLOR_SPECTRUM}

    def _get(url, **_kwargs):
        if calls is not None:
            calls.append(url)
        body = spectra.get(url, MANOS_STATUSES)

        class _Response:
            @staticmethod
            def raise_for_status():
                pass

        _Response.text = body
        _Response.content = body.encode()
        return _Response

    monkeypatch.setattr(surveys.requests, "get", _get)


def test_the_status_table_is_indexed_case_insensitively(monkeypatch):
    _serve_manos(monkeypatch)
    assert set(surveys.manos_observations()) == {"2008 ev5", "2016 cf194"}


def test_the_spectrum_with_the_most_points_wins(monkeypatch):
    _serve_manos(monkeypatch)
    wave, reflectance, error = surveys.manos_spectrum("2008 EV5")
    # The 4-point visible spectrum, not the 2-point colour-derived one: more points give the
    # classifier's resampling less to interpolate across.
    assert wave == [0.45, 0.60, 0.70, 0.80]
    assert reflectance[2] == 0.86 and error == [0.01, 0.01, 0.02, 0.01]


def test_an_all_zero_error_column_means_not_reported(monkeypatch):
    _serve_manos(monkeypatch, spectra={"https://example/vis.dat": COLOR_SPECTRUM,
                                       "https://example/color.dat": COLOR_SPECTRUM})
    _wave, _reflectance, error = surveys.manos_spectrum("2008 EV5")
    assert error is None, "zeros are 'not measured', not 'measured perfectly'"


def test_a_body_the_survey_never_observed_has_no_spectrum(monkeypatch):
    _serve_manos(monkeypatch)
    assert surveys.manos_spectrum("NOT-A-BODY") is None
    assert surveys.manos_spectrum("2016 CF194") is None


def test_downloaded_spectra_are_cached(monkeypatch):
    calls = []
    _serve_manos(monkeypatch, calls=calls)
    surveys.manos_spectrum("2008 EV5")
    before = len(calls)
    surveys.manos_spectrum("2008 EV5")
    assert len(calls) == before, "spectra are re-read from disk, not re-downloaded"


def test_a_spectrum_with_no_separator_is_skipped(monkeypatch):
    _serve_manos(monkeypatch, spectra={"https://example/vis.dat": "0.45, 0.9, 0.0",
                                       "https://example/color.dat": COLOR_SPECTRUM})
    wave, _reflectance, _error = surveys.manos_spectrum("2008 EV5")
    assert wave == [0.45, 0.60], "falls back to the spectrum that did parse"


def test_a_malformed_status_table_leaves_the_survey_empty(monkeypatch):
    _serve_manos(monkeypatch, spectra={})
    monkeypatch.setattr(surveys.requests, "get", _serve("{ not json"))
    assert surveys.manos_observations() == {}
