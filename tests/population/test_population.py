"""Tests for the SBDB population fetch (prospector.population).

The network call is faked so these run offline and deterministically. They pin the contract the
screen relies on: element columns come back numeric, unparseable or poorly-determined orbits are
dropped, and the disk cache short-circuits a second fetch. The SPK fetch (for the high-fidelity
solver's target ephemeris) is faked the same way.
"""
from datetime import date

import pandas as pd
import pytest

from prospector import population
from prospector.population import sbdb


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    """Stands in for requests.Session, recording how many times it was queried."""

    def __init__(self, payload):
        self._payload = payload
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return _FakeResponse(self._payload)


_PAYLOAD = {
    "signature": {"version": population.SBDB_QUERY_API_VERSION},
    "fields": population.FIELDS,
    "data": [
        # spkid, pdes, full_name, H, condition_code, a, e, i, om, w, ma, epoch
        ["2000433", "433", "433 Eros", "10.4", "0", "1.458", "0.2228", "10.83",
         "304.3", "178.9", "271.1", "2461000.5"],
        ["x", "junk", "bad elements", "20", "0", "n/a", "0.1", "5",
         "10.0", "20.0", "30.0", "2461000.5"],            # unparseable a -> dropped
        ["y", "uncert", "uncertain orbit", "21", "9", "2.0", "0.2", "8",
         "10.0", "20.0", "30.0", "2461000.5"],            # bad condition code -> dropped
    ],
}


def test_clean_keeps_only_good_rows():
    raw = pd.DataFrame(_PAYLOAD["data"], columns=population.FIELDS)
    out = sbdb._clean(raw, condition_code_max=7)
    assert len(out) == 1
    assert out.loc[0, "pdes"] == "433"
    assert out["a"].dtype.kind == "f"   # numeric


def test_fetch_uses_cache_on_second_call(tmp_path):
    session = _FakeSession(_PAYLOAD)
    first = population.fetch_population(h_max=18.0, session=session, cache_dir=tmp_path)
    assert session.calls == 1
    assert len(first) == 1

    # Second call with the same terms reads the parquet cache, no new query.
    second = population.fetch_population(h_max=18.0, session=session, cache_dir=tmp_path)
    assert session.calls == 1
    pd.testing.assert_frame_equal(first, second)


def test_refresh_bypasses_cache(tmp_path):
    session = _FakeSession(_PAYLOAD)
    population.fetch_population(h_max=18.0, session=session, cache_dir=tmp_path)
    population.fetch_population(h_max=18.0, session=session, cache_dir=tmp_path, refresh=True)
    assert session.calls == 2


# ---- SPK kernel fetch ----

class _FakeText:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


# ---------------------------------------------------------------------------
# mission-epoch element refresh (the post-flyby correctness fix)
# ---------------------------------------------------------------------------

_HORIZONS_BLOCK = """$$SOE
2462714.500000000 = A.D. 2030-Aug-01 00:00:00.0000 TDB
 EC= 1.890893E-01 QR= 8.945175E-01 IN= 2.221070E+00
 OM= 2.037311E+02 W = 1.265107E+02 Tp=  2462600.5
 N = 8.508462E-01 MA= 9.700000E+01 TA= 1.100000E+02
 A = 1.103004E+00 AD= 1.311491E+00 PR= 4.231081E+02
$$EOE"""


def test_parse_horizons_elements_block():
    from prospector.population import parse_horizons_elements
    out = parse_horizons_elements(_HORIZONS_BLOCK)
    # The keys overlap (A vs MA/TA/AD, W vs OM): each must land on its own value.
    assert out["a"] == pytest.approx(1.103004)
    assert out["e"] == pytest.approx(0.1890893)
    assert out["i"] == pytest.approx(2.22107)
    assert out["om"] == pytest.approx(203.7311)
    assert out["w"] == pytest.approx(126.5107)
    assert out["ma"] == pytest.approx(97.0)
    assert out["epoch"] == pytest.approx(2462714.5)


def test_parse_horizons_elements_rejects_garbage():
    from prospector.population import parse_horizons_elements
    with pytest.raises(ValueError):
        parse_horizons_elements("No matches found.")


class _DeadSession:
    def get(self, *a, **k):
        raise OSError("no network")


_ROW = {"pdes": "99942", "spkid": 2099942, "a": 0.9224, "e": 0.1912,
        "i": 3.34, "om": 203.9, "w": 126.6, "ma": 50.0, "epoch": 2461000.5}


def test_refresh_raises_rather_than_silently_using_stale_elements(tmp_path):
    """A failed refresh must stop, not continue on elements from another epoch.

    Un-refreshed elements do not give a slightly worse trajectory. Across a planetary close
    approach they give a rendezvous with an orbit the body does not fly, and the only trace is a
    field nobody reads, a wrong answer that looks like a right one.
    """

    from prospector.population import refresh_target_elements

    with pytest.raises(RuntimeError, match="could not refresh the orbit"):
        refresh_target_elements(_ROW, date(2030, 8, 1), cache_dir=tmp_path,
                                session=_DeadSession())
    # The message has to say how to proceed deliberately, not just that it failed.
    try:
        refresh_target_elements(_ROW, date(2030, 8, 1), cache_dir=tmp_path,
                                session=_DeadSession())
    except RuntimeError as exc:
        assert "allow_stale=True" in str(exc)


def test_refresh_degrades_only_when_the_caller_asks(tmp_path):
    """``allow_stale=True`` is the documented escape hatch for a caller that prefers a coarse
    answer to none. It warns, and the row says where its numbers came from."""

    from prospector.population import refresh_target_elements

    with pytest.warns(UserWarning, match="Horizons element refresh failed"):
        out = refresh_target_elements(_ROW, date(2030, 8, 1), cache_dir=tmp_path,
                                      session=_DeadSession(), allow_stale=True)
    assert out["a"] == _ROW["a"] and out["epoch"] == _ROW["epoch"]
    assert out["elements_source"] == "sbdb-cache"


def test_refresh_uses_the_disk_cache(tmp_path):
    # A cached (body, epoch) never touches the network at all.
    import json as _json

    from prospector.population import refresh_target_elements

    epoch = date(2030, 8, 1)
    cache = tmp_path / "horizons_elements" / f"2099942_{epoch.isoformat()}.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(_json.dumps({"a": 1.103, "e": 0.189, "i": 2.221,
                                  "om": 203.7, "w": 126.5, "ma": 97.0,
                                  "epoch": 2462714.5}))

    class _MustNotFetch:
        def get(self, *a, **k):
            raise AssertionError("network touched despite cache")

    row = {"pdes": "99942", "spkid": 2099942, "a": 0.9224, "epoch": 2461000.5}
    out = refresh_target_elements(row, epoch, cache_dir=tmp_path,
                                  session=_MustNotFetch())
    assert out["a"] == pytest.approx(1.103)
    assert out["i"] == pytest.approx(2.221)
    assert out["elements_source"] == "horizons"


# ---------------------------------------------------------------------------
# the main-belt slice
# ---------------------------------------------------------------------------

class _RecordingSession(_FakeSession):
    """A fake session that also keeps the query parameters it was asked with."""

    def __init__(self, payload):
        super().__init__(payload)
        self.params = []

    def get(self, url, params=None, timeout=None):
        self.params.append(dict(params or {}))
        return super().get(url, params=params, timeout=timeout)


def test_the_main_belt_is_asked_for_by_orbit_class(tmp_path):
    """SBDB has no main-belt group, so the slice is the union of the three belt classes; the
    near-Earth slice stays a group query, and the two never share a cache file."""
    session = _RecordingSession(_PAYLOAD)
    population.fetch_population(h_max=12.0, group=population.MAIN_BELT,
                                session=session, cache_dir=tmp_path)
    population.fetch_population(h_max=12.0, group=population.NEAR_EARTH,
                                session=session, cache_dir=tmp_path)
    belt, neo = session.params
    assert belt["sb-class"] == "IMB,MBA,OMB" and "sb-group" not in belt
    assert neo["sb-group"] == "neo" and "sb-class" not in neo
    assert "H|LE|12.0" in belt["sb-cdata"]
    assert len(list(tmp_path.glob("sbdb_mba_*.parquet"))) == 1
    assert len(list(tmp_path.glob("sbdb_neo_*.parquet"))) == 1


def test_load_population_merges_the_bright_main_belt(monkeypatch):
    from prospector.population import catalog

    calls = []

    def fake_fetch(h_max, group, **kwargs):
        calls.append((group, h_max))
        if group == population.MAIN_BELT:
            return pd.DataFrame([["2000016", "16", "16 Psyche", 6.2, 0, 2.92, 0.13, 3.1,
                                  150.0, 229.0, 300.0, 2461000.5]], columns=population.FIELDS)
        return pd.DataFrame([["2099942", "99942", "99942 Apophis", 19.7, 0, 0.92, 0.19, 3.3,
                              204.0, 126.0, 215.0, 2461000.5]], columns=population.FIELDS)

    monkeypatch.setattr(catalog, "fetch_population", fake_fetch)
    df = catalog.load_population(h_max=25.0)
    assert calls == [(population.NEAR_EARTH, 25.0), (population.MAIN_BELT, catalog.MAIN_BELT_H_MAX)]
    names = df["pdes"].astype(str).tolist()
    assert names[:2] == ["99942", "16"], "near-Earth first, then the belt, then the planets"
    assert "Venus" in names
    assert (df.loc[df["pdes"] == "16", population.BODY_CLASS_COL] == population.ASTEROID).all()
    # Opting out drops the belt, and the slider's cutoff never reaches it.
    calls.clear()
    small = catalog.load_population(h_max=25.0, main_belt_h_max=None, include_planets=False)
    assert calls == [(population.NEAR_EARTH, 25.0)] and small["pdes"].tolist() == ["99942"]


def test_a_failed_main_belt_fetch_warns_and_keeps_the_screen_up(monkeypatch):
    """Offline with the near-Earth slice cached: the belt is left out with a warning, rather than
    the whole population load failing over the part that is almost never reachable anyway."""
    import requests

    from prospector.population import catalog

    def fake_fetch(h_max, group, **kwargs):
        if group == population.MAIN_BELT:
            raise requests.ConnectionError("offline")
        return pd.DataFrame([["2099942", "99942", "99942 Apophis", 19.7, 0, 0.92, 0.19, 3.3,
                              204.0, 126.0, 215.0, 2461000.5]], columns=population.FIELDS)

    monkeypatch.setattr(catalog, "fetch_population", fake_fetch)
    with pytest.warns(UserWarning, match="main-belt catalog unavailable"):
        df = catalog.load_population(h_max=25.0)
    assert df["pdes"].astype(str).tolist()[0] == "99942" and "Mars" in df["pdes"].tolist()
