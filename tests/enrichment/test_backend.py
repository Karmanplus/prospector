"""Tests for the assembly of one characterized target.

``characterize`` is where the sources meet the models, and the failure modes it has to survive are
the ones that would otherwise cost the screen a reachable target: a catalogue that returns nothing,
a body with no rotation period, a body with no size at all. Every source is stubbed here so the
assembly is what is under test, not the network.

The stubs replace attributes on the source *modules*, because that is what ``backend`` calls.
Patching a name re-exported elsewhere would leave the real lookups running.
"""
import pytest

from prospector.enrichment import backend
from prospector.enrichment.measurements import Measurement
from prospector.enrichment.sources import ssodnet


@pytest.fixture
def sources(monkeypatch):
    """Every source stubbed, returning whatever the test puts in the returned dict."""
    state = {
        "body": ssodnet.Body(input_id="341843", name="(341843) 2008 EV5",
                             semi_major_axis_au=0.958, eccentricity=0.083,
                             inclination_deg=7.44),
        "astorb": {name: [] for name in ("albedo", "taxonomy", "period_h", "tumbling",
                                         "lightcurve_quality")},
        "ondrejov": [],
        "overrides": {},
        "tier_override": None,
    }

    monkeypatch.setattr(backend.ssodnet, "lookup", lambda _id: state["body"])
    monkeypatch.setattr(backend.astorb, "lookup", lambda _id: state["astorb"])
    monkeypatch.setattr(backend.surveys, "ondrejov_period", lambda _id: state["ondrejov"])
    monkeypatch.setattr(backend.overrides, "for_target", lambda _id: state["overrides"])
    monkeypatch.setattr(backend.overrides, "tier_override", lambda _id: state["tier_override"])
    monkeypatch.setattr(backend, "_hydration_class", lambda *a, **k: None)
    return state


def _characterize(**kwargs):
    return backend.characterize("341843", spin_catalog=([], []), **kwargs)


def test_an_unresolvable_body_is_skipped_not_faked(sources):
    sources["body"] = ssodnet.Body(input_id="NOT-A-BODY", resolved=False)
    assert _characterize() is None


def test_the_join_key_is_the_identifier_that_was_asked_for(sources):
    """The canonical name deliberately differs from the input for named bodies, so joining on
    it would silently lose every target whose name resolved."""
    row = _characterize()
    assert row["input_id"] == "341843"
    assert row["name"] == "(341843) 2008 EV5"


def test_a_body_with_nothing_measured_still_produces_a_row(sources):
    row = _characterize()
    assert row["tier"] == "C"                       # uncharacterized, not rejected
    assert row["diameter_m"] is None and row["cohesion_pa"] is None


def test_measurements_from_every_source_reach_the_row(sources):
    sources["body"].abs_mag = [Measurement(20.0, "ssodnet", preferred=True)]
    sources["astorb"]["albedo"] = [Measurement(0.09, "astorb")]
    sources["astorb"]["taxonomy"] = [Measurement("C", "astorb")]
    sources["ondrejov"] = [Measurement(3.7, "ondrejov")]

    row = _characterize()
    assert row["albedo"] == 0.09 and row["taxonomy"] == "C" and row["period_h"] == 3.7
    assert row["diameter_m"] == pytest.approx(430.0, rel=0.05)


def test_an_override_outranks_the_catalogues(sources):
    sources["astorb"]["albedo"] = [Measurement(0.09, "astorb")]
    sources["overrides"] = {"albedo": [Measurement(0.30, "override", preferred=True)]}
    assert _characterize()["albedo"] == 0.30


def test_a_tier_override_replaces_the_computed_tier(sources):
    sources["tier_override"] = "S"
    assert _characterize()["tier"] == "S"


def test_cohesion_needs_both_a_size_and_a_spin(sources):
    sources["body"].abs_mag = [Measurement(20.0, "ssodnet", preferred=True)]
    # Size but no period: the structural question cannot be asked from this body's own spin.
    assert _characterize()["cohesion_pa"] is None

    sources["ondrejov"] = [Measurement(3.7, "ondrejov")]
    assert _characterize()["cohesion_pa"] is not None


def test_a_body_with_no_size_gets_no_structural_verdict(sources):
    # Without an absolute magnitude there is no diameter, and without a diameter neither structural
    # measure means anything. Both stay unknown rather than defaulting.
    sources["ondrejov"] = [Measurement(3.7, "ondrejov")]
    row = _characterize()
    assert row["diameter_m"] is None
    assert row["cohesion_pa"] is None and row["p_gt_pcrit"] is None


def test_perihelion_history_needs_the_published_table(sources):
    sources["body"].abs_mag = [Measurement(20.0, "ssodnet", preferred=True)]
    # The table is a gigabyte-scale optional download; without it the column is simply absent.
    assert _characterize(toliou_table=None)["q_min_au"] is None


def test_perihelion_history_needs_a_full_set_of_elements(sources):
    import numpy as np

    table = np.zeros((1, 30))
    table[0, :4] = [0.958, 0.083, 7.44, 20.0]
    sources["body"].abs_mag = [Measurement(20.0, "ssodnet", preferred=True)]
    sources["body"].eccentricity = None             # SsODNet had no orbit for this body
    assert _characterize(toliou_table=table)["q_min_au"] is None


def test_a_tumbling_flag_reaches_the_tier(sources):
    sources["body"].abs_mag = [Measurement(20.0, "ssodnet", preferred=True)]
    sources["astorb"]["taxonomy"] = [Measurement("C", "astorb")]
    sources["ondrejov"] = [Measurement(2.0, "ondrejov")]
    settled = _characterize()["tier"]

    sources["astorb"]["tumbling"] = [Measurement(True, "astorb")]
    assert backend.tiers.demote(settled) == _characterize()["tier"]


def test_every_contract_column_is_present(sources):
    from prospector.enrichment.schema import ENRICH_COLUMNS

    # normalize() would fill a missing column with NA, which would hide a property the backend
    # quietly stopped emitting. Checking here means the gap surfaces as a test failure.
    assert set(_characterize()) == set(ENRICH_COLUMNS)
