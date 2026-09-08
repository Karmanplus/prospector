"""Tests for measurement provenance and the rule that chooses between disagreeing sources.

Which value wins decides what the tool reports for every body measured more than once, so the
precedence is pinned here rather than left to whichever source happened to be appended last.
"""
import pytest

from prospector.enrichment.measurements import (
    Measurement,
    consolidate,
    is_missing,
    major_class,
    preferred,
    preferred_float,
    preferred_str,
)


def test_a_flagged_measurement_beats_every_unflagged_one():
    values = [Measurement(0.1, "ssodnet"), Measurement(0.2, "astorb", preferred=True),
              Measurement(0.3, "ondrejov")]
    assert preferred(values) == 0.2


def test_the_last_measurement_wins_when_nothing_is_flagged():
    # Order carries meaning: consolidate() puts the more systematically curated catalogues later,
    # so the fallback reaches for them first.
    values = [Measurement(0.1, "ssodnet"), Measurement(0.2, "astorb")]
    assert preferred(values) == 0.2


def test_an_unusable_flagged_value_falls_through():
    # A source can flag a value as preferred and still have nothing in it; that must not shadow a
    # real measurement from a lower-ranked source.
    values = [Measurement(0.4, "ssodnet"), Measurement(None, "override", preferred=True)]
    assert preferred(values) == 0.4


def test_nothing_measured_is_none_not_a_default():
    assert preferred([]) is None
    assert preferred(None) is None
    assert preferred([Measurement("N/A", "astorb"), Measurement(float("nan"), "ssodnet")]) is None


def test_consolidation_orders_sources_by_rank():
    ssodnet = [Measurement("Ch", "ssodnet")]
    astorb = [Measurement("C", "astorb")]
    # Passed in the other order, the ranking still puts astorb last, so the source order is a
    # property of the data, not of the call site.
    assert [m.source for m in consolidate(astorb, ssodnet)] == ["ssodnet", "astorb"]


def test_an_unranked_source_sorts_last_rather_than_being_dropped():
    known = [Measurement(1.0, "ssodnet")]
    novel = [Measurement(2.0, "some-new-survey")]
    combined = consolidate(known, novel)
    assert [m.source for m in combined] == ["ssodnet", "some-new-survey"]
    assert preferred(combined) == 2.0


def test_consolidation_tolerates_absent_groups():
    assert consolidate(None, [Measurement(1.0, "astorb")], None) == [Measurement(1.0, "astorb")]


def test_float_coercion_rejects_non_numbers():
    assert preferred_float([Measurement("3.5", "astorb")]) == 3.5
    assert preferred_float([Measurement("Cb", "astorb")]) is None
    assert preferred_float([]) is None


def test_string_coercion_rejects_non_strings():
    assert preferred_str([Measurement("  Ch ", "astorb")]) == "Ch"
    assert preferred_str([Measurement(0.3, "astorb")]) is None


@pytest.mark.parametrize("value", [None, float("nan"), "", "  ", "N/A", "none", "null", "-"])
def test_stated_absences_are_treated_as_missing(value):
    assert is_missing(value)


@pytest.mark.parametrize("value", [0, 0.0, "C", "0.05", False])
def test_real_values_are_not_missing(value):
    # Zero and False are values. Treating them as absent would silently discard a measured albedo
    # of zero or a "not tumbling" flag.
    assert not is_missing(value)


@pytest.mark.parametrize("taxonomy,expected", [
    ("C", "C"), ("Cb", "C"), ("  ch ", "C"), ("X", "X"), ("(V)", "V"),
    (None, None), ("", None), ("N/A", None), ("123", None),
])
def test_major_class_reduces_a_taxonomy_to_its_complex(taxonomy, expected):
    assert major_class(taxonomy) == expected
