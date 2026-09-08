"""Tests for the return-destination library (prospector.launch ReturnDestination).

The homebound mirror of the launch library: the shipped destinations near Earth and the Moon load
and round-trip, their rough capture costs are pinned so a refactor cannot drift them, and the model
rejects a negative delta-v.
"""
import pytest
from pydantic import ValidationError

from prospector.launch import (
    ReturnDestination,
    list_return_destinations,
    load_return_destinations,
    save_return_destination,
)


def test_shipped_return_library_loads():
    cat = load_return_destinations()
    assert {"EML1", "EML2", "SEL2", "LDRO", "LEO"} <= set(cat)
    assert list_return_destinations() == sorted(cat)


def test_pinned_destination_numbers():
    cat = load_return_destinations()
    assert cat["EML2"].arrival_vinf_kms == pytest.approx(0.4)
    assert cat["EML2"].insertion_dv_kms == pytest.approx(0.7)
    assert cat["SEL2"].insertion_dv_kms == pytest.approx(0.3)
    # Propulsive LEO capture (no aerocapture modeled) is the expensive outlier.
    assert cat["LEO"].insertion_dv_kms > cat["EML2"].insertion_dv_kms


def test_negative_dv_rejected():
    with pytest.raises(ValidationError):
        ReturnDestination(name="bad", arrival_vinf_kms=-1.0, insertion_dv_kms=0.5)
    with pytest.raises(ValidationError):
        ReturnDestination(name="bad", arrival_vinf_kms=0.4, insertion_dv_kms=-0.5)


def test_save_round_trips(tmp_path):
    d = ReturnDestination(name="X", arrival_vinf_kms=0.2, insertion_dv_kms=0.6, description="x")
    save_return_destination(d, "X", return_dir=tmp_path)
    cat = load_return_destinations(tmp_path)
    assert cat["X"].insertion_dv_kms == pytest.approx(0.6)
    assert cat["X"].arrival_vinf_kms == pytest.approx(0.2)
