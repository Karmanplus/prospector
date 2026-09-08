"""Tests for the hydration classifier.

The forward pass is reimplemented here in a few lines of numpy rather than loaded from a
machine-learning library, so the probabilities are pinned against the reference the trained network
produced. Those values come from the training toolkit's own evaluation of the shipped weights on
the shipped validation set, so this checks the reimplementation, not itself.

The spectral half of the module needs an optional package and is not exercised here; what is
checked is that its absence degrades to "unknown" instead of raising.
"""
import numpy as np
import pytest

from prospector.enrichment.models import hydration

# Validation features (albedo, band depth, z0..z3) with the class probabilities the trained network
# assigns them, in HYDRATION_CLASSES order.
_REFERENCE = [
    ([0.0717, -0.0074, -0.464608, -0.574516, 0.0546419, 0.140547],
     {"H3": 0.110180, "H4": 0.037237, "H5": 0.003355, "H12": 0.849228}),
    ([0.248, -0.0421, -0.255875, 0.0511324, -0.245092, 0.0133414],
     {"H3": 0.000001, "H4": 0.003712, "H5": 0.996287, "H12": 0.000000}),
    ([0.037, 0.0237, -0.284779, -0.460335, 0.122859, -0.0207366],
     {"H3": 0.138559, "H4": 0.014111, "H5": 0.006017, "H12": 0.841314}),
]


def test_the_model_ships_with_the_package():
    assert hydration.MODEL_PATH.is_file(), "the trained weights must travel with the code"


def test_the_model_reports_the_expected_classes():
    model = hydration.load_model()
    assert sorted(model.class_names) == sorted(hydration.HYDRATION_CLASSES)


@pytest.mark.parametrize("features,expected", _REFERENCE)
def test_probabilities_match_the_trained_network(features, expected):
    model = hydration.load_model()
    got = dict(zip(model.class_names, model.probabilities(np.array(features))))
    for name, probability in expected.items():
        assert got[name] == pytest.approx(probability, abs=5e-4)


def test_probabilities_sum_to_one():
    model = hydration.load_model()
    for features, _expected in _REFERENCE:
        assert model.probabilities(np.array(features)).sum() == pytest.approx(1.0)


def test_a_missing_model_file_is_reported_as_unavailable(tmp_path):
    # Not an IOError: the caller treats every "no hydration class" cause the same way, and a body
    # must never be dropped because one display column could not be computed.
    with pytest.raises(hydration.HydrationUnavailable, match="not found"):
        hydration.load_model(tmp_path / "missing.npz")


def test_a_corrupt_model_file_is_reported_as_unavailable(tmp_path):
    broken = tmp_path / "broken.npz"
    broken.write_bytes(b"not an npz")
    with pytest.raises(hydration.HydrationUnavailable):
        hydration.load_model(broken)


def test_classification_without_the_optional_package_is_unavailable(monkeypatch):
    real_import = __import__

    def _no_classy(name, *args, **kwargs):
        if name == "classy":
            raise ImportError("classy is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _no_classy)
    with pytest.raises(hydration.HydrationUnavailable, match="classy"):
        hydration.published_spectrum("Bennu")
