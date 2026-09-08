"""Tests for the expected-minimum-perihelion lookup.

The published grid is over a gigabyte, so it is not in the repository and is not downloaded here.
These tests build a tiny table with the same column layout, which is what the reduction actually
depends on: four element columns, then 26 cumulative-probability columns for the thresholds 0.05 to
1.30 AU.
"""
import numpy as np
import pytest

from prospector.enrichment.models import perihelion


def _row(a, e, i, h, first_threshold_reached):
    """One grid row whose cumulative curve steps to certainty at the given threshold index."""
    cumulative = np.zeros(26)
    cumulative[first_threshold_reached:] = 1.0
    return np.concatenate(([a, e, i, h], cumulative, np.zeros(130)))


def test_a_certain_bin_returns_that_bin_midpoint():
    # A body certain to have reached the 0.2 AU threshold and no closer: the expectation is the
    # midpoint of the 0.15-0.20 bin.
    table = np.array([_row(1.5, 0.4, 5.0, 20.0, first_threshold_reached=3)])
    assert perihelion.expected_min_perihelion_au(1.5, 0.4, 5.0, 20.0, table) == pytest.approx(0.175)


def test_the_closest_grid_cell_is_the_one_used():
    table = np.array([
        _row(1.0, 0.1, 2.0, 18.0, first_threshold_reached=1),    # midpoint 0.075
        _row(2.5, 0.7, 30.0, 24.0, first_threshold_reached=10),  # midpoint 0.525
    ])
    near_first = perihelion.expected_min_perihelion_au(1.05, 0.12, 3.0, 18.2, table)
    near_second = perihelion.expected_min_perihelion_au(2.4, 0.68, 28.0, 23.5, table)
    assert near_first == pytest.approx(0.075)
    assert near_second == pytest.approx(0.525)


def test_the_axes_are_weighted_so_inclination_does_not_dominate():
    """Inclination spans 180 degrees and semimajor axis a few AU. Without normalising by each
    axis's range, a half-AU difference would be invisible next to a few degrees of tilt and
    the wrong cell would be chosen for nearly every body."""
    table = np.array([
        _row(1.0, 0.1, 10.0, 20.0, first_threshold_reached=1),    # right orbit, 5 deg off
        _row(3.0, 0.1, 5.0, 20.0, first_threshold_reached=20),    # exact tilt, 2 AU off
    ])
    # 2 AU of semimajor axis is a much larger fraction of its range than 5 degrees is of
    # inclination's, so the first row must win.
    assert perihelion.expected_min_perihelion_au(1.0, 0.1, 5.0, 20.0, table) == pytest.approx(0.075)


def test_a_body_that_never_came_close_scores_high():
    far = np.array([_row(2.0, 0.2, 10.0, 20.0, first_threshold_reached=25)])
    close = np.array([_row(2.0, 0.2, 10.0, 20.0, first_threshold_reached=0)])
    assert (perihelion.expected_min_perihelion_au(2.0, 0.2, 10.0, 20.0, far)
            > perihelion.expected_min_perihelion_au(2.0, 0.2, 10.0, 20.0, close))


def test_an_unreachable_download_degrades_to_none(tmp_path, monkeypatch):
    """The table is fetched on first use, and that fetch is allowed to fail. Perihelion
    history is one display column; a body still characterizes without it, so the failure must
    come back as None rather than as an exception out of the middle of a screen."""
    calls = []

    def _unreachable(url, **_kwargs):
        calls.append(url)
        raise ConnectionError("no network in tests")

    monkeypatch.setattr(perihelion.requests, "get", _unreachable)
    assert perihelion.load_toliou_table(tmp_path / "absent.dat") is None
    assert calls == [perihelion.TOLIOU_URL]


def test_an_unreadable_table_degrades_to_none(tmp_path):
    broken = tmp_path / "broken.dat"
    broken.write_text("not a table\n")
    assert perihelion.load_toliou_table(broken) is None
