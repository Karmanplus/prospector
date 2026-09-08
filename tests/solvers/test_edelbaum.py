"""Tests for the Edelbaum low-thrust dV estimate (prospector.solvers.edelbaum).

These pin the calibrated physics so a refactor can't silently change it: the vis-viva scale, the
near-zero cost of Earth's own orbit, and, most importantly, the intercept-strategy "rescue" of
eccentric Earth-crossers that the spiral term alone over-charges. The rescue is what keeps
genuinely reachable targets from being dropped.
"""
import numpy as np
import pandas as pd

from prospector.constants import EARTH_ORBIT_ECC, EARTH_ORBIT_INC_DEG, EARTH_ORBIT_SMA_AU
from prospector.solvers import edelbaum


def test_circular_speed_at_1au():
    assert abs(edelbaum.circular_speed(1.0) - 29.78) < 0.05


def test_earth_to_earth_is_cheap():
    # Earth's own orbit should cost essentially nothing.
    dv = float(edelbaum.lowthrust_dv(EARTH_ORBIT_SMA_AU, EARTH_ORBIT_ECC, EARTH_ORBIT_INC_DEG))
    assert dv < 0.5


def test_intercept_rescues_eccentric_earth_crosser():
    # q=0.8, Q=1.5, i=2 deg  ->  a=1.15, e~=0.3043. The spiral term over-charges this
    # (large |de|), but the intercept strategy meets it cheaply near aphelion.
    a, e, i = 1.15, 0.30434783, 2.0
    dv = float(edelbaum.lowthrust_dv(a, e, i))
    spiral = float(edelbaum.spiral_dv(a, e, i))
    assert dv < 5.0          # intercept rescue keeps it well inside a real budget
    assert dv < spiral       # min() picked the cheaper intercept strategy


def test_vectorized_matches_scalar():
    a = np.array([1.0, 1.15, 2.5])
    e = np.array([0.0167, 0.30434783, 0.1])
    i = np.array([0.0, 2.0, 15.0])
    vec = edelbaum.lowthrust_dv(a, e, i)
    for k in range(len(a)):
        assert np.isclose(vec[k], float(edelbaum.lowthrust_dv(a[k], e[k], i[k])))


def test_evaluate_dataframe_adds_dv_column():
    df = pd.DataFrame({"a": [1.0, 5.0], "e": [0.0167, 0.1], "i": [0.0, 25.0]})
    out = edelbaum.evaluate_dataframe(df)
    assert "lowthrust_dv" in out.columns
    assert "reachable" not in out.columns          # no budget -> no verdict


def test_evaluate_dataframe_flags_reachable_when_budget_given():
    df = pd.DataFrame({"a": [1.0, 5.0], "e": [0.0167, 0.1], "i": [0.0, 25.0]})
    out = edelbaum.evaluate_dataframe(df, dv_budget=7.0)
    assert bool(out["reachable"].iloc[0]) is True    # ~Earth orbit: reachable
    assert bool(out["reachable"].iloc[1]) is False   # far + inclined: not


def test_unparseable_elements_are_unreachable():
    df = pd.DataFrame({"a": ["n/a"], "e": [0.1], "i": [3.0]})
    out = edelbaum.evaluate_dataframe(df, dv_budget=7.0)
    assert not bool(out["reachable"].iloc[0])
    assert np.isnan(out["lowthrust_dv"].iloc[0])
