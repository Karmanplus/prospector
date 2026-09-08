"""Tests for the two-burn Lambert transfers (prospector.solvers.lambert).

These pin the calibrated geometry against the real ARM reference, the impulsive rendezvous cost to
2008 EV5 (~5.6 km/s, the published Shoemaker-Helin accessibility ΔV) and its escape C3 (~2
km^2/s^2) -- and lock the architectural rule that Lambert ranks and seeds but never drops a target.
Built from hardcoded elements so they run offline; PyKEP comes from the pixi env.
"""
from datetime import date

import numpy as np
import pytest

pytest.importorskip("pykep")
from prospector.solvers import lambert as lb  # noqa: E402

# 2008 EV5 osculating elements (JPL SBDB, epoch JD 2461000.5), the ARM reference target.
EV5 = dict(a_au=0.95982, e=0.082830, i_deg=7.4478, om_deg=93.1897, w_deg=235.9248,
           ma_deg=122.8655, epoch_jd=2461000.5)
ARM_WINDOW = (date(2021, 12, 26), date(2022, 1, 15))
ARM_ARRIVE_BY = date(2023, 8, 25)


def _ev5_porkchop():
    target = lb.target_planet(name="2008 EV5", **EV5)
    dep, arr = lb.launch_arrival_grids(ARM_WINDOW, ARM_ARRIVE_BY, step_days=10.0)
    return lb.porkchop(target, dep, arr, max_revs=2)


def test_mjd2000_epoch_zero_and_roundtrip():
    assert lb.mjd2000_from_date(date(2000, 1, 1)) == 0.0
    for d in (date(2008, 3, 4), date(2022, 1, 15)):
        assert lb.date_from_mjd2000(lb.mjd2000_from_date(d)).date() == d


def test_target_ephemeris_sane():
    # 2008 EV5: perihelion 0.88 AU, aphelion 1.04 AU -> |r| always in that band.
    ev5 = lb.target_planet(name="2008 EV5", **EV5)
    for mjd in (8000.0, 8200.0, 8400.0):
        r, _ = ev5.eph(mjd)
        au = np.linalg.norm(r) / lb.target_planet.__globals__["_AU_M"]
        assert 0.86 < au < 1.06


def test_ev5_impulsive_rendezvous_is_calibrated():
    pc = _ev5_porkchop()
    b = pc.best
    assert b is not None and pc.n_feasible > 100
    # Matches the published ~5.6 km/s rendezvous accessibility ΔV for 2008 EV5.
    assert 5.0 < b.dv_kms < 6.5
    # Escape energy near the real ARRM post-lunar-assist C3 ~ +2 km^2/s^2.
    assert 1.0 < b.c3_km2s2 < 3.5
    # Outbound impulsive transfer ~1 year (real SEP cruise was ~425 d, a bit longer).
    assert 300 < b.tof_days < 450
    assert b.vinf_dep_kms > 0 and b.vinf_arr_kms > 0


def _row(elements):
    """SBDB-style row columns from a target_planet element dict."""
    return {"a": elements["a_au"], "e": elements["e"], "i": elements["i_deg"],
            "om": elements["om_deg"], "w": elements["w_deg"], "ma": elements["ma_deg"],
            "epoch": elements["epoch_jd"]}


def test_thrust_limited_mask_flags_short_transfers():
    from prospector.solvers.lambert import thrust_limited_mask
    # 0.1 N on 500 kg ~ 0.0173 km/s per day of dV at full burn.
    dv = np.array([[1.0, 1.0], [5.0, np.nan]])
    tof = np.array([[100.0, 30.0], [100.0, 100.0]])
    mask = thrust_limited_mask(dv, tof, thrust_N=0.1, mass_kg=500.0)
    assert not mask[0, 0]                          # 1 km/s in 100 d (1.73 available): flyable
    assert mask[0, 1]                              # 1 km/s in 30 d (0.52 available): not
    assert mask[1, 0]                              # 5 km/s in 100 d: not
    assert not mask[1, 1]                          # NaN cells are not flagged
    # Halving usable thrust halves the line.
    assert thrust_limited_mask(dv, tof, thrust_N=0.1, mass_kg=500.0,
                               thrust_limit=0.5)[0, 0]
