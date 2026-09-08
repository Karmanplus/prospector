"""Seed-diverse parallel multi-start of the outbound cruise (prospector.trades.pipeline).

The heavy Sims-Flanagan solve is stubbed at the single per-start seam
(``pipeline._outbound_start``) so no pygmo optimization runs; the tests pin the selection logic
(best-of), the seed spread (departure x flight-time families incl. the clicked cell), and the
serial path. Units: MJD2000 days.
"""
from types import SimpleNamespace

from prospector import config
from prospector.solvers import lambert as lb
from prospector.trades import pipeline
from prospector.trades.pipeline import multistart as ms


def _sol(feasible, mismatch, final_mass_kg, dv_kms=1.0, tof_days=200.0):
    """A deterministic stand-in for a SimsFlanaganSolution (only the fields the callers read)."""
    return SimpleNamespace(feasible=feasible, mismatch=mismatch, final_mass_kg=final_mass_kg,
                           dv_kms=dv_kms, tof_days=tof_days)


def test_better_sol_prefers_feasible_then_more_mass():
    feas_light = _sol(True, 0.0, 400.0)
    feas_heavy = _sol(True, 0.0, 450.0)
    infeas = _sol(False, 1e-6, 500.0)          # more mass but not matchpoint-closed
    assert pipeline._better_sol(feas_light, None) is True
    assert pipeline._better_sol(feas_light, infeas) is True    # feasible beats infeasible
    assert pipeline._better_sol(infeas, feas_light) is False
    assert pipeline._better_sol(feas_heavy, feas_light) is True  # among feasible, more mass wins
    assert pipeline._better_sol(_sol(False, 0.2, 9), _sol(False, 0.5, 9)) is True  # else lower mismatch


def test_diverse_seed_cells_include_the_click_and_span_flight_time():
    rc = config.default_resolved()
    click = (lb.mjd2000_from_date(rc.departure_window[0]) + 3.0,
             lb.mjd2000_from_date(rc.departure_window[0]) + 3.0 + 160.0)
    cells = pipeline._diverse_seed_cells(rc, click[0], click[1], n=9)
    assert cells[0] == click                       # the clicked cell is always first
    assert len(cells) <= 9 and len(cells) >= 3
    tofs = [arr - dep for dep, arr in cells]
    # A spread of trajectory families, not just the (short) clicked one.
    assert max(tofs) - min(tofs) > 150.0
    arrive = lb.mjd2000_from_date(rc.mission.arrive_by)
    assert all(arr <= arrive + 1e-6 for _dep, arr in cells)   # never past the deadline


def test_diverse_seed_cells_respect_the_flight_time_cap():
    rc = config.default_resolved()
    dep0 = lb.mjd2000_from_date(rc.departure_window[0])
    # A clicked cell WAY past the cap plus the spread must all stay within max_tof.
    cells = pipeline._diverse_seed_cells(rc, dep0 + 3.0, dep0 + 3.0 + 400.0, n=9, max_tof=210.0)
    assert cells
    assert all((arr - dep) <= 210.0 + 1e-6 for dep, arr in cells)


def test_multistart_passes_the_cap_through_not_the_ceiling(monkeypatch):
    rc = config.default_resolved()
    dep0 = lb.mjd2000_from_date(rc.departure_window[0])
    seed = SimpleNamespace(dep_mjd2000=dep0 + 2.0, arr_mjd2000=dep0 + 2.0 + 150.0)
    seen = []

    def fake_start(args):
        _cfg, _row, dep, arr, _revs, kw = args
        seen.append(kw.get("max_tof_days"))
        assert (arr - dep) <= 210.0 + 1e-6            # no seed exceeds the cap
        return _sol(True, 0.0, 400.0, tof_days=arr - dep)

    monkeypatch.setattr(ms, "_outbound_start", fake_start)
    pipeline._multistart_outbound(rc, {"a": 1.1}, seed, sf_kwargs={"max_tof_days": 210.0},
                                  n_starts=9, workers=1, max_revs=2, scaled=lambda *a: None)
    assert seen and all(c == 210.0 for c in seen)     # the cap is forwarded, never overridden


def test_multistart_keeps_the_best_start(monkeypatch):
    rc = config.default_resolved()
    seed = SimpleNamespace(dep_mjd2000=lb.mjd2000_from_date(rc.departure_window[0]) + 2.0,
                           arr_mjd2000=lb.mjd2000_from_date(rc.departure_window[0]) + 2.0 + 150.0)
    calls = []

    def fake_start(args):
        _cfg, _row, dep, arr, _revs, _kw = args
        calls.append((dep, arr))
        # Longer transfers are cheaper here: more final mass -> the best. This is the whole point,
        # a short clicked seed must not win when a longer family beats it.
        return _sol(True, 0.0, 300.0 + (arr - dep))

    monkeypatch.setattr(ms, "_outbound_start", fake_start)
    best = pipeline._multistart_outbound(rc, {"a": 1.1, "e": 0.1, "i": 3.0}, seed,
                                         sf_kwargs={"nseg": 12}, n_starts=9, workers=1,
                                         max_revs=2, scaled=lambda *a: None)
    assert len(calls) >= 3                                   # every diverse seed was run
    best_possible = max(300.0 + (arr - dep) for dep, arr in calls)
    assert best.final_mass_kg == best_possible              # it kept the best start


def test_multistart_falls_back_when_every_start_fails(monkeypatch):
    rc = config.default_resolved()
    seed = SimpleNamespace(dep_mjd2000=lb.mjd2000_from_date(rc.departure_window[0]) + 2.0,
                           arr_mjd2000=lb.mjd2000_from_date(rc.departure_window[0]) + 2.0 + 150.0)
    monkeypatch.setattr(ms, "_outbound_start", lambda args: None)   # all starts raised
    monkeypatch.setattr(ms.lb, "planet_from_row", lambda row: object())
    sentinel = _sol(False, 0.9, 123.0)
    monkeypatch.setattr(ms.sf, "solve_for_config",
                        lambda *a, **k: sentinel)            # single honest clicked-seed solve
    best = pipeline._multistart_outbound(rc, {"a": 1.1}, seed, sf_kwargs={}, n_starts=6,
                                         workers=1, max_revs=2, scaled=lambda *a: None)
    assert best is sentinel



def test_only_the_clicked_start_gets_the_warm_vector(monkeypatch):
    """A grid vector belongs to the cell it was solved at. Every start used to receive it and
    override its own seed, which made nine starts into one; now the other starts keep their
    two-burn guesses."""
    rc = config.default_resolved()
    dep0 = lb.mjd2000_from_date(rc.departure_window[0])
    seed = SimpleNamespace(dep_mjd2000=dep0 + 2.0, arr_mjd2000=dep0 + 2.0 + 150.0)
    got = []

    def fake_start(args):
        _cfg, _row, dep, arr, _revs, kw = args
        got.append((dep, arr, kw.get("x0")))
        return _sol(True, 0.0, 400.0, tof_days=arr - dep)

    monkeypatch.setattr(ms, "_outbound_start", fake_start)
    pipeline._multistart_outbound(rc, {"a": 1.1}, seed, sf_kwargs={"x0": [1.0, 2.0], "nseg": 12},
                                  n_starts=9, workers=1, max_revs=2, scaled=lambda *a: None)
    with_x0 = [(d, a) for d, a, x in got if x is not None]
    assert with_x0 == [(seed.dep_mjd2000, seed.arr_mjd2000)]
    assert len(got) >= 3 and sum(1 for *_r, x in got if x is None) == len(got) - 1
