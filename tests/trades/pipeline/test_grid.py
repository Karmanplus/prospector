"""The converged low-thrust date grid (prospector.trades.pipeline.grid).

The Sims-Flanagan solve is stubbed at the single per-cell seam (``grid._cell``) so no pygmo
optimization runs; these pin how the sweep behaves, the axis order, what may start from what, and
what a failed cell is allowed to cost. The physics has its own tests in
``tests/solvers/test_simsflanagan.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from prospector import config
from prospector.solvers import lambert as lb
from prospector.trades.pipeline import grid as G


@pytest.fixture(autouse=True)
def _no_horizons(monkeypatch):
    """Never re-osculate over the network in a test. Patched on ``grid`` itself, not on the
    population facade: a function resolves names in its own module, so patching the facade would
    leave the grid calling the real thing."""
    monkeypatch.setattr(G, "_mission_elements", lambda rc, row: dict(row))


def _stub(monkeypatch, closes):
    """Replace the per-cell solve with a recorder. ``closes(dep, tof) -> bool`` decides which cells
    converge; every call's warm-start vector is recorded so the chain can be inspected."""
    seen = []

    def fake(args):
        _cfg, _row, dep, tof, x0, _kw = args
        seen.append({"dep": float(dep), "tof": float(tof), "x0": x0})
        if not closes(dep, tof):
            return None
        # A vector whose contents identify the cell that produced it, so inheritance is traceable.
        return {"dep_mjd2000": float(dep), "tof_days": float(tof),
                "dv_kms": 3.0 + 0.001 * dep + 0.002 * tof,
                "final_mass_kg": 500.0, "feasible": True, "mismatch": 1e-6,
                "decision_vector": [float(dep), float(tof)]}

    monkeypatch.setattr(G, "_cell", fake)
    return seen


def test_the_march_runs_from_the_longest_flight_time_down():
    """Long transfers are the easy ones -- propellant margin to spare -- and short ones are where
    the feasibility wall is. Marching toward the wall means every hard cell inherits a solved
    neighbour; marching away from it would seed the easy cells from nothing."""
    rc = config.default_resolved()
    dep, tof = G.grid_axes(rc, n_dep=4, n_tof=6)
    assert tof[0] > tof[-1], "the flight-time axis must be handed back longest first"
    assert np.all(np.diff(tof) < 0)
    # Departures span the window a day inside each edge, see
    # ``test_departures_sit_on_whole_days_inside_the_launch_window`` for why, and the flight-time
    # ceiling is measured from the FIRST departure: the earliest liftoff flies the longest trip.
    w0 = lb.mjd2000_from_date(rc.departure_window[0])
    w1 = lb.mjd2000_from_date(rc.departure_window[1])
    assert w0 < dep[0] <= w0 + 2.0
    assert w1 - 2.0 <= dep[-1] < w1
    arrive = lb.mjd2000_from_date(rc.mission.arrive_by)
    assert dep[0] + tof[0] <= arrive + 1e-6
    assert dep[-1] + tof[0] > arrive, "later departures have cells past the deadline, marked, not solved"


def test_each_departure_chains_along_flight_time_not_across_departures(monkeypatch):
    """The chain runs along one departure's own column. Seeding across departures instead would
    couple the columns, so a single bad departure could poison the whole grid rather than one
    column -- and the columns are what makes the departure axis independent evidence."""
    rc = config.default_resolved()
    seen = _stub(monkeypatch, lambda d, t: True)
    out = G.lowthrust_grid(rc, {"pdes": "X", "full_name": "X"}, n_dep=3, n_tof=3, workers=1)

    dep, tof = G.grid_axes(rc, n_dep=3, n_tof=3)
    blocked = np.asarray(out["past_deadline"], bool)
    for i in range(3):
        # A departure's first flyable column (the longest it can fly to the deadline) starts cold;
        # each later one inherits from the SAME departure's previous flyable column. The stub's
        # vector is [dep, tof] of whatever produced it, so inheritance is traceable.
        columns = [j for j in range(3) if not blocked[i, j]]
        assert columns, "every departure can fly at least the shortest trip"
        for k, j in enumerate(columns):
            cell = next(c for c in seen
                        if c["tof"] == pytest.approx(tof[j]) and c["dep"] == pytest.approx(dep[i]))
            if k == 0:
                assert cell["x0"] is None, (i, j)
            else:
                assert cell["x0"] is not None
                assert cell["x0"][0] == pytest.approx(dep[i]), (i, j)
                assert cell["x0"][1] == pytest.approx(tof[columns[k - 1]]), (i, j)


def test_a_failed_cell_costs_one_cell_and_never_seeds_its_successors(monkeypatch):
    """A vector that did not close is not a solution, so propagating it would hand the rest of the
    column a starting point that satisfies nothing. The chain carries the last CONVERGED vector
    instead, which is what makes one bad cell cost one cell rather than cascading down a column.
    """
    rc = config.default_resolved()
    dep, tof = G.grid_axes(rc, n_dep=2, n_tof=4)
    # Fail the whole middle column.
    seen = _stub(monkeypatch, lambda d, t: t != pytest.approx(tof[1]))
    out = G.lowthrust_grid(rc, {"pdes": "X", "full_name": "X"}, n_dep=2, n_tof=4, workers=1)

    ok = np.asarray(out["feasible"], bool)
    blocked = np.asarray(out["past_deadline"], bool)
    assert not ok[:, 1].any(), "the middle column was supposed to fail"
    assert ok[:, 2].all() and ok[:, 3].all(), "the failure cascaded"
    assert ok[0, 0], "the first departure flies the longest trip"

    # The column after the failure inherits the last converged one, skipping the hole. Only the
    # first departure can fly the longest trip; a later one whose longest is past the deadline has
    # nothing converged before the hole and starts the column after it cold.
    for i in range(2):
        after = next(c for c in seen
                     if c["tof"] == pytest.approx(tof[2]) and c["dep"] == pytest.approx(dep[i]))
        if blocked[i, 0]:
            assert after["x0"] is None
        else:
            assert after["x0"] is not None
            assert after["x0"][1] == pytest.approx(tof[0]), "a failed vector was propagated"


def test_a_grid_that_solves_nothing_still_returns_a_readable_surface(monkeypatch):
    """Nothing converging is a possible outcome, not an error. The surface must come back full of
    NaN with no cell marked feasible, so a caller renders "not found" rather than raising."""
    rc = config.default_resolved()
    _stub(monkeypatch, lambda d, t: False)
    out = G.lowthrust_grid(rc, {"pdes": "X", "full_name": "X"}, n_dep=3, n_tof=3, workers=1)
    assert out["n_feasible"] == 0 and out["n_cells"] == 9
    assert not np.asarray(out["feasible"], bool).any()
    assert np.isnan(np.asarray(out["dv_kms"], float)).all()
    assert all(v is None for col in out["decision_vectors"] for v in col)
    # And the curve reports the gaps as gaps rather than dropping them.
    fr = G.cheapest_per_flight_time(out)
    assert len(fr) == 3 and all(p["dv_kms"] is None for p in fr)


def test_the_frontier_is_the_column_minimum_and_the_spread_is_what_it_hides(monkeypatch):
    """The cheapest cell in each column is the flight-time trade -- which is why the grid replaces a
    separate flight-time trade. The spread across departures within a column is the thing that
    frontier cannot show, and on some targets it is the largest gradient in the trade space.
    """
    rc = config.default_resolved()
    _stub(monkeypatch, lambda d, t: True)
    out = G.lowthrust_grid(rc, {"pdes": "X", "full_name": "X"}, n_dep=4, n_tof=3, workers=1)

    dv = np.asarray(out["dv_kms"], float)          # NaN where a departure cannot reach the deadline
    fr = G.cheapest_per_flight_time(out)
    assert [p["tof_days"] for p in fr] == sorted(p["tof_days"] for p in fr), "frontier sorts by tof"
    for p in fr:
        j = out["tof_days"].index(p["tof_days"])
        assert p["dv_kms"] == pytest.approx(np.nanmin(dv[:, j]))
        assert p["dep_mjd2000"] == pytest.approx(out["dep_mjd2000"][int(np.nanargmin(dv[:, j]))])

    for p in G.departure_spread(out):
        j = out["tof_days"].index(p["tof_days"])
        col = dv[:, j]
        if p["n"] < 2:                     # one flyable departure: no spread to speak of
            assert p["lo_kms"] is None and p["hi_kms"] is None
            continue
        assert p["lo_kms"] == pytest.approx(np.nanmin(col))
        assert p["hi_kms"] == pytest.approx(np.nanmax(col))
        assert p["spread_pct"] == pytest.approx(100.0 * (np.nanmax(col) / np.nanmin(col) - 1.0))


def test_the_whole_grid_shares_one_segment_count(monkeypatch):
    """A warm start may not be resampled between segment counts, so the grid and anything seeded
    from it must agree on nseg. Passing it per cell is how that would drift."""
    rc = config.default_resolved()
    seen = []

    def fake(args):
        seen.append(args[5].get("nseg"))
        return None

    monkeypatch.setattr(G, "_cell", fake)
    G.lowthrust_grid(rc, {"pdes": "X", "full_name": "X"}, n_dep=3, n_tof=3, nseg=10, workers=1)
    assert set(seen) == {10}, seen


def test_a_deadline_too_tight_for_any_transfer_yields_an_empty_grid_not_an_error():
    """An unreachable target is a normal answer for a screening tool. Returning one flight time at
    the ceiling lets the caller draw an empty grid; raising would take the workspace down."""
    rc = config.default_resolved()
    tight = rc.model_copy(update={"mission": rc.mission.model_copy(
        update={"arrive_by": rc.departure_window[1]})})
    dep, tof = G.grid_axes(tight, n_dep=4, n_tof=6, min_tof_days=150.0)
    assert dep.size == 4 and tof.size == 1


def _polish_stub(monkeypatch, grid, better_by):
    """Replace the free-departure polish with a recorder.

    ``better_by(tof) -> float`` is how much cheaper the polish comes back than the cell it started
    from; None means it did not converge. The baseline is read off ``grid``'s own column minimum,
    so the gain is the only thing that varies, a stub with an independent baseline would make the
    polish look better or worse for reasons the test is not about.
    """
    base = {round(p["tof_days"], 3): p["dv_kms"] for p in G.cheapest_per_flight_time(grid)}
    seen = []

    def fake(args):
        _cfg, _row, tof, x0, _kw = args
        seen.append({"tof": float(tof), "x0": x0})
        gain = better_by(float(tof))
        if gain is None:
            return None
        return {"tof_days": float(tof), "solved_tof_days": float(tof) + 0.61,
                "dep_mjd2000": 12345.0, "dv_kms": base[round(float(tof), 3)] - gain,
                "final_mass_kg": 501.0, "feasible": True, "mismatch": 1e-6,
                "decision_vector": [0.0]}

    monkeypatch.setattr(G, "_polish", fake)
    return seen


def test_a_polished_point_is_keyed_by_the_flight_time_that_was_asked_for(monkeypatch):
    """The polish pins flight time to a tolerance, so the SOLVED duration drifts a little off the
    requested one. Keying the curve on the solved value matches nothing, and the merge then
    silently discarded every polished point while still returning a full curve, the improvement
    looked like it simply never helped. The key is the request; the solved value rides alongside.
    """
    rc = config.default_resolved()
    _stub(monkeypatch, lambda d, t: True)
    out = G.lowthrust_grid(rc, {"pdes": "X", "full_name": "X"}, n_dep=3, n_tof=3, workers=1)

    _polish_stub(monkeypatch, out, lambda tof: 0.05)
    pol = G.polish_best_per_flight_time(rc, {"pdes": "X", "full_name": "X"}, out, workers=1)
    assert pol, "the polish returned nothing to merge"
    # The requested flight times are the grid's own axis values...
    assert {round(p["tof_days"], 3) for p in pol} == {round(t, 3) for t in out["tof_days"]}
    # ...and every one of them actually reaches the merged frontier.
    merged = G.best_per_flight_time(out, pol)
    assert all(p["source"] == "polish" for p in merged), merged
    # The solved duration is kept, but is not the key.
    assert all(p["solved_tof_days"] != p["tof_days"] for p in pol)


def test_the_merge_only_ever_moves_the_frontier_down(monkeypatch):
    """A refinement that could make the displayed curve worse would be a strange thing to run by
    default, so the merge takes whichever of the cell and its polish is better, and ignores a
    polish that failed to converge."""
    rc = config.default_resolved()
    _stub(monkeypatch, lambda d, t: True)
    out = G.lowthrust_grid(rc, {"pdes": "X", "full_name": "X"}, n_dep=3, n_tof=3, workers=1)
    grid_curve = {round(p["tof_days"], 3): p["dv_kms"] for p in G.cheapest_per_flight_time(out)}
    tofs = sorted(grid_curve)

    # One column improves, one is worse, one fails to converge at all.
    def gain(tof):
        if round(tof, 3) == tofs[0]:
            return 0.10
        if round(tof, 3) == tofs[1]:
            return -0.50            # polish came back WORSE
        return None                 # polish did not converge

    _polish_stub(monkeypatch, out, gain)
    pol = G.polish_best_per_flight_time(rc, {"pdes": "X", "full_name": "X"}, out, workers=1)
    merged = {round(p["tof_days"], 3): p for p in G.best_per_flight_time(out, pol)}

    assert merged[tofs[0]]["source"] == "polish"
    assert merged[tofs[0]]["dv_kms"] < grid_curve[tofs[0]]
    for t in tofs[1:]:
        assert merged[t]["source"] == "grid", t
        assert merged[t]["dv_kms"] == pytest.approx(grid_curve[t]), t


def test_the_polish_starts_from_the_grid_and_skips_columns_with_no_cell(monkeypatch):
    """It is a refinement of the grid, not an independent instrument: it warm-starts from the
    column's best cell. A column where nothing converged has no cell to start from, and a cold
    solve there would be a different thing reporting into the same curve."""
    rc = config.default_resolved()
    dep, tof = G.grid_axes(rc, n_dep=2, n_tof=3)
    _stub(monkeypatch, lambda d, t: t != pytest.approx(tof[1]))     # middle column all fails
    out = G.lowthrust_grid(rc, {"pdes": "X", "full_name": "X"}, n_dep=2, n_tof=3, workers=1)

    seen = _polish_stub(monkeypatch, out, lambda tof: 0.01)
    G.polish_best_per_flight_time(rc, {"pdes": "X", "full_name": "X"}, out, workers=1)
    asked = {round(c["tof"], 3) for c in seen}
    assert round(float(tof[1]), 3) not in asked, "polished a column with no converged cell"
    assert asked == {round(float(tof[0]), 3), round(float(tof[2]), 3)}
    assert all(c["x0"] is not None for c in seen), "the polish must start from the grid's cell"


def test_the_grid_runs_under_the_solver_terms_it_is_given(monkeypatch):
    """The grid's cells must run under the same terms the cruise does. Left to the solver's own
    defaults it solved every cell at a 1 km/s departure excess regardless of the config's knob, so
    the surface described a different mission than the trajectory flown from it, and a vector
    stored under those bounds could not be rebuilt under the config's at all.

    The terms are echoed back so a rebuild uses what the cell was solved under, not what the rail
    happens to say later.
    """
    rc = config.default_resolved()
    seen = []

    def fake(args):
        seen.append(args[5])
        return None

    monkeypatch.setattr(G, "_cell", fake)
    out = G.lowthrust_grid(rc, {"pdes": "X", "full_name": "X"}, n_dep=2, n_tof=2, workers=1,
                           vinf_dep_kms=0.25, vinf_arr_kms=0.05, max_duty_cycle=0.8)
    assert seen, "no cells were attempted"
    for kw in seen:
        assert kw["vinf_dep_kms"] == pytest.approx(0.25)
        assert kw["vinf_arr_kms"] == pytest.approx(0.05)
        assert kw["max_duty_cycle"] == pytest.approx(0.8)
    echoed = out["sf_kwargs"]
    assert echoed["vinf_dep_kms"] == pytest.approx(0.25)
    assert echoed["max_duty_cycle"] == pytest.approx(0.8)
    # How many restarts the grid used is its own business, not a term a rebuild reproduces.
    assert "restarts" not in echoed


def test_the_polish_runs_under_the_same_terms_the_cells_did(monkeypatch):
    """The polish frees the departure date and nothing else.

    Reading only the segment count off the grid left the rest to the solver's defaults, which meant
    a 1 km/s departure excess the cells never had. The free speed then came back as an improvement:
    on Apophis the curve read up to 1.19 km/s under its own cells, against the 0.05 a real search
    wins, and the cheapest point landed mid-curve on a number no cell held.
    """
    rc = config.default_resolved()
    _stub(monkeypatch, lambda dep, tof: True)
    out = G.lowthrust_grid(rc, {"pdes": "X", "full_name": "X"}, n_dep=2, n_tof=2, workers=1,
                           vinf_dep_kms=0.001, vinf_arr_kms=0.05, max_duty_cycle=0.8)

    seen = []

    def fake(args):
        seen.append(args[4])
        return None

    monkeypatch.setattr(G, "_polish", fake)
    G.polish_best_per_flight_time(rc, {"pdes": "X", "full_name": "X"}, out, workers=1)
    assert seen, "the polish attempted nothing"
    for kw in seen:
        assert kw["vinf_dep_kms"] == pytest.approx(0.001)
        assert kw["vinf_arr_kms"] == pytest.approx(0.05)
        assert kw["max_duty_cycle"] == pytest.approx(0.8)
        assert int(kw["nseg"]) == int(out["nseg"])


def test_a_bad_call_raises_instead_of_reading_as_an_unreachable_target(monkeypatch):
    """A TypeError from the cell solve is a wrong call, not a hard cell. Every cell fails the same
    way, so swallowing it reports a target as unreachable when the code is simply wrong, which is
    what hid a duplicate-keyword bug that emptied whole grids while the surface still
    rendered as "nothing converged"."""
    rc = config.default_resolved()

    def boom(args):
        raise TypeError("got multiple values for keyword argument 'window_slack_days'")

    monkeypatch.setattr(G, "_cell", boom)
    with pytest.raises(TypeError, match="multiple values"):
        G.lowthrust_grid(rc, {"pdes": "X", "full_name": "X"}, n_dep=2, n_tof=2, workers=1)


def test_departures_sit_on_whole_days_inside_the_launch_window():
    """Each cell pins its departure to a narrow window around its own epoch, and that pin has to be
    expressible as a date plus a slack; the only form the solver's bounds take. A fractional epoch
    cannot be, and rounding it widened the slack enough to push the bounds back outside the launch
    window, which produced cells departing before the pad was available.

    So the axis sits on whole days, a day inside each edge, leaving room for the pin itself.
    """
    rc = config.default_resolved()
    dep, _tof = G.grid_axes(rc, n_dep=6, n_tof=4)
    w0 = lb.mjd2000_from_date(rc.departure_window[0])
    w1 = lb.mjd2000_from_date(rc.departure_window[1])
    assert np.all(dep == np.round(dep)), dep
    assert dep.min() >= w0 + 1.0 - 1e-9, (dep.min(), w0)
    assert dep.max() <= w1 - 1.0 + 1e-9, (dep.max(), w1)
    # A pin of the default width around any of them stays inside the window.
    assert dep.min() - 0.75 >= w0 and dep.max() + 0.75 <= w1

    # A window too short for that collapses to a single mid-window day rather than raising.
    tight = rc.model_copy(update={"mission": rc.mission.model_copy(
        update={"launch_window": (rc.mission.launch_window[0], rc.mission.launch_window[0])})})
    dep2, _ = G.grid_axes(tight, n_dep=6, n_tof=4)
    assert dep2.size == 1 and dep2[0] == np.round(dep2[0])


def test_cells_that_cannot_reach_the_deadline_are_not_solved_and_read_as_such(monkeypatch):
    """The flight-time axis reaches to what the EARLIEST departure can fly, so later departures
    have cells that would arrive after the deadline. Those are never sent to the solver, they are
    marked, and the count of cells still adds up."""
    rc = config.default_resolved()
    seen = _stub(monkeypatch, closes=lambda dep, tof: True)
    out = G.lowthrust_grid(rc, {"pdes": "X", "full_name": "X"}, n_dep=4, n_tof=4, workers=1)
    dep, tof = np.asarray(out["dep_mjd2000"]), np.asarray(out["tof_days"])
    arrive = lb.mjd2000_from_date(rc.mission.arrive_by)
    blocked = np.asarray(out["past_deadline"], bool)
    # The longest flight time is the earliest departure's, so the latest departure cannot fly it.
    assert dep[0] + tof[0] <= arrive + 1e-6
    assert blocked[-1, 0], "the last departure at the longest flight time is past the deadline"
    assert not blocked[0].any(), "the first departure can fly every flight time on the axis"
    # Blocked cells were never solved; every other cell was.
    solved = {(s["dep"], s["tof"]) for s in seen}
    for i in range(dep.size):
        for j in range(tof.size):
            assert ((float(dep[i]), float(tof[j])) in solved) == (not blocked[i, j])
    assert not np.asarray(out["feasible"], bool)[blocked].any()
    assert out["n_cells"] == dep.size * tof.size


def test_the_frontier_prefers_cells_whose_ceilings_settled(monkeypatch):
    """An unsettled cell still fires where the array cannot power it and reads cheaper than it
    is. It stands in on the frontier only where nothing in its column settled."""
    rc = config.default_resolved()
    seen = []

    def fake(args):
        _cfg, _row, dep, tof, x0, _kw = args
        seen.append((dep, tof))
        return {"dep_mjd2000": float(dep), "tof_days": float(tof),
                "dv_kms": 3.0 + 0.001 * dep + 0.002 * tof, "final_mass_kg": 500.0,
                "feasible": True, "mismatch": 1e-6, "settled": False,
                "decision_vector": [float(dep), float(tof)]}

    monkeypatch.setattr(G, "_cell", fake)
    out = G.lowthrust_grid(rc, {"pdes": "X", "full_name": "X"}, n_dep=3, n_tof=2, workers=1)
    settled = np.asarray(out["settled"], bool)
    assert not settled[np.asarray(out["feasible"], bool)].any()
    # Nothing settled anywhere: the frontier falls back to the plain column minimum.
    fr = G.cheapest_per_flight_time(out)
    assert all(p["dv_kms"] is not None for p in fr)
    # Mark one dearer cell per column settled and the frontier moves to it.
    dv = np.asarray(out["dv_kms"], float)
    for j in range(dv.shape[1]):
        i_dear = int(np.nanargmax(dv[:, j]))
        out["settled"][i_dear][j] = True
    fr = G.cheapest_per_flight_time(out)
    for p in fr:
        j = out["tof_days"].index(p["tof_days"])
        assert p["dv_kms"] == pytest.approx(np.nanmax(dv[:, j]))


def test_each_cell_keeps_the_thrust_terms_it_was_solved_under(monkeypatch):
    """A vector alone no longer fixes the trajectory: the cruise follows the Sun, so a cell settles
    on per-segment thrust ceilings and Isps of its own path. They are stored beside the vector,
    cell by cell, so a click rebuilds the cell's own trajectory rather than a different one from
    the same vector (measured, that read as not converged while the grid showed the cell closing).
    A stub that records no terms leaves None, never a KeyError."""
    rc = config.default_resolved()

    def fake(args):
        _cfg, _row, dep, tof, _x0, _kw = args
        out = {"dep_mjd2000": float(dep), "tof_days": float(tof), "dv_kms": 3.0,
               "final_mass_kg": 500.0, "feasible": True, "mismatch": 1e-6,
               "decision_vector": [float(dep), float(tof)]}
        if tof > 200.0:            # the shorter cells stand for an older stub with no terms
            out["leg_terms"] = {"seg_caps": [0.9, 0.7], "thrust_N": 0.232, "isp_s": 1751.0,
                                "seg_isp_s": [1783.0, 1712.0]}
        return out

    monkeypatch.setattr(G, "_cell", fake)
    out = G.lowthrust_grid(rc, {"pdes": "X", "full_name": "X"}, n_dep=2, n_tof=3, workers=1)
    tof = np.asarray(out["tof_days"], float)
    ok = np.asarray(out["feasible"], bool)
    assert len(out["leg_terms"]) == 2 and all(len(col) == 3 for col in out["leg_terms"])
    for i in range(2):
        for j in range(3):
            terms = out["leg_terms"][i][j]
            if not ok[i, j]:
                assert terms is None
            elif tof[j] > 200.0:
                assert terms == {"seg_caps": [0.9, 0.7], "thrust_N": 0.232, "isp_s": 1751.0,
                                 "seg_isp_s": [1783.0, 1712.0]}
            else:
                assert terms is None


def test_leg_terms_are_json_safe_lists_from_the_solution():
    class Sol:
        seg_caps = np.array([0.9, 0.7])
        seg_isp_s = np.array([1783.0, 1712.0])
        thrust_N = 0.232
        isp_s = 1751.0

    assert G.leg_terms(Sol()) == {"seg_caps": [0.9, 0.7], "thrust_N": 0.232, "isp_s": 1751.0,
                                  "seg_isp_s": [1783.0, 1712.0]}

    class Bare:
        thrust_N, isp_s = 0.1, 1500.0

    assert G.leg_terms(Bare()) == {"seg_caps": None, "thrust_N": 0.1, "isp_s": 1500.0,
                                   "seg_isp_s": None}
