"""Smoke tests for the reachability figures.

These confirm the figures build, carry traces, and respond to the delta-v budget: the budget is the
filter, so the dome has to grow when the budget grows. They check the wiring and that coupling, not
what the pixels look like.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from prospector import figures
from prospector.figures import products
from prospector.solvers import edelbaum


def _population(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "full_name": [f"obj {k}" for k in range(n)],
        "pdes": [str(k) for k in range(n)],
        "a": rng.uniform(0.7, 3.5, n),
        "e": rng.uniform(0.0, 0.6, n),
        "i": rng.uniform(0.0, 35.0, n),
    })
    return edelbaum.evaluate_dataframe(df, dv_budget=7.0)


def test_all_figures_build():
    df = _population()
    for fig in (
        figures.reachability_dome_3d(7.0, df, config_name="t"),
        figures.element_distributions(df, dv_budget=7.0, config_name="t"),
    ):
        assert isinstance(fig, go.Figure)
        assert fig.data


def test_dome_builds_without_a_population():
    # The dome is the vehicle's filter; it must render from the budget alone.
    fig = figures.reachability_dome_3d(6.0)
    assert any(isinstance(t, go.Isosurface) for t in fig.data)


def test_dome_grows_with_budget():
    # A bigger budget reaches a larger circular orbit, so the size axis must extend.
    small = figures.reachability_dome_3d(4.0).layout.scene.xaxis.range[1]
    large = figures.reachability_dome_3d(9.0).layout.scene.xaxis.range[1]
    assert large > small


def test_reference_marker_is_added():
    df = _population()
    ref = {"a": 0.958, "e": 0.083, "i": 7.4, "label": "2008 EV5", "dv": 3.0}
    fig = figures.reachability_dome_3d(7.0, df, reference=ref)
    assert any(getattr(t, "name", None) == "2008 EV5" for t in fig.data)


def test_distributions_overlay_reachable_subset():
    df = _population()
    fig = figures.element_distributions(df, dv_budget=7.0)
    names = {getattr(t, "name", None) for t in fig.data}
    assert "all" in names and "reachable" in names


def _circle(r=1.0, n=80):
    t = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(t), r * np.sin(t), np.zeros(n)])


def test_trajectory_3d_is_thrust_coloured_and_animated():
    fine = np.linspace([1, 0, 0], [-1, 0.2, 0.13], 200)   # smooth densified arc
    days = np.linspace(0, 500, 200)
    thr = np.clip(np.sin(days / 60), 0, 1)
    nodes_d = np.linspace(0, 500, 25)
    nodes_u = np.clip(np.sin(nodes_d / 60), 0, 1)
    earth_track = np.linspace([1, 0, 0], [0.9, 0.4, 0], 200)
    target_track = np.linspace([0.96, 0, 0.1], [-1, 0.2, 0.13], 200)
    fig = figures.trajectory_3d(fine, thr, days, nodes_d, nodes_u, _circle(1.0), _circle(0.96),
                              earth_track, target_track, "2022-01-01", "2023-05-15",
                              "2008 EV5", 3.5, 500)
    assert fig.frames                                   # has a synced time scrubber
    transfer = next(t for t in fig.data if getattr(t, "name", None) == "transfer")
    assert list(transfer.line.color) == list(thr)      # arc coloured by throttle
    # the synced thrust profile is in percent (0-100), below the 3D scene
    thrust = next(t for t in fig.data if getattr(t, "fill", None) == "tozeroy")
    assert thrust.y.max() <= 100 + 1e-6


def test_throttle_hover_carries_the_segments_duty_cycle():
    """Hovering a segment of the thrust profile must report its duty cycle, not just its share
    of rated thrust.

    The two differ whenever the array cannot power full thrust. The plotted percentage is
    referred to RATED so it can share a frame with the power ceiling; the duty cycle is the raw
    throttle, which, Isp being constant, is the share of that segment the engine spends
    firing. At a 50%-of-rated ceiling a segment plotted at 20% is the engine running 40% of the
    time, and sizing anything off the 20% understates it by half."""
    nd = np.array([0.0, 100.0, 200.0, 300.0])
    u = np.array([1.0, 0.5, 0.0, 0.4])
    fine = np.linspace([1, 0, 0], [-1, 0.2, 0.13], 4)

    def _hover_trace(fig):
        return next(t for t in fig.data
                    if "duty cycle" in (getattr(t, "hovertemplate", None) or ""))

    for fig in (figures.trajectory_3d(fine, u, nd, nd, u, _circle(1.0), _circle(0.96),
                                      fine, fine, animate=False, controls=False,
                                      max_throttle_pct=50.0),
                figures.trajectory_diagnostics(nd, u, u * 0, u, u * 0, np.full(4, 5.0),
                                               np.ones(4), np.zeros(4),
                                               max_throttle_pct=50.0)):
        trace = _hover_trace(fig)
        duty = np.ravel(np.asarray(trace.customdata, float))
        assert np.allclose(duty, u * 100.0), "hover must quote the raw throttle as a percentage"
        assert np.allclose(np.asarray(trace.y, float), u * 50.0), "the plot stays referred to rated"
        # The two readings are named apart, so neither can be mistaken for the other.
        assert "of rated" in trace.hovertemplate and "duty cycle" in trace.hovertemplate

    # The peak the hover shows agrees with the shared derivation every other consumer reads, off
    # the same history, one number quoted two ways would be worse than not quoting it.
    card = products.cruise_duty_cycle({"fine_times_days": nd.tolist(),
                                       "fine_throttle": u.tolist()})
    diag = _hover_trace(figures.trajectory_diagnostics(
        nd, u, u * 0, u, u * 0, np.full(4, 5.0), np.ones(4), np.zeros(4), max_throttle_pct=50.0))
    assert np.isclose(np.ravel(np.asarray(diag.customdata, float)).max() / 100.0, card["peak"])


def test_power_limited_markers_render():
    # The array-power thrust ceiling shows on both throttle plots, and the degradation plot marks
    # the loss beyond which full thrust can't be held (the power-limited zone).
    fine = np.linspace([1, 0, 0], [-1, 0.2, 0.13], 80)
    days = np.linspace(0, 400, 80)
    thr = np.full(80, 0.4)
    nd = np.linspace(0, 400, 20)
    nu = np.full(20, 0.4)
    # trajectory_3d (mixed 3D scene + xy throttle row): the cap is a dashed trace at 45%.
    fig = figures.trajectory_3d(fine, thr, days, nd, nu, _circle(1.0), _circle(0.96),
                              np.linspace([1, 0, 0], [0.9, 0.4, 0], 80),
                              np.linspace([0.96, 0, 0.1], [-1, 0.2, 0.13], 80),
                              target_i_deg=5.0, animate=False, controls=False,
                              max_throttle_pct=45.0)
    assert any(getattr(getattr(t, "line", None), "dash", None) == "dash"
               and list(getattr(t, "y", []) or []) == [45.0, 45.0] for t in fig.data)
    # trajectory_diagnostics (all 2D): the cap is an hline shape at 45%.
    diag = figures.trajectory_diagnostics(nd, nu, nu * 0, nu, nu * 0, np.full(20, 5.0),
                                        np.ones(20), np.zeros(20), max_throttle_pct=45.0)
    assert any(s.y0 == s.y1 == 45.0 for s in (diag.layout.shapes or []))
    # The ceiling shows even at full power capability (line at 100%), so "no power limit" is explicit.
    full = figures.trajectory_3d(fine, thr, days, nd, nu, _circle(1.0), _circle(0.96),
                               np.linspace([1, 0, 0], [0.9, 0.4, 0], 80),
                               np.linspace([0.96, 0, 0.1], [-1, 0.2, 0.13], 80),
                               target_i_deg=5.0, animate=False, controls=False,
                               max_throttle_pct=100.0)
    assert any(getattr(getattr(t, "line", None), "dash", None) == "dash"
               and list(getattr(t, "y", []) or []) == [100.0, 100.0] for t in full.data)
    # degradation_profile: a power-limited zone above the full-thrust floor.
    t = np.linspace(0, 120, 100)
    pf = np.clip(1 - np.linspace(0, 0.5, 100), 0, 1)
    deg = figures.degradation_profile(t, pf, power_floor_pct=37.5)
    assert len(deg.layout.shapes or []) >= 2          # hrect (zone) + hline (floor)
    assert not figures.degradation_profile(t, pf).layout.shapes   # none without a floor
    # Headroom case: a high floor the curve never reaches is still shown, and the y-axis is framed
    # to keep it in view (so "we have margin" is visible, not off-screen).
    head = figures.degradation_profile(t, np.clip(1 - np.linspace(0, 0.1, 100), 0, 1),
                                     power_floor_pct=55.0)
    assert len(head.layout.shapes or []) >= 2
    assert head.layout.yaxis.range is not None and head.layout.yaxis.range[1] >= 55.0


def test_trajectory_diagnostics_builds():
    nd = np.linspace(0, 500, 21)
    thr = np.clip(np.sin(nd / 50), 0, 1)
    radial = 0.2 * np.cos(nd / 40)
    transverse = 0.6 * np.clip(np.sin(nd / 50), 0, 1)
    normal = 0.5 * np.sin(nd / 60)
    inc = 5 + 3 * np.sin(nd / 100)
    a = 1.0 + 0.05 * nd / 500
    e = 0.1 * np.ones_like(nd)
    speed = 29.8 + 0.6 * nd / 500
    fig = figures.trajectory_diagnostics(nd, thr, radial, transverse, normal, inc, a, e,
                                       speed_kms=speed, earth_speed_dep=29.8,
                                       target_speed_arr=30.4, target_a=0.96, target_e=0.08,
                                       target_i=7.4, target_name="2008 EV5")
    # throttle + 3 thrust-direction + speed + i + a + e
    assert isinstance(fig, go.Figure) and len(fig.data) >= 7
    speed_trace = next(t for t in fig.data
                       if "km/s" in (getattr(t, "hovertemplate", "") or ""))
    assert float(np.max(speed_trace.y)) <= 31.0      # speed panel carries the heliocentric speed

    # Without speed it still builds (older saved runs): 5 panels, no speed trace.
    bare = figures.trajectory_diagnostics(nd, thr, radial, transverse, normal, inc, a, e)
    assert isinstance(bare, go.Figure) and len(bare.data) >= 6


def test_trajectory_3d_thrust_cones_are_optional():
    # Cones (the thrust vector field) appear only when enabled and the node vectors are supplied.
    fine = np.linspace([1, 0, 0], [-1, 0.1, 0.05], 40)
    days = np.linspace(0, 400, 40)
    thr = np.clip(np.sin(days / 50), 0, 1)
    nodes_d = np.linspace(0, 400, 17)
    nodes_u = np.clip(np.sin(nodes_d / 50), 0, 1)
    e_tr = np.linspace([1, 0, 0], [0.9, 0.4, 0], 40)
    t_tr = np.linspace([0.96, 0, 0.05], [-1, 0.1, 0.05], 40)
    node_pos = np.linspace([1, 0, 0], [-0.96, 0.1, 0.05], 18)
    node_vec = np.tile([0.0, 0.6, 0.2], (18, 1))      # a thrust direction at every node
    # RTN components whose dominant axis cycles radial / transverse / normal -> 3 colour groups.
    rtn = np.zeros((18, 3))
    rtn[0::3, 0] = 0.6        # radial-dominant
    rtn[1::3, 1] = 0.6        # transverse-dominant
    rtn[2::3, 2] = 0.6        # normal-dominant
    common = dict(target_i_deg=7.0, node_pos_au=node_pos, node_thrust_vec=node_vec, node_rtn=rtn)
    on = figures.trajectory_3d(fine, thr, days, nodes_d, nodes_u, _circle(1.0), _circle(0.96),
                             e_tr, t_tr, "d", "a", "EV5", 3.5, 400, show_cones=True, **common)
    off = figures.trajectory_3d(fine, thr, days, nodes_d, nodes_u, _circle(1.0), _circle(0.96),
                              e_tr, t_tr, "d", "a", "EV5", 3.5, 400, show_cones=False, **common)
    cones = [t for t in on.data if isinstance(t, go.Cone)]
    assert len(cones) == 3                                      # one colour trace per steering axis
    assert not any(isinstance(t, go.Cone) for t in off.data)   # absent when disabled (default)


def test_trajectory_3d_zaxis_frames_to_inclination():
    fine = np.linspace([1, 0, 0], [-1, 0.02, 0.013], 60)   # small z so the inclination sets the frame
    days = np.linspace(0, 500, 60)
    thr = np.clip(np.sin(days / 60), 0, 1)
    nodes_d = np.linspace(0, 500, 21)
    nodes_u = np.clip(np.sin(nodes_d / 60), 0, 1)
    e_tr = np.linspace([1, 0, 0], [0.9, 0.4, 0], 60)
    t_tr = np.linspace([0.96, 0, 0.01], [-1, 0.02, 0.013], 60)

    def frame(i):
        fig = figures.trajectory_3d(fine, thr, days, nodes_d, nodes_u, _circle(1.0), _circle(0.96),
                                  e_tr, t_tr, "d", "a", "EV5", 3.5, 500, target_i_deg=i)
        zr = fig.layout.scene.zaxis.range
        return zr[1] - zr[0], fig.layout.scene.aspectratio.z

    span1, asp1 = frame(1.0)
    span20, asp20 = frame(20.0)
    assert span20 > span1        # the z RANGE scales up with inclination (1° floor -> 20°)
    assert asp20 > asp1          # to scale: bigger inclination -> a taller (true-angle) z box


def test_distributions_clip_outliers():
    # An extreme orbit must not blow out the axis frame: the panel stays on the bulk and the
    # outlier is counted, not framed.
    df = _population()
    df.loc[0, "a"] = 500.0          # absurd semimajor axis
    fig = figures.element_distributions(df, dv_budget=7.0)
    a_range = fig.layout.xaxis.range  # first subplot is the 'a' panel
    assert a_range[1] < 50.0          # frame ignores the 500 AU outlier


def test_spiral_plots_build():
    # The launch-phase views: geocentric spiral (Earth to scale), the dv-vs-vinf tradeoff curve
    # with its breakeven line and markers, and the time diagnostics.
    t = np.linspace(0, 200, 300)
    pos = np.column_stack([7000 * np.cos(t), 7000 * np.sin(t), 100 * t])
    fig = figures.spiral_3d(pos, t, dv_kms=7.5, tof_days=200)
    assert any(d.type == "surface" for d in fig.data)            # Earth sphere present
    assert fig.layout.scene.aspectmode == "data"                  # geocentric, to scale
    eq = [d for d in fig.data if d.name == "equator"]
    assert eq                                                     # the out-of-plane reference
    # No rotation: Earth's equatorial plane is FLAT, its rim lies exactly in z = 0.
    rim = next(d for d in eq if d.type == "scatter3d")
    assert np.allclose(np.asarray(rim.z, float), 0.0)
    assert any(d.name == "ecliptic" for d in fig.data)            # ecliptic shown tilted, for reference
    assert not any(d.name == "sun" for d in fig.data)             # sun removed (was confusing)

    fig = figures.spiral_diagnostics(t, np.linspace(-30, 1, 300), np.full(300, 28.5),
                                   np.linspace(650, 420, 300))
    assert len(fig.data) == 3


def test_report_figures_build():
    # The standalone-document figures: array power through the belts, cumulative propellant over
    # the mission, and the dry-mass allocation. Each carries an explicit width/height so the report
    # rasterizes it at the right aspect.
    t = np.linspace(0, 120, 200)
    pt = figures.propellant_timeline(t, np.linspace(0, 180, 200),
                                   np.linspace(0, 400, 300), np.linspace(0, 60, 300),
                                   usable_kg=240)
    names = {d.name for d in pt.data}
    assert "Earth escape" in names and "heliocentric cruise" in names

    ma = figures.mass_allocation([("Solar array", 40), ("Thrusters", 44), ("Tank", 30),
                                ("Structure", 100), ("Payload + margin", 30)],
                               dry_mass_kg=250)
    assert ma.data and ma.data[0].orientation == "h"


def test_distance_profile_ranges_and_traces():
    # Two straight 3-sample tracks: spacecraft leaves Earth (0,0,0) toward a target parked at
    # (2,0,0). Earth range should grow 0 -> 2; target range should shrink 2 -> 0.
    t = [0.0, 50.0, 100.0]
    sc = [[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]
    earth = [[0.0, 0, 0], [0.0, 0, 0], [0.0, 0, 0]]
    target = [[2.0, 0, 0], [2.0, 0, 0], [2.0, 0, 0]]
    fig = figures.distance_profile(t, sc, earth, target, target_name="X")
    assert isinstance(fig, go.Figure) and len(fig.data) == 2
    to_earth = next(tr for tr in fig.data if tr.name == "to Earth")
    to_target = next(tr for tr in fig.data if tr.name == "to X")
    assert list(to_earth.y) == [0.0, 1.0, 2.0]        # grows away from Earth
    assert list(to_target.y) == [2.0, 1.0, 0.0]        # closes on the target (rendezvous)


def test_distance_profile_truncates_to_shortest_track():
    # Mismatched lengths must not raise, align to the shortest.
    fig = figures.distance_profile([0.0, 1.0], [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                                 [[0, 0, 0], [0, 0, 0]], [[3, 0, 0], [3, 0, 0], [3, 0, 0]],
                                 target_name="Y")
    assert all(len(tr.y) == 2 for tr in fig.data)


def test_distance_profile_sun_trace_and_transitions():
    # The optional Sun range rides as a third trace and the phase seams draw as dashed vertical
    # lines (whole-mission clock).
    t = [0.0, 50.0, 100.0]
    sc = [[1.0, 0, 0], [1.2, 0, 0], [1.4, 0, 0]]
    still = [[1.0, 0, 0]] * 3
    fig = figures.distance_profile(t, sc, still, still, sun_au=[1.0, 1.2, 1.4],
                                 transitions=[(50.0, "cruise")], target_name="X")
    sun = next(tr for tr in fig.data if tr.name == "to Sun")
    assert list(sun.y) == [1.0, 1.2, 1.4]
    assert any(getattr(s, "x0", None) == 50.0 for s in fig.layout.shapes)


def test_mission_power_timeline_splits_the_three_effects():
    # Per non-empty phase: an array-output trace and an at-thrusters trace (the conversion loss).
    # Plus THREE effect lines in the bottom panel, belt degradation, irradiance, and cell
    # efficiency, each continuous across all phases. Empty phases are skipped.
    fig = figures.mission_power_timeline(
        [("Earth escape", [0.0, 100.0], [4880.0, 3600.0], [3675.0, 2712.0],
          [100.0, 90.0], [100.0, 100.0], [100.0, 98.0], [79.0, 62.0], [26.6, 27.6]),
         ("cruise", [100.0, 400.0], [3600.0, 2000.0], [2712.0, 1507.0],
          [90.0, 90.0], [125.0, 58.0], [96.0, 109.0], [70.0, 5.0], [27.1, 31.0]),
         ("return", [], [], [], [], [], [], [], [])],            # empty phase skipped
        transitions=[(100.0, "departure")], nameplate_W=4880.0)
    names = [tr.name for tr in fig.data]
    assert "Earth escape · array output" in names
    assert "Earth escape · at thrusters" in names
    assert "cruise · array output" in names
    assert "return · array output" not in names                 # empty phase skipped
    # The three effects are separate, continuous lines (one per effect, not per phase).
    for effect in ("belt degradation", "irradiance (sun distance)",
                   "cell efficiency (temperature)"):
        assert names.count(effect) == 1
    # The thermal panel: the panel temperature itself and the absolute cell efficiency, each one
    # continuous line across the phases, on their own row (efficiency on the secondary axis).
    temp = next(tr for tr in fig.data if tr.name == "array temperature")
    eff = next(tr for tr in fig.data if tr.name == "cell efficiency")
    assert [v for v in temp.y if v is not None] == [79.0, 62.0, 70.0, 5.0]
    assert [v for v in eff.y if v is not None] == [26.6, 27.6, 27.1, 31.0]
    assert temp.yaxis != eff.yaxis and temp.xaxis == eff.xaxis
    assert fig.layout.yaxis3.title.text == "panel °C"
    # Irradiance legitimately exceeds 100% sunward (the point of the split).
    irr = next(tr for tr in fig.data if tr.name == "irradiance (sun distance)")
    assert max(v for v in irr.y if v is not None) > 100.0
    deg = next(tr for tr in fig.data if tr.name == "belt degradation")
    assert max(v for v in deg.y if v is not None) <= 100.0       # degradation never exceeds 100%
    assert any(getattr(s, "y0", None) == 4880.0 for s in fig.layout.shapes)    # nameplate line


def test_engine_performance_timeline_two_panels():
    # Available vs flown thrust in the top panel, Isp in the bottom; None rows are skipped.
    fig = figures.engine_performance_timeline(
        [("cruise", [0.0, 100.0], [200.0, 150.0], [180.0, 140.0], [1500.0, 1400.0]),
         ("return", [100.0, 200.0], [150.0, 180.0], None, None)],
        transitions=[(100.0,'return')], rated_thrust_mN=220.0)
    names = [tr.name for tr in fig.data]
    assert "cruise · available" in names and "cruise · flown" in names
    assert "return · available" in names and "return · flown" not in names
    isp = next(tr for tr in fig.data if tr.name == "cruise · Isp")
    assert list(isp.y) == [1500.0, 1400.0]


def test_dome_marks_the_focus_and_picked_bodies():
    """A chosen target has to be findable in a cloud of thousands, so it gets its own marker.

    Drawn as separate traces rather than by recolouring a scatter point, because the population
    scatter is subsampled (the picked body may not be in it) and a target can sit outside the
    reachable set entirely, where there is no point to recolour.
    """
    fig = figures.reachability_dome_3d(
        8.0, grid=12,
        highlights=[{"a": 1.1, "e": 0.2, "i": 5.0, "role": "picked", "label": None},
                    {"a": 1.1, "e": 0.2, "i": 5.0, "role": "focus", "label": "99942"}])
    names = [getattr(t, "name", None) for t in fig.data]
    assert "focus target" in names and "picked in table" in names
    focus = next(t for t in fig.data if t.name == "focus target")
    assert focus.x[0] == pytest.approx(1.1) and focus.z[0] == pytest.approx(5.0)
    assert focus.text == ("99942",)
    # The two roles differ in shape as well as colour, so the picture survives in greyscale.
    picked = next(t for t in fig.data if t.name == "picked in table")
    assert focus.marker.symbol != picked.marker.symbol


def test_dome_skips_a_highlight_it_cannot_place():
    """A body with a missing or unparseable element must be left out, not drawn at the origin --
    where it would read as a target sharing Earth's own orbit."""
    fig = figures.reachability_dome_3d(
        8.0, grid=12,
        highlights=[{"a": float("nan"), "e": 0.1, "i": 1.0, "role": "focus", "label": "bad"},
                    {"a": 1.0, "role": "focus"},                       # missing e / i
                    {"a": 1.0, "e": 0.1, "i": 1.0, "role": "not-a-role"}])
    names = [getattr(t, "name", None) for t in fig.data]
    assert "focus target" not in names and "picked in table" not in names


def _grid_surface():
    """A small converged-grid surface, with one cell the search did not close."""
    dep = np.array([10000.0, 10010.0, 10020.0])
    tof = np.array([500.0, 400.0, 300.0])          # longest first, as the march returns it
    dv = np.array([[3.4, 3.1, 3.6], [3.3, 3.0, 3.7], [3.5, 3.2, np.nan]])
    mass = np.array([[400.0, 430.0, 380.0], [410.0, 440.0, 370.0], [395.0, 425.0, np.nan]])
    ok = np.isfinite(dv)
    return dep, tof, dv, mass, ok


def test_the_converged_grid_never_calls_a_missing_cell_infeasible():
    """A cell the search did not close means the optimizer found nothing at this effort, not that
    the cell cannot be flown. Only the optimizer's own convergence supports either claim, and the
    surface this replaces had an infeasibility verdict that fired on 16, 21 and 18 flyable cells
    across three reference targets. So the wording is the safety property, and it is pinned.
    """
    dep, tof, dv, mass, ok = _grid_surface()
    fig = figures.converged_grid(dep, tof, dv_kms=dv, final_mass_kg=mass, feasible=ok)
    cells = next(t for t in fig.data if t.name == "cells")
    missing = [h for h in cells.text if "no trajectory found" in h]
    assert len(missing) == 1, cells.text
    joined = " ".join(cells.text).lower()
    assert "infeasible" not in joined and "cannot be flown" not in joined
    assert "unreachable" not in joined
    # The not-found layer is a flat slab, not a ranked gradient: a non-converged cell has no cost
    # to rank, and a gradient would imply it did.
    slab = next(t for t in fig.data if t.name == "not found")
    assert slab.zmin == 0.0 and slab.zmax == 1.0
    flat = {c[1] for c in slab.colorscale}
    assert len(flat) == 1, "not-found is one flat colour, not a ranked gradient"

    # And it must not be confusable with an over-budget cell, which is a different fact: that
    # trajectory exists and is simply too expensive for this vehicle.
    priced = figures.converged_grid(dep, tof, dv_kms=dv, final_mass_kg=mass, feasible=ok,
                                    colour_by="dv", dv_budget=3.15)
    grey = next(t for t in priced.data if t.name == "over budget")
    assert flat.isdisjoint({c[1] for c in grey.colorscale}), (
        "not-found and over-budget render the same colour")


def test_hot_means_good_in_both_colour_fields():
    """The field flips between delivered mass (more is better) and ΔV (less is better). If the scale
    did not flip with it, the same picture would read as its own opposite, so the reversal is tied
    to the field rather than fixed."""
    dep, tof, dv, mass, ok = _grid_surface()
    by_mass = figures.converged_grid(dep, tof, dv_kms=dv, final_mass_kg=mass, feasible=ok,
                                     colour_by="mass")
    by_dv = figures.converged_grid(dep, tof, dv_kms=dv, final_mass_kg=mass, feasible=ok,
                                   colour_by="dv")
    assert by_mass.data[0].reversescale is False
    assert by_dv.data[0].reversescale is True
    assert "mass" in by_mass.data[0].colorbar.title.text
    assert "ΔV" in by_dv.data[0].colorbar.title.text
    # The field only ever spans converged cells, so one NaN cannot drag the ramp.
    assert np.isfinite(by_dv.data[0].zmin) and np.isfinite(by_dv.data[0].zmax)
    assert by_dv.data[0].zmin == pytest.approx(np.nanmin(dv))


def test_the_grid_axis_says_a_departure_is_not_a_liftoff():
    """The y axis is the cruise start, which for a SEP escape is the liftoff plus the whole spiral.
    Leaving that implicit is what made the two easy to conflate, so the axis carries the offset and
    every hover carries the liftoff date."""
    dep, tof, dv, mass, ok = _grid_surface()
    fig = figures.converged_grid(dep, tof, dv_kms=dv, final_mass_kg=mass, feasible=ok,
                                 target_name="Apophis", liftoff_offset_days=160.0)
    assert "liftoff + 160" in fig.layout.yaxis.title.text
    assert "flight time" in fig.layout.xaxis.title.text
    assert "launch" not in (fig.layout.title.text or "").lower()
    cells = next(t for t in fig.data if t.name == "cells")
    assert all("liftoff ~" in h for h in cells.text)


def test_the_best_per_flight_time_curve_rides_on_the_grid_that_produced_it():
    """The best cell in each column is the flight-time trade. Drawing it on the same axes as the
    grid is what makes a separate trade tab unnecessary, and a re-solved point is marked so a
    reader can tell which numbers came from the extra pass."""
    dep, tof, dv, mass, ok = _grid_surface()
    frontier = [{"tof_days": 500.0, "dv_kms": 3.3, "dep_mjd2000": 10010.0, "source": "grid"},
                {"tof_days": 400.0, "dv_kms": 2.9, "dep_mjd2000": 10010.0, "source": "polish"},
                {"tof_days": 300.0, "dv_kms": None, "dep_mjd2000": None, "source": "grid"}]
    fig = figures.converged_grid(dep, tof, dv_kms=dv, final_mass_kg=mass, feasible=ok,
                                 frontier=frontier)
    line = next(t for t in fig.data if t.name == "best per flight time")
    assert list(line.x) == [500.0, 400.0], "a flight time with no trajectory must not be drawn"
    # A re-solved point has to read differently from a grid point, checked by comparing the two
    # rather than by looking for a particular word, so rewording the hover does not fail this.
    plain = [dict(p, source="grid") for p in frontier]
    same = figures.converged_grid(dep, tof, dv_kms=dv, final_mass_kg=mass, feasible=ok,
                                  frontier=plain)
    plain_line = next(t for t in same.data if t.name == "best per flight time")
    assert line.text[1] != plain_line.text[1], "a polished point reads the same as a grid point"
    assert line.text[0] == plain_line.text[0], "a grid point was marked as re-solved"
    star = next(t for t in fig.data if t.name == "best")
    assert float(star.x[0]) == 400.0, "the star must sit on the curve's cheapest point"


def test_a_grid_where_nothing_converged_still_renders():
    """Nothing closing is a possible answer, and the workspace has to draw it rather than raise."""
    dep, tof, _dv, mass, _ok = _grid_surface()
    dead = np.full((3, 3), np.nan)
    fig = figures.converged_grid(dep, tof, dv_kms=dead, final_mass_kg=dead,
                                 feasible=np.zeros((3, 3), bool))
    cells = next(t for t in fig.data if t.name == "cells")
    assert all("no trajectory found" in h for h in cells.text)
    assert not any(t.name in ("best per flight time", "best") for t in fig.data)


def test_only_affordable_cells_get_colour_and_the_ramp_spans_them():
    """Colour means AFFORDABLE. A cell whose cruise ΔV exceeds what the tank holds after escape is
    not an option, so it recedes to grey, still drawn and still clickable, because this grid's ΔV
    is converged rather than estimated and how far over matters.

    The ramp then spans only those affordable cells, clipped near their top, and both halves
    matter. Spanning every converged cell lets a few at the feasibility wall stretch the scale over
    a range the basin does not occupy: measured on Apophis, 88% of cells fell in the bottom 20% of
    that ramp, which is why three quarters of the field read as one flat colour. Clipping brings it
    to 35%.
    """
    dep = np.array([10000.0, 10010.0, 10020.0, 10030.0, 10040.0, 10050.0])
    tof = np.array([500.0, 400.0])
    # A tight basin plus two cells far up at the wall, the shape that flattened the field.
    dv = np.array([[3.00, 3.05], [3.02, 3.08], [3.04, 3.10],
                   [3.06, 3.12], [3.08, 6.20], [3.10, 6.40]])
    mass = 500.0 - 30.0 * dv
    ok = np.ones_like(dv, bool)
    budget = 4.5

    fig = figures.converged_grid(dep, tof, dv_kms=dv, final_mass_kg=mass, feasible=ok,
                                 colour_by="dv", dv_budget=budget)
    names = [t.name for t in fig.data]
    assert "over budget" in names, "cells above the budget were not greyed"
    field = fig.data[0]
    # The ramp never reaches the over-budget cells...
    assert field.zmax < budget, (field.zmin, field.zmax)
    # ...and it is much tighter than the full spread, which is the whole point.
    assert field.zmax - field.zmin < 0.5 * (dv.max() - dv.min())

    # An over-budget cell is named as a shortfall against capability, not as a verdict on the
    # transfer: the trajectory is real and flyable, this vehicle just cannot afford it.
    cells = next(t for t in fig.data if t.name == "cells")
    over = [h for h in cells.text if "over the cruise budget" in h]
    assert len(over) == 2, cells.text
    assert not any("infeasible" in h.lower() or "not found" in h for h in over)

    # With no budget given, nothing greys, the caller opted out of the comparison.
    plain = figures.converged_grid(dep, tof, dv_kms=dv, final_mass_kg=mass, feasible=ok,
                                   colour_by="dv")
    assert "over budget" not in [t.name for t in plain.data]


def test_the_mass_field_clips_at_its_own_useful_end():
    """More mass is better, so the crowded end is the HIGH one and the clip belongs low. Clipping
    the wrong end would put the whole contrast on the handful of near-worthless cells."""
    dep = np.linspace(10000.0, 10050.0, 6)
    tof = np.array([500.0, 400.0])
    mass = np.array([[400.0, 398.0], [396.0, 394.0], [392.0, 390.0],
                     [388.0, 386.0], [384.0, 60.0], [380.0, 40.0]])
    dv = 10.0 - 0.01 * mass
    ok = np.ones_like(mass, bool)
    fig = figures.converged_grid(dep, tof, dv_kms=dv, final_mass_kg=mass, feasible=ok,
                                 colour_by="mass")
    field = fig.data[0]
    assert field.zmax == pytest.approx(mass.max())
    assert field.zmin > mass.min(), "the sparse low end still owns part of the ramp"
    assert field.reversescale is False


def test_mission_power_timeline_without_thermal_series_leaves_that_panel_empty():
    # Older seven-field phases still draw the power and effect panels; the thermal panel stays
    # empty rather than raising.
    fig = figures.mission_power_timeline(
        [("cruise", [0.0, 100.0], [3600.0, 2000.0], [2712.0, 1507.0],
          [90.0, 90.0], [125.0, 58.0], [96.0, 109.0])])
    names = [tr.name for tr in fig.data]
    assert "cruise · array output" in names
    assert "array temperature" not in names and "cell efficiency" not in names
