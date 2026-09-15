"""The flyby epoch is marked on every time-axis chart: a dashed vertical in the planet's colour on
the thrust profile under the trajectory, on each diagnostics panel, and on the mission-profile
timelines through the shared phase-seam markers."""
import numpy as np

from prospector import figures
from prospector.figures.theme import FLYBY_GREY, MUTED
from prospector.figures.trajectory import _phase_transition_lines, flyby_marker


def _circle(r, n=24):
    th = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(th), r * np.sin(th), np.zeros(n)])


def test_flyby_marker_is_the_planets_colour():
    assert flyby_marker(412.5, "Mars") == (412.5, "Mars flyby", FLYBY_GREY)
    assert flyby_marker(1.0, "") == (1.0, "flyby", FLYBY_GREY)


def test_the_thrust_profile_and_the_diagnostics_carry_the_line():
    nd = np.array([0.0, 100.0, 200.0, 300.0])
    u = np.array([1.0, 0.5, 0.0, 0.4])
    fine = np.linspace([1, 0, 0], [-1, 0.2, 0.13], 4)
    fig = figures.trajectory_3d(fine, u, nd, nd, u, _circle(1.0), _circle(0.96), fine, fine,
                                animate=False, controls=False, flyby_day=140.0, flyby_name="Mars")
    mark = next(t for t in fig.data if getattr(t, "name", "") == "flyby-mark")
    assert list(mark.x) == [140.0, 140.0] and mark.line.color == FLYBY_GREY
    assert any("Mars flyby" in (a.text or "") for a in fig.layout.annotations)
    plain = figures.trajectory_3d(fine, u, nd, nd, u, _circle(1.0), _circle(0.96), fine, fine,
                                  animate=False, controls=False)
    assert not any(getattr(t, "name", "") == "flyby-mark" for t in plain.data)

    diag = figures.trajectory_diagnostics(nd, u, u * 0, u, u * 0, np.full(4, 5.0), np.ones(4),
                                          np.zeros(4), flyby_day=140.0, flyby_name="Mars")
    lines = [s for s in diag.layout.shapes if s.type == "line" and s.line.color == FLYBY_GREY]
    assert len(lines) == len({s.yref for s in lines}) >= 4          # one per panel
    assert all(s.x0 == 140.0 for s in lines)


def test_phase_seams_take_a_colour_and_default_to_grey():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Scatter(x=[0, 10], y=[0, 1]), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 10], y=[0, 1]), row=2, col=1)
    _phase_transition_lines(fig, [(2.0, "Earth departure"), flyby_marker(6.0, "Mars")], rows=(1, 2))
    colours = sorted({s.line.color for s in fig.layout.shapes})
    assert colours == sorted({MUTED, FLYBY_GREY})
    assert len(fig.layout.shapes) == 4 and len(fig.layout.annotations) == 2
