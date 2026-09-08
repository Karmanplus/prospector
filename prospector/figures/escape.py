"""Launch-phase figures: the geocentric escape spiral and its dV-vs-v-infinity tradeoff.

The 3D climb is drawn inside a rendering of the radiation belts, so the part that damages the
arrays is visible rather than left to the imagination. The trade curve is where the escape meets
the cruise: what the spiral pays to deliver each departure speed.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from prospector.constants import (
    EARTH_EQUATORIAL_RADIUS_KM,
    EARTH_OBLIQUITY_RAD,
    MOON_ORBIT_TILT_RAD,
)
from prospector.figures.theme import (
    ASTEROID_ORANGE,
    BG,
    DOME,
    EARTH,
    EARTH_BLUE,
    ECCEN,
    GRID,
    MUTED,
    PLAY_GREEN,
    SPACECRAFT,
    SURFACE,
    TEXT,
    _layout,
)
from prospector.solvers import spiral as _spiral


def _orbit_plane_camera(pos: np.ndarray, distance: float = 2.1) -> tuple[dict, dict]:
    """A camera that looks face-on at the spiral, along its mean orbit-plane normal.

    The spiral lies close to a single plane (slowly precessing), so the winding only reads as a
    spiral when viewed perpendicular to that plane; any in-plane view collapses it to a line. The
    plane normal is the average of consecutive position cross products (the orbital
    angular-momentum direction, weighted naturally toward the wider loops). The view-up is world-Z
    projected off the view axis, so the framing stays sensible for any inclination (and never
    degenerates when the normal happens to point along Z). Returns Plotly ``eye``/``up`` dicts;
    falls back to a top-down view for a too-short path.
    """
    if pos.ndim != 2 or len(pos) < 3:
        return dict(x=0.0, y=0.0, z=distance), dict(x=0.0, y=1.0, z=0.0)
    h = np.cross(pos[:-1], pos[1:]).sum(axis=0)
    norm = float(np.linalg.norm(h))
    n = h / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
    z = np.array([0.0, 0.0, 1.0])
    up = z - np.dot(z, n) * n                 # world-up with the view axis removed
    up = up / np.linalg.norm(up) if np.linalg.norm(up) > 1e-3 else np.array([0.0, 1.0, 0.0])
    eye = n * distance
    return dict(x=float(eye[0]), y=float(eye[1]), z=float(eye[2])), \
        dict(x=float(up[0]), y=float(up[1]), z=float(up[2]))


# Van Allen radiation belts: drawn as a translucent 3D VOLUME of the spiral solver's damage field
# (go.Volume), so there are no hard planes to clutter the crossing. Opacity scales with the local
# dose rate, so a near-zero rate is almost transparent and the inner-proton core is bright. The
# belt reads as a
# soft body the near-polar spiral threads through, and a glance shows where the damage concentrates.


OUTER_BELT_KM = (3.0 * _spiral.RE_KM, 6.0 * _spiral.RE_KM)   # outer electron belt (L 3-6)


BELT_EXTENT_KM = 1.2 * OUTER_BELT_KM[1]                     # the field is negligible past here (~7 Re)


MOON_DIST_KM = 384400.0              # mean lunar distance, as a familiar scale marker




def belt_volume(extent_km, *, model=None, n: int = 40, opacity: float = 0.6) -> go.Volume:
    """The radiation-belt damage field as a translucent 3D volume (geocentric km).

    Samples the dose rate on a cube clipped to the belt region (the field is negligible past
    ``BELT_EXTENT_KM``, so a far escape view doesn't grid empty space) and renders it with
    ``opacityscale`` mapping intensity to opacity: near-zero dose is transparent, the diffuse outer
    electron belt reads as a faint haze and the inner-proton core glows, so both belts show, the
    lesser one subtly, with no hard rings.

    PERFORMANCE: a volume's payload is ``n^3`` coordinates, which dominates the figure's size, so
    ``n`` is kept modest and the field is lightly Gaussian-smoothed (the smoothing, not a finer grid,
    is what removes the faceted, crinkled look at this resolution), and the grid is sent as
    integer km / rounded intensity, halving the bytes on the wire. ``model`` is the
    :class:`~prospector.spacecraft.radiation.RadiationModel` flown (None uses the default scenario); the belt
    GEOMETRY is the same across scenarios, only the magnitude differs, and the volume self-normalizes.
    """
    rate_fn = model.ddd_rate if model is not None else _spiral.belt_ddd_rate
    cap = min(float(extent_km), BELT_EXTENT_KM)
    g = np.linspace(-cap, cap, n)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    value = np.log10(rate_fn(np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)) + 1e6)
    # Smooth the coarse grid so the sharp proton-plateau shoulders do not alias into facets. This
    # is what keeps it a soft cloud at a payload-friendly resolution.
    from scipy.ndimage import gaussian_filter
    value = gaussian_filter(value.reshape(X.shape), sigma=1.2).ravel()
    # Floor low enough to admit the outer electron belt's skirts (it is ~10x fainter than the
    # proton core), so it renders as a cloud rather than being clipped away.
    floor = float(np.log10(0.005 * _spiral.PROTON_DDD_CORE + 1e6))
    vmax = max(float(value.max()), floor + 1.0)
    return go.Volume(
        x=np.round(X.ravel()).astype(np.int32), y=np.round(Y.ravel()).astype(np.int32),
        z=np.round(Z.ravel()).astype(np.int32), value=np.round(value, 3),
        isomin=floor, isomax=vmax, opacity=opacity, surface_count=15,
        colorscale="Inferno", showscale=False,
        # Do not draw the field on the bounding-box faces, since those caps render as bright square
        # patches on the cube sides; only the soft interior cloud is wanted.
        caps=dict(x_show=False, y_show=False, z_show=False),
        # Relative opacity vs normalized intensity: transparent at the floor, a gentle ramp so the
        # fainter electron belt is a visible haze, climbing to the bright proton core.
        opacityscale=[[0.0, 0.0], [0.06, 0.0], [0.22, 0.06], [0.5, 0.16],
                      [0.78, 0.4], [1.0, 1.0]],
        hoverinfo="skip", name="belts")


def _densify_path(pos, days, factor=3):
    """Cubic-spline-upsample an (N,3) path by ``factor`` for a smoother line, parameterized by
    cumulative chord length so the dense curve follows the flown geometry. Returns
    ``(pos_dense, days_dense)``; a path too short to work with comes back unchanged."""
    pos = np.asarray(pos, float)
    days = np.asarray(days, float)
    if len(pos) < 4 or factor <= 1:
        return pos, days
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))))
    keep = np.concatenate(([True], np.diff(s) > 0))     # drop zero-length segments for the spline
    if keep.sum() < 4 or s[-1] <= 0:
        return pos, days
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(s[keep], pos[keep], axis=0)
    s_dense = np.linspace(s[keep][0], s[keep][-1], int(len(pos) * factor))
    return cs(s_dense), np.interp(s_dense, s[keep], days[keep])


def _add_moon(fig: go.Figure, extent: float) -> None:
    """Draw the lunar orbit as a tilted ring, with a Moon marker, to ground the scale. The
    belts hug Earth, the Moon sits far out at 384,400 km. Skipped when the view is zoomed
    in tighter than the Moon's distance (it would be off-frame)."""
    if extent < MOON_DIST_KM:
        return
    c, s = np.cos(MOON_ORBIT_TILT_RAD), np.sin(MOON_ORBIT_TILT_RAD)
    phi = np.linspace(0, 2 * np.pi, 120)
    ring = np.column_stack([MOON_DIST_KM * np.cos(phi), MOON_DIST_KM * np.sin(phi),
                            np.zeros_like(phi)]) @ np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]).T
    fig.add_trace(go.Scatter3d(
        x=ring[:, 0], y=ring[:, 1], z=ring[:, 2], mode="lines",
        line=dict(color=MUTED, width=1.5, dash="dot"), opacity=0.6,
        name="Moon's orbit", hovertemplate="Moon's orbit · 384,400 km<extra></extra>"))
    m = ring[15]                                    # a representative point on the ring
    fig.add_trace(go.Scatter3d(
        x=[m[0]], y=[m[1]], z=[m[2]], mode="markers+text",
        marker=dict(size=7, color="#c8ccd8", line=dict(color=BG, width=1)),
        text=["  Moon"], textposition="middle right", textfont=dict(color=MUTED, size=11),
        hovertemplate="Moon · 384,400 km from Earth<extra></extra>"))


def spiral_3d(positions_km, times_days, *, title="Escape spiral",
              dv_kms=None, tof_days=None, animate=False, n_frames=60,
              controls=False, show_belts=False, show_moon=False,
              max_extent_km=None, show_equator=True, radiation_model=None) -> go.Figure:
    """The geocentric escape spiral (km): Earth to scale, the path colored by elapsed time.

    Positions are relative to Earth's equator, which is the propagator's own frame, plotted as
    flown, with Earth's equatorial plane as the flat z = 0 disc (no rotation). The Earth sphere is
    rendered at its true radius so the early, tightly wound revolutions read against it; color
    encodes elapsed time, so the slow climb out of the well versus the fast exit is visible at a
    glance. The faint ring tilted by Earth's axial tilt is the plane of Earth's orbit, for
    reference.

    ``show_belts`` overlays the radiation-belt damage field as a translucent 3D volume whose
    opacity tracks the local dose (see :func:`belt_volume`), so the near-polar spiral threads
    through a soft body rather than crossing hard planes; ``radiation_model`` is the scenario flown
    (None = default). ``show_moon`` adds the lunar orbit + a Moon marker to ground the scale.
    ``max_extent_km`` clips the view to a cube of that half-size: the climb winds for ~90% of its
    samples within a few times the outer belt, then shoots far out to escape, so a belt-scaled box
    keeps the near-Earth winding and the belts legible while the hyperbolic tail simply leaves the
    frame. ``None`` auto-fits the whole path.
    """
    days = np.asarray(times_days, float)
    pos = np.asarray(positions_km, float)        # geocentric equatorial, plotted as flown
    extent = float(np.max(np.linalg.norm(pos, axis=1))) if len(pos) else EARTH_EQUATORIAL_RADIUS_KM * 10
    if max_extent_km is not None:
        extent = float(max_extent_km)            # belt-scaled view: reference rings track it too
    cos_e, sin_e = np.cos(EARTH_OBLIQUITY_RAD), np.sin(EARTH_OBLIQUITY_RAD)

    fig = go.Figure()
    # Earth, to scale, solid (a sphere is rotation-invariant).
    u, v = np.linspace(0, 2 * np.pi, 36), np.linspace(0, np.pi, 18)
    fig.add_trace(go.Surface(
        x=EARTH_EQUATORIAL_RADIUS_KM * np.outer(np.cos(u), np.sin(v)),
        y=EARTH_EQUATORIAL_RADIUS_KM * np.outer(np.sin(u), np.sin(v)),
        z=EARTH_EQUATORIAL_RADIUS_KM * np.outer(np.ones_like(u), np.cos(v)),
        colorscale=[[0, EARTH_BLUE], [1, EARTH_BLUE]], opacity=1.0,
        showscale=False, hoverinfo="skip"))
    # Rotation axis through the poles: a thick black bearing line whose tips poke past the solid
    # globe (the segment inside Earth is hidden; this only orients the view, north up).
    paxis = EARTH_EQUATORIAL_RADIUS_KM * 1.5
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[-paxis, paxis], mode="lines",
        line=dict(color="#000000", width=7), hoverinfo="skip", name="poles"))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[paxis * 1.12, -paxis * 1.12], mode="text",
        text=["N", "S"], textfont=dict(color=MUTED, size=12), hoverinfo="skip"))

    # Earth's equatorial plane: NO filled disc (it would tint the scene and hide the belt field) --
    # just a black ring hugging the globe's equator (the orientation marker, crisp against the blue
    # Earth and the field) plus a thin outer reference ring, ring-only like the ecliptic.
    phi = np.linspace(0, 2 * np.pi, 72)
    rim = extent * 1.05
    if show_equator:
        eqr = EARTH_EQUATORIAL_RADIUS_KM * 1.02
        fig.add_trace(go.Scatter3d(
            x=eqr * np.cos(phi), y=eqr * np.sin(phi), z=np.zeros_like(phi), mode="lines",
            line=dict(color="#000000", width=5), hoverinfo="skip", name="equator"))
        fig.add_trace(go.Scatter3d(
            x=rim * np.cos(phi), y=rim * np.sin(phi), z=np.zeros_like(phi),
            mode="lines", line=dict(color=EARTH_BLUE, width=1.5, dash="dot"), opacity=0.4,
            name="equator", hovertemplate="Earth's equatorial plane<extra></extra>"))
    # The ecliptic (Earth's orbit plane), tilted by the obliquity, for reference.
    ecl = np.column_stack([rim * 0.6 * np.cos(phi), rim * 0.6 * np.sin(phi),
                           np.zeros_like(phi)]) @ np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_e, -sin_e], [0.0, sin_e, cos_e]]).T
    fig.add_trace(go.Scatter3d(
        x=ecl[:, 0], y=ecl[:, 1], z=ecl[:, 2], mode="lines",
        line=dict(color=EARTH, width=1.5, dash="dot"), opacity=0.4,
        name="ecliptic", hovertemplate="ecliptic (Earth's orbit plane)<extra></extra>"))

    # Radiation belts: a translucent 3D volume of the damage field, opacity tracking the dose so
    # only the hot zones read and the spiral threads through a soft body (no hard planes). Drawn
    # before the path so the spiral reads on top.
    if show_belts:
        fig.add_trace(belt_volume(extent, model=radiation_model))
    if show_moon:
        _add_moon(fig, extent)

    # Spline-densify the path for a smooth line (the integrator samples only a handful of points
    # per early revolution, which otherwise reads as polygonal). The flown path can be ~20k
    # samples; plotted in full and densified it is the bulk of the figure's payload, so cap it
    # first. A hundreds-of-revolution winding reads cleanly at a few thousand points. Display only:
    # the scrubber and camera still use the flown samples.
    if len(pos) > 10000:
        keep = np.unique(np.linspace(0, len(pos) - 1, 10000).astype(int))
        pos_line, days_line = pos[keep], days[keep]
    else:
        pos_line, days_line = pos, days
    pos_d, days_d = _densify_path(pos_line, days_line, factor=2)
    # Send the line as integer km with rounded day and altitude: visually identical, far smaller.
    fig.add_trace(go.Scatter3d(
        x=np.round(pos_d[:, 0]).astype(np.int32), y=np.round(pos_d[:, 1]).astype(np.int32),
        z=np.round(pos_d[:, 2]).astype(np.int32), mode="lines", name="spiral",
        line=dict(color=np.round(days_d, 2), colorscale="Viridis", width=4,
                  colorbar=dict(title="day", x=1.0, len=0.7)),
        customdata=np.round(np.column_stack(
            [days_d, np.linalg.norm(pos_d, axis=1) - EARTH_EQUATORIAL_RADIUS_KM]), 1),
        hovertemplate="day %{customdata[0]:.1f}<br>alt %{customdata[1]:,.0f} km<extra></extra>"))
    fig.add_trace(go.Scatter3d(
        x=[pos[-1, 0]], y=[pos[-1, 1]], z=[pos[-1, 2]], mode="markers+text",
        marker=dict(size=5, color=DOME, line=dict(color=BG, width=1)),
        text=["  escape"], textposition="top center", textfont=dict(color=DOME),
        hoverinfo="skip"))

    if animate and len(pos) > 2:
        _add_spiral_scrubber(fig, pos, days, n_frames, controls)

    bits = []
    if dv_kms is not None:
        bits.append(f"dV {dv_kms:.2f} km/s")
    if tof_days is not None:
        bits.append(f"{tof_days:.0f} d ({tof_days / 365.25:.2f} yr)")
    _layout(fig, title + ("  ·  " + "  ·  ".join(bits) if bits else ""))
    axis = dict(backgroundcolor=SURFACE, gridcolor=GRID, color=MUTED, title="km")
    z_axis = dict(axis, title="km off equator")
    # Look perpendicular to the orbit plane so the winding reads as a spiral; any in-plane view
    # (the default angled one) collapses the loops to a line. The belt volume reads from this angle
    # too, since it is a 3D body rather than a plane that needs facing.
    _cam_eye, _cam_up = _orbit_plane_camera(pos)
    scene = dict(xaxis=axis, yaxis=dict(axis), zaxis=z_axis, aspectmode="data",
                 uirevision="spiral", camera=dict(eye=_cam_eye, up=_cam_up))
    if max_extent_km is not None:
        # A fixed equal-sided box centered on Earth: the belts and the near-Earth winding fill the
        # frame and the far hyperbolic tail clips out, rather than the tail dictating a zoomed-out
        # view that shrinks the belt crossings to a speck.
        rng = [-extent, extent]
        scene.update(aspectmode="cube", xaxis=dict(axis, range=rng),
                     yaxis=dict(axis, range=rng), zaxis=dict(z_axis, range=rng))
    fig.update_layout(height=620, showlegend=False, scene=scene)
    return fig


def _add_spiral_scrubber(fig, pos, days, n_frames, controls) -> None:
    """Add a moving spacecraft marker (named ``scrub-marker``) along the spiral.

    With ``controls=True`` also build the animation frames + Plotly's own Play/slider chrome.
    With ``controls=False`` only the marker trace is added (no frames): an external control --
    the app's own mission timeline, moves it with ``Plotly.restyle`` on that one trace, which
    never recomputes the heavy spiral line, so a high-resolution path stays cheap to scrub."""
    def _marker(p):
        return go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers", name="scrub-marker",
                            marker=dict(size=5, color=SPACECRAFT, line=dict(color=BG, width=1)),
                            hoverinfo="skip")

    moving = [len(fig.data)]
    fig.add_trace(_marker(pos[0]))
    if not controls:
        return
    idx = np.unique(np.linspace(0, len(pos) - 1, min(n_frames, len(pos))).astype(int))
    fig.frames = [go.Frame(name=f"{days[k]:.0f}", traces=moving, data=[_marker(pos[k])])
                  for k in idx]
    steps = [dict(method="animate", label=f"{days[k]:.0f}",
                  args=[[f"{days[k]:.0f}"], dict(mode="immediate",
                        frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
             for k in idx]
    play_args = [None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True,
                            transition=dict(duration=0))]
    pill = dict(type="buttons", y=0.0, yanchor="top", xanchor="left", showactive=False,
                borderwidth=0, pad=dict(t=4, b=4, l=10, r=10), font=dict(color=BG, size=14))
    fig.update_layout(
        updatemenus=[dict(**pill, x=0.0, bgcolor=PLAY_GREEN,
                          buttons=[dict(label="▶", method="animate", args=play_args)])],
        sliders=[dict(active=0, x=0.06, len=0.9, y=0.0, yanchor="top",
                      currentvalue=dict(prefix="day ", font=dict(color=TEXT)),
                      font=dict(color=MUTED), steps=steps)])


def spiral_diagnostics(times_days, energy_km2_s2, inc_deg, mass_kg) -> go.Figure:
    """The spiral over time: the energy climb, the plane, and the propellant it costs.

    The energy crossing zero is the moment it gets free of Earth. The tilt shows any steering, or
    just the drift from Earth's bulge and the Sun and Moon if there is none. The mass shows the
    propellant going down, and where it flattens out the engine is off in eclipse or throttled back
    for lack of power.
    """
    t = np.asarray(times_days, float)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=("specific orbital energy (km²/s²) - 0 = escape",
                        "geocentric inclination (deg)", "stack mass (kg)"))
    fig.add_trace(go.Scatter(x=t, y=np.asarray(energy_km2_s2, float), mode="lines",
                             line=dict(color=DOME, width=2),
                             hovertemplate="day %{x:.0f}<br>E %{y:.2f}<extra></extra>"),
                  row=1, col=1)
    fig.add_hline(y=0.0, line=dict(color=MUTED, width=1, dash="dot"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.asarray(inc_deg, float), mode="lines",
                             line=dict(color=ASTEROID_ORANGE, width=2),
                             hovertemplate="day %{x:.0f}<br>i %{y:.2f}°<extra></extra>"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.asarray(mass_kg, float), mode="lines",
                             line=dict(color=ECCEN, width=2),
                             hovertemplate="day %{x:.0f}<br>%{y:.1f} kg<extra></extra>"),
                  row=3, col=1)
    _layout(fig, "Spiral diagnostics")
    fig.update_layout(height=560, showlegend=False)
    fig.update_xaxes(gridcolor=GRID, color=MUTED, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, color=MUTED, zeroline=False)
    fig.update_xaxes(title="days from injection", row=3, col=1)
    fig.update_annotations(font=dict(color=TEXT, size=12))
    return fig
