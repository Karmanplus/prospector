"""Heliocentric cruise figures: the flown arc, its diagnostics, and the trades around it.

The 3D trajectory colours the path by how hard the engine is running, and can carry a slider that
walks the spacecraft and both bodies along it together. The diagnostics split the thrust into its
radial, along-track and out-of-plane parts next to the orbit properties each one changes, which is
what makes "out of plane" mean something concrete.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from prospector.figures.theme import (
    ASTEROID_ORANGE,
    BG,
    DEPARTURE,
    DOME,
    EARTH,
    EARTH_BLUE,
    ECCEN,
    GRID,
    MUTED,
    PAUSE_AMBER,
    PLAY_GREEN,
    SPACECRAFT,
    SURFACE,
    TEXT,
    THRUST_SCALE,
    _layout,
)


def _phase_transition_lines(fig: go.Figure, transitions) -> None:
    """Dashed vertical markers where one phase of the mission hands over to the next.

    ``transitions`` is ``[(day, label), ...]`` on the figure's own clock. Every whole-mission
    timeline uses this, so the boundaries look the same on every chart.
    """
    for day, label in transitions or ():
        fig.add_vline(x=float(day), line=dict(color=MUTED, width=1.2, dash="dash"),
                      annotation_text=str(label), annotation_position="top",
                      annotation_font=dict(color=MUTED, size=10))


def distance_profile(times_days, sc_pos_au, earth_pos_au, target_pos_au, *,
                     sun_au=None, transitions=(),
                     target_name: str = "target",
                     title: str = "Range to Earth & target") -> go.Figure:
    """Spacecraft range to Earth and to the target across the cruise (AU vs days from departure).

    Worked out from the position tracks of the spacecraft, Earth and the target sampled at the same
    times, so nothing has to be looked up. The distance to Earth starts near zero at departure and
    grows; the distance to the target shrinks to near zero on arrival, since it is a rendezvous. 1
    AU is about 149.6 million km. ``sun_au``, optional and sampled the same way, adds the distance
    to the Sun, which is what drives the array output. ``transitions`` marks where the phases
    change (:func:`_phase_transition_lines`).
    """
    t = np.asarray(times_days, float)
    sc = np.asarray(sc_pos_au, float)
    earth = np.asarray(earth_pos_au, float)
    tgt = np.asarray(target_pos_au, float)
    n = min(len(t), len(sc), len(earth), len(tgt))
    t, sc, earth, tgt = t[:n], sc[:n], earth[:n], tgt[:n]
    d_earth = np.linalg.norm(sc - earth, axis=1)
    d_target = np.linalg.norm(sc - tgt, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=d_earth, mode="lines", name="to Earth",
        line=dict(color=EARTH_BLUE, width=2),
        hovertemplate="day %{x:.0f}<br>%{y:.3f} AU to Earth<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=t, y=d_target, mode="lines", name=f"to {target_name}",
        line=dict(color=ASTEROID_ORANGE, width=2),
        customdata=[target_name] * len(t),
        hovertemplate="day %{x:.0f}<br>%{y:.3f} AU to %{customdata}<extra></extra>"))
    if sun_au is not None:
        s = np.asarray(sun_au, float)[:n]
        fig.add_trace(go.Scatter(
            x=t[:len(s)], y=s, mode="lines", name="to Sun",
            line=dict(color=EARTH, width=2, dash="dot"),
            hovertemplate="day %{x:.0f}<br>%{y:.3f} AU from the Sun<extra></extra>"))
    _phase_transition_lines(fig, transitions)
    fig.update_xaxes(title="days from departure", gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title="distance (AU)", gridcolor=GRID, zeroline=False, rangemode="tozero")
    return _layout(fig, title)


def trajectory_3d(fine_pos_au, fine_throttle, fine_days, node_days, node_throttle,
                  earth_orbit_au, target_orbit_au, earth_track_au, target_track_au,
                  dep_label="", arr_label="", target_name="", dv_kms=None, tof_days=None,
                  target_i_deg=None, node_pos_au=None, node_thrust_vec=None, node_rtn=None,
                  show_cones=False, animate=True, n_frames=60, controls=True,
                  max_throttle_pct=None) -> go.Figure:
    """The converged low-thrust trajectory (AU) over a synced thrust profile, made readable.

    On top, the smoothed path coloured by throttle, red for coasting and green for full thrust,
    with Earth's orbit in blue and the target's in orange. The starting point is a different blue
    so it is not mistaken for Earth, and the vertical axis is to scale so tilts and thrust arrows
    show their real angles. Below, the thrust over time as a percentage. One Play button and a time
    slider move the three spheres along their paths and a cursor along the thrust profile together.

    Given ``show_cones`` and the per-node thrust vectors, which are ``node_pos_au`` for positions
    and ``node_thrust_vec`` for directions with length giving the throttle, a field of arrows is
    drawn along the path showing which way the engine is pushing. It is off by default. Arrow
    length is the throttle. Given ``node_rtn``, the radial, along-track and out-of-plane parts at
    each node, every arrow is coloured by which part of the orbit its thrust is mainly changing:
    orange out of plane for the tilt, blue along-track for the size, violet radial for the shape.
    The same colours appear in the diagnostics.
    """
    fine = np.asarray(fine_pos_au, float)
    days = np.asarray(fine_days, float)
    earth = np.asarray(earth_orbit_au, float)
    tgt = np.asarray(target_orbit_au, float)
    e_track = np.asarray(earth_track_au, float)
    t_track = np.asarray(target_track_au, float)
    nd = np.asarray(node_days, float)
    # Throttle is a fraction of the stack's PHYSICAL max thrust, which the post-escape array power
    # caps below RATED. Express it as a fraction of RATED (× the array-power ceiling) so the
    # profile and its ceiling line share one frame. Otherwise 90% of the reduced thrust reads as
    # "90%" against a 69%-of-rated ceiling and looks like a violation (see trajectory_diagnostics).
    scale = (float(max_throttle_pct) if max_throttle_pct is not None
             and 0.0 < float(max_throttle_pct) <= 100.5 else 100.0)
    thr = np.clip(np.asarray(fine_throttle, float), 0.0, 1.0) * (scale / 100.0)   # fraction of RATED
    nu = np.clip(np.asarray(node_throttle, float), 0.0, 1.0) * scale              # percent of RATED
    # The same number read operationally. Isp is constant across the solve, so the share of a
    # segment's available velocity change the optimizer took is the share of that segment the
    # engine spends firing, which is the duty cycle. It parts company with the plotted percentage
    # once the array caps thrust below rated: at an 80%-of-rated ceiling, a segment
    # plotted at 40% of rated is the engine firing half the time, not two-fifths of it.
    duty = np.clip(np.asarray(node_throttle, float), 0.0, 1.0) * 100.0
    order = np.argsort(nd)                 # node times aren't in order where the halves meet
    nd, nu, duty = nd[order], nu[order], duty[order]
    tof = float(days.max()) if days.size else 1.0

    fig = make_subplots(rows=2, cols=1, row_heights=[0.82, 0.18], vertical_spacing=0.10,
                        specs=[[{"type": "scene"}], [{"type": "xy"}]])
    # --- scene (row 1) ---
    fig.add_trace(go.Scatter3d(
        x=earth[:, 0], y=earth[:, 1], z=earth[:, 2], mode="lines", name="Earth orbit",
        line=dict(color=EARTH_BLUE, width=2), opacity=0.5, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=tgt[:, 0], y=tgt[:, 1], z=tgt[:, 2], mode="lines", name=f"{target_name} orbit",
        line=dict(color=ASTEROID_ORANGE, width=2), opacity=0.5, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers+text", marker=dict(size=5, color="#ffd24a"),
        text=["Sun"], textposition="top center", textfont=dict(color="#ffd24a"),
        hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=fine[:, 0], y=fine[:, 1], z=fine[:, 2], mode="lines", name="transfer",
        line=dict(color=thr, colorscale=THRUST_SCALE, cmin=0, cmax=1, width=7,
                  colorbar=dict(title="throttle %", x=1.0, y=0.62, len=0.5,
                                tickvals=[0, 0.25, 0.5, 0.75, 1],
                                ticktext=["0", "25", "50", "75", "100"])),
        customdata=np.column_stack([days, thr * 100]),
        hovertemplate="day %{customdata[0]:.0f}<br>throttle %{customdata[1]:.0f}%<extra></extra>"),
        row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[fine[0, 0]], y=[fine[0, 1]], z=[fine[0, 2]], mode="markers+text",
        marker=dict(size=6, color=DEPARTURE, symbol="circle", line=dict(color=BG, width=1)),
        text=[f"  depart {dep_label}"], textposition="top center",
        textfont=dict(color=DEPARTURE), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[fine[-1, 0]], y=[fine[-1, 1]], z=[fine[-1, 2]], mode="markers+text",
        marker=dict(size=6, color=ASTEROID_ORANGE, symbol="diamond", line=dict(color=BG, width=1)),
        text=[f"  arrive {arr_label}"], textposition="bottom center",
        textfont=dict(color=ASTEROID_ORANGE), hoverinfo="skip"), row=1, col=1)
    # Optional thrust-direction cones (off by default): an arrow at each thrusting node pointing
    # the way the engine pushes. Arrow LENGTH = throttle (go.Cone size = vector-norm × sizeref).
    # COLOUR = the element the thrust is mainly buying (matches the diagnostics panel): normal ->
    # inclination (orange), transverse -> semi-major axis (blue), radial -> eccentricity (violet).
    if show_cones and node_pos_au is not None and node_thrust_vec is not None:
        pos = np.asarray(node_pos_au, float)
        vec = np.asarray(node_thrust_vec, float)
        if pos.shape == vec.shape and pos.ndim == 2 and pos.shape[1] == 3:
            on = np.linalg.norm(vec, axis=1) > 0.02
            size = 1.0 * (float(np.max(np.linalg.norm(pos[:, :2], axis=1))) or 1.0)
            cone_kw = dict(anchor="tail", sizemode="absolute", sizeref=size,
                           showscale=False, hoverinfo="skip")
            rtn = np.asarray(node_rtn, float) if node_rtn is not None else None
            if rtn is not None and rtn.shape == pos.shape:
                # go.Cone can't take a per-cone colour, so draw one single-colour trace per
                # dominant steering axis (whichever |component| is largest at that node).
                dominant = np.argmax(np.abs(rtn), axis=1)        # 0 radial, 1 transverse, 2 normal
                for axis_idx, color in ((2, ASTEROID_ORANGE), (1, EARTH_BLUE), (0, ECCEN)):
                    m = on & (dominant == axis_idx)
                    if m.any():
                        fig.add_trace(go.Cone(
                            x=pos[m, 0], y=pos[m, 1], z=pos[m, 2],
                            u=vec[m, 0], v=vec[m, 1], w=vec[m, 2],
                            colorscale=[[0, color], [1, color]], **cone_kw), row=1, col=1)
            elif on.any():                                       # no RTN: fall back to one colour
                fig.add_trace(go.Cone(
                    x=pos[on, 0], y=pos[on, 1], z=pos[on, 2],
                    u=vec[on, 0], v=vec[on, 1], w=vec[on, 2],
                    colorscale=[[0, DOME], [1, DOME]], **cone_kw), row=1, col=1)
    # --- thrust profile (row 2) ---
    fig.add_trace(go.Scatter(
        x=nd, y=nu, mode="lines", line=dict(color=DOME, width=2, shape="hv"),
        fill="tozeroy", fillcolor="rgba(55,194,196,0.20)", customdata=duty,
        hovertemplate="day %{x:.0f}<br>throttle %{y:.0f}% of rated"
                      "<br>duty cycle %{customdata:.0f}% of this segment<extra></extra>"),
        row=2, col=1)
    # The array-power ceiling: the most thrust the degraded, post-escape array can power, as a
    # fraction of full thrust. Always drawn (even at 100%) so it is clear whether power is the
    # limit: a line at 100% means full thrust is available; a lower line is the power ceiling the
    # throttle cannot exceed. Drawn as a trace rather than add_hline, since the figure mixes a 3D
    # scene with this xy row.
    if max_throttle_pct is not None and 0.0 < float(max_throttle_pct) <= 100.5:
        cap = min(100.0, float(max_throttle_pct))
        label = "full power, no thrust limit" if cap >= 99.5 else f"array-power ceiling {cap:.0f}%"
        fig.add_trace(go.Scatter(
            x=[0.0, tof], y=[cap, cap], mode="lines", hoverinfo="skip", showlegend=False,
            line=dict(color=EARTH, width=1.5, dash="dash")), row=2, col=1)
        fig.add_annotation(x=0.0, y=cap, text=label, showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(color=EARTH, size=10),
                           row=2, col=1)

    have_bodies = e_track.shape[0] == fine.shape[0] and t_track.shape[0] == fine.shape[0]
    if animate and fine.shape[0] > 2:
        _add_scrubber(fig, fine, days, thr, e_track, t_track, have_bodies, n_frames,
                      controls=controls)

    bits = []
    if dv_kms is not None:
        bits.append(f"dV {dv_kms:.2f} km/s")
    if tof_days is not None:
        bits.append(f"{tof_days:.0f} d ({tof_days/365.25:.2f} yr)")
    if target_i_deg is not None:
        bits.append(f"tilt {float(target_i_deg):.1f}°")
    title = (f"{target_name} - low-thrust trajectory" if target_name
             else "Low-thrust trajectory")
    _layout(fig, title + ("  ·  " + "  ·  ".join(bits) if bits else ""))

    # Z-axis: rendered to scale (1 AU z == 1 AU x) and never exaggerated, so the tilt and every
    # thrust arrow read at their true angle. (A fixed z stretch tilts out-of-plane thrust arrows
    # several times too steep and over-states tiny inclinations.) The z RANGE is framed to the
    # orbit but never tighter than a 1deg envelope, so a sub-1deg orbit reads as small rather than
    # being zoomed up. Ticks are position (AU); the title names the true orbital tilt.
    x_axis = dict(title="x (AU)", backgroundcolor=SURFACE, gridcolor=GRID, color=MUTED)
    y_axis = dict(title="y (AU)", backgroundcolor=SURFACE, gridcolor=GRID, color=MUTED)
    z_axis = dict(title="z (AU)", backgroundcolor=SURFACE, gridcolor=GRID, color=MUTED)
    z_aspect = 0.55                          # fallback when no inclination is supplied
    if target_i_deg is not None:
        planar = np.concatenate([earth[:, :2], tgt[:, :2], fine[:, :2]])
        R = float(np.max(np.linalg.norm(planar, axis=1))) if planar.size else 1.0
        z_data = float(np.max(np.abs(np.concatenate([earth[:, 2], tgt[:, 2], fine[:, 2]]))))
        z_half = max(np.sin(np.radians(max(1.0, float(target_i_deg)))) * R, z_data, 1e-3) * 1.05
        z_axis["range"] = [-z_half, z_half]
        z_axis["title"] = f"z (AU) · orbit tilt {float(target_i_deg):.1f}° (to scale)"
        # Frame x/y tight to the orbits (just a hair of margin) so they nearly fill the borders.
        x_axis["range"] = [-R * 1.05, R * 1.05]
        y_axis["range"] = [-R * 1.05, R * 1.05]
        z_aspect = z_half / (R * 1.05)       # to scale in every case
    fig.update_layout(
        height=820, showlegend=False, margin=dict(l=10, r=10, t=46, b=70),
        # uirevision pins the user's camera across redraws, so pressing Play or dragging the time
        # slider animates the bodies without snapping the view back to default.
        scene=dict(
            xaxis=x_axis, yaxis=y_axis, zaxis=z_axis, uirevision="trajectory",
            aspectmode="manual", aspectratio=dict(x=1.0, y=1.0, z=z_aspect),
            camera=dict(eye=dict(x=1.05, y=1.05, z=0.65))))
    fig.update_xaxes(title="days from departure", range=[0, tof], gridcolor=GRID,
                     color=MUTED, zeroline=False, row=2, col=1)
    fig.update_yaxes(title="throttle (%)", range=[0, 105], gridcolor=GRID, color=MUTED,
                     zeroline=False, row=2, col=1)
    return fig


def _add_scrubber(fig, fine, days, thr, e_track, t_track, have_bodies, n_frames,
                  controls=True) -> None:
    """Add the moving spheres (3D, named ``scrub-sc`` / ``scrub-earth`` / ``scrub-ast``) + a
    thrust-profile cursor (``scrub-cursor``). With ``controls=True`` it also builds the animation
    frames and Plotly's own Play button and slider. With ``controls=False`` only the markers are
    added and no frames, because something outside, namely the app's own mission timeline, moves
    them
    with ``Plotly.restyle`` on those traces. That never redraws the heavy path, so dragging
    through a high-resolution trajectory stays cheap."""
    moving = []           # trace indices the frames update, in the order added below

    def _sphere(pos, color, name):
        return go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2]], mode="markers", name=name,
                            marker=dict(size=6, color=color, line=dict(color=BG, width=1)),
                            hoverinfo="skip")

    moving.append(len(fig.data))
    fig.add_trace(_sphere(fine[0], SPACECRAFT, "scrub-sc"), row=1, col=1)
    if have_bodies:
        moving.append(len(fig.data))
        fig.add_trace(_sphere(e_track[0], EARTH_BLUE, "scrub-earth"), row=1, col=1)
        moving.append(len(fig.data))
        fig.add_trace(_sphere(t_track[0], ASTEROID_ORANGE, "scrub-ast"), row=1, col=1)
    moving.append(len(fig.data))
    fig.add_trace(go.Scatter(x=[days[0], days[0]], y=[0, 105], mode="lines",
                             line=dict(color=SPACECRAFT, width=1.5, dash="dot"),
                             name="scrub-cursor", hoverinfo="skip"), row=2, col=1)   # the cursor

    if not controls:
        return                          # external control restyles the markers; no frames needed

    idx = np.unique(np.linspace(0, len(fine) - 1, min(n_frames, len(fine))).astype(int))
    frames = []
    for k in idx:
        data = [_sphere(fine[k], SPACECRAFT, "scrub-sc")]
        if have_bodies:
            data.append(_sphere(e_track[k], EARTH_BLUE, "scrub-earth"))
            data.append(_sphere(t_track[k], ASTEROID_ORANGE, "scrub-ast"))
        data.append(go.Scatter(x=[days[k], days[k]], y=[0, 105], mode="lines",
                               line=dict(color=SPACECRAFT, width=1.5, dash="dot")))
        frames.append(go.Frame(name=f"{days[k]:.0f}", traces=moving, data=data))
    fig.frames = frames

    steps = [dict(method="animate", label=f"{days[k]:.0f}",
                  args=[[f"{days[k]:.0f}"], dict(mode="immediate",
                        frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
             for k in idx]
    # Two single-character buttons in distinct, high-contrast colours (the default dark-on-dark
    # rendered as unreadable white-on-white). Plotly buttons are pill-shaped, not true circles.
    play_args = [None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True,
                            transition=dict(duration=0))]
    pause_args = [[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))]
    pill = dict(type="buttons", y=1.0, yanchor="top", xanchor="left", showactive=False,
                borderwidth=0, pad=dict(t=4, b=4, l=10, r=10), font=dict(color=BG, size=14))
    fig.update_layout(
        updatemenus=[
            dict(**pill, x=0.0, bgcolor=PLAY_GREEN,
                 buttons=[dict(label="▶", method="animate", args=play_args)]),
            dict(**pill, x=0.05, bgcolor=PAUSE_AMBER,
                 buttons=[dict(label="❚❚", method="animate", args=pause_args)]),
        ],
        sliders=[dict(active=0, x=0.11, len=0.85, y=0.0, yanchor="top",
                      currentvalue=dict(prefix="day ", font=dict(color=TEXT)),
                      font=dict(color=MUTED), steps=steps)])


def trajectory_diagnostics(node_days, throttle, radial, transverse, normal, i_deg, a_au, e,
                           speed_kms=None, earth_speed_dep=None, target_speed_arr=None,
                           target_a=None, target_e=None, target_i=None,
                           target_name="", max_throttle_pct=None) -> go.Figure:
    """How the trajectory is flown, node by node, for reading and checking it.

    Every panel shares the same time axis, days from departure, and they run cause before effect.
    First what the engine is doing: total throttle, then the thrust split into its parts in the
    orbit's own frame, as signed fractions of maximum thrust, showing which part of the orbit each
    burn is changing and in which direction. These components vary smoothly, with no angle wrapping
    around:

        out of plane (orange) -> tilts the orbit          -> inclination
        along-track  (blue)   -> adds or removes energy   -> orbit size
        radial       (violet) -> pushes out or in         -> orbit shape

    Then the effect: speed, if given, followed by tilt, size and shape, each with a dotted line at
    the value it is aiming for. The speed panel answers what the spacecraft left at and whether it
    ends up matching the asteroid: it starts at Earth's speed plus the departure excess and is
    driven to the asteroid's speed by arrival. Read top to bottom, a burst out of plane moves the
    tilt, an along-track burst moves the size, and a radial one moves the shape. A tilt offset that
    is there from day one came from the launch, not the engine. All of it comes straight from the
    position, velocity and throttle at each node.
    """
    nd = np.asarray(node_days, float)
    order = np.argsort(nd)                  # node times aren't in order where the halves meet
    nd = nd[order]
    # The solver's throttle + thrust components are fractions of the stack's PHYSICAL max thrust,
    # which the post-escape array power caps BELOW rated. Plot everything as a fraction of RATED
    # (scale by the array-power ceiling = thrust_N/rated), so the throttle trace and the ceiling
    # line share one frame. Otherwise a throttle at 90% of the reduced thrust reads as "90%"
    # against a 69%-of-rated ceiling and looks like it broke the ceiling when it did not.
    scale = (float(max_throttle_pct) if max_throttle_pct is not None
             and 0.0 < float(max_throttle_pct) <= 100.5 else 100.0)
    thr = np.clip(np.asarray(throttle, float)[order], 0.0, 1.0) * scale
    # Unscaled, the throttle is the share of each segment the engine spends firing (see
    # trajectory_3d): the duty cycle, which the plotted percentage only equals at full power.
    duty = np.clip(np.asarray(throttle, float)[order], 0.0, 1.0) * 100.0
    rad = np.asarray(radial, float)[order] * scale
    tan = np.asarray(transverse, float)[order] * scale
    nor = np.asarray(normal, float)[order] * scale
    inc = np.asarray(i_deg, float)[order]
    a = np.asarray(a_au, float)[order]
    ecc = np.asarray(e, float)[order]
    has_speed = speed_kms is not None

    titles = ["throttle (% of rated)", "thrust components (% of rated, signed)"]
    if has_speed:
        titles.append("heliocentric speed (km/s)")
    titles += ["inclination (°)", "semi-major axis (AU)", "eccentricity"]
    nrows = len(titles)
    fig = make_subplots(rows=nrows, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                        subplot_titles=titles)

    fig.add_trace(go.Scatter(
        x=nd, y=thr, mode="lines", line=dict(color=DOME, width=2, shape="hv"),
        fill="tozeroy", fillcolor="rgba(55,194,196,0.18)", showlegend=False, customdata=duty,
        hovertemplate="day %{x:.0f}<br>%{y:.0f}% of rated"
                      "<br>duty cycle %{customdata:.0f}% of this segment<extra></extra>"),
        row=1, col=1)
    # The array-power ceiling: the most thrust the post-escape array can power, as a fraction of
    # full thrust. Always drawn (even at 100%) so it is clear whether power is the limit: 100% =
    # full thrust available; a lower line is the power ceiling the throttle can't exceed.
    if max_throttle_pct is not None and 0.0 < float(max_throttle_pct) <= 100.5:
        cap = min(100.0, float(max_throttle_pct))
        label = "full power, no thrust limit" if cap >= 99.5 else f"array-power ceiling {cap:.0f}%"
        fig.add_hline(y=cap, line=dict(color=EARTH, width=1.5, dash="dash"),
                      annotation_text=label, annotation_position="top left",
                      annotation_font=dict(color=EARTH, size=10), row=1, col=1)
        # Keep the ceiling in view even when the throttle rides well below it (so the line doesn't
        # clip off the top of an auto-fit panel).
        fig.update_yaxes(range=[0, 105], row=1, col=1)
    # Thrust split in the orbit frame, signed, showing what each burn changes and which way. Filled
    # to zero (the "allocation" read), each colour matching the element it drives. Vector
    # components are smooth, so there is no wraparound to handle.
    for comp, color, fillc, label in (
            (nor, ASTEROID_ORANGE, "rgba(255,159,67,0.22)", "normal → i"),
            (tan, EARTH_BLUE, "rgba(74,163,255,0.22)", "transverse → a"),
            (rad, ECCEN, "rgba(197,140,240,0.22)", "radial → e")):
        fig.add_trace(go.Scatter(
            x=nd, y=comp, mode="lines", name=label, showlegend=True,
            line=dict(color=color, width=2), fill="tozeroy", fillcolor=fillc,
            hovertemplate=label + "<br>day %{x:.0f}<br>%{y:.0f}%<extra></extra>"), row=2, col=1)
    fig.add_hline(y=0.0, line=dict(color=GRID, width=1), row=2, col=1)

    row = 3
    if has_speed:
        spd = np.asarray(speed_kms, float)[order]
        fig.add_trace(go.Scatter(
            x=nd, y=spd, mode="lines+markers", line=dict(color=SPACECRAFT, width=2),
            marker=dict(size=3), showlegend=False,
            hovertemplate="day %{x:.0f}<br>%{y:.2f} km/s<extra></extra>"), row=row, col=1)
        if earth_speed_dep is not None:
            fig.add_hline(y=earth_speed_dep, line=dict(color=EARTH_BLUE, width=1, dash="dot"),
                          annotation_text="Earth @ departure", annotation_position="bottom left",
                          annotation_font=dict(color=EARTH_BLUE, size=10), row=row, col=1)
        if target_speed_arr is not None:
            fig.add_hline(y=target_speed_arr, line=dict(color=ASTEROID_ORANGE, width=1, dash="dot"),
                          annotation_text="asteroid @ arrival", annotation_position="top left",
                          annotation_font=dict(color=ASTEROID_ORANGE, size=10), row=row, col=1)
        row += 1

    fig.add_trace(go.Scatter(
        x=nd, y=inc, mode="lines+markers", line=dict(color=ASTEROID_ORANGE, width=2),
        marker=dict(size=3), showlegend=False,
        hovertemplate="day %{x:.0f}<br>i %{y:.3f}°<extra></extra>"), row=row, col=1)
    if target_i is not None:
        fig.add_hline(y=target_i, line=dict(color=ASTEROID_ORANGE, width=1, dash="dot"), row=row, col=1)
    row += 1
    fig.add_trace(go.Scatter(
        x=nd, y=a, mode="lines+markers", line=dict(color=EARTH_BLUE, width=2),
        marker=dict(size=3), showlegend=False,
        hovertemplate="day %{x:.0f}<br>a %{y:.4f} AU<extra></extra>"), row=row, col=1)
    if target_a is not None:
        fig.add_hline(y=target_a, line=dict(color=EARTH_BLUE, width=1, dash="dot"), row=row, col=1)
    row += 1
    fig.add_trace(go.Scatter(
        x=nd, y=ecc, mode="lines+markers", line=dict(color=ECCEN, width=2),
        marker=dict(size=3), showlegend=False,
        hovertemplate="day %{x:.0f}<br>e %{y:.4f}<extra></extra>"), row=row, col=1)
    if target_e is not None:
        fig.add_hline(y=target_e, line=dict(color=ECCEN, width=1, dash="dot"), row=row, col=1)

    title = (f"{target_name} - trajectory diagnostics" if target_name
             else "Trajectory diagnostics")
    _layout(fig, title + "  ·  dotted = target")
    # Place the legend beside the steering-angle panel (row 2) it labels, not over row 1.
    panel_h = (1.0 - 0.035 * (nrows - 1)) / nrows
    row2_center = 1.0 - 1.5 * panel_h - 0.035
    fig.update_layout(height=150 * nrows + 60, margin=dict(l=10, r=140, t=46, b=10),
                      legend=dict(orientation="v", x=1.01, xanchor="left", y=row2_center,
                                  yanchor="middle", bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    fig.update_xaxes(gridcolor=GRID, color=MUTED, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, color=MUTED, zeroline=False)
    fig.update_xaxes(title="days from departure", row=nrows, col=1)
    fig.update_annotations(font=dict(color=TEXT, size=12))   # subplot titles -> readable
    return fig
