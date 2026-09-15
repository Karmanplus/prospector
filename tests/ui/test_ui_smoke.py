"""Headless smoke test for the NiceGUI app's Find-targets render path.

Uses NiceGUI's pure-Python ``User`` simulation (no browser) to build the page with a project open
and the Find-targets workspace showing a (pre-computed, in-memory) reachable set -- asserting the
whole render path (top bar, canvas with the reachability dome, the targets dock) constructs without
raising. State is seeded directly so the test exercises rendering deterministically, without the
live async fetch or the detached characterization subprocess (the data pipeline and the physics
have their own tests).
"""
import json
import os
import time
from datetime import date

import pandas as pd
import pytest
from nicegui import ui
from nicegui.testing import User

from prospector import enrichment, paths
from prospector.config import Mission
from prospector.solvers import edelbaum
from tests import library as lib
from ui import state
from ui.state import S


def test_design_id_is_unique_across_a_model_param_sweep():
    """A --sweep-param run yields rows identical but for the swept knob; the design_id must
    fold the param in, or they collide on one row key, the bug that made a row click
    highlight the whole group and corrupted the grid's sort/selection tracking."""
    from ui import vehicle_search as vs
    df = pd.DataFrame({
        "engine": ["e", "e", "e"], "propellant_key": ["xenon"] * 3,
        "dry_kg": [170.0] * 3, "prop_kg": [200.0] * 3, "n_engines": [4] * 3,
        "param_power_margin_pct": [10.0, 20.0, 30.0],   # same vehicle, three solar-margin points
    })
    assert vs.with_design_id(df)["design_id"].nunique() == 3
    # With no swept param the identity is unchanged (back-compat with non-sweep sessions).
    plain = vs.with_design_id(df.drop(columns=["param_power_margin_pct"]))
    assert plain["design_id"].iloc[0] == "e|xenon|170.0|200.0|4"


def test_sweep_form_counts_the_same_axis_the_search_builds():
    """The form quotes how many trajectory solves a sweep will run, which is only useful if it
    counts the axis the search actually builds. Both walk lo..hi inclusive at the given step, so
    they have to agree exactly, or the form under-quotes a run that then takes twice as long.
    """
    import numpy as np

    from ui.workspaces.vehicle import _axis_steps

    for lo, hi, step in [(150, 350, 25), (150, 350, 10), (0, 400, 100), (200, 200, 25),
                         (50, 1000, 7), (150, 355, 25)]:
        built = np.arange(lo, hi + step / 2, step)
        assert _axis_steps(lo, hi, step) == len(built), f"{lo}..{hi} by {step}"
    assert _axis_steps(150, 350, 0) == 1, "a zero step is one point, not a division by zero"


def test_design_slug_is_deterministic_and_referenceable():
    """A design's memorable name is a pure function of its identity: the same design_id always
    maps to the same adjective-noun name, so a good config stays referenceable across table
    reloads and re-runs without ever being persisted."""
    from ui import vehicle_search as vs
    name = vs.design_slug("e|xenon|170.0|200.0|4")
    assert name == vs.design_slug("e|xenon|170.0|200.0|4")   # stable across calls
    assert name.count("-") == 1 and all(name.split("-"))      # adjective-noun shape
    # with_design_id attaches a per-design name, unique across distinct designs.
    df = pd.DataFrame({
        "engine": ["e", "e", "f"], "propellant_key": ["xenon", "krypton", "xenon"],
        "dry_kg": [170.0, 170.0, 220.0], "prop_kg": [200.0, 200.0, 245.0],
        "n_engines": [4, 4, 2]})
    out = vs.with_design_id(df)
    assert "slug" in out.columns and out["slug"].nunique() == len(out)


def test_design_slugs_breaks_collisions_stably():
    """When two DISTINCT designs hash to the same base name, the session mapping keeps them
    distinct with a numeric suffix, assigned order-independently so a later evaluation never
    renames an earlier one as rows stream in."""
    from collections import defaultdict

    from ui import vehicle_search as vs
    buckets: dict[str, list] = defaultdict(list)
    for i in range(400):                       # enough distinct ids to force a base-name collision
        did = f"eng|xenon|{i}.0|100.0|1"
        buckets[vs.design_slug(did)].append(did)
    collide = next(ids for ids in buckets.values() if len(ids) >= 2)[:2]
    m1 = vs.design_slugs(collide)
    m2 = vs.design_slugs(list(reversed(collide)))
    assert m1 == m2                            # order-independent -> a design's name is stable
    assert len(set(m1.values())) == 2          # distinct names despite the base collision


def test_power_stat_is_array_power_not_engine_draw():
    """The vehicle's power headline is the SOLAR ARRAY's BOL power (what limits a power-limited
    SEP thruster), with its sizing margin in the label -- not the engine stack's summed rated
    draw, which is not a useful figure. A 0 W (unmodeled) array reads as '—', never '0.0 kW'."""
    from types import SimpleNamespace

    from prospector.config import EngineMount, Vehicle
    from ui import components as c
    v = Vehicle(name="x", dry_mass=200, fuel_mass=300, solar_power_W=8500.0,
                array_margin_pct=55.0, engines=[EngineMount(type="e", count=2)])
    rc = SimpleNamespace(vehicle=v, total_thrust_mN=1000.0, effective_isp=2000.0,
                         total_dv_capability=9.0, usable_propellant_kg=300.0,
                         escape_propellant_kg=40.0, cruise_start_mass_kg=460.0)
    drive = dict(c.vehicle_drive_rows(rc))
    assert "Power" not in drive                       # the engine-stack draw stat is gone
    array_label = next(lbl for lbl in drive if lbl.startswith("Array power"))
    assert "+55% margin" in array_label and drive[array_label] == "8.5 kW"
    # A 0 W array (not modeled) reads as em-dash, and with no explicit margin the label is bare.
    v0 = v.model_copy(update={"solar_power_W": 0.0, "array_margin_pct": None})
    drive0 = dict(c.vehicle_drive_rows(SimpleNamespace(**{**rc.__dict__, "vehicle": v0})))
    assert drive0["Array power"] == "—"


def test_adopting_a_trade_names_the_vehicle_by_its_slug():
    """A swept design carries its memorable name onto the working vehicle (and thus its saved
    filename); a legacy row without one falls back to the numeric spec name."""
    from ui.workspaces.vehicle import _vehicle_from_trade
    named = _vehicle_from_trade(
        {"engine": "e", "dry_kg": 200.0, "prop_kg": 200.0, "n_engines": 1,
         "slug": "gilded-lynx"}, {})
    assert named.name == "gilded-lynx"
    legacy = _vehicle_from_trade(
        {"engine": "e", "dry_kg": 200.0, "prop_kg": 200.0, "n_engines": 1}, {})
    assert legacy.name.startswith("trade ")


def test_adopting_a_trade_carries_its_sized_bol_array():
    """Adopting a swept design for trajectory planning must copy its SIZED beginning-of-life
    array power onto the vehicle. Without it solar_power_W defaults to 0, which the escape
    spiral reads as power-rich (no degradation), so the array reads as 0 in the report and
    belt-degradation throttling silently vanishes from the trajectory."""
    from prospector.spacecraft import buildability
    from ui.workspaces.vehicle import _vehicle_from_trade
    eng = lib.engine_key()
    rec = {"engine": eng, "dry_kg": 220.0, "prop_kg": 245.0,
           "n_engines": 4, "propellant_key": "xenon", "bol_power_W": 5979.0}
    v = _vehicle_from_trade(rec, {})
    assert v.solar_power_W == 5979.0
    assert v.dry_mass == 220.0 and v.fuel_mass == 245.0
    assert [(m.type, m.count) for m in v.engines] == [(eng, 4)]
    # The drag/SRP area is sized from the same array (the SEP bus's dominant surface), so the
    # editor's drag field is populated too, not left at 0.
    assert v.area_m2 > 0.0 and v.area_m2 == pytest.approx(buildability.array_area_m2(5979.0))
    # A swept-margin design carries the margin it was sized at, so the vehicle page's array
    # derivation reproduces its BOL power; a design that didn't sweep margin leaves it None (= the
    # build model's default).
    assert v.array_margin_pct is None
    swept = _vehicle_from_trade({**rec, "param_power_margin_pct": 15.0}, {})
    assert swept.array_margin_pct == 15.0
    # A legacy row without bol_power_W stays at 0 power and 0 area, the "power-rich"/unmodeled
    # mode, not a crash.
    legacy = _vehicle_from_trade(
        {"engine": lib.engine_key(), "dry_kg": 200.0, "prop_kg": 200.0, "n_engines": 1}, {})
    assert legacy.solar_power_W == 0.0 and legacy.area_m2 == 0.0


def test_vehicle_page_array_derivation_matches_core_sizing():
    """The vehicle page sizes the array from the margin via the same chain as the core
    buildability sizing, so the number shown/derived on the page equals required_bol_W and the
    solar_power_W written onto the vehicle for the spiral to fly."""
    from prospector.config import EngineMount, Vehicle
    from prospector.spacecraft import buildability
    from prospector.spacecraft.buildability import array_sizing_terms, with_sized_array
    from prospector.spacecraft.propulsion import load_engines
    v = Vehicle(name="t", dry_mass=200.0, fuel_mass=225.0,
                engines=[EngineMount(type=lib.engine_key(), count=2)], array_margin_pct=10.0)
    model, cat = buildability.load_bus_model(), load_engines()
    terms = array_sizing_terms(v, cat, model)
    thrusters_W = cat[lib.engine_key()].power_W * 2
    # Including the conversion chain the mounted engines' PPU wiring implies; the page must size
    # against the same chain the spiral will fly on, whichever side those engines are wired to.
    chain = buildability.assembly_thruster_chain_eff(v.mounts, cat, model)
    assert terms["bol"] == pytest.approx(
        buildability.required_bol_W(1, thrusters_W, model, margin_pct=10.0,
                                    thruster_chain_eff=chain), rel=1e-9)
    assert terms["thruster_eff"] == pytest.approx(chain, rel=1e-12)
    # Writing it onto the vehicle stores that array power, rounded, for the spiral to read.
    sized = with_sized_array(v, cat, model)
    assert sized.solar_power_W == pytest.approx(round(terms["bol"], 1))
    assert sized.area_m2 > 0.0


def test_engine_power_curve_paste_parses():
    """The engine editor lets you paste a datasheet power/thrust/Isp table. The parser must skip a
    header line, accept comma OR whitespace separators, sort by power, and reject malformed input --
    so a table copied straight from a spec sheet becomes a valid throttle curve."""
    from ui.workspaces.project import _parse_power_curve

    pts, msg = _parse_power_curve("Power\tThrust\tIsp\n1000 60 1500\n400, 20, 1200\n650, 40, 1400")
    assert pts is not None and [p.power_W for p in pts] == [400.0, 650.0, 1000.0]   # header skipped, sorted
    assert [p.isp_s for p in pts] == [1200.0, 1400.0, 1500.0]
    assert "3 points" in msg

    assert _parse_power_curve("") == (None, "")                    # blank -> no curve (constant Isp)
    assert _parse_power_curve("   \n  ") == (None, "")
    assert _parse_power_curve("400, 20, 1200")[0] is None        # a single point is not a curve
    assert _parse_power_curve("400, 20")[0] is None              # wrong column count
    assert _parse_power_curve("400 1 1500\n400 2 1500")[0] is None  # duplicate power point


def test_engine_power_curve_paste_survives_how_spreadsheets_actually_export():
    """A real datasheet paste keeps its units and whatever separator the spreadsheet chose.

    Each of these came back as "no curve" once: the row failed to split into three floats, the
    header-skip rule swallowed it, and every row failing meant an empty result that is
    indistinguishable from a deliberately blank box."""
    from ui.workspaces.project import _parse_power_curve

    for label, text in (
        ("units kept in the cells", "4000 W, 120 mN, 3200 s\n7000 W, 210 mN, 3700 s"),
        ("semicolons, as a comma-decimal locale exports", "4000; 120; 3200\n7000; 210; 3700"),
        ("thousands separators", "4,000, 120, 3200\n7,000, 210, 3700"),
        ("a title line carrying a part number", "HT-2200 performance\n4000, 120, 3200\n7000, 210, 3700"),
    ):
        pts, msg = _parse_power_curve(text)
        assert pts is not None, f"{label}: {msg}"
        assert [p.power_W for p in pts] == [4000.0, 7000.0], label
        assert [p.thrust_mN for p in pts] == [120.0, 210.0], label
        assert [p.isp_s for p in pts] == [3200.0, 3700.0], label


def test_engine_power_curve_never_reports_no_curve_for_text_that_was_pasted():
    """Text in the box that yields nothing must say so, not read as "no curve".

    The box keeps showing what was pasted, so a silent empty result looks like the engine has
    no curve rather than like the paste was not understood, and the engine then saves with a
    curve the user believes they entered, flying at constant Isp instead."""
    from ui.workspaces.project import _parse_power_curve

    for text in ("hello there\nsome notes", "n/a", "see attached sheet"):
        pts, msg = _parse_power_curve(text)
        assert pts is None and msg, f"{text!r} was accepted silently"

    # A blank box is the one case that stays quiet: it means constant Isp on purpose.
    assert _parse_power_curve("") == (None, "")
    assert _parse_power_curve("  \n\t ") == (None, "")

    # A row that is data but the wrong width still names itself, so the offending line is findable.
    pts, msg = _parse_power_curve("4000, 120, 3200\n7000, 210, 3700, 42")
    assert pts is None and "7000, 210, 3700, 42" in msg


@pytest.mark.skipif(os.name != "posix", reason="signal-0 liveness probe is POSIX-only")
def test_crashed_sweep_reads_as_stopped_via_dead_pid(tmp_path):
    """A sweep that dies mid-run leaves status.json saying 'running'. Rather than wait out the
    15-minute staleness window (during which the Compare-vehicles page treats it as live and
    rebuilds every poll tick), a recorded-but-dead PID reads as 'stopped' immediately -- while a
    live PID keeps the session 'running' even when its status stamp is fresh."""
    import subprocess
    import sys

    from ui import vehicle_search as vs

    status = {"state": "running", "n_evaluated": 0}

    # A live process (our own PID) with a fresh stamp stays running.
    alive = tmp_path / "alive"
    alive.mkdir()
    (alive / "status.json").write_text("{}")           # fresh mtime
    (alive / "pid").write_text(str(os.getpid()))
    assert vs.session_state(alive, status) == "running"

    # A dead PID reads as stopped immediately, no staleness needed. Spawn a trivial child and reap
    # it so its PID is gone.
    proc = subprocess.Popen([sys.executable, "-c", ""])
    proc.wait()
    dead = tmp_path / "dead"
    dead.mkdir()
    (dead / "status.json").write_text("{}")            # fresh mtime -> staleness would not fire
    (dead / "pid").write_text(str(proc.pid))
    assert vs.session_state(dead, status) == "stopped"

    # No PID file (a session predating PID recording): the mtime staleness guard still applies.
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "status.json").write_text("{}")
    assert vs.session_state(legacy, status) == "running"          # fresh -> still running
    old = time.time() - (vs.STALE_S + 60)
    os.utime(legacy / "status.json", (old, old))
    assert vs.session_state(legacy, status) == "stopped"          # stale -> stopped


def test_cruise_run_clears_when_the_config_changes():
    """A finished cruise run is shown only while it still matches the live config: editing the
    vehicle (mass or engine) or moving the departure v-infinity marks the shown trajectory
    stale so the window clears, rather than lingering with a 'needs re-run' warning while its
    total-mission numbers silently mix with a re-flown escape."""
    from prospector import jobs
    from prospector.config import EngineMount, Mission, ResolvedConfig, Screening, Vehicle
    from prospector.launch import LaunchOrbit
    from prospector.spacecraft.propulsion import Engine
    from ui.workspaces import flight

    cat = {"E": Engine(name="E", isp_s=1800, thrust_mN=250, power_W=5000)}
    launches = {"LV": LaunchOrbit(name="lv", perigee_alt_km=200, apogee_alt_km=200,
                                  inclination_deg=28.5, escape_provided=True)}
    mis = Mission(launch_orbit="LV", launch_window=(date(2027, 1, 1), date(2027, 3, 1)),
                  arrive_by=date(2029, 6, 1))
    veh = Vehicle(name="A", dry_mass=600, fuel_mass=400,
                  engines=[EngineMount(type="E", count=1)], solar_power_W=5000)
    rc = ResolvedConfig.build(mis, veh, Screening(), cat, launches=launches).model_copy(
        update={"departure_vinf_kms": 1.0})
    run_id = jobs.record_run(rc.model_dump(mode="json"),
                             {"pdes": "X", "a": 1.2, "e": 0.1, "i": 5, "om": 0, "w": 0,
                              "ma": 0, "epoch": 2461000.5},
                             9000.0, 9300.0, options={"available_power_W": None})
    try:
        assert flight._run_matches_config(run_id, rc)                       # its own config
        heavier = ResolvedConfig.build(mis, veh.model_copy(update={"dry_mass": 700.0}),
                                       Screening(), cat, launches=launches).model_copy(
            update={"departure_vinf_kms": 1.0})
        assert not flight._run_matches_config(run_id, heavier)              # vehicle mass edit
        cat2 = {"E": Engine(name="E", isp_s=2000, thrust_mN=300, power_W=6000)}
        rebuilt_engine = ResolvedConfig.build(mis, veh, Screening(), cat2, launches=launches).model_copy(
            update={"departure_vinf_kms": 1.0})
        assert not flight._run_matches_config(run_id, rebuilt_engine)       # engine swap
        assert not flight._run_matches_config(run_id, rc.model_copy(
            update={"departure_vinf_kms": 2.0}))                            # v-infinity change
    finally:
        import shutil
        shutil.rmtree(jobs.run_dir(run_id), ignore_errors=True)


def test_timeline_play_ignores_slider_echo():
    """During play the tick is the single clock for the playhead. The slider's own on_change --
    including the client echoing the tick's programmatic value sets back a frame or two stale --
    must not move S.timeline_t, or the thumb jumps around and the plot marker/date readout
    freeze on the bouncing value. Paused, a manual scrub still moves it."""
    from ui.state import S
    from ui.workspaces import flight
    prev_playing, prev_t = S.timeline_playing, S.timeline_t
    try:
        S.timeline_playing = True
        S.timeline_t = 0.5
        flight._set_t(0.2)              # a stale slider echo arriving mid-play
        assert S.timeline_t == 0.5      # ignored: the tick owns the playhead
        S.timeline_playing = False
        flight._set_t(0.3)              # paused: a manual scrub moves it
        assert S.timeline_t == 0.3
    finally:
        S.timeline_playing, S.timeline_t = prev_playing, prev_t


def test_typing_a_grid_axis_count_never_rebuilds_the_rail_it_is_typed_into(monkeypatch):
    """The axis inputs fire on every keystroke, so a handler that rebuilds the rail replaces the
    box mid-word: focus is lost after one character and the half-typed value is painted back over.
    Neither count is drawn anywhere -- both are read when Run is pressed -- so nothing needs a
    redraw. The colour is what the field is drawn by, so that one refreshes the canvas.
    """
    from ui.state import S
    from ui.workspaces import flight
    prev = (S.grid_n_dep, S.grid_n_tof, S.grid_colour, S.grid_fast)
    from ui import topbar
    hits = []
    for name in ("_rail", "_canvas", "_timeline"):
        monkeypatch.setattr(getattr(flight, name), "refresh",
                            lambda n=name, **_: hits.append(n))
    monkeypatch.setattr(topbar.header, "refresh", lambda **_: hits.append("topbar"))
    try:
        for ch in (1, 16):              # "1", then "16": what typing 16 actually sends
            flight._set_grid(n_dep=ch)
        flight._set_grid(n_tof=8)
        flight._set_grid(fast=True)
        assert hits == [], f"typing rebuilt {hits}"
        assert S.grid_n_dep == 16       # the finished number, not the first digit
        assert S.grid_n_tof == 8
        flight._set_grid(colour="mass")
        assert hits == ["_canvas"], hits
        # Held to what the inputs offer, so a fat-fingered 99 cannot launch a 9801-cell grid.
        flight._set_grid(n_dep=99, n_tof=0)
        assert S.grid_n_dep == 24
        assert S.grid_n_tof == 8        # 0 is not a value; the last good one stands
    finally:
        S.grid_n_dep, S.grid_n_tof, S.grid_colour, S.grid_fast = prev


_FAKE_POP = pd.DataFrame({
    "spkid": [2099942, 2101955], "pdes": ["99942", "101955"],
    "full_name": [" 99942 Apophis (2004 MN4)", " 101955 Bennu"],
    "H": [19.7, 20.2], "condition_code": [0, 0],
    "a": [0.922, 1.126], "e": [0.191, 0.204], "i": [3.34, 6.03],
    "om": [204.4, 2.06], "w": [126.4, 66.2], "ma": [180.0, 101.7], "epoch": [2461000.5, 2461000.5],
})


@pytest.mark.anyio
@pytest.mark.nicegui_main_file("ui/app.py")
async def test_find_targets_renders(user: User) -> None:
    projects = state.available_projects()
    assert projects, "no studies in configs/studies to open"

    # Seed an open project with a pre-screened reachable set (skips the async auto-screen and the
    # characterization subprocess: the render path is what's under test here).
    state.open_project(projects[0])
    df = edelbaum.evaluate_dataframe(_FAKE_POP.copy(), dv_budget=20.0)
    S.population, S.population_h = _FAKE_POP.copy(), round(S.screening.h_max, 1)
    S.screen_df = enrichment.apply_selection(df, S.desirability)
    S.enrich_reachable = None

    await user.open("/")
    await user.should_see("Find targets")        # the workspace tabs (a project is open)
    await user.should_see("Reachable targets")   # the bottom targets dock

    # The reachable set reached the targets table (cell data lives in the table's rows prop, not as
    # discrete DOM elements, so assert on the prop rather than via should_see).
    table = next(iter(user.find(ui.table).elements))
    names = [str(r.get("full_name", "")) for r in table.rows]
    assert any("Apophis" in n for n in names), names
    assert len(table.rows) == 2


@pytest.mark.anyio
@pytest.mark.nicegui_main_file("ui/app.py")
async def test_project_config_renders(user: User) -> None:
    """The Project workspace builds its full mission/vehicle editors + derived stats, and the
    launch-type table is populated from the launch library -- without touching disk."""
    projects = state.available_projects()
    assert projects, "no studies in configs/studies to open"
    state.open_project(projects[0])
    S.workspace = "project"
    # Force a return trip so the conditional return-leg fields render too (the default mission is
    # one-way, leaving that branch otherwise uncovered).
    S.mission = S.mission.model_copy(update={
        "return_trip": True, "return_by": S.mission.arrive_by})

    await user.open("/")
    await user.should_see("How it leaves Earth")   # the mission editor section heading
    await user.should_see("Calculated")            # the derived-stats strip
    await user.should_see("Payload collected (kg)")  # a return-leg field (toggle is on)

    # The launch-type table is the (only) table on this workspace, fed from the launch library.
    table = next(iter(user.find(ui.table).elements))
    keys = {r["key"] for r in table.rows}
    assert {"LEO", "TLI"} <= keys, keys
    # The open study's launch type starts selected.
    assert table.selected and table.selected[0]["key"] == S.mission.launch_orbit


def test_curve_from_triples_holds_both_editors_to_the_same_rules() -> None:
    """The row editor and the paste box share one validator, so a curve entered either way is
    accepted, rejected and summarised identically."""
    from ui.workspaces.project import _curve_from_triples, _parse_power_curve

    pts, msg = _curve_from_triples([(7000.0, 210.0, 3700.0), (4000.0, 120.0, 3200.0)])
    assert [p.power_W for p in pts] == [4000.0, 7000.0]      # sorted by power, not entry order
    assert "2 points" in msg

    # The same curve pasted as text produces the same points and the same summary.
    pasted, pasted_msg = _parse_power_curve("7000, 210, 3700\n4000, 120, 3200")
    assert [p.power_W for p in pasted] == [p.power_W for p in pts]
    assert pasted_msg == msg

    assert _curve_from_triples([]) == (None, "")                          # no rows -> no curve
    assert _curve_from_triples([(4000.0, 120.0, 3200.0)])[0] is None      # one point is not a curve
    assert _curve_from_triples([(4000.0, 120.0, 3200.0)] * 2)[0] is None  # duplicate power
    for bad in ((0.0, 120.0, 3200.0), (4000.0, -1.0, 3200.0), (4000.0, 120.0, 0.0)):
        pts, msg = _curve_from_triples([bad, (7000.0, 210.0, 3700.0)])
        assert pts is None and msg, bad


@pytest.mark.anyio
@pytest.mark.nicegui_main_file("ui/app.py")
async def test_engine_library_curve_is_an_editable_row_per_point(user: User) -> None:
    """The throttle curve is a row of number fields per operating point, not a text blob.

    Builds the real dialog, which is where a closure defined out of order (the status label is
    created after the rows that report into it) would raise rather than merely look wrong."""
    projects = state.available_projects()
    assert projects, "no studies in configs/studies to open"
    state.open_project(projects[0])
    S.workspace = "project"

    await user.open("/")
    user.find("Engine library").click()
    await user.should_see("Power throttle curve (optional)")
    await user.should_see("Add point")             # rows can be added, as elsewhere in the app
    await user.should_see("Paste a datasheet table")   # the fast path in is kept, tucked away

    # The status line describes whatever the rows currently hold.
    await user.should_see("no curve")


def test_curve_rows_add_edit_and_remove() -> None:
    """The row state, driven directly.

    Kept off the User harness: its deferred refresh means a click that adds a row is not
    observable as new fields on the next tick, which says nothing about whether the row was
    added. The rendering has its own test above; this pins the behaviour."""
    import ui.workspaces.project as proj

    # The rows and their label are module state; drop the label so this test cannot reach a widget
    # belonging to a client another test already tore down.
    proj._curve_status = None
    proj._set_curve_rows([])
    assert proj._curve_triples() == []
    assert proj._curve_from_triples(proj._curve_triples()) == (None, "")   # no rows -> no curve

    proj._add_curve_point()
    proj._add_curve_point()
    assert len(proj._curve_rows) == 2
    # A row still entirely blank is not yet a point, so a half-built curve is not an error.
    assert proj._curve_triples() == []

    for i, (p_W, t_mN, isp) in enumerate(((7000.0, 210.0, 3700.0), (4000.0, 120.0, 3200.0))):
        proj._set_curve_point(i, "p", p_W)
        proj._set_curve_point(i, "t", t_mN)
        proj._set_curve_point(i, "i", isp)
    pts, msg = proj._curve_from_triples(proj._curve_triples())
    assert [pt.power_W for pt in pts] == [4000.0, 7000.0]      # sorted, whatever order entered
    assert "2 points" in msg

    # Removing one leaves a single point, which is not a curve, and says so.
    proj._remove_curve_point(0)
    assert len(proj._curve_rows) == 1
    pts, msg = proj._curve_from_triples(proj._curve_triples())
    assert pts is None and "at least 2 points" in msg

    # A partly-filled row is reported rather than silently dropped.
    proj._set_curve_rows([])
    proj._add_curve_point()
    proj._set_curve_point(0, "p", 4000.0)
    pts, msg = proj._curve_from_triples(proj._curve_triples())
    assert pts is None and msg, "a half-filled row must not vanish"
    proj._set_curve_rows([])


def test_enabling_return_trip_seeds_a_valid_mission(monkeypatch) -> None:
    """Flipping the Return-trip switch must update the working mission even though the
    return-leg fields are built by a refreshable that hasn't rendered yet on that tick.

    The toggle's ``_apply_mission`` runs before the return widgets exist in ``_w``, so the fields
    fall back to the prior mission (return_by to arrive_by). Without that fallback the build failed
    "return_trip is set but return_by is missing", the editor swallowed it, and the toggle silently
    dropped -- which is the bug this pins. Driven directly (no NiceGUI client) so it's
    deterministic and free of the User harness's deferred-refresh race.
    """
    from types import SimpleNamespace

    import ui.workspaces.project as proj

    one_way = Mission(name="m", launch_orbit="TLI",
                      launch_window=(date(2028, 1, 1), date(2028, 3, 31)),
                      arrive_by=date(2029, 3, 1), return_trip=False)
    S.mission = one_way
    proj._launch_orbit = one_way.launch_orbit
    proj._mission_err = None                              # _show_error returns early on None
    monkeypatch.setattr(proj, "_mark_dirty", lambda: None)
    monkeypatch.setattr(proj._stats, "refresh", lambda *a, **k: None)
    # Only the base widgets + the return switch (ON); the return-leg widgets are not in _w yet, as
    # on the toggle tick before the refreshable rebuilds.
    proj._w = {
        "mis_name": SimpleNamespace(value="m"),
        "mis_open": SimpleNamespace(value="2028-01-01"),
        "mis_close": SimpleNamespace(value="2028-03-31"),
        "mis_arrive": SimpleNamespace(value="2029-03-01"),
        "mis_return": SimpleNamespace(value=True),
    }
    proj._apply_mission()
    assert S.mission.return_trip is True                  # the toggle took effect
    assert S.mission.return_by is not None                # seeded valid, so it resolves/saves
    assert S.mission.return_destination                   # a destination is set


def test_editing_one_mission_date_carries_the_others() -> None:
    """A mission date edit must land, whatever order the three are edited in.

    The window's ends and the arrival are ordered by definition, so editing one passes through a
    combination ``Mission`` refuses, pushing the window out a year inverts it until the closing
    date is edited too. That used to be reported inline and the whole edit dropped, which read as
    the date "resetting to what it was": the field kept the typed text while the model kept the old
    value, so the old date came back the next time the panel was built.

    The edited date is authoritative and the ones after it move with it, keeping the window's
    length and the transfer duration -- moving a mission later moves all of it."""
    import ui.workspaces.project as proj

    base = Mission(name="m", launch_orbit="TLI",
                   launch_window=(date(2028, 1, 1), date(2028, 3, 31)),
                   arrive_by=date(2029, 3, 1), return_trip=False)
    window_days = (base.launch_window[1] - base.launch_window[0]).days
    transfer_days = (base.arrive_by - base.launch_window[1]).days

    # Pushing the opening date out two years carries the window and the transfer with it.
    opens, closes, arrives, _, moved = proj._reconcile_mission_dates(
        base, date(2030, 1, 1), base.launch_window[1], base.arrive_by, None, "mis_open")
    assert opens == date(2030, 1, 1), "the edited date must be kept"
    assert (closes - opens).days == window_days, "the window kept its length"
    assert (arrives - closes).days == transfer_days, "the transfer kept its duration"
    assert moved and Mission(**{**base.model_dump(), "launch_window": (opens, closes),
                                "arrive_by": arrives})

    # Pushing the closing date past the arrival carries the arrival, not the opening.
    opens, closes, arrives, _, moved = proj._reconcile_mission_dates(
        base, base.launch_window[0], date(2029, 6, 1), base.arrive_by, None, "mis_close")
    assert (opens, closes) == (base.launch_window[0], date(2029, 6, 1))
    assert arrives > closes and "arrive by" in moved

    # Pulling the arrival in front of the window close is a deadline, so the window closes at it
    # rather than sliding backwards, the opening date the user did not touch stays put.
    opens, closes, arrives, _, moved = proj._reconcile_mission_dates(
        base, base.launch_window[0], base.launch_window[1], date(2028, 2, 1), None, "mis_arrive")
    assert (opens, closes, arrives) == (date(2028, 1, 1), date(2028, 2, 1), date(2028, 2, 1))
    assert "launch closes" in moved

    # An edit that is already consistent moves nothing and reports nothing.
    out = proj._reconcile_mission_dates(
        base, base.launch_window[0], date(2028, 2, 15), base.arrive_by, None, "mis_close")
    assert out == (date(2028, 1, 1), date(2028, 2, 15), date(2029, 3, 1), None, [])

    # A return leg follows the arrival wherever it goes.
    *_, returns, moved = proj._reconcile_mission_dates(
        base, date(2030, 1, 1), base.launch_window[1], base.arrive_by, date(2029, 9, 1),
        "mis_open")
    assert returns >= date(2031, 3, 1) and "return by" in moved


def test_mission_date_edit_reaches_the_model_and_the_other_fields(monkeypatch) -> None:
    """Driven through ``_apply_mission``: the edit reaches ``S.mission``, and the fields that
    moved are written back so what is shown matches what is held. The field being typed into is
    never rewritten -- doing that mid-keystroke would fight the typing."""
    from types import SimpleNamespace

    import ui.workspaces.project as proj

    S.mission = Mission(name="m", launch_orbit="TLI",
                        launch_window=(date(2028, 1, 1), date(2028, 3, 31)),
                        arrive_by=date(2029, 3, 1), return_trip=False)
    proj._launch_orbit = "TLI"
    proj._mission_err = None
    monkeypatch.setattr(proj, "_mark_dirty", lambda: None)
    monkeypatch.setattr(proj._stats, "refresh", lambda *a, **k: None)
    monkeypatch.setattr(proj.ui, "notify", lambda *a, **k: None)
    proj._w = {"mis_name": SimpleNamespace(value="m"),
               "mis_open": SimpleNamespace(value="2030-01-01"),      # typed: two years later
               "mis_close": SimpleNamespace(value="2028-03-31"),     # still the old value
               "mis_arrive": SimpleNamespace(value="2029-03-01"),
               "mis_return": SimpleNamespace(value=False)}
    proj._apply_mission(edited="mis_open")

    assert S.mission.launch_window[0] == date(2030, 1, 1), "the edit was dropped again"
    assert proj._w["mis_open"].value == "2030-01-01", "the edited field must not be rewritten"
    # The two that moved now show what the model holds, rather than the values they were left at.
    assert proj._w["mis_close"].value == S.mission.launch_window[1].isoformat()
    assert proj._w["mis_arrive"].value == S.mission.arrive_by.isoformat()

    # Text mid-way through being typed leaves the model on its previous value rather than raising.
    before = S.mission.launch_window[0]
    proj._w["mis_open"].value = "2030-01-"
    proj._apply_mission(edited="mis_open")
    assert S.mission.launch_window[0] == before


def test_return_plot_gated_on_convergence() -> None:
    """A return leg the optimizer didn't close is not a flyable trajectory: the Trajectory and
    Diagnostics views and the timeline all suppress it (only its status is reported)."""
    import ui.workspaces.flight as fl

    converged = {"return": {"sf": {"feasible": True, "mismatch": 1e-6, "tof_days": 300.0}}}
    unconverged = {"return": {"sf": {"feasible": False, "mismatch": 1.0, "tof_days": 300.0}}}
    errored = {"return": {"error": "window too short"}}
    assert fl._return_plottable(converged) is True
    assert fl._return_plottable(unconverged) is False     # optimization failed -> no plot
    assert fl._return_plottable(errored) is False         # never ran -> no plot
    assert fl._return_plottable({}) is False              # one-way mission


@pytest.mark.anyio
@pytest.mark.nicegui_main_file("ui/app.py")
async def test_plan_trajectory_renders(user: User) -> None:
    """The Plan-trajectory workspace builds the budget strip, the gated Earth-escape →
    Trajectory → Diagnostics sub-tabs, the escape rail, and the mission timeline, without
    running any solver (the escape sub-tab is the landing view, and no job is dispatched)."""
    projects = state.available_projects()
    assert projects, "no studies in configs/studies to open"
    state.open_project(projects[0])
    S.workspace = "flight"
    # A focus target with full elements, so the porkchop control offers a Compute (not the "no
    # target in focus" message) once the Trajectory sub-tab is reachable.
    df = edelbaum.evaluate_dataframe(_FAKE_POP.copy(), dv_budget=20.0)
    from prospector import jobs
    S.focus = jobs.jsonable_row(df.iloc[0])

    await user.open("/")
    await user.should_see("Total ΔV")            # the journey strip's total-ΔV stat
    await user.should_see("Departure")           # the journey strip's first node
    await user.should_see("Arrival")             # the journey strip's last node
    await user.should_see("Earth escape")        # the gated sub-tabs
    await user.should_see("Leaving Earth")       # the escape rail section


# A swept model setting rides along as a param_<key> column, the way a --sweep-param session
# records it, so the test also covers the Model-params column group.
_COMBOS_CSV = (
    "dry_kg,prop_kg,wet_kg,n_engines,engine,success,buildable,margin_kms,"
    "total_dv_kms,payload_capacity_kg,capability_kms,param_power_margin_pct,why\n"
    f"150,300,450,3,{lib.engine_key()},True,True,0.42,8.10,46,8.52,20,ok\n"
    f"200,300,500,3,{lib.engine_key()},True,False,0.15,8.30,18,8.45,30,over budget\n"
    f"260,300,560,3,{lib.engine_key()},False,False,-0.30,8.80,5,8.50,40,over budget\n"
)


@pytest.mark.anyio
@pytest.mark.nicegui_main_file("ui/app.py")
async def test_compare_vehicles_renders(user: User, monkeypatch, tmp_path) -> None:
    """The Compare-vehicles workspace builds the in-view tabs, the live-filter rail, and the
    sweep runner, and loads a (synthetic, on-disk) sweep session into the per-design grid --
    without launching the search script (the data layer reads the session artifacts directly)."""
    from ui.workspaces import vehicle

    # A self-contained, timestamp-named session on disk so the test never depends on runs/ contents
    # (only dated sessions are listed). Redirecting the run root moves the search root with it,
    # since vehicle_search resolves it per call.
    projects = state.available_projects()
    assert projects, "no studies in configs/studies to open"
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    sess = tmp_path / "vehicle_search" / "20300101-120000"
    sess.mkdir(parents=True)
    # Filed under the study about to be opened: the list shows a project its own sweeps only.
    (sess / "session.json").write_text(
        json.dumps({"target": "99942 Apophis", "pdes": "99942", "mission": lib.mission_key(),
                    "study": projects[0], "engine": lib.engine_key(),
                    "pairs": [[lib.engine_key(), 3]]}))
    (sess / "status.json").write_text('{"state": "done", "n_evaluated": 3}')
    (sess / "combos.csv").write_text(_COMBOS_CSV)

    state.open_project(projects[0])
    vehicle.reset()
    vehicle._session = "20300101-120000"
    vehicle._view = "Table"                       # land on the per-design grid
    vehicle._cols_on.add("params")                # reveal the swept model-setting column
    S.workspace = "vehicle"
    S.focus = {"pdes": "99942", "full_name": "99942 Apophis", "i": 3.34, "a": 0.922}

    await user.open("/")
    await user.should_see("New sweep")            # the sweep-runner launcher
    await user.should_see("Recent sweeps")        # the sessions panel
    await user.should_see("Build masses")          # a Table-view column-group toggle (in the rail)

    # The per-design grid renders the session, and the default filters (buildable + flyable both
    # on) narrow the three synthetic designs to the one that closes and builds. Find the grid among
    # the page's tables by its decision columns.
    grids = [t for t in user.find(ui.table).elements if "Fuel margin (kg)" in
             {c.get("label") for c in (t.columns or [])}]
    assert grids, "the per-design grid did not render"
    rows = grids[0].rows
    assert len(rows) == 1, rows                   # only the viable design passes the default filters
    assert rows[0]["Dry (kg)"] == 150 and rows[0]["Builds"] == "✓"
    # The memorable name leads the grid as the primary identifier: an adjective-noun slug the user
    # references instead of the numeric params.
    labels = {c.get("label") for c in (grids[0].columns or [])}
    assert "Name" in labels, labels
    assert rows[0]["Name"] and rows[0]["Name"].count("-") >= 1
    # The swept model setting shows as its own labeled column, so its value is readable per row.
    assert "Solar array margin (%)" in labels, labels
    assert rows[0]["Solar array margin (%)"] == 20


def test_every_flight_subtab_is_wired_to_a_view_and_the_escape_gate():
    """Adding a Plan-trajectory sub-tab and forgetting to wire it fails silently -- the tab renders
    a blank canvas. So the tab list, the render branches, and the escape gate are checked against
    each other: every tab past the escape must have a branch and must be gated on it."""
    import inspect

    from ui.workspaces import flight
    src = inspect.getsource(flight)
    tabs = [name for name, _icon in flight._SUBTABS]
    assert tabs[0] == "Earth escape"                 # the gate itself, never gated
    for name in tabs:
        assert f'S.flight_tab == "{name}"' in src or name == "Diagnostics", \
            f"sub-tab {name!r} has no render branch"   # Diagnostics is the else branch
    gate = src.split("disabled_tip")[0].rsplit("disabled=", 1)[-1]
    for name in tabs[1:]:
        assert f'"{name}"' in gate, f"sub-tab {name!r} is not gated behind the Earth escape"


def test_the_screen_loads_its_catalog_on_a_cold_start(monkeypatch):
    """Regression: the reachability screen's catalog load must work with nothing cached.

    ``_compute_base`` once took a parameter named ``population``, which shadowed the ``population``
    MODULE inside the function, so a repo-wide rename turned the catalog load into an attribute
    lookup on the parameter. With nothing cached the parameter is None, so every first screen of
    every session raised AttributeError, surfaced to the user as "SBDB fetch failed". Nothing else
    catches this: the failure needs the uncached path, which the rest of the suite seeds around.
    """
    import pandas as pd

    from prospector.config import EngineMount, ResolvedConfig, Screening, Vehicle
    from prospector.population import catalog, planets
    from prospector.spacecraft.propulsion import load_engines
    from ui.workspaces import target as tw

    monkeypatch.setattr(catalog, "fetch_population", lambda **kw: pd.DataFrame({
        "spkid": [2099942], "pdes": ["99942"], "full_name": ["99942 Apophis"], "H": [19.7],
        "condition_code": [0], "a": [0.9224], "e": [0.1914], "i": [3.331],
        "om": [204.4], "w": [126.7], "ma": [180.0], "epoch": [2460800.5]}))
    monkeypatch.setattr(tw, "_enrichment_frame", lambda pdes: None)
    rc = ResolvedConfig.build(
        Mission(launch_orbit="LEO"),
        Vehicle(name="v", dry_mass=250.0, fuel_mass=325.0,
                engines=[EngineMount(type=lib.engine_key(), count=4)]),
        Screening(), load_engines())

    df, cached, cached_h = tw._compute_base(rc, None, None, None)   # cold: nothing cached
    assert "lowthrust_dv" in df.columns and "reachable" in df.columns
    assert cached_h == pytest.approx(round(rc.screening.h_max, 1))
    assert len(cached) == len(df) and len(df) > 1        # the asteroid plus the planets
    assert "Mars" in set(df["pdes"])                    # planets reach the screen
    assert set(df[planets.BODY_CLASS_COL]) == {planets.ASTEROID, planets.PLANET}


def test_unavailable_characterization_reads_calm_not_failed():
    """A default install has no enrichment dependencies, so the desirability axis is simply
    absent; that is the expected state, not a fault, and it says nothing about the screen the
    user is looking at. The worker records it with a distinct message; this pins that the two
    ERROR causes stay distinguishable, because keying on the wording of one exception raised in
    another package is exactly how the calm path stops matching without anyone noticing.
    """
    from prospector import jobs
    from ui.workspaces.target import _is_unavailable

    unavailable = {"state": jobs.ERROR, "message": "enrichment unavailable",
                   "error": "enrichment needs the 'rocks' package to resolve small-body "
                            "identities; install the enrichment extra"}
    assert _is_unavailable(unavailable)
    # The word appears in only one of the two fields, and which one is not this layer's business.
    assert _is_unavailable({"state": jobs.ERROR, "message": "enrichment unavailable", "error": ""})
    assert _is_unavailable({"state": jobs.ERROR, "message": "", "error": "rocks is not installed"})
    # A genuine failure must still read as one.
    assert not _is_unavailable(
        {"state": jobs.ERROR, "message": "enrichment failed",
         "error": "no targets could be characterized (the catalogues may be unreachable)"})


def test_escape_sun_distance_is_heliocentric_au():
    """The Mission profile view computes each escape sample's distance from the Sun, and it is
    the only caller of that helper, so nothing else notices when a constant it reaches for
    gets renamed. It did: the helper divided by an ``AU`` attribute on the spiral module that a
    units sweep had turned into ``AU_M``, and the whole view raised AttributeError on open.

    Asserting the value rather than merely calling it also pins the unit. A geocentric escape stays
    within a Lunar distance of Earth, so every sample must land within a percent of 1 AU; dividing
    by kilometres instead of metres would put it near 1e-3.
    """
    import numpy as np

    from ui.workspaces.flight import _escape_sun_distance_au

    positions_km = np.array([[7000.0, 0.0, 0.0], [0.0, 7500.0, 0.0], [0.0, 0.0, 400000.0]])
    au = _escape_sun_distance_au(positions_km, np.array([0.0, 100.0, 200.0]))
    assert au.shape == (3,)
    assert np.all((0.9 < au) & (au < 1.1)), au


def test_mass_curve_rows_parse_into_segments():
    """The array mass curve is structured data edited as pasted rows, like a throttle curve.

    A malformed row raises rather than being dropped: a curve missing a band would size every array
    in that band off the neighbouring segment, and nothing downstream would look wrong.
    """
    import pytest as _pytest

    from ui.settings import parse_mass_curve

    rows = "lo_W hi_W w0 kg0 w1 kg1\n5000, 100000, 5000, 50, 100000, 1000\n50, 500, 50, 1.2, 500, 5"
    out = parse_mass_curve(rows)
    assert [s["lo_W"] for s in out] == [50.0, 5000.0], "header skipped, segments sorted by power"
    assert out[0]["kg0"] == 1.2

    assert parse_mass_curve("") is None            # blank keeps whatever is stored
    assert parse_mass_curve("   \n\n") is None
    with _pytest.raises(ValueError, match="6 numbers"):
        parse_mass_curve("50, 500, 50, 1.2")
    with _pytest.raises(ValueError, match="upper power"):
        parse_mass_curve("500, 50, 50, 1.2, 500, 5")


def test_catalog_add_and_delete_reach_the_config_library(tmp_path):
    """Adding and removing a launch type has to work from the dialog, or the only way to change
    the library is editing YAML by hand - which is where a launch type nobody can see comes from.

    Both edits are held in the working copy until Save, so Cancel costs nothing and one bad number
    cannot leave the library half-written.
    """
    from prospector.launch import (
        LaunchOrbit,
        delete_launch_orbit,
        load_launch_orbits,
        save_launch_orbit,
    )
    from ui.settings import _add_entry, _Catalog, _commit_catalog, _remove_entry

    launch_dir = tmp_path / "launches"
    save_launch_orbit(LaunchOrbit(name="leo", perigee_alt_km=400.0, apogee_alt_km=400.0,
                                  inclination_deg=28.5), "LEO", launch_dir)
    save_launch_orbit(LaunchOrbit(name="gone", perigee_alt_km=500.0, apogee_alt_km=500.0,
                                  inclination_deg=0.0), "GONE", launch_dir)

    class _Box:
        def __init__(self, value=""): self.value = value

    cat = _Catalog(
        entries=load_launch_orbits(launch_dir), fields={}, noun="launch type", blurb="",
        blank=lambda key: LaunchOrbit(name=key, perigee_alt_km=400.0, apogee_alt_km=400.0,
                                      inclination_deg=28.5),
        writer=lambda model, key: save_launch_orbit(model, key, launch_dir),
        remover=lambda key: delete_launch_orbit(key, launch_dir),
        text_fields=("name",), bool_fields=("escape_provided",))

    assert not _add_entry(cat, "MEO")
    _remove_entry(cat, "GONE")
    assert set(cat.entries) == {"LEO", "MEO"}
    assert set(load_launch_orbits(launch_dir)) == {"LEO", "GONE"}, "the library moved before Save"

    # A key has to survive being a filename, and cannot shadow an entry already there.
    assert _add_entry(cat, "../etc/passwd")
    assert _add_entry(cat, "")
    assert _add_entry(cat, "LEO")
    assert set(cat.entries) == {"LEO", "MEO"}

    # The panel would have built these; the working copy's own values stand in for untouched boxes.
    cat.fields = {k: {f: _Box(getattr(v, f)) for f in type(v).model_fields}
                  for k, v in cat.entries.items()}
    cat.fields["MEO"]["apogee_alt_km"] = _Box(20000.0)
    _commit_catalog(cat, LaunchOrbit)

    written = load_launch_orbits(launch_dir)
    assert set(written) == {"LEO", "MEO"}
    assert written["MEO"].apogee_alt_km == 20000.0
    assert not (launch_dir / "GONE.yaml").exists()


def test_catalog_save_keeps_untouched_fields():
    """Editing one coefficient of a catalog entry must not blank the rest of it."""
    from prospector.launch import LaunchOrbit
    from ui.settings import _save_catalog

    stored = LaunchOrbit(name="GTO", perigee_alt_km=250.0, apogee_alt_km=35786.0,
                         inclination_deg=27.0, escape_provided=False, min_thrust_alt_km=23622.0)
    written = {}

    class _Widget:
        def __init__(self, value): self.value = value

    # Only the thrust floor is touched; the name box is left blank, which must not erase it.
    fields = {"GTO": {"min_thrust_alt_km": _Widget(21000.0), "name": _Widget("GTO")}}
    _save_catalog(fields, {"GTO": stored}, LaunchOrbit,
                  lambda model, key: written.__setitem__(key, model))
    assert written["GTO"].min_thrust_alt_km == 21000.0
    assert written["GTO"].apogee_alt_km == 35786.0, "an untouched field was lost"
    assert written["GTO"].inclination_deg == 27.0


def test_catalog_save_round_trips_a_boolean():
    """The escape-provided switch decides whether the launch vehicle delivers escape outright,
    so flipping it silently rewrites the whole mission's budget - on a lunar-assisted TLI it is
    the difference between a free departure and a spiral the spacecraft has to fly itself.

    A switch is the one widget whose meaningful value is falsey, so a save that skips blank inputs
    has to let it through. Both states round-trip, and an untouched save preserves it. Turning it
    off needs a drop-off orbit to spiral out of, so that save is refused until one is typed in.
    """
    from pydantic import ValidationError

    from prospector.launch import LaunchOrbit
    from ui.settings import _save_catalog

    class _Widget:
        def __init__(self, value): self.value = value

    stored = LaunchOrbit(name="TLI", escape_provided=True)
    written = {}
    save = lambda model, key: written.__setitem__(key, model)  # noqa: E731

    def widgets(**overrides):
        base = {f: _Widget(getattr(stored, f)) for f in LaunchOrbit.model_fields}
        base.update({k: _Widget(v) for k, v in overrides.items()})
        return {"TLI": base}

    _save_catalog(widgets(), {"TLI": stored}, LaunchOrbit, save)
    assert written["TLI"].escape_provided is True, "an untouched switch was not preserved"

    with pytest.raises(ValidationError, match="perigee_alt_km"):
        _save_catalog(widgets(escape_provided=False), {"TLI": stored}, LaunchOrbit, save)

    _save_catalog(widgets(escape_provided=False, perigee_alt_km=185.0, apogee_alt_km=400000.0,
                          inclination_deg=28.5), {"TLI": stored}, LaunchOrbit, save)
    assert written["TLI"].escape_provided is False, "switching it off did not persist"
    assert written["TLI"].apogee_alt_km == 400000.0

    off = written["TLI"]
    _save_catalog(widgets(escape_provided=True), {"TLI": off}, LaunchOrbit, save)
    assert written["TLI"].escape_provided is True, "switching it on did not persist"
    assert written["TLI"].apogee_alt_km is None, "the orbit it no longer flies was kept"


def test_unbuildable_vehicle_reads_red_on_the_top_bar_chip():
    """A vehicle whose bus overruns its dry-mass budget has no payload capacity left -- a dead
    design. The top-bar chip is the only vehicle chrome visible from every workspace, so the
    shortfall has to surface there (red, signed, with the diagnosis) instead of only inside the
    build dialog: found after a trajectory is solved, it has already wasted the solve."""
    from prospector.config import EngineMount, ResolvedConfig, Screening, Vehicle
    from prospector.launch import LaunchOrbit
    from prospector.spacecraft.propulsion import Engine
    from ui import topbar
    from ui.components import vehicle_build_mass_rows
    from ui.workspaces import project as proj

    cat = {"E": Engine(name="E", isp_s=1800, thrust_mN=250, power_W=5000)}
    launches = {"LV": LaunchOrbit(name="lv", perigee_alt_km=200, apogee_alt_km=200,
                                  inclination_deg=28.5, escape_provided=True)}
    mis = Mission(launch_orbit="LV", launch_window=(date(2027, 1, 1), date(2027, 3, 1)),
                  arrive_by=date(2029, 6, 1))
    mounts = [EngineMount(type="E", count=4)]

    # A dry mass far under what four thrusters, their arrays and tank already weigh.
    starved = ResolvedConfig.build(mis, Vehicle(name="starved", dry_mass=60, fuel_mass=400,
                                               engines=mounts, solar_power_W=20000),
                                   Screening(), cat, launches=launches)
    build = proj.assess_build(starved)
    assert build is not None and build["buildable"] is False
    shortfall = topbar._payload_shortfall(build)
    assert shortfall is not None and shortfall > 0
    assert shortfall == pytest.approx(-build["payload_capacity_kg"])
    # The diagnosis names the bill, so the chip's tooltip has something to say beyond "no".
    assert "kg dry" in build["build_why"]

    # The grid renders the deficit with its sign, which is what turns the row red, a value that
    # lost its minus sign would render as an ordinary mass.
    row = dict(vehicle_build_mass_rows(build))["Payload+mgn"]
    assert row.startswith("-"), row

    # A design with room to spare stays out of the red path entirely.
    roomy = ResolvedConfig.build(mis, Vehicle(name="roomy", dry_mass=3000, fuel_mass=400,
                                             engines=[EngineMount(type="E", count=1)],
                                             solar_power_W=5000),
                                 Screening(), cat, launches=launches)
    assert topbar._payload_shortfall(proj.assess_build(roomy)) is None
    # An absent assessment is not a failed one: no build model, no red.
    assert topbar._payload_shortfall(None) is None


def test_a_sub_kilogram_payload_shortfall_still_reads_as_a_shortfall():
    """Rounding a deficit of a few hundred grams to a bare "0 kg" would show an over-budget
    design as break-even -- the one case where the distinction decides buildability."""
    from ui import topbar
    from ui.components import _kg

    assert _kg(-0.4) == "-0.4 kg"
    assert topbar._signed_kg(-0.4) == "-0.4 kg"
    assert _kg(-42.0) == "-42 kg"          # ordinary magnitudes stay whole
    assert topbar._signed_kg(-42.0) == "-42 kg"
    assert topbar._signed_kg(120.0) == "+120 kg"


def test_array_mass_curve_rows_validate_before_they_can_be_saved():
    """The array mass curve is a row per power band, like an engine's throttle curve. The anchor
    check is the load-bearing one: the mass is interpolated as
    ``kg0 + (kg1 - kg0)/(w1 - w0) * (P - w0)``, so two anchors at the same power divide by zero.
    That curve validates fine as a model and only fails later, when something asks it for a
    mass -- so it has to be caught at the editor, not at the model."""
    from prospector.spacecraft.arrays import ArrayMassSegment
    from ui import settings

    good = [(50.0, 500.0, 50.0, 1.2, 500.0, 5.0),
            (500.0, 5000.0, 500.0, 5.0, 5000.0, 50.0)]
    segs, msg = settings.segments_from_sextets(list(reversed(good)))
    assert [s["lo_W"] for s in segs] == [50.0, 500.0], "segments sort by band"
    assert "2 segments" in msg and "kW" in msg

    # Two anchors at one power: a model-valid segment whose mass_kg() raises.
    same_anchor = [(50.0, 500.0, 500.0, 1.2, 500.0, 5.0)]
    segs, msg = settings.segments_from_sextets(same_anchor)
    assert segs is None and "different powers" in msg
    with pytest.raises(ZeroDivisionError):
        ArrayMassSegment(**dict(zip(settings._MASS_COLS, same_anchor[0], strict=True))).mass_kg(100.0)

    for bad, expect in (([(500.0, 50.0, 50.0, 1.2, 500.0, 5.0)], "upper power"),
                        ([(50.0, 500.0, 50.0, 0.0, 500.0, 5.0)], "anchor mass"),
                        ([(-1.0, 500.0, 50.0, 1.2, 500.0, 5.0)], "positive")):
        segs, msg = settings.segments_from_sextets(bad)
        assert segs is None, bad
        assert expect in msg, msg

    # No rows at all is not an error: the stored curve stands, since an array with no curve cannot
    # be sized at all.
    assert settings.segments_from_sextets([]) == (None, "")


def test_array_mass_curve_rows_add_edit_and_remove():
    """The row state is the editor. Exercised directly (no live client) so the add/edit/remove
    path is pinned independently of whether a refreshable redraws in the test harness."""
    from prospector.spacecraft import buildability
    from ui import settings

    settings._mass_status = None            # no label bound to a torn-down client
    stored = buildability.load_bus_model().array_mass_curve
    settings._set_mass_rows(stored)
    assert len(settings._mass_rows) == len(stored)
    assert settings._mass_rows[0]["lo_W"] == stored[0].lo_W

    settings._add_mass_segment()
    assert set(settings._mass_rows[-1]) == set(settings._MASS_COLS)
    # A row that is still entirely blank is not a segment yet, so it must not fail validation.
    segs, msg = settings.segments_from_sextets(settings._mass_sextets())
    assert segs is not None and msg, msg

    for col, val in zip(settings._MASS_COLS, (5000.0, 9000.0, 5000.0, 40.0, 9000.0, 70.0),
                        strict=True):
        settings._set_mass_cell(len(settings._mass_rows) - 1, col, val)
    segs, _ = settings.segments_from_sextets(settings._mass_sextets())
    assert (5000.0, 9000.0) in [(s["lo_W"], s["hi_W"]) for s in segs]

    settings._remove_mass_segment(len(settings._mass_rows) - 1)
    segs, _ = settings.segments_from_sextets(settings._mass_sextets())
    assert len(segs) == len(stored)
    # Dicts round-trip as well as model objects, so a pasted curve reloads into the rows.
    settings._set_mass_rows(segs)
    assert settings._mass_rows[0]["lo_W"] == segs[0]["lo_W"]


@pytest.mark.anyio
@pytest.mark.nicegui_main_file("ui/app.py")
async def test_global_settings_dialog_builds_with_the_mass_curve_rows(user: User) -> None:
    """The Global-settings button is labelled (not a bare icon) and its dialog builds every
    panel, including the array mass curve's row editor."""
    from ui import settings

    projects = state.available_projects()
    state.open_project(projects[0])
    await user.open("/")
    await user.should_see("Global settings")          # the top-bar button carries its name

    # Clicked rather than called: a dialog is built into the client's layout slot, so opening it
    # out of band has nowhere to attach.
    user.find("Global settings").click()
    await user.should_see("Array mass curve")
    await user.should_see("Add segment")
    # The stored curve is loaded into the rows, not left for the user to retype.
    assert settings._mass_rows, "the dialog opened with an empty curve"


# ======================================================================================
# Restoring a project's runs across sessions (ui/session.py)
# ======================================================================================

def _seed_solved_project(tmp_path, monkeypatch, *, pdes="X", vinf=1.0):
    """Open a study, point it at a converged cruise run, and hand back what it took.

    The run is recorded through ``jobs`` as the worker would, so the restore path reads a real job
    spec (its stored config is what every staleness check compares against).
    """
    from prospector import jobs
    from ui import session

    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    projects = state.available_projects()
    state.open_project(projects[0])
    S.departure_vinf = vinf
    S.focus = {"pdes": pdes, "full_name": pdes, "a": 1.2, "e": 0.1, "i": 5.0,
               "om": 0.0, "w": 0.0, "ma": 0.0, "epoch": 2461000.5}
    rc = state.resolved()

    from ui.workspaces import flight
    run_id = jobs.record_run(rc.model_dump(mode="json"), dict(S.focus), 9000.0, 9300.0,
                             options={"available_power_W": flight._available_cruise_power_W(rc)})
    # A converged result: what the restore requires before it will adopt the run.
    result = jobs.read_result(run_id) or {}
    result["sf"] = {"feasible": True, "mismatch": 0.0, "dv_kms": 4.2, "tof_days": 300.0,
                    "dep_mjd2000": 9000.0}
    jobs.write_result(run_id, result)
    jobs.write_status(run_id, state=jobs.DONE)
    S.solve_run_id = run_id
    S.grid_sel = (9000.0, 300.0)          # the picked (departure, flight time) cell
    session.remember()
    return projects[0], run_id


def test_a_solved_trajectory_comes_back_when_the_project_is_reopened(tmp_path, monkeypatch):
    """The expensive thing is the solve, and it is already on disk with the config it was flown
    for. Reopening a project must bring it back rather than showing an empty window beside a
    converged run, and must say it did, so a restored result is never mistaken for a fresh one.
    """
    from ui import session

    project, run_id = _seed_solved_project(tmp_path, monkeypatch)
    restored = state.open_project(project)          # the reopen under test

    assert S.solve_run_id == run_id, "the converged cruise was not re-adopted"
    assert "Trajectory" in restored, restored
    assert S.departure_vinf == 1.0, "the v-infinity knob the run was flown at came back"
    assert S.focus and S.focus["pdes"] == "X"
    # The picked cell does not come back on its own: it is a coordinate into the transfer grid, so
    # without that surface restored there is nothing for it to point at. Restoring it anyway would
    # leave a selection marker on a plot that no longer exists.
    assert S.grid_sel is None, "a cell was restored with no surface to index into"
    # And the workspace agrees the run is live, through its own gate rather than ours.
    from ui.workspaces import flight
    assert flight._cruise_result(S.solve_run_id) is not None
    assert session.snapshot()["solve_run_id"] == run_id


def test_a_restored_trajectory_is_dropped_when_the_vehicle_changed(tmp_path, monkeypatch):
    """The whole hazard of restoring is showing a result from a configuration that no longer
    exists. A dry-mass edit between sessions moves the cruise-start mass, so the stored run
    describes a different spacecraft and must not come back."""
    project, run_id = _seed_solved_project(tmp_path, monkeypatch)

    # Edit the vehicle the way the Project workspace would, and persist it as the study default so
    # reopening loads the heavier bus.
    heavier = S.vehicle.model_copy(update={"dry_mass": S.vehicle.dry_mass + 250.0})
    S.vehicle = heavier
    state.persist_study(S.study_name, project)

    restored = state.open_project(project)
    assert S.solve_run_id is None, "a run flown for a lighter vehicle was restored"
    assert "Trajectory" not in restored
    assert S.grid_sel is None, "the picked cell outlived the surface it points into"
    assert S.grid_run_id is None, "a grid solved for a lighter vehicle was restored"


def test_a_restored_trajectory_is_dropped_when_it_was_flown_to_another_body(tmp_path, monkeypatch):
    """The trajectory signature is config-only -- it names no target -- so a run flown to one
    asteroid matches a config now aimed at another. Without a separate target check the restore
    would put the wrong body's trajectory on screen under the right vehicle."""
    from prospector.figures import products
    from ui.workspaces import flight

    project, run_id = _seed_solved_project(tmp_path, monkeypatch, pdes="Apophis")
    S.focus = {**S.focus, "pdes": "2008 EV5", "full_name": "2008 EV5"}
    from ui import session
    session.remember()

    rc = state.resolved()
    # The config check alone would happily adopt it; only the target check refuses.
    assert products.cruise_run_matches(run_id, rc, flight._available_cruise_power_W(rc),
                                       unknown_ok=False)
    state.open_project(project)
    assert S.solve_run_id is None, "a trajectory to a different body was restored"


def test_an_unreadable_run_is_never_adopted(tmp_path, monkeypatch):
    """Mid-session, a run whose config cannot be read is KEPT: the user launched it and yanking
    it away would be worse. On adoption the same silence is no evidence, and adopting on it puts
    a trajectory nobody asked for on screen. Same predicate, opposite default."""
    from prospector import jobs
    from prospector.figures import products
    from ui.workspaces import flight

    project, run_id = _seed_solved_project(tmp_path, monkeypatch)
    rc = state.resolved()
    (jobs.run_dir(run_id) / "job.json").write_text("{}")     # a spec with no config in it

    assert products.cruise_run_matches(run_id, rc, flight._available_cruise_power_W(rc)) is True
    assert products.cruise_run_matches(run_id, rc, flight._available_cruise_power_W(rc),
                                       unknown_ok=False) is False
    state.open_project(project)
    assert S.solve_run_id is None


def test_a_hand_broken_session_file_restores_nothing_and_never_raises(tmp_path, monkeypatch):
    """The pointer file is disposable, so a truncated or hand-edited one must degrade to "start
    clean" rather than take the app down on open."""
    from ui import session

    project, _ = _seed_solved_project(tmp_path, monkeypatch)
    path = session._path(project)
    for text in ("", "{", "[]", '{"departure_vinf": "fast", "solve_run_id": 17}'):
        path.write_text(text)
        assert state.open_project(project) == []
        assert S.solve_run_id is None
        # The baseline is the mission's planned speed (zero when it has none), never the junk.
        assert S.departure_vinf == float(S.mission.departure_vinf_kms or 0.0), \
            "a junk knob value was trusted"


def test_the_session_pointer_only_writes_when_the_state_moves(tmp_path, monkeypatch):
    """It is written from a page timer, so an unchanged snapshot must not rewrite the file --
    otherwise the app touches disk every few seconds for the life of the session."""
    from ui import session

    project, _ = _seed_solved_project(tmp_path, monkeypatch)
    path = session._path(project)
    before = path.read_text()
    stamp = path.stat().st_mtime_ns

    session.remember()
    assert path.stat().st_mtime_ns == stamp, "an unchanged snapshot rewrote the file"

    S.departure_vinf = 2.5
    session.remember()
    assert path.read_text() != before
    assert json.loads(path.read_text())["departure_vinf"] == 2.5


def test_a_numpy_valued_focus_row_survives_the_round_trip(tmp_path, monkeypatch):
    """The focus target is a population row, so its values arrive as numpy scalars. json cannot
    take those: an unhandled one would make every write fail silently and nothing would ever be
    restored -- the failure mode that looks exactly like the feature not existing."""
    import numpy as np

    from ui import session

    project, _ = _seed_solved_project(tmp_path, monkeypatch)
    S.focus = {"pdes": "X", "a": np.float64(1.23), "i": np.float32(4.5), "n": np.int64(7)}
    session.remember()
    stored = json.loads(session._path(project).read_text())["focus"]
    assert stored["a"] == pytest.approx(1.23) and isinstance(stored["a"], float)
    assert stored["n"] == 7


def test_a_flown_escape_spiral_comes_back_with_its_project(tmp_path, monkeypatch):
    """The escape was deliberately never adopted from disk: a spiral that silently reads as
    "already solved" is worse than re-running it. Restoring the one the PROJECT points at is a
    different thing from scanning for a stranger's, the project recorded it, and reopening
    announces it, so it comes back, still gated on the escape fingerprint.

    The spiral knobs have to come back with it: duty cycle, run-time cap, steering and the
    radiation scenario are all IN that fingerprint, so returning with defaults would reject the run
    for the wrong reason.
    """
    from prospector import jobs
    from ui import session

    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: None)   # lay the job down only
    projects = state.available_projects()
    state.open_project(projects[0])
    if state.resolved().launch.escape_provided:
        pytest.skip("the example study's launch vehicle provides escape outright")

    S.launch_duty, S.launch_years = 82.0, 6.5      # non-default, so the fingerprint depends on them
    fingerprint = state.spiral_fingerprint(state.resolved())
    run_id = jobs.submit_spiral(state.resolved().model_dump(mode="json"),
                                options=state.spiral_options(), fingerprint=fingerprint)
    jobs.write_spiral_summary(run_id, {
        "status": "escaped", "dv_at_escape_kms": 7.0, "dv_kms": 7.4, "tof_days": 220.0,
        "vinf_kms": 1.0, "target_vinf_kms": 1.0, "propellant_kg": 233.0,
        "curve_vinf_kms": [0.0, 1.0, 2.0], "curve_dv_kms": [7.0, 7.4, 8.1],
        "curve_tof_days": [200.0, 220.0, 250.0]})
    jobs.write_spiral_status(run_id, state=jobs.DONE, progress=1.0)
    S.launch_run_id = run_id
    session.remember()

    restored = state.open_project(projects[0])
    assert S.launch_run_id == run_id, "the flown escape spiral was not re-adopted"
    assert "Earth escape" in restored, restored
    assert (S.launch_duty, S.launch_years) == (82.0, 6.5), "the knobs it was flown at came back"
    # Through the app's own gate, and priced into the budget: a restored spiral refines the escape
    # charge as one flown a minute ago does.
    rc = state.resolved()
    assert state.session_spiral_run(rc) == run_id
    assert rc.escape_dv_refined == pytest.approx(7.0, abs=0.5)
    from ui.workspaces import flight
    assert flight._escape_ready(rc), "the Trajectory sub-tab stayed gated behind a flown escape"

    # A spiral knob edit between sessions moves the fingerprint, so it must not come back.
    S.launch_duty = 55.0
    session.remember()
    assert "Earth escape" not in state.open_project(projects[0])
    assert S.launch_run_id is None


def test_the_mission_clock_runs_escape_straight_into_cruise(tmp_path, monkeypatch):
    """There is no interval between the escape ending and the cruise beginning, so the timeline
    must not contain one.

    The launch date and the cruise departure are the same choice offset by the spiral, and the
    departure window is the launch window shifted by exactly that, so liftoff is derived by
    back-dating the departure, and it always lands inside the launch window. Pinning liftoff to the
    window open instead manufactured a gap of up to the window's whole width, which read as the
    vehicle loitering. It cannot loiter: past escape it is on its own heliocentric orbit.
    """
    from prospector import jobs
    from prospector.config import EngineMount, ResolvedConfig, Screening, Vehicle
    from prospector.launch import LaunchOrbit
    from prospector.spacecraft.propulsion import Engine
    from ui.workspaces import flight

    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    cat = {"E": Engine(name="E", isp_s=1800, thrust_mN=250, power_W=5000)}
    launches = {"LV": LaunchOrbit(name="lv", perigee_alt_km=200, apogee_alt_km=200,
                                  inclination_deg=28.5)}
    mis = Mission(launch_orbit="LV", launch_window=(date(2028, 7, 1), date(2028, 7, 31)),
                  arrive_by=date(2031, 6, 1))
    veh = Vehicle(name="V", dry_mass=600, fuel_mass=400,
                  engines=[EngineMount(type="E", count=1)], solar_power_W=5000)
    rc = ResolvedConfig.build(mis, veh, Screening(), cat, launches=launches)

    # The departure window is the launch window shifted by the escape, so every departure the
    # solver may pick back-dates to a liftoff inside the launch window; there is never a gap.
    for dep in (rc.departure_window[0], rc.departure_window[1]):
        sched = rc.departure_schedule(dep)
        assert sched["liftoff_in_window"] is True, dep
        assert mis.launch_window[0] <= sched["liftoff_if_no_coast"] <= mis.launch_window[1]

    # The span at the LATEST departure, the case that used to report a full window of waiting.
    latest = rc.departure_window[1]
    run_id = jobs.record_run(rc.model_dump(mode="json"),
                             {"pdes": "X", "a": 1.2, "e": 0.1, "i": 5, "om": 0, "w": 0,
                              "ma": 0, "epoch": 2461000.5},
                             0.0, 0.0, options={"available_power_W": None})
    result = jobs.read_result(run_id) or {}
    dep_mjd = (latest - date(2000, 1, 1)).days
    result["sf"] = {"feasible": True, "mismatch": 0.0, "dv_kms": 4.0, "tof_days": 400.0,
                    "dep_mjd2000": float(dep_mjd)}
    jobs.write_result(run_id, result)
    jobs.write_status(run_id, state=jobs.DONE)
    S.solve_run_id = run_id
    # ``_timeline_span`` reads the live config through ``_cruise_result`` to decide whether the run
    # is still current. Pin that to the config the run was recorded under, so the test exercises
    # the span rather than whatever AppState an earlier test happened to leave behind.
    monkeypatch.setattr(flight, "_resolve", lambda: rc)
    try:
        span = flight._timeline_span(rc)
        assert "wait_days" not in span, "the timeline still carries a wait phase"
        # Liftoff is back-dated from the departure, not pinned to the window open.
        assert span["liftoff"] == rc.departure_schedule(latest)["liftoff_if_no_coast"]
        assert span["liftoff"] != mis.launch_window[0], "liftoff was pinned, not derived"
        # And the clock is exactly its two legs, with nothing between them.
        assert span["total_days"] == pytest.approx(span["esc_days"] + span["cruise_days"])
        fr = flight._phase_fracs(span)
        assert fr["escape_end"] == pytest.approx(fr["cruise_start"]), "a gap survives in the clock"
    finally:
        S.solve_run_id = None
        import shutil
        shutil.rmtree(jobs.run_dir(run_id), ignore_errors=True)


def test_the_trajectory_workspace_reads_the_converged_grid_not_the_impulsive_one():
    """The surface the user picks from is the converged grid. The impulsive porkchop it replaced was
    instant, which is exactly why it could only ever be an approximation, and measured against
    converged solves it was anti-correlated with the truth, hid 90% of the flyable window on one
    target, and its rejection verdict fired on cells that fly.

    So the old compute path is gone rather than merely unused: a surface that must never be shown
    is a hazard sitting in the workspace, and the separate flight-time trade tab goes with it,
    since that curve is the grid's own per-column minimum.
    """
    from ui.workspaces import flight

    assert hasattr(flight, "_run_grid") and hasattr(flight, "_grid_plot")
    for gone in ("_compute_porkchop", "_tof_trade_view", "_run_tof_trade",
                 "_solved_cell", "_porkchop_plot"):
        assert not hasattr(flight, gone), f"{gone} survived"
    assert "Flight-time trade" not in [t[0] for t in flight._SUBTABS]
    # And the retired state is off AppState, not merely unread; a dataclass would happily let a
    # stale attribute be set back on by any caller that still remembered it.
    fields = set(type(S).__dataclass_fields__)
    assert {"solve_porkchop", "tof_trade", "solve_tof_step_days"}.isdisjoint(fields)
    assert {"grid_run_id", "grid_sel", "grid_colour"} <= fields


def test_the_grid_in_force_clears_when_the_config_or_the_target_moves(tmp_path, monkeypatch):
    """A grid is a picture of one vehicle flying to one body, so a grid whose config no longer
    matches describes a mission that no longer exists. It clears rather than lingering behind a
    warning, the same discipline a stale cruise gets, and for the stronger reason that its cells
    are the seeds a refine would start from."""
    from prospector import jobs
    from prospector.config import EngineMount, ResolvedConfig, Screening, Vehicle
    from prospector.launch import LaunchOrbit
    from prospector.spacecraft.propulsion import Engine
    from ui.workspaces import flight

    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: None)
    cat = {"E": Engine(name="E", isp_s=1800, thrust_mN=250, power_W=5000)}
    launches = {"LV": LaunchOrbit(name="lv", perigee_alt_km=200, apogee_alt_km=200,
                                  inclination_deg=28.5, escape_provided=True)}
    mis = Mission(launch_orbit="LV", launch_window=(date(2027, 1, 1), date(2027, 3, 1)),
                  arrive_by=date(2029, 6, 1))
    veh = Vehicle(name="V", dry_mass=600, fuel_mass=400,
                  engines=[EngineMount(type="E", count=1)], solar_power_W=5000)
    rc = ResolvedConfig.build(mis, veh, Screening(), cat, launches=launches)
    focus = {"pdes": "X", "full_name": "X", "a": 1.2, "e": 0.1, "i": 5.0,
             "om": 0.0, "w": 0.0, "ma": 0.0, "epoch": 2461000.5}

    run_id = jobs.submit_grid(rc.model_dump(mode="json"), focus, options={})
    jobs.write_grid_result(run_id, {"dep_mjd2000": [9000.0], "tof_days": [300.0],
                                    "dv_kms": [[4.0]], "final_mass_kg": [[500.0]],
                                    "feasible": [[True]], "decision_vectors": [[[0.0]]],
                                    "n_feasible": 1, "n_cells": 1, "frontier": []})
    jobs.write_grid_status(run_id, state=jobs.DONE, progress=1.0)

    S.grid_run_id = run_id
    S.focus = dict(focus)
    try:
        monkeypatch.setattr(flight, "_resolve", lambda: rc)
        assert flight._grid_result() is not None, "its own config and target should hold"

        # A heavier vehicle moves the cruise-start mass, so every cell describes a different ship.
        heavier = ResolvedConfig.build(mis, veh.model_copy(update={"dry_mass": 900.0}),
                                      Screening(), cat, launches=launches)
        monkeypatch.setattr(flight, "_resolve", lambda: heavier)
        assert flight._grid_result() is None, "a grid for a lighter vehicle stayed in force"

        # Same config, different body: the trajectory signature names no target, so this needs its
        # own check or the grid for one asteroid would be shown under another.
        monkeypatch.setattr(flight, "_resolve", lambda: rc)
        S.focus = {**focus, "pdes": "2008 EV5"}
        assert flight._grid_result() is None, "a grid for another body stayed in force"
    finally:
        S.grid_run_id, S.grid_sel, S.focus = None, None, None


def test_clicking_a_grid_cell_records_a_readable_run_without_re_solving(tmp_path, monkeypatch):
    """A click shows the cell's trajectory, and it does so by recording it through the ordinary run
    channel, so the trajectory view, the mission profile and the PDF all read it unchanged.

    The spec that lands on disk is JSON, so nothing date-shaped may reach it. The pin's own terms
    (launch_window, the flight-time bounds) are dates and bounds, and recording them raised "Object
    of type date is not JSON serializable" on every timer tick. They are reconstructible from the
    epochs recorded beside them, so they are dropped from the record rather than serialized.
    """
    import json as _json

    import numpy as np

    from prospector import jobs
    from prospector.config import EngineMount, ResolvedConfig, Screening, Vehicle
    from prospector.launch import LaunchOrbit
    from prospector.solvers import lambert as lb
    from prospector.spacecraft.propulsion import Engine
    from prospector.trades.pipeline import grid as G
    from prospector.trades.pipeline import solve as sv
    from ui.workspaces import flight

    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: None)
    # No Horizons in a test. Patched on each module that CALLS it, a function resolves names in its
    # own module, so patching the population facade would leave both callers on the real thing.
    monkeypatch.setattr(G, "_mission_elements", lambda rc_, r: dict(r))
    monkeypatch.setattr(sv, "_mission_elements", lambda rc_, r: dict(r))
    # A high-thrust bus so a 2x2 grid closes quickly; the physics has its own tests.
    cat = {"E": Engine(name="E", isp_s=2000, thrust_mN=4000, power_W=5000)}
    launches = {"LV": LaunchOrbit(name="lv", perigee_alt_km=400, apogee_alt_km=400,
                                  inclination_deg=0.0, escape_provided=True)}
    mis = Mission(launch_orbit="LV", launch_window=(date(2028, 1, 1), date(2028, 2, 1)),
                  arrive_by=date(2029, 8, 1))
    veh = Vehicle(name="v", dry_mass=800, fuel_mass=600,
                  engines=[EngineMount(type="E", count=1)], solar_power_W=5000)
    rc = ResolvedConfig.build(mis, veh, Screening(), cat, launches=launches)
    row = {"pdes": "X", "full_name": "X", "a": 1.15, "e": 0.06, "i": 3.0, "om": 20.0,
           "w": 40.0, "ma": 10.0,
           "epoch": lb.mjd2000_from_date(date(2028, 1, 1)) + 51544.5}

    surface = G.lowthrust_grid(rc, row, n_dep=2, n_tof=2, nseg=6, restarts=1, workers=1,
                               vinf_dep_kms=1.0, vinf_arr_kms=0.1)
    if not surface["n_feasible"]:
        pytest.skip("no cell converged on this synthetic setup")
    surface["frontier"] = G.best_per_flight_time(surface, [])

    run_id = jobs.submit_grid(rc.model_dump(mode="json"), row, options={})
    jobs.write_grid_result(run_id, surface)
    jobs.write_grid_status(run_id, state=jobs.DONE, progress=1.0)

    ok = np.asarray(surface["feasible"], bool)
    i, j = (int(a) for a in np.argwhere(ok)[0])
    cell = (float(surface["dep_mjd2000"][i]), float(surface["tof_days"][j]))

    S.grid_run_id, S.focus = run_id, dict(row)
    try:
        monkeypatch.setattr(flight, "_resolve", lambda: rc)
        monkeypatch.setattr(flight, "_grid_result", lambda: surface)
        flight._show_grid_cell(cell)

        assert S.solve_run_id, "the click recorded no run"
        # The spec on disk must be JSON, read it back the way every consumer does.
        spec = _json.loads((jobs.run_dir(S.solve_run_id) / "job.json").read_text())
        assert "launch_window" not in (spec.get("options") or {})
        assert spec["dep_mjd2000"] == pytest.approx(cell[0])
        assert spec["arr_mjd2000"] == pytest.approx(cell[0] + cell[1])
        # And the full result is there, with no solve having run.
        result = jobs.read_result(S.solve_run_id)
        assert result and result["sf"]["feasible"] is True
        assert jobs.read_status(S.solve_run_id)["state"] == jobs.DONE
        assert jobs.read_summary(S.solve_run_id)["dv_kms"] > 0
        # The rebuilt ΔV is the grid's own number for that cell, not a fresh optimization, and so
        # is its matchpoint mismatch: the rebuild reproduces the cell's trajectory exactly.
        assert result["sf"]["dv_kms"] == pytest.approx(
            float(np.asarray(surface["dv_kms"], float)[i, j]), rel=1e-6)
        assert result["sf"]["mismatch"] == pytest.approx(
            float(np.asarray(surface["mismatch"], float)[i, j]), rel=1e-6, abs=1e-12)
        # The grid recorded the cell's thrust terms and the click handed them to the rebuild.
        spec_opts = spec.get("options") or {}
        assert surface["leg_terms"][i][j] is not None
        assert spec_opts["thrust_N"] == pytest.approx(surface["leg_terms"][i][j]["thrust_N"])
    finally:
        S.grid_run_id, S.grid_sel, S.solve_run_id, S.focus = None, None, None, None


def test_a_grid_cell_click_rebuilds_against_the_cells_own_thrust_terms(monkeypatch):
    """The cruise operating point follows the Sun, so a grid cell settles on per-segment thrust
    ceilings and Isps of its own path. The click must rebuild the vector against exactly those,
    or the same vector flies a different thrust history and reads as not converged while the
    porkchop beside it shows the cell closing (measured: mismatch 1.5e-5 became 3.6e-3). A refined
    point carries its own terms; a plain cell's come from the grid's cube."""
    import numpy as np

    from prospector import config
    from prospector.trades import pipeline
    from ui.workspaces import flight
    S.grid_run_id, S.focus = "grid-x", {"pdes": "X", "full_name": "X"}
    rc = config.default_resolved()
    cell_terms = {"seg_caps": [0.9, 0.7, 0.5], "thrust_N": 0.232, "isp_s": 1751.0,
                  "seg_isp_s": [1783.0, 1712.0, 1700.0]}
    polish_terms = {"seg_caps": [0.8, 0.8, 0.8], "thrust_N": 0.230, "isp_s": 1740.0,
                    "seg_isp_s": None}
    grid = {"dep_mjd2000": [9000.0, 9010.0], "tof_days": [300.0, 350.0],
            "dv_kms": [[4.0, 3.8], [4.1, 3.9]], "final_mass_kg": [[500.0] * 2] * 2,
            "feasible": [[True, True], [True, True]], "nseg": 3,
            "decision_vectors": [[[1.0], [2.0]], [[3.0], [4.0]]],
            "leg_terms": [[cell_terms, None], [None, None]],
            "sf_kwargs": {"nseg": 3, "available_power_W": 4000.0},
            "target_row": dict(S.focus),
            "polished": [{"dep_mjd2000": 9005.0, "tof_days": 350.0, "solved_tof_days": 350.2,
                          "dv_kms": 3.7, "final_mass_kg": 505.0, "feasible": True,
                          "decision_vector": [9.0], "leg_terms": polish_terms}]}
    seen = []

    def capture(rc_, row, x, **kw):
        seen.append((list(np.asarray(x, float)), kw))
        raise RuntimeError("stop here: the rebuild itself is not under test")

    monkeypatch.setattr(pipeline, "result_from_decision", capture)
    monkeypatch.setattr(flight.ui, "notify", lambda *a, **k: None)
    try:
        monkeypatch.setattr(flight, "_resolve", lambda: rc)
        monkeypatch.setattr(flight, "_grid_result", lambda: grid)
        flight._show_grid_cell((9000.0, 300.0))                 # a plain cell, with terms
        flight._show_grid_cell((9010.0, 300.0))                 # a plain cell, without
        flight._show_grid_cell((9005.0, 350.0))                 # the refined point
    finally:
        S.grid_run_id, S.grid_sel, S.solve_run_id, S.focus = None, None, None, None
    assert [x for x, _kw in seen] == [[1.0], [3.0], [9.0]]
    with_terms, without, refined = (kw for _x, kw in seen)
    for k, v in cell_terms.items():
        assert with_terms[k] == v
    assert with_terms["available_power_W"] == 4000.0 and with_terms["nseg"] == 3
    assert not any(k in without for k in cell_terms), "a cell with no recorded terms adds none"
    assert refined["seg_caps"] == polish_terms["seg_caps"]
    assert refined["thrust_N"] == polish_terms["thrust_N"]
    assert "seg_isp_s" not in refined, "a None term is left for the rebuild to derive"
