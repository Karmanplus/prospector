"""The vehicle design-space search: its pure helpers and its launch contract.

The search driver itself runs Sims-Flanagan solves and takes minutes, so it is not exercised here.
What is covered is everything that can be wrong without a solve: the grid and pairing expansion
that decides what gets evaluated, the config each combo is built from, and the argv the app spawns
the search with. That last one matters because the Compare-vehicles workspace launches this module
as a subprocess, a flag renamed on either side would otherwise surface only when a user clicks Run
sweep.
"""
from __future__ import annotations

import pytest

from prospector.config import load_mission
from prospector.launch import load_launch_orbits
from prospector.spacecraft import buildability
from prospector.spacecraft.propulsion import load_engines
from prospector.trades import design_search as ds
from tests import library as lib

# ---------------------------------------------------------------------------
# engine/count pairings
# ---------------------------------------------------------------------------

def test_parse_pairs_expands_lists_and_ranges():
    catalog = load_engines()
    key = next(iter(sorted(catalog)))
    pairs = ds.parse_pairs([f"{key}:1,3-5"], catalog)
    assert pairs == [(key, 1), (key, 3), (key, 4), (key, 5)]


def test_parse_pairs_preserves_order_and_drops_exact_repeats():
    catalog = load_engines()
    key = next(iter(sorted(catalog)))
    assert ds.parse_pairs([f"{key}:2,2,1"], catalog) == [(key, 2), (key, 1)]


def test_parse_pairs_rejects_unknown_engine():
    with pytest.raises(SystemExit, match="unknown engine"):
        ds.parse_pairs(["no-such-engine:1"], load_engines())


def test_parse_pairs_rejects_missing_and_nonpositive_counts():
    catalog = load_engines()
    key = next(iter(sorted(catalog)))
    with pytest.raises(SystemExit, match="needs counts"):
        ds.parse_pairs([key], catalog)
    with pytest.raises(SystemExit, match="must be >= 1"):
        ds.parse_pairs([f"{key}:0"], catalog)


# ---------------------------------------------------------------------------
# swept model axes
# ---------------------------------------------------------------------------

def test_parse_model_axes_accepts_a_sweepable_key():
    key = next(iter(buildability.SWEEPABLE))
    assert ds.parse_model_axes([f"{key}=10,20"]) == {key: [10.0, 20.0]}


def test_parse_model_axes_rejects_an_unsupported_key():
    with pytest.raises(SystemExit, match="unknown sweep key"):
        ds.parse_model_axes(["not_a_knob=1,2"])


def test_parse_model_axes_rejects_non_numeric_values():
    key = next(iter(buildability.SWEEPABLE))
    with pytest.raises(SystemExit, match="must be numbers"):
        ds.parse_model_axes([f"{key}=fast"])


def test_model_combinations_is_the_cartesian_product():
    combos = ds.model_combinations({"a": [1.0, 2.0], "b": [3.0]})
    assert combos == [{"a": 1.0, "b": 3.0}, {"a": 2.0, "b": 3.0}]


def test_model_combinations_with_no_axes_is_one_empty_override():
    """No sweep axes must still evaluate exactly once, at the model's configured defaults."""
    assert ds.model_combinations({}) == [{}]


# ---------------------------------------------------------------------------
# per-combo config assembly
# ---------------------------------------------------------------------------

def test_build_config_applies_the_swept_masses_and_engine_count():
    catalog, launches = load_engines(), load_launch_orbits()
    mission = load_mission(lib.mission_key())
    key = lib.engine_key()
    rc, estimated = ds.build_config(mission, catalog, launches, dry_kg=200.0, prop_kg=300.0,
                                    n_engines=4, engine_key=key)
    assert (rc.vehicle.dry_mass, rc.vehicle.fuel_mass) == (200.0, 300.0)
    assert rc.vehicle.engine_count == 4
    # Four engines draw four times one engine's power and produce four times its thrust.
    assert rc.total_thrust_mN == pytest.approx(4 * catalog[key].thrust_mN)
    assert rc.total_power_W == pytest.approx(4 * catalog[key].power_W)
    # The engine is authored on xenon, so running it on xenon is measured, not estimated.
    assert estimated is False


def test_build_config_flags_a_non_native_gas_as_estimated():
    """A gas the engine has no authored numbers for is scaled, and must be flagged as such."""
    catalog, launches = load_engines(), load_launch_orbits()
    mission = load_mission(lib.mission_key())
    _rc, estimated = ds.build_config(mission, catalog, launches, dry_kg=200.0, prop_kg=300.0,
                                     n_engines=1, engine_key=lib.engine_key(),
                                     propellant_key="krypton")
    assert estimated is True


def test_build_config_uses_an_unmargined_screening_factor():
    """The search compares capability against a solved requirement, so a screening margin
    would double-count. It must resolve at exactly 1.0."""
    rc, _ = ds.build_config(load_mission(lib.mission_key()), load_engines(),
                            load_launch_orbits(), dry_kg=200.0, prop_kg=300.0,
                            n_engines=3, engine_key=lib.engine_key())
    assert rc.screening.dv_margin_factor == 1.0


# ---------------------------------------------------------------------------
# the launch contract: the argv the app spawns the search with
# ---------------------------------------------------------------------------

def test_the_mission_must_be_named_not_defaulted():
    """No default mission key: one would tie this module to a particular library's contents, and a
    recipient with their own library would get a search pointed at a mission they do not have."""
    import pytest
    with pytest.raises(SystemExit):
        ds.parse_args([])
    args = ds.parse_args(["--mission", lib.mission_key()])
    load_mission(args.mission)          # raises if the key is not in the active library
    assert args.buildability is True


def test_launch_accepts_the_flags_the_app_passes():
    """The Compare-vehicles workspace builds this argv; a rename here breaks the app silently."""
    args = ds.parse_args([
        "--out", "/tmp/session", "--target", "99942",
        "--mission", lib.mission_key(), "--h-max", "25.2",
        "--pairs", f"{lib.engine_key()}:3-4", "--propellants", "xenon", "krypton",
        "--dry-min", "150", "--dry-max", "250", "--dry-step", "50",
        "--prop-min", "200", "--prop-max", "300", "--prop-step", "100",
        "--max-wet", "700", "--duty", "0.9", "--vinf", "0.0", "1.2",
        "--nseg", "20", "--maxeval", "1500", "--restarts", "2", "--retries", "1",
        "--parallel", "4", "--workers", "0", "--sweep-param", "power_margin_pct=0,20",
        "--no-buildability", "--anchor", "--persist-winner",
    ])
    assert args.out == "/tmp/session"
    assert args.pairs == [f"{lib.engine_key()}:3-4"]
    assert args.propellants == ["xenon", "krypton"]
    assert args.vinf == [0.0, 1.2]
    assert args.buildability is False
    assert args.anchor is True
    assert ds.parse_model_axes(args.sweep_param) == {"power_margin_pct": [0.0, 20.0]}


def test_launch_out_defaults_under_the_session_root():
    """Sessions must land where the workspace looks for them, or a run is invisible in the app."""
    args = ds.parse_args(["--mission", lib.mission_key()])
    assert args.out.startswith(str(ds.DEFAULT_OUT_ROOT))
