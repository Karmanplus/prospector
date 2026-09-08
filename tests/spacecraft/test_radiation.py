"""Tests for the selectable solar-array radiation model (prospector.spacecraft.radiation).

The model is built from physics + literature so the mission total FALLS OUT of the flown path (it
is not calibrated to an external tool's number). These tests pin the structural invariants, the
belt geometry, the coverglass shielding direction, proton dominance, and that selecting a gentler
scenario moves the total down, and assert the SSO worst-case loss lands in a defensible BAND rather
than a fitted value.
"""
import numpy as np
import pytest

from prospector.spacecraft.radiation import (
    DEFAULT_RADIATION_MODEL,
    default_radiation_model,
    list_radiation_models,
    load_radiation_model,
    load_radiation_models,
)

RE = 6371.0


def _eq(L):
    """A geocentric point on the equator at L shells out (km)."""
    return np.array([L * RE, 0.0, 0.0])


def test_library_loads_and_default_reproduces_constants():
    catalog = load_radiation_models()
    assert {"ap8min-worstcase", "ap8max-nominal"} <= set(catalog)
    assert DEFAULT_RADIATION_MODEL in catalog
    m = default_radiation_model()
    # Cores anchored to the OMERE reference run (proton incl. ESP; electron ~800x below proton).
    assert m.proton_ddd_core == pytest.approx(4.6e9)
    assert m.electron_ddd_core == pytest.approx(5.8e6)
    # GaAs/Ge single-junction NRL curve (reverse-engineered from an OMERE run to 4 sig figs).
    assert m.cell_ddd_ref == pytest.approx(1.1e9) and m.cell_ddd_coef == pytest.approx(0.2904)
    assert tuple(m.proton_plateau_l) == (1.6, 2.7)
    # Default cover is the real multi-layer stack (212.5 um @ 1.64 g/cm3), shielded below the
    # reference; shielding is 1.0 exactly AT the reference areal density.
    assert m.coverglass_um == pytest.approx(212.5) and m.coverglass_density_g_cm3 == pytest.approx(1.64)
    assert m.with_coverglass(75.0, 1.9866).proton_shield() == pytest.approx(1.0)
    assert m.proton_shield() < 1.0                        # thicker/denser cover than the reference
    assert DEFAULT_RADIATION_MODEL in list_radiation_models()


def test_unknown_scenario_raises_rather_than_substituting_one():
    """A misspelled scenario key must be an error, not different physics.

    Substituting the default (or the bare class defaults) would change the dose the spiral flies
    and the array it sizes, and the only symptom would be a setting that stopped mattering.
    """
    with pytest.raises(KeyError, match="no radiation model"):
        load_radiation_model("nope-not-real")
    # The message names what is available, so the typo is fixable from the error alone.
    try:
        load_radiation_model("nope-not-real")
    except KeyError as exc:
        assert DEFAULT_RADIATION_MODEL in str(exc)


def test_missing_library_raises_rather_than_falling_back(tmp_path):
    """The belt physics is file-based and has no built-in fallback: an absent library is a
    broken install, and an empty catalog would let every caller silently use class defaults."""
    with pytest.raises(FileNotFoundError, match="no radiation-model library"):
        load_radiation_models(tmp_path / "not-there")
    with pytest.raises(FileNotFoundError):
        load_radiation_model("ap8min-worstcase", radiation_dir=tmp_path / "not-there")


def test_belt_geometry_toroidal():
    m = default_radiation_model()
    # The damaging-proton plateau glows on the equator; the poles and the dense atmosphere are cold.
    assert m.ddd_rate(_eq(2.0)) > 1e8                      # mid-plateau, equator
    assert m.ddd_rate(np.array([0.0, 0.0, 2.0 * RE])) < 1e3   # over the pole
    assert m.ddd_rate(np.array([1.05 * RE, 0.0, 0.0])) < 1e3  # ~300 km alt, below the belt floor
    # L-shell rises off the equator (toroidal), running high toward the pole.
    assert m.lshell(_eq(2.0)) == pytest.approx(2.0, rel=1e-6)
    assert m.lshell(np.array([RE, 0.0, RE])) > 2.0


def test_proton_dominates_electron():
    # Displacement damage in GaAs is overwhelmingly proton (OMERE: electrons ~0.12%, ~840:1). The
    # cores are set ~800:1 to reproduce that, not the ~10:1 of the raw belt fluxes.
    m = default_radiation_model()
    assert m.proton_ddd_core / m.electron_ddd_core == pytest.approx(793.1, rel=1e-2)
    proton_core, _ = m.rate_components(_eq(2.0))           # in the proton plateau
    _, electron_core = m.rate_components(_eq(m.electron_peak_l))   # at the electron peak
    assert proton_core > 100.0 * electron_core            # protons dwarf electrons


def test_coverglass_shielding_monotonic():
    m = default_radiation_model()
    thin, thick = m.with_coverglass(50.0), m.with_coverglass(300.0)
    # More cover stops more soft protons -> strictly less proton dose (thinner shields less).
    assert thin.proton_shield() > m.proton_shield() > thick.proton_shield()
    # Shielding is driven by areal DENSITY: at the same thickness, a denser cover shields more.
    assert m.with_coverglass(200.0, 2.5).proton_shield() < m.with_coverglass(200.0, 1.2).proton_shield()
    # Electrons out-range a thin cover, so their shielding moves far less than the protons'.
    p_ratio = thick.proton_shield() / thin.proton_shield()
    e_ratio = thick.electron_shield() / thin.electron_shield()
    assert p_ratio < e_ratio                               # protons attenuate much more steeply

    # Integrated loss on a fixed belt-crossing path falls monotonically as the cover thickens.
    pos, days = _belt_crossing_path()
    losses = [1.0 - m.with_coverglass(c).power_profile(pos, days)[2][-1]
              for c in (50.0, 100.0, 200.0, 400.0)]
    assert all(a > b for a, b in zip(losses, losses[1:]))  # strictly decreasing


def test_model_selection_moves_the_total_down():
    pos, days = _belt_crossing_path()
    worst = load_radiation_model("ap8min-worstcase")
    worst_end = worst.power_profile(pos, days)[2][-1]
    gentle = load_radiation_model("ap8max-nominal").power_profile(pos, days)[2][-1]
    # A thicker cover is reached with the live thickness knob (no separate preset needed).
    glass = worst.with_coverglass(400.0).power_profile(pos, days)[2][-1]
    # A gentler proton environment and a thicker cover both leave MORE power at the end.
    assert gentle > worst_end
    assert glass > worst_end


def test_power_profile_monotone_and_clamped():
    pos, days = _belt_crossing_path()
    in_belt, ddd, frac = default_radiation_model().power_profile(pos, days)
    assert frac[0] == pytest.approx(1.0) and 0.0 <= frac[-1] <= 1.0
    assert np.all(np.diff(frac) <= 1e-12)                  # never recovers (permanent damage)
    assert np.all(np.diff(ddd) >= -1e-9)                   # dose only accumulates
    assert in_belt.dtype == bool and in_belt.any()
    # Degenerate path -> empty (no crash).
    assert default_radiation_model().power_profile(np.zeros((1, 3)), np.zeros(1))[2].size == 0


def test_sso_worst_case_loss_in_defensible_band():
    # A near-polar climb that punches THROUGH the inner proton belt for many revolutions, the worst
    # belt geometry. With the GaAs/Ge cell behind the default 212 um multi-layer cover the total
    # loss lands ~30-40%; at the thin 75 um / 1.99 reference it is ~44%, consistent with a
    # dedicated OMERE AP8-MIN run (~43% for a matched SSO vehicle). A physics-based bracket, not a
    # fit.
    pos, days = _sso_climb_path()
    m = default_radiation_model()
    loss = 1.0 - m.power_profile(pos, days)[2][-1]                 # default 212 um cover
    assert 0.25 < loss < 0.45
    ref_loss = 1.0 - m.with_coverglass(75.0, 1.9866).power_profile(pos, days)[2][-1]
    assert 0.35 < ref_loss < 0.55                                  # thin reference cover
    # Cover areal density is the dominant lever: much more cover meaningfully cuts the loss.
    loss_500 = 1.0 - m.with_coverglass(500.0).power_profile(pos, days)[2][-1]
    assert loss_500 < 0.75 * loss
    # Electrons are a negligible fraction of the displacement dose (OMERE: ~0.12%); on this path
    # they stay well under a couple percent at any cover, protons dominate, as they must for GaAs.
    proton, electron = m.rate_components(pos)
    dose_p = float(np.sum(0.5 * (proton[:-1] + proton[1:]) * np.diff(days)))
    dose_e = float(np.sum(0.5 * (electron[:-1] + electron[1:]) * np.diff(days)))
    assert dose_e / (dose_p + dose_e) < 0.02


# ---------------------------------------------------------------------------
# synthetic geocentric paths (km, days), cheap stand-ins for a propagated spiral
# ---------------------------------------------------------------------------

def _belt_crossing_path(n: int = 4000):
    """A rising, mildly inclined spiral that dwells in the proton plateau -- a fixed path on which
    different models/coverglasses are compared (the geometry is held constant, only the model varies)."""
    th = np.linspace(0.0, 60.0 * np.pi, n)
    r = np.linspace(2.0 * RE, 5.0 * RE, n)                 # climbs across the proton plateau + slot
    inc = np.radians(20.0)
    pos = np.column_stack([r * np.cos(th),
                           r * np.sin(th) * np.cos(inc),
                           r * np.sin(th) * np.sin(inc)])
    days = np.linspace(0.0, 120.0, n)
    return pos, days


def _sso_climb_path(n: int = 8000):
    """A near-polar (SSO-like) climb from a low circular orbit out through the belts to escape --
    crosses the equatorial proton belt at every node for hundreds of revolutions."""
    th = np.linspace(0.0, 300.0 * np.pi, n)
    r = np.linspace(1.1 * RE, 9.0 * RE, n)
    inc = np.radians(97.0)
    pos = np.column_stack([r * np.cos(th),
                           r * np.sin(th) * np.cos(inc),
                           r * np.sin(th) * np.sin(inc)])
    days = np.linspace(0.0, 175.0, n)
    return pos, days
