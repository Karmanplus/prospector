"""Tests for the mission-target tier rules.

The tier is the single number a user filters on, so every branch is pinned. Two properties matter
beyond the individual cases: unknowns must never promote a body, and they must never be the reason
a body is demoted below the "uncharacterized" tier; an enrichment gap is not evidence against a
target.
"""
import pytest

from prospector.enrichment import tiers

# A structurally sound body: needs almost no cohesion to survive its spin.
SOUND = dict(cohesion_pa=0.2)
# One that needs real strength, but not implausibly much.
MARGINAL = dict(cohesion_pa=2.0)
# One no rubble-pile interpretation survives.
DISQUALIFYING = dict(cohesion_pa=9.0)


def test_valuable_and_sound_is_the_top_tier():
    assert tiers.assign(taxonomy="C", **SOUND) == "S"
    assert tiers.assign(albedo=0.04, **SOUND) == "S", "dark enough counts without a taxonomy"


def test_sound_but_not_valuable_is_one_tier_down():
    assert tiers.assign(taxonomy="S", **SOUND) == "A"


def test_valuable_but_only_marginally_sound_is_one_tier_down():
    assert tiers.assign(taxonomy="C", **MARGINAL) == "A"


def test_valuable_with_nothing_known_about_structure_is_two_tiers_down():
    assert tiers.assign(taxonomy="C") == "B"


def test_marginal_structure_alone_is_two_tiers_down():
    assert tiers.assign(taxonomy="S", **MARGINAL) == "B"


def test_neither_established_is_uncharacterized_not_rejected():
    # C is "nothing is known", which the desirability filter keeps unless a floor is set.
    assert tiers.assign() == "C"
    assert tiers.assign(taxonomy="S") == "C"


def test_disqualifying_cohesion_overrides_everything():
    # Even a perfectly valuable composition cannot rescue a body that cannot be a rubble pile.
    assert tiers.assign(taxonomy="C", **DISQUALIFYING) == "D"


def test_a_high_stability_margin_substitutes_for_a_measured_cohesion():
    """The two structural measures answer the same question from different directions -- this
    body's own spin, and the spin distribution of its size class, so either satisfying the
    bar is enough. A body with only one of them must not be penalised for the missing other."""
    assert tiers.assign(taxonomy="C", stability=0.95) == "S"
    assert tiers.assign(taxonomy="C", stability=0.75) == "A"
    assert tiers.assign(taxonomy="C", stability=0.5) == "B"


def test_metallic_bodies_drop_one_tier():
    assert tiers.assign(taxonomy="M", **SOUND) == "B"     # would be A on structure alone
    assert tiers.assign(taxonomy="M", **DISQUALIFYING) == "D", "D is the floor"


def test_a_fast_tumbling_body_drops_one_tier():
    fast = dict(taxonomy="C", period_h=2.0, **SOUND)
    assert tiers.assign(**fast) == "S"
    assert tiers.assign(tumbling=True, **fast) == "A"


def test_a_slow_tumbling_body_is_not_demoted():
    # Tumbling matters because of what it does to rendezvous; at a long period it does not.
    slow = dict(taxonomy="C", period_h=40.0, **SOUND)
    assert tiers.assign(tumbling=True, **slow) == "S"


def test_tumbling_with_an_unknown_period_is_not_demoted():
    # The demotion is about a fast tumbler specifically. Without a period there is no evidence of
    # that, and an unknown must not cost a target a tier.
    assert tiers.assign(taxonomy="C", tumbling=True, **SOUND) == "S"


@pytest.mark.parametrize("value", [None, float("nan")])
def test_unknown_properties_never_promote(value):
    assert tiers.assign(taxonomy=None, albedo=value, cohesion_pa=value, stability=value) == "C"


def test_demote_bottoms_out():
    assert [tiers.demote(t) for t in tiers.TIER_ORDER] == ["A", "B", "C", "D", "D"]
