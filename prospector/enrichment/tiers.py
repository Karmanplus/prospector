"""Mission-target tiers: one letter summarising whether a body is worth going to.

Reachability says a target can be reached. This says whether it is worth reaching, on two questions
that a mining mission lives or dies by:

**Is there anything there?** Carbonaceous bodies carry the hydrated minerals and volatiles that
make an asteroid worth visiting. That means the C, B and D complexes, plus anything dark enough
to be one whether or not it has been classified. A bright silicate body is a rock.

**Can a spacecraft work on it?** A body that needs essentially no cohesion to survive its own
spin (:mod:`.models.cohesion`) is a gravity-bound rubble pile: it holds together, it can be
anchored to, and material can be removed from it without the whole thing responding. One that
needs several pascals is held by something else and behaves unpredictably when disturbed --
the worst outcome being a body that comes apart under a spacecraft.

The two questions are asked separately and then combined, so the letter says *why*:

===== ==========================================================================
``S`` valuable composition and structurally sound: the targets worth flying
``A`` one of the two clearly satisfied, the other marginal or unmet
``B`` one of the two satisfied, the other unknown or failed
``C`` neither established; not ruled out, just uncharacterized
``D`` structurally disqualifying: cohesion beyond what a rubble pile explains
===== ==========================================================================

Two demotions apply on top. An M-type (metallic) body drops one tier: its composition is
interesting for other reasons but not for volatiles. A tumbling body with a short period drops one
grade as well, since a tumbling body spinning fast is the hardest case there is to rendezvous with
and station-keep against.

Unknowns never promote. A body with no measurements lands in ``C``, which the desirability filter
keeps unless the user asks for a floor, consistent with the rule that this gaps do not reject
targets.
"""
from __future__ import annotations

import math

from prospector.enrichment.measurements import major_class as taxonomy_major

# Tiers best to worst. The index is the rank compared against a user's tier floor.
TIER_ORDER: tuple[str, ...] = ("S", "A", "B", "C", "D")

# Spectral complexes whose members carry hydrated minerals and volatiles.
CARBONACEOUS_COMPLEXES = frozenset({"C", "B", "D"})

# Albedo at or below which a body is dark enough to be treated as carbonaceous even with no
# taxonomy. Bright silicates sit well above this; the carbonaceous complexes sit well below.
DARK_ALBEDO = 0.1

# Cohesion thresholds, in pascals. Below `SOUND` the body is gravity-bound and behaves like a
# rubble pile. Between `SOUND` and `MARGINAL` it needs some strength but stays plausible. Above
# `DISQUALIFYING` no rubble-pile interpretation survives.
SOUND_COHESION_PA = 1.0
MARGINAL_COHESION_PA = 3.0
DISQUALIFYING_COHESION_PA = 5.0

# Probability that the body rotates more slowly than its disruption limit (:mod:`.models.spin`).
# The high bar counts as structurally sound on its own; the lower one counts only as marginal.
SOUND_STABILITY = 0.9
MARGINAL_STABILITY = 0.7

# A tumbling body spinning faster than this (hours) is demoted.
TUMBLING_PERIOD_H = 5.0

_DEMOTION = {"S": "A", "A": "B", "B": "C", "C": "D", "D": "D"}


def _known(value: float | None) -> bool:
    return value is not None and not math.isnan(value)


def demote(tier: str) -> str:
    """One tier worse, bottoming out at ``D``."""
    return _DEMOTION.get(tier, tier)


def assign(*, taxonomy: str | None = None, albedo: float | None = None,
           cohesion_pa: float | None = None, stability: float | None = None,
           period_h: float | None = None, tumbling: bool = False) -> str:
    """The mission-target tier for one body.

    ``cohesion_pa`` is the cohesion its measured spin requires; ``stability`` is ``P(period >
    critical period)`` from the spin distribution. The two answer the same structural question from
    different directions, one from this body's own measurement and one from its size class, so
    either satisfying the bar is enough and a body with only one of them is not penalised for the
    missing other.
    """
    major = taxonomy_major(taxonomy)
    valuable = (major in CARBONACEOUS_COMPLEXES
                or (_known(albedo) and albedo <= DARK_ALBEDO))

    sound = (_known(cohesion_pa) and cohesion_pa < SOUND_COHESION_PA) or \
            (_known(stability) and stability >= SOUND_STABILITY)
    marginal = (_known(cohesion_pa) and SOUND_COHESION_PA <= cohesion_pa <= MARGINAL_COHESION_PA) \
        or (_known(stability) and stability >= MARGINAL_STABILITY)

    if _known(cohesion_pa) and cohesion_pa > DISQUALIFYING_COHESION_PA:
        tier = "D"
    elif valuable and sound:
        tier = "S"
    elif sound or (valuable and marginal):
        tier = "A"
    elif valuable or marginal:
        tier = "B"
    else:
        tier = "C"

    if major == "M":
        tier = demote(tier)
    if tumbling and _known(period_h) and period_h <= TUMBLING_PERIOD_H:
        tier = demote(tier)
    return tier
