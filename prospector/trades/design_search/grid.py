"""Parsing what to search: engine pairings and swept model axes.

Argument handling only, and all of it fails loudly. An unknown engine name, a missing count, a
sweep value that is not a number, or a setting that cannot be swept raises rather than being
skipped: quietly dropping one would run a search covering less ground than was asked for.
"""
from __future__ import annotations

from prospector.spacecraft import buildability


def parse_model_axes(specs: list[str] | None) -> dict[str, list[float]]:
    """Parse ``--sweep-param KEY=v1,v2,...`` specs into ``{key: [values]}``.

    Keys are checked against :data:`prospector.spacecraft.buildability.SWEEPABLE`, so a setting
    that cannot be swept fails immediately instead of sweeping a value that goes nowhere.
    Repeating a key keeps the last list given for it."""
    axes: dict[str, list[float]] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise SystemExit(f"--sweep-param needs KEY=v1,v2,..., got {spec!r}")
        key, raw = spec.split("=", 1)
        key = key.strip()
        if key not in buildability.SWEEPABLE:
            raise SystemExit(f"unknown sweep key {key!r}; choose from "
                             f"{', '.join(buildability.SWEEPABLE)}")
        try:
            values = [float(v) for v in raw.split(",") if v.strip() != ""]
        except ValueError as exc:
            raise SystemExit(f"--sweep-param {key} values must be numbers, "
                             f"got {raw!r}") from exc
        if not values:
            raise SystemExit(f"--sweep-param {key} lists no values")
        axes[key] = values
    return axes


def model_combinations(axes: dict[str, list[float]]) -> list[dict]:
    """Cartesian product of the swept model axes -> one override dict per combination.

    No axes -> a single empty override, i.e. the model's configured defaults."""
    combos: list[dict] = [{}]
    for key, values in axes.items():
        combos = [{**c, key: v} for c in combos for v in values]
    return combos


def parse_pairs(specs: list[str], catalog: dict) -> list[tuple[str, int]]:
    """Parse engine/count pairings: ``ENGINE:1,2`` or ``ENGINE:3-6`` -> [(key, n)...].

    Counts accept comma-separated lists and inclusive ranges ('1,2', '3-6', '1,3-5'), and engine
    names are checked against the library. Order is kept and exact repeats dropped, so one run can
    sweep, say, one or two big thrusters as well as three to six small ones.
    """
    pairs, seen = [], set()
    for spec in specs:
        key, _, counts = spec.partition(":")
        key = key.strip()
        if key not in catalog:
            raise SystemExit(f"unknown engine {key!r} in pairing {spec!r}; "
                             f"available: {', '.join(sorted(catalog))}")
        if not counts.strip():
            raise SystemExit(f"pairing {spec!r} needs counts, e.g. "
                             f"'{key}:1,2' or '{key}:3-6'")
        for token in counts.split(","):
            token = token.strip()
            try:
                if "-" in token:
                    a, b = token.split("-", 1)
                    ns = list(range(int(a), int(b) + 1))
                else:
                    ns = [int(token)]
            except ValueError:
                raise SystemExit(f"bad count {token!r} in pairing {spec!r}") from None
            for n in ns:
                if n < 1:
                    raise SystemExit(f"engine count must be >= 1 (got {n} in {spec!r})")
                if (key, n) not in seen:
                    seen.add((key, n))
                    pairs.append((key, n))
    return pairs
