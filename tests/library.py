"""Keys resolved from whichever config library the suite is running against.

Tests must not name engines, missions, or studies literally. A literal key ties the suite to one
library's contents, and there are two libraries that matter: the repository's own (vendor engines,
company mission concepts) and the shipped example set. A suite bound to either one cannot verify
the other, and a suite bound to the private one cannot be distributed at all.

So tests ask for "an engine" or "a study" and get whatever the active library holds. Selection is
deterministic (sorted) so a failure is reproducible, and each helper raises with a clear message
rather than returning something empty, a test that silently ran against no engine would pass for
the wrong reason.

Where a test needs a specific CAPABILITY rather than just any engine, a throttle curve, a
particular thrust class; it should build that engine inline instead of hunting the library for one.
:func:`curved_engine` exists for the throttle-curve case.
"""
from __future__ import annotations

from prospector.config import list_missions, list_studies
from prospector.spacecraft.propulsion import Engine, PowerPoint, load_engines


def engine_keys() -> list[str]:
    """Every engine key in the active library, sorted. Raises if the library is empty."""
    keys = sorted(load_engines())
    if not keys:
        raise AssertionError("the active engine library is empty; tests need at least one engine")
    return keys


def engine_key() -> str:
    """One engine key from the active library -- when a test needs *an* engine, not a specific one."""
    return engine_keys()[0]


def engine_key_at(index: int) -> str:
    """The nth engine key, wrapping. For tests that need two or three DISTINCT engines (a mixed
    assembly, a multi-engine sweep) without caring which."""
    keys = engine_keys()
    return keys[index % len(keys)]


def study_key() -> str:
    """One study key from the active library. Raises if the library ships none."""
    keys = sorted(list_studies())
    if not keys:
        raise AssertionError("the active library ships no studies; tests need at least one")
    return keys[0]


def mission_key() -> str:
    """One mission key from the active library. Raises if the library ships none."""
    keys = sorted(list_missions())
    if not keys:
        raise AssertionError("the active library ships no missions; tests need at least one")
    return keys[0]


def curved_engine(key: str = "curved") -> tuple[str, Engine]:
    """An engine with a measured throttle curve, built here rather than found in the library.

    Thrust AND Isp fall together along the curve when the bus cannot supply rated power, which is
    the behaviour some tests exist to check. Whether any given library happens to ship an engine
    with curve data is not something a test should depend on.
    """
    return key, Engine(
        name="Curved test thruster", source="synthetic - test fixture",
        isp_s=2000.0, thrust_mN=100.0, power_W=2000.0, mass_kg=10.0,
        power_curve=[PowerPoint(power_W=800.0, thrust_mN=35.0, isp_s=1500.0),
                     PowerPoint(power_W=1400.0, thrust_mN=70.0, isp_s=1800.0),
                     PowerPoint(power_W=2000.0, thrust_mN=100.0, isp_s=2000.0)])
