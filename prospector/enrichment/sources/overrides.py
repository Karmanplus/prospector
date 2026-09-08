"""Hand-entered physical properties that outrank every catalogue.

Published databases lag the literature, disagree, and occasionally carry a value that is simply
wrong for a body someone on the team has looked at closely. This file is where that knowledge goes:
a designation, the properties to override, and nothing else. Every value it supplies is flagged
preferred, so it wins against any survey.

It lives in the config library rather than in source, like every other input the tool reads (see
``configs/`` and ``$PROSPECTOR_CONFIG_DIR``), and it is optional: an installation with nothing to
correct simply does not have the file.

Format, keyed by the designation the screen uses:

.. code-block:: yaml

    "2008 EV5":
      taxonomy: C
      albedo: 0.12
    "1998 KG3":
      period_h: 13.335
      tier: A

Recognised properties are :data:`OVERRIDABLE`. ``tier`` is special: it replaces the computed
mission tier outright rather than feeding into it, for a body whose tier was decided by a judgement
the model does not capture.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from prospector import paths
from prospector.enrichment.measurements import Measurement

# Properties an override may set. Anything else in the file is ignored rather than rejected, so a
# comment key or a note field costs nothing, but a typo in one of these is silently inert, which is
# why the set is small and fixed.
OVERRIDABLE: tuple[str, ...] = ("abs_mag", "albedo", "taxonomy", "period_h", "tier")

_cache: tuple[Path, dict[str, dict]] | None = None


def overrides_path() -> Path:
    """The override file in the active config library. Resolved per call, never at import."""
    return paths.config_dir() / "enrichment-overrides.yaml"


def load(path: Path | None = None) -> dict[str, dict]:
    """Every override in the file, keyed by designation. Empty when there is no file.

    Cached against the resolved path, so pointing the config library somewhere else picks up that
    library's overrides rather than the previous one's.
    """
    global _cache
    target = Path(path) if path is not None else overrides_path()
    if _cache is not None and _cache[0] == target:
        return _cache[1]

    entries: dict[str, dict] = {}
    if target.is_file():
        try:
            loaded = yaml.safe_load(target.read_text()) or {}
        except Exception:
            loaded = {}
        if isinstance(loaded, dict):
            entries = {str(key).strip(): value for key, value in loaded.items()
                       if isinstance(value, dict)}
    _cache = (target, entries)
    return entries


def for_target(identifier: str) -> dict[str, list[Measurement]]:
    """The overrides for one body, as measurement lists flagged preferred.

    ``tier`` is not returned here: it is not a measurement of anything, and :func:`tier_override`
    keeps it separate so it cannot be consolidated into a property list by accident.
    """
    entry = load().get(identifier.strip(), {})
    return {name: [Measurement(entry[name], source="override", preferred=True)]
            for name in OVERRIDABLE if name != "tier" and entry.get(name) is not None}


def tier_override(identifier: str) -> str | None:
    """The hand-assigned mission tier for one body, if the file sets one."""
    value = load().get(identifier.strip(), {}).get("tier")
    return str(value).strip().upper() if isinstance(value, str) and value.strip() else None
