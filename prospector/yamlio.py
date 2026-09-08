"""Writing a config file back without destroying what it says about itself.

Every file in the config library is annotated by hand: why a coefficient has the value it has,
which paper it came from, which of two scenarios is the cautious one. Those comments are the only
place that reasoning lives, since the numbers cannot carry it themselves.

A plain ``yaml.safe_dump`` writes the file out from scratch, so every comment in it disappears the
first time anyone presses Save. That was survivable while the library was edited by hand and saving
was rare. It is not survivable once the app is how these values get changed, because then the
library loses a bit of its own documentation each time somebody adjusts a number.

So :func:`dump_preserving_comments` edits the existing file in place, keeping the same keys, order
and comments and changing only the values. It falls back to a plain write for a file that does not
exist yet. Keys the model no longer has are dropped and new ones are appended, so the file keeps up
with the model.

:func:`warn_unknown_keys` covers the other half of the problem. Pydantic ignores a key it does not
recognise, which is deliberate: it lets a file survive a coefficient being renamed instead of
failing to load. But ignoring it without saying so means a value somebody tuned can sit in the file
looking authoritative while nothing reads it, and the only symptom is a setting that stopped
mattering.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import yaml

try:                                        # pragma: no cover - exercised by both branches
    from ruamel.yaml import YAML as _RuamelYAML
except ImportError:                         # pragma: no cover
    _RuamelYAML = None


def _round_tripper():
    """A ruamel round-trip parser configured to leave formatting alone, or None if absent."""
    if _RuamelYAML is None:
        return None
    ruamel = _RuamelYAML()
    ruamel.preserve_quotes = True
    # Never re-wrap a long line: rewrapping produces a diff on lines nobody edited, which buries
    # the one value that changed.
    ruamel.width = 4096
    # Write an absent value as ``null``, not as nothing after the colon. Both parse the same, but
    # the round-trip default is the empty form, so every file holding a null would come back
    # rewritten the first time anything else in the library was saved, producing a diff on a file
    # nobody edited, which is the noise this module exists to avoid.
    ruamel.representer.add_representer(
        type(None), lambda dumper, _value: dumper.represent_scalar("tag:yaml.org,2002:null", "null"))
    return ruamel


def dump_preserving_comments(data: dict[str, Any], path: str | Path) -> Path:
    """Write ``data`` to ``path``, keeping the comments and layout already there.

    Values are updated in place in the existing document, so the file's comments, key order and
    quoting survive. Keys absent from ``data`` are removed (the model no longer has them) and keys
    it adds are appended.

    Falls back to a plain dump when the file is new, or when the round-trip parser is not
    installed. In that case the write still succeeds and only the comments are lost, which is the
    behaviour without this module at all.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ruamel = _round_tripper()

    if ruamel is not None and path.is_file():
        try:
            with path.open("r", encoding="utf-8") as handle:
                document = ruamel.load(handle)
        except Exception:
            document = None
        if isinstance(document, dict):
            _update_in_place(document, data)
            with path.open("w", encoding="utf-8") as handle:
                ruamel.dump(document, handle)
            return path

    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _update_in_place(document: dict, data: dict[str, Any]) -> None:
    """Reconcile a loaded document with new values, touching as little as possible.

    A value that already matches is left completely alone: assigning it back would replace the node
    carrying its comment with a bare number, and the comment would go with it.
    """
    for key, value in data.items():
        if key in document and document[key] == value:
            continue
        document[key] = value
    for key in [k for k in document if k not in data]:
        del document[key]


def validated(model_cls, path: str | Path):
    """Parse ``path`` and validate it as ``model_cls``, warning about keys the model ignores.

    The one place a config file becomes a model, so the "this key does nothing" warning cannot be
    forgotten by a loader that was written later.
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    if isinstance(data, dict):
        warn_unknown_keys(data, model_cls.model_fields, path)
    return model_cls.model_validate(data)


def warn_unknown_keys(data: dict[str, Any], fields, source: str | Path) -> None:
    """Warn about keys in a config file that the model does not read.

    Loading puts up with them, and this makes that visible. A file carrying a renamed or retired
    coefficient looks authoritative, since somebody tuned that number, while doing nothing at all,
    and without a warning the only way to find out is to notice that changing it has no effect.
    """
    unknown = [k for k in data if k not in fields]
    if unknown:
        warnings.warn(
            f"{source}: ignoring {len(unknown)} key(s) this model does not read: "
            f"{', '.join(sorted(unknown))}",
            stacklevel=3,
        )
