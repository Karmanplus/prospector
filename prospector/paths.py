"""The repository's well-known directories, resolved once.

Anything that needs the config library, the caches or the run directories imports them from here
rather than walking up from its own ``__file__``. That walk is a trap: how many ``.parent`` hops it
takes depends on how deeply the module sits, so moving a module into a subpackage quietly redirects
its paths. A config directory that resolves to nothing does not raise: the loader returns an empty
library and the caller falls back to defaults, which shows up as physics that stopped responding to
a setting rather than as an error.

This module stays at the top of the package so its own hop count never changes.
"""
from __future__ import annotations

import os
from pathlib import Path

# prospector/paths.py -> prospector/ -> the repository root.
REPO_ROOT = Path(__file__).resolve().parent.parent

# The environment variable that relocates the config library. Set it to run the tool against a
# different set of engines, missions, vehicles, studies, and sizing coefficients than the one in
# the repository, which is what lets an installation keep its own library outside this tree, and
# what lets the test suite prove the tool works without the repository's own.
CONFIG_DIR_ENV = "PROSPECTOR_CONFIG_DIR"

# The working library, used when nothing overrides it. Untracked by git: it holds an installation's
# own engines, missions, vehicles and sizing coefficients, which are operator data rather than
# source, and are frequently under agreements that must not travel with the code. A fresh checkout
# has no such directory and creates one by copying the example library below.
BUNDLED_CONFIG_DIR = REPO_ROOT / "configs"

# The tracked template: a complete, self-consistent library of synthetic parts that ships with the
# code. It is the starting point for a new installation and the library the test suite runs
# against, so a test cannot come to depend on data that is not in the repository.
EXAMPLE_CONFIG_DIR = REPO_ROOT / "examples" / "configs"


def config_name(name: str) -> str:
    """Check that a config name is one file name inside its kind's directory, never a path.

    The app slugs the names it saves, but a shared library is hand-edited YAML, and a study whose
    ``vehicle: ../x`` would otherwise read a file outside the library. Every loader and saver that
    turns a name into ``<kind>/<name>.yaml`` passes it through here first."""
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"config name {name!r} must be a plain file name, not a path")
    return name


def config_dir() -> Path:
    """The active config library: engines, propellants, launches, returns, radiation, missions,
    vehicles, studies, and the build model.

    Worked out on every call, never captured at import. A module-level constant would be read once,
    when the first importer loads, so anything set afterwards, whether an override or a test
    fixture, would be quietly ignored while every path still looked right. The run root is read per
    call for the same reason.

    ``$PROSPECTOR_CONFIG_DIR`` wins when set, and ``~`` is expanded. The directory does not have to
    exist here: each loader decides for itself whether a missing library is an error, and most do
    raise, since quietly falling back to defaults would report a different vehicle from the
    configured one.
    """
    override = os.environ.get(CONFIG_DIR_ENV)
    return Path(override).expanduser() if override else BUNDLED_CONFIG_DIR

# Gitignored on-disk caches: the SBDB population parquet, Horizons elements, SPK kernels, and the
# per-object enrichment cache.
DATA_DIR = REPO_ROOT / "data"

# Background-job working directories, one subtree per channel.
RUNS_DIR = REPO_ROOT / "runs"

# The detached-job entry point that the job channels spawn.
WORKER = REPO_ROOT / "worker.py"
