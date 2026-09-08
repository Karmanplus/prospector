"""Session bookkeeping for one design-space search: its status file and its stop signal.

The app polls ``status.json``, and asks it to stop by dropping a ``cancel`` file, the same way the
worker jobs do. Kept apart from the search itself so everything passing between the search and
whatever is watching it fits in one short module.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from prospector.paths import REPO_ROOT


class Cancelled(Exception):
    """Raised between evaluations when the session's ``cancel`` file appears (the app's
    Stop button); the search winds down and writes whatever it has."""


def write_session_status(out: Path, *, state: str, message: str,
                         n_evaluated: int = 0,
                         progress: dict | None = None) -> None:
    """Write the session's ``status.json`` in one go, since the app's vehicle search tab polls
    it the same way it polls the worker jobs. ``progress`` carries what the tab's status bar
    counts against, which is ``n_planned``, the combination count the axes imply."""
    payload = {"state": state, "message": message, "n_evaluated": int(n_evaluated),
               "updated": datetime.now().isoformat(timespec="seconds")}
    if progress:
        payload.update(progress)
    tmp = out / "status.json.tmp"
    tmp.write_text(json.dumps(payload))
    tmp.replace(out / "status.json")


# Where sessions land, and where propagated escape curves are cached between them.
DEFAULT_OUT_ROOT = REPO_ROOT / "runs" / "vehicle_search"


SPIRAL_CACHE_DIR = DEFAULT_OUT_ROOT / "spiral_cache"
