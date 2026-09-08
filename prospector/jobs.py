"""
Background jobs: submit a detached worker, poll its status, read its result.

A trajectory solve takes minutes, so it must not block the UI. No threads and no task queue: a job
is a subprocess (``worker.py``) working in its own directory, writing ``status.json`` as it goes,
which the UI polls. This module writes the job spec, starts the worker, and reads the status and
result back. No UI dependency, so the worker and the tests use it too.

Five kinds of job share one mechanism, :class:`JobChannel`, differing only in where their runs
live, which argument the worker takes, whether they detach, and what their status holds:

    SOLVE    runs/<id>/            per-target cruise solve
    ENRICH   runs/enrichment/<id>/ target characterization; not detached, per-target progress
    SPIRAL   runs/spiral/<id>/     escape spiral (a property of the config)
    SWEEP    runs/sweep/<id>/      the escape/cruise trade for one target

Every channel writes the same files, so the UI polls them all alike:

    job.json     the input spec                                    [written by submit]
    status.json  {state, message, error, ...}                       [written by worker]
    summary.json compact run metrics, cheap to list                 [written by worker]
    result.pkl   the result dict (numpy arrays)                     [written by worker]
    worker.log   the worker's stdout/stderr                         [worker process]

Status is written to a temporary file and renamed, so a poll never catches a half-written one, and
every read falls back to a placeholder. A job that has not started looks the same as one whose
status is briefly unreadable, and both mean "keep polling". The ``list_*_runs`` functions read only
the small JSON files, never the heavy result, so a long history stays cheap to draw.

The run root is read as ``paths.RUNS_DIR`` on each call rather than captured at import, so
redirecting it moves every channel at once, which is how the tests stay off real run data. A
module-level import could not be redirected, and the failure would be silent.

The spiral channel also stores a fingerprint (:func:`launch.escape_fingerprint`) of the setup it
was flown for. A finished run whose fingerprint matches replaces the estimated escape budget
(:func:`latest_spiral_match`); editing the vehicle stops it matching.
"""
from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from prospector import paths
from prospector.paths import REPO_ROOT

# status.json states
QUEUED, RUNNING, DONE, ERROR = "queued", "running", "done", "error"


# The status shape common to every channel; a channel may add or replace keys.
_BASE_STATUS = {"state": QUEUED, "stage": "", "progress": 0.0, "message": "", "error": ""}


def age_s(path: Path) -> float:
    """Seconds since ``path`` was last written; infinite when it cannot be stat'd.

    Used to tell whether a process that stopped writing its status has died or is just slow, so a
    file that cannot be reached has to read as infinitely old rather than as fresh.
    """
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return float("inf")


def _read_json(path: Path, *, require_object: bool = False):
    """Parse a small JSON file, or None if it is missing, unreadable, or the wrong shape.

    Every poll goes through this. A job that has not written its status yet and one whose file is
    briefly unreadable both mean "keep polling" rather than "raise". This is the one place here
    where swallowing an error is right, because a missing file is an expected state rather than a
    failure. It returns None rather than a default, so a caller cannot mistake it for real content.

    ``require_object=True`` also rejects valid JSON that is not an object, for callers that go on
    to index the result.
    """
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return None if (require_object and not isinstance(data, dict)) else data




# ---------------------------------------------------------------------------
# The one job mechanism
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JobChannel:
    """One background-job family: where its runs live and how its worker is spawned.

    ``subdir`` is the channel's directory under ``runs/``; None puts runs straight in ``runs/``.
    ``worker_arg`` is the extra argument ``worker.py`` takes to pick this mode, None meaning a
    solve. ``prefix`` goes in front of every file name so two channels can share a directory.
    ``status_defaults`` is returned as-is before anything has been written.

    ``detached`` starts the worker in its own session so it outlives the UI. Enrichment sets it
    False so it stays in the app's process group and a terminal Ctrl-C reaches it.
    """

    name: str
    subdir: str | None = None
    worker_arg: str | None = None
    detached: bool = True
    prefix: str = ""
    status_defaults: dict = field(default_factory=lambda: dict(_BASE_STATUS))

    # ---- locations ----

    @property
    def root(self) -> Path:
        root = paths.RUNS_DIR
        return root / self.subdir if self.subdir else root

    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def path(self, run_id: str, stem: str) -> Path:
        """One of this channel's files inside a run directory, with the channel's prefix."""
        return self.run_dir(run_id) / f"{self.prefix}{stem}"

    # ---- the job spec ----

    def new_run(self, spec: dict, message: str = "submitted") -> str:
        """Create a run directory, write its ``job.json``, mark it queued; return the new id."""
        run_id = uuid.uuid4().hex[:12]
        self.run_dir(run_id).mkdir(parents=True, exist_ok=True)
        self.path(run_id, "job.json").write_text(json.dumps(spec, indent=2))
        self.write_status(run_id, state=QUEUED, message=message)
        return run_id

    def read_job(self, run_id: str) -> dict:
        return json.loads(self.path(run_id, "job.json").read_text())

    # ---- status ----

    def write_status(self, run_id: str, **fields) -> None:
        """Atomically write this channel's status file, defaulting any key not supplied."""
        payload = {**self.status_defaults, **fields}
        if "progress" in payload and payload["progress"] is not None:
            payload["progress"] = float(payload["progress"])
        self._write_atomic(self.path(run_id, "status.json"), json.dumps(payload))

    def read_status(self, run_id: str) -> dict:
        data = _read_json(self.path(run_id, "status.json"))
        return data if isinstance(data, dict) else dict(self.status_defaults)

    # ---- result + summary ----

    def write_result(self, run_id: str, result: dict) -> None:
        self.path(run_id, "result.pkl").write_bytes(pickle.dumps(result))

    def read_result(self, run_id: str) -> dict | None:
        path = self.path(run_id, "result.pkl")
        return pickle.loads(path.read_bytes()) if path.is_file() else None

    def write_summary(self, run_id: str, summary: dict) -> None:
        self.path(run_id, "summary.json").write_text(json.dumps(summary, indent=2))

    def read_summary(self, run_id: str) -> dict:
        data = _read_json(self.path(run_id, "summary.json"))
        return data if isinstance(data, dict) else {}

    # ---- lifecycle ----

    def spawn(self, run_id: str) -> None:
        """Launch ``worker.py`` for this run, streaming its output to the run's log."""
        argv = [sys.executable, str(paths.WORKER), run_id]
        if self.worker_arg:
            argv.append(self.worker_arg)
        log = open(self.path(run_id, "worker.log"), "w")
        try:
            subprocess.Popen(argv, cwd=str(REPO_ROOT), stdout=log,
                             stderr=subprocess.STDOUT,
                             start_new_session=self.detached)
        finally:
            # The child kept its own dup'd fd; don't leak ours in the UI process.
            log.close()

    def delete(self, run_id: str) -> None:
        """Remove a run's directory and everything in it (idempotent)."""
        shutil.rmtree(self.run_dir(run_id), ignore_errors=True)

    @staticmethod
    def _write_atomic(path: Path, text: str) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text)
        tmp.replace(path)          # atomic rename, so a poll never sees a half-written file


SOLVE = JobChannel("solve")
SPIRAL = JobChannel("spiral", subdir="spiral", worker_arg="spiral")
SWEEP = JobChannel("sweep", subdir="sweep", worker_arg="sweep",
                   status_defaults={**_BASE_STATUS, "point": None})
GRID = JobChannel("grid", subdir="grid", worker_arg="grid",
                  status_defaults={**_BASE_STATUS, "cells_done": 0, "cells_total": 0})
ENRICH = JobChannel("enrichment", subdir="enrichment", worker_arg="enrich", detached=False,
                    status_defaults={"state": QUEUED, "done": 0, "total": 0, "dropped": 0,
                                     "message": "", "error": ""})


def is_terminal(status: dict) -> bool:
    return status.get("state") in (DONE, ERROR)


def jsonable_row(row: dict) -> dict:
    """Coerce a population row (possibly a pandas Series) to plain JSON-safe scalars.

    Not the same thing as a ``json.dumps(default=...)`` hook: this walks a mapping and unwraps each
    numpy value in place. Naming the two alike made them look interchangeable when neither can do
    the other's job.
    """
    items = row.items() if hasattr(row, "items") else dict(row).items()
    out = {}
    for k, v in items:
        try:
            out[str(k)] = v.item() if hasattr(v, "item") else v
        except (ValueError, AttributeError):
            out[str(k)] = v
    return out


# ---------------------------------------------------------------------------
# Solve channel: one target's cruise solve
# ---------------------------------------------------------------------------

def run_dir(run_id: str) -> Path:
    return SOLVE.run_dir(run_id)


def record_run(config: dict, target_row: dict, dep_mjd2000: float, arr_mjd2000: float,
               label: str = "", porkchop: dict | None = None,
               options: dict | None = None) -> str:
    """Lay down a run record without spawning a worker; return the id.

    For results worked out elsewhere. The departure-speed sweep already solves every point itself,
    so its winning point is recorded through this and lands in the same place as a solve the user
    started by hand, appearing in the run history and reloading the same way. The caller writes the
    result, summary and final status. :func:`submit` is the version that also starts a worker.
    """
    run_id = SOLVE.new_run({"label": label, "config": config, "target": jsonable_row(target_row),
                            "dep_mjd2000": float(dep_mjd2000),
                            "arr_mjd2000": float(arr_mjd2000), "options": options or {}})
    if porkchop is not None:
        (SOLVE.run_dir(run_id) / "porkchop.json").write_text(json.dumps(porkchop))
    return run_id


def submit(config: dict, target_row: dict, dep_mjd2000: float, arr_mjd2000: float,
           label: str = "", porkchop: dict | None = None,
           options: dict | None = None) -> str:
    """Lay down a job spec and spawn a detached worker; return the new run id.

    ``config`` is a serialized :class:`ResolvedConfig` (``rc.model_dump(mode="json")``), so the job
    captures the study as it stands right now, unsaved edits and all, rather than an older copy
    from disk. ``target_row`` has to carry the target's full orbit (a, e, i, om, w, ma, epoch) plus
    its identifiers, so the worker can rebuild the body without going to the network.
    ``dep_mjd2000`` and ``arr_mjd2000`` are the cell picked on the porkchop, which the solve starts
    from. ``porkchop`` is the JSON-safe version of what that cell was picked from, stored so
    reopening the run can redraw it. ``options`` are solver settings passed straight through.
    """
    run_id = record_run(config, target_row, dep_mjd2000, arr_mjd2000,
                        label=label, porkchop=porkchop, options=options)
    SOLVE.spawn(run_id)
    return run_id


def write_status(run_id: str, **kwargs) -> None:
    SOLVE.write_status(run_id, **kwargs)


def read_status(run_id: str) -> dict:
    return SOLVE.read_status(run_id)


def read_job(run_id: str) -> dict:
    return SOLVE.read_job(run_id)


def write_result(run_id: str, result: dict) -> None:
    SOLVE.write_result(run_id, result)


def read_result(run_id: str) -> dict | None:
    return SOLVE.read_result(run_id)


def write_summary(run_id: str, summary: dict) -> None:
    SOLVE.write_summary(run_id, summary)


def read_summary(run_id: str) -> dict:
    return SOLVE.read_summary(run_id)


def delete_run(run_id: str) -> None:
    SOLVE.delete(run_id)


def read_porkchop(run_id: str) -> dict | None:
    """The stored porkchop context for a run, or None if it wasn't captured / is unreadable."""
    data = _read_json(SOLVE.run_dir(run_id) / "porkchop.json")
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# Enrichment channel: describing the reachable set
#
# Results live in a global per-object cache (``data/enrichment/``), so any job's work benefits
# every other. Each screen still gets its own run directory, so its status reflects that screen's
# count. ``latest.txt`` marks the newest submission, and an older worker winds down when it sees
# a newer one.
# ---------------------------------------------------------------------------

def write_enrich_status(run_id: str, **kwargs) -> None:
    """Write one enrichment job's per-target status (the worker calls this as it goes)."""
    ENRICH.run_dir(run_id).mkdir(parents=True, exist_ok=True)
    ENRICH.write_status(run_id, **kwargs)


def read_enrich_status(run_id: str) -> dict:
    return ENRICH.read_status(run_id)


def read_enrich_ids(run_id: str) -> list[str]:
    return json.loads((ENRICH.run_dir(run_id) / "ids.json").read_text())


def read_enrich_options(run_id: str) -> dict:
    data = _read_json(ENRICH.run_dir(run_id) / "options.json")
    return data if isinstance(data, dict) else {}


def enrich_is_current(run_id: str) -> bool:
    """True if ``run_id`` is the latest enrichment submission (a worker stops when it isn't)."""
    try:
        return (ENRICH.root / "latest.txt").read_text().strip() == run_id
    except OSError:
        return True   # no marker -> assume current (don't stop a lone job)


def cancel_enrich(run_id: str) -> None:
    """Request cancellation; the worker stops at the next target and keeps what it has already
    characterized (everything is cached per target)."""
    d = ENRICH.run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "cancel").write_text("1")


def enrich_cancelled(run_id: str) -> bool:
    return (ENRICH.run_dir(run_id) / "cancel").exists()


def enrich_should_continue(run_id: str) -> bool:
    """The predicate a worker checks to know whether to keep going: still the latest job and not
    cancelled. Drivers poll this per target so a cancel or supersession takes effect promptly."""
    return enrich_is_current(run_id) and not enrich_cancelled(run_id)


def submit_enrich(identifiers: list[str], options: dict | None = None) -> str:
    """Dispatch a background characterization of ``identifiers``; return its run id.

    Each call starts a fresh job and marks it the latest, replacing any older one, which stops at
    its next convenient point. The caller polls :func:`read_enrich_status`.
    """
    ids = [str(x) for x in identifiers]
    run_id = uuid.uuid4().hex[:12]
    d = ENRICH.run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "ids.json").write_text(json.dumps(ids))
    (d / "options.json").write_text(json.dumps(options or {}))
    write_enrich_status(run_id, state=QUEUED, total=len(ids), message="submitted")
    (ENRICH.root / "latest.txt").write_text(run_id)   # supersede any older worker
    ENRICH.spawn(run_id)
    return run_id


# ---------------------------------------------------------------------------
# Spiral channel: the launch-phase escape spiral
#
# A spiral run is a property of the *config* (launch type + vehicle), not of any target, so it
# lives apart from the per-target solves.
# ---------------------------------------------------------------------------

def spiral_run_dir(run_id: str) -> Path:
    return SPIRAL.run_dir(run_id)


def submit_spiral(config: dict, options: dict | None = None,
                  fingerprint: dict | None = None, label: str = "") -> str:
    """Lay down a spiral job and spawn a detached worker; return the new run id.

    ``config`` is the serialized live :class:`ResolvedConfig`; ``options`` are
    :func:`solvers.spiral.solve` knobs (``target_vinf_kms``, ``duty_cycle``, ...); ``fingerprint``
    identifies the physical setup for later matching.
    """
    run_id = SPIRAL.new_run({"label": label, "config": config, "options": options or {},
                             "fingerprint": fingerprint or {}})
    SPIRAL.spawn(run_id)
    return run_id


def write_spiral_status(run_id: str, **kwargs) -> None:
    SPIRAL.write_status(run_id, **kwargs)


def read_spiral_status(run_id: str) -> dict:
    return SPIRAL.read_status(run_id)


def read_spiral_job(run_id: str) -> dict:
    return SPIRAL.read_job(run_id)


def write_spiral_result(run_id: str, result: dict) -> None:
    SPIRAL.write_result(run_id, result)


def read_spiral_result(run_id: str) -> dict | None:
    return SPIRAL.read_result(run_id)


def write_spiral_summary(run_id: str, summary: dict) -> None:
    SPIRAL.write_summary(run_id, summary)


def read_spiral_summary(run_id: str) -> dict:
    return SPIRAL.read_summary(run_id)


def delete_spiral_run(run_id: str) -> None:
    SPIRAL.delete(run_id)


# ---------------------------------------------------------------------------
# Sweep channel: the escape/cruise trade for one target
#
# Runs the cruise solve at several departure speeds and prices each escape off the spiral curve,
# so the cheapest total is visible. Per-target like a solve, but it produces a points table
# rather than one trajectory. Its status carries the latest finished ``point``.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Grid channel: the solved date grid for one target
#
# Per-target like a solve, but a whole surface of trajectories rather than one. Detached because
# it costs tens of seconds. Its status streams the cell count, so the map fills in as it goes.
# ---------------------------------------------------------------------------

def grid_run_dir(run_id: str) -> Path:
    return GRID.run_dir(run_id)


def submit_grid(config: dict, target_row: dict, options: dict | None = None,
                label: str = "") -> str:
    """Lay down a grid job and spawn a detached worker; return the new run id.

    ``config`` is the serialized live :class:`ResolvedConfig` and ``target_row`` carries the
    target's full orbit and identifiers, as for a solve. ``options`` carry the grid terms --
    ``n_dep``, ``n_tof``, ``nseg``, ``restarts``, ``min_tof_days`` and ``polish``, plus
    ``available_power_W``, which has to match the cruise solve's, or the grid describes a different
    vehicle from the one the trajectory is flown with.
    """
    run_id = GRID.new_run({"label": label, "config": config,
                           "target": jsonable_row(target_row), "options": options or {}})
    GRID.spawn(run_id)
    return run_id


def write_grid_status(run_id: str, **kwargs) -> None:
    """Write a grid job's status. ``cells_done``/``cells_total`` report progress, so a
    poller can report how much of the surface exists yet."""
    GRID.write_status(run_id, **kwargs)


def read_grid_status(run_id: str) -> dict:
    return GRID.read_status(run_id)


def read_grid_job(run_id: str) -> dict:
    return GRID.read_job(run_id)


def write_grid_result(run_id: str, result: dict) -> None:
    GRID.write_result(run_id, result)


def read_grid_result(run_id: str) -> dict | None:
    return GRID.read_result(run_id)


def write_grid_summary(run_id: str, summary: dict) -> None:
    GRID.write_summary(run_id, summary)


def read_grid_summary(run_id: str) -> dict:
    return GRID.read_summary(run_id)


def sweep_run_dir(run_id: str) -> Path:
    return SWEEP.run_dir(run_id)


def submit_sweep(config: dict, target_row: dict, options: dict | None = None,
                 label: str = "") -> str:
    """Lay down a sweep job and spawn a detached worker; return the new run id.

    ``config`` is the serialized live :class:`ResolvedConfig` and ``target_row`` carries the
    target's full orbit and identifiers, as for a solve. ``options`` carry the sweep terms --
    ``vinf_values``, the departure speeds to try in km/s, or ``vinf_cap_kms`` for the default set,
    and ``curve``, the spiral's delta-v against departure speed. Any solver settings are are passed
    on to every point.
    """
    run_id = SWEEP.new_run({"label": label, "config": config,
                            "target": jsonable_row(target_row), "options": options or {}})
    SWEEP.spawn(run_id)
    return run_id


def write_sweep_status(run_id: str, **kwargs) -> None:
    """Write a sweep job's status. ``point`` is the latest finished sweep point (a JSON-safe
    dict), streamed so a poller can grow the tradeoff curve live."""
    SWEEP.write_status(run_id, **kwargs)


def read_sweep_status(run_id: str) -> dict:
    return SWEEP.read_status(run_id)


def read_sweep_job(run_id: str) -> dict:
    return SWEEP.read_job(run_id)


def write_sweep_result(run_id: str, result: dict) -> None:
    SWEEP.write_result(run_id, result)


def read_sweep_result(run_id: str) -> dict | None:
    return SWEEP.read_result(run_id)


def write_sweep_summary(run_id: str, summary: dict) -> None:
    SWEEP.write_summary(run_id, summary)


def read_sweep_summary(run_id: str) -> dict:
    return SWEEP.read_summary(run_id)


def delete_sweep_run(run_id: str) -> None:
    SWEEP.delete(run_id)

