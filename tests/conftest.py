"""Shared pytest configuration.

Registers NiceGUI's pure-Python ``user`` fixture for the NiceGUI app smoke test. The
``user_plugin`` is used rather than the combined ``plugin`` so the test suite needs no
selenium/browser dependency (the combined plugin's ``Screen`` fixture imports selenium).

The ``user`` fixture is async, so the async tests are driven by anyio (already in the environment
via Starlette) pinned to the asyncio backend -- no extra test dependency.
"""
import contextlib
import multiprocessing
import os
import pathlib
import shutil
import signal
import sys
import threading
from pathlib import Path

import pytest

from prospector import paths

pytest_plugins = ["nicegui.testing.user_plugin"]

# Start worker processes by spawning, not forking.
#
# The pools these tests drive fork by default, which is right where they normally run: inside the
# single-threaded detached worker. Pytest is not that. Several tests start threads, and forking a
# threaded process can deadlock the child on a lock nobody will release. It is timing-dependent,
# so it passes here and hangs on a two-core runner, where it took the suite green and then held
# the job open until it was cancelled.
multiprocessing.set_start_method("spawn", force=True)

# Point the suite at the EXAMPLE config library, not the repository's own.
#
# Nothing in the suite may depend on proprietary data. The bundled library holds vendor engine
# performance and company mission concepts; a test that names one of those keys silently makes the
# suite undistributable, and the failure only shows up when somebody tries to hand the tool over.
# Reading the example library means such a dependency fails here instead. It also proves the tool
# works on a library it did not grow up with, which is what a recipient will be doing.
#
# Set here at import rather than in a fixture: a session fixture runs after collection, and test
# modules that read the library while being imported would already have failed. On a machine that
# happens to have a `configs/` directory the default path resolves and hides it, so this only
# shows up on a clean checkout.
#
# An externally-set PROSPECTOR_CONFIG_DIR is honoured, so this can be pointed at the bundled
# library to check the private data still resolves.
if not os.environ.get(paths.CONFIG_DIR_ENV):
    os.environ[paths.CONFIG_DIR_ENV] = str(paths.REPO_ROOT / "examples" / "configs")


@pytest.fixture(autouse=True)
def _keep_main_module():
    """Put ``sys.modules['__main__']`` back after every test.

    NiceGUI's ``user`` fixture runs the app module through ``runpy`` under that name and drops the
    entry when it tears down. Spawning a worker process reads it to work out what the child should
    import, so any later test that starts a pool dies with ``KeyError: '__main__'``. The order
    tests run in is shuffled, so without this the failure lands on a different test each run, or
    on none.
    """
    saved = sys.modules.get("__main__")
    yield
    if saved is not None:
        sys.modules["__main__"] = saved


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    session.config._prospector_exit_status = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """Reap anything still running once pytest is done.

    concurrent.futures joins every executor's worker at interpreter shutdown, so one child that
    outlives the run leaves pytest hanging with its results already printed: the suite reads as
    passed and the process never exits. Anything still here is a leak, so name it and end it
    rather than block on it. Children are read from /proc because a leaked process need not be a
    multiprocessing one, and multiprocessing.active_children() would not see it.
    """
    for child in multiprocessing.active_children():
        child.terminate()
        child.join(timeout=5)

    kids = pathlib.Path("/proc/self/task")
    if not kids.is_dir():
        return
    pids: set[int] = set()
    for task in kids.iterdir():
        try:
            pids.update(int(p) for p in (task / "children").read_text().split())
        except (OSError, ValueError):
            pass
    for pid in sorted(pids):
        try:
            cmd = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue
        print(f"\nreaping a process that outlived the session: {pid} {cmd.strip()[:120]}")
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)

    # A thread cannot be reaped the way a process can, and a non-daemon one that never finishes
    # blocks interpreter shutdown after the results are already printed: the suite reads as passed
    # and the process never exits. Name any that are still running, then leave without waiting.
    blocking = [t for t in threading.enumerate()
                if t is not threading.current_thread() and not t.daemon]
    if not blocking:
        return
    # Kaleido runs its browser on a non-daemon thread and does not always end it, and Python joins
    # every non-daemon thread before it exits. That join is what leaves a run reporting its
    # results and then never finishing. Nothing here owns state worth waiting for by this point,
    # so say what is still up and go. Runs last, so a plugin that writes on its way out, coverage
    # among them, has already done so.
    for t in blocking:
        print(f"not waiting for a thread that outlived the session: {t.name} ({t})")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(getattr(config, "_prospector_exit_status", 0)))


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def isolate_runs(tmp_path_factory, monkeypatch):
    """Point the run root at a scratch directory for every test.

    The job channels resolve ``paths.RUNS_DIR`` at call time, so this one redirect moves all five
    of them. It is autouse because the failure mode it prevents is silent rather than loud: a test
    that submits a job without redirecting would write into the developer's real ``runs/`` tree and
    read whatever is already there. That happened: four enrichment tests once passed only because
    they redirected, and reading real state made them pass for the wrong reason.

    Tests that assert on file layout still redirect to their own ``tmp_path`` explicitly; this is
    the floor under them, not a replacement.
    """
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path_factory.mktemp("runs"))


@pytest.fixture(autouse=True)
def isolate_config_library(tmp_path_factory, monkeypatch):
    """Give every test its own copy of the config library, so none can edit the real one.

    The counterpart to :func:`isolate_runs`, and autouse for the same reason: the failure is
    silent. The library the suite reads is ``examples/configs/``, which is TRACKED, and the app's
    save paths are ordinary library writers -- so a test that exercises one (``persist_study``,
    ``save_vehicle``, the engine-library dialog) rewrites the repository's own example data. The
    suite still passes; the corruption surfaces later as an unexplained diff, or as a test that
    only passes because a previous run left the value it wanted behind.

    A copy rather than a redirect to an empty directory: nearly every test needs a library that
    resolves, and the point is to keep the writes off the tracked one, not to remove the data. The
    library is small (tens of files), so copying it per test costs less than the debugging one
    silent overwrite costs.

    Tests that assert on library layout still redirect explicitly; ``paths.config_dir()`` reads the
    environment on every call, so a later override wins over this one.
    """
    source = Path(os.environ[paths.CONFIG_DIR_ENV])
    if not source.is_dir():
        return                      # a suite pointed at a library that isn't there: leave it alone
    scratch = tmp_path_factory.mktemp("configs") / source.name
    shutil.copytree(source, scratch)
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(scratch))
