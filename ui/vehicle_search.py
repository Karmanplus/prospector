"""Data access for the Compare-vehicles workspace: read + launch design-space sweeps.

A thin layer, with no UI framework in it, over the sessions
:mod:`prospector.trades.design_search` writes under ``runs/vehicle_search/<session>/``. The
search writes ``status.json`` as it goes plus one line of ``evaluations.jsonl`` per vehicle
evaluated, then leaves ``result.json``, ``combos.csv`` and ``REPORT.md`` behind. This module
finds those sessions, loads whatever rows they left, and offers the one thing that changes
anything: starting a fresh search as a detached subprocess, the same arrangement the worker jobs
use, so it survives the app restarting and the workspace just polls its status.

No ``nicegui`` import here. This is the plain data the workspace (``workspaces/vehicle.py``) draws,
the same way the ``prospector`` core feeds the other workspaces.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from prospector import jobs, paths
from prospector.paths import REPO_ROOT
from prospector.spacecraft import buildability
from prospector.spacecraft.propulsion import load_engines


def search_root() -> Path:
    """Where design-search sessions live. Resolved on each call, like the job channels, so a
    redirected run root (the test suite) moves this too."""
    return paths.RUNS_DIR / "vehicle_search"


def example_root() -> Path:
    """Sweeps shipped with the example library. Listed like the user's own, never written to."""
    return paths.EXAMPLE_RUNS_DIR / "vehicle_search"
# The search is launched as a module, not a script path, so this layer depends on the library's CLI
# contract rather than on a file's location on disk.
SEARCH_MODULE = "prospector.trades.design_search"
# A running search prints one verdict per evaluation, minutes apart; a status this stale with no
# terminal state means the process is gone, not slow.
STALE_S = 900

# The verdict columns the workspace renders, kept in one place so the loaders agree.
_VERDICT_COLUMNS = (
    "dry_kg", "prop_kg", "wet_kg", "prop_dry_ratio", "n_engines", "engine",
    "propellant", "propellant_key", "propellant_estimated", "anchor",
    "thrust_mN_per_kg", "power_W_per_kg", "thrust_mN_per_W",
    "buildable", "build_why", "lifetime_ok", "lifetime_why",
    "prop_per_thruster_kg", "throughput_margin_kg", "mission_ignitions",
    "payload_capacity_kg", "min_dry_kg", "eol_factor", "belt_days", "bol_power_W",
    "array_kg", "pcdu_kg", "thruster_sys_kg", "tank_kg", "fixed_bus_kg",
    "build_cost_musd", "success", "why", "margin_kms", "dv_short_kms", "prop_margin_kg",
    "capability_kms", "escape_dv_kms", "cruise_dv_kms", "total_dv_kms",
    "spiral_tof_days", "cruise_tof_days", "total_tof_days", "n_converged", "elapsed_s",
    # Escape power adequacy (the array must carry full thrust through end-of-life).
    "power_limited", "power_required_W", "power_available_end_W",
    # Return-leg columns (NaN/None for one-way sessions and sessions predating the feature).
    "return_dv_kms", "return_prop_kg", "return_tof_days", "return_insertion_dv_kms",
    "insertion_prop_kg", "delivered_payload_kg", "total_prop_kg", "round_trip_days",
)

_BUILD_COLUMNS = (
    "buildable", "build_why", "propellant", "lifetime_ok", "lifetime_why",
    "prop_per_thruster_kg", "throughput_margin_kg", "mission_ignitions",
    "payload_capacity_kg", "min_dry_kg", "eol_factor", "belt_days", "bol_power_W",
    "array_kg", "pcdu_kg", "thruster_sys_kg", "tank_kg", "fixed_bus_kg", "build_cost_musd",
)


# ---------------------------------------------------------------------------
# session discovery & loading (plain file reads; no state, no mutation)
# ---------------------------------------------------------------------------

# Sessions are named for their launch instant (``YYYYMMDD-HHMMSS``); only dated ones are listed
# (ad-hoc/CLI runs with arbitrary --out names are skipped), and the timestamp is the sort key.
_STAMP_FMT = "%Y%m%d-%H%M%S"


def _session_time(name: str) -> datetime | None:
    try:
        return datetime.strptime(name, _STAMP_FMT)
    except ValueError:
        return None


def list_sessions() -> list[dict]:
    """The dated search sessions, newest first: name, a readable launch time, state, target,
    and the session metadata. Sessions whose directory name isn't a timestamp are skipped."""
    sessions = []
    roots = [(search_root(), False), (example_root(), True)]
    for root, example in roots:
        if root.is_dir():
            sessions.extend(_sessions_in(root, example))
    sessions.sort(key=lambda s: s["stamp"], reverse=True)
    return sessions


def _sessions_in(root: Path, example: bool) -> list[dict]:
    sessions = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name == "spiral_cache":
            continue
        when = _session_time(d.name)
        if when is None:
            continue                      # not a dated run -> not shown
        status = read_json(d / "status.json")
        artifacts = [p for p in (d / "result.json", d / "combos.csv",
                                 d / "evaluations.jsonl") if p.is_file()]
        if status is None and not artifacts:
            continue
        meta = read_json(d / "session.json") or {}
        sessions.append({"name": d.name, "dir": d, "when": when.strftime("%m/%d/%y %H:%M"),
                         "stamp": when, "state": session_state(d, status),
                         "message": (status or {}).get("message", ""),
                         "target": meta.get("target"), "meta": meta, "example": example})
    return sessions


def sessions_for_project(sessions: list[dict], study_key: str | None,
                         mission_key: str | None, target_pdes: str | None) -> list[dict]:
    """The sessions that belong to one project, in the order given.

    A session records the study that launched it (``session.json["study"]``), and that is the
    match. Sessions from before the study was recorded carry only the mission key and the target,
    so those are claimed by a project whose mission AND default target match both; a project with
    no default target claims none of them, since the mission alone is shared by many studies.
    """
    if not study_key:
        return list(sessions)
    kept = []
    for s in sessions:
        meta = s.get("meta") or {}
        study = meta.get("study")
        if study:
            if study == study_key:
                kept.append(s)
            continue
        if (mission_key and target_pdes and meta.get("mission") == mission_key
                and str(meta.get("pdes") or "") == str(target_pdes)):
            kept.append(s)
    return kept


def session_state(d: Path, status: dict | None) -> str:
    """'running' / 'done' / 'error' / 'stopped' for one session directory.

    ``status.json`` is authoritative when present, with one correction: a session it calls
    'running' whose process has gone, having crashed or been killed before writing a terminal
    state, is reported 'stopped'. Liveness is read two ways: the recorded PID is probed directly
    (instant, the common case) and, as a portable backstop for sessions with no PID or on platforms
    where the probe is unavailable, a status stamp older than ``STALE_S`` is treated as dead. A
    session with only artifacts is 'done'.
    """
    if status is not None:
        state = status.get("state", "")
        if state == "running":
            pid = _read_pid(d)
            if pid is not None and not _process_alive(pid):
                return "stopped"
            if jobs.age_s(d / "status.json") > STALE_S:
                return "stopped"
        return state
    return "done" if (d / "result.json").is_file() else "stopped"


def load_rows(d: Path) -> pd.DataFrame:
    """Every evaluated vehicle in one session, best designs first.

    Prefers the final ``combos.csv``; falls back to the live ``evaluations.jsonl`` stream so a
    session is visible at whatever fidelity it has reached.
    """
    if (d / "combos.csv").is_file():
        try:
            return order(_with_mission_times(pd.read_csv(d / "combos.csv"), d))
        except pd.errors.EmptyDataError:
            pass                          # a freshly created, header-less file -> try the stream
    jsonl = d / "evaluations.jsonl"
    if jsonl.is_file():
        rows = []
        for line in jsonl.read_text().splitlines():
            try:
                v = json.loads(line)
            except json.JSONDecodeError:
                continue                      # a half-written tail line; the next poll has it
            # The fixed verdict columns plus any swept model settings (param_<key>), so a
            # session's --sweep-param axes survive the projection and become plottable
            # (the final combos.csv keeps them as plain columns).
            rows.append({**{k: v.get(k) for k in _VERDICT_COLUMNS},
                         **{k: val for k, val in v.items() if k.startswith("param_")}})
        if rows:
            return order(pd.DataFrame(rows))
    return pd.DataFrame()


def _with_mission_times(df: pd.DataFrame, d: Path) -> pd.DataFrame:
    """Recover the best-point columns (mission times + the ΔV ledger) for sessions whose
    ``combos.csv`` predates them, derived from each verdict's best sweep point in the
    ``evaluations.jsonl`` stream. Sessions already carrying the columns pass through."""
    have = all(c in df.columns and df[c].notna().any()
               for c in ("cruise_tof_days", "cruise_dv_kms"))
    if df.empty or have:
        return df
    jsonl = d / "evaluations.jsonl"
    if not jsonl.is_file():
        return df
    best: dict[tuple, dict] = {}
    for line in jsonl.read_text().splitlines():
        try:
            v = json.loads(line)
        except json.JSONDecodeError:
            continue
        bp = v.get("best_point") or {}
        cruise = bp.get("tof_days")
        key = (str(v.get("engine")), float(v.get("dry_kg", 0)),
               float(v.get("prop_kg", 0)), int(v.get("n_engines", 0)))
        best[key] = {
            "cruise_tof_days": cruise,
            "total_tof_days": (None if cruise is None else
                               float(bp.get("escape_tof_days") or 0.0) + float(cruise)),
            "cruise_dv_kms": bp.get("cruise_dv_kms"),
            "total_dv_kms": bp.get("total_dv_kms"),
        }
    if not best:
        return df
    df = df.copy()
    keys = [(str(r.get("engine")), float(r.get("dry_kg", 0)),
             float(r.get("prop_kg", 0)), int(r.get("n_engines", 0)))
            for _, r in df.iterrows()]
    for col in ("cruise_tof_days", "total_tof_days", "cruise_dv_kms", "total_dv_kms"):
        if col not in df.columns or df[col].isna().all():
            df[col] = [best.get(k, {}).get(col) for k in keys]
    return df


def flag(df: pd.DataFrame, column: str) -> pd.Series:
    """A boolean view of a verdict column that may arrive as bools, CSV strings, or NaN --
    anything not affirmatively true is False."""
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].map(lambda v: v is True or str(v).strip().lower() == "true")


def order(df: pd.DataFrame) -> pd.DataFrame:
    """Best designs first. A design that both flies and can be built, at the lightest wet mass,
    leads, the goal being the lightest workable vehicle. Then designs that fly but cannot be
    built, then the ones that miss, ordered by how far."""
    if df.empty:
        return df
    df = df.copy()
    df["_both"] = flag(df, "success") & flag(df, "buildable")
    df["_wet"] = pd.to_numeric(df.get("wet_kg"), errors="coerce")
    # Ranked by the propellant left in the tank, the verdict's own quantity (a session without the
    # column, an older one, simply ranks on the rest).
    df["_margin"] = pd.to_numeric(df.get("prop_margin_kg"), errors="coerce")
    df = df.sort_values(["_both", "success", "_wet", "_margin"],
                        ascending=[False, False, True, False], na_position="last")
    return df.drop(columns=["_both", "_wet", "_margin"]).reset_index(drop=True)


def with_buildability(rows: pd.DataFrame, engine_key: str | None,
                      build_model: str = buildability.DEFAULT_BUS_MODEL) -> pd.DataFrame:
    """Backfill the buildability/cost columns for rows that don't carry them.

    The assessment (``prospector.spacecraft.buildability``) is a pure closed-form function of (dry,
    prop, engine count, engine spec, gas), so sessions recorded before it existed get it computed
    live, with no re-run. Sessions whose script already wrote the columns pass through untouched.

    The engine is converted to each row's working gas (``propellant_key``, default xenon) before
    assessing, so a live recompute prices the gas that was swept and never quietly falls back to
    xenon. The configured build model is loaded once for the whole frame.
    """
    if not len(rows):
        return rows
    if "buildable" in rows.columns and rows["buildable"].notna().all():
        return rows
    try:
        catalog = load_engines()
        # Rows from a session that predates per-row engine keys need one to assess against; take
        # the session's own engine, else the library's first key. Never a literal: a hardcoded key
        # would be absent from a different library and every row would fail to assess.
        default = engine_key or (sorted(catalog)[0] if catalog else None)
        keys = (rows["engine"].fillna(default) if "engine" in rows.columns
                else pd.Series(default, index=rows.index))
        from prospector.spacecraft.propellants import engine_on_propellant, load_propellants
        propellants = load_propellants()
        model = buildability.load_bus_model(name=build_model)
        gases = (rows["propellant_key"].fillna("xenon") if "propellant_key" in rows.columns
                 else pd.Series("xenon", index=rows.index))
        igns = (rows["revolutions"] if "revolutions" in rows.columns
                else pd.Series(None, index=rows.index))
        assessed = [buildability.assess_engine(
            dry_kg=float(r.dry_kg), prop_kg=float(r.prop_kg),
            n_engines=int(r.n_engines),
            engine=engine_on_propellant(catalog[k], gas, propellants)[0],
            belt_days=(float(b) if pd.notna(b) else None),
            ignitions=(float(g) if pd.notna(g) else None),
            propellants=propellants, model=model)
            for r, k, gas, b, g in zip(rows.itertuples(), keys, gases,
                                       rows.get("belt_days", pd.Series(None, index=rows.index)),
                                       igns)]
    except Exception:
        return rows                      # an unknown engine never blocks the table
    rows = rows.copy()
    for k in _BUILD_COLUMNS:
        rows[k] = [a[k] for a in assessed]
    return rows


def in_flight(d: Path) -> str | None:
    """The design currently being processed: the last evaluation streamed, summarized
    'dry 180 · prop 92 · 1×thruster-a'."""
    jsonl = d / "evaluations.jsonl"
    if not jsonl.is_file():
        return None
    last = None
    for line in jsonl.read_text().splitlines():
        if line.strip():
            last = line
    if last is None:
        return None
    try:
        v = json.loads(last)
        return (f"dry {float(v['dry_kg']):.0f} · prop {float(v['prop_kg']):.0f} · "
                f"{int(v['n_engines'])}×{v.get('engine', '?')}")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


_STATE_TAG = {"running": "… running", "done": "✓ done", "error": "✕ error",
              "stopped": "stopped"}


def session_label(s: dict) -> str:
    """One human line identifying a session: state, target, engine stack."""
    meta = s.get("meta") or {}
    pairs = meta.get("pairs")
    if pairs:
        by_engine: dict = {}
        for key, n in pairs:
            by_engine.setdefault(key, []).append(int(n))
        engine = " + ".join(f"{k} ×{'/'.join(str(n) for n in sorted(ns))}"
                            for k, ns in by_engine.items())
    else:
        engine = meta.get("engine")
        if engine and meta.get("engines"):
            engine = f"{engine} ×{'/'.join(str(int(n)) for n in meta['engines'])}"
    return "  ·  ".join(filter(None, [state_tag(s), s.get("target"), engine]))


def state_tag(s: dict) -> str:
    """The state as shown; a shipped example reads as such rather than as a run of the user's."""
    if s.get("example") and s["state"] == "done":
        return "✓ example"
    return _STATE_TAG.get(s["state"], s["state"])


# ---------------------------------------------------------------------------
# what makes one sweep different from the next
# ---------------------------------------------------------------------------

def _fmt_range(value) -> str:
    """A ``[min, max, step]`` triple as ``min-max/step``, or ``value`` alone when the range is
    a single point (min == max), since a step over nothing says nothing."""
    lo, hi, step = (float(v) for v in value)
    return f"{lo:g}" if lo == hi else f"{lo:g}-{hi:g}/{step:g}"


def _fmt_axes(value) -> str:
    """The swept model settings as ``key=v1,v2,v3`` clauses, which are what a sweep explores,
    so it is named rather than counted."""
    return "  ".join(f"{k}={','.join(f'{v:g}' for v in vals)}" for k, vals in value.items())


# The session-setup fields the history distinguishes sweeps by, in the order they read. Each is a
# (label, session.json key, formatter); a falsy or missing value contributes nothing. Target and
# engine stack are left out, since they have their own columns and are what a user scans by.
_SETUP_FIELDS = (
    ("mission", "mission", str),
    ("mode", "mode", str),
    ("gas", "propellants", lambda v: "+".join(str(g) for g in v)),
    ("dry", "dry_range_kg", _fmt_range),
    ("prop", "prop_range_kg", _fmt_range),
    ("max wet", "max_wet_kg", lambda v: f"{float(v):g} kg"),
    ("thrust", "duty", lambda v: f"{float(v) * 100:.0f}%"),
    ("array", "array_W", lambda v: f"{float(v) / 1000:.1f} kW stated"),
    ("swept", "sweep_axes", _fmt_axes),
)


def session_setup(sessions: list[dict]) -> dict:
    """Split the listed sessions' setup into what distinguishes each one and what they all share.

    A history row has room for the handful of settings that differ, not for the dozen a
    sweep is defined by. So each field is rendered for every session and then placed by whether it
    varies: a field taking more than one value across the list is DISTINGUISHING and goes in that
    session's own row; a field every session agrees on is stated once, for all of them.

    A swept model axis is always distinguishing - it is the question the sweep was run to answer,
    and two sessions sweeping the same axis over different values are not the same sweep.

    Returns ``{"rows": {session_name: text}, "common": text}``. With a single session nothing can
    vary, so its whole setup lands in ``common`` - which is the honest reading: there is nothing to
    tell it apart from.
    """
    rendered: dict[str, dict[str, str]] = {}
    for s in sessions:
        meta = s.get("meta") or {}
        cells = {}
        for label, key, fmt in _SETUP_FIELDS:
            value = meta.get(key)
            if not value:
                continue
            try:
                cells[label] = fmt(value)
            except (TypeError, ValueError):
                continue                  # a malformed field describes nothing; leave it out
        rendered[s["name"]] = cells

    varies = set()
    for label, _key, _fmt in _SETUP_FIELDS:
        values = {cells.get(label) for cells in rendered.values()}
        if len(values) > 1 or (label == "swept" and any(values)):
            varies.add(label)

    def line(labels: set, cells: dict) -> str:
        return "  ·  ".join(f"{label} {cells[label]}" for label, _k, _f in _SETUP_FIELDS
                            if label in labels and label in cells)

    shared = {label for label, _k, _f in _SETUP_FIELDS if label not in varies}
    first = next(iter(rendered.values()), {})
    return {"rows": {name: line(varies, cells) for name, cells in rendered.items()},
            "common": line(shared, first)}


def design_id(engine, dry, prop, n_eng, propellant=None, params: dict | None = None) -> str:
    """A stable identity per evaluated design, so a selection sticks to the design rather than
    to a grid row position that re-sorts as new evaluations stream in. Includes the working
    gas so two rows that differ only by propellant (e.g. xenon vs krypton) stay distinct and
    a click highlights one. ``params`` are the swept model settings (``param_<key>``
    values): a model-setting sweep produces several rows identical in (engine, gas, masses,
    count) but differing only by a swept knob, so the knob values must be part of the identity
    or those rows collide on one key, which makes a click highlight the whole group and
    corrupts the table's row tracking (sort/selection)."""
    def _f(v):
        try:
            return f"{float(v):.1f}"
        except (TypeError, ValueError):
            return "?"
    try:
        n = int(float(n_eng))
    except (TypeError, ValueError):
        n = "?"
    gas = str(propellant) if propellant not in (None, "") else "?"
    base = f"{engine}|{gas}|{_f(dry)}|{_f(prop)}|{n}"
    if params:
        base += "|" + ",".join(f"{k}={_f(params[k])}" for k in sorted(params))
    return base


# A memorable adjective-noun name per design, so a good config can be recognized and referenced
# ("gilded-lynx") instead of recalled by its numeric params. Deterministic: the same design_id
# always yields the same name, so a name is stable across table reloads and re-runs and never has
# to be stored, since it follows from the design identity alone. 64 × 64 = 4096 base combinations;
# sessions larger than that get numeric disambiguation suffixes (see design_slugs) so distinct
# designs never share a name within one session.
_SLUG_ADJECTIVES = (
    "amber", "arctic", "azure", "bold", "brave", "bright", "bronze", "calm",
    "clever", "cobalt", "coral", "crimson", "dapper", "eager", "electric", "fabled",
    "fleet", "gallant", "gentle", "gilded", "glacial", "golden", "granite", "hardy",
    "ivory", "jade", "keen", "lively", "lucid", "lunar", "mellow", "mighty",
    "noble", "nimble", "opal", "polar", "prime", "quick", "quiet", "radiant",
    "rapid", "rugged", "rust", "sable", "scarlet", "sharp", "silent", "silver",
    "slate", "solar", "sonic", "stark", "steady", "stellar", "sterling", "stormy",
    "swift", "teal", "titan", "valiant", "vivid", "wily", "zephyr", "zesty",
)
_SLUG_NOUNS = (
    "albatross", "auk", "badger", "bison", "cobra", "comet", "condor", "cougar",
    "crane", "dragon", "eagle", "falcon", "ferret", "fox", "gannet", "gazelle",
    "griffin", "hawk", "heron", "ibex", "jackal", "jaguar", "kestrel", "kite",
    "lark", "lynx", "mako", "marlin", "marten", "meteor", "mongoose", "narwhal",
    "nebula", "ocelot", "orca", "osprey", "otter", "owl", "panther", "petrel",
    "puffin", "puma", "quokka", "raptor", "raven", "rhino", "sable", "salmon",
    "serval", "shrike", "sparrow", "stag", "stoat", "swift", "talon", "tapir",
    "tern", "tiger", "vixen", "vulture", "walrus", "weasel", "wolf", "wolverine",
)


def design_slug(design_id: str) -> str:
    """A memorable ``adjective-noun`` name derived deterministically from a design identity.

    Uses a stable content hash (``hashlib.md5``, not the builtin ``hash``, which is salted per
    process, so it would give a different name every app launch and defeat the whole point of a
    referenceable name). The same ``design_id`` always maps to the same base name; collisions
    between distinct designs in one session are broken by ``design_slugs``.
    """
    digest = hashlib.md5(str(design_id).encode()).digest()
    n = int.from_bytes(digest[:8], "big")
    adj = _SLUG_ADJECTIVES[(n >> 8) % len(_SLUG_ADJECTIVES)]
    noun = _SLUG_NOUNS[n % len(_SLUG_NOUNS)]
    return f"{adj}-{noun}"


def design_slugs(design_ids) -> dict[str, str]:
    """Map each distinct design identity to a unique memorable name within one collection.

    The base name is a pure function of the identity (``design_slug``); when two DISTINCT
    identities hash to the same base name, the collision is broken by a ``-2``/``-3`` suffix
    assigned in sorted-identity order so a given design's name is stable as rows stream in
    (a later evaluation never renames an earlier one)."""
    seen: dict[str, int] = {}
    out: dict[str, str] = {}
    for did in sorted({str(d) for d in design_ids}):
        base = design_slug(did)
        count = seen.get(base, 0) + 1
        seen[base] = count
        out[did] = base if count == 1 else f"{base}-{count}"
    return out


def with_design_id(rows: pd.DataFrame) -> pd.DataFrame:
    """Attach the stable ``design_id`` (every selection and pinned detail keys on it) and its
    memorable ``slug`` name (the human-facing identifier in the table, plot, and HUD)."""
    if not len(rows):
        return rows
    rows = rows.copy()
    # The swept model-setting columns are part of a design's identity (see design_id): without
    # them, every point of a --sweep-param run collides on one key.
    param_cols = [c for c in rows.columns if str(c).startswith("param_")]
    rows["design_id"] = [
        design_id(r.get("engine"), r.get("dry_kg"), r.get("prop_kg"),
                  r.get("n_engines"), r.get("propellant_key"),
                  params={c: r.get(c) for c in param_cols} if param_cols else None)
        for _, r in rows.iterrows()]
    names = design_slugs(rows["design_id"])
    rows["slug"] = [names[did] for did in rows["design_id"]]
    return rows


# ---------------------------------------------------------------------------
# launching a search (the one mutation: spawn the script, detached)
# ---------------------------------------------------------------------------

def submit_search(cli_args: list[str]) -> str:
    """Spawn the search script detached against a fresh session dir; return its name.

    Mirrors the worker-job pattern: the child survives app restarts (``start_new_session``), its
    console goes to ``console.log`` in the session dir, and an initial ``status.json`` is stamped
    here so the session is listable before the search's own first write.
    """
    name = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = search_root() / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "status.json").write_text(json.dumps(
        {"state": "running", "message": "starting…", "n_evaluated": 0,
         "updated": datetime.now().isoformat(timespec="seconds")}))
    log = open(out / "console.log", "w")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", SEARCH_MODULE, "--out", str(out), *cli_args],
            cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log.close()      # the detached child keeps its own dup'd fd
    # Record the child PID so the workspace can tell a crashed sweep from a slow one without
    # waiting out the mtime staleness window. A dedicated file (not status.json) survives the
    # search's own status rewrites untouched.
    (out / "pid").write_text(str(proc.pid))
    return name


def request_stop(d: Path) -> None:
    """Ask a running session to wind down: it stops after the evaluation in flight and keeps
    everything finished so far (the search polls for this ``cancel`` file between combos)."""
    (d / "cancel").write_text("1")


def stop_requested(d: Path) -> bool:
    return (d / "cancel").exists()


def read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None




def _read_pid(d: Path) -> int | None:
    """The sweep's recorded child PID, or None when the session predates PID recording or the
    file is unreadable (then callers fall back to the mtime staleness guard)."""
    try:
        return int((d / "pid").read_text().strip())
    except (OSError, ValueError):
        return None


def _process_alive(pid: int) -> bool:
    """Whether ``pid`` is a live process, via the POSIX signal-0 probe. Off POSIX (or on any
    ambiguous error) cannot be told apart cheaply, so report alive and let the mtime guard
    catch a dead run, and never a false 'dead' that would hide a running sweep. (A dead
    PID later reused by an unrelated process is the one false 'alive'; the staleness guard still
    backstops it.)"""
    if os.name != "posix" or not pid or pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False                      # definitively gone
    except (PermissionError, OSError):
        return True                       # exists (just not ours) or can't probe
    return True
