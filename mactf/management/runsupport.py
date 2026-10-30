"""Argument types, run selection, and the run listing/report shared by the run commands.

Document 16, section 8.1.2. ``runs``, ``inspect``, ``checkpoints``, ``logs``, ``run``, and
``redteam`` all select runs the same way and describe them the same way; this module is
the one place that happens. Module-level imports are the standard library, the exceptions,
and the cheap core modules of section 7 (none of them imports torch).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

from ..db import Registry
from ..exceptions import CommandError, InvalidArgument, ResourceNotFound
from ..orchestration.recovery import (
    ACTIVE_STATUSES,
    FINAL_STATUSES,
    RESUMABLE_STATUSES,
    validate_run_id,
)
from ..orchestration.runlock import LockInfo, RunLock, lock_directory
from ..orchestration.runview import decode_run, liveness, resolve_run_id, run_duration
from .durations import format_duration, format_timestamp, parse_duration, parse_when
from .exitcodes import ExitCode
from .output import Column

ALL_STATUSES = ACTIVE_STATUSES | RESUMABLE_STATUSES | FINAL_STATUSES
STATUS_GROUPS = {
    "active": ACTIVE_STATUSES,
    "resumable": RESUMABLE_STATUSES,
    "final": FINAL_STATUSES,
}
SECTIONS = ("episodes", "rules", "checkpoints", "warnings", "adjudications", "aggregations", "resumes")


# -- argparse types: a bad value is a usage error naming the value ----------------------
def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} must be at least 1")
    return value


def non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"{text!r} must not be negative")
    return value


def run_id_arg(text: str) -> str:
    try:
        return validate_run_id(text)
    except InvalidArgument as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


_DEVICE = re.compile(r"^(auto|cpu|mps|cuda(:\d+)?)$")


def device_arg(text: str) -> str:
    if not _DEVICE.match(text):
        raise argparse.ArgumentTypeError(f"{text!r} is not a device; use auto, cpu, mps, cuda, or cuda:N")
    return text


def status_arg(text: str) -> frozenset[str]:
    key = text.strip().lower()
    if key in STATUS_GROUPS:
        return frozenset(STATUS_GROUPS[key])
    if key in ALL_STATUSES:
        return frozenset({key})
    raise argparse.ArgumentTypeError(
        f"unknown status {text!r}; use one of {', '.join(sorted(ALL_STATUSES))} or a group: "
        f"{', '.join(sorted(STATUS_GROUPS))}"
    )


def duration_arg(text: str) -> float:
    try:
        return parse_duration(text)
    except CommandError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def when_arg(text: str) -> float:
    try:
        return parse_when(text)
    except CommandError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def flatten_statuses(values: Sequence[frozenset[str]] | None) -> list[str] | None:
    """The union of every ``--status`` given, or ``None`` when there was none."""
    if not values:
        return None
    merged: set[str] = set()
    for value in values:
        merged |= value
    return sorted(merged)


# -- registry and run selection --------------------------------------------------------
def require_registry(command: Any) -> Registry:
    """The registry, or ``ResourceNotFound`` when no database exists yet. Never creates one."""
    registry = command.registry_for_read()
    if registry is None:
        raise ResourceNotFound(
            f"no run registry at {command.need_settings().database_path()}; nothing has been run yet"
        )
    return registry


def target_run(command: Any, text: str | None, *, statuses: Collection[str] | None = None) -> tuple[Registry, str]:
    """``(registry, run_id)`` for a user's run argument (id, unique prefix, or ``latest``)."""
    registry = require_registry(command)
    return registry, resolve_run_id(registry, text, statuses=statuses)


def lock_dir_for(command: Any) -> Path:
    return lock_directory(command.need_settings())


def holder_of(command: Any, run_id: str) -> LockInfo | None:
    return RunLock.holder(lock_dir_for(command), run_id)


def exclusive_run(command: Any, run_id: str) -> RunLock:
    """A lock to hold while mutating a run; ``acquire`` raises ``StateConflict`` if a process drives it."""
    return RunLock(lock_dir_for(command), run_id, owner="cli")


def declined(command: Any) -> int:
    command.out.result("aborted")
    return int(ExitCode.FAILURE)


#: The positional every run-taking verb shares. These dicts are shared between verbs, so
#: ``_declare`` copies each kwargs dict before it pops ``"completer"`` from it.
RUN_OPTIONAL = (("run",), {"nargs": "?", "default": "latest", "completer": "run_id",
                           "help": "Run id, a unique prefix of one, or 'latest' (the default)."})
RUN_REQUIRED = (("run",), {"completer": "run_id",
                           "help": "Run id, a unique prefix of one, or 'latest'."})


# -- run listing ----------------------------------------------------------------------
def _three(value: float) -> str:
    return f"{value:.3f}"


RUN_COLUMNS = (
    Column("id", "ID"),
    Column("status", "STATUS"),
    Column("liveness", "STATE"),
    Column("progress", "EPOCHS", align="r"),
    Column("kappa", "KAPPA", align="r", fmt=_three),
    Column("entropy", "ENTROPY", align="r", fmt=_three),
    Column("started_at", "STARTED", fmt=format_timestamp),
    Column("duration", "TOOK", align="r", fmt=format_duration),
)


def run_listing(registry: Registry, rows: Sequence[Mapping[str, Any]], lock_dir: Path,
                *, now: float | None = None) -> list[dict[str, Any]]:
    """One flat, JSON-ready dict per run row: the shape of ``runs list`` items."""
    counts = registry.episode_counts([row["id"] for row in rows])
    items: list[dict[str, Any]] = []
    for row in rows:
        decoded = decode_run(row)
        settings = decoded.get("settings")
        planned = settings.get("epochs") if isinstance(settings, dict) else None
        done = counts.get(row["id"], 0)
        items.append({
            "id": row["id"],
            "status": row["status"],
            "liveness": liveness(lock_dir, row),
            "episodes": done,
            "planned_epochs": planned,
            "progress": f"{done}/{planned}" if planned else str(done),
            "kappa": row.get("kappa"),
            "entropy": row.get("entropy"),
            "started_at": row.get("started_at"),
            "finished_at": row.get("finished_at"),
            "duration": run_duration(row, now=now),
            "notes": row.get("notes") or "",
        })
    return items


# -- run report (runs show, inspect) ---------------------------------------------------
def _scores(card: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    peers = card.get("peers") if isinstance(card, Mapping) else None
    peers = peers if isinstance(peers, Mapping) else {}
    scores = {
        name: (round(payload["final_score"], 4)
               if isinstance(payload, Mapping) and isinstance(payload.get("final_score"), (int, float))
               else None)
        for name, payload in peers.items()
    }
    void = [name for name, payload in peers.items() if isinstance(payload, Mapping) and payload.get("void")]
    return scores, void


def collect_run_report(registry: Registry, run_id: str, *, sections: Collection[str],
                       lock_dir: Path, now: float | None = None) -> dict[str, Any]:
    """Everything known about one run. ``sections`` (a subset of ``SECTIONS``) adds lists.

    Always present: ``run`` (decoded row), ``liveness``, ``lock``, ``duration``, ``counts``.
    A requested section appears as a top-level key; an unrequested one is absent, so the
    default payload stays small.
    """
    unknown = set(sections) - set(SECTIONS)
    if unknown:
        raise InvalidArgument(f"unknown section(s) {sorted(unknown)}; known: {', '.join(SECTIONS)}")
    row = registry.get_run(run_id)
    if row is None:
        raise ResourceNotFound(f"no run {run_id!r}")
    holder = RunLock.holder(lock_dir, run_id)
    rule_counts = registry.rule_counts(run_id)
    checkpoints = registry.checkpoints(run_id)
    warnings = registry.all_warnings(run_id)
    adjudications = registry.adjudications(run_id)
    aggregations = registry.aggregations(run_id)
    report: dict[str, Any] = {
        "run": decode_run(row),
        "liveness": liveness(lock_dir, row),
        "lock": holder.as_dict() if holder is not None else None,
        "duration": run_duration(row, now=now),
        "counts": {
            "episodes": registry.count_episodes(run_id),
            "checkpoints": len(checkpoints),
            "rule_events": sum(rule_counts.values()),
            "warnings": len(warnings),
            "adjudications": len(adjudications),
            "aggregations": len(aggregations),
        },
    }
    if "episodes" in sections:
        items = []
        for episode in registry.episodes(run_id):
            scores, void = _scores(episode.get("scorecard") or {})
            items.append({"episode": episode["episode"], "started_at": episode.get("started_at"),
                          "finished_at": episode.get("finished_at"), "scores": scores, "void": void,
                          "scorecard": episode.get("scorecard")})
        report["episodes"] = items
    if "rules" in sections:
        report["rules"] = rule_counts
    if "checkpoints" in sections:
        report["checkpoints"] = checkpoints
    if "warnings" in sections:
        report["warnings"] = warnings
    if "adjudications" in sections:
        report["adjudications"] = adjudications
    if "aggregations" in sections:
        report["aggregations"] = aggregations
    if "resumes" in sections:
        report["resumes"] = registry.resume_reports(run_id)
    return report


def render_run_report(out: Any, report: Mapping[str, Any]) -> None:
    """The one text renderer for a run report."""
    run = report["run"]
    out.result(f"run {run['id']} [{run['status']}] {report['liveness']}")
    runtime = run.get("runtime") if isinstance(run.get("runtime"), dict) else {}
    lock = report.get("lock")
    out.kv([
        ("started", format_timestamp(run.get("started_at"))),
        ("finished", format_timestamp(run.get("finished_at"))),
        ("took", format_duration(report["duration"]) if report.get("duration") is not None else "-"),
        ("kappa", _three(run["kappa"]) if run.get("kappa") is not None else "-"),
        ("entropy", _three(run["entropy"]) if run.get("entropy") is not None else "-"),
        ("device", runtime.get("device") or "-"),
        ("notes", run.get("notes") or "-"),
        ("driven by", f"pid {lock['pid']} on {lock['host']} ({lock['owner']})" if lock else "-"),
    ])
    counts = report["counts"]
    out.result("  " + "  ".join(f"{name}={value}" for name, value in counts.items()))
    if "episodes" in report:
        out.result("episodes")
        out.table(
            [{"episode": item["episode"],
              "scores": ", ".join(f"{peer}={value if value is not None else '-'}"
                                  for peer, value in item["scores"].items()) or "-",
              "void": ", ".join(item["void"]) or "-"} for item in report["episodes"]],
            [Column("episode", "EP", align="r"), Column("scores", "SCORES"), Column("void", "VOID")],
            empty="(no episodes)",
        )
    if "rules" in report:
        out.result("rule events")
        out.table([{"rule_id": rule, "count": count} for rule, count in report["rules"].items()],
                  [Column("rule_id", "RULE"), Column("count", "COUNT", align="r")],
                  empty="(no rule events)")
    if "checkpoints" in report:
        out.result("checkpoints")
        out.table(report["checkpoints"],
                  [Column("episode", "EP", align="r"), Column("kappa", "KAPPA", align="r", fmt=_three),
                   Column("entropy", "ENTROPY", align="r", fmt=_three),
                   Column("score", "SCORE", align="r", fmt=lambda value: f"{value:.4f}"),
                   Column("path", "PATH")], empty="(no checkpoints)")
    if "warnings" in report:
        out.result("warnings")
        out.table(report["warnings"],
                  [Column("peer", "PEER"), Column("rule_id", "RULE"), Column("episode", "EP", align="r"),
                   Column("severity", "SEV"), Column("active", "ACTIVE"),
                   Column("expiry_episode", "EXPIRES", align="r")], empty="(no warnings)")
    if "adjudications" in report:
        out.result("adjudications")
        out.table(report["adjudications"],
                  [Column("peer", "PEER"), Column("rule_id", "RULE"), Column("episode", "EP", align="r"),
                   Column("decision", "DECISION"), Column("adjudicator", "BY"),
                   Column("rationale", "RATIONALE", max_width=60)], empty="(no adjudications)")
    if "aggregations" in report:
        out.result("aggregations")
        rows = []
        for item in report["aggregations"]:
            body = item.get("report") if isinstance(item.get("report"), Mapping) else {}
            rows.append({
                "episode": item["episode"], "kappa": body.get("kappa"), "entropy": body.get("entropy"),
                "weights": ", ".join(f"{peer}={weight:.3f}"
                                     for peer, weight in (body.get("weights") or {}).items()),
                "fallback": body.get("fallback"),
            })
        out.table(rows,
                  [Column("episode", "EP", align="r"), Column("kappa", "KAPPA", align="r", fmt=_three),
                   Column("entropy", "ENTROPY", align="r", fmt=_three), Column("weights", "WEIGHTS"),
                   Column("fallback", "FALLBACK")], empty="(no aggregations)")
    if "resumes" in report:
        out.result("resume reports")
        out.table(report["resumes"],
                  [Column("id", "ID", align="r"), Column("checkpoint_episode", "FROM", align="r"),
                   Column("started_epoch", "STARTED", align="r"),
                   Column("warnings_restored", "WARNINGS", align="r")], empty="(none)")


def add_component_modules(settings, modules) -> None:
    """``--component-module M``: append to ``COMPONENTS.MODULES`` for this process only.

    Nothing is written to a settings file. The modules are loaded like any listed in
    ``conf/components.yaml``, so the import surface is the operator's (document 17, 16.1.5).
    """
    if not modules:
        return
    from ..components import ensure_components_loaded
    from ..components.known import is_dotted_name

    for module in modules:
        if not is_dotted_name(module):
            raise CommandError.usage(f"--component-module {module!r} is not a dotted module name")
    block = dict(settings.get("COMPONENTS") or {})
    listed = list(block.get("MODULES", ()))
    block["MODULES"] = [*listed, *[m for m in modules if m not in listed]]
    settings["COMPONENTS"] = block
    ensure_components_loaded(settings)
