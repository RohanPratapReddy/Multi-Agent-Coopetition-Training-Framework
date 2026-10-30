"""``runs`` - list, inspect, wait for, resume, cancel, reap, and delete runs. Section 8.12.

Everything here reads the registry and the cross-process run locks. No verb imports
``torch`` except ``resume``, which trains.
"""

from __future__ import annotations

import time
from typing import Any

from ..driver import RunDriver, plan_run, present_run_end
from ..durations import format_duration
from ..exitcodes import ExitCode
from ..group import GroupCommand, Sub
from ..interrupt import StopFlag
from ..output import Column, format_bytes
from ..runsupport import (
    RUN_COLUMNS,
    RUN_OPTIONAL,
    RUN_REQUIRED,
    SECTIONS,
    collect_run_report,
    declined,
    device_arg,
    duration_arg,
    exclusive_run,
    flatten_statuses,
    holder_of,
    lock_dir_for,
    positive_int,
    render_run_report,
    require_registry,
    run_listing,
    status_arg,
    target_run,
    when_arg,
)
from ...exceptions import CommandError, ResourceNotFound, StateConflict
from ...orchestration.recovery import (
    ACTIVE_STATUSES,
    FINAL_STATUSES,
    RESUMABLE_STATUSES,
    delete_run_artifacts,
    reap_stale,
)
from ...orchestration.runlock import RunLock
from ...orchestration.runview import liveness, resolve_run_id

_FILTERS = (
    (("--status",), {"action": "append", "type": status_arg, "default": None, "metavar": "STATUS",
                     "completer": "run_status",
                     "help": "A status, or active, resumable, final. Repeatable."}),
    (("--since",), {"type": when_arg, "default": None, "metavar": "WHEN",
                    "help": "Started at or after: 2h, 3d, 2026-10-01, 2026-10-01T08:30."}),
    (("--until",), {"type": when_arg, "default": None, "metavar": "WHEN",
                    "help": "Started at or before WHEN."}),
    (("--prefix",), {"default": None, "help": "Only runs whose id starts with this."}),
)
_PAGING = (
    (("--limit",), {"type": positive_int, "default": 20, "help": "Rows to show (default 20)."}),
    (("--all",), {"action": "store_true", "default": False, "help": "Every matching run."}),
)
_SECTION_FLAGS = (
    *[((f"--{name}",), {"action": "store_true", "default": False,
                        "help": f"Include the {name} section."}) for name in SECTIONS],
    (("--all-sections",), {"action": "store_true", "default": False, "help": "Include every section."}),
)


class Command(GroupCommand):
    help = "List, inspect, wait for, resume, cancel, and delete runs."
    category = "Runs"
    output_formats = ("text", "json", "jsonl", "yaml")
    subcommands = (
        Sub("list", "List runs, newest first.", aliases=("ls",),
            arguments=(*_FILTERS, *_PAGING, (("--reverse",), {"action": "store_true", "default": False,
                                                              "help": "Oldest first."})),
            exclusive=(("--limit", "--all"),),
            examples=("mactf runs list", "mactf runs list --status failed --since 2d",
                      "mactf runs list --status active --format jsonl")),
        Sub("show", "Everything known about one run.", output_formats=("text", "json", "yaml"),
            arguments=(RUN_OPTIONAL, *_SECTION_FLAGS),
            examples=("mactf runs show", "mactf runs show 9f3a --episodes --rules")),
        Sub("wait", "Block until a run stops being active.", output_formats=("text", "json"),
            arguments=(RUN_OPTIONAL,
                       (("--timeout",), {"type": duration_arg, "default": None, "metavar": "DURATION",
                                         "help": "Stop waiting after this long (exit 124)."}),
                       (("--interval",), {"type": duration_arg, "default": 2.0, "metavar": "DURATION",
                                          "help": "Poll interval (default 2s)."})),
            examples=("mactf runs wait 9f3a --timeout 2h",)),
        Sub("resume", "Continue a paused or interrupted run under its own id.",
            output_formats=("text", "json", "jsonl"),
            arguments=(RUN_OPTIONAL,
                       (("--epochs",), {"type": positive_int, "default": None,
                                        "help": "Total epochs (not additional epochs)."}),
                       (("--device",), {"type": device_arg, "default": None,
                                        "help": "auto, cpu, mps, cuda, or cuda:N."}),
                       (("--timeout",), {"type": duration_arg, "default": None, "metavar": "DURATION",
                                         "help": "Wall-clock budget; pauses at the next epoch boundary (exit 124)."}),
                       (("--on-interrupt",), {"choices": ["pause", "cancel"], "default": "pause",
                                              "help": "What Ctrl-C/SIGTERM makes of the run row."}),
                       (("--log-file",), {"default": None, "metavar": "PATH",
                                          "help": "Write the per-run JSONL log here."}),
                       (("--no-log",), {"action": "store_true", "default": False,
                                        "help": "Write no per-run log."})),
            exclusive=(("--log-file", "--no-log"),),
            examples=("mactf runs resume", "mactf runs resume nightly-01 --epochs 40")),
        Sub("cancel", "Mark a run nobody is driving as cancelled.", destructive=True,
            output_formats=("text", "json"),
            arguments=(RUN_REQUIRED,
                       (("--reason",), {"default": None, "help": "Recorded in the run's notes."}),
                       (("--dry-run",), {"action": "store_true", "default": False,
                                         "help": "Say what would happen; change nothing."})),
            examples=("mactf runs cancel 9f3a --reason \"bad seed\"",)),
        Sub("reap", "Mark active runs with no live process as interrupted.", destructive=True,
            output_formats=("text", "json"),
            arguments=((("--dry-run",), {"action": "store_true", "default": False,
                                         "help": "List the runs that would be reaped."}),),
            examples=("mactf runs reap --dry-run", "mactf runs reap --yes")),
        Sub("delete", "Delete runs, their registry rows, checkpoints, and logs.", aliases=("rm",),
            destructive=True, output_formats=("text", "json"),
            arguments=((("run",), {"nargs": "*", "completer": "run_id",
                                   "help": "Run ids or unique prefixes."}),
                       (("--older-than",), {"type": duration_arg, "default": None, "metavar": "DURATION",
                                            "help": "Runs started more than this long ago."}),
                       (("--status",), {"action": "append", "type": status_arg, "default": None,
                                        "metavar": "STATUS", "completer": "run_status",
                                        "help": "With --older-than: which statuses (default: final)."}),
                       (("--keep-files",), {"action": "store_true", "default": False,
                                            "help": "Delete rows only; leave checkpoint and log files."}),
                       (("--dry-run",), {"action": "store_true", "default": False,
                                         "help": "Say what would be deleted; delete nothing."})),
            examples=("mactf runs delete 9f3a --dry-run", "mactf runs delete --older-than 30d --yes")),
    )

    # -- list -----------------------------------------------------------------
    def sub_list(self, **options):
        since, until = options["since"], options["until"]
        if since is not None and until is not None and since > until:
            raise CommandError.usage("--since is later than --until")
        registry = self.registry_for_read()
        filters = {"status": flatten_statuses(options["status"]), "since": since,
                   "until": until, "id_prefix": options["prefix"]}
        total = registry.count_runs(**filters) if registry is not None else 0
        limit = max(total, 1) if options["all"] else options["limit"]
        rows = (registry.list_runs(limit=limit, order="asc" if options["reverse"] else "desc", **filters)
                if registry is not None and total else [])
        items = run_listing(registry, rows, lock_dir_for(self)) if rows else []
        if self.out.fmt == "jsonl":
            for item in items:
                self.out.jsonl(item)
            return 0
        payload = {"total": total, "count": len(items), "items": items}

        def text() -> None:
            self.out.table(items, RUN_COLUMNS, empty="no runs match")
            if total > len(items):
                self.stderr.write(f"showing {len(items)} of {total} (use --all or --limit to see more)\n")
        self.out.emit(payload, text=text)
        return 0

    # -- show -----------------------------------------------------------------
    def sub_show(self, **options):
        registry, run_id = target_run(self, options["run"])
        sections = set(SECTIONS) if options["all_sections"] else {n for n in SECTIONS if options.get(n)}
        report = collect_run_report(registry, run_id, sections=sections, lock_dir=lock_dir_for(self))
        self.out.emit(report, text=lambda: render_run_report(self.out, report))
        return 0

    # -- wait -----------------------------------------------------------------
    def sub_wait(self, **options):
        registry, run_id = target_run(self, options["run"])
        lock_dir = lock_dir_for(self)
        began = time.monotonic()
        with StopFlag(timeout=options["timeout"], announce=self.out.warn,
                      message="no longer waiting; the run itself is unaffected") as stop:
            while True:
                row = registry.get_run(run_id)
                if row is None:
                    raise ResourceNotFound(f"run {run_id} was deleted while waiting")
                state = liveness(lock_dir, row)
                self.out.debug(f"{run_id} {row['status']} {state}")
                if row["status"] not in ACTIVE_STATUSES or state == "stale":
                    break
                remaining = options["interval"]
                while remaining > 0 and not stop.wait(min(remaining, 0.5)):
                    remaining -= 0.5
                if stop.is_set():
                    break
        waited = time.monotonic() - began
        status = row["status"]
        if stop.timed_out:
            code = int(ExitCode.TIMEOUT)
        elif stop.is_set():
            code = int(ExitCode.INTERRUPTED)
        elif state == "stale":
            code = int(ExitCode.FAILURE)
        else:
            code = int(ExitCode.OK if status in ("completed", "dry_run") else ExitCode.FAILURE)
        payload = {"run_id": run_id, "status": status, "liveness": state, "waited": round(waited, 3),
                   "timed_out": stop.timed_out, "exit_code": code}

        def text() -> None:
            if stop.timed_out:
                self.out.result(f"run {run_id} is still {status} after {format_duration(waited)}")
            elif state == "stale":
                self.out.result(f"run {run_id} [{status}] has no live process; "
                                f"`{self.prog} runs reap` marks it interrupted")
            else:
                self.out.result(f"run {run_id} [{status}] after {format_duration(waited)}")
        self.out.emit(payload, text=text)
        return code

    # -- resume ---------------------------------------------------------------
    def sub_resume(self, **options):
        from ...checks import run_checks

        outcome = run_checks()
        for message in outcome.errors:
            self.out.error(str(message))
        if not outcome.ok:
            raise CommandError(f"system checks failed ({outcome.summary()})")
        plan = plan_run(self, {"resume": options["run"], "epochs": options["epochs"],
                               "device": options["device"]})
        result = RunDriver(self).execute(
            plan, timeout=options["timeout"], on_interrupt=options["on_interrupt"],
            log_path=options["log_file"], log_enabled=not options["no_log"])
        present_run_end(self, plan, result)
        return result.exit_code

    # -- cancel ---------------------------------------------------------------
    def sub_cancel(self, **options):
        registry, run_id = target_run(self, options["run"], statuses=ACTIVE_STATUSES | RESUMABLE_STATUSES)
        row = registry.get_run(run_id)
        holder = holder_of(self, run_id)
        if holder is not None:
            raise StateConflict(
                f"run {run_id} is being driven by pid {holder.pid} on {holder.host} ({holder.owner}); "
                "stop it there with Ctrl-C, or POST /runs/{id}/cancel on the API that owns it",
                run_id=run_id, pid=holder.pid)
        payload = {"run_id": run_id, "was": row["status"], "status": "cancelled",
                   "dry_run": options["dry_run"]}
        if options["dry_run"]:
            self.out.emit(payload, text=lambda: self.out.result(f"would cancel run {run_id} ({row['status']})"))
            return 0
        if not self.confirm(f"Cancel run {run_id} ({row['status']})? A cancelled run cannot be resumed."):
            return declined(self)
        # The registry here is read-only; the write goes through a writable one.
        writer = self.registry_for_write()
        with exclusive_run(self, run_id):
            current = writer.get_run(run_id)
            if current is None or current["status"] not in ACTIVE_STATUSES | RESUMABLE_STATUSES:
                raise StateConflict(f"run {run_id} is no longer cancellable "
                                    f"({current['status'] if current else 'deleted'})", run_id=run_id)
            writer.set_run_status(run_id, "cancelled",
                                  notes=options["reason"] or "cancelled from the command line")
        self.out.emit(payload, text=lambda: self.out.result(f"run {run_id} [cancelled]"))
        return 0

    # -- reap -----------------------------------------------------------------
    def sub_reap(self, **options):
        registry = require_registry(self)
        lock_dir = lock_dir_for(self)
        pending = reap_stale(registry, lock_dir, dry_run=True)
        payload = {"dry_run": options["dry_run"], "count": len(pending), "reaped": pending}
        if pending and not options["dry_run"]:
            if not self.confirm(f"Mark {len(pending)} stale run(s) interrupted?"):
                return declined(self)
            payload["reaped"] = reap_stale(self.registry_for_write(), lock_dir)
            payload["count"] = len(payload["reaped"])

        def text() -> None:
            self.out.table(payload["reaped"], [Column("run_id", "RUN"), Column("was", "WAS"),
                                               Column("reaped", "REAPED")], empty="no stale runs")
            if payload["reaped"]:
                self.out.result(("would mark" if options["dry_run"] else "marked")
                                + f" {payload['count']} run(s) interrupted; "
                                  f"continue one with `{self.prog} runs resume <id>`")
        self.out.emit(payload, text=text)
        return 0

    # -- delete ---------------------------------------------------------------
    def sub_delete(self, **options):
        registry = require_registry(self)
        lock_dir = lock_dir_for(self)
        explicit = bool(options["run"])
        if explicit == (options["older_than"] is not None):
            raise CommandError.usage("name runs to delete, or pass --older-than (not both, not neither)")
        if explicit:
            run_ids = list(dict.fromkeys(resolve_run_id(registry, text) for text in options["run"]))
        else:
            statuses = flatten_statuses(options["status"]) or sorted(FINAL_STATUSES)
            rows = registry.list_runs(limit=max(registry.count_runs(), 1), status=statuses,
                                      until=time.time() - options["older_than"])
            run_ids = [row["id"] for row in rows]

        eligible: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        for run_id in run_ids:
            holder = RunLock.holder(lock_dir, run_id)
            if holder is not None:
                skipped.append({"run_id": run_id, "reason": f"driven by pid {holder.pid} on {holder.host}"})
            else:
                eligible.append(registry.get_run(run_id))
        deleted: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []

        if options["dry_run"]:
            for row in eligible:
                artifacts = ({"files": [], "freed_bytes": 0} if options["keep_files"]
                             else delete_run_artifacts(self.settings, registry, row["id"], dry_run=True))
                deleted.append({"run_id": row["id"], "status": row["status"], "rows": {},
                                "files": artifacts["files"], "freed_bytes": artifacts["freed_bytes"]})
        elif eligible:
            if not self.confirm(f"Delete {len(eligible)} run(s) and "
                                f"{'their registry rows only' if options['keep_files'] else 'their files'}?"):
                return declined(self)
            writer = self.registry_for_write()
            for row in eligible:
                run_id = row["id"]
                try:
                    with exclusive_run(self, run_id):
                        artifacts = ({"files": [], "freed_bytes": 0, "problems": []} if options["keep_files"]
                                     else delete_run_artifacts(self.settings, writer, run_id))
                        if artifacts["problems"]:
                            failed.append({"run_id": run_id, "problems": artifacts["problems"]})
                            continue                     # keep the row: it still names the files left
                        rows = writer.delete_run(run_id)
                except StateConflict as exc:
                    skipped.append({"run_id": run_id, "reason": str(exc)})
                    continue
                deleted.append({"run_id": run_id, "status": row["status"], "rows": rows,
                                "files": artifacts["files"], "freed_bytes": artifacts["freed_bytes"]})

        payload = {"dry_run": options["dry_run"], "deleted": deleted, "skipped": skipped, "failed": failed,
                   "freed_bytes": sum(item["freed_bytes"] for item in deleted)}
        self.out.emit(payload, text=lambda: self._delete_text(payload))
        if failed:
            return int(ExitCode.FAILURE)
        if explicit and skipped and not deleted:
            return int(ExitCode.CONFLICT)
        return int(ExitCode.OK)

    def _delete_text(self, payload: dict[str, Any]) -> None:
        verb = "would delete" if payload["dry_run"] else "deleted"
        for item in payload["deleted"]:
            rows = sum(item["rows"].values())
            self.out.result(f"{verb} run {item['run_id']} [{item['status']}] "
                            f"{(str(rows) + ' row(s), ') if rows else ''}{len(item['files'])} file(s), "
                            f"{format_bytes(item['freed_bytes'])}")
        for item in payload["skipped"]:
            self.out.warn(f"skipped {item['run_id']}: {item['reason']}")
        for item in payload["failed"]:
            self.out.error(f"kept {item['run_id']}: " + "; ".join(item["problems"]))
        if not payload["deleted"] and not payload["skipped"] and not payload["failed"]:
            self.out.result("no runs match")
