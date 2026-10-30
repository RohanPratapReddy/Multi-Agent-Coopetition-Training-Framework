"""``logs`` - read a run's JSONL log, with ``--follow``. Document 16, section 8.14.

The CLI writes the per-run log (``run``); this command reads it with the filters the API
has and a ``--follow`` it cannot have.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..durations import format_timestamp
from ..exitcodes import ExitCode
from ..group import GroupCommand, Sub
from ..interrupt import StopFlag
from ..runsupport import RUN_OPTIONAL, lock_dir_for, positive_int, target_run
from ...exceptions import CommandError, ResourceNotFound
from ...orchestration.runview import liveness
from ...training.logging import (
    LogRecord,
    complete_size,
    follow_jsonl,
    run_log_location,
    tail_jsonl,
)

_LOG_FIELDS = {name for name in LogRecord.__dataclass_fields__}


def render_log_line(record: Mapping[str, Any]) -> str:
    """One text line for a log record: the layout the trainer's console sink prints.

    Unknown keys are ignored and missing keys defaulted, so a record written by a newer or
    an older schema still renders.
    """
    values = {key: record[key] for key in _LOG_FIELDS if key in record}
    for key in ("kind", "run_id", "node_path", "level", "message"):
        if key in values and not isinstance(values[key], str):
            values[key] = str(values[key])
    fields = values.get("fields")
    if fields is not None and not isinstance(fields, Mapping):
        values["fields"] = {"value": fields}
    for key in ("depth", "epoch", "step"):
        if key in values:
            try:
                values[key] = int(values[key])
            except (TypeError, ValueError):
                values.pop(key)
    for key in ("wall_time", "monotonic", "duration"):
        if key in values:
            try:
                values[key] = float(values[key])
            except (TypeError, ValueError):
                values.pop(key)
    built = LogRecord(**values)
    stamp = format_timestamp(built.wall_time) if built.wall_time else "-"
    return f"{stamp} {built.to_text()}"


class Command(GroupCommand):
    help = "Read a run's JSONL log."
    category = "Runs"
    output_formats = ("text", "json", "jsonl")
    subcommands = (
        Sub("tail", "Print the last records of a run's log; --follow keeps printing new ones.",
            arguments=(RUN_OPTIONAL,
                       (("-n", "--lines"), {"type": positive_int, "default": 20, "metavar": "N",
                                            "help": "Records to print (default 20)."}),
                       (("-f", "--follow"), {"action": "store_true", "default": False,
                                             "help": "Keep printing records as they are written."}),
                       (("--kind",), {"default": None, "completer": "log_kind",
                                      "help": "run, epoch, step, loss, score, aggregation, warning, ..."}),
                       (("--level",), {"choices": ["debug", "info", "warning", "error", "critical"],
                                       "default": None, "help": "Only records of this level."}),
                       (("--node",), {"default": None, "metavar": "NODE_PATH",
                                      "help": "Only records of this node."}),
                       (("--file",), {"default": None, "metavar": "PATH", "completer": "file",
                                      "help": "Read this JSONL file instead of a run's log."})),
            examples=("mactf logs tail", "mactf logs tail 9f3a -n 100 --kind epoch",
                      "mactf logs tail -f --level warning", "mactf logs tail --format jsonl | jq .fields")),
        Sub("path", "Print where a run's log is (or would be).", output_formats=("text", "json"),
            arguments=(RUN_OPTIONAL,),
            examples=("tail -f $(mactf logs path)",)),
    )

    def _resolve(self, options: Mapping[str, Any]) -> tuple[Path, str | None, str | None]:
        """``(path, run_id for filtering, run id for liveness)``."""
        if options.get("file"):
            return Path(options["file"]).expanduser(), None, None
        registry, run_id = target_run(self, options["run"])
        location = run_log_location(self.settings, run_id)
        return location.path, (run_id if location.shared else None), run_id

    def sub_path(self, **options):
        registry, run_id = target_run(self, options["run"])
        location = run_log_location(self.settings, run_id)
        payload = {"run_id": run_id, "path": str(location.path), "shared": location.shared,
                   "exists": location.exists}
        self.out.emit(payload, text=lambda: self.out.result(str(location.path)))
        return 0

    def sub_tail(self, **options):
        path, filter_run, live_run = self._resolve(options)
        if options["follow"] and self.out.fmt == "json":
            raise CommandError.usage("--follow streams; use --format jsonl or text")
        filters = {"node_path": options["node"], "kind": options["kind"],
                   "level": options["level"], "run_id": filter_run}
        if not path.exists() and not options["follow"]:
            raise ResourceNotFound(
                f"no log at {path}; the run was started with --no-log, or before per-run logs existed")
        end = complete_size(path)
        records = tail_jsonl(path, options["lines"], end=end, **filters)
        if self.out.fmt == "json":
            payload = {"run_id": live_run, "path": str(path), "count": len(records), "items": records}
            self.out.emit(payload, text=lambda: None)
            return 0
        for record in records:
            self._print(record)
        if not options["follow"]:
            return 0

        registry = self.registry_for_read() if live_run is not None else None
        lock_dir = lock_dir_for(self) if live_run is not None else None
        with StopFlag() as stop:                  # the first Ctrl-C ends the follow quietly
            def finished() -> bool:
                if stop.is_set():
                    return True
                if live_run is None or registry is None:
                    return False                                    # --file: until Ctrl-C
                row = registry.get_run(live_run)
                return row is None or liveness(lock_dir, row) != "live"

            try:
                for record in follow_jsonl(path, stop=finished, offset=end, **filters):
                    self._print(record)
            except KeyboardInterrupt:
                return int(ExitCode.INTERRUPTED)
        return int(ExitCode.INTERRUPTED) if stop.is_set() else int(ExitCode.OK)

    def _print(self, record: Mapping[str, Any]) -> None:
        if self.out.fmt == "jsonl":
            self.out.jsonl(record)
        else:
            self.out.result(render_log_line(record))
