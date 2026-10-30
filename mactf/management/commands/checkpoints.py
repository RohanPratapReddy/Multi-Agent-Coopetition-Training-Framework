"""``checkpoints`` - list, verify, roll back, and prune a run's checkpoints. Section 8.13.

Every verb is one of the section 7.2 functions in ``orchestration.recovery``; the verbs
only choose arguments and print. ``torch`` is imported only when a file is verified.
"""

from __future__ import annotations

import argparse
from typing import Any

from ..durations import format_timestamp
from ..exitcodes import ExitCode
from ..group import GroupCommand, Sub
from ..output import Column, format_bytes
from ..runsupport import (
    RUN_OPTIONAL,
    RUN_REQUIRED,
    declined,
    exclusive_run,
    holder_of,
    non_negative_int,
    positive_int,
    require_registry,
    target_run,
)
from ...db import Registry
from ...exceptions import CommandError, ResourceNotFound, StateConflict
from ...orchestration import recovery
from ...orchestration.recovery import checkpoint_entry, checkpoint_roots, find_checkpoint, verify_checkpoint
from ...orchestration.runview import resolve_run_id


def episode_arg(text: str) -> int | str:
    if text.lower() == "latest":
        return "latest"
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an episode number or 'latest'") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"{text!r} must not be negative")
    return value


def checkpoint_row(registry: Registry, run_id: str, episode: int | str) -> dict[str, Any]:
    if episode == "latest":
        row = registry.latest_checkpoint(run_id)
        if row is None:
            raise ResourceNotFound(f"run {run_id!r} has no checkpoints")
        return row
    return find_checkpoint(registry, run_id, int(episode))


CHECKPOINT_COLUMNS = (
    Column("episode", "EP", align="r"),
    Column("kappa", "KAPPA", align="r", fmt=lambda value: f"{value:.3f}"),
    Column("entropy", "ENTROPY", align="r", fmt=lambda value: f"{value:.3f}"),
    Column("score", "SCORE", align="r", fmt=lambda value: f"{value:.4f}"),
    Column("size_bytes", "SIZE", align="r", fmt=format_bytes),
    Column("verified", "OK", fmt=lambda value: "yes" if value else "NO"),
    Column("path", "PATH"),
)


class Command(GroupCommand):
    help = "List, verify, roll back, and prune a run's checkpoints."
    category = "Runs"
    output_formats = ("text", "json", "yaml")
    subcommands = (
        Sub("list", "List a run's checkpoints.",
            arguments=(RUN_OPTIONAL,
                       (("--verify",), {"action": "store_true", "default": False,
                                        "help": "Hash and verify each file (reads every checkpoint)."}),
                       (("--valid-only",), {"action": "store_true", "default": False,
                                            "help": "Only checkpoints that verify; implies --verify."}),
                       (("--best",), {"action": "store_true", "default": False,
                                      "help": "Only the *-best checkpoints."})),
            examples=("mactf checkpoints list", "mactf checkpoints list 9f3a --verify")),
        Sub("show", "Describe and verify one checkpoint.",
            arguments=(RUN_REQUIRED, (("episode",), {"type": episode_arg,
                                                     "help": "Episode number or 'latest'."})),
            examples=("mactf checkpoints show 9f3a latest",)),
        Sub("verify", "Verify checkpoint files; exit 1 if any fails.",
            arguments=(RUN_REQUIRED, (("episode",), {"type": episode_arg, "nargs": "?", "default": None,
                                                     "help": "One episode; default: all."})),
            examples=("mactf checkpoints verify latest", "mactf checkpoints verify 9f3a 12")),
        Sub("rollback", "Make an earlier checkpoint the newest one, so a resume continues from it.",
            destructive=True,
            arguments=(RUN_REQUIRED, (("episode",), {"type": non_negative_int,
                                                     "help": "The episode to make newest."}),
                       (("--dry-run",), {"action": "store_true", "default": False,
                                         "help": "Say what would happen; change nothing."})),
            examples=("mactf checkpoints rollback 9f3a 12 --dry-run",)),
        Sub("prune", "Delete all but the newest checkpoints of a run.", destructive=True,
            arguments=((("run",), {"nargs": "?", "default": None, "completer": "run_id",
                                   "help": "Run id or unique prefix."}),
                       (("--all",), {"action": "store_true", "default": False,
                                     "help": "Every run no process is driving."}),
                       (("--keep",), {"type": positive_int, "default": 3,
                                      "help": "Newest checkpoints to keep per run (default 3)."}),
                       (("--no-keep-best",), {"action": "store_true", "default": False,
                                              "help": "Do not protect *-best checkpoints."}),
                       (("--dry-run",), {"action": "store_true", "default": False,
                                         "help": "Say what would be deleted; delete nothing."})),
            examples=("mactf checkpoints prune 9f3a --keep 2", "mactf checkpoints prune --all --dry-run")),
    )

    # -- list -----------------------------------------------------------------
    def sub_list(self, **options):
        registry, run_id = target_run(self, options["run"])
        deep = options["verify"] or options["valid_only"]
        items = [checkpoint_entry(row, deep=deep) for row in registry.checkpoints(run_id)]
        if options["best"]:
            items = [item for item in items if item["best"]]
        if options["valid_only"]:
            items = [item for item in items if item["verified"]]
        payload = {"run_id": run_id, "count": len(items), "items": items}

        def text() -> None:
            self.out.result(f"run {run_id}")
            self.out.table(items, CHECKPOINT_COLUMNS, empty="no checkpoints")
        self.out.emit(payload, text=text)
        return 0

    # -- show -----------------------------------------------------------------
    def sub_show(self, **options):
        registry, run_id = target_run(self, options["run"])
        entry = checkpoint_entry(checkpoint_row(registry, run_id, options["episode"]), deep=True)
        self.out.emit(entry, text=lambda: self._show_text(entry))
        return 0

    def _show_text(self, entry: dict[str, Any]) -> None:
        size = entry.get("size_bytes")
        verified = entry.get("verified")
        self.out.kv([
            ("run", entry["run_id"]), ("episode", entry["episode"]), ("path", entry["path"]),
            ("size", format_bytes(size) if size is not None else None),
            ("created", format_timestamp(entry["created_at"]) if entry.get("created_at") else None),
            ("kappa", None if entry.get("kappa") is None else f"{entry['kappa']:.3f}"),
            ("entropy", None if entry.get("entropy") is None else f"{entry['entropy']:.3f}"),
            ("score", None if entry.get("score") is None else f"{entry['score']:.4f}"),
            ("best", "yes" if entry.get("best") else "no"),
            ("verified", None if verified is None else ("yes" if verified else "NO")),
            ("sections", ", ".join(entry.get("sections") or []) or None),
            ("schema", entry.get("schema_version")),
        ])
        for problem in entry.get("problems") or []:
            self.out.result(f"  problem: {problem}")
        if not entry.get("exists", True):
            self.out.result("  problem: file does not exist")

    # -- verify ---------------------------------------------------------------
    def sub_verify(self, **options):
        registry, run_id = target_run(self, options["run"])
        rows = ([checkpoint_row(registry, run_id, options["episode"])] if options["episode"] is not None
                else registry.checkpoints(run_id))
        items = [{"episode": int(row["episode"]), "path": row["path"],
                  **verify_checkpoint(row["path"], run_id)} for row in rows]
        failed = [item for item in items if not item["ok"]]
        payload = {"run_id": run_id, "ok": not failed, "count": len(items), "failed": len(failed),
                   "items": items}

        def text() -> None:
            for item in items:
                state = "ok" if item["ok"] else "FAILED"
                self.out.result(f"  episode {item['episode']:>4} {state}  {item['path']}")
                for problem in item["problems"]:
                    self.out.result(f"      {problem}")
            self.out.result(f"{len(items) - len(failed)} of {len(items)} checkpoint(s) verified"
                            if items else "no checkpoints")
        self.out.emit(payload, text=text)
        return int(ExitCode.FAILURE) if failed else int(ExitCode.OK)

    # -- rollback -------------------------------------------------------------
    def sub_rollback(self, **options):
        registry, run_id = target_run(self, options["run"])
        episode = options["episode"]
        if options["dry_run"]:
            live = holder_of(self, run_id) is not None
            entry = recovery.rollback(registry, run_id, episode, live=live, source="cli", dry_run=True)
            self.out.emit(entry, text=lambda: self.out.result(
                f"would roll run {run_id} back to episode {episode}: archive "
                f"{len(entry['would_archive'])} file(s), remove {entry['would_remove_rows']} row(s)"))
            return 0
        newer = [row for row in registry.checkpoints(run_id) if int(row["episode"]) > episode]
        if not self.confirm(f"Roll run {run_id} back to episode {episode}? {len(newer)} newer checkpoint(s) "
                            "are archived as *.rolledback and forgotten."):
            return declined(self)
        writer = self.registry_for_write()
        with exclusive_run(self, run_id):
            entry = recovery.rollback(writer, run_id, episode, live=False, source="cli")

        def text() -> None:
            self.out.result(f"run {run_id} rolled back to episode {episode}; archived "
                            f"{len(entry['archived'])} file(s), removed {entry['rows_removed']} row(s)")
            self.out.result(f"continue with: {self.prog} run --resume {run_id}")
        self.out.emit(entry, text=text)
        return 0

    # -- prune ----------------------------------------------------------------
    def sub_prune(self, **options):
        registry = require_registry(self)
        if bool(options["run"]) == options["all"]:
            raise CommandError.usage("name a run, or pass --all (not both, not neither)")
        run_ids = ([resolve_run_id(registry, options["run"])] if options["run"]
                   else [row["id"] for row in registry.list_runs(limit=max(registry.count_runs(), 1))])
        roots = checkpoint_roots(self.settings)
        keep_best = not options["no_keep_best"]
        reports: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        for run_id in run_ids:
            if holder_of(self, run_id) is not None:
                skipped.append({"run_id": run_id, "reason": "being driven by a live process"})
                continue
            plan = recovery.prune(registry, run_id, keep=options["keep"], keep_best=keep_best,
                                  roots=roots, live=False, dry_run=True)
            if plan["removed"]:
                reports.append(plan)
        if not options["dry_run"] and reports:
            freed = sum(plan["freed_bytes"] for plan in reports)
            if not self.confirm(f"Delete {sum(len(plan['removed']) for plan in reports)} checkpoint file(s) "
                                f"from {len(reports)} run(s) ({format_bytes(freed)})?"):
                return declined(self)
            writer = self.registry_for_write()
            applied = []
            for plan in reports:
                run_id = plan["run_id"]
                try:
                    with exclusive_run(self, run_id):
                        applied.append(recovery.prune(writer, run_id, keep=options["keep"],
                                                      keep_best=keep_best, roots=roots, live=False))
                except StateConflict as exc:
                    skipped.append({"run_id": run_id, "reason": str(exc)})
            reports = applied
        payload = {"dry_run": options["dry_run"], "keep": options["keep"], "keep_best": keep_best,
                   "runs": reports, "skipped": skipped,
                   "freed_bytes": sum(plan["freed_bytes"] for plan in reports)}
        self.out.emit(payload, text=lambda: self._prune_text(payload))
        problems = [item for plan in reports for item in plan["removed"] if item["problem"]]
        return int(ExitCode.FAILURE) if problems else int(ExitCode.OK)

    def _prune_text(self, payload: dict[str, Any]) -> None:
        for plan in payload["runs"]:
            removed = plan["removed"]
            self.out.result(f"run {plan['run_id']}: kept episodes {plan['kept']}, "
                            f"{'would remove' if payload['dry_run'] else 'removed'} "
                            f"{len(removed)} file(s), {format_bytes(plan['freed_bytes'])}")
            for item in removed:
                if item["problem"]:
                    self.out.error(f"episode {item['episode']} {item['path']}: {item['problem']}")
        for item in payload["skipped"]:
            self.out.warn(f"skipped {item['run_id']}: {item['reason']}")
        if not payload["runs"] and not payload["skipped"]:
            self.out.result("nothing to prune")
        else:
            self.out.result(f"{'would free' if payload['dry_run'] else 'freed'} "
                            f"{format_bytes(payload['freed_bytes'])}")
