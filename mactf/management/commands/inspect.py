"""``inspect`` - a run's scorecards, rule counts, and checkpoints. Document 16, section 8.7.

The compatibility front for ``runs show`` and ``runs list``: it delegates to the same
``collect_run_report``, so the two cannot disagree.
"""

from __future__ import annotations

from ..base import BaseCommand
from ..runsupport import (
    RUN_COLUMNS,
    SECTIONS,
    collect_run_report,
    lock_dir_for,
    render_run_report,
    run_listing,
    target_run,
)
from ...exceptions import ResourceNotFound


class Command(BaseCommand):
    help = "Inspect runs, episodes, rule counts, and checkpoints."
    category = "Runs"
    output_formats = ("text", "json")
    examples = ("mactf inspect", "mactf inspect 9f3a --episodes --rules", "mactf inspect --runs --json")
    arguments = [
        (("target",), {"nargs": "?", "default": None, "completer": "run_id",
                       "help": "Run id, or omit for the latest."}),
        (("--runs",), {"action": "store_true", "default": False, "help": "List recent runs."}),
        *[((f"--{name}",), {"action": "store_true", "default": False,
                            "help": f"Include the {name} section."}) for name in SECTIONS],
    ]

    def handle(self, **options):
        registry = self.registry_for_read()
        if options["runs"]:
            rows = registry.list_runs(limit=25) if registry is not None else []
            # --json keeps printing the registry rows exactly as before: a bare array.
            self.out.emit(rows, text=lambda: self._list(registry, rows))
            return 0
        if registry is None or registry.count_runs() == 0:
            if options["target"] is None:
                self.out.emit({"run": None}, text=lambda: self.out.result(
                    f"no runs recorded; try `{self.prog} run --epochs 1`"))
                return 0
            raise ResourceNotFound("no runs recorded")
        registry, run_id = target_run(self, options["target"] or "latest")
        sections = {name for name in SECTIONS if options.get(name)}
        report = collect_run_report(registry, run_id, sections=sections, lock_dir=lock_dir_for(self))
        self.out.emit(report, text=lambda: render_run_report(self.out, report))
        return 0

    def _list(self, registry, rows) -> None:
        items = run_listing(registry, rows, lock_dir_for(self)) if rows else []
        self.out.table(items, RUN_COLUMNS, empty="no runs recorded")
