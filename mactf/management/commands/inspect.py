"""``inspect`` - show a run's scorecards, rule counts, and checkpoints."""

from __future__ import annotations

import json

from ..base import BaseCommand
from ...conf import get_settings
from ...db import Registry


class Command(BaseCommand):
    help = "Inspect runs, episodes, rule counts, and checkpoints."
    arguments = [
        (("target",), {"nargs": "?", "default": None, "help": "Run id, or omit for the latest."}),
        (("--runs",), {"action": "store_true", "default": False, "help": "List recent runs."}),
        (("--episodes",), {"action": "store_true", "default": False}),
        (("--rules",), {"action": "store_true", "default": False}),
        (("--checkpoints",), {"action": "store_true", "default": False}),
        (("--json",), {"action": "store_true", "default": False}),
    ]

    def handle(self, **options):
        settings = get_settings()
        registry = Registry(settings.database_path())
        target = options.get("target")

        if options.get("runs"):
            runs = registry.list_runs(limit=25)
            if options.get("json"):
                self.stdout.write(json.dumps(runs, indent=2, default=str) + "\n")
                return 0
            if not runs:
                self.success("no runs recorded")
                return 0
            for run in runs:
                self.success(
                    f"{run['id']}  {run['status']:<10} "
                    f"kappa={run['kappa'] if run['kappa'] is not None else '-'} "
                    f"entropy={run['entropy'] if run['entropy'] is not None else '-'}"
                )
            return 0

        if target is None:
            latest = registry.list_runs(limit=1)
            if not latest:
                self.success("no runs recorded; try `manage.py run --epochs 1`")
                return 0
            target = latest[0]["id"]

        run = registry.get_run(target)
        if run is None:
            self.success(f"no such run: {target}")
            return 1

        self.success(f"run {target} [{run['status']}]")
        if not options.get("episodes") and not options.get("rules") and not options.get("checkpoints"):
            self.success(json.dumps(run, indent=2, default=str))
            return 0

        if options.get("episodes"):
            for episode in registry.episodes(target):
                card = episode["scorecard"]
                scores = {
                    peer: round(payload["final_score"], 4)
                    for peer, payload in card.get("peers", {}).items()
                }
                self.success(f"  episode {episode['episode']}: {scores}")

        if options.get("rules"):
            counts = registry.rule_counts(target)
            if not counts:
                self.success("  no rule events")
            for rule_id, count in counts.items():
                self.success(f"  {rule_id}: {count}")

        if options.get("checkpoints"):
            for checkpoint in registry.checkpoints(target):
                self.success(
                    f"  ep{checkpoint['episode']:>3} kappa={checkpoint['kappa']:.3f} "
                    f"score={checkpoint['score']:.4f} {checkpoint['path']}"
                )
        return 0
