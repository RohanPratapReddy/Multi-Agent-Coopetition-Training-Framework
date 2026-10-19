"""``run`` - execute a training run."""

from __future__ import annotations

from ..base import BaseCommand
from ...conf import get_settings
from ...exceptions import CommandError
from ...orchestration.orchestrator import Orchestrator, RunConfig


class Command(BaseCommand):
    help = "Run the multi-agent coopetition training loop."
    requires_system_checks = True
    arguments = [
        (("--epochs",), {"type": int, "default": None}),
        (("--peers",), {"type": int, "default": None, "help": "Limit the number of peers."}),
        (("--device",), {"default": None, "help": "auto, cpu, cuda, cuda:0, or mps."}),
        (("--seed",), {"type": int, "default": None}),
        (("--redteam",), {"action": "store_true", "default": False}),
        (("--dry-run",), {"action": "store_true", "default": False}),
        (("--resume",), {"default": None, "help": "Resume from the latest checkpoint."}),
        (("--run-id",), {"default": None}),
        (("--json",), {"action": "store_true", "default": False}),
    ]

    def handle(self, **options):
        settings = get_settings()
        config = RunConfig(
            epochs=options.get("epochs"),
            peers=options.get("peers"),
            device=options.get("device"),
            seed=options.get("seed"),
            dry_run=bool(options.get("dry_run")),
            redteam=bool(options.get("redteam")),
            resume_from=True if options.get("resume") else None,
            run_id=options.get("run_id"),
        )
        orchestrator = Orchestrator(settings)
        try:
            summary = orchestrator.run(config)
        except Exception as exc:  # noqa: BLE001 - surface a clean CLI failure
            raise CommandError(f"training failed: {exc}") from exc

        if options.get("json"):
            import json

            self.stdout.write(json.dumps(summary, indent=2, default=str) + "\n")
            return 0

        self.success(f"run {summary['run_id']} [{summary['status']}]")
        runtime = summary.get("runtime", {})
        self.success(f"  device   : {runtime.get('device')} dtype={runtime.get('dtype')}")
        self.success(f"  peers    : {', '.join(summary.get('peers', []))}")
        self.success(f"  sandbox  : {summary.get('sandbox', 'none')}")
        for note in summary.get("notes", []):
            self.success(f"  note     : {note}")
        for entry in summary.get("history", []):
            weights = ", ".join(
                f"{peer}={weight:.3f}" for peer, weight in entry.get("weights", {}).items()
            )
            self.success(
                f"  epoch {entry['epoch']}: score={entry['score']:.4f} "
                f"kappa={entry['kappa']:.3f} entropy={entry['entropy']:.3f} [{weights}]"
            )
        return 0
