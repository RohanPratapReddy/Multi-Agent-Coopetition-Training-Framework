"""``run`` - execute (start, bound, stop, continue) a training run. Document 16, section 8.2."""

from __future__ import annotations

from ..base import BaseCommand
from ..driver import RESUME_AUTO, DriveResult, RunDriver, RunPlan, plan_run, present_run_end
from ..runsupport import add_component_modules, device_arg, duration_arg, positive_int, run_id_arg


class Command(BaseCommand):
    help = "Run the multi-agent coopetition training loop."
    description = (
        "Run the epoch loop. Progress prints one line per epoch. Ctrl-C once pauses the run at the\n"
        "next epoch boundary so `--resume` can continue it; Ctrl-C twice aborts."
    )
    category = "Training"
    requires_system_checks = True
    output_formats = ("text", "json", "jsonl")
    examples = (
        "mactf run --epochs 3",
        "mactf run --epochs 20 --timeout 2h --run-id nightly-01",
        "mactf run --resume nightly-01 --epochs 40",
        "mactf run --branch-from nightly-01:12 --epochs 30 --set COOPETITION.WEIGHT_CAP=0.5",
        "mactf run --epochs 2 --format jsonl | jq -c 'select(.event == \"epoch\") | .score'",
    )
    arguments = [
        (("--epochs",), {"type": positive_int, "default": None,
                         "help": "Total epochs (not additional epochs when resuming)."}),
        (("--peers",), {"type": positive_int, "default": None, "help": "Limit the number of peers."}),
        (("--device",), {"type": device_arg, "default": None, "help": "auto, cpu, mps, cuda, or cuda:N."}),
        (("--seed",), {"type": int, "default": None}),
        (("--redteam",), {"action": "store_true", "default": False,
                          "help": "Force the stress pass on for this run."}),
        (("--no-redteam",), {"action": "store_true", "default": False,
                             "help": "Force the stress pass off for this run."}),
        (("--task",), {"default": None, "help": "Task source name, overriding TRAINING.TASK."}),
        (("--dry-run",), {"action": "store_true", "default": False,
                          "help": "Build the swarm and record a dry_run row; train nothing."}),
        (("--resume",), {"nargs": "?", "const": RESUME_AUTO, "default": None, "metavar": "RUN",
                         "completer": "resumable_run_id",
                         "help": "Continue a paused or interrupted run under its own id."}),
        (("--branch-from",), {"default": None, "metavar": "RUN[:EPISODE]", "completer": "run_id",
                              "help": "Start a new run from a checkpoint of another run."}),
        (("--run-id",), {"type": run_id_arg, "default": None, "help": "Id for the run (default: generated)."}),
        (("--timeout",), {"type": duration_arg, "default": None, "metavar": "DURATION",
                          "help": "Wall-clock budget; the run pauses at the next epoch boundary (exit 124)."}),
        (("--on-interrupt",), {"choices": ["pause", "cancel"], "default": "pause",
                               "help": "What Ctrl-C/SIGTERM makes of the run row."}),
        (("--log-file",), {"default": None, "metavar": "PATH",
                           "help": "Write the per-run JSONL log here instead of the default location."}),
        (("--no-log",), {"action": "store_true", "default": False, "help": "Write no per-run log."}),
        (("--component-module",), {"action": "append", "default": None, "metavar": "MODULE",
                                   "dest": "component_modules",
                                   "help": "Import MODULE and register its components for this run; repeat it."}),
    ]
    exclusive = (
        ("--resume", "--branch-from", "--dry-run"),
        ("--redteam", "--no-redteam"),
        ("--log-file", "--no-log"),
    )

    def handle(self, **options):
        settings = self.need_settings()
        add_component_modules(settings, options.get("component_modules"))
        if (settings.get("ORCHESTRATOR") or {}).get("ANNOUNCE"):
            for note in settings.get("_CONF_NOTES") or []:
                self.out.info(f"  conf     : {note}")
        plan = plan_run(self, options)
        result = RunDriver(self).execute(
            plan, timeout=options["timeout"], on_interrupt=options["on_interrupt"],
            log_path=options["log_file"], log_enabled=not options["no_log"],
        )
        self._finish(plan, result)
        return result.exit_code

    def _finish(self, plan: RunPlan, result: DriveResult) -> None:
        present_run_end(self, plan, result)
