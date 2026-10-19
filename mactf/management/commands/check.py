"""``check`` - run system checks."""

from __future__ import annotations

from ..base import BaseCommand
from ...checks import run_checks


class Command(BaseCommand):
    help = "Run framework system checks (rules, scoring, peers, runtime, sandbox)."
    requires_system_checks = False
    arguments = [
        (("--fail-level",), {"default": "ERROR", "choices": ["ERROR", "WARNING", "INFO"]}),
        (("--quiet",), {"action": "store_true", "default": False}),
    ]

    def handle(self, **options):
        outcome = run_checks()
        if not options.get("quiet"):
            for message in outcome.messages:
                stream = self.stderr if message.level == "ERROR" else self.stdout
                stream.write(f"{message}\n")
        self.success(outcome.summary())
        return 0 if outcome.ok else 1
