"""``check`` - run system checks (document 16, section 8.5)."""

from __future__ import annotations

from dataclasses import asdict

from ...checks import ERROR, list_checks, run_checks
from ..base import BaseCommand
from ..exitcodes import ExitCode
from ..output import Column


class Command(BaseCommand):
    help = "Run framework system checks (rules, scoring, peers, runtime, sandbox)."
    category = "Project"
    requires_system_checks = False
    output_formats = ("text", "json")
    examples = (
        "mactf check",
        "mactf check --fail-level WARNING",
        "mactf check --app mactf.contrib.api --format json",
        "mactf check --list",
    )
    arguments = [
        (("--fail-level",), {"default": ERROR, "choices": ["ERROR", "WARNING", "INFO"],
                             "help": "The lowest level that makes the command fail."}),
        (("--app",), {"default": None, "completer": "check_app", "help": "Only this app's checks."}),
        (("--tag",), {"action": "append", "default": None, "completer": "check_tag",
                      "help": "Only checks carrying this tag; repeatable."}),
        (("--id",), {"action": "append", "default": None, "metavar": "PREFIX",
                     "help": "Only messages whose id starts with PREFIX (mactf.E0); repeatable."}),
        (("--list",), {"action": "store_true", "default": False,
                       "help": "List the registered checks and exit; runs none."}),
    ]

    def handle(self, **options):
        if options["list"]:
            rows = [asdict(info) for info in list_checks()]
            self.out.emit(
                {"checks": rows},
                text=lambda: self.out.table(rows, [
                    Column("name", "CHECK"), Column("app", "APP"),
                    Column("tags", "TAGS", fmt=lambda tags: ",".join(tags) or "-"),
                    Column("requires", "REQUIRES"), Column("doc", "WHAT", max_width=60)]),
            )
            return 0
        outcome = run_checks(fail_level=options["fail_level"], app=options["app"],
                             tags=options["tag"] or (), ids=options["id"] or ())
        payload = outcome.as_dict()
        settings = self.need_settings()
        payload["conf_notes"] = list(settings.get("_CONF_NOTES") or [])
        quiet = self.out.verbosity < 1 or bool(options.get("quiet"))

        def text() -> None:
            if not quiet:
                for note in payload["conf_notes"]:
                    self.out.info(f"conf: {note}")
                for message in outcome.messages:
                    if message.level == ERROR:
                        self.out.error(str(message))
                    else:
                        self.out.result(str(message))
            self.out.result(outcome.summary())

        self.out.emit(payload, text=text)
        return int(ExitCode.OK if outcome.ok else ExitCode.FAILURE)
