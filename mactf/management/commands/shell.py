"""``shell`` - an interactive or scripted Python shell with MACTF bootstrapped. Section 8.9."""

from __future__ import annotations

import code
import traceback
from typing import Any

from ..base import BaseCommand
from ..exitcodes import ExitCode
from ... import apps
from ...exceptions import CapabilityUnavailable


class Command(BaseCommand):
    help = "Open an interactive shell with settings and apps loaded."
    category = "Project"
    output_formats = ("text",)
    examples = ("mactf shell", "mactf shell -c \"print(settings.database_path())\"",
                "mactf shell --interface plain < script.py")
    arguments = [
        (("--interface",), {"default": "auto", "choices": ["auto", "ipython", "bpython", "plain"],
                            "help": "Which interactive interpreter to use."}),
        (("-c", "--command"), {"default": None, "metavar": "CODE",
                               "help": "Execute CODE in the namespace and exit."}),
    ]

    def handle(self, **options):
        settings = self.settings
        apps.populate()
        namespace = self._namespace(settings)

        source = options["command"]
        if source is None and not self._interactive_stdin():
            source = self.stdin.read()
        if source is not None:
            return self._execute(source, namespace)

        interface = options["interface"]
        if interface in ("auto", "ipython"):
            try:
                from IPython import start_ipython  # type: ignore[import-not-found]
            except ImportError:
                if interface == "ipython":
                    raise CapabilityUnavailable("IPython is not installed; `pip install ipython`") from None
            else:
                start_ipython(argv=[], user_ns=namespace)
                return 0
        if interface in ("auto", "bpython"):
            try:
                import bpython  # type: ignore[import-not-found]
            except ImportError:
                if interface == "bpython":
                    raise CapabilityUnavailable("bpython is not installed; `pip install bpython`") from None
            else:
                bpython.embed(locals_=namespace)
                return 0
        code.interact(banner=self._banner(settings, namespace), local=namespace)
        return 0

    def _interactive_stdin(self) -> bool:
        isatty = getattr(self.stdin, "isatty", None)
        return bool(isatty and isatty())

    def _namespace(self, settings: Any) -> dict[str, Any]:
        from ... import db, orchestration, rules, sandbox, scoring
        from ...conf import get_settings
        from ...coopetition import adapter, aggregation, peer, temperature

        def get_registry():
            """The registry, created and migrated on first call."""
            return self.registry_for_write()

        return {
            "settings": settings, "apps": apps, "rules": rules, "scoring": scoring,
            "sandbox": sandbox, "db": db, "orchestration": orchestration,
            "aggregation": aggregation, "adapter": adapter, "peer": peer,
            "temperature": temperature, "get_settings": get_settings,
            "get_registry": get_registry,
        }

    def _banner(self, settings: Any, namespace: dict[str, Any]) -> str:
        labels = [config.label for config in apps.apps.get_app_configs()]
        return (
            "MACTF shell\n"
            f"settings={settings.module} apps={labels}\n"
            "Names available: " + ", ".join(sorted(namespace))
        )

    def _execute(self, source: str, namespace: dict[str, Any]) -> int:
        try:
            compiled = compile(source, "<mactf shell>", "exec")
        except SyntaxError:
            traceback.print_exc(file=self.stderr)
            return int(ExitCode.FAILURE)
        try:
            exec(compiled, namespace)                          # noqa: S102 - the operator's own code
        except SystemExit as exc:
            if isinstance(exc.code, int):
                return int(exc.code)
            return int(ExitCode.OK if exc.code is None else ExitCode.FAILURE)
        except Exception:                                      # noqa: BLE001 - report as python does
            traceback.print_exc(file=self.stderr)
            return int(ExitCode.FAILURE)
        return 0
