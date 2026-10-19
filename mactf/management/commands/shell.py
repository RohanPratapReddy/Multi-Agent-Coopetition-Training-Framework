"""``shell`` - an interactive Python shell with MACTF bootstrapped."""

from __future__ import annotations

import code

from ..base import BaseCommand
from ... import apps
from ...conf import get_settings


class Command(BaseCommand):
    help = "Open an interactive shell with settings and apps loaded."
    arguments = [
        (("--interface",), {"default": "auto", "choices": ["auto", "ipython", "bpython", "plain"]}),
    ]

    def handle(self, **options):
        settings = get_settings()
        apps.populate()

        from .. import db, orchestration, rules, sandbox, scoring
        from ...coopetition import aggregation, adapter, peer, temperature

        namespace = {
            "settings": settings,
            "apps": apps,
            "rules": rules,
            "scoring": scoring,
            "sandbox": sandbox,
            "db": db,
            "orchestration": orchestration,
            "aggregation": aggregation,
            "adapter": adapter,
            "peer": peer,
            "temperature": temperature,
            "get_settings": get_settings,
        }
        banner = (
            "MACTF shell\n"
            f"settings={settings.module} apps={[config.label for config in apps.apps.get_app_configs()]}\n"
            "Names available: " + ", ".join(sorted(namespace))
        )

        interface = options.get("interface", "auto")
        if interface in {"auto", "ipython"}:
            try:
                from IPython import start_ipython  # type: ignore[import-not-found]

                start_ipython(argv=[], user_ns=namespace)
                return 0
            except ImportError:
                if interface == "ipython":
                    self.success("IPython is not installed")
                    return 1
        if interface in {"auto", "bpython"}:
            try:
                import bpython  # type: ignore[import-not-found]

                bpython.embed(locals_=namespace)
                return 0
            except ImportError:
                if interface == "bpython":
                    self.success("bpython is not installed")
                    return 1

        code.interact(banner=banner, local=namespace)
        return 0
