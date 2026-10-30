"""mactf version: the framework version, and optionally what it runs on."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

from ..base import BaseCommand

COMPONENTS = ("torch", "PyYAML", "numpy", "scipy", "peft", "transformers", "bitsandbytes",
              "uvicorn", "ipython", "bpython")


def installed(distribution: str) -> str | None:
    """The installed version from package metadata; never imports the package."""
    from importlib import metadata

    try:
        return metadata.version(distribution)
    except Exception:  # noqa: BLE001 - absent, or metadata corrupt: both mean "not usable"
        return None


class Command(BaseCommand):
    help = "Print the framework version."
    category = "Meta"
    requires_settings = False
    output_formats = ("text", "json")
    arguments = [(("--components",), {
        "action": "store_true", "default": False,
        "help": "Also print Python, the platform, and installed dependencies."})]
    examples = ("mactf version", "mactf version --components", "mactf version --format json")

    def handle(self, **options):
        import mactf

        payload = {
            "name": "mactf", "version": mactf.__version__, "release": list(mactf.VERSION),
            "python": {"version": platform.python_version(),
                       "implementation": platform.python_implementation(),
                       "executable": sys.executable},
            "platform": platform.platform(),
            "install": str(Path(mactf.__file__).resolve().parent),
            "project": {"settings_module": os.environ.get("MACTF_SETTINGS_MODULE"),
                        "root": os.environ.get("MACTF_DISCOVERED_ROOT")},
            "components": {name: installed(name) for name in COMPONENTS},
        }

        def text() -> None:
            if not options["components"]:
                self.out.result(payload["version"])      # bare, as it has always been: scripts parse it
                return
            self.out.result(f"mactf {payload['version']}")
            python = payload["python"]
            project = payload["project"]
            rows = [
                ("python", f"{python['version']} ({python['implementation']}) {python['executable']}"),
                ("platform", payload["platform"]), ("install", payload["install"]),
                ("project", (project["settings_module"] or "(framework defaults)")
                 + (f" ({project['root']})" if project["root"] else "")),
            ]
            rows += [(name, version or "not installed")
                     for name, version in payload["components"].items()]
            self.out.kv(rows)

        self.out.emit(payload, text=text)
        return 0
