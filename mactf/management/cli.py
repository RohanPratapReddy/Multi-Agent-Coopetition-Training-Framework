"""Console entry point for ``mactf``, ``python -m mactf``, and ``manage.py``.

The distribution exposes one script, ``mactf``. Like ``django-admin`` it works from any
directory; unlike it, it also *finds the project*: when neither ``--settings`` nor
``MACTF_SETTINGS_MODULE`` names one, it looks for a generated ``manage.py`` at or above
the working directory (document 16, section 4.3) and falls back to the built-in default
settings only when there is none, so a fresh install can run ``mactf check`` before any
project exists.

A generated ``manage.py`` keeps its own explicit settings module and calls
``execute_from_command_line`` directly; discovery is not involved.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence


def _tolerate_console_encoding() -> None:
    """Print ``?`` instead of raising ``UnicodeEncodeError`` on a narrow Windows code page."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):
            pass                                        # a closed or exotic stream: leave it


def _add_path(directory: Path) -> None:
    text = str(directory)
    if text not in sys.path:
        sys.path.insert(0, text)


def main(argv: Sequence[str] | None = None) -> int:
    """Run a management command and return a process exit code."""
    from .bootstrap import discover_project, discovery_disabled, settings_chosen

    _tolerate_console_encoding()
    args = list(argv) if argv is not None else list(sys.argv[1:])
    if not settings_chosen(args):
        found = None if discovery_disabled(args) else discover_project()
        if found is not None:
            _add_path(found.pythonpath)
            os.environ["MACTF_SETTINGS_MODULE"] = found.module
            os.environ["MACTF_DISCOVERED_ROOT"] = str(found.root)
        else:
            os.environ.setdefault("MACTF_SETTINGS_MODULE", "mactf.settings_default")
    from . import execute_from_command_line

    return execute_from_command_line(["mactf", *args])


if __name__ == "__main__":  # pragma: no cover - module executed directly
    raise SystemExit(main())
