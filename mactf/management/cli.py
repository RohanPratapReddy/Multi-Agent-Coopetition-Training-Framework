"""Console entry point for ``mactf``, ``python -m mactf``, and ``manage.py``.

The distribution exposes one script, ``mactf``, which behaves like Django's
``django-admin``: it discovers settings through ``MACTF_SETTINGS_MODULE`` and
falls back to the built-in default settings module so a fresh install can run
``mactf check`` before any project exists.

A generated ``manage.py`` keeps its own explicit settings module, so both entry
points delegate to the same :func:`mactf.management.execute_from_command_line`.
"""

from __future__ import annotations

import os
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run a management command and return a process exit code."""
    os.environ.setdefault("MACTF_SETTINGS_MODULE", "mactf.settings_default")

    from . import execute_from_command_line

    args = list(argv) if argv is not None else list(sys.argv[1:])
    return execute_from_command_line(["mactf", *args])


if __name__ == "__main__":  # pragma: no cover - module executed directly
    raise SystemExit(main())
