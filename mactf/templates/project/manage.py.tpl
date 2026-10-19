#!/usr/bin/env python
"""Django-style entry point for the {{ project_name }} MACTF project."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The project directory *is* the package directory, so its parent goes on the
# path and everything created at runtime (var/, checkpoints) stays beside it.
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))


def main() -> int:
    os.environ.setdefault("MACTF_SETTINGS_MODULE", "{{ project_name }}.settings")
    from mactf.management import execute_from_command_line

    return execute_from_command_line([sys.argv[0], *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
