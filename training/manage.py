#!/usr/bin/env python
"""Django-style entry point for the MACTF training project."""

from __future__ import annotations

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def main() -> int:
    os.environ.setdefault("MACTF_SETTINGS_MODULE", "training.settings")
    from mactf.management import execute_from_command_line

    return execute_from_command_line([sys.argv[0], *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
