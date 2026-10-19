"""Allow ``python -m mactf <command>``."""

from __future__ import annotations

from .management.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
