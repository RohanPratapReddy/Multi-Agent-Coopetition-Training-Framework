"""``migrate`` - apply run-registry schema migrations."""

from __future__ import annotations

from ..base import BaseCommand
from ...conf import get_settings
from ...db import Registry, SCHEMA_VERSION


class Command(BaseCommand):
    help = "Create or upgrade the run registry database."
    arguments = [
        (("--database",), {"default": None, "help": "Override the sqlite path."}),
        (("--check",), {"action": "store_true", "default": False, "help": "Report only."}),
    ]

    def handle(self, **options):
        settings = get_settings()
        registry = Registry(options.get("database") or settings.database_path())
        current = registry.schema_version()
        if options.get("check"):
            self.success(f"schema version: {current} (target {SCHEMA_VERSION})")
            return 0 if current >= SCHEMA_VERSION else 1
        applied = registry.migrate()
        if applied:
            self.success(f"applied migrations: {', '.join(str(version) for version in applied)}")
        else:
            self.success(f"no migrations needed (schema at {registry.schema_version()})")
        self.success(f"database: {registry.path}")
        return 0
