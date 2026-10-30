"""``migrate`` - apply run-registry schema migrations. Document 16, section 8.6.

``--check`` and ``--plan`` only look: they never create the database file (a missing file is
schema version 0 with everything pending) and they count app migrations, not just core's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..base import BaseCommand
from ..exitcodes import ExitCode
from ...db import SCHEMA_VERSION, MigrationFailed, Registry, migration_plan
from ...exceptions import CommandError


class Command(BaseCommand):
    help = "Create or upgrade the run registry database."
    category = "Project"
    output_formats = ("text", "json")
    examples = ("mactf migrate", "mactf migrate --check", "mactf migrate --plan --format json")
    arguments = [
        (("--database",), {"default": None, "metavar": "PATH", "completer": "file",
                           "help": "Override the sqlite path."}),
        (("--check",), {"action": "store_true", "default": False,
                        "help": "Report only; exit 1 if the schema is behind or a migration is pending."}),
        (("--plan",), {"action": "store_true", "default": False,
                       "help": "List pending migrations; apply none."}),
    ]
    exclusive = (("--check", "--plan"),)

    def handle(self, **options):
        path = Path(options["database"] or self.need_settings().database_path()).expanduser()
        plan = migration_plan()
        exists = path.exists()
        read_only = bool(options["check"] or options["plan"])
        registry: Registry | None
        if read_only:
            registry = Registry(path, read_only=True) if exists else None
        else:
            registry = Registry(path)
        current = registry.schema_version() if registry is not None and exists else 0
        core_pending = max(0, SCHEMA_VERSION - current)
        if registry is not None and exists:
            pending = [(app, version) for app, version, _ in registry.pending_plan(plan)]
        else:
            pending = [(app, version) for app, version, _ in plan]
        payload: dict[str, Any] = {
            "database": str(path), "exists": exists, "schema_version": current,
            "target_version": SCHEMA_VERSION, "core_pending": core_pending,
            "pending": [{"app": app, "version": version} for app, version in pending],
            "applied": [], "applied_apps": [],
            "mode": "check" if options["check"] else "plan" if options["plan"] else "apply",
        }
        if read_only:
            payload["up_to_date"] = core_pending == 0 and not pending
            self.out.emit(payload, text=lambda: self._text(payload))
            return int(ExitCode.OK if payload["up_to_date"] or options["plan"] else ExitCode.FAILURE)

        assert registry is not None
        payload["applied"] = list(registry.migrate())
        try:
            payload["applied_apps"] = [{"app": app, "version": version}
                                       for app, version in registry.migrate_apps(plan)]
        except MigrationFailed as exc:
            raise CommandError(str(exc)) from exc
        payload["exists"] = True
        payload["schema_version"] = registry.schema_version()
        payload["pending"] = []
        payload["core_pending"] = 0
        payload["up_to_date"] = True
        self.out.emit(payload, text=lambda: self._text(payload))
        return 0

    def _text(self, payload: Mapping[str, Any]) -> None:
        out = self.out
        note = "" if payload["exists"] or payload["mode"] == "apply" else " (not created yet)"
        out.result(f"database: {payload['database']}{note}")
        out.result(f"schema version: {payload['schema_version']} (target {payload['target_version']})")
        for item in payload["pending"]:
            out.result(f"  pending {item['app']}/{item['version']}")
        if payload["applied"]:
            out.result(f"applied migrations: {', '.join(str(version) for version in payload['applied'])}")
        for item in payload["applied_apps"]:
            out.result(f"applied {item['app']}/{item['version']}")
        if payload["mode"] == "apply" and not payload["applied"] and not payload["applied_apps"]:
            out.result(f"no migrations needed (schema at {payload['schema_version']})")
        if payload["mode"] != "apply":
            out.result("up to date" if payload["up_to_date"] else "migrations are pending; run `migrate`")
