"""A schema-3 database meets the schema-4 build (document 17 section 14.7.2)."""

from __future__ import annotations

import time

from mactf.db import MIGRATIONS, Registry


def _v3(path) -> None:
    registry = Registry(path)
    with registry.connect() as connection:
        for migration in MIGRATIONS:
            if migration.version > 3:
                break
            for statement in migration.statements:
                connection.execute(statement)
            if migration.step is not None:
                migration.step(connection)
            connection.execute(
                "INSERT OR REPLACE INTO mactf_schema (version, applied_at) VALUES (?, ?)",
                (migration.version, time.time()),
            )


def test_a_version_3_database_refuses_a_new_run_until_migrated(project):
    path = project.root / "v3.sqlite3"
    _v3(path)
    out = project.invoke("migrate", "--check", "--database", str(path), "--format", "json")
    assert out.code == 1
    payload = out.json()
    assert payload["schema_version"] == 3 and payload["core_pending"] == 1 and not payload["up_to_date"]
    applied = project.invoke("migrate", "--database", str(path), "--format", "json")
    assert applied.code == 0 and applied.json()["applied"], applied.stderr
    assert Registry(path).schema_version() == 4
    again = project.invoke("migrate", "--check", "--database", str(path), "--format", "json")
    assert again.code == 0 and again.json()["core_pending"] == 0 and again.json()["up_to_date"] is True
