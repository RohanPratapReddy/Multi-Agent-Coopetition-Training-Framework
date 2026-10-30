"""Registry migration 4 and ``record_components`` (document 17 section 14.6; invariant P7).

Migration 4 is the one irreversible step of document 17 (section 20.4): one nullable column,
no backfill, nothing existing changes.
"""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from mactf.db import MIGRATIONS, SCHEMA_VERSION, Registry, _migration_4_step
from mactf.exceptions import ImproperlyConfigured

MANIFEST = {
    "weight_update": {
        "TRAINER.WEIGHT_UPDATE": {
            "schema": 1, "kind": "weight_update", "name": "ema_softmax", "origin": "project",
            "version": "1", "spec_hash": "h", "params": {"alpha": 0.5}, "slot": "TRAINER.WEIGHT_UPDATE",
        }
    }
}


def _v3_database(path) -> Registry:
    """A database at exactly version 3: migrations 1-3 applied, nothing newer."""
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
    return registry


def _columns(registry: Registry, table: str = "runs") -> dict[str, tuple]:
    with registry.connect() as connection:
        return {row[1]: tuple(row) for row in connection.execute(f"PRAGMA table_info({table})")}


def _run(registry: Registry, run_id: str = "run-1", **kwargs):
    return registry.start_run(
        run_id, settings={"epochs": 3}, runtime={"device": "cpu"}, notes=kwargs.get("notes", "hello")
    )


def test_the_schema_version_is_4_and_the_last_migration_is_this_one():
    assert SCHEMA_VERSION == 4 == MIGRATIONS[-1].version
    assert [m.version for m in MIGRATIONS] == [1, 2, 3, 4]


# -- 22 -----------------------------------------------------------------------------------
def test_migration_4_adds_one_nullable_column(tmp_path):
    registry = _v3_database(tmp_path / "v3.sqlite3")
    assert registry.schema_version() == 3 and "components_json" not in _columns(registry)
    _run(registry, "old-run", notes="written before")
    with registry.connect() as connection:
        before = dict(connection.execute("SELECT * FROM runs WHERE id='old-run'").fetchone())
        tables_before = {r[0] for r in connection.execute("SELECT name FROM sqlite_master")}
    columns_before = set(_columns(registry))

    assert registry.migrate() == [4]
    assert registry.schema_version() == 4
    columns = _columns(registry)
    assert set(columns) - columns_before == {"components_json"}
    name, declared = columns["components_json"][1], columns["components_json"][2]
    assert (name, declared.upper()) == ("components_json", "TEXT")
    assert columns["components_json"][3] == 0 and columns["components_json"][4] is None   # nullable, no default

    with registry.connect() as connection:
        after = dict(connection.execute("SELECT * FROM runs WHERE id='old-run'").fetchone())
        tables_after = {r[0] for r in connection.execute("SELECT name FROM sqlite_master")}
    assert after.pop("components_json") is None
    assert after == before                                       # not one existing value changed
    # no new table either (``mactf_migrations`` is the app-migration bookkeeping ``migrate`` always creates)
    assert {t for t in tables_after if "mactf_migrations" not in t} == tables_before
    assert registry.components("old-run") is None
    assert ("core", 4) in registry.applied_migrations()


def test_a_version_4_reader_works_on_a_version_3_file_for_reads(tmp_path):
    registry = _v3_database(tmp_path / "v3.sqlite3")
    _run(registry, "old-run")
    assert registry.get_run("old-run")["components"] is None     # no column: still a row, still None
    assert registry.components("old-run") is None
    assert [row["id"] for row in registry.list_runs()] == ["old-run"]


# -- 23 -----------------------------------------------------------------------------------
def test_migration_4_is_idempotent(tmp_path):
    registry = _v3_database(tmp_path / "v3.sqlite3")
    statements: list[str] = []
    with registry.connect() as connection:
        connection.set_trace_callback(statements.append)
        _migration_4_step(connection)
        _migration_4_step(connection)                            # a re-run after a crash
        _migration_4_step(connection)
        connection.set_trace_callback(None)
    alters = [s for s in statements if s.strip().upper().startswith("ALTER TABLE")]
    assert len(alters) == 1 and "components_json" in alters[0]
    assert registry.migrate() == [4]
    assert registry.migrate() == []                              # the registry side is idempotent too
    assert list(_columns(registry)).count("components_json") == 1


def test_migration_4_survives_a_database_that_already_has_the_column(tmp_path):
    registry = _v3_database(tmp_path / "v3.sqlite3")
    with registry.connect() as connection:
        connection.execute("ALTER TABLE runs ADD COLUMN components_json TEXT")   # a hand-rolled build
    assert registry.migrate() == [4]
    assert registry.schema_version() == 4


def test_a_fresh_database_goes_straight_to_4(tmp_path):
    registry = Registry(tmp_path / "fresh.sqlite3")
    assert registry.migrate() == [1, 2, 3, 4]
    assert "components_json" in _columns(registry)


# -- 24, 25, 26, 27 -----------------------------------------------------------------------
def test_record_components_round_trip(tmp_path):
    registry = Registry(tmp_path / "r.sqlite3")
    registry.migrate()
    _run(registry)
    registry.record_components("run-1", MANIFEST, source="engine")
    assert registry.components("run-1") == MANIFEST
    changed = {**MANIFEST, "optimizer": {"OPTIMIZER": {"name": "Lion"}}}
    registry.record_components("run-1", changed, source="checkpoint")
    assert registry.components("run-1") == changed               # re-written, no history
    with registry.connect() as connection:
        raw = connection.execute("SELECT components_json FROM runs WHERE id='run-1'").fetchone()[0]
    assert json.loads(raw) == changed and raw == json.dumps(changed, sort_keys=True)


def test_record_components_for_a_missing_run_is_refused(tmp_path):
    registry = Registry(tmp_path / "r.sqlite3")
    registry.migrate()
    with pytest.raises(ImproperlyConfigured, match="no run 'ghost'.*start_run"):
        registry.record_components("ghost", MANIFEST)
    assert registry.components("ghost") is None


def test_empty_manifest_stores_null(tmp_path):
    registry = Registry(tmp_path / "r.sqlite3")
    registry.migrate()
    _run(registry)
    for empty in (None, {}):
        registry.record_components("run-1", MANIFEST)
        registry.record_components("run-1", empty)
        assert registry.components("run-1") is None
        with registry.connect() as connection:
            assert connection.execute("SELECT components_json FROM runs").fetchone()[0] is None


def test_get_run_decodes_components(tmp_path):
    registry = Registry(tmp_path / "r.sqlite3")
    registry.migrate()
    _run(registry)
    row = registry.get_run("run-1")
    assert row["components"] is None and "components_json" not in row
    registry.record_components("run-1", MANIFEST)
    row = registry.get_run("run-1")
    assert row["components"] == MANIFEST and "components_json" not in row
    # the other JSON columns keep their pre-doc-17 form (text), so no reader breaks
    assert isinstance(row["settings"], str) and json.loads(row["settings"]) == {"epochs": 3}
    assert registry.get_run("nope") is None
    assert "components_json" in registry.list_runs()[0]           # list views keep raw rows


def test_a_hand_edited_row_is_kept_as_text_not_a_failure(tmp_path):
    registry = Registry(tmp_path / "r.sqlite3")
    registry.migrate()
    _run(registry)
    with registry.connect() as connection:
        connection.execute("UPDATE runs SET components_json = 'not json'")
    assert registry.get_run("run-1")["components"] == "not json"


def test_record_components_on_a_version_3_file_says_to_migrate(tmp_path):
    registry = _v3_database(tmp_path / "v3.sqlite3")
    _run(registry)
    with pytest.raises(ImproperlyConfigured, match="migrate"):
        registry.record_components("run-1", MANIFEST)


def test_checkpoint_writer_survives_a_version_3_registry(tmp_path):
    """The writer treats a registry that cannot take the manifest as 'the file is authoritative'."""
    from mactf.orchestration.checkpoint_v2 import MACTFCheckpoint, CheckpointContext

    registry = _v3_database(tmp_path / "v3.sqlite3")
    _run(registry, "persist-run")
    writer = MACTFCheckpoint(tmp_path / "ckpt", run_id="persist-run", registry=registry)
    record = writer.save(
        CheckpointContext(identity_extra={"components": MANIFEST}), reason="manual", episode=0
    )
    assert record.path.exists()


def test_db_migration(tmp_path):
    """The exit-criterion summary: 3 -> 4 in place, then the manifest round-trips."""
    registry = _v3_database(tmp_path / "v3.sqlite3")
    _run(registry)
    assert registry.migrate() == [4]
    registry.record_components("run-1", MANIFEST)
    assert registry.get_run("run-1")["components"] == MANIFEST
    with sqlite3.connect(tmp_path / "v3.sqlite3") as raw:
        assert raw.execute("SELECT MAX(version) FROM mactf_schema").fetchone()[0] == 4
