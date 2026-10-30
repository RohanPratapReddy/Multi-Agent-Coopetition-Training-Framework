"""Migration 2: schema upgrade, idempotency, and the new persistence methods."""

from __future__ import annotations

import sqlite3
import time

import pytest

from mactf.db import MIGRATIONS, SCHEMA_VERSION, Registry
from mactf.rules.severity import Severity
from mactf.scoring.warning import WarningPolicy, WarningRecord, WarningRegistry


def _v1_database(path) -> Registry:
    """Build a database at exactly version 1 by applying only migration 1."""
    registry = Registry(path)
    with registry.connect() as connection:
        connection.execute(
            "CREATE TABLE mactf_schema (version INTEGER PRIMARY KEY, applied_at REAL NOT NULL)"
        )
        for statement in MIGRATIONS[0].statements:
            connection.execute(statement)
        connection.execute(
            "INSERT INTO mactf_schema (version, applied_at) VALUES (1, ?)", (time.time(),)
        )
    return registry


def _columns(registry: Registry, table: str) -> set[str]:
    with registry.connect() as connection:
        return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def _tables(registry: Registry) -> set[str]:
    with registry.connect() as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    return {row["name"] for row in rows}


def test_v1_database_migrates_to_v2(tmp_path):
    registry = _v1_database(tmp_path / "v1.sqlite3")
    assert registry.schema_version() == 1
    assert registry.migrate() == [2, 3]
    assert registry.schema_version() == SCHEMA_VERSION == 3


def test_new_tables_exist(tmp_path):
    registry = Registry(tmp_path / "fresh.sqlite3")
    registry.migrate()
    tables = _tables(registry)
    for name in ("warnings", "adjudications", "aggregations", "resume_reports"):
        assert name in tables, f"{name} missing after migration 2"


def test_rule_events_gains_reversal_columns(tmp_path):
    registry = _v1_database(tmp_path / "v1.sqlite3")
    registry.migrate()
    columns = _columns(registry, "rule_events")
    assert "reversed_at" in columns
    assert "reversed_by" in columns


def test_migration_is_idempotent(tmp_path):
    registry = Registry(tmp_path / "fresh.sqlite3")
    assert registry.migrate() == [1, 2, 3]
    assert registry.migrate() == []


def test_second_migration_on_v1_does_not_duplicate_columns(tmp_path):
    registry = _v1_database(tmp_path / "v1.sqlite3")
    registry.migrate()
    # Force a re-run of the step: it must not raise on the existing columns.
    with registry.connect() as connection:
        MIGRATIONS[1].step(connection)
    assert _columns(registry, "rule_events") >= {"reversed_at", "reversed_by"}


def test_fresh_database_goes_straight_to_v2(tmp_path):
    registry = Registry(tmp_path / "fresh.sqlite3")
    assert registry.schema_version() == 0
    registry.migrate()
    assert registry.schema_version() == 3
    assert {"warnings", "adjudications", "aggregations", "resume_reports"} <= _tables(registry)


# -- warning persistence -------------------------------------------------


def _record(rule_id="P6", episode=0, **kwargs) -> WarningRecord:
    defaults = dict(
        rule_id=rule_id,
        episode=episode,
        severity=Severity.MINOR,
        evidence="ignored suggestion",
        expiry_episode=episode + 5,
    )
    defaults.update(kwargs)
    return WarningRecord(**defaults)


def test_record_and_read_warning(tmp_path):
    registry = Registry(tmp_path / "db.sqlite3")
    registry.migrate()
    registry.record_warning("run1", "peer_a", _record())
    active = registry.active_warnings("run1")
    assert len(active) == 1
    assert active[0]["peer"] == "peer_a"
    assert active[0]["rule_id"] == "P6"
    assert active[0]["active"] == 1


def test_record_warning_is_idempotent_per_episode(tmp_path):
    registry = Registry(tmp_path / "db.sqlite3")
    registry.migrate()
    registry.record_warning("run1", "peer_a", _record())
    registry.record_warning("run1", "peer_a", _record(evidence="changed"))
    active = registry.active_warnings("run1")
    assert len(active) == 1
    assert active[0]["evidence"] == "changed"


def test_expire_warning_clears_active(tmp_path):
    registry = Registry(tmp_path / "db.sqlite3")
    registry.migrate()
    registry.record_warning("run1", "peer_a", _record(episode=2))
    registry.expire_warning("run1", "peer_a", "P6", 2, expired_at=time.time())
    assert registry.active_warnings("run1") == []
    rows = registry.all_warnings("run1")
    assert rows[0]["expired_at"] is not None


def test_restore_warnings_rebuilds_registry(tmp_path):
    registry = Registry(tmp_path / "db.sqlite3")
    registry.migrate()
    for episode in range(3):
        registry.record_warning("run1", "peer_a", _record(episode=episode))

    target = WarningRegistry(WarningPolicy())
    restored = registry.restore_warnings("run1", target, peer="peer_a")
    assert restored == 3
    # Restore reproduces the persisted state exactly; W_CAP is enforced when a
    # new warning is issued, not by silently dropping history on resume.
    assert target.count("peer_a", "P6") == 3
    assert len(target.all_active("peer_a")["P6"]) == 3


def test_restore_derives_clean_streak(tmp_path):
    """The streak is derived from episode distance, not stored."""
    registry = Registry(tmp_path / "db.sqlite3")
    registry.migrate()
    registry.record_warning("run1", "peer_a", _record(episode=0))

    target = WarningRegistry(WarningPolicy())
    registry.restore_warnings("run1", target, peer="peer_a", episode=4)
    # The warning was at episode 0 and the run is at 4, so the streak is 3.
    assert target._clean_streak["peer_a"]["P6"] == 3

    registry.record_warning("run1", "peer_b", _record(rule_id="P11", episode=4))
    second = WarningRegistry(WarningPolicy())
    registry.restore_warnings("run1", second, peer="peer_b", episode=7)
    assert second._clean_streak["peer_b"]["P11"] == 2


def test_resumed_warning_escalates_at_k_esc(tmp_path):
    """Framework.md W3: three active warnings escalate on the next occurrence."""
    registry = Registry(tmp_path / "db.sqlite3")
    registry.migrate()
    for episode in range(2):
        registry.record_warning("run1", "peer_a", _record(episode=episode))

    policy = WarningPolicy(k_esc=3, n_win=5, n_decay=3, w_cap=2, p_warn=5.0)
    restored = WarningRegistry(policy)
    registry.restore_warnings("run1", restored, peer="peer_a")
    outcome = restored.issue("peer_a", "P6", severity="S1", episode=2)
    assert outcome.escalated is not None
    assert outcome.penalty == pytest.approx(5.0)


# -- adjudications -------------------------------------------------------


def test_adjudication_round_trip(tmp_path):
    registry = Registry(tmp_path / "db.sqlite3")
    registry.migrate()
    identifier = registry.record_adjudication(
        "run1",
        peer="peer_a",
        rule_id="P1",
        episode=3,
        decision="overturned",
        severity="S3",
        rationale="the citation resolves; the auditor used a stale index",
        adjudicator="reviewer@example",
    )
    assert identifier > 0
    rows = registry.adjudications("run1")
    assert len(rows) == 1
    assert rows[0]["decision"] == "overturned"
    assert rows[0]["adjudicator"] == "reviewer@example"


def test_adjudication_without_run_id(tmp_path):
    registry = Registry(tmp_path / "db.sqlite3")
    registry.migrate()
    registry.record_adjudication(
        None,
        peer="peer_a",
        rule_id="P6",
        episode=0,
        decision="upheld",
        rationale="ignored",
        adjudicator="reviewer",
    )
    assert registry.adjudications("run1") == []


def test_mark_events_reversed(tmp_path):
    registry = Registry(tmp_path / "db.sqlite3")
    registry.migrate()
    registry.record_events(
        "run1", 3, "peer_a", [{"rule_id": "P1", "magnitude": 1.0, "severity": "S3"}]
    )
    assert registry.mark_events_reversed("run1", "peer_a", "P1", 3, by="reviewer") == 1
    assert registry.mark_events_reversed("run1", "peer_a", "P1", 3, by="reviewer") == 0
    with registry.connect() as connection:
        row = connection.execute("SELECT * FROM rule_events").fetchone()
    assert row["reversed_by"] == "reviewer"
    assert row["reversed_at"] is not None


# -- aggregations --------------------------------------------------------


class _Report:
    def __init__(self, episode: int) -> None:
        self.episode = episode

    def as_dict(self):
        return {"episode": self.episode, "kappa": 1.0 + self.episode, "weights": {}}


def test_aggregation_history_is_ordered(tmp_path):
    registry = Registry(tmp_path / "db.sqlite3")
    registry.migrate()
    for episode in (2, 0, 1):
        registry.record_aggregation("run1", episode, _Report(episode))
    rows = registry.aggregations("run1")
    assert [row["episode"] for row in rows] == [0, 1, 2]
    assert rows[2]["report"]["kappa"] == 3.0


def test_aggregation_is_replaced_per_episode(tmp_path):
    registry = Registry(tmp_path / "db.sqlite3")
    registry.migrate()
    registry.record_aggregation("run1", 0, _Report(0))
    registry.record_aggregation("run1", 0, _Report(0))
    assert len(registry.aggregations("run1")) == 1


def test_latest_aggregation(tmp_path):
    registry = Registry(tmp_path / "db.sqlite3")
    registry.migrate()
    assert registry.latest_aggregation("run1") is None
    registry.record_aggregation("run1", 0, _Report(0))
    registry.record_aggregation("run1", 1, _Report(1))
    latest = registry.latest_aggregation("run1")
    assert latest is not None and latest["episode"] == 1


# -- resume reports ------------------------------------------------------


def test_resume_report_round_trip(tmp_path):
    registry = Registry(tmp_path / "db.sqlite3")
    registry.migrate()
    registry.record_resume_report(
        "run1",
        checkpoint_episode=2,
        started_epoch=3,
        warning_count=4,
        expired_count=1,
        report={"source": "checkpoint"},
    )
    rows = registry.resume_reports("run1")
    assert len(rows) == 1
    assert rows[0]["checkpoint_episode"] == 2
    assert rows[0]["started_epoch"] == 3
    assert rows[0]["warnings_restored"] == 4
    assert rows[0]["report"]["source"] == "checkpoint"
