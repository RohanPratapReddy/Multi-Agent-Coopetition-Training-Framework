"""The run registry: a small, versioned sqlite store for runs and episodes.

Modeled on Django's ``migrate``: schema changes are numbered, applied in order,
and recorded in ``mactf_schema`` so a second ``migrate`` is a no-op. JSON columns
hold scorecards so the schema stays stable while scoring evolves.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .exceptions import ImproperlyConfigured

SCHEMA_VERSION = 1

MIGRATIONS: dict[int, list[str]] = {
    1: [
        """
        CREATE TABLE IF NOT EXISTS mactf_schema (
            version INTEGER PRIMARY KEY,
            applied_at REAL NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            started_at REAL NOT NULL,
            finished_at REAL,
            status TEXT NOT NULL,
            settings TEXT NOT NULL,
            runtime TEXT NOT NULL,
            sandbox TEXT,
            kappa REAL,
            entropy REAL,
            notes TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS episodes (
            run_id TEXT NOT NULL,
            episode INTEGER NOT NULL,
            started_at REAL NOT NULL,
            finished_at REAL,
            scorecard TEXT NOT NULL,
            PRIMARY KEY (run_id, episode),
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS rule_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            episode INTEGER NOT NULL,
            peer TEXT NOT NULL,
            rule_id TEXT NOT NULL,
            magnitude REAL NOT NULL,
            severity TEXT NOT NULL,
            evidence TEXT,
            source TEXT,
            created_at REAL NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            episode INTEGER NOT NULL,
            path TEXT NOT NULL,
            kappa REAL,
            entropy REAL,
            score REAL,
            created_at REAL NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_rule_events_run
        ON rule_events (run_id, episode, peer)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_checkpoints_run
        ON checkpoints (run_id, episode)
        """,
    ]
}


@dataclass
class RunRecord:
    id: str
    started_at: float
    status: str
    settings: dict[str, Any]
    runtime: dict[str, Any]

    def describe(self) -> str:
        return f"run {self.id} [{self.status}] device={self.runtime.get('device')}"


class Registry:
    """Thin sqlite wrapper with JSON helpers and idempotent migrations."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_settings(cls, settings: Any) -> "Registry":
        return cls(settings.database_path())

    # -- connection ------------------------------------------------------
    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    # -- migrations ------------------------------------------------------
    @staticmethod
    def current_version(connection: sqlite3.Connection) -> int:
        try:
            row = connection.execute("SELECT MAX(version) AS version FROM mactf_schema").fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row["version"] or 0) if row else 0

    def migrate(self) -> list[int]:
        """Apply pending migrations and return the versions applied."""
        applied: list[int] = []
        with self.connect() as connection:
            version = self.current_version(connection)
            for target in sorted(MIGRATIONS):
                if target <= version:
                    continue
                for statement in MIGRATIONS[target]:
                    connection.execute(statement)
                connection.execute(
                    "INSERT OR REPLACE INTO mactf_schema (version, applied_at) VALUES (?, ?)",
                    (target, time.time()),
                )
                applied.append(target)
        return applied

    def schema_version(self) -> int:
        with self.connect() as connection:
            return self.current_version(connection)

    # -- runs ------------------------------------------------------------
    def start_run(
        self,
        run_id: str,
        *,
        settings: dict[str, Any],
        runtime: dict[str, Any],
        sandbox: dict[str, Any] | None = None,
        notes: str = "",
    ) -> RunRecord:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (id, started_at, status, settings, runtime, sandbox, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    time.time(),
                    "running",
                    json.dumps(settings, default=str),
                    json.dumps(runtime, default=str),
                    json.dumps(sandbox, default=str) if sandbox else None,
                    notes,
                ),
            )
        return RunRecord(
            id=run_id,
            started_at=time.time(),
            status="running",
            settings=settings,
            runtime=runtime,
        )

    def finish_run(
        self,
        run_id: str,
        *,
        status: str = "completed",
        kappa: float | None = None,
        entropy: float | None = None,
        notes: str | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET finished_at = ?, status = ?, kappa = COALESCE(?, kappa),
                    entropy = COALESCE(?, entropy), notes = COALESCE(?, notes)
                WHERE id = ?
                """,
                (time.time(), status, kappa, entropy, notes, run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def list_runs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    # -- episodes --------------------------------------------------------
    def record_episode(
        self, run_id: str, episode: int, scorecard: dict[str, Any], *, started_at: float | None = None
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO episodes
                (run_id, episode, started_at, finished_at, scorecard)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    episode,
                    started_at or time.time(),
                    time.time(),
                    json.dumps(scorecard, default=str),
                ),
            )

    def episodes(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM episodes WHERE run_id = ? ORDER BY episode", (run_id,)
            ).fetchall()
        payload = []
        for row in rows:
            item = _row_to_dict(row)
            item["scorecard"] = json.loads(item["scorecard"])
            payload.append(item)
        return payload

    def latest_episode(self, run_id: str) -> int | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT MAX(episode) AS episode FROM episodes WHERE run_id = ?", (run_id,)
            ).fetchone()
        return int(row["episode"]) if row and row["episode"] is not None else None

    # -- rule events -----------------------------------------------------
    def record_events(
        self, run_id: str, episode: int, peer: str, events: list[dict[str, Any]]
    ) -> None:
        if not events:
            return
        now = time.time()
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO rule_events
                (run_id, episode, peer, rule_id, magnitude, severity, evidence, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        episode,
                        peer,
                        event["rule_id"],
                        float(event.get("magnitude", 1.0)),
                        event.get("severity", "S2"),
                        event.get("evidence", ""),
                        event.get("source", ""),
                        now,
                    )
                    for event in events
                ],
            )

    def rule_counts(self, run_id: str) -> dict[str, int]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT rule_id, COUNT(*) AS count FROM rule_events
                WHERE run_id = ? GROUP BY rule_id ORDER BY count DESC
                """,
                (run_id,),
            ).fetchall()
        return {row["rule_id"]: int(row["count"]) for row in rows}

    # -- checkpoints -----------------------------------------------------
    def record_checkpoint(
        self,
        run_id: str,
        episode: int,
        path: Path | str,
        *,
        kappa: float | None = None,
        entropy: float | None = None,
        score: float | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO checkpoints (run_id, episode, path, kappa, entropy, score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, episode, str(path), kappa, entropy, score, time.time()),
            )

    def checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY episode", (run_id,)
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def latest_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY episode DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        return _row_to_dict(row) if row else None


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def ensure_registry(settings: Any) -> Registry:
    """Build a registry and guarantee the schema exists."""
    registry = Registry.from_settings(settings)
    version = registry.schema_version()
    if version < SCHEMA_VERSION:
        applied = registry.migrate()
        if not applied:
            raise ImproperlyConfigured(
                f"schema at version {version} needs {SCHEMA_VERSION} but no migration applied"
            )
    return registry
