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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .exceptions import ImproperlyConfigured

SCHEMA_VERSION = 3


@dataclass(frozen=True)
class Migration:
    """One schema version.

    Most versions are pure SQL and use ``statements``. A version that must
    inspect the existing schema first (``ALTER TABLE ... ADD COLUMN`` is not
    idempotent) supplies a ``step`` callable instead, which receives an open
    connection and is responsible for its own idempotency.
    """

    version: int
    statements: tuple[str, ...] = ()
    step: Callable[[sqlite3.Connection], None] | None = None


def _add_columns_if_absent(
    connection: sqlite3.Connection, table: str, columns: dict[str, str]
) -> None:
    """Add each column only when ``PRAGMA table_info`` does not already list it."""
    existing = {
        row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for name, declaration in columns.items():
        if name in existing:
            continue
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")


_MIGRATION_1 = (
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
)


def _migration_2_step(connection: sqlite3.Connection) -> None:
    """Warning, adjudication, aggregation, and reversal persistence (doc 04.4)."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS warnings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            peer TEXT NOT NULL,
            rule_id TEXT NOT NULL,
            episode INTEGER NOT NULL,
            severity TEXT NOT NULL,
            evidence TEXT,
            expiry_episode INTEGER NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            created_at REAL NOT NULL,
            expired_at REAL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_warnings_run_peer_rule
        ON warnings (run_id, peer, rule_id, active)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS adjudications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            peer TEXT NOT NULL,
            rule_id TEXT NOT NULL,
            episode INTEGER NOT NULL,
            decision TEXT NOT NULL,
            severity TEXT,
            rationale TEXT NOT NULL,
            adjudicator TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_adjudications_run
        ON adjudications (run_id, peer, rule_id)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS aggregations (
            run_id TEXT NOT NULL,
            episode INTEGER NOT NULL,
            report TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (run_id, episode)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS resume_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            checkpoint_episode INTEGER,
            started_epoch INTEGER NOT NULL,
            warnings_restored INTEGER NOT NULL DEFAULT 0,
            warnings_expired INTEGER NOT NULL DEFAULT 0,
            report TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_resume_reports_run ON resume_reports (run_id)"
    )
    _add_columns_if_absent(
        connection,
        "rule_events",
        {"reversed_at": "REAL", "reversed_by": "TEXT"},
    )


def _migration_3_step(connection: sqlite3.Connection) -> None:
    """The points layer (doc 11 section 8.3).

    ``scorecards`` is one row per peer per episode so the points breakdown is
    queryable without parsing the episode blob. Existing rows are left with NULL
    points (not computed), never a fabricated zero.
    """
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS scorecards (
            run_id TEXT NOT NULL,
            episode INTEGER NOT NULL,
            peer TEXT NOT NULL,
            final_score REAL NOT NULL,
            points TEXT,
            points_table_version TEXT,
            created_at REAL NOT NULL,
            PRIMARY KEY (run_id, episode, peer)
        )
        """
    )
    # A database that already carries the table (a hand-rolled earlier build)
    # still needs the new columns; ``_add_columns_if_absent`` is idempotent.
    _add_columns_if_absent(
        connection, "scorecards", {"points": "TEXT", "points_table_version": "TEXT"}
    )
    _add_columns_if_absent(connection, "episodes", {"normalization": "TEXT"})
    _add_columns_if_absent(
        connection, "aggregations", {"signal_source": "TEXT NOT NULL DEFAULT 'final_score'"}
    )


MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, statements=_MIGRATION_1),
    Migration(2, step=_migration_2_step),
    Migration(3, step=_migration_3_step),
)

#: Version -> migration, for callers that want to look one up.
MIGRATIONS_BY_VERSION: dict[int, Migration] = {
    migration.version: migration for migration in MIGRATIONS
}


CORE_LABEL = "core"

#: ``runs.status`` values that end a run (doc 01 section 4).
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "dry_run", "interrupted"})
#: Rows ``start_run`` may reuse rather than insert a duplicate for.
RESTARTABLE_STATUSES = frozenset({"queued", "paused", "resuming", "interrupted"})

_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS mactf_migrations (
    app        TEXT NOT NULL,
    version    INTEGER NOT NULL,
    applied_at REAL NOT NULL,
    PRIMARY KEY (app, version)
)
"""


class BigAutoField:
    """The default primary-key column type named by ``DEFAULT_AUTO_FIELD``."""

    sql_type = "INTEGER PRIMARY KEY AUTOINCREMENT"


class MigrationFailed(ImproperlyConfigured):
    """An app's migrations were rolled back; ``app`` says which."""

    def __init__(self, app: str, cause: BaseException) -> None:
        super().__init__(f"migrations for app {app!r} failed and were rolled back: {cause}")
        self.app = app
        self.cause = cause


def migration_plan(
    app_migrations: Mapping[str, Mapping[int, list[str]]] | None = None,
) -> list[tuple[str, int, list[str]]]:
    """``(app_label, version, statements)`` in apply order: core, then apps by label.

    ``app_migrations`` defaults to what the installed apps declare through
    ``AppConfig.get_migrations()``. A core migration that is a Python ``step`` has
    no statements and appears with an empty list; it is still a version.
    """
    plan: list[tuple[str, int, list[str]]] = [
        (CORE_LABEL, migration.version, list(migration.statements))
        for migration in sorted(MIGRATIONS, key=lambda item: item.version)
    ]
    if app_migrations is None:
        app_migrations = _declared_app_migrations()
    for label in sorted(app_migrations):
        if label == CORE_LABEL:
            raise ImproperlyConfigured("an app may not use the reserved label 'core'")
        for version in sorted(app_migrations[label]):
            if isinstance(version, bool) or not isinstance(version, int) or version < 1:
                raise ImproperlyConfigured(
                    f"app {label!r} migration versions are integers from 1, got {version!r}"
                )
            plan.append((label, version, list(app_migrations[label][version])))
    return plan


def _declared_app_migrations() -> dict[str, dict[int, list[str]]]:
    from . import apps as apps_module

    if not apps_module.apps._ready:
        return {}
    declared: dict[str, dict[int, list[str]]] = {}
    for config in apps_module.apps.get_app_configs():
        migrations = config.get_migrations()
        if migrations:
            declared[config.label] = migrations
    return declared


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

    def __init__(
        self, path: Path | str, *, read_only: bool = False, busy_timeout: float | None = None
    ) -> None:
        self.path = Path(path)
        self.read_only = bool(read_only)
        self.busy_timeout = 30.0 if busy_timeout is None else float(busy_timeout)
        if not self.read_only and self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)       # never for a reader

    @classmethod
    def from_settings(cls, settings: Any, **kwargs: Any) -> "Registry":
        return cls(settings.database_path(), **kwargs)

    # -- connection ------------------------------------------------------
    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            # ``mode=ro`` refuses to create the file and refuses every write; a missing
            # file is sqlite3.OperationalError, which callers treat as "no registry".
            uri = f"{self.path.resolve().as_uri()}?mode=ro"
            connection = sqlite3.connect(uri, uri=True, timeout=self.busy_timeout)
        else:
            connection = sqlite3.connect(self.path, timeout=self.busy_timeout)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            if not self.read_only:
                connection.execute("PRAGMA journal_mode = WAL")       # converting is a write
            yield connection
            if not self.read_only:
                connection.commit()
        except Exception:
            if not self.read_only:
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
            for migration in sorted(MIGRATIONS, key=lambda item: item.version):
                if migration.version <= version:
                    continue
                for statement in migration.statements:
                    connection.execute(statement)
                if migration.step is not None:
                    migration.step(connection)
                connection.execute(
                    "INSERT OR REPLACE INTO mactf_schema (version, applied_at) VALUES (?, ?)",
                    (migration.version, time.time()),
                )
                applied.append(migration.version)
            self._ensure_bookkeeping(connection)
        return applied

    # -- app-scoped migrations (doc 13 section 2.2) --------------------------------
    @staticmethod
    def _ensure_bookkeeping(connection: sqlite3.Connection) -> None:
        """Create ``mactf_migrations`` and seed ``core`` from ``mactf_schema``."""
        connection.execute(_MIGRATIONS_TABLE)
        current = Registry.current_version(connection)
        if current:
            connection.executemany(
                "INSERT OR IGNORE INTO mactf_migrations (app, version, applied_at) VALUES (?, ?, ?)",
                [(CORE_LABEL, version, time.time()) for version in range(1, current + 1)],
            )

    def applied_migrations(self) -> set[tuple[str, int]]:
        """Every ``(app, version)`` recorded, ``core`` included."""
        with self.connect() as connection:
            if self.read_only:
                # A reader must not create the bookkeeping table; an older file simply
                # lacks it, and its core versions are the ones ``mactf_schema`` lists.
                applied: set[tuple[str, int]] = set()
                try:
                    rows = connection.execute("SELECT app, version FROM mactf_migrations").fetchall()
                    applied |= {(row["app"], int(row["version"])) for row in rows}
                except sqlite3.OperationalError:
                    pass
                applied |= {(CORE_LABEL, version) for version in range(1, self.current_version(connection) + 1)}
                return applied
            self._ensure_bookkeeping(connection)
            rows = connection.execute("SELECT app, version FROM mactf_migrations").fetchall()
        return {(row["app"], int(row["version"])) for row in rows}

    def pending_plan(
        self, plan: list[tuple[str, int, list[str]]] | None = None
    ) -> list[tuple[str, int, list[str]]]:
        """The entries of ``plan`` not yet recorded, in apply order."""
        plan = migration_plan() if plan is None else plan
        applied = self.applied_migrations()
        return [entry for entry in plan if (entry[0], entry[1]) not in applied]

    def migrate_apps(
        self, plan: list[tuple[str, int, list[str]]] | None = None
    ) -> list[tuple[str, int]]:
        """Apply pending *app* migrations; core is handled by :meth:`migrate`.

        Each app's pending statements run in one transaction. A failure rolls that
        app back and raises :class:`MigrationFailed` naming it; apps applied before
        it stay applied.
        """
        done: list[tuple[str, int]] = []
        pending = [entry for entry in self.pending_plan(plan) if entry[0] != CORE_LABEL]
        by_app: dict[str, list[tuple[int, list[str]]]] = {}
        for app, version, statements in pending:
            by_app.setdefault(app, []).append((version, statements))
        for app in sorted(by_app):
            with self.connect() as connection:
                connection.execute("BEGIN")
                try:
                    for version, statements in sorted(by_app[app]):
                        for statement in statements:
                            connection.execute(statement)
                        connection.execute(
                            "INSERT INTO mactf_migrations (app, version, applied_at) VALUES (?, ?, ?)",
                            (app, version, time.time()),
                        )
                        done.append((app, version))
                except Exception as exc:
                    connection.rollback()
                    done[:] = [item for item in done if item[0] != app]
                    raise MigrationFailed(app, exc) from exc
        return done

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
            existing = connection.execute(
                "SELECT status FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            values = (
                time.time(),
                "running",
                json.dumps(settings, default=str),
                json.dumps(runtime, default=str),
                json.dumps(sandbox, default=str) if sandbox else None,
                notes,
            )
            if existing is not None and existing["status"] in RESTARTABLE_STATUSES:
                # A row the run manager queued, or a paused/interrupted run being
                # resumed under the same id: reuse it instead of colliding on the key.
                connection.execute(
                    """
                    UPDATE runs SET started_at = ?, status = ?, settings = ?, runtime = ?,
                        sandbox = ?, notes = ?, finished_at = NULL WHERE id = ?
                    """,
                    (*values, run_id),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO runs (id, started_at, status, settings, runtime, sandbox, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (run_id, *values),
                )
        return RunRecord(
            id=run_id,
            started_at=time.time(),
            status="running",
            settings=settings,
            runtime=runtime,
        )

    def queue_run(
        self, run_id: str, *, settings: dict[str, Any], notes: str = ""
    ) -> None:
        """Insert a ``queued`` row so a run is visible before its thread starts."""
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (id, started_at, status, settings, runtime, notes)
                VALUES (?, ?, 'queued', ?, '{}', ?)
                """,
                (run_id, time.time(), json.dumps(settings, default=str), notes),
            )

    def set_run_status(
        self,
        run_id: str,
        status: str,
        *,
        notes: str | None = None,
        finished: bool | None = None,
    ) -> None:
        """Write a status the run manager observed.

        ``finished`` defaults to "terminal statuses get a ``finished_at``"; a
        ``paused`` or ``resuming`` row keeps ``finished_at`` NULL.
        """
        if finished is None:
            finished = status in TERMINAL_STATUSES
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE runs SET status = ?, notes = COALESCE(?, notes),
                    finished_at = CASE WHEN ? THEN COALESCE(finished_at, ?) ELSE NULL END
                WHERE id = ?
                """,
                (status, notes, 1 if finished else 0, time.time(), run_id),
            )

    def runs_with_status(self, statuses: Iterable[str]) -> list[dict[str, Any]]:
        wanted = list(statuses)
        if not wanted:
            return []
        marks = ",".join("?" for _ in wanted)
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM runs WHERE status IN ({marks}) ORDER BY started_at", wanted
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

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

    def list_runs(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        status: Iterable[str] | None = None,
        since: float | None = None,
        until: float | None = None,
        id_prefix: str | None = None,
        order: str = "desc",
    ) -> list[dict[str, Any]]:
        """Runs newest first (or oldest first with ``order="asc"``), optionally filtered.

        ``status`` is ``IN (...)``; ``since``/``until`` bound ``started_at`` (POSIX seconds);
        ``id_prefix`` is a ``LIKE 'prefix%'`` with wildcards escaped.
        """
        if order not in ("desc", "asc"):
            raise ValueError(f"order must be 'desc' or 'asc', not {order!r}")
        where, params = _run_filter(status, since, until, id_prefix)
        direction = "DESC" if order == "desc" else "ASC"
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM runs{where} ORDER BY started_at {direction}, id {direction} "
                "LIMIT ? OFFSET ?",
                (*params, int(limit), max(0, int(offset))),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def count_runs(
        self,
        *,
        status: Iterable[str] | None = None,
        since: float | None = None,
        until: float | None = None,
        id_prefix: str | None = None,
    ) -> int:
        where, params = _run_filter(status, since, until, id_prefix)
        with self.connect() as connection:
            return int(
                connection.execute(f"SELECT COUNT(*) AS n FROM runs{where}", params).fetchone()["n"]
            )

    def episode_counts(self, run_ids: Sequence[str]) -> dict[str, int]:
        """Episodes per run in one ``GROUP BY`` per 500 ids (SQLite caps bound variables)."""
        counts = {run_id: 0 for run_id in run_ids}
        ids = list(dict.fromkeys(run_ids))
        with self.connect() as connection:
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                rows = connection.execute(
                    "SELECT run_id, COUNT(*) AS n FROM episodes WHERE run_id IN "
                    f"({','.join('?' for _ in chunk)}) GROUP BY run_id",
                    chunk,
                ).fetchall()
                for row in rows:
                    counts[row["run_id"]] = int(row["n"])
        return counts

    def delete_run(self, run_id: str) -> dict[str, int]:
        """Delete a run and everything that references it; rows removed per table.

        Tables are discovered from ``sqlite_master`` (apps add tables through
        ``migration_plan``), never listed, so a new app table cannot leave orphans.
        Table names come from the database itself; ``run_id`` is bound.
        """
        removed: dict[str, int] = {}
        with self.connect() as connection:                  # one transaction
            tables = [
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                )
            ]
            for table in tables:
                if table == "runs":
                    continue
                columns = {info[1] for info in connection.execute(f'PRAGMA table_info("{table}")')}
                if "run_id" not in columns:
                    continue
                count = connection.execute(
                    f'DELETE FROM "{table}" WHERE run_id = ?', (run_id,)
                ).rowcount
                if count:
                    removed[table] = int(count)
            count = connection.execute("DELETE FROM runs WHERE id = ?", (run_id,)).rowcount
            if count:
                removed["runs"] = int(count)
        return removed

    def delete_checkpoint_rows(self, row_ids: Sequence[int]) -> int:
        """Forget checkpoint rows by primary key (a ``-best`` and a regular row can share an episode)."""
        ids = [int(item) for item in row_ids]
        if not ids:
            return 0
        removed = 0
        with self.connect() as connection:
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                cursor = connection.execute(
                    f"DELETE FROM checkpoints WHERE id IN ({','.join('?' for _ in chunk)})", chunk
                )
                removed += int(cursor.rowcount or 0)
        return removed

    def count_episodes(self, run_id: str) -> int:
        with self.connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM episodes WHERE run_id = ?", (run_id,)
                ).fetchone()["n"]
            )

    def rule_events(
        self,
        run_id: str,
        *,
        peer: str | None = None,
        rule_id: str | None = None,
        episode: int | None = None,
        severity: str | None = None,
        source: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Filtered ``rule_events`` rows plus the unpaged total.

        This is the audit surface for framework.md section 12: individual events are
        returned as stored, never aggregated.
        """
        clauses, params = ["run_id = ?"], [run_id]
        for column, value in (
            ("peer", peer), ("rule_id", rule_id), ("episode", episode),
            ("severity", severity), ("source", source),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = " AND ".join(clauses)
        with self.connect() as connection:
            total = int(
                connection.execute(f"SELECT COUNT(*) AS n FROM rule_events WHERE {where}", params)
                .fetchone()["n"]
            )
            rows = connection.execute(
                f"SELECT * FROM rule_events WHERE {where} ORDER BY episode, id LIMIT ? OFFSET ?",
                [*params, int(limit), max(0, int(offset))],
            ).fetchall()
        return [_row_to_dict(row) for row in rows], total

    # -- episodes --------------------------------------------------------
    def record_episode(
        self,
        run_id: str,
        episode: int,
        scorecard: dict[str, Any],
        *,
        started_at: float | None = None,
        normalization: dict[str, Any] | None = None,
    ) -> None:
        now = time.time()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO episodes
                (run_id, episode, started_at, finished_at, scorecard, normalization)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    episode,
                    started_at or now,
                    now,
                    json.dumps(scorecard, default=str),
                    json.dumps(normalization, default=str) if normalization is not None else None,
                ),
            )
            peers = scorecard.get("peers") if isinstance(scorecard, dict) else None
            for peer, card in (peers or {}).items():
                if not isinstance(card, dict):
                    continue
                points = card.get("points")
                connection.execute(
                    """
                    INSERT OR REPLACE INTO scorecards
                    (run_id, episode, peer, final_score, points, points_table_version, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        episode,
                        str(peer),
                        float(card.get("final_score", 0.0)),
                        json.dumps(points, default=str) if points is not None else None,
                        (points or {}).get("table_version") if isinstance(points, dict) else None,
                        now,
                    ),
                )

    def scorecards(self, run_id: str, *, episode: int | None = None) -> list[dict[str, Any]]:
        """Per-peer scorecard rows, with ``points`` decoded (``None`` = not computed)."""
        query = "SELECT * FROM scorecards WHERE run_id = ?"
        params: list[Any] = [run_id]
        if episode is not None:
            query += " AND episode = ?"
            params.append(int(episode))
        with self.connect() as connection:
            rows = connection.execute(query + " ORDER BY episode, peer", params).fetchall()
        payload = []
        for row in rows:
            item = _row_to_dict(row)
            item["points"] = json.loads(item["points"]) if item.get("points") else None
            payload.append(item)
        return payload

    def episodes(
        self, run_id: str, *, limit: int | None = None, offset: int = 0
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM episodes WHERE run_id = ? ORDER BY episode"
        params: list[Any] = [run_id]
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params += [int(limit), max(0, int(offset))]
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        payload = []
        for row in rows:
            item = _row_to_dict(row)
            item["scorecard"] = json.loads(item["scorecard"])
            if item.get("normalization"):
                item["normalization"] = json.loads(item["normalization"])
            payload.append(item)
        return payload

    def get_episode(self, run_id: str, episode: int) -> dict[str, Any] | None:
        """One decoded episode row, or ``None``."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM episodes WHERE run_id = ? AND episode = ?", (run_id, int(episode))
            ).fetchone()
        if row is None:
            return None
        item = _row_to_dict(row)
        item["scorecard"] = json.loads(item["scorecard"])
        if item.get("normalization"):
            item["normalization"] = json.loads(item["normalization"])
        return item

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

    def delete_checkpoints_after(self, run_id: str, episode: int) -> int:
        """Forget checkpoint rows newer than ``episode`` (the files are the caller's to archive)."""
        with self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM checkpoints WHERE run_id = ? AND episode > ?", (run_id, int(episode))
            )
            return int(cursor.rowcount or 0)

    def latest_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY episode DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        return _row_to_dict(row) if row else None

    # -- warnings --------------------------------------------------------
    def record_warning(self, run_id: str, peer: str, record: Any) -> None:
        """Persist one warning record.

        Idempotent per ``(run_id, peer, rule_id, episode)`` so an epoch that is
        rescored does not double-count its warnings.
        """
        with self.connect() as connection:
            existing = connection.execute(
                """
                SELECT id FROM warnings
                WHERE run_id = ? AND peer = ? AND rule_id = ? AND episode = ?
                """,
                (run_id, peer, record.rule_id, int(record.episode)),
            ).fetchone()
            if existing is not None:
                connection.execute(
                    """
                    UPDATE warnings
                    SET severity = ?, evidence = ?, expiry_episode = ?, active = ?
                    WHERE id = ?
                    """,
                    (
                        _severity_label(record.severity),
                        record.evidence,
                        int(record.expiry_episode),
                        int(bool(record.active)),
                        existing["id"],
                    ),
                )
                return
            connection.execute(
                """
                INSERT INTO warnings
                (run_id, peer, rule_id, episode, severity, evidence, expiry_episode,
                 active, created_at, expired_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    peer,
                    record.rule_id,
                    int(record.episode),
                    _severity_label(record.severity),
                    record.evidence,
                    int(record.expiry_episode),
                    int(bool(record.active)),
                    time.time(),
                    None if record.active else time.time(),
                ),
            )

    def expire_warning(
        self,
        run_id: str,
        peer: str,
        rule_id: str,
        episode: int,
        *,
        expired_at: float,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE warnings
                SET active = 0, expired_at = ?
                WHERE run_id = ? AND peer = ? AND rule_id = ? AND episode <= ? AND active = 1
                """,
                (expired_at, run_id, peer, rule_id, episode),
            )

    def active_warnings(self, run_id: str, *, peer: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM warnings WHERE run_id = ? AND active = 1"
        parameters: list[Any] = [run_id]
        if peer is not None:
            query += " AND peer = ?"
            parameters.append(peer)
        query += " ORDER BY peer, rule_id, episode"
        with self.connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [_row_to_dict(row) for row in rows]

    def all_warnings(self, run_id: str, *, peer: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM warnings WHERE run_id = ?"
        parameters: list[Any] = [run_id]
        if peer is not None:
            query += " AND peer = ?"
            parameters.append(peer)
        query += " ORDER BY peer, rule_id, episode"
        with self.connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [_row_to_dict(row) for row in rows]

    def restore_warnings(
        self,
        run_id: str,
        registry: Any,
        *,
        peer: str | None = None,
        episode: int | None = None,
    ) -> int:
        """Repopulate a ``WarningRegistry`` from persisted warnings.

        ``_clean_streak`` is *derived* rather than stored, because a stored
        counter drifts the moment a warning expires by decay instead of by a new
        violation: the streak is ``episode - last active warning episode - 1``,
        floored at zero, evaluated at the run's latest episode. Storing it would
        be a second source of truth for the same fact.

        ``episode`` defaults to the run's latest recorded episode, falling back
        to the newest warning's episode when the run has no episodes yet.
        """
        from .rules.severity import Severity
        from .scoring.warning import WarningRecord

        rows = self.all_warnings(run_id, peer=peer)
        if not rows:
            return 0

        reference = episode
        if reference is None:
            reference = self.latest_episode(run_id)
        if reference is None:
            reference = max(int(row["episode"]) for row in rows)

        restored = 0
        by_peer: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for row in rows:
            by_peer.setdefault(str(row["peer"]), {}).setdefault(str(row["rule_id"]), []).append(row)

        for peer_name, rules in by_peer.items():
            _seed_peer_dict(registry, peer_name)
            for rule_id, entries in rules.items():
                entries.sort(key=lambda item: int(item["episode"]))
                active_entries = [entry for entry in entries if int(entry["active"]) == 1]
                for entry in entries:
                    record = WarningRecord(
                        rule_id=rule_id,
                        episode=int(entry["episode"]),
                        severity=Severity.parse(entry["severity"]),
                        evidence=str(entry["evidence"] or ""),
                        expiry_episode=int(entry["expiry_episode"]),
                        active=bool(entry["active"]),
                    )
                    _seed_warning(registry, peer_name, rule_id, record)
                    restored += 1
                last_active = max(
                    (int(entry["episode"]) for entry in active_entries), default=None
                )
                streak = (
                    0 if last_active is None else max(0, int(reference) - last_active - 1)
                )
                _seed_streak(registry, peer_name, rule_id, streak)
        return restored

    # -- adjudications ---------------------------------------------------
    def record_adjudication(self, run_id: str | None, **fields: Any) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO adjudications
                (run_id, peer, rule_id, episode, decision, severity, rationale,
                 adjudicator, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    fields["peer"],
                    fields["rule_id"],
                    int(fields["episode"]),
                    fields["decision"],
                    fields.get("severity"),
                    fields["rationale"],
                    fields["adjudicator"],
                    fields.get("created_at") or time.time(),
                ),
            )
            return int(cursor.lastrowid or 0)

    def adjudications(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM adjudications WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def mark_events_reversed(
        self, run_id: str, peer: str, rule_id: str, episode: int, *, by: str
    ) -> int:
        """Mark the matching rule events as reversed, returning the count.

        Reversal is recorded rather than deleted: an adjudicator overturning a
        finding is part of the audit trail (framework.md W5), and deleting the
        original event would hide that the finding was ever made.
        """
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE rule_events
                SET reversed_at = ?, reversed_by = ?
                WHERE run_id = ? AND peer = ? AND rule_id = ? AND episode = ?
                  AND reversed_at IS NULL
                """,
                (time.time(), by, run_id, peer, rule_id, episode),
            )
            return int(cursor.rowcount or 0)

    # -- aggregations ----------------------------------------------------
    def record_aggregation(self, run_id: str, episode: int, report: Any) -> None:
        payload = report.as_dict() if hasattr(report, "as_dict") else dict(report)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO aggregations
                (run_id, episode, report, created_at, signal_source)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    int(episode),
                    json.dumps(payload, default=str),
                    time.time(),
                    str(payload.get("signal_source") or "final_score"),
                ),
            )

    def aggregations(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM aggregations WHERE run_id = ? ORDER BY episode", (run_id,)
            ).fetchall()
        payload = []
        for row in rows:
            item = _row_to_dict(row)
            item["report"] = json.loads(item["report"])
            payload.append(item)
        return payload

    def latest_aggregation(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM aggregations WHERE run_id = ? ORDER BY episode DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        item = _row_to_dict(row)
        item["report"] = json.loads(item["report"])
        return item

    # -- resume reports --------------------------------------------------
    def record_resume_report(
        self,
        run_id: str,
        *,
        checkpoint_episode: int | None,
        started_epoch: int,
        warning_count: int = 0,
        expired_count: int = 0,
        report: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO resume_reports
                (run_id, checkpoint_episode, started_epoch, warnings_restored,
                 warnings_expired, report, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    checkpoint_episode,
                    int(started_epoch),
                    int(warning_count),
                    int(expired_count),
                    json.dumps(report or {}, default=str),
                    time.time(),
                ),
            )

    def resume_reports(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM resume_reports WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
        payload = []
        for row in rows:
            item = _row_to_dict(row)
            item["report"] = json.loads(item["report"])
            payload.append(item)
        return payload


def _severity_label(value: Any) -> str:
    return getattr(value, "label", str(value))


def _seed_peer_dict(registry: Any, peer: str) -> None:
    import logging

    logger = logging.getLogger("mactf.db")
    if not hasattr(registry, "_warnings") or not hasattr(registry, "_clean_streak"):
        logger.warning("restore_warnings received a registry without warning state")
        return
    registry._warnings.setdefault(peer, {})
    registry._clean_streak.setdefault(peer, {})


def _seed_warning(registry: Any, peer: str, rule_id: str, record: Any) -> None:
    registry._warnings[peer].setdefault(rule_id, []).append(record)


def _seed_streak(registry: Any, peer: str, rule_id: str, streak: int) -> None:
    registry._clean_streak[peer][rule_id] = int(streak)


def _run_filter(
    status: Iterable[str] | None,
    since: float | None,
    until: float | None,
    id_prefix: str | None,
) -> tuple[str, list[Any]]:
    """The shared ``WHERE`` for ``list_runs`` and ``count_runs``.

    An explicitly empty ``status`` means "no filter", not "match nothing".
    """
    clauses: list[str] = []
    params: list[Any] = []
    wanted = list(status) if status is not None else []
    if wanted:
        clauses.append(f"status IN ({','.join('?' for _ in wanted)})")
        params.extend(wanted)
    if since is not None:
        clauses.append("started_at >= ?")
        params.append(float(since))
    if until is not None:
        clauses.append("started_at <= ?")
        params.append(float(until))
    if id_prefix:
        escaped = id_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        clauses.append("id LIKE ? ESCAPE '\\'")
        params.append(escaped + "%")
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


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
    if _declared_app_migrations():
        registry.migrate_apps()
    return registry
