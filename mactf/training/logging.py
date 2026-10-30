"""Typed logging with four sinks (document 09 section 2).

The design rule is that a record is a dataclass, never a formatted string. A sink
renders; it does not decide what a field means. That is what lets ``ScoreRecord``
carry exactly ``PeerScorecard.to_dict()``, so a score can be recovered from a log
even though log bytes themselves are not part of the determinism guarantee.

The logger is a signal *receiver* only. It never sends, which is what makes
"logging cannot change training" true by construction.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence

from ..conf_files import is_authored
from ..exceptions import ImproperlyConfigured

__all__ = [
    "LogLocation",
    "complete_size",
    "follow_jsonl",
    "open_run_logger",
    "run_log_location",
    "run_log_path",
    "tail_jsonl",
    "AggregationRecord",
    "CheckpointRecord",
    "ConsoleSink",
    "DEFAULT_REDACT",
    "EpochRecord",
    "FileSink",
    "HealthRecord",
    "JsonlSink",
    "LogRecord",
    "LogSink",
    "LossRecord",
    "MACTFLogger",
    "MemorySink",
    "ResumeRecord",
    "ScoreRecord",
    "StepRecord",
    "TensorBoardSink",
    "WandBSink",
    "WarningRecord",
]

#: Field-name fragments that are redacted before any sink sees a value.
#:
#: Matched case-insensitively as substrings, so ``hf_token`` and
#: ``Authorization`` are both caught by their stems.
DEFAULT_REDACT: tuple[str, ...] = (
    "token",
    "secret",
    "password",
    "api_key",
    "authorization",
    "hf_token",
    "wandb_key",
)

_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


@dataclass(frozen=True)
class LogRecord:
    """The base record. Every field is data, not prose."""

    kind: str = "log"
    run_id: str = ""
    node_path: str = ""
    depth: int = 0
    epoch: int = 0
    step: int = 0
    wall_time: float = 0.0
    monotonic: float = 0.0
    duration: float = 0.0
    level: str = "info"
    message: str = ""
    fields: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    # -- rendering -------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        payload = {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "node_path": self.node_path,
            "depth": self.depth,
            "epoch": self.epoch,
            "step": self.step,
            "wall_time": self.wall_time,
            "monotonic": self.monotonic,
            "duration": self.duration,
            "level": self.level,
            "message": self.message,
        }
        if self.fields:
            payload["fields"] = _jsonable(self.fields)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, default=str)

    def to_text(self) -> str:
        parts = [f"{self.level.upper():8s}", f"{self.kind:11s}"]
        if self.run_id:
            parts.append(self.run_id)
        if self.node_path:
            parts.append(self.node_path)
        if self.epoch:
            parts.append(f"ep={self.epoch}")
        if self.step:
            parts.append(f"step={self.step}")
        if self.duration:
            parts.append(f"dur={self.duration:.3f}s")
        text = " ".join(parts)
        if self.message:
            text += f" {self.message}"
        if self.fields:
            pairs = ", ".join(f"{key}={value}" for key, value in self.fields.items())
            text += f" [{pairs}]"
        return text

    def matches(self, *, node_path: str | None = None, kind: str | None = None) -> bool:
        if node_path is not None and self.node_path != node_path:
            return False
        if kind is not None and self.kind != kind:
            return False
        return True


@dataclass(frozen=True)
class EpochRecord(LogRecord):
    kind: str = "epoch"


@dataclass(frozen=True)
class StepRecord(LogRecord):
    kind: str = "step"


@dataclass(frozen=True)
class LossRecord(LogRecord):
    kind: str = "loss"


@dataclass(frozen=True)
class ScoreRecord(LogRecord):
    kind: str = "score"


@dataclass(frozen=True)
class AggregationRecord(LogRecord):
    kind: str = "aggregation"


@dataclass(frozen=True)
class WarningRecord(LogRecord):
    kind: str = "warning"


@dataclass(frozen=True)
class CheckpointRecord(LogRecord):
    kind: str = "checkpoint"


@dataclass(frozen=True)
class HealthRecord(LogRecord):
    kind: str = "health"


@dataclass(frozen=True)
class ResumeRecord(LogRecord):
    kind: str = "resume"


def _jsonable(value: Any) -> Any:
    """Make a value JSON-safe without inventing fields.

    Tensors become ``{"shape", "dtype", "values"}`` for small payloads and a shape
    summary otherwise, which is what a log reader actually needs.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        torch = None
    if torch is not None and torch.is_tensor(value):
        payload: dict[str, Any] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
        }
        if value.numel() <= 64:
            payload["values"] = value.detach().cpu().tolist()
        return payload
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------
class LogSink(Protocol):
    name: str

    def write(self, record: LogRecord) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...


class _CountingSink:
    """Shared record counting, so every sink reports the same shape of state."""

    name = "sink"

    def __init__(self) -> None:
        self.count = 0
        self.closed = False

    def _count(self, record: LogRecord) -> None:
        self.count += 1

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.flush()
        self.closed = True

    def state_dict(self) -> dict[str, Any]:
        return {"name": self.name, "records": self.count, "closed": self.closed}


class _TextFormatter:
    """Plain or JSON, matching ``mactf.telemetry``'s two formatters."""

    def __init__(self, *, json_format: bool) -> None:
        self.json_format = bool(json_format)

    def format(self, record: LogRecord) -> str:
        return record.to_json() if self.json_format else record.to_text()


class ConsoleSink(_CountingSink):
    """Writes to stderr, never stdout: stdout belongs to CLI ``--json`` output."""

    name = "console"

    def __init__(self, *, json_format: bool = False, stream: Any = None) -> None:
        super().__init__()
        self.formatter = _TextFormatter(json_format=json_format)
        self.stream = stream if stream is not None else sys.stderr

    def write(self, record: LogRecord) -> None:
        self._count(record)
        self.stream.write(self.formatter.format(record) + "\n")

    def flush(self) -> None:
        try:
            self.stream.flush()
        except (ValueError, OSError):  # pragma: no cover - closed stream at exit
            pass


class FileSink(_CountingSink):
    """Appends to a text file with size-based rotation."""

    name = "file"

    def __init__(
        self,
        path: str | Path,
        *,
        json_format: bool = False,
        max_bytes: int = 10_485_760,
        backup_count: int = 5,
        encoding: str = "utf-8",
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.formatter = _TextFormatter(json_format=json_format)
        self.max_bytes = max(0, int(max_bytes))
        self.backup_count = max(0, int(backup_count))
        self.encoding = encoding
        self._handle: Any = None
        self._size = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._size = self.path.stat().st_size
        self._handle = self.path.open("a", encoding=self.encoding, newline="\n")

    def write(self, record: LogRecord) -> None:
        self._count(record)
        line = self.formatter.format(record) + "\n"
        encoded = len(line.encode(self.encoding))
        if self.max_bytes and self._size + encoded > self.max_bytes:
            self._rotate()
        self._handle.write(line)
        self._size += encoded

    def _rotate(self) -> None:
        self._handle.close()
        if self.backup_count:
            oldest = self.path.with_suffix(self.path.suffix + f".{self.backup_count}")
            if oldest.exists():
                oldest.unlink()
            for index in range(self.backup_count - 1, 0, -1):
                source = self.path.with_suffix(self.path.suffix + f".{index}")
                if source.exists():
                    source.rename(self.path.with_suffix(self.path.suffix + f".{index + 1}"))
            if self.path.exists():
                self.path.rename(self.path.with_suffix(self.path.suffix + ".1"))
        else:
            self.path.unlink(missing_ok=True)
        self._handle = self.path.open("a", encoding=self.encoding, newline="\n")
        self._size = 0

    def flush(self) -> None:
        if self._handle is not None and not self._handle.closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())

    def close(self) -> None:
        if self._handle is not None and not self._handle.closed:
            self.flush()
            self._handle.close()
        self.closed = True

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload.update({"path": str(self.path), "bytes": self._size})
        return payload


class JsonlSink(_CountingSink):
    """One JSON object per line, fsynced on flush.

    This is the format ``GET /runs/{id}/logs/tail`` reads, so it has no rotation:
    a shipper needs a stable file, and truncating it mid-run loses records the API
    promised.
    """

    name = "jsonl"

    def __init__(self, path: str | Path, *, encoding: str = "utf-8") -> None:
        super().__init__()
        self.path = Path(path)
        self.encoding = encoding
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding=self.encoding, newline="\n")
        # Seek to the logical end so an offset-based resume appends rather than
        # overwriting a partially written last line.
        self._handle.seek(0, os.SEEK_END)
        self.offset = self._handle.tell()

    def write(self, record: LogRecord) -> None:
        self._count(record)
        line = record.to_json() + "\n"
        self._handle.write(line)
        self.offset += len(line.encode(self.encoding))

    def flush(self) -> None:
        if self._handle is not None and not self._handle.closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())

    def close(self) -> None:
        if self._handle is not None and not self._handle.closed:
            self.flush()
            self._handle.close()
        self.closed = True

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload.update({"path": str(self.path), "offset": self.offset})
        return payload


class MemorySink(_CountingSink):
    """A bounded in-memory ring, backing :meth:`MACTFLogger.tail`."""

    name = "memory"

    def __init__(self, buffer_size: int = 1000) -> None:
        super().__init__()
        self.records: deque[LogRecord] = deque(maxlen=max(1, int(buffer_size)))

    def write(self, record: LogRecord) -> None:
        self._count(record)
        self.records.append(record)

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload["buffered"] = len(self.records)
        return payload


class TensorBoardSink(_CountingSink):
    """Scalars to a ``SummaryWriter``; requires ``mactf[tensorboard]``."""

    name = "tensorboard"

    def __init__(self, log_dir: str | Path, *, writer: Any = None) -> None:
        super().__init__()
        if writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415
            except ImportError as exc:
                raise ImproperlyConfigured(
                    "TensorBoardSink requires tensorboard; install with "
                    '`pip install "mactf[tensorboard]"`'
                ) from exc
            writer = SummaryWriter(log_dir=str(log_dir))
        self.writer = writer

    def write(self, record: LogRecord) -> None:
        self._count(record)
        step = int(record.step or record.epoch)
        tag_prefix = f"{record.run_id}/" if record.run_id else ""
        if record.duration:
            self.writer.add_scalar(f"{tag_prefix}duration", record.duration, step)
        for key, value in record.fields.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.writer.add_scalar(f"{tag_prefix}{key}", float(value), step)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()
        self.closed = True


class WandBSink(_CountingSink):
    """Scalars to Weights & Biases; requires ``mactf[wandb]``."""

    name = "wandb"

    def __init__(self, *, run: Any = None, project: str | None = None, config: Any = None) -> None:
        super().__init__()
        if run is None:
            try:
                import wandb  # noqa: PLC0415 - optional dependency
            except ImportError as exc:
                raise ImproperlyConfigured(
                    "WandBSink requires wandb; install with "
                    '`pip install "mactf[wandb]"`'
                ) from exc
            run = wandb.init(project=project, config=dict(config or {}))
        self.run = run

    def write(self, record: LogRecord) -> None:
        self._count(record)
        payload: dict[str, Any] = {"step": int(record.step or record.epoch)}
        if record.duration:
            payload["duration"] = record.duration
        for key, value in record.fields.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                payload[str(key)] = float(value)
        if payload:
            self.run.log(payload)

    def close(self) -> None:
        finish = getattr(self.run, "finish", None)
        if callable(finish):
            finish()
        self.closed = True


# ---------------------------------------------------------------------------
# The logger
# ---------------------------------------------------------------------------
class MACTFLogger:
    """Typed records, multiple sinks, optional async writer."""

    def __init__(
        self,
        settings: Any | None = None,
        *,
        run_id: str | None = None,
        level: str | None = None,
        sinks: Sequence[LogSink] | None = None,
        buffer_size: int = 1000,
        redact: Sequence[str] = (),
        async_write: bool = True,
        queue_size: int = 8192,
        schema_version: int = 1,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        telemetry = _telemetry_settings(settings)
        # Document 15 section 7.2 C13: a project that wrote LOGGER gets its level and
        # sinks from there; everyone else keeps the TELEMETRY path unchanged.
        use_logger_block = settings is not None and is_authored(settings, "LOGGER")
        logger_config = _logger_config(settings) if use_logger_block else {}
        self.run_id = run_id or telemetry.get("RUN_ID") or ""
        self.level = str(
            level or (logger_config.get("LEVEL") if use_logger_block else None)
            or telemetry.get("LEVEL", "INFO")
        ).upper()
        if self.level not in _LEVELS:
            raise ImproperlyConfigured(
                f"unknown log level {self.level!r}; use one of " + ", ".join(_LEVELS)
            )
        self.schema_version = int(schema_version)
        self.clock = clock
        self.monotonic = monotonic
        self.redact = tuple(str(fragment).lower() for fragment in redact) or (
            DEFAULT_REDACT
            if not redact
            else ()
        )
        self.buffer_size = int(buffer_size)
        self.dropped_records = 0
        self.sink_errors: list[str] = []
        self._lock = threading.RLock()
        self._closed = False

        if sinks is None:
            if use_logger_block:
                sinks = self._build_sinks(logger_config, run_id=self.run_id, settings=settings)
            else:
                sinks = self._default_sinks(telemetry)

        self.memory = MemorySink(buffer_size=self.buffer_size)
        self.sinks: list[LogSink] = list(sinks)
        # The memory sink is always present, so ``tail()`` works regardless of
        # configuration (document 09 section 2.2).
        if not any(getattr(sink, "name", "") == "memory" for sink in self.sinks):
            self.sinks.append(self.memory)

        self.async_write = bool(async_write)
        self._queue: queue.Queue[LogRecord | None] | None = None
        self._thread: threading.Thread | None = None
        self._error_records: list[LogRecord] = []
        if self.async_write:
            self._queue = queue.Queue(maxsize=max(1, int(queue_size)))
            self._thread = threading.Thread(
                target=self._writer_loop, name=f"mactf-log-{self.run_id or 'anon'}", daemon=True
            )
            self._thread.start()

    # -- construction ----------------------------------------------------
    @classmethod
    def from_settings(cls, settings: Any, *, run_id: str | None = None) -> "MACTFLogger":
        """Build a logger from ``LOGGER``, falling back to ``TELEMETRY``.

        A sink named in ``SINKS`` that needs a path and has none raises here, with
        the key name, rather than at the first write. Relative paths resolve
        against ``PATHS["ROOT"]`` when ``settings`` can resolve it.
        """
        config = _logger_config(settings)
        sinks = cls._build_sinks(config, run_id=run_id, settings=settings)
        return cls(
            settings,
            run_id=run_id,
            level=str(config.get("LEVEL", "INFO")),
            sinks=sinks,
            buffer_size=int(config.get("BUFFER_SIZE", 1000)),
            redact=tuple(config.get("REDACT") or ()),
            async_write=bool(config.get("ASYNC_WRITE", True)),
            queue_size=int(config.get("QUEUE_SIZE", 8192)),
            schema_version=int(config.get("SCHEMA_VERSION", 1)),
        )

    @staticmethod
    def _build_sinks(
        config: Mapping[str, Any], *, run_id: str | None = None, settings: Any = None
    ) -> list[LogSink]:
        """The sinks ``SINKS`` names, in order. ``memory`` is added by ``__init__``."""
        json_format = bool(config.get("JSON", False))
        built: list[LogSink] = []
        for name in config.get("SINKS", ["console", "file"]):
            key = str(name).lower()
            if key == "console":
                built.append(ConsoleSink(json_format=json_format))
            elif key == "file":
                built.append(
                    FileSink(
                        _required_path(config, "LOG_FILE", key, settings),
                        json_format=json_format,
                        max_bytes=int(config.get("ROTATE_BYTES", 10_485_760)),
                        backup_count=int(config.get("ROTATE_BACKUPS", 5)),
                    )
                )
            elif key == "jsonl":
                built.append(JsonlSink(_required_path(config, "JSONL_FILE", key, settings)))
            elif key == "tensorboard":
                built.append(
                    TensorBoardSink(_required_path(config, "TENSORBOARD_DIR", key, settings))
                )
            elif key == "wandb":
                built.append(WandBSink(project=config.get("WANDB_PROJECT"), config=dict(config)))
            else:
                raise ImproperlyConfigured(f"unknown LOGGER sink {name!r}")
        return built

    @staticmethod
    def _default_sinks(telemetry: Mapping[str, Any]) -> list[LogSink]:
        sinks: list[LogSink] = []
        json_format = bool(telemetry.get("JSON", False))
        if bool(telemetry.get("CONSOLE", True)):
            sinks.append(ConsoleSink(json_format=json_format))
        log_file = telemetry.get("LOG_FILE")
        if log_file:
            sinks.append(
                FileSink(
                    log_file,
                    json_format=json_format,
                    max_bytes=int(telemetry.get("ROTATE_BYTES", 10_485_760)),
                    backup_count=int(telemetry.get("ROTATE_BACKUPS", 5)),
                )
            )
        jsonl_file = telemetry.get("JSONL_FILE")
        if jsonl_file:
            sinks.append(JsonlSink(jsonl_file))
        return sinks

    # -- redaction -------------------------------------------------------
    def _redact(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        if not self.redact:
            return dict(fields)
        cleaned: dict[str, Any] = {}
        for key, value in fields.items():
            lowered = str(key).lower()
            cleaned[key] = (
                "***" if any(fragment in lowered for fragment in self.redact) else value
            )
        return cleaned

    # -- record construction --------------------------------------------
    def _record(
        self,
        cls: type[LogRecord],
        *,
        message: str = "",
        level: str = "info",
        node_path: str = "",
        depth: int = 0,
        epoch: int = 0,
        step: int = 0,
        duration: float = 0.0,
        fields: Mapping[str, Any] | None = None,
    ) -> LogRecord:
        return cls(  # type: ignore[call-arg]
            run_id=self.run_id,
            node_path=str(node_path or ""),
            depth=int(depth or 0),
            epoch=int(epoch or 0),
            step=int(step or 0),
            wall_time=float(self.clock()),
            monotonic=float(self.monotonic()),
            duration=float(duration or 0.0),
            level=str(level).lower(),
            message=message,
            fields=self._redact(fields or {}),
            schema_version=self.schema_version,
        )

    # -- writing ---------------------------------------------------------
    def log(self, record: LogRecord) -> None:
        """Enqueue or write a record. Never raises because a sink failed."""
        if self._closed:
            return
        severity = _LEVELS.get(record.level.upper(), 20)
        if severity < _LEVELS.get(self.level, 20):
            return
        # The memory sink is written by the single consumer in async mode so the
        # buffer and the files cannot disagree about order.
        if self._queue is None:
            self._write(record)
            return
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            if severity >= _LEVELS["ERROR"]:
                # Losing an error is not acceptable, so block rather than drop.
                self._queue.put(record)
                return
            self.dropped_records += 1

    def _write(self, record: LogRecord) -> None:
        for sink in self.sinks:
            try:
                sink.write(record)
            except Exception as error:  # noqa: BLE001 - a sink must not fail a step
                message = f"{getattr(sink, 'name', type(sink).__name__)}: {error}"
                if message not in self.sink_errors:
                    self.sink_errors.append(message)
                self.dropped_records += 1

    def _writer_loop(self) -> None:
        assert self._queue is not None
        while True:
            try:
                record = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if record is None:
                self._queue.task_done()
                break
            self._write(record)
            self._queue.task_done()

    # -- convenience methods --------------------------------------------
    def info(self, message: str, **fields: Any) -> None:
        self.log(self._record(LogRecord, message=message, level="info", fields=fields))

    def debug(self, message: str, **fields: Any) -> None:
        self.log(self._record(LogRecord, message=message, level="debug", fields=fields))

    def warning(self, message: str, **fields: Any) -> None:
        self.log(self._record(LogRecord, message=message, level="warning", fields=fields))

    def error(self, message: str, *, exc_info: bool = False, **fields: Any) -> None:
        payload = dict(fields)
        if exc_info:
            import traceback

            payload.setdefault("traceback", traceback.format_exc())
        self.log(self._record(LogRecord, message=message, level="error", fields=payload))

    def epoch(self, result: Any, **fields: Any) -> None:
        payload = dict(fields)
        payload.setdefault("steps", int(getattr(result, "steps", 0)))
        payload.setdefault("mean_loss", float(getattr(result, "mean_loss", 0.0)))
        payload.setdefault("completed", bool(getattr(result, "completed", True)))
        record = self._record(
            EpochRecord,
            epoch=int(getattr(result, "epoch", 0)),
            message=getattr(result, "reason", "") or "",
            fields=payload,
        )
        self.log(record)

    def step(self, node: Any, result: Any, **fields: Any) -> None:
        path = node if isinstance(node, str) else getattr(node, "path", "")
        depth = 0 if isinstance(node, str) else int(getattr(node, "depth", 0) or 0)
        payload = dict(fields)
        payload.setdefault("loss", float(getattr(result, "loss_value", 0.0)))
        payload.setdefault("stepped", bool(getattr(result, "stepped", True)))
        metrics = getattr(result, "metrics", None) or {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                payload.setdefault(str(key), float(value))
        self.log(
            self._record(
                StepRecord,
                node_path=path,
                depth=depth,
                fields=payload,
            )
        )

    def loss(self, breakdown: Any, *, node_path: str = "", **fields: Any) -> None:
        payload = dict(fields)
        if hasattr(breakdown, "as_dict"):
            payload.update(breakdown.as_dict())
        elif isinstance(breakdown, Mapping):
            payload.update(dict(breakdown))
        self.log(self._record(LossRecord, node_path=node_path, fields=payload))

    def score(self, card: Any, **fields: Any) -> None:
        payload = dict(fields)
        if hasattr(card, "to_dict"):
            payload["card"] = card.to_dict()
            payload.setdefault("final_score", float(getattr(card, "final_score", 0.0)))
        else:
            payload["card"] = _jsonable(card)
        self.log(
            self._record(
                ScoreRecord,
                node_path=str(getattr(card, "peer", "") or ""),
                epoch=int(getattr(card, "episode", 0) or 0),
                fields=payload,
            )
        )

    def aggregation(self, report: Any, **fields: Any) -> None:
        payload = dict(fields)
        if hasattr(report, "as_dict"):
            payload.update(report.as_dict())
        elif isinstance(report, Mapping):
            payload.update(dict(report))
        self.log(self._record(AggregationRecord, fields=payload))

    def checkpoint(self, record: Any, *, reason: str = "") -> None:
        payload: dict[str, Any] = {"reason": reason}
        describe = getattr(record, "describe", None)
        if callable(describe):
            payload["record"] = describe()
        path = getattr(record, "path", None)
        if path is not None:
            payload["path"] = str(path)
            payload["name"] = Path(path).name
        if hasattr(record, "episode"):
            payload.setdefault("episode", int(record.episode))
        self.log(self._record(CheckpointRecord, fields=payload))

    def health(self, report: Any, **fields: Any) -> None:
        payload = dict(fields)
        if hasattr(report, "as_dict"):
            payload.update(report.as_dict())
        elif isinstance(report, Mapping):
            payload.update(dict(report))
        status = str(payload.get("status", "")).lower()
        level = "error" if status == "critical" else ("warning" if status == "degraded" else "info")
        self.log(
            self._record(
                HealthRecord,
                level=level,
                epoch=int(payload.get("epoch", 0) or 0),
                fields=payload,
            )
        )

    def resume(self, report: Any, **fields: Any) -> None:
        payload = dict(fields)
        if hasattr(report, "as_dict"):
            payload.update(report.as_dict())
        elif isinstance(report, Mapping):
            payload.update(dict(report))
        self.log(
            self._record(
                ResumeRecord,
                epoch=int(payload.get("episode", 0) or 0),
                fields=payload,
            )
        )

    # -- reading ---------------------------------------------------------
    def tail(
        self,
        n: int = 100,
        *,
        node_path: str | None = None,
        kind: str | None = None,
    ) -> list[LogRecord]:
        records = [
            record
            for record in list(self.memory.records)
            if record.matches(node_path=node_path, kind=kind)
        ]
        return records[-int(n):] if n else records

    # -- lifecycle -------------------------------------------------------
    def flush(self, timeout: float | None = None) -> None:
        if self._queue is not None:
            deadline = None if timeout is None else time.monotonic() + timeout
            while not self._queue.empty():
                if deadline is not None and time.monotonic() > deadline:
                    self.error("log flush timed out with records still queued")
                    return
                time.sleep(0.005)
            # A short grace period so the consumer drains the last item it took.
            time.sleep(0.01)
        for sink in self.sinks:
            try:
                sink.flush()
            except Exception as error:  # noqa: BLE001
                message = f"{getattr(sink, 'name', 'sink')} flush: {error}"
                if message not in self.sink_errors:
                    self.sink_errors.append(message)

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        if self._queue is not None:
            self._queue.put(None)
            if self._thread is not None:
                self._thread.join(timeout=5.0)
                if self._thread.is_alive():
                    self.error("log writer thread did not join within 5s")
            self._queue = None
            self._thread = None
        for sink in self.sinks:
            try:
                sink.close()
            except Exception as error:  # noqa: BLE001
                message = f"{getattr(sink, 'name', 'sink')} close: {error}"
                if message not in self.sink_errors:
                    self.sink_errors.append(message)
        self._closed = True

    def __enter__(self) -> "MACTFLogger":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- persistence -----------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "level": self.level,
            "schema_version": self.schema_version,
            "dropped_records": self.dropped_records,
            "sink_errors": list(self.sink_errors),
            "sinks": [sink.state_dict() for sink in self.sinks],
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        state = dict(payload or {})
        if "run_id" in state and not self.run_id:
            self.run_id = str(state["run_id"])
        self.dropped_records = int(state.get("dropped_records", self.dropped_records))
        self.sink_errors = [str(item) for item in state.get("sink_errors", self.sink_errors)]
        by_name = {
            str(item.get("name")): item for item in state.get("sinks", []) if isinstance(item, Mapping)
        }
        for sink in self.sinks:
            offset = by_name.get(getattr(sink, "name", ""), {}).get("offset")
            if offset is not None and hasattr(sink, "offset"):
                sink.offset = int(offset)

    def describe(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "level": self.level,
            "sinks": [getattr(sink, "name", type(sink).__name__) for sink in self.sinks],
            "async": self.async_write,
            "dropped_records": self.dropped_records,
            "sink_errors": list(self.sink_errors),
        }

    # -- child -----------------------------------------------------------
    def child(self, suffix: str) -> "MACTFLogger":
        """A logger sharing this one's sinks and buffer, with a suffixed run id.

        Used by a distributed worker or a sandbox probe so its records stay
        attributable without a second file.
        """
        clone = object.__new__(MACTFLogger)
        clone.run_id = f"{self.run_id}.{suffix}" if self.run_id else str(suffix)
        clone.level = self.level
        clone.schema_version = self.schema_version
        clone.clock = self.clock
        clone.monotonic = self.monotonic
        clone.redact = self.redact
        clone.buffer_size = self.buffer_size
        clone.dropped_records = 0
        clone.sink_errors = self.sink_errors  # shared, so a failure is reported once
        clone._lock = self._lock
        clone._closed = False
        clone.memory = self.memory
        clone.sinks = self.sinks
        clone.async_write = self.async_write
        clone._queue = self._queue
        clone._thread = self._thread
        clone._error_records = self._error_records
        return clone

    # -- signal wiring ---------------------------------------------------
    def connect(self, signals_module: Any = None) -> "MACTFLogger":
        """Subscribe to framework signals.

        Only connects; never sends. That direction is what makes logging a pure
        observer.
        """
        if signals_module is None:
            from .. import signals as signals_module

        module = signals_module

        def on_phase_started(sender: Any, phase: str = "", **kwargs: Any) -> None:
            self._phase_started[phase] = float(self.monotonic())

        def on_phase_finished(sender: Any, phase: str = "", **kwargs: Any) -> None:
            started = self._phase_started.pop(phase, None)
            duration = 0.0 if started is None else float(self.monotonic()) - started
            self.log(
                self._record(
                    StepRecord,
                    message=f"phase {phase}",
                    duration=duration,
                    fields={"phase": phase, "keys": sorted(kwargs)},
                )
            )

        def on_peer_scored(sender: Any, card: Any = None, ledger: Any = None) -> None:
            if card is not None:
                self.score(card)

        def on_episode_scored(sender: Any, scorecard: Any = None) -> None:
            if scorecard is None:
                return
            self.log(
                self._record(
                    EpochRecord,
                    epoch=int(getattr(scorecard, "episode", 0)),
                    fields={"scores": dict(scorecard.scores)},
                )
            )

        def on_warning(sender: Any, action: str = "issue", **kwargs: Any) -> None:
            self.log(
                self._record(
                    WarningRecord,
                    fields={"action": action, **{k: v for k, v in kwargs.items() if k != "sender"}},
                )
            )

        def on_aggregation(sender: Any, result: Any = None, **kwargs: Any) -> None:
            report = getattr(result, "report", None) or kwargs.get("report")
            if report is not None:
                self.aggregation(report)

        def on_checkpoint(sender: Any, record: Any = None, restored: bool = False, **kwargs: Any) -> None:
            self.checkpoint(record, reason="restored" if restored else "")

        def on_rollback(sender: Any, **kwargs: Any) -> None:
            self.log(
                self._record(
                    CheckpointRecord,
                    level="warning",
                    fields={"reason": str(kwargs.get("reason", "rollback"))},
                )
            )

        def on_incident(sender: Any, **kwargs: Any) -> None:
            self.log(
                self._record(
                    HealthRecord,
                    level="error",
                    fields={key: _jsonable(value) for key, value in kwargs.items()},
                )
            )

        self._phase_started = {}
        module.phase_started.connect(on_phase_started)
        module.phase_finished.connect(on_phase_finished)
        module.episode_peer_scored.connect(on_peer_scored)
        module.episode_scored.connect(on_episode_scored)
        module.warning_issued.connect(
            lambda sender, **kw: on_warning(sender, action="issue", **kw)
        )
        module.warning_expired.connect(
            lambda sender, **kw: on_warning(sender, action="expire", **kw)
        )
        module.warning_escalated.connect(
            lambda sender, **kw: on_warning(sender, action="escalate", **kw)
        )
        module.aggregation_completed.connect(on_aggregation)
        module.checkpoint_written.connect(on_checkpoint)
        module.checkpoint_rolled_back.connect(on_rollback)
        module.integrity_incident.connect(on_incident)
        self._connected = True
        return self


def _logger_config(settings: Any) -> dict[str, Any]:
    """``LOGGER`` when the project wrote it, else ``TELEMETRY`` widened to its shape.

    "Wrote it" is :func:`mactf.conf_files.is_authored`: ``DEFAULTS`` carries a
    ``LOGGER`` block, and a framework default must not hide a project's
    ``TELEMETRY.LEVEL``.
    """
    if settings is not None and is_authored(settings, "LOGGER"):
        block = _block_of(settings, "LOGGER")
        if block:
            return block
    telemetry = _telemetry_settings(settings)
    sinks: list[str] = []
    if bool(telemetry.get("CONSOLE", True)):
        sinks.append("console")
    if telemetry.get("LOG_FILE"):
        sinks.append("file")
    if telemetry.get("JSONL_FILE"):
        sinks.append("jsonl")
    return {
        "LEVEL": telemetry.get("LEVEL", "INFO"),
        "JSON": telemetry.get("JSON", False),
        "SINKS": sinks,
        "LOG_FILE": telemetry.get("LOG_FILE"),
        "JSONL_FILE": telemetry.get("JSONL_FILE"),
        "ROTATE_BYTES": telemetry.get("ROTATE_BYTES", 10_485_760),
        "ROTATE_BACKUPS": telemetry.get("ROTATE_BACKUPS", 5),
        "ASYNC_WRITE": True,
        "BUFFER_SIZE": 1000,
    }


def _block_of(settings: Any, name: str) -> dict[str, Any]:
    """Read block ``name`` from a settings object or mapping; ``{}`` when absent."""
    if isinstance(settings, Mapping):
        block = settings.get(name)
    else:
        getter = getattr(settings, "get", None)
        block = getter(name, None) if callable(getter) else getattr(settings, name, None)
    return dict(block) if isinstance(block, Mapping) else {}


def _required_path(
    config: Mapping[str, Any], key: str, sink: str, settings: Any
) -> Path:
    """The path a sink needs, or an error naming the key rather than a TypeError."""
    value = config.get(key)
    if not value:
        raise ImproperlyConfigured(f"LOGGER.SINKS includes {sink!r} but {key} is unset")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    resolver = getattr(settings, "path", None)
    if callable(resolver):
        return resolver("ROOT") / path
    return path


def _telemetry_settings(settings: Any) -> dict[str, Any]:
    """Read the ``TELEMETRY`` block from a settings object or mapping."""
    if settings is None:
        return {}
    if isinstance(settings, Mapping):
        block = settings.get("TELEMETRY")
        return dict(block) if isinstance(block, Mapping) else {}
    getter = getattr(settings, "get", None)
    if callable(getter):
        block = getter("TELEMETRY", None)
        if isinstance(block, Mapping):
            return dict(block)
    block = getattr(settings, "TELEMETRY", None)
    if isinstance(block, Mapping):
        return dict(block)
    if isinstance(settings, Mapping):
        return dict(settings)
    return {}


# ---------------------------------------------------------------------------
# Per-run log files (document 16 section 7.5)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LogLocation:
    """Where a run's JSONL log lives."""

    path: Path
    shared: bool      # True: one file for every run (LOGGER.JSONL_FILE); records carry run_id
    exists: bool


def _root_of(settings: Any) -> Path:
    resolver = getattr(settings, "path", None)
    return Path(resolver("ROOT")) if callable(resolver) else Path.cwd()


def _resolve_against_root(settings: Any, value: Any) -> Path:
    target = Path(str(value)).expanduser()
    return target if target.is_absolute() else _root_of(settings) / target


def run_log_location(settings: Any, run_id: str) -> LogLocation:
    """Resolve the JSONL file for ``run_id``; the first matching rule wins.

    1. ``LOGGER.JSONL_FILE`` (or ``TELEMETRY`` widened to it): that file, shared by every run.
    2. ``<LOGGER.RUNS_DIR>/<run_id>/events.jsonl`` (default ``var/runs``): the run's own file.
    3. ``<PATHS.LOGS>/<run_id>.jsonl`` only when it already exists and (2) does not: the
       location the API read before the CLI wrote logs, kept so existing files stay visible.
    """
    from ..orchestration.recovery import validate_run_id

    validate_run_id(run_id)
    config = _logger_config(settings)
    shared_file = config.get("JSONL_FILE")
    if shared_file:
        path = _resolve_against_root(settings, shared_file)
        return LogLocation(path=path, shared=True, exists=path.exists())
    runs_dir = config.get("RUNS_DIR") or _block_of(settings, "LOGGER").get("RUNS_DIR") or "var/runs"
    own = _resolve_against_root(settings, runs_dir) / run_id / "events.jsonl"
    if not own.exists():
        resolver = getattr(settings, "path", None)
        if callable(resolver):
            try:
                legacy = Path(resolver("LOGS")) / f"{run_id}.jsonl"
            except ImproperlyConfigured:
                legacy = None
            if legacy is not None and legacy.exists():
                return LogLocation(path=legacy, shared=False, exists=True)
    return LogLocation(path=own, shared=False, exists=own.exists())


def run_log_path(settings: Any, run_id: str) -> Path:
    """``run_log_location(settings, run_id).path``."""
    return run_log_location(settings, run_id).path


def open_run_logger(settings: Any, run_id: str, *, path: Path | None = None) -> "MACTFLogger":
    """A synchronous logger writing only JSONL to ``path`` (default ``run_log_path``).

    ``async_write=False`` because the CLI is the only writer and wants each epoch on disk
    before the next begins.
    """
    target = Path(path) if path is not None else run_log_path(settings, run_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    config = _logger_config(settings)
    return MACTFLogger(
        settings,
        run_id=run_id,
        level=str(config.get("LEVEL", "INFO")),
        sinks=[JsonlSink(target)],
        async_write=False,
    )


def _matches(
    record: Mapping[str, Any], node_path: str | None, kind: str | None,
    level: str | None, run_id: str | None,
) -> bool:
    if node_path is not None and record.get("node_path") != node_path:
        return False
    if kind is not None and record.get("kind") != kind:
        return False
    if level is not None and str(record.get("level", "")).lower() != level:
        return False
    if run_id is not None and record.get("run_id") not in (None, "", run_id):
        return False       # a shared file holds every run; keep this run's and unstamped lines
    return True


def tail_jsonl(
    path: Path, n: int, *, node_path: str | None = None, kind: str | None = None,
    level: str | None = None, run_id: str | None = None, max_scan: int = 64 * 1024 * 1024,
    end: int | None = None,
) -> list[dict[str, Any]]:
    """The last ``n`` matching records, oldest first, reading backwards from the end.

    Reads 64 KiB blocks from the end, stops as soon as ``n`` records match or ``max_scan``
    bytes have been examined, ignores blank and malformed lines, and returns ``[]`` for a
    missing file. ``end`` starts the backward scan at that byte offset instead of the end.
    """
    path = Path(path)
    if n < 1 or not path.exists():
        return []
    wanted_level = level.lower() if level else None
    kept: list[dict[str, Any]] = []
    carry = b""
    scanned = 0
    with path.open("rb") as handle:
        size = handle.seek(0, os.SEEK_END)
        position = size if end is None else min(max(int(end), 0), size)
        while position > 0 and len(kept) < n and scanned < max_scan:
            step = min(65536, position)
            position -= step
            handle.seek(position)
            block = handle.read(step) + carry
            scanned += step
            lines = block.split(b"\n")
            carry = lines[0] if position > 0 else b""    # the first piece may be a partial line
            pending = lines[1:] if position > 0 else lines
            for raw in reversed(pending):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw.decode("utf-8", errors="replace"))
                except ValueError:
                    continue
                if not isinstance(record, dict) or not _matches(
                        record, node_path, kind, wanted_level, run_id):
                    continue
                kept.append(record)
                if len(kept) >= n:
                    break
    kept.reverse()
    return kept


def complete_size(path: Path) -> int:
    """Byte offset just past the last newline in ``path``; ``0`` for a missing or newline-free file."""
    path = Path(path)
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        position = handle.seek(0, os.SEEK_END)
        while position > 0:
            step = min(65536, position)
            position -= step
            handle.seek(position)
            block = handle.read(step)
            index = block.rfind(b"\n")
            if index != -1:
                return position + index + 1
    return 0


def _identity(stat: os.stat_result) -> tuple[int, int] | None:
    """(device, inode) when the platform reports one; ``None`` otherwise."""
    return (stat.st_dev, stat.st_ino) if stat.st_ino else None


def follow_jsonl(
    path: Path, *, stop: Callable[[], bool], from_end: bool = True, poll: float = 0.25,
    node_path: str | None = None, kind: str | None = None, level: str | None = None,
    run_id: str | None = None, sleep: Callable[[float], None] = time.sleep,
    offset: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield records as they are appended, until ``stop()`` is true.

    Only whole lines are yielded: a line the writer is mid-way through is held until its
    newline arrives. A file that shrinks (truncated) or is replaced (rotation: a different
    inode, or a size below the read offset) is re-read from its start. A file that does not
    exist yet is waited for. After ``stop()`` turns true the generator drains what is already
    on disk once more and returns. Offsets are bytes; lines are decoded one at a time, so a
    multi-byte character split across a poll never corrupts.
    """
    path = Path(path)
    wanted_level = level.lower() if level else None
    position: int | None = None
    identity: tuple[int, int] | None = None
    pending = b""
    if offset is not None:
        position = max(0, int(offset))
    elif from_end and path.exists():
        position = complete_size(path)

    def read_new() -> list[dict[str, Any]]:
        nonlocal position, identity, pending
        try:
            stat = path.stat()
        except OSError:
            position, identity, pending = (None if position is None else 0), None, b""
            return []
        current = _identity(stat)
        rotated = identity is not None and current is not None and current != identity
        if position is None:
            position = 0
        if rotated or stat.st_size < position:
            position, pending = 0, b""
        identity = current
        if stat.st_size == position:
            return []
        try:
            with path.open("rb") as handle:
                handle.seek(position)
                chunk = handle.read()
        except OSError:
            return []
        position += len(chunk)
        data = pending + chunk
        lines = data.split(b"\n")
        pending = lines.pop()               # an unterminated tail waits for its newline
        records: list[dict[str, Any]] = []
        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            if isinstance(record, dict) and _matches(record, node_path, kind, wanted_level, run_id):
                records.append(record)
        return records

    while True:
        finished = bool(stop())
        yield from read_new()
        if finished:
            return
        sleep(poll)
