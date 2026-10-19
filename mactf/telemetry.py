"""Telemetry: structured logging with optional JSON output."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        return json.dumps(payload, default=str)


class PlainFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} {record.name}: {record.getMessage()}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(
    *,
    level: str = "INFO",
    json_output: bool = False,
    console: bool = True,
    log_file: Path | str | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the root MACTF logger once."""
    global _CONFIGURED
    logger = logging.getLogger("mactf")
    if _CONFIGURED and not force:
        logger.setLevel(level.upper())
        return logger

    logger.setLevel(level.upper())
    logger.handlers.clear()
    logger.propagate = False
    formatter: logging.Formatter = JsonFormatter() if json_output else PlainFormatter()

    if console:
        stream = logging.StreamHandler(stream=sys.stderr)
        stream.setFormatter(formatter)
        logger.addHandler(stream)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _CONFIGURED = True
    return logger


def get_logger(name: str = "mactf") -> logging.Logger:
    return logging.getLogger(name if name.startswith("mactf") else f"mactf.{name}")


def log_event(logger: logging.Logger, message: str, **fields: Any) -> None:
    """Log a message with structured fields attached."""
    logger.info(message, extra={"extra_fields": fields})


def reset_logging() -> None:
    """Clear handlers. Intended for tests."""
    global _CONFIGURED
    logger = logging.getLogger("mactf")
    logger.handlers.clear()
    _CONFIGURED = False
