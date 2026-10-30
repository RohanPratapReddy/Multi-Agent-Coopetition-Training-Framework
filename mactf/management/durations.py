"""Durations and times for command-line options (document 16, section 6.5)."""

from __future__ import annotations

import re
import time
from datetime import datetime

from ..exceptions import CommandError

_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0}
_GROUP = re.compile(r"(\d+(?:\.\d+)?)([smhdw])", re.IGNORECASE)
_WHOLE = re.compile(r"(?:\d+(?:\.\d+)?[smhdw])+", re.IGNORECASE)
_BARE = re.compile(r"\d+(?:\.\d+)?")


def parse_duration(text: str) -> float:
    """'90', '90s', '15m', '2h', '3d', '1w', '1h30m' -> seconds. Bare numbers are seconds."""
    raw = text.strip() if isinstance(text, str) else str(text)
    if _BARE.fullmatch(raw):
        return float(raw)
    if raw and _WHOLE.fullmatch(raw):
        return sum(float(number) * _UNITS[unit.lower()] for number, unit in _GROUP.findall(raw))
    raise CommandError.usage(
        f"invalid duration {text!r}; use forms like 90s, 15m, 2h, 3d, 1h30m"
    )


def format_duration(seconds: float) -> str:
    """0.4s, 12.3s, 4m07s, 2h03m, 3d04h."""
    seconds = max(0.0, float(seconds))
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    whole = int(round(seconds))
    days, rest = divmod(whole, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s"


def format_timestamp(ts: float | None) -> str:
    """'2026-10-22 22:53:07' in local time; '-' for None or an unrepresentable value."""
    if ts is None:
        return "-"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OverflowError, OSError, TypeError):
        return "-"


def parse_when(text: str, *, now: float | None = None) -> float:
    """A duration ago ('2h' -> now - 7200) or an ISO-8601 date/time -> a POSIX timestamp."""
    reference = time.time() if now is None else now
    raw = text.strip()
    try:
        return reference - parse_duration(raw)
    except CommandError:
        pass
    iso = raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw   # Python 3.10 rejects 'Z'
    try:
        return datetime.fromisoformat(iso).timestamp()               # naive means local time
    except ValueError:
        raise CommandError.usage(
            f"invalid time {text!r}; use a duration ago (2h, 3d, 1h30m) or an ISO-8601 "
            f"date or time (2026-10-22, 2026-10-22T14:30, 2026-10-22T14:30:00Z)"
        ) from None
