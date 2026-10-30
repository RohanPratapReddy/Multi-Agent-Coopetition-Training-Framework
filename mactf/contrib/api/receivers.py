"""Signal receivers: a process-wide tally the admin capabilities route reports."""

from __future__ import annotations

import threading
import time
from collections import defaultdict

from ... import signals

_LOCK = threading.Lock()
_COUNTS: dict[str, int] = defaultdict(int)
_LAST: dict[str, float] = {}

_TRACKED = (
    "episode_scored",
    "warning_issued",
    "warning_escalated",
    "warning_expired",
    "integrity_incident",
    "aggregation_completed",
    "checkpoint_written",
    "checkpoint_rolled_back",
)


def _make(name: str):
    def receiver(sender, **kwargs) -> None:
        with _LOCK:
            _COUNTS[name] += 1
            _LAST[name] = time.time()

    receiver.__name__ = f"count_{name}"
    return receiver


_RECEIVERS = {name: _make(name) for name in _TRACKED}
for _name, _receiver in _RECEIVERS.items():
    getattr(signals, _name).connect(_receiver, weak=False)


def signal_tally() -> dict[str, dict[str, float | int]]:
    """``{signal: {"count": n, "last": epoch_seconds}}`` since process start."""
    with _LOCK:
        return {name: {"count": _COUNTS[name], "last": _LAST.get(name, 0.0)} for name in _TRACKED}
