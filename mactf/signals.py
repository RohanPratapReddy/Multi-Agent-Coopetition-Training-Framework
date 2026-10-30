"""A minimal, dependency-free signal dispatcher modeled on Django signals."""

from __future__ import annotations

import logging
import threading
import weakref
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_EMPTY = object()


class Signal:
    """A named event that receivers can connect to.

    Receivers may return ``None`` to continue. Returning a non-``None`` value is
    recorded and returned to the sender, which lets a receiver veto an
    aggregation or a checkpoint.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._receivers: list[tuple[Any, Callable[..., Any]]] = []
        self._lock = threading.RLock()

    def connect(self, receiver: Callable[..., Any], *, weak: bool = True) -> Callable[..., Any]:
        with self._lock:
            if weak:
                for original, ref in self._receivers:
                    target = ref() if isinstance(ref, weakref.WeakMethod) else ref
                    if target is receiver or original is receiver:
                        return receiver
                try:
                    ref: Any = weakref.WeakMethod(receiver)  # type: ignore[arg-type]
                except TypeError:
                    ref = receiver
            else:
                ref = receiver
            self._receivers.append((receiver, ref))
        return receiver

    def disconnect(self, receiver: Callable[..., Any]) -> bool:
        with self._lock:
            for index, (original, ref) in enumerate(self._receivers):
                target = ref() if isinstance(ref, weakref.WeakMethod) else ref
                if original is receiver or target is receiver:
                    del self._receivers[index]
                    return True
        return False

    def has_receivers(self) -> bool:
        with self._lock:
            return any(
                (ref() is not None) if isinstance(ref, weakref.WeakMethod) else ref is not None
                for _, ref in self._receivers
            )

    def send(self, sender: Any, **named: Any) -> list[tuple[Any, Any]]:
        responses: list[tuple[Any, Any]] = []
        with self._lock:
            snapshot = list(self._receivers)
        for original, ref in snapshot:
            if isinstance(ref, weakref.WeakMethod):
                receiver = ref()
                if receiver is None:
                    with self._lock:
                        try:
                            self._receivers.remove((original, ref))
                        except ValueError:
                            pass
                    continue
            else:
                receiver = ref
            try:
                response = receiver(signal=self, sender=sender, **named)
            except Exception:  # noqa: BLE001 - receivers must never break training
                logger.exception("receiver %r failed for signal %s", receiver, self.name)
                responses.append((receiver, _EMPTY))
                continue
            responses.append((receiver, _EMPTY if response is None else response))
        return responses

    def first_response(self, sender: Any, **named: Any) -> Any:
        """Return the first non-``None`` receiver response, else ``None``."""
        for _, response in self.send(sender, **named):
            if response is not _EMPTY:
                return response
        return None


# Episode lifecycle
episode_started = Signal("episode_started")
episode_peer_scored = Signal("episode_peer_scored")
episode_scored = Signal("episode_scored")

# Scoring lifecycle
warning_issued = Signal("warning_issued")
warning_expired = Signal("warning_expired")
warning_escalated = Signal("warning_escalated")
integrity_incident = Signal("integrity_incident")

# Aggregation lifecycle
aggregation_ready = Signal("aggregation_ready")
aggregation_completed = Signal("aggregation_completed")

# Distillation lifecycle (document 14 section 12.1)
distillation_step = Signal("distillation_step")
distillation_teacher_error = Signal("distillation_teacher_error")
distillation_sigma_collapsed = Signal("distillation_sigma_collapsed")
distillation_fallback = Signal("distillation_fallback")
# Distillation conformance (document 14 section 4.6)
distillation_conformance_violation = Signal("distillation_conformance_violation")
distillation_trust_collapsed = Signal("distillation_trust_collapsed")

# Training lifecycle
phase_started = Signal("phase_started")
phase_progress = Signal("phase_progress")
phase_finished = Signal("phase_finished")
checkpoint_written = Signal("checkpoint_written")
checkpoint_rolled_back = Signal("checkpoint_rolled_back")

# Topology lifecycle (document 17 section 12.8)
tree_mutated = Signal("tree_mutated")

# Startup
app_ready = Signal("app_ready")
