"""Cooperative interrupt handling for long commands (document 16, section 6.6)."""

from __future__ import annotations

import signal
import threading
import time
from typing import Callable

_DEFAULT_MESSAGE = "stop requested: finishing epoch; press Ctrl-C again to abort"


class StopFlag:
    """The first SIGINT/SIGTERM asks for a stop; the second aborts.

    Also carries an optional wall-clock deadline so ``run --timeout`` and Ctrl-C share
    one ``should_stop`` callable. ``StopFlag`` does not call ``sys.exit`` and does not
    touch the work it guards; the caller polls :meth:`is_set` at its own boundaries.
    """

    def __init__(
        self,
        *,
        timeout: float | None = None,
        announce: Callable[[str], None] = lambda message: None,
        message: str = _DEFAULT_MESSAGE,
    ) -> None:
        self._deadline = None if timeout is None else time.monotonic() + float(timeout)
        self._announce = announce
        self._message = message
        self._event = threading.Event()
        self._signal_name: str | None = None
        self._previous: dict[int, object] = {}
        self._installed = False

    # -- signals ---------------------------------------------------------
    @staticmethod
    def _signals() -> list[int]:
        names = ["SIGINT", "SIGTERM", "SIGBREAK"]
        return [getattr(signal, name) for name in names if hasattr(signal, name)]

    def __enter__(self) -> "StopFlag":
        if threading.current_thread() is threading.main_thread():
            for number in self._signals():
                try:
                    self._previous[number] = signal.signal(number, self._handle)
                except (ValueError, OSError):  # pragma: no cover - exotic embedding
                    continue
            self._installed = True
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._installed:
            for number, handler in self._previous.items():
                try:
                    signal.signal(number, handler)  # type: ignore[arg-type]
                except (ValueError, OSError):  # pragma: no cover
                    pass
        self._previous.clear()
        self._installed = False

    def _handle(self, number: int, frame: object) -> None:
        if self._event.is_set() and self._signal_name is not None:
            raise KeyboardInterrupt()              # the second signal aborts
        self._signal_name = signal.Signals(number).name
        self._event.set()
        self._announce(self._message)

    # -- state -----------------------------------------------------------
    def _remaining(self) -> float | None:
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - time.monotonic())

    def is_set(self) -> bool:
        """A signal was received or the deadline passed."""
        if self._event.is_set():
            return True
        remaining = self._remaining()
        return remaining is not None and remaining <= 0.0

    def wait(self, timeout: float | None = None) -> bool:
        """Block until :meth:`is_set` or ``timeout``; returns :meth:`is_set`."""
        remaining = self._remaining()
        limits = [value for value in (timeout, remaining) if value is not None]
        self._event.wait(min(limits) if limits else None)
        return self.is_set()

    @property
    def reason(self) -> str | None:
        if self._signal_name is not None:
            return "signal"
        remaining = self._remaining()
        if remaining is not None and remaining <= 0.0:
            return "timeout"
        return None

    @property
    def signal_name(self) -> str | None:
        return self._signal_name

    @property
    def timed_out(self) -> bool:
        return self.reason == "timeout"
