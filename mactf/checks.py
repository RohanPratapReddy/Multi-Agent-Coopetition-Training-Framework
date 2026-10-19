"""System checks, modeled on Django's ``check`` framework."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

ERROR = "ERROR"
WARNING = "WARNING"
INFO = "INFO"


@dataclass(frozen=True)
class CheckMessage:
    level: str
    message: str
    hint: str | None = None
    id: str | None = None
    app: str | None = None

    def __str__(self) -> str:
        parts = [f"{self.level}{(':' + self.id) if self.id else ''}: {self.message}"]
        if self.hint:
            parts.append(f"HINT: {self.hint}")
        return "\n".join(parts)

    @property
    def is_fatal(self) -> bool:
        return self.level == ERROR


@dataclass
class CheckResult:
    messages: list[CheckMessage] = field(default_factory=list)
    checked: int = 0

    @property
    def errors(self) -> list[CheckMessage]:
        return [message for message in self.messages if message.level == ERROR]

    @property
    def warnings(self) -> list[CheckMessage]:
        return [message for message in self.messages if message.level == WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        counts = {level: 0 for level in (ERROR, WARNING, INFO)}
        for message in self.messages:
            counts[message.level] = counts.get(message.level, 0) + 1
        return (
            f"System check identified {counts[ERROR]} error(s), "
            f"{counts[WARNING]} warning(s), {counts[INFO]} info message(s) "
            f"across {self.checked} check(s)."
        )


_registry: dict[str, list[Callable[[], Any]]] = {}


def register(check: Callable[[], Any], *, app: str = "core") -> Callable[[], Any]:
    _registry.setdefault(app, []).append(check)
    return check


def check(**tags: Any):
    """Decorator registering a system check."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        register(func)
        return func

    return decorator


def _normalize(result: Any, app: str) -> list[CheckMessage]:
    if result is None:
        return []
    if isinstance(result, CheckMessage):
        return [result]
    if isinstance(result, (list, tuple)):
        messages: list[CheckMessage] = []
        for item in result:
            messages.extend(_normalize(item, app))
        return messages
    raise TypeError(f"check returned unsupported value {result!r}")


def run_checks(*, fail_level: str = ERROR) -> CheckResult:
    """Run every registered check and collect messages."""
    outcome = CheckResult()
    for app, checks in _registry.items():
        for check in checks:
            outcome.checked += 1
            try:
                messages = _normalize(check(), app)
            except Exception as exc:  # noqa: BLE001 - a broken check is an error
                messages = [
                    CheckMessage(
                        level=ERROR,
                        message=f"check {getattr(check, '__name__', check)!r} raised {exc!r}",
                        hint="Fix the check or the object it inspects.",
                        app=app,
                    )
                ]
            outcome.messages.extend(messages)
    return outcome
