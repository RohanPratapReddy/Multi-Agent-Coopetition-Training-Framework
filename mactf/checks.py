"""System checks, modeled on Django's ``check`` framework.

Document 16, section 7.7. ``fail_level`` is now read, ``@check`` tags are kept, messages are
stamped with the app that produced them, and a run can be narrowed to an app, a tag, or a
message-id prefix. Every default invocation behaves as it did before.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

ERROR = "ERROR"
WARNING = "WARNING"
INFO = "INFO"

_RANK = {INFO: 0, WARNING: 1, ERROR: 2}
LEVELS = (ERROR, WARNING, INFO)


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

    def as_dict(self) -> dict[str, Any]:
        return {"level": self.level, "id": self.id, "app": self.app,
                "message": self.message, "hint": self.hint}


@dataclass
class CheckResult:
    messages: list[CheckMessage] = field(default_factory=list)
    checked: int = 0
    fail_level: str = ERROR

    @property
    def errors(self) -> list[CheckMessage]:
        return [message for message in self.messages if message.level == ERROR]

    @property
    def warnings(self) -> list[CheckMessage]:
        return [message for message in self.messages if message.level == WARNING]

    def failing(self) -> list[CheckMessage]:
        """Messages at or above ``fail_level``."""
        threshold = _RANK[self.fail_level]
        return [message for message in self.messages if _RANK.get(message.level, 0) >= threshold]

    @property
    def ok(self) -> bool:
        return not self.failing()

    def counts(self) -> dict[str, int]:
        counts = {level: 0 for level in LEVELS}
        for message in self.messages:
            counts[message.level] = counts.get(message.level, 0) + 1
        return counts

    def summary(self) -> str:
        counts = self.counts()
        return (
            f"System check identified {counts[ERROR]} error(s), "
            f"{counts[WARNING]} warning(s), {counts[INFO]} info message(s) "
            f"across {self.checked} check(s)."
        )

    def as_dict(self) -> dict[str, Any]:
        """The JSON shape of ``mactf check --format json`` (without ``conf_notes``)."""
        return {
            "ok": self.ok,
            "fail_level": self.fail_level,
            "summary": self.summary(),
            "checked": self.checked,
            "counts": self.counts(),
            "messages": [message.as_dict() for message in self.messages],
        }


@dataclass(frozen=True)
class CheckInfo:
    """One registered check, as ``check --list`` reports it."""

    name: str          # module + function, e.g. "mactf.checks.check_rules"
    app: str
    tags: tuple[str, ...]
    requires: str | None
    doc: str           # first line of the docstring


_registry: dict[str, list[Callable[[], Any]]] = {}

#: ``check -> dotted app name`` for checks that only make sense while that app is installed.
_requires: dict[Callable[[], Any], str] = {}

#: ``check -> tags`` recorded by ``@check(**tags)`` / ``register(tags=...)``.
_tags: dict[Callable[[], Any], frozenset[str]] = {}


def register(
    check: Callable[[], Any], *, app: str = "core", requires: str | None = None,
    tags: Collection[str] = (),
) -> Callable[[], Any]:
    """Register a check. ``requires`` is an ``INSTALLED_APPS`` entry the check depends on.

    The registry is process-wide and outlives a settings reload, so a check registered
    by an app that a later configuration no longer installs must not keep reporting.
    ``tags`` label the check for ``run_checks(tags=...)`` and ``check --tag``.
    """
    registered = _registry.setdefault(app, [])
    if check not in registered:  # ready() may run again after a registry reset
        registered.append(check)
    if requires is not None:
        _requires[check] = requires
    if tags:
        _tags[check] = frozenset(_tags.get(check, frozenset())) | frozenset(str(t) for t in tags)
    return check


def _applicable(check: Callable[[], Any]) -> bool:
    needed = _requires.get(check)
    if needed is None:
        return True
    from .conf import get_settings

    return needed in (get_settings().get("INSTALLED_APPS") or [])


def check(*tag_names: str, **tags: Any):
    """Decorator registering a system check.

    ``@check("rules")`` and ``@check(rules=True)`` both tag the check ``rules`` (a keyword
    whose value is falsy is not a tag).
    """
    labels = [*tag_names, *(name for name, enabled in tags.items() if enabled)]

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        register(func, tags=labels)
        return func

    return decorator


def _normalize(result: Any, app: str) -> list[CheckMessage]:
    if result is None:
        return []
    if isinstance(result, CheckMessage):
        return [result if result.app is not None else dataclasses.replace(result, app=app)]
    if isinstance(result, (list, tuple)):
        messages: list[CheckMessage] = []
        for item in result:
            messages.extend(_normalize(item, app))
        return messages
    raise TypeError(f"check returned unsupported value {result!r}")


def _name_of(check: Callable[..., Any]) -> str:
    module = getattr(check, "__module__", "") or ""
    function = getattr(check, "__qualname__", None) or getattr(check, "__name__", repr(check))
    return f"{module}.{function}" if module else str(function)


def _validate_level(fail_level: str) -> str:
    level = str(fail_level).upper()
    if level not in _RANK:
        raise ValueError(f"fail_level must be one of {', '.join(LEVELS)}, not {fail_level!r}")
    return level


def run_checks(
    *, fail_level: str = ERROR, app: str | None = None, tags: Collection[str] = (),
    ids: Collection[str] = (),
) -> CheckResult:
    """Run the registered checks and collect messages.

    ``app`` keeps only that app's checks; ``tags`` keeps checks carrying *any* listed tag;
    ``ids`` filters *messages* by prefix (``"mactf.E0"`` keeps ``mactf.E001``, ``mactf.E002``).
    ``checked`` counts the checks actually run. ``ValueError`` for an unknown ``fail_level``.
    """
    outcome = CheckResult(fail_level=_validate_level(fail_level))
    wanted_tags = {str(tag) for tag in tags}
    prefixes = tuple(str(prefix) for prefix in ids)
    for owner, checks in list(_registry.items()):
        if app is not None and owner != app:
            continue
        for check_function in list(checks):
            if not _applicable(check_function):
                continue
            if wanted_tags and not (wanted_tags & _tags.get(check_function, frozenset())):
                continue
            outcome.checked += 1
            try:
                messages = _normalize(check_function(), owner)
            except Exception as exc:  # noqa: BLE001 - a broken check is an error
                messages = [
                    CheckMessage(
                        level=ERROR,
                        message=f"check {getattr(check_function, '__name__', check_function)!r} raised {exc!r}",
                        hint="Fix the check or the object it inspects.",
                        app=owner,
                    )
                ]
            if prefixes:
                messages = [m for m in messages if m.id and m.id.startswith(prefixes)]
            outcome.messages.extend(messages)
    return outcome


def list_checks() -> list[CheckInfo]:
    """Every registered check that applies to the current settings, in registration order."""
    infos: list[CheckInfo] = []
    for owner, checks in _registry.items():
        for check_function in checks:
            if not _applicable(check_function):
                continue
            doc = (getattr(check_function, "__doc__", "") or "").strip().splitlines()
            infos.append(CheckInfo(
                name=_name_of(check_function),
                app=owner,
                tags=tuple(sorted(_tags.get(check_function, frozenset()))),
                requires=_requires.get(check_function),
                doc=doc[0].strip() if doc else "",
            ))
    return infos
