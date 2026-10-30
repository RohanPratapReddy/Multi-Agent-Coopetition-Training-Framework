"""Backward-compatible views over the registries (document 17 section 3.6).

The dicts the rest of the code and the tests already import stay as objects with the same
names. Each becomes a :class:`LegacyTable`: a real ``dict`` subclass (so
``isinstance(x, dict)``, ``json.dumps``, ``sorted(x)`` and ``dict(x)`` keep working) whose
reads are plain dict reads of a mirror kept in step by the registry, and whose writes are
routed through the registry so they get origin tracking, collision rules and lock checks.
"""

from __future__ import annotations

from typing import Any, Mapping

from .kinds import Origin
from .registry import HUB, ComponentRegistry, Hub

_UNSET: Any = object()


class LegacyTable(dict):
    """``name -> factory`` view of the *base layer* of one registry.

    Overlays are deliberately invisible here: this object is shared, process-wide state
    and a run's private parts must not leak into it. Framework code reads the registry
    (``HUB.registry(kind).get``); only third-party code that still holds the dict sees
    this view.
    """

    __slots__ = ("_kind", "_hub")

    def __init__(
        self,
        kind_id: str,
        builtins: Mapping[str, Any] | None = None,
        *,
        hub: Hub | None = None,
    ) -> None:
        super().__init__()
        self._kind = kind_id
        self._hub = hub if hub is not None else HUB
        registry = self._registry()
        registry.attach_mirror(self)
        for name, factory in (builtins or {}).items():
            registry.register_builtin(name, factory)

    def _registry(self) -> ComponentRegistry[Any]:
        return self._hub.registry(self._kind)

    @property
    def kind_id(self) -> str:
        return self._kind

    # -- writes route through the registry --------------------------------
    def __setitem__(self, name: str, factory: Any) -> None:
        self._registry().register(
            name, factory, origin=Origin.RUNTIME, replace=name in self, via="legacy"
        )

    def __delitem__(self, name: str) -> None:
        if name not in self:
            raise KeyError(name)
        self._registry().unregister(name, via="legacy")

    def pop(self, name: str, default: Any = _UNSET) -> Any:
        if name not in self:
            if default is _UNSET:
                raise KeyError(name)
            return default
        factory = dict.__getitem__(self, name)
        self._registry().unregister(name, via="legacy")
        return factory

    def popitem(self) -> tuple[str, Any]:
        if not self:
            raise KeyError("popitem(): table is empty")
        name = next(reversed(list(self)))
        return name, self.pop(name)

    def clear(self) -> None:
        for name in list(self):
            self._registry().unregister(name, via="legacy")

    def setdefault(self, name: str, default: Any = None) -> Any:
        if name not in self:
            self[name] = default
        return dict.__getitem__(self, name)

    def update(self, *args: Any, **kwargs: Any) -> None:
        merged: dict[str, Any] = dict(*args, **kwargs)
        for name, factory in merged.items():
            self[name] = factory

    def __ior__(self, other: Mapping[str, Any]) -> "LegacyTable":
        self.update(other)
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        # Pickle as a plain snapshot; a table cannot be re-attached in another process.
        return (dict, (dict(self),))

    def copy(self) -> dict[str, Any]:
        return dict(self)
