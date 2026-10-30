"""Declarative parameter descriptions (document 17 section 3.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MISSING: Any = object()


@dataclass(frozen=True)
class ParamSpec:
    """Optional, declarative description of one constructor parameter.

    A component that supplies a ``params_schema`` gets typed validation *before* its
    constructor runs. A component that does not still gets unknown-key rejection from
    constructor reflection.
    """

    type: type | tuple[type, ...] | None = None
    default: Any = MISSING
    required: bool = False
    choices: tuple[Any, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    doc: str = ""

    def problems(self, name: str, value: Any) -> list[str]:
        found: list[str] = []
        if self.type is not None:
            declared = self.type if isinstance(self.type, tuple) else (self.type,)
            allowed = declared
            numeric_only = all(t in (int, float) for t in declared)
            # An integer is a valid float: YAML writes ``lr: 1`` and means 1.0.
            if float in allowed and int not in allowed:
                allowed = (*allowed, int)
            if isinstance(value, bool) and numeric_only:
                found.append(f"{name}: expected a number, got bool")
            elif not isinstance(value, allowed):
                names = "/".join(t.__name__ for t in declared)
                found.append(f"{name}: expected {names}, got {type(value).__name__}")
        if self.choices and value not in self.choices:
            found.append(f"{name}: {value!r} is not one of {list(self.choices)}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if self.minimum is not None and value < self.minimum:
                found.append(f"{name}: {value} is below the minimum {self.minimum}")
            if self.maximum is not None and value > self.maximum:
                found.append(f"{name}: {value} is above the maximum {self.maximum}")
        return found

    def describe(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"required": self.required, "doc": self.doc}
        if self.type is not None:
            allowed = self.type if isinstance(self.type, tuple) else (self.type,)
            payload["type"] = "/".join(t.__name__ for t in allowed)
        if self.default is not MISSING:
            payload["default"] = self.default
        if self.choices:
            payload["choices"] = list(self.choices)
        if self.minimum is not None:
            payload["minimum"] = self.minimum
        if self.maximum is not None:
            payload["maximum"] = self.maximum
        return payload
