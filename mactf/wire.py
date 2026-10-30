"""Shared helpers for the hand-written ``from_dict`` classmethods (document 13 section 5).

Three rules every ``from_dict`` follows, enforced here so they cannot drift apart:

* a missing required key raises ``KeyError("Class.field")`` rather than defaulting;
* unknown keys are ignored, so an older client can read a newer server;
* ``None`` is preserved: only an *absent* key takes a default.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def need(owner: str, payload: Mapping[str, Any], key: str) -> Any:
    """``payload[key]``, or a ``KeyError`` naming ``owner.key``."""
    if not isinstance(payload, Mapping):
        raise TypeError(f"{owner}.from_dict expects a mapping, got {type(payload).__name__}")
    try:
        return payload[key]
    except KeyError:
        raise KeyError(f"{owner}.{key}") from None


def opt(payload: Mapping[str, Any], key: str, default: Any) -> Any:
    """``payload[key]`` when present (even if ``None``), else ``default``."""
    return payload[key] if key in payload else default


def floats(mapping: Mapping[str, Any] | None) -> dict[str, float]:
    return {str(key): float(value) for key, value in (mapping or {}).items()}


def strings(values: Any) -> tuple[str, ...]:
    return tuple(str(value) for value in (values or ()))
