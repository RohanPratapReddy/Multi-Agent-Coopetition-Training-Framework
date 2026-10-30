"""PEP 562 helper: packages whose public names live in submodules that import ``torch``."""

from __future__ import annotations

import importlib
from typing import Any, Callable, Mapping


def lazy_exports(
    package: str, exports: Mapping[str, str], namespace: dict[str, Any]
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """``__getattr__`` / ``__dir__`` for a package whose public names live in submodules.

    ``exports`` maps a public name to the submodule that defines it. The first access
    imports the submodule and caches the value in the package namespace, so
    ``__getattr__`` runs at most once per name.
    """

    def __getattr__(name: str) -> Any:
        module = exports.get(name)
        if module is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        value = getattr(importlib.import_module(f".{module}", package), name)
        namespace[name] = value
        return value

    def __dir__() -> list[str]:
        return sorted(set(namespace) | set(exports))

    return __getattr__, __dir__
