"""Testing utilities for MACTF. Exports resolve lazily: ``mactf.testing.cli`` needs no torch."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._lazy import lazy_exports

_EXPORTS: dict[str, str] = {
    "Client": "client", "TestResponse": "client",
    "Harness": "harness", "ToyPeer": "harness", "build_harness": "harness", "make_ledger": "harness",
    "CliResult": "cli", "invoke": "cli",
    "synthetic_backend": "strategy", "sweep": "strategy", "written": "strategy",
    "tree_of": "tree", "simulator": "tree", "findings": "tree",
}

if TYPE_CHECKING:  # pragma: no cover
    from .cli import CliResult, invoke
    from .client import Client, TestResponse
    from .harness import Harness, ToyPeer, build_harness, make_ledger
    from .strategy import synthetic_backend, sweep, written
    from .tree import findings, simulator, tree_of

__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS, globals())
