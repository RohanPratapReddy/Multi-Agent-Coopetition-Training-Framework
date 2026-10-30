"""Distributed runs: a coordinator owns scoring and aggregation, workers own forward/backward.

Importable without the API app: only ``views`` and ``urls`` import ``mactf.http``. Names are
resolved lazily so a worker process that only needs :class:`Worker` does not load the
coordinator, and ``import mactf.distributed`` itself loads no torch.
"""

from __future__ import annotations

import importlib
from typing import Any

default_app_config = "mactf.distributed.apps.DistributedConfig"

_EXPORTS = {
    "Coordinator": "coordinator",
    "RemoteAdapter": "coordinator",
    "get_coordinator": "coordinator",
    "reset_coordinator": "coordinator",
    "Worker": "worker",
    "WorkerClient": "worker",
    "ParameterExchange": "protocol",
    "TaskAssignment": "protocol",
    "TaskResult": "protocol",
    "WorkerHeartbeat": "protocol",
    "WorkerRegistration": "protocol",
    "decode_tensors": "transport",
    "encode_tensors": "transport",
    "vector_checksum": "transport",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'mactf.distributed' has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value
