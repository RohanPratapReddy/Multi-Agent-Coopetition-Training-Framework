"""The wire dataclasses of distributed mode (document 05 section 3).

Every class is frozen, JSON-only (``to_dict``/``from_dict``) and torch-free: tensors travel
inside ``*_b64`` strings produced by :mod:`mactf.distributed.transport`. Fields beyond the
document's list are marked ``extension`` and carry defaults, so a message from a peer that
does not know them still parses.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..wire import need, opt

__all__ = [
    "ParameterExchange", "TaskAssignment", "TaskResult", "WorkerHeartbeat", "WorkerRegistration",
]


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in (value or ()))


def _names(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in (value or ()))


@dataclass(frozen=True)
class WorkerRegistration:
    worker_id: str
    hostname: str
    capabilities: Mapping[str, Any] = field(default_factory=dict)  # device, dtype, torch version
    max_peers: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "hostname": self.hostname,
            "capabilities": dict(self.capabilities),
            "max_peers": self.max_peers,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkerRegistration":
        owner = "WorkerRegistration"
        return cls(
            worker_id=str(need(owner, payload, "worker_id")),
            hostname=str(need(owner, payload, "hostname")),
            capabilities=dict(payload.get("capabilities") or {}),
            max_peers=int(opt(payload, "max_peers", 1)),
        )


@dataclass(frozen=True)
class WorkerHeartbeat:
    worker_id: str
    epoch: int
    peers: tuple[str, ...] = ()
    load: float = 0.0
    free_memory_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "epoch": self.epoch,
            "peers": list(self.peers),
            "load": self.load,
            "free_memory_bytes": self.free_memory_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkerHeartbeat":
        owner = "WorkerHeartbeat"
        return cls(
            worker_id=str(need(owner, payload, "worker_id")),
            epoch=int(need(owner, payload, "epoch")),
            peers=_names(payload.get("peers")),
            load=float(opt(payload, "load", 0.0)),
            free_memory_bytes=int(opt(payload, "free_memory_bytes", 0)),
        )


@dataclass(frozen=True)
class TaskAssignment:
    """One epoch's local-update work for a worker.

    ``payload_b64`` is a ``torch.save`` of ``{"states", "targets"}`` plus, for every assigned
    peer, ``"parameters::<peer>"`` (its current flat vector) and ``"reference"`` (extension:
    the worker holds no state between epochs, so a restarted worker is indistinguishable).
    """

    run_id: str
    epoch: int
    task_id: str
    states_shape: tuple[int, ...]
    targets_shape: tuple[int, ...]
    peer_names: tuple[str, ...]
    payload_b64: str
    timeout: float
    checksum: str = ""  # extension: blake2b of the serialized payload
    config: Mapping[str, Any] = field(default_factory=dict)  # extension: loss config, signals, voids

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "epoch": self.epoch,
            "task_id": self.task_id,
            "states_shape": list(self.states_shape),
            "targets_shape": list(self.targets_shape),
            "peer_names": list(self.peer_names),
            "payload_b64": self.payload_b64,
            "timeout": self.timeout,
            "checksum": self.checksum,
            "config": dict(self.config),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskAssignment":
        owner = "TaskAssignment"
        return cls(
            run_id=str(need(owner, payload, "run_id")),
            epoch=int(need(owner, payload, "epoch")),
            task_id=str(need(owner, payload, "task_id")),
            states_shape=_shape(need(owner, payload, "states_shape")),
            targets_shape=_shape(need(owner, payload, "targets_shape")),
            peer_names=_names(need(owner, payload, "peer_names")),
            payload_b64=str(need(owner, payload, "payload_b64")),
            timeout=float(need(owner, payload, "timeout")),
            checksum=str(opt(payload, "checksum", "")),
            config=dict(payload.get("config") or {}),
        )


@dataclass(frozen=True)
class TaskResult:
    """What a worker computed: one flat parameter vector per assigned peer."""

    task_id: str
    worker_id: str
    peer_names: tuple[str, ...]
    parameters_b64: str
    checksum: str
    metrics: Mapping[str, float] = field(default_factory=dict)  # local loss, grad norm, duration
    ok: bool = True
    error: str | None = None
    epoch: int = -1  # extension: lets the coordinator key the barrier without a lookup
    run_id: str = ""  # extension

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "peer_names": list(self.peer_names),
            "parameters_b64": self.parameters_b64,
            "checksum": self.checksum,
            "metrics": {str(k): float(v) for k, v in self.metrics.items()},
            "ok": self.ok,
            "error": self.error,
            "epoch": self.epoch,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskResult":
        owner = "TaskResult"
        return cls(
            task_id=str(need(owner, payload, "task_id")),
            worker_id=str(need(owner, payload, "worker_id")),
            peer_names=_names(need(owner, payload, "peer_names")),
            parameters_b64=str(opt(payload, "parameters_b64", "")),
            checksum=str(opt(payload, "checksum", "")),
            metrics={str(k): float(v) for k, v in (payload.get("metrics") or {}).items()},
            ok=bool(opt(payload, "ok", True)),
            error=opt(payload, "error", None),
            epoch=int(opt(payload, "epoch", -1)),
            run_id=str(opt(payload, "run_id", "")),
        )


@dataclass(frozen=True)
class ParameterExchange:
    """The aggregated vector of one epoch, as the coordinator publishes it."""

    run_id: str
    epoch: int
    parameters_b64: str
    checksum: str
    reference_b64: str | None
    kappa: float
    reference_checksum: str | None = None  # extension
    reset_optimizers: bool = True  # extension: workers clear optimizer state exactly as the host does

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "epoch": self.epoch,
            "parameters_b64": self.parameters_b64,
            "checksum": self.checksum,
            "reference_b64": self.reference_b64,
            "kappa": self.kappa,
            "reference_checksum": self.reference_checksum,
            "reset_optimizers": self.reset_optimizers,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ParameterExchange":
        owner = "ParameterExchange"
        return cls(
            run_id=str(need(owner, payload, "run_id")),
            epoch=int(need(owner, payload, "epoch")),
            parameters_b64=str(need(owner, payload, "parameters_b64")),
            checksum=str(need(owner, payload, "checksum")),
            reference_b64=opt(payload, "reference_b64", None),
            kappa=float(need(owner, payload, "kappa")),
            reference_checksum=opt(payload, "reference_checksum", None),
            reset_optimizers=bool(opt(payload, "reset_optimizers", True)),
        )
