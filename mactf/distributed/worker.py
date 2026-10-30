"""The worker: a client of the coordinator that owns forward/backward passes (document 05 section 5).

A worker process imports this module and nothing from ``mactf.http``'s server side: it never
constructs a WSGI application. Its only channel is :class:`WorkerClient`, which speaks the
six ``worker`` routes over ``urllib``.
"""

from __future__ import annotations

import copy
import logging
import os
import socket
import threading
import time
from collections.abc import Mapping
from typing import Any

import torch

from ..exceptions import DistributedError, ImproperlyConfigured, ParameterIntegrityError
from .protocol import (
    ParameterExchange,
    TaskAssignment,
    TaskResult,
    WorkerHeartbeat,
    WorkerRegistration,
)
from .transport import decode_tensors, encode_tensors

__all__ = ["Worker", "WorkerClient"]

logger = logging.getLogger("mactf.distributed.worker")

_DEFAULT_PREFIX = "/worker/v1"


class WorkerClient:
    """The client side of the six worker routes; every method returns the decoded protocol object."""

    def __init__(
        self,
        coordinator_url: str,
        *,
        token: str | None = None,
        prefix: str = _DEFAULT_PREFIX,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff: float = 0.5,
    ) -> None:
        from ..client.transport import HTTPTransport

        self.prefix = "/" + prefix.strip("/")
        self._transport = HTTPTransport(
            coordinator_url, token=token, timeout=timeout, max_retries=max_retries, backoff=backoff,
            probe=False, prefix=self.prefix,
        )

    def _call(self, method: str, route: str, *, json: Any = None, params: Mapping[str, Any] | None = None,
              headers: Mapping[str, str] | None = None) -> Any:
        from ..client.errors import raise_for_response

        response = raise_for_response(
            self._transport.request(method, f"{self.prefix}/{route}", json=json, params=params, headers=headers)
        )
        return response.json() if response.content else {}

    def register(self, registration: WorkerRegistration) -> dict[str, Any]:
        return dict(self._call("POST", "register", json=registration.to_dict()))

    def heartbeat(self, heartbeat: WorkerHeartbeat) -> dict[str, Any]:
        return dict(self._call("POST", "heartbeat", json=heartbeat.to_dict()))

    def tasks(self, worker_id: str, *, wait: float = 20.0) -> TaskAssignment | None:
        body = self._call("GET", "tasks", params={"worker_id": worker_id, "wait": wait})
        task = body.get("task") if isinstance(body, Mapping) else None
        return TaskAssignment.from_dict(task) if task else None

    def results(self, result: TaskResult) -> dict[str, Any]:
        # The task id doubles as the idempotency key: a replayed submission is answered, not re-run.
        return dict(self._call("POST", "results", json=result.to_dict(), headers={"Idempotency-Key": result.task_id}))

    def exchange(self, worker_id: str, run_id: str, epoch: int, *, wait: float = 20.0) -> ParameterExchange | None:
        body = self._call(
            "POST", "exchange", json={"worker_id": worker_id, "run_id": run_id, "epoch": epoch, "wait": wait}
        )
        found = body.get("exchange") if isinstance(body, Mapping) else None
        return ParameterExchange.from_dict(found) if found else None

    def deregister(self, worker_id: str) -> None:
        self._call("POST", "deregister", json={"worker_id": worker_id})


class Worker:
    def __init__(
        self,
        *,
        coordinator_url: str,
        token: str | None = None,
        device: str = "auto",
        max_peers: int = 1,
        heartbeat_interval: float = 5.0,
        worker_id: str | None = None,
        hostname: str | None = None,
        settings: Any = None,
        prefix: str = _DEFAULT_PREFIX,
        poll_wait: float = 20.0,
        client: WorkerClient | None = None,
    ) -> None:
        if settings is None:
            from ..conf import get_settings

            settings = get_settings()
        block = dict(settings.get("DISTRIBUTED") or {})
        self.settings = settings
        self.hostname = hostname or socket.gethostname()
        self.worker_id = worker_id or block.get("WORKER_ID") or f"{self.hostname}-{os.getpid()}"
        self.max_peers = int(max_peers)
        self.heartbeat_interval = float(heartbeat_interval)
        self.poll_wait = float(poll_wait)
        self.device = device
        self.client = client or WorkerClient(coordinator_url, token=token, prefix=prefix, timeout=poll_wait + 30.0)
        self.peers: dict[str, Any] = {}
        self.reference: Any = None
        self._runtime: Any = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._beat: threading.Thread | None = None
        self._epoch = -1
        self._busy = False
        self.registered = False
        self.completed_epochs = 0

    # -- runtime and peers ------------------------------------------------------------
    @property
    def runtime(self) -> Any:
        if self._runtime is None:
            from ..runtime import Runtime, resolve_device, resolve_dtype

            requested = self.settings["DEVICE"] if self.device in (None, "auto") else self.device
            device = resolve_device(requested)
            self._runtime = Runtime(
                device=device,
                dtype=resolve_dtype(self.settings["DTYPE"], device),
                amp=False,
                seed=int(self.settings["SEED"]),
                deterministic=bool(self.settings["DETERMINISTIC"]),
            )
        return self._runtime

    def build_peers(self, names: tuple[str, ...] | list[str]) -> dict[str, Any]:
        """Build (once) the peers named in an assignment through the same path the host uses."""
        from ..coopetition.peer import build_peer, build_reference, require_same_layout
        from ..orchestration.orchestrator import _default_modules

        missing = [name for name in names if name not in self.peers]
        if missing:
            modules = _default_modules(self.settings)
            training = self.settings["TRAINING"]
            for name in missing:
                if name not in modules:
                    raise ImproperlyConfigured(
                        f"peer {name!r} is not defined in this worker's PEERS setting; the project "
                        "package and settings must be deployed to every host"
                    )
                self.peers[name] = build_peer(
                    copy.deepcopy(modules[name]),
                    name=name,
                    runtime=self.runtime,
                    learning_rate=float(training["LEARNING_RATE"]),
                    action_dim=int(training["ACTION_DIM"]),
                    state_dim=int(training["STATE_DIM"]),
                )
            require_same_layout(self.peers)
        if self.reference is None and self.peers:
            self.reference = build_reference(next(iter(self.peers.values())), runtime=self.runtime)
        return {name: self.peers[name] for name in names}

    # -- one epoch ---------------------------------------------------------------------
    def run_epoch(self, assignment: TaskAssignment, peers: dict[str, Any]) -> TaskResult:
        """Decode the assignment, run the local update for its peers, return their parameter vectors."""
        from ..orchestration.phases import run_local_step
        from ..rl.loss import LossConfig

        started = time.time()
        try:
            tensors = decode_tensors(
                assignment.payload_b64,
                assignment.checksum,
                expected=[("states", assignment.states_shape), ("targets", assignment.targets_shape)],
            )
            config = assignment.config
            runtime = self.runtime
            states = runtime.to(tensors["states"])
            loss = LossConfig(**dict(config.get("loss") or {}))
            max_norm = config.get("max_grad_norm")
            reference = self.reference
            if "reference" in tensors:
                self._load(reference, tensors["reference"], "reference")
            metrics: dict[str, float] = {}
            out: dict[str, torch.Tensor] = {}
            for name in assignment.peer_names:
                peer = peers[name]
                key = f"parameters::{name}"
                if key not in tensors:
                    raise ParameterIntegrityError(f"assignment carries no {key!r}")
                self._load(peer, tensors[key], name)
                breakdown = run_local_step(
                    peer,
                    reference,
                    states,
                    float((config.get("signals") or {})[name]),
                    runtime=runtime,
                    void=bool((config.get("void") or {}).get(name, False)),
                    floor=float((config.get("floor") or {}).get(name, 0.0)),
                    loss_config=loss,
                    max_grad_norm=float(max_norm) if max_norm else None,
                )
                for metric, value in breakdown.items():
                    metrics[f"{name}.{metric}"] = float(value)
                out[name] = peer.adapter.export_parameters().detach().cpu()
            metrics["duration"] = time.time() - started
            blob, checksum = encode_tensors(out, compress=bool(self.settings.get("DISTRIBUTED", {}).get("COMPRESS", True)))
            return TaskResult(
                task_id=assignment.task_id, worker_id=self.worker_id, peer_names=assignment.peer_names,
                parameters_b64=blob, checksum=checksum, metrics=metrics, ok=True,
                epoch=assignment.epoch, run_id=assignment.run_id,
            )
        except Exception as error:  # noqa: BLE001 - a failed task is reported, never swallowed
            logger.exception("task %s failed", assignment.task_id)
            return TaskResult(
                task_id=assignment.task_id, worker_id=self.worker_id, peer_names=assignment.peer_names,
                parameters_b64="", checksum="", metrics={"duration": time.time() - started}, ok=False,
                error=f"{type(error).__name__}: {error}", epoch=assignment.epoch, run_id=assignment.run_id,
            )

    @staticmethod
    def _load(peer: Any, vector: torch.Tensor, name: str) -> None:
        local = peer.adapter.parameter_shapes()
        total = sum(int(torch.Size(shape).numel()) for _, shape in local)
        if int(vector.numel()) != total:
            raise ParameterIntegrityError(
                f"parameters for {name!r} have shape {tuple(vector.shape)} but this worker's peer "
                f"has {total} parameters: the worker is running a different peer layout"
            )
        peer.adapter.import_parameters(vector.to(next(peer.adapter.parameters()).device))

    def apply_exchange(self, exchange: ParameterExchange) -> None:
        """Load the coordinator's aggregated vector into every local peer and the reference."""
        parameters = decode_tensors(exchange.parameters_b64, exchange.checksum)["parameters"]
        for name, peer in self.peers.items():
            self._load(peer, parameters, name)
            if exchange.reset_optimizers:
                peer.optimizer.state.clear()
        if exchange.reference_b64 and self.reference is not None:
            vector = decode_tensors(exchange.reference_b64, exchange.reference_checksum or "")["reference"]
            self._load(self.reference, vector, "reference")

    # -- the loop ------------------------------------------------------------------------
    def register(self) -> dict[str, Any]:
        info = self.runtime.describe()
        reply = self.client.register(
            WorkerRegistration(
                worker_id=self.worker_id,
                hostname=self.hostname,
                capabilities=info,
                max_peers=self.max_peers,
            )
        )
        self.registered = True
        return reply

    def _heartbeat_loop(self) -> None:
        from ..client.errors import NotFound

        while not self._stop.wait(self.heartbeat_interval):
            try:
                self.client.heartbeat(
                    WorkerHeartbeat(
                        worker_id=self.worker_id,
                        epoch=self._epoch,
                        peers=tuple(self.peers),
                        load=1.0 if self._busy else 0.0,
                        free_memory_bytes=_free_memory(),
                    )
                )
            except NotFound:  # the coordinator restarted: join again under the same id
                try:
                    self.register()
                except Exception:  # noqa: BLE001 - try again on the next beat
                    logger.warning("worker %s could not re-register", self.worker_id)
            except Exception as error:  # noqa: BLE001 - a missed beat is retried, not fatal
                logger.warning("heartbeat failed: %s", error)

    def step(self) -> bool:
        """Poll once; run, report and sync one assignment if there was one. Returns whether it ran."""
        from ..client.errors import Conflict, NotFound

        try:
            assignment = self.client.tasks(self.worker_id, wait=self.poll_wait)
        except NotFound:
            self.register()
            return False
        if assignment is None:
            return False
        self._busy = True
        self._epoch = assignment.epoch
        try:
            try:
                peers = self.build_peers(assignment.peer_names)
            except Exception as error:  # noqa: BLE001 - reported as a failed task
                result = TaskResult(
                    task_id=assignment.task_id, worker_id=self.worker_id, peer_names=assignment.peer_names,
                    parameters_b64="", checksum="", ok=False, error=f"{type(error).__name__}: {error}",
                    epoch=assignment.epoch, run_id=assignment.run_id,
                )
            else:
                result = self.run_epoch(assignment, peers)
            try:
                self.client.results(result)
            except Conflict:  # a replayed submission the coordinator already holds
                pass
            if result.ok:
                exchange = self.client.exchange(self.worker_id, assignment.run_id, assignment.epoch, wait=self.poll_wait)
                if exchange is not None:
                    self.apply_exchange(exchange)
                self.completed_epochs += 1
        finally:
            self._busy = False
        return True

    def run_forever(self) -> None:
        """Register, heartbeat in the background, and serve assignments until :meth:`shutdown`."""
        self.register()
        self._beat = threading.Thread(target=self._heartbeat_loop, name=f"{self.worker_id}-beat", daemon=True)
        self._beat.start()
        try:
            while not self._stop.is_set():
                try:
                    self.step()
                except DistributedError:
                    raise
                except Exception as error:  # noqa: BLE001 - keep serving across transient network errors
                    if self._stop.is_set():
                        break
                    logger.warning("worker loop error: %s", error)
                    self._stop.wait(min(1.0, self.heartbeat_interval))
        finally:
            if self.registered:
                self._leave()

    def start(self) -> "Worker":
        """Run :meth:`run_forever` on a daemon thread."""
        self._thread = threading.Thread(target=self.run_forever, name=self.worker_id, daemon=True)
        self._thread.start()
        return self

    def _leave(self) -> None:
        if self._deregister:
            try:
                self.client.deregister(self.worker_id)
            except Exception:  # noqa: BLE001 - the coordinator may already be gone
                pass
        self.registered = False

    _deregister = True

    def shutdown(self, *, deregister: bool = True, timeout: float = 10.0) -> None:
        """Stop heartbeating and polling. ``deregister=False`` simulates a crash (tests, chaos drills)."""
        self._deregister = deregister
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)


def _free_memory() -> int:
    try:
        if torch.cuda.is_available():
            return int(torch.cuda.mem_get_info()[0])
    except Exception:  # noqa: BLE001 - best effort telemetry
        pass
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))  # type: ignore[attr-defined]
    except (AttributeError, ValueError, OSError):
        return 0
