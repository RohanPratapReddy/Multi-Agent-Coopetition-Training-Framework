"""The coordinator: membership, sharding, the epoch barrier and aggregation (document 05 section 4).

The coordinator owns scoring and aggregation; workers own the forward/backward pass. It is
installed on a run's :class:`~mactf.orchestration.phases.PhaseContext` as ``distributor``;
``local_update`` and ``aggregate`` in the phases hand over to it only while
:meth:`Coordinator.active` is true (at least one live worker). With none, the single-host
path runs untouched.

This module imports no HTTP code: the worker routes (``views.py``) call into it.
"""

from __future__ import annotations

import collections
import dataclasses
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from .. import signals
from ..exceptions import DistributedError, ParameterIntegrityError
from .protocol import (
    ParameterExchange,
    TaskAssignment,
    TaskResult,
    WorkerHeartbeat,
    WorkerRegistration,
)
from .transport import decode_tensors, encode_tensors

if TYPE_CHECKING:  # pragma: no cover
    from ..coopetition.aggregation import AggregateResult, AggregationReport, Aggregator

__all__ = [
    "Coordinator", "RemoteAdapter", "UnknownWorker", "WorkerConflict", "get_coordinator",
    "reset_coordinator",
]

_KEEP_EXCHANGES = 4
_POLL = 0.1


class UnknownWorker(DistributedError):
    """A heartbeat, task poll or result from a worker the coordinator does not know (HTTP 404)."""


class WorkerConflict(DistributedError):
    """A registration that reuses an id from another host, or exceeds ``MAX_WORKERS`` (HTTP 409)."""


class RemoteAdapter:
    """Exposes a worker-provided vector as a peer adapter for :meth:`Aggregator.aggregate`.

    It lives here, not in ``mactf.coopetition``, because it exists only for distribution.
    ``module`` is the coordinator's own copy of the peer (or an empty module), which is where
    non-parameter buffers come from: workers ship parameters only.
    """

    def __init__(
        self,
        vector: torch.Tensor,
        *,
        shapes: Sequence[tuple[str, tuple[int, ...]]] | None = None,
        module: nn.Module | None = None,
    ) -> None:
        self._vector = vector.detach().clone()
        self._shapes = [(str(n), tuple(s)) for n, s in (shapes or [])]
        self.module = module if module is not None else nn.Module()

    def export_parameters(self) -> torch.Tensor:
        return self._vector.clone()

    def parameter_shapes(self) -> list[tuple[str, tuple[int, ...]]]:
        return list(self._shapes)

    def import_parameters(self, flat: torch.Tensor, *, strict: bool = True) -> int:
        self._vector = flat.detach().clone()
        return int(flat.numel())


@dataclass
class _Member:
    registration: WorkerRegistration
    registered_at: float
    last_seen: float
    epoch: int = -1
    peers: tuple[str, ...] = ()
    load: float = 0.0
    free_memory_bytes: int = 0
    status: str = "active"  # active | suspect

    def describe(self, now: float) -> dict[str, Any]:
        return {
            "worker_id": self.registration.worker_id,
            "hostname": self.registration.hostname,
            "capabilities": dict(self.registration.capabilities),
            "max_peers": self.registration.max_peers,
            "status": self.status,
            "epoch": self.epoch,
            "peers": list(self.peers),
            "load": self.load,
            "free_memory_bytes": self.free_memory_bytes,
            "seconds_since_heartbeat": round(max(0.0, now - self.last_seen), 3),
        }


@dataclass
class _Task:
    assignment: TaskAssignment
    worker_id: str
    expected: dict[str, tuple[int, ...]]
    state: str = "queued"  # queued | running | done | rejected | lost


@dataclass
class _EpochState:
    """What one epoch's dispatch produced, consumed by the aggregation phase."""

    run_id: str
    epoch: int
    vectors: dict[str, torch.Tensor] = field(default_factory=dict)
    results: list[TaskResult] = field(default_factory=list)
    missing: dict[str, list[str]] = field(default_factory=dict)  # worker_id -> peers
    rejected: dict[str, str] = field(default_factory=dict)  # worker_id -> reason


class Coordinator:
    def __init__(
        self,
        *,
        settings: Any,
        registry: Any = None,
        heartbeat_timeout: float | None = None,
        max_workers: int | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        block = dict(settings.get("DISTRIBUTED") or {}) if settings is not None else {}
        self.settings = settings
        self.registry = registry
        self.heartbeat_timeout = float(
            heartbeat_timeout if heartbeat_timeout is not None else block.get("WORKER_HEARTBEAT_TIMEOUT", 30.0)
        )
        self.max_workers = int(max_workers if max_workers is not None else block.get("MAX_WORKERS", 16))
        self.task_timeout = float(block.get("WORKER_TASK_TIMEOUT", 600.0))
        self.compress = bool(block.get("COMPRESS", True))
        self._clock = clock
        self._cv = threading.Condition()
        self._members: dict[str, _Member] = {}
        self._queues: dict[str, collections.deque[str]] = {}
        self._tasks: dict[str, _Task] = {}
        self._results: dict[tuple[int, str], TaskResult] = {}
        self._decoded: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
        self._exchanges: collections.OrderedDict[tuple[str, int], ParameterExchange] = collections.OrderedDict()
        self._epochs: dict[tuple[str, int], _EpochState] = {}
        self.current_run: str | None = None
        self.layout: list[tuple[str, tuple[int, ...]]] | None = None

    # -- membership ----------------------------------------------------------------
    def register(self, registration: WorkerRegistration) -> Mapping[str, Any]:
        if not registration.worker_id.strip():
            raise DistributedError("worker_id must be a non-empty string")
        if registration.max_peers < 1:
            raise DistributedError("max_peers must be at least 1")
        now = self._clock()
        with self._cv:
            self._reap_locked(now)
            existing = self._members.get(registration.worker_id)
            if existing is not None and existing.registration.hostname != registration.hostname:
                raise WorkerConflict(
                    f"worker id {registration.worker_id!r} is already registered from host "
                    f"{existing.registration.hostname!r}; a spoofed or recycled identity is refused"
                )
            if existing is None and len(self._members) >= self.max_workers:
                raise WorkerConflict(
                    f"the coordinator already has {self.max_workers} workers (DISTRIBUTED.MAX_WORKERS)"
                )
            if existing is not None:  # a restarted worker: whatever it held is gone
                self._drop_tasks_locked(registration.worker_id, state="lost")
            self._members[registration.worker_id] = _Member(registration, now, now)
            self._queues[registration.worker_id] = collections.deque()
            self._cv.notify_all()
            return {
                "worker_id": registration.worker_id,
                "heartbeat_timeout": self.heartbeat_timeout,
                "task_timeout": self.task_timeout,
                "compress": self.compress,
                "run_id": self.current_run,
                "layout": [[name, list(shape)] for name, shape in (self.layout or [])],
                "peers": sorted(self._configured_peers()),
                "members": len(self._members),
            }

    def heartbeat(self, heartbeat: WorkerHeartbeat) -> Mapping[str, Any]:
        now = self._clock()
        with self._cv:
            self._reap_locked(now)
            member = self._members.get(heartbeat.worker_id)
            if member is None:
                raise UnknownWorker(f"worker {heartbeat.worker_id!r} is not registered; register again")
            member.last_seen = now
            member.epoch = heartbeat.epoch
            member.peers = tuple(heartbeat.peers)
            member.load = heartbeat.load
            member.free_memory_bytes = heartbeat.free_memory_bytes
            return {"ok": True, "run_id": self.current_run, "members": len(self._members), "status": member.status}

    def deregister(self, worker_id: str) -> None:
        with self._cv:
            if self._members.pop(worker_id, None) is None:
                raise UnknownWorker(f"worker {worker_id!r} is not registered")
            self._drop_tasks_locked(worker_id, state="lost")
            self._queues.pop(worker_id, None)
            self._cv.notify_all()

    def members(self) -> list[Mapping[str, Any]]:
        now = self._clock()
        with self._cv:
            self._reap_locked(now)
            return [self._members[key].describe(now) for key in sorted(self._members)]

    def reap_stale(self) -> list[str]:
        with self._cv:
            return self._reap_locked(self._clock())

    def has_epoch(self, context: Any) -> bool:
        """True when this run's epoch was dispatched to workers and awaits aggregation."""
        return (context.run_id, int(context.epoch)) in self._epochs

    def active(self) -> bool:
        """True while at least one live worker is registered: the switch into distributed mode."""
        with self._cv:
            self._reap_locked(self._clock())
            return any(member.status == "active" for member in self._members.values())

    def _reap_locked(self, now: float) -> list[str]:
        stale = [
            worker_id
            for worker_id, member in self._members.items()
            if now - member.last_seen > self.heartbeat_timeout
        ]
        for worker_id in stale:
            self._members.pop(worker_id, None)
            self._drop_tasks_locked(worker_id, state="lost")
            self._queues.pop(worker_id, None)
        if stale:
            self._cv.notify_all()
        return stale

    def _drop_tasks_locked(self, worker_id: str, *, state: str) -> None:
        for task in self._tasks.values():
            if task.worker_id == worker_id and task.state in ("queued", "running"):
                task.state = state

    def _configured_peers(self) -> list[str]:
        try:
            return [str(entry["name"]) for entry in (self.settings.get("PEERS") or [])]
        except Exception:  # noqa: BLE001 - a malformed PEERS block is reported elsewhere
            return []

    # -- epoch orchestration --------------------------------------------------------
    def plan(self, epoch: int, peers: Sequence[str]) -> dict[str, list[str]]:
        """Assign ``peers`` to live workers in contiguous, balanced blocks sorted by worker id.

        Three peers over two workers give ``{w1: [p1, p2], w2: [p3]}``. A worker never gets more
        than its ``max_peers``; the shares are a pure function of the membership, so two
        coordinators with the same view produce the same plan.
        """
        with self._cv:
            self._reap_locked(self._clock())
            live = [
                (worker_id, member.registration.max_peers)
                for worker_id, member in sorted(self._members.items())
                if member.status == "active"
            ]
        names = list(peers)
        capacity = sum(cap for _, cap in live)
        if not live or capacity < len(names):
            raise DistributedError(
                f"{len(names)} peer(s) cannot be placed on {len(live)} worker(s) with total capacity "
                f"{capacity}; unassigned peers: {', '.join(names[capacity:]) or ', '.join(names)}"
            )
        base, extra = divmod(len(names), len(live))
        shares = [min(cap, base + (1 if index < extra else 0)) for index, (_, cap) in enumerate(live)]
        remaining = len(names) - sum(shares)
        index = 0
        while remaining > 0:  # a small cap pushed its share onto the others
            if shares[index % len(live)] < live[index % len(live)][1]:
                shares[index % len(live)] += 1
                remaining -= 1
            index += 1
        plan: dict[str, list[str]] = {}
        cursor = 0
        for (worker_id, _), share in zip(live, shares):
            if share:
                plan[worker_id] = names[cursor : cursor + share]
                cursor += share
        return plan

    def submit(
        self,
        worker_id: str,
        assignment: TaskAssignment,
        *,
        expected: Mapping[str, tuple[int, ...]] | None = None,
    ) -> None:
        """Queue ``assignment`` for ``worker_id``; ``expected`` maps each peer to its vector shape."""
        with self._cv:
            if worker_id not in self._members:
                raise UnknownWorker(f"worker {worker_id!r} is not registered")
            task = _Task(assignment=assignment, worker_id=worker_id, expected=dict(expected or {}))
            self._tasks[assignment.task_id] = task
            self._queues.setdefault(worker_id, collections.deque()).append(assignment.task_id)
            self._cv.notify_all()

    def next_task(self, worker_id: str, *, wait: float = 20.0) -> TaskAssignment | None:
        """Long-poll: the next queued assignment for ``worker_id``, or ``None`` after ``wait`` seconds."""
        deadline = self._clock() + max(0.0, wait)
        with self._cv:
            while True:
                now = self._clock()
                self._reap_locked(now)
                member = self._members.get(worker_id)
                if member is None:
                    raise UnknownWorker(f"worker {worker_id!r} is not registered")
                member.last_seen = now  # polling is proof of life
                queue = self._queues.get(worker_id)
                while queue:
                    task = self._tasks.get(queue.popleft())
                    if task is not None and task.state == "queued":
                        task.state = "running"
                        return task.assignment
                if now >= deadline:
                    return None
                self._cv.wait(timeout=min(_POLL * 5, deadline - now))

    def submit_result(self, result: TaskResult) -> str:
        """Validate and store one result; return ``accepted`` or raise after recording the rejection."""
        with self._cv:
            task = self._tasks.get(result.task_id)
            if task is None:
                raise DistributedError(f"unknown task {result.task_id!r}")
            if task.worker_id != result.worker_id:
                raise DistributedError(f"task {result.task_id!r} was not assigned to worker {result.worker_id!r}")
            if task.state not in ("running", "queued"):
                raise DistributedError(f"task {result.task_id!r} is already {task.state}")
            member = self._members.get(result.worker_id)
            reason: str | None = None
            decoded: dict[str, torch.Tensor] = {}
            if not result.ok:
                reason = f"worker reported failure: {result.error or 'unspecified'}"
            elif tuple(result.peer_names) != tuple(task.assignment.peer_names):
                reason = (
                    f"returned peers {list(result.peer_names)} but was assigned "
                    f"{list(task.assignment.peer_names)}"
                )
                if member is not None:
                    member.status = "suspect"
            else:
                try:
                    decoded = decode_tensors(
                        result.parameters_b64,
                        result.checksum,
                        expected=[(name, task.expected[name]) for name in task.assignment.peer_names if name in task.expected],
                    )
                    missing = [name for name in task.assignment.peer_names if name not in decoded]
                    if missing:
                        reason = f"result lacks parameters for {missing}"
                except ParameterIntegrityError as error:
                    reason = str(error)
            epoch = task.assignment.epoch
            if reason is not None:
                task.state = "rejected"
                state = self._epochs.get((task.assignment.run_id, epoch))
                if state is not None:
                    state.rejected[result.worker_id] = reason
                self._cv.notify_all()
                incident = (result.worker_id, list(task.assignment.peer_names), reason)
            else:
                task.state = "done"
                self._results[(epoch, result.task_id)] = result
                self._decoded[(epoch, result.task_id)] = decoded
                self._cv.notify_all()
                incident = None
        if incident is not None:
            signals.integrity_incident.send(
                sender=self,
                reason=f"worker {incident[0]} result rejected: {incident[2]}",
                workers=[incident[0]],
                peers=incident[1],
            )
            raise DistributedError(f"result rejected: {incident[2]}")
        return "accepted"

    def wait_for_results(
        self, epoch: int, expected: int, *, timeout: float, run_id: str | None = None
    ) -> list[TaskResult]:
        """Block until ``expected`` tasks of ``epoch`` are settled (done, rejected or lost) or time runs out.

        On timeout every worker still holding a task is reaped and its peers are reported
        missing; the epoch proceeds with what arrived. Returns the accepted results.
        """
        deadline = self._clock() + timeout
        with self._cv:
            while True:
                now = self._clock()
                self._reap_locked(now)
                mine = [
                    t for t in self._tasks.values()
                    if t.assignment.epoch == epoch and (run_id is None or t.assignment.run_id == run_id)
                ]
                settled = [t for t in mine if t.state in ("done", "rejected", "lost")]
                if len(settled) >= expected:
                    break
                if now >= deadline:
                    for task in mine:
                        if task.state in ("queued", "running"):
                            task.state = "lost"
                            self._members.pop(task.worker_id, None)
                            self._queues.pop(task.worker_id, None)
                    self._cv.notify_all()
                    break
                self._cv.wait(timeout=min(_POLL, max(0.0, deadline - now)))
            return [
                self._results[(epoch, task.assignment.task_id)]
                for task in mine
                if (epoch, task.assignment.task_id) in self._results
            ]

    # -- the distributed phases ------------------------------------------------------
    def local_update(
        self,
        context: Any,
        *,
        loss_config: Any = None,
        max_grad_norm: float | None = None,
    ) -> dict[str, dict[str, float]]:
        """Dispatch the epoch's local updates to workers and gather their parameter vectors."""
        from ..training_utils import state_targets_for

        signals.phase_started.send(sender=context, phase="local_update")
        signal_map = context.signal_by_peer()
        trained = [
            name
            for name in context.peers
            if name in signal_map and any(name in card.peers for card in context.scorecards)
        ]
        states, targets = state_targets_for(context)
        epoch = int(context.epoch)
        state = _EpochState(run_id=context.run_id, epoch=epoch)
        self._epochs[(context.run_id, epoch)] = state
        self.current_run = context.run_id
        shapes = next(iter(context.peers.values())).adapter.parameter_shapes() if context.peers else []
        self.layout = [(str(n), tuple(s)) for n, s in shapes]

        statistics: dict[str, dict[str, float]] = {}
        if states is None or not trained:
            context.metrics["local_update"] = statistics
            signals.phase_finished.send(sender=context, phase="local_update", statistics=statistics)
            return statistics

        self._purge(context.run_id, epoch)
        plan = self.plan(epoch, trained)
        loss = dataclasses.asdict(loss_config) if loss_config is not None else {}
        for worker_id, names in plan.items():
            cards = {n: [c.peers[n] for c in context.scorecards if n in c.peers] for n in names}
            tensors: dict[str, torch.Tensor] = {"states": states.detach().cpu(), "targets": targets.detach().cpu()}
            for name in names:
                tensors[f"parameters::{name}"] = context.peers[name].adapter.export_parameters().detach().cpu()
            tensors["reference"] = context.reference.adapter.export_parameters().detach().cpu()
            payload_b64, checksum = encode_tensors(tensors, compress=self.compress)
            assignment = TaskAssignment(
                run_id=context.run_id,
                epoch=epoch,
                task_id=uuid.uuid4().hex,
                states_shape=tuple(states.shape),
                targets_shape=tuple(targets.shape),
                peer_names=tuple(names),
                payload_b64=payload_b64,
                timeout=self.task_timeout,
                checksum=checksum,
                config={
                    "loss": loss,
                    "max_grad_norm": max_grad_norm,
                    "signals": {n: float(signal_map[n]) for n in names},
                    "void": {n: bool(cards[n][0].is_void) for n in names},
                    "floor": {n: sum(c.floor for c in cards[n]) / max(1, len(cards[n])) for n in names},
                },
            )
            self.submit(
                worker_id,
                assignment,
                expected={name: (int(context.peers[name].adapter.export_parameters().numel()),) for name in names},
            )

        results = self.wait_for_results(epoch, len(plan), timeout=self.task_timeout, run_id=context.run_id)
        with self._cv:
            for task in self._tasks.values():
                if task.assignment.epoch == epoch and task.assignment.run_id == context.run_id and task.state == "lost":
                    state.missing[task.worker_id] = list(task.assignment.peer_names)
            for result in results:
                if result.run_id not in ("", context.run_id):
                    continue
                for name, vector in self._decoded[(epoch, result.task_id)].items():
                    state.vectors[name] = vector
                state.results.append(result)
                for key, value in result.metrics.items():
                    peer, _, metric = key.partition(".")
                    if metric:
                        statistics.setdefault(peer, {})[metric] = float(value)
            # Peers the single-host loop would also skip keep their coordinator-side vector.
            for name, peer in context.peers.items():
                if name not in trained:
                    state.vectors[name] = peer.adapter.export_parameters().detach().cpu()
            missing_peers = sorted({p for peers in state.missing.values() for p in peers})
            rejected = dict(state.rejected)
        if state.missing:
            signals.integrity_incident.send(
                sender=self,
                reason=(
                    f"epoch {epoch}: worker(s) {', '.join(sorted(state.missing))} did not return; "
                    f"peer(s) {', '.join(missing_peers)} contribute zero weight"
                ),
                workers=sorted(state.missing),
                peers=missing_peers,
            )
        context.metrics["distributed"] = {
            "epoch": epoch,
            "workers": sorted(plan),
            "missing_workers": sorted(state.missing),
            "missing_peers": missing_peers,
            "rejected": rejected,
        }
        context.metrics["local_update"] = statistics
        signals.phase_finished.send(sender=context, phase="local_update", statistics=statistics)
        return statistics

    def _purge(self, run_id: str, epoch: int) -> None:
        """Forget settled tasks of this run up to ``epoch`` so a re-run epoch starts from a clean barrier."""
        with self._cv:
            for task_id, task in list(self._tasks.items()):
                a = task.assignment
                if a.run_id == run_id and a.epoch <= epoch and task.state in ("done", "rejected", "lost"):
                    del self._tasks[task_id]
                    self._results.pop((a.epoch, task_id), None)
                    self._decoded.pop((a.epoch, task_id), None)

    def aggregate(
        self,
        results: Sequence[TaskResult],
        report: "AggregationReport",
        *,
        extra: Mapping[str, torch.Tensor] | None = None,
        reference: Any = None,
        modules: Mapping[str, nn.Module] | None = None,
        aggregator: "Aggregator | None" = None,
    ) -> "AggregateResult":
        """Weighted sum of worker vectors under ``report``; the coordinator, never a worker, weighs them."""
        aggregator = aggregator or self._aggregator
        if aggregator is None:
            raise DistributedError("no aggregator is bound; pass one or call bind()")
        vectors: dict[str, torch.Tensor] = dict(extra or {})
        for result in results:
            task = self._tasks.get(result.task_id)
            decoded = self._decoded.get((task.assignment.epoch, result.task_id)) if task is not None else None
            if decoded is None:
                decoded = decode_tensors(result.parameters_b64, result.checksum)
            vectors.update(decoded)
        adapters = {
            name: RemoteAdapter(vector, module=(modules or {}).get(name)) for name, vector in vectors.items()
        }
        return aggregator.aggregate(adapters, report, reference=reference or self._reference)

    _aggregator: Any = None
    _reference: Any = None

    def bind(self, *, aggregator: Any = None, reference: Any = None) -> None:
        """The aggregator (and reference adapter) :meth:`aggregate` uses when none is passed."""
        self._aggregator = aggregator
        self._reference = reference

    def aggregate_epoch(
        self,
        context: Any,
        aggregator: "Aggregator",
        *,
        reset_optimizers: bool = True,
        reference_ema: float = 0.05,
    ) -> "AggregationReport":
        """Phase 5 for a distributed epoch: weight, aggregate, apply on the host, publish to workers."""
        from ..orchestration.phases import _peer_confidence, _peer_void

        signals.phase_started.send(sender=context, phase="aggregation")
        state = self._epochs.pop((context.run_id, int(context.epoch)), None)
        if state is None:
            raise DistributedError(f"no distributed local update was run for epoch {context.epoch}")
        names = list(context.peers)
        signal_map = context.signal_by_peer()
        scores = [signal_map.get(name, 0.0) for name in names]
        # A peer whose host dropped out contributes zero weight, never a default (framework.md 10.1.4).
        eligible = [name in state.vectors and not _peer_void(context, name) for name in names]
        confidence = [_peer_confidence(context, name) for name in names]
        report = aggregator.compute_weights(names, scores, eligible=eligible, confidence=confidence)
        context.reports.append(report)

        adapters = {name: context.peers[name].adapter for name in names}
        result = self.aggregate(
            state.results,
            report,
            extra={n: v for n, v in state.vectors.items() if n not in {p for r in state.results for p in r.peer_names}},
            reference=context.reference.adapter,
            modules={name: adapters[name].module for name in names},
            aggregator=aggregator,
        )
        aggregator.apply(
            adapters,
            result,
            reference=context.reference.adapter,
            reference_ema=reference_ema,
            reset_optimizers=reset_optimizers,
            optimizers={name: peer.optimizer for name, peer in context.peers.items()},
        )
        self._publish(context, result.parameters, report, reset_optimizers=reset_optimizers)
        context.metrics["aggregation"] = report.as_dict()
        signals.phase_finished.send(sender=context, phase="aggregation", report=report)
        return report

    def _publish(self, context: Any, parameters: torch.Tensor, report: Any, *, reset_optimizers: bool) -> None:
        params_b64, params_sum = encode_tensors({"parameters": parameters.detach().cpu()}, compress=self.compress)
        reference_vector = context.reference.adapter.export_parameters().detach().cpu()
        ref_b64, ref_sum = encode_tensors({"reference": reference_vector}, compress=self.compress)
        exchange = ParameterExchange(
            run_id=context.run_id,
            epoch=int(context.epoch),
            parameters_b64=params_b64,
            checksum=params_sum,
            reference_b64=ref_b64,
            kappa=float(report.kappa),
            reference_checksum=ref_sum,
            reset_optimizers=reset_optimizers,
        )
        with self._cv:
            self._exchanges[(context.run_id, int(context.epoch))] = exchange
            while len(self._exchanges) > _KEEP_EXCHANGES:
                self._exchanges.popitem(last=False)
            self._cv.notify_all()

    def exchange(self, run_id: str, epoch: int, *, wait: float = 20.0) -> ParameterExchange | None:
        """The aggregated vector of ``(run_id, epoch)``, waiting up to ``wait`` seconds for it to exist."""
        deadline = self._clock() + max(0.0, wait)
        with self._cv:
            while True:
                found = self._exchanges.get((run_id, epoch))
                if found is not None:
                    return found
                now = self._clock()
                if now >= deadline:
                    return None
                self._cv.wait(timeout=min(_POLL * 5, deadline - now))


# -- the process-wide coordinator ------------------------------------------------------
_LOCK = threading.Lock()
_COORDINATOR: Coordinator | None = None


def get_coordinator(settings: Any = None) -> Coordinator:
    """The one coordinator of this process, built lazily from the current settings."""
    global _COORDINATOR
    with _LOCK:
        if _COORDINATOR is None:
            if settings is None:
                from ..conf import get_settings

                settings = get_settings()
            _COORDINATOR = Coordinator(settings=settings)
        return _COORDINATOR


def reset_coordinator() -> None:
    global _COORDINATOR
    with _LOCK:
        _COORDINATOR = None
