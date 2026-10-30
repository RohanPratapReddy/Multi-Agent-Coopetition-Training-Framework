"""Live runs: one thread per run, an event bus, and the run state machine.

``RunManager`` owns the threads that drive :class:`~mactf.orchestration.orchestrator.Orchestrator`
and the transitions of ``runs.status``; ``EventBus`` carries what happens inside a
run to whoever is watching (the SSE view, the client iterator, a test). Neither
imports ``mactf.http``: they are as usable from the CLI as from a request handler.

The concurrency model is plain threads (document 01 section 3):

* one daemon thread per run, so a request never occupies a worker for a whole run;
* cancellation is cooperative, polled by the orchestrator at epoch boundaries;
* the bus is a bounded ring buffer guarded by one ``Condition``; subscribers block
  on it with a heartbeat timeout, so there is no event loop anywhere.
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import enum
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from .. import signals
from ..conf import Settings
from ..db import TERMINAL_STATUSES, Registry, ensure_registry
from ..exceptions import ImproperlyConfigured, MACTFError
from .orchestrator import Orchestrator, RunConfig

logger = logging.getLogger("mactf.live")

__all__ = [
    "CapacityExceeded",
    "EVENT_NAMES",
    "EventBus",
    "RunHandle",
    "RunManager",
    "RunExists",
    "RunNotFound",
    "RunState",
    "RunStateError",
    "Subscription",
    "jsonable",
]

#: The fixed event vocabulary of document 02 section 3.4.
EVENT_NAMES = (
    "run", "phase", "epoch", "warning", "incident", "checkpoint", "rollback",
    "terminal", "heartbeat",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class RunNotFound(LookupError):
    """No run with this id, live or stored."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"no run {run_id!r}")
        self.run_id = run_id


class RunStateError(MACTFError):
    """A requested transition is not in the state graph (HTTP 409)."""

    def __init__(self, run_id: str, current: str, action: str) -> None:
        super().__init__(f"cannot {action} run {run_id!r}: it is {current}")
        self.run_id = run_id
        self.current_status = current
        self.action = action


class RunExists(MACTFError):
    """``start`` was given a ``run_id`` that is already taken (HTTP 409)."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"run id {run_id!r} is already in use")
        self.run_id = run_id


class CapacityExceeded(MACTFError):
    """``MAX_CONCURRENT_RUNS`` is reached (HTTP 429 with ``Retry-After``)."""

    def __init__(self, limit: int, retry_after: float = 5.0) -> None:
        super().__init__(f"{limit} run(s) already active; retry in {retry_after:g}s")
        self.limit = limit
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
class RunState(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    RESUMING = "resuming"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DRY_RUN = "dry_run"
    INTERRUPTED = "interrupted"

    @property
    def terminal(self) -> bool:
        return self.value in TERMINAL_STATUSES

    @property
    def live(self) -> bool:
        """Occupying (or about to occupy) a run thread."""
        return self in (RunState.QUEUED, RunState.RUNNING, RunState.RESUMING)

    def can_become(self, target: "RunState") -> bool:
        return target in _TRANSITIONS.get(self, frozenset())


_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.QUEUED: frozenset({RunState.RUNNING, RunState.FAILED, RunState.CANCELLED}),
    RunState.RUNNING: frozenset(
        {RunState.COMPLETED, RunState.CANCELLED, RunState.FAILED, RunState.PAUSED, RunState.DRY_RUN}
    ),
    RunState.PAUSED: frozenset({RunState.RUNNING, RunState.RESUMING, RunState.CANCELLED}),
    RunState.RESUMING: frozenset({RunState.RUNNING, RunState.FAILED, RunState.CANCELLED}),
    RunState.INTERRUPTED: frozenset({RunState.RESUMING}),
    RunState.COMPLETED: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.CANCELLED: frozenset(),
    RunState.DRY_RUN: frozenset(),
}


# ---------------------------------------------------------------------------
# JSON coercion
# ---------------------------------------------------------------------------
def jsonable(value: Any, *, _depth: int = 0) -> Any:
    """Coerce ``value`` to something ``json.dumps`` accepts, never raising."""
    if _depth > 8:
        return str(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, enum.Enum):
        return jsonable(value.value, _depth=_depth + 1)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v, _depth=_depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset, collections.deque)):
        return [jsonable(v, _depth=_depth + 1) for v in value]
    for attribute in ("to_dict", "as_dict", "describe"):
        method = getattr(value, attribute, None)
        if callable(method):
            try:
                return jsonable(method(), _depth=_depth + 1)
            except Exception:  # noqa: BLE001 - fall through to the next strategy
                continue
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value), _depth=_depth + 1)
    item = getattr(value, "item", None)
    if callable(item):  # numpy / torch scalars
        try:
            return jsonable(item(), _depth=_depth + 1)
        except Exception:  # noqa: BLE001
            pass
    return str(value)


# ---------------------------------------------------------------------------
# Event bus
# ---------------------------------------------------------------------------
class _Channel:
    __slots__ = ("events", "next_id", "closed", "terminal")

    def __init__(self, capacity: int) -> None:
        self.events: collections.deque[dict[str, Any]] = collections.deque(maxlen=capacity)
        self.next_id = 1
        self.closed = False
        self.terminal: dict[str, Any] | None = None


class EventBus:
    """A per-run ring buffer with blocking subscriptions.

    ``publish`` appends one JSON-safe event and wakes every subscriber. Ids are
    monotonic per run and never reused, even across a pause/resume, so
    ``Last-Event-ID`` stays meaningful for the life of the run.
    """

    def __init__(self, capacity: int = 1024, *, heartbeat: float = 15.0, max_runs: int = 64) -> None:
        if capacity < 1:
            raise ImproperlyConfigured("EventBus capacity must be >= 1")
        if heartbeat <= 0:
            raise ImproperlyConfigured("EventBus heartbeat must be positive")
        self.capacity = int(capacity)
        self.heartbeat = float(heartbeat)
        self.max_runs = int(max_runs)
        self._channels: dict[str, _Channel] = {}
        self._cond = threading.Condition(threading.Lock())

    @classmethod
    def from_settings(cls, settings: Any) -> "EventBus":
        api = settings.get("API") or {}
        http = settings.get("HTTP") or {}
        return cls(
            int(api.get("EVENT_BUFFER", 1024)), heartbeat=float(http.get("SSE_HEARTBEAT", 15.0))
        )

    # -- producing ----------------------------------------------------------
    def publish(self, run_id: str, event: str, data: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if event not in EVENT_NAMES:
            raise ValueError(f"unknown event name {event!r}; expected one of {EVENT_NAMES}")
        payload = jsonable(dict(data or {}))
        payload.setdefault("run_id", run_id)
        with self._cond:
            channel = self._channels.get(run_id)
            if channel is None:
                channel = self._channels[run_id] = _Channel(self.capacity)
                self._evict_locked(keep=run_id)
            elif channel.closed and event != "terminal":
                channel.closed = False  # a resumed run reopens its stream
                channel.terminal = None
            stamped = {"id": channel.next_id, "event": event, "data": payload}
            channel.next_id += 1
            channel.events.append(stamped)
            if event == "terminal":
                channel.closed = True
                channel.terminal = stamped
            self._cond.notify_all()
        return stamped

    def _evict_locked(self, *, keep: str) -> None:
        if len(self._channels) <= self.max_runs:
            return
        for run_id in list(self._channels):
            if run_id != keep and self._channels[run_id].closed:
                del self._channels[run_id]
                if len(self._channels) <= self.max_runs:
                    return

    def close(self, run_id: str) -> None:
        """End every subscription to ``run_id`` without a terminal event."""
        with self._cond:
            channel = self._channels.get(run_id)
            if channel is not None:
                channel.closed = True
            self._cond.notify_all()

    # -- reading --------------------------------------------------------------
    def has(self, run_id: str) -> bool:
        with self._cond:
            return run_id in self._channels

    def history(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        with self._cond:
            channel = self._channels.get(run_id)
            return [e for e in channel.events if e["id"] > after] if channel else []

    def latest_id(self, run_id: str) -> int:
        with self._cond:
            channel = self._channels.get(run_id)
            return channel.next_id - 1 if channel else 0

    def oldest_id(self, run_id: str) -> int | None:
        with self._cond:
            channel = self._channels.get(run_id)
            return channel.events[0]["id"] if channel and channel.events else None

    def terminal_event(self, run_id: str) -> dict[str, Any] | None:
        with self._cond:
            channel = self._channels.get(run_id)
            return channel.terminal if channel else None

    def subscribe(self, run_id: str, last_id: int | None = None) -> "Subscription":
        """Replay everything after ``last_id`` (all of the buffer if ``None``), then follow."""
        return Subscription(self, run_id, last_id)


class Subscription:
    """A blocking cursor over one run's events. Iterable; ends after ``terminal``."""

    def __init__(self, bus: EventBus, run_id: str, last_id: int | None) -> None:
        self.bus = bus
        self.run_id = run_id
        self._cursor = int(last_id) if last_id is not None else 0
        self._lost: int | None = None
        self._done = False
        self._cancelled = False
        if last_id is not None:
            oldest = bus.oldest_id(run_id)
            # Events (last_id, oldest) were dropped from the ring: say so, once.
            if oldest is not None and int(last_id) < oldest - 1:
                self._lost = oldest
                self._cursor = oldest - 1

    def cancel(self) -> None:
        """Wake a blocked ``next`` and end the iteration (client disconnect)."""
        with self.bus._cond:
            self._cancelled = True
            self.bus._cond.notify_all()

    close = cancel

    @property
    def done(self) -> bool:
        return self._done

    def _heartbeat(self) -> dict[str, Any]:
        return {"id": None, "event": "heartbeat", "data": {"run_id": self.run_id, "ts": time.time()}}

    def next(self, timeout: float | None = None) -> dict[str, Any] | None:
        """The next event; a ``heartbeat`` after ``timeout`` (default: the bus
        heartbeat) of silence; ``None`` once the stream has ended."""
        wait_for = self.bus.heartbeat if timeout is None else float(timeout)
        deadline = time.monotonic() + wait_for
        cond = self.bus._cond
        with cond:
            while True:
                if self._cancelled or self._done:
                    return None
                if self._lost is not None:
                    oldest, self._lost = self._lost, None
                    return {"id": None, "event": "stream_lost", "data": {"oldest": oldest, "run_id": self.run_id}}
                channel = self.bus._channels.get(self.run_id)
                if channel is not None and channel.events:
                    first = channel.events[0]["id"]
                    index = max(0, self._cursor + 1 - first)
                    if index < len(channel.events):
                        event = channel.events[index]
                        self._cursor = event["id"]
                        if event["event"] == "terminal":
                            self._done = True
                        return event
                if channel is not None and channel.closed:
                    self._done = True
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._heartbeat()
                cond.wait(remaining)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        while True:
            event = self.next()
            if event is None:
                return
            yield event


# ---------------------------------------------------------------------------
# Run handle and manager
# ---------------------------------------------------------------------------
class RunHandle:
    """What the manager knows about one run it started."""

    def __init__(self, run_id: str, config: RunConfig) -> None:
        self.run_id = run_id
        self.config = config
        self.state = RunState.QUEUED
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.error: str | None = None
        self.summary: dict[str, Any] | None = None
        self.epochs_completed = 0
        self.thread: threading.Thread | None = None
        self.orchestrator: Orchestrator | None = None
        #: Real components a caller attached (``trainer``, ``logger``, ``watchdog``,
        #: ``checkpoint``); the API's observability routes read them.
        self.components: dict[str, Any] = {}
        self.done = threading.Event()
        self._cancel = threading.Event()
        self._pause = threading.Event()
        self.lock = threading.RLock()

    @property
    def status(self) -> str:
        return self.state.value

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    @property
    def pause_requested(self) -> bool:
        return self._pause.is_set()

    def should_stop(self) -> bool:
        return self._cancel.is_set() or self._pause.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the run's thread ends; ``True`` if it did."""
        return self.done.wait(timeout)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "epochs_completed": self.epochs_completed,
            "error": self.error,
        }

    def __repr__(self) -> str:
        return f"<RunHandle {self.run_id} {self.status}>"


_REQUEST_FIELDS = {f.name for f in dataclasses.fields(RunConfig)}


def _config_from(request: RunConfig | Mapping[str, Any] | None) -> RunConfig:
    if request is None:
        return RunConfig()
    if isinstance(request, RunConfig):
        return copy.copy(request)
    unknown = sorted(set(request) - _REQUEST_FIELDS)
    if unknown:
        raise ImproperlyConfigured(
            f"unknown run option(s) {unknown}; expected {sorted(_REQUEST_FIELDS)}"
        )
    values = dict(request)
    resume = values.get("resume_from")
    if isinstance(resume, str) and resume:
        values["resume_from"] = Path(resume)
    return RunConfig(**values)


class RunManager:
    """Starts, watches, cancels and resumes runs, one thread each."""

    def __init__(
        self,
        settings: Settings,
        *,
        registry: Registry | None = None,
        bus: EventBus | None = None,
        modules: Mapping[str, Any] | None = None,
        orchestrator_factory: Callable[[Settings, Registry], Orchestrator] | None = None,
        max_concurrent: int | None = None,
        cancel_grace: float | None = None,
    ) -> None:
        api = settings.get("API") or {}
        self.settings = settings
        self.registry = registry or ensure_registry(settings)
        self.bus = bus or EventBus.from_settings(settings)
        self.modules = dict(modules) if modules else None
        self._factory = orchestrator_factory or self._default_factory
        if max_concurrent is None:
            configured = api.get("MAX_CONCURRENT_RUNS")
            max_concurrent = int(configured) if configured else None
        self.max_concurrent = max_concurrent
        self.cancel_grace = float(
            cancel_grace if cancel_grace is not None else api.get("CANCEL_GRACE_SECONDS", 10.0)
        )
        self._slots = (
            threading.BoundedSemaphore(max_concurrent) if max_concurrent else None
        )
        self._handles: dict[str, RunHandle] = {}
        self._by_thread: dict[int, str] = {}
        self._lock = threading.RLock()
        self._connected: list[tuple[signals.Signal, Callable[..., Any]]] = []
        self._connect_signals()

    # -- construction helpers ------------------------------------------------
    def _default_factory(self, settings: Settings, registry: Registry) -> Orchestrator:
        return Orchestrator(settings, modules=self.modules, registry=registry)

    def _run_settings(self) -> Settings:
        """A private copy: ``Orchestrator.prepare`` writes DEVICE/SEED into settings."""
        return Settings(self.settings.as_dict(), module=self.settings.module)

    # -- signals -> events --------------------------------------------------
    def _connect_signals(self) -> None:
        table: list[tuple[signals.Signal, Callable[..., Any]]] = [
            (signals.phase_started, self._on_phase_started),
            (signals.phase_finished, self._on_phase_finished),
            (signals.episode_scored, self._on_episode_scored),
            (signals.episode_peer_scored, self._on_peer_scored),
            (signals.warning_issued, lambda **kw: self._on_warning("issued", **kw)),
            (signals.warning_escalated, lambda **kw: self._on_warning("escalated", **kw)),
            (signals.warning_expired, lambda **kw: self._on_warning("expired", **kw)),
            (signals.integrity_incident, self._on_incident),
            (signals.aggregation_completed, self._on_aggregation),
            (signals.checkpoint_written, self._on_checkpoint),
            (signals.checkpoint_rolled_back, self._on_rollback),
        ]
        for signal, receiver in table:
            signal.connect(receiver, weak=False)
            self._connected.append((signal, receiver))

    def close(self) -> None:
        """Disconnect from the global signals. Idempotent."""
        for signal, receiver in self._connected:
            signal.disconnect(receiver)
        self._connected.clear()

    def _current_run(self) -> str | None:
        """The run a signal belongs to: the sending thread's, else the only live one."""
        with self._lock:
            run_id = self._by_thread.get(threading.get_ident())
            if run_id is not None:
                return run_id
            live = [h.run_id for h in self._handles.values() if h.state.live]
            return live[0] if len(live) == 1 else None

    def _publish(self, run_id: str, event: str, data: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Publish on the bus and mirror the event into the run's log buffer."""
        stamped = self.bus.publish(run_id, event, data)
        handle = self.get_handle(run_id)
        logger_ = handle.components.get("logger") if handle is not None else None
        if logger_ is not None and event != "heartbeat":
            try:
                level = "warning" if event in ("warning", "incident", "rollback") else "info"
                getattr(logger_, level)(
                    event, **{k: v for k, v in stamped["data"].items() if k != "run_id"}
                )
            except Exception:  # noqa: BLE001 - logging never breaks a run
                logger.debug("could not mirror %s event into the run log", event, exc_info=True)
        return stamped

    def attach(self, run_id: str, **components: Any) -> RunHandle:
        """Hand the manager real components for a run: ``trainer``, ``logger``,
        ``watchdog``, ``checkpoint``. The observability routes read these."""
        handle = self.get_handle(run_id)
        if handle is None:
            raise RunNotFound(run_id)
        unknown = sorted(set(components) - {"trainer", "logger", "watchdog", "checkpoint"})
        if unknown:
            raise ImproperlyConfigured(f"cannot attach {unknown}")
        handle.components.update({k: v for k, v in components.items() if v is not None})
        return handle

    def component(self, run_id: str, name: str) -> Any | None:
        handle = self.get_handle(run_id)
        return handle.components.get(name) if handle is not None else None

    def tree(self, run_id: str) -> Any | None:
        """The live :class:`PeerTree` for a run the manager holds, else ``None``.

        An attached trainer's tree wins. Otherwise a flat tree is built over the
        orchestrator's peers (the swarm *is* a flat tree), with each node's ``peer``
        bound to the live peer so inference can run over it.
        """
        handle = self.get_handle(run_id)
        if handle is None:
            return None
        trainer = handle.components.get("trainer")
        if trainer is not None and getattr(trainer, "tree", None) is not None:
            return trainer.tree
        orchestrator = handle.orchestrator
        if orchestrator is None or not orchestrator.peers:
            return None
        from ..coopetition.tree import PeerTree

        tree = PeerTree.flat(list(orchestrator.peers))
        for name, peer in orchestrator.peers.items():
            node = tree.get(name)
            node.peer = peer
        return tree

    def _emit(self, event: str, data: Mapping[str, Any]) -> None:
        run_id = self._current_run()
        if run_id is not None:
            self._publish(run_id, event, data)

    def _on_phase_started(self, signal=None, sender=None, phase=None, **_: Any) -> None:
        self._emit("phase", {"phase": phase, "status": "started"})

    def _on_phase_finished(self, signal=None, sender=None, phase=None, **_: Any) -> None:
        self._emit("phase", {"phase": phase, "status": "finished"})

    def _on_aggregation(self, signal=None, sender=None, result=None, **_: Any) -> None:
        self._emit("phase", {"phase": "aggregation", "status": "completed"})

    def _on_episode_scored(self, signal=None, sender=None, scorecard=None, **_: Any) -> None:
        episode = getattr(scorecard, "episode", None)
        self._emit("run", {"kind": "episode_scored", "episode": episode})

    def _on_peer_scored(self, signal=None, sender=None, card=None, **_: Any) -> None:
        self._emit(
            "run",
            {
                "kind": "episode_peer_scored",
                "peer": getattr(card, "peer", None),
                "episode": getattr(card, "episode", None),
                "final_score": getattr(card, "final_score", None),
            },
        )

    def _on_warning(
        self, kind: str, signal=None, sender=None, peer=None, rule_id=None,
        record=None, episode=None, severity=None, **_: Any,
    ) -> None:
        self._emit(
            "warning",
            {
                "kind": kind,
                "peer": peer,
                "rule_id": rule_id,
                "episode": episode if episode is not None else getattr(record, "episode", None),
                "severity": severity if severity is not None else getattr(record, "severity", None),
            },
        )

    def _on_incident(self, signal=None, sender=None, reason=None, workers=None, peers=None, **_: Any) -> None:
        data: dict[str, Any] = {"reason": reason}
        if workers is not None:
            data["workers"] = list(workers)
        if peers is not None:
            data["peers"] = list(peers)
        self._emit("incident", data)

    def _on_checkpoint(self, signal=None, sender=None, record=None, restored=False, **_: Any) -> None:
        self._emit(
            "checkpoint",
            {
                "episode": getattr(record, "episode", None),
                "path": getattr(record, "path", None),
                "restored": bool(restored),
            },
        )

    def _on_rollback(self, signal=None, sender=None, reason=None, episode=None, **_: Any) -> None:
        self._emit("rollback", {"reason": reason, "episode": episode})

    # -- starting -------------------------------------------------------------
    def start(self, request: RunConfig | Mapping[str, Any] | None = None) -> RunHandle:
        """Queue a run and start its thread. Raises :class:`CapacityExceeded`."""
        config = _config_from(request)
        if config.resume_from is True:
            raise ImproperlyConfigured(
                "resume_from=True continues a run under its own id; use resume(run_id). "
                "start() accepts an explicit checkpoint file to branch a new run from."
            )
        if config.resume_from not in (None, False) and not Path(config.resume_from).exists():
            raise ImproperlyConfigured(f"resume_from path does not exist: {config.resume_from}")
        if not self._acquire_slot():
            raise CapacityExceeded(int(self.max_concurrent or 0))
        try:
            orchestrator = self._factory(self._run_settings(), self.registry)
            run_id = config.run_id or orchestrator.run_id
            config.run_id = run_id
            orchestrator.run_id = run_id
            with self._lock:
                if run_id in self._handles or self.registry.get_run(run_id) is not None:
                    raise RunExists(run_id)
                handle = RunHandle(run_id, config)
                handle.orchestrator = orchestrator
                self._handles[run_id] = handle
            self.registry.queue_run(
                run_id,
                settings={"device": config.device, "epochs": config.epochs, "queued": True},
            )
        except BaseException:
            self._release_slot()
            raise
        self._publish(run_id, "run", {"status": "queued"})
        self._spawn(handle)
        return handle

    def _spawn(self, handle: RunHandle) -> None:
        thread = threading.Thread(
            target=self._run, args=(handle,), name=f"mactf-run-{handle.run_id}", daemon=True
        )
        handle.thread = thread
        handle.done.clear()
        thread.start()

    def _acquire_slot(self, grace: float = 2.0) -> bool:
        """Take a run slot, tolerating a run that is terminal but still winding down.

        ``Orchestrator.run`` writes the terminal row before the run thread releases
        its slot, so a client that sees ``completed`` and immediately submits again
        would otherwise lose that race and get a spurious 429.
        """
        if self._slots is None:
            return True
        if self._slots.acquire(blocking=False):
            return True
        with self._lock:
            candidates = [h for h in self._handles.values() if not h.done.is_set()]
        # The terminal row is written before the handle's state flips and before the
        # slot is released, so a run that has just finished can still look live.
        winding_down = [h for h in candidates if h.state.terminal or self._row_is_terminal(h.run_id)]
        if not winding_down:
            return False
        deadline = time.monotonic() + grace
        for handle in winding_down:
            handle.done.wait(max(0.0, deadline - time.monotonic()))
        return self._slots.acquire(blocking=False)

    def _row_is_terminal(self, run_id: str) -> bool:
        try:
            row = self.registry.get_run(run_id)
        except Exception:  # noqa: BLE001 - a registry hiccup must not turn into a 500
            return False
        return bool(row) and row["status"] in ("completed", "failed", "cancelled")

    def _release_slot(self) -> None:
        if self._slots is not None:
            try:
                self._slots.release()
            except ValueError:  # released more than acquired; never fatal
                logger.debug("run slot released twice")

    # -- the run thread -------------------------------------------------------
    def _transition(self, handle: RunHandle, target: RunState, *, notes: str | None = None) -> None:
        with handle.lock:
            if handle.state is target:
                return
            if not handle.state.can_become(target):
                raise RunStateError(handle.run_id, handle.status, f"move to {target.value}")
            handle.state = target
            if target is RunState.RUNNING and handle.started_at is None:
                handle.started_at = time.time()
            if target.terminal:
                handle.finished_at = time.time()
        # ``start_run`` (inside Orchestrator.run) owns the running row; the others
        # are written here so the registry never lags a state the API reports.
        if target is not RunState.RUNNING:
            self.registry.set_run_status(handle.run_id, target.value, notes=notes)

    def _run(self, handle: RunHandle) -> None:
        run_id = handle.run_id
        with self._lock:
            self._by_thread[threading.get_ident()] = run_id
        try:
            self._execute(handle)
        except BaseException as exc:  # noqa: BLE001 - the thread must always settle the run
            logger.exception("run %s crashed", run_id)
            self._fail(handle, exc)
        finally:
            with self._lock:
                self._by_thread.pop(threading.get_ident(), None)
            self._release_slot()
            handle.done.set()
            self._trim()

    def _execute(self, handle: RunHandle) -> None:
        orchestrator = handle.orchestrator
        assert orchestrator is not None
        config = handle.config
        if "logger" not in handle.components:
            from ..training.logging import MACTFLogger

            # Memory only: tail() works, nothing is written unless the caller attaches sinks.
            handle.components["logger"] = MACTFLogger(
                self.settings, run_id=handle.run_id, sinks=[], async_write=False, level="DEBUG"
            )
        if handle.cancel_requested:  # cancelled while still queued
            self._finish(handle, RunState.CANCELLED, {"epochs_completed": 0})
            return
        try:
            orchestrator.prepare(config)
        except Exception as exc:  # noqa: BLE001
            self._fail(handle, exc)
            return
        self._transition(handle, RunState.RUNNING)
        self._publish(handle.run_id, "run", {"status": "running"})
        try:
            summary = orchestrator.run(
                config, should_stop=handle.should_stop, on_epoch=lambda entry: self._on_epoch(handle, entry)
            )
        except BaseException as exc:  # noqa: BLE001
            self._fail(handle, exc, already_recorded=True)
            return
        handle.summary = {k: v for k, v in summary.items() if k != "history"}
        handle.epochs_completed = int(summary.get("epochs_completed", handle.epochs_completed) or 0)
        status = RunState(summary.get("status", "completed"))
        if status is RunState.CANCELLED and handle.pause_requested and not handle.cancel_requested:
            self._pause(handle)
        else:
            self._finish(handle, status, summary)

    def _on_epoch(self, handle: RunHandle, entry: Mapping[str, Any]) -> None:
        handle.epochs_completed += 1
        self._publish(
            handle.run_id,
            "epoch",
            {
                "epoch": entry.get("epoch"),
                "score": entry.get("score"),
                "kappa": entry.get("kappa"),
                "entropy": entry.get("entropy"),
                "weights": entry.get("weights"),
                "duration": entry.get("duration"),
            },
        )

    def _pause(self, handle: RunHandle) -> None:
        with handle.lock:
            handle.state = RunState.PAUSED
            handle._pause.clear()
        self.registry.set_run_status(handle.run_id, "paused", finished=False)
        self._publish(handle.run_id, "run", {"status": "paused", "epochs_completed": handle.epochs_completed})

    def _finish(self, handle: RunHandle, state: RunState, summary: Mapping[str, Any]) -> None:
        with handle.lock:
            handle.state = state
            handle.finished_at = time.time()
        # Orchestrator.run already finished the row for completed/cancelled/dry_run.
        stored = self.registry.get_run(handle.run_id)
        if stored is None or stored["status"] != state.value:
            self.registry.set_run_status(handle.run_id, state.value)
        self._publish(
            handle.run_id,
            "terminal",
            {"status": state.value, "epochs": summary.get("epochs_completed", handle.epochs_completed)},
        )

    def _fail(self, handle: RunHandle, exc: BaseException, *, already_recorded: bool = False) -> None:
        message = f"{type(exc).__name__}: {exc}"[:500]
        handle.error = message
        with handle.lock:
            handle.state = RunState.FAILED
            handle.finished_at = time.time()
        try:
            self.registry.set_run_status(handle.run_id, "failed", notes=message)
        except Exception:  # noqa: BLE001
            logger.exception("could not record failure of %s", handle.run_id)
        self._publish(
            handle.run_id, "terminal",
            {"status": "failed", "error": message, "epochs": handle.epochs_completed},
        )

    def _trim(self, keep: int = 256) -> None:
        with self._lock:
            finished = [h for h in self._handles.values() if h.state.terminal]
            for handle in sorted(finished, key=lambda h: h.finished_at or 0)[: max(0, len(finished) - keep)]:
                del self._handles[handle.run_id]

    # -- control ---------------------------------------------------------------
    def live_run_ids(self) -> list[str]:
        """Ids of runs the manager is driving right now (queued, running or resuming)."""
        with self._lock:
            return [h.run_id for h in self._handles.values() if h.state.live]

    def get_handle(self, run_id: str) -> RunHandle | None:
        with self._lock:
            return self._handles.get(run_id)

    def active(self) -> list[RunHandle]:
        with self._lock:
            return [h for h in self._handles.values() if h.state.live]

    def status(self, run_id: str) -> str | None:
        """The live state if the manager holds the run, else the stored status."""
        handle = self.get_handle(run_id)
        if handle is not None:
            return handle.status
        row = self.registry.get_run(run_id)
        return row["status"] if row else None

    def cancel(self, run_id: str, *, wait: bool = True) -> str:
        """Request cooperative cancellation. Returns ``"cancelling"`` while the
        thread has not yet reached an epoch boundary, else the final status.

        Raises :class:`RunNotFound`, or :class:`RunStateError` for a terminal run.
        """
        handle = self.get_handle(run_id)
        if handle is None:
            row = self.registry.get_run(run_id)
            if row is None:
                raise RunNotFound(run_id)
            if row["status"] in TERMINAL_STATUSES:
                raise RunStateError(run_id, row["status"], "cancel")
            if row["status"] == "paused":  # a paused row from an earlier process
                self.registry.set_run_status(run_id, "cancelled")
                self._publish(run_id, "terminal", {"status": "cancelled", "epochs": 0})
                return "cancelled"
            raise RunStateError(run_id, row["status"], "cancel")
        with handle.lock:
            if handle.state.terminal:
                raise RunStateError(run_id, handle.status, "cancel")
            if handle.state is RunState.PAUSED:
                handle.state = RunState.CANCELLED
                handle.finished_at = time.time()
                self.registry.set_run_status(run_id, "cancelled")
                self._publish(run_id, "terminal", {"status": "cancelled", "epochs": handle.epochs_completed})
                handle.done.set()
                return "cancelled"
            handle._cancel.set()
        if wait:
            handle.wait(self.cancel_grace)
        return handle.status if handle.state.terminal else "cancelling"

    def pause(self, run_id: str) -> RunHandle:
        """Ask a running run to stop at the next epoch boundary and become ``paused``."""
        handle = self.get_handle(run_id)
        if handle is None:
            if self.registry.get_run(run_id) is None:
                raise RunNotFound(run_id)
            raise RunStateError(run_id, self.status(run_id) or "unknown", "pause")
        with handle.lock:
            if handle.state is not RunState.RUNNING:
                raise RunStateError(run_id, handle.status, "pause")
            handle._pause.set()
        return handle

    def resume(self, run_id: str, *, epochs: int | None = None) -> RunHandle:
        """Continue a ``paused`` (or ``interrupted``) run under the same id."""
        handle = self.get_handle(run_id)
        row = self.registry.get_run(run_id)
        if handle is None and row is None:
            raise RunNotFound(run_id)
        current = handle.status if handle is not None else row["status"]
        if current not in ("paused", "interrupted"):
            raise RunStateError(run_id, current, "resume")
        if not self._acquire_slot():
            raise CapacityExceeded(int(self.max_concurrent or 0))
        try:
            stored: dict[str, Any] = {}
            if row is not None:
                import json

                try:
                    stored = json.loads(row["settings"] or "{}")
                except ValueError:
                    stored = {}
            total = epochs or stored.get("epochs")
            config = RunConfig(
                epochs=int(total) if total else None,
                device=stored.get("device"),
                run_id=run_id,
                resume_from=True,
            )
            orchestrator = self._factory(self._run_settings(), self.registry)
            orchestrator.run_id = run_id
            if handle is None:
                handle = RunHandle(run_id, config)
                handle.state = RunState.INTERRUPTED
                with self._lock:
                    self._handles[run_id] = handle
            with handle.lock:
                handle.config = config
                handle.orchestrator = orchestrator
                handle._cancel.clear()
                handle._pause.clear()
                handle.error = None
                handle.finished_at = None
            self._transition(handle, RunState.RESUMING)
        except BaseException:
            self._release_slot()
            raise
        self._publish(run_id, "run", {"status": "resuming"})
        self._spawn(handle)
        return handle

    def reap_orphans(self) -> list[str]:
        """Rewrite ``running``/``queued``/``resuming`` rows nothing is driving to ``interrupted``.

        The only place the manager writes a status it did not observe. ``paused``
        rows survive: a paused run has a checkpoint by construction.
        """
        reaped: list[str] = []
        for row in self.registry.runs_with_status(("running", "queued", "resuming")):
            handle = self.get_handle(row["id"])
            if handle is not None and handle.state.live:
                continue
            self.registry.set_run_status(
                row["id"], "interrupted", notes="process ended while the run was active"
            )
            self._publish(row["id"], "terminal", {"status": "interrupted", "epochs": 0})
            reaped.append(row["id"])
        return reaped

    # -- reading ------------------------------------------------------------------
    def get(self, run_id: str) -> dict[str, Any] | None:
        """The stored row with the live status laid over it."""
        row = self.registry.get_run(run_id)
        handle = self.get_handle(run_id)
        if row is None and handle is None:
            return None
        merged = dict(row or {"id": run_id})
        if handle is not None:
            merged["status"] = handle.status
            merged["epochs_completed"] = handle.epochs_completed
            if handle.error:
                merged["error"] = handle.error
        return merged

    def subscribe(self, run_id: str, last_id: int | None = None) -> Subscription:
        """A subscription for a live, recent or stored run.

        A stored terminal run with no buffer (an earlier process) gets a closed
        stream carrying one synthesized ``terminal`` event, so a late client still
        learns the outcome.
        """
        if self.bus.has(run_id):
            return self.bus.subscribe(run_id, last_id)
        row = self.registry.get_run(run_id)
        if row is None:
            raise RunNotFound(run_id)
        status = row["status"]
        if status in TERMINAL_STATUSES:
            self._publish(run_id, "terminal", {"status": status, "epochs": self.registry.latest_episode(run_id)})
        else:
            self._publish(run_id, "run", {"status": status})
        return self.bus.subscribe(run_id, last_id)

    def wait(self, run_id: str, timeout: float | None = None) -> str | None:
        handle = self.get_handle(run_id)
        if handle is not None:
            handle.wait(timeout)
        return self.status(run_id)

    # -- shutdown ------------------------------------------------------------------
    def shutdown(self, timeout: float = 10.0) -> list[str]:
        """Cancel every live run, join the threads, mark stragglers ``interrupted``."""
        deadline = time.monotonic() + timeout
        handles = self.active()
        for handle in handles:
            handle._cancel.set()
        stragglers: list[str] = []
        for handle in handles:
            handle.wait(max(0.0, deadline - time.monotonic()))
            if not handle.done.is_set():
                stragglers.append(handle.run_id)
                self.registry.set_run_status(
                    handle.run_id, "interrupted", notes="server shut down before the run reached a boundary"
                )
        self.close()
        return stragglers
