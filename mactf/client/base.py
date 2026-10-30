"""Transport protocol and the few shapes that exist only in the client (document 03 sections 3-4).

``RunHandle``, ``Event``, ``Page``, ``Run``, ``Episode``, ``RuleEvent`` and ``Checkpoint``
are transport shapes, not domain models: the API projects a storage row into them, and
no engine class holds that projection. Every other value the client returns is the
framework's own class (``PeerScorecard``, ``AggregationReport``, ``CapabilityReport``,
``PointsTable``, ``NormalizationReport``, ``ExecResult``, ``PeerTree``) rebuilt through its
``from_dict``, or a :class:`Report` where the API does not fix the shape.

Nothing here imports torch or an engine package at module scope.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from ..http.response import Response
from ..wire import need, opt

T = TypeVar("T")

#: Statuses after which a run never changes again, unless resumed after an interruption.
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "dry_run", "interrupted"})


@dataclass(frozen=True)
class RunHandle:
    """What ``POST /runs`` answered: an id, a status and the links to follow."""

    run_id: str
    status: str
    created_at: float
    links: Mapping[str, str] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """The run id, spelled the way :class:`Run` spells it."""
        return self.run_id

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunHandle":
        owner = "RunHandle"
        return cls(
            run_id=str(need(owner, payload, "run_id")),
            status=str(need(owner, payload, "status")),
            created_at=float(opt(payload, "created_at", 0.0) or 0.0),
            links=dict(payload.get("links") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "created_at": self.created_at,
            "links": dict(self.links),
        }


@dataclass(frozen=True)
class Event:
    """One server-sent event. ``heartbeat`` frames carry no data and no id."""

    id: int | None
    event: str
    data: Mapping[str, Any]

    @property
    def type(self) -> str:
        return self.event

    @property
    def epoch(self) -> int | None:
        """The epoch the event reports, when it reports one."""
        value = self.data.get("epoch") if isinstance(self.data, Mapping) else None
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    @classmethod
    def from_frame(cls, frame: Mapping[str, Any]) -> "Event":
        raw_id = frame.get("id")
        try:
            event_id = int(raw_id) if raw_id not in (None, "") else None
        except (TypeError, ValueError):
            event_id = None
        if frame.get("data") is None:
            return cls(id=event_id, event="heartbeat", data={})
        data = frame["data"]
        return cls(id=event_id, event=str(frame.get("event") or "message"), data=data if isinstance(data, Mapping) else {"value": data})


@dataclass(frozen=True)
class Page(Generic[T]):
    """One page of a collection. Iterating yields ``items``; it never fetches."""

    items: tuple[T, ...]
    total: int
    limit: int
    offset: int
    has_more: bool

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> T:
        return self.items[index]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], parse: Any = lambda item: item) -> "Page[Any]":
        owner = "Page"
        return cls(
            items=tuple(parse(item) for item in need(owner, payload, "items")),
            total=int(need(owner, payload, "total")),
            limit=int(need(owner, payload, "limit")),
            offset=int(need(owner, payload, "offset")),
            has_more=bool(need(owner, payload, "has_more")),
        )


@dataclass(frozen=True)
class Run:
    """The ``runs`` row as the API projects it."""

    id: str
    status: str
    started_at: float | None
    finished_at: float | None
    epochs: int | None
    config: Mapping[str, Any]
    runtime: Mapping[str, Any] = field(default_factory=dict)
    sandbox: Mapping[str, Any] = field(default_factory=dict)
    kappa: float | None = None
    entropy: float | None = None
    epochs_completed: int = 0
    notes: str | None = None
    error: str | None = None

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Run":
        owner = "Run"
        config = dict(payload.get("settings") or payload.get("config") or {})
        epochs = config.get("epochs")
        return cls(
            id=str(need(owner, payload, "id")),
            status=str(need(owner, payload, "status")),
            started_at=_maybe_float(opt(payload, "started_at", None)),
            finished_at=_maybe_float(opt(payload, "finished_at", None)),
            epochs=int(epochs) if epochs is not None else None,
            config=config,
            runtime=dict(payload.get("runtime") or {}),
            sandbox=dict(payload.get("sandbox") or {}),
            kappa=_maybe_float(opt(payload, "kappa", None)),
            entropy=_maybe_float(opt(payload, "entropy", None)),
            epochs_completed=int(opt(payload, "epochs_completed", 0) or 0),
            notes=opt(payload, "notes", None),
            error=opt(payload, "error", None),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "settings": dict(self.config),
            "runtime": dict(self.runtime),
            "sandbox": dict(self.sandbox),
            "kappa": self.kappa,
            "entropy": self.entropy,
            "epochs_completed": self.epochs_completed,
            "notes": self.notes,
            "error": self.error,
        }


@dataclass(frozen=True)
class Episode:
    """The ``episodes`` row: the episode scorecard and, when calibrated, its normalization."""

    run_id: str
    episode: int
    started_at: float | None
    finished_at: float | None
    scorecard: Any  # mactf.scoring.scorecard.EpisodeScorecard
    normalization: Any | None = None  # mactf.scoring.normalize.NormalizationReport

    @property
    def peers(self) -> Mapping[str, Any]:
        """``{peer: PeerScorecard}`` of this episode."""
        return self.scorecard.peers

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Episode":
        from ..scoring.normalize import NormalizationReport
        from ..scoring.scorecard import EpisodeScorecard

        owner = "Episode"
        normalization = payload.get("normalization")
        return cls(
            run_id=str(need(owner, payload, "run_id")),
            episode=int(need(owner, payload, "episode")),
            started_at=_maybe_float(opt(payload, "started_at", None)),
            finished_at=_maybe_float(opt(payload, "finished_at", None)),
            scorecard=EpisodeScorecard.from_dict(need(owner, payload, "scorecard")),
            normalization=NormalizationReport.from_dict(normalization) if normalization else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "episode": self.episode,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "scorecard": self.scorecard.to_dict(),
            "normalization": self.normalization.as_dict() if self.normalization is not None else None,
        }


@dataclass(frozen=True)
class RuleEvent:
    """One stored ledger event, with the severity the server resolved for it."""

    id: int
    run_id: str
    episode: int
    peer: str
    rule_id: str
    magnitude: float
    severity: str
    evidence: str
    source: str
    created_at: float | None
    reversed_at: float | None = None
    reversed_by: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuleEvent":
        owner = "RuleEvent"
        return cls(
            id=int(need(owner, payload, "id")),
            run_id=str(need(owner, payload, "run_id")),
            episode=int(need(owner, payload, "episode")),
            peer=str(need(owner, payload, "peer")),
            rule_id=str(need(owner, payload, "rule_id")),
            magnitude=float(need(owner, payload, "magnitude")),
            severity=str(need(owner, payload, "severity")),
            evidence=str(opt(payload, "evidence", "") or ""),
            source=str(opt(payload, "source", "") or ""),
            created_at=_maybe_float(opt(payload, "created_at", None)),
            reversed_at=_maybe_float(opt(payload, "reversed_at", None)),
            reversed_by=opt(payload, "reversed_by", None),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "run_id": self.run_id, "episode": self.episode, "peer": self.peer,
            "rule_id": self.rule_id, "magnitude": self.magnitude, "severity": self.severity,
            "evidence": self.evidence, "source": self.source, "created_at": self.created_at,
            "reversed_at": self.reversed_at, "reversed_by": self.reversed_by,
        }


@dataclass(frozen=True)
class Checkpoint:
    """A checkpoint row, with what verification found when the server was asked to verify."""

    run_id: str
    episode: int
    path: str
    created_at: float | None = None
    kappa: float | None = None
    entropy: float | None = None
    score: float | None = None
    size_bytes: int | None = None
    verified: bool | None = None
    sections: tuple[str, ...] = ()
    best: bool = False
    exists: bool = True
    problems: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Checkpoint":
        owner = "Checkpoint"
        return cls(
            run_id=str(need(owner, payload, "run_id")),
            episode=int(need(owner, payload, "episode")),
            path=str(need(owner, payload, "path")),
            created_at=_maybe_float(opt(payload, "created_at", None)),
            kappa=_maybe_float(opt(payload, "kappa", None)),
            entropy=_maybe_float(opt(payload, "entropy", None)),
            score=_maybe_float(opt(payload, "score", None)),
            size_bytes=opt(payload, "size_bytes", None),
            verified=opt(payload, "verified", None),
            sections=tuple(str(s) for s in (payload.get("sections") or ())),
            best=bool(opt(payload, "best", False)),
            exists=bool(opt(payload, "exists", True)),
            problems=tuple(str(p) for p in (payload.get("problems") or ())),
        )


class Report(Mapping[str, Any]):
    """An immutable view of a JSON object the API does not pin to a class.

    Reads like the dict it came from (``report["status"]``) and like an object
    (``report.status``); nested objects are wrapped the same way, lists become tuples.
    ``to_dict()`` returns the original structure, so ``Report.from_dict(r.to_dict()) == r``.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_data", dict(data))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Report":
        return cls(payload)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def __getitem__(self, key: str) -> Any:
        return _wrap(self._data[key])

    def __getattr__(self, name: str) -> Any:
        try:
            return _wrap(self._data[name])
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("a Report is read-only")

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Report):
            return self._data == other._data
        if isinstance(other, Mapping):
            return self._data == dict(other)
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        keys = ", ".join(list(self._data)[:6])
        return f"Report({keys}{', ...' if len(self._data) > 6 else ''})"


def _wrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        return Report(value)
    if isinstance(value, list):
        return tuple(_wrap(item) for item in value)
    return value


def _maybe_float(value: Any) -> float | None:
    return None if value is None else float(value)


@runtime_checkable
class Transport(Protocol):
    """What the namespaces need from a way of reaching the API.

    ``path`` is a route path built with ``reverse()``, never a route name. Both
    implementations return a :class:`mactf.http.Response`, so there is exactly one
    response type in the framework.
    """

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response: ...

    def stream(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Iterator[Event]: ...


def now() -> float:
    return time.time()


__all__ = [
    "Checkpoint", "Episode", "Event", "Page", "Report", "Response", "Run", "RunHandle", "RuleEvent",
    "TERMINAL_STATUSES", "Transport",
]
