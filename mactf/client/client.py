"""``MACTFClient`` and its namespaces (document 03 sections 5-6).

Every method is a route: it builds the path with :func:`~mactf.client.routes.path_for`,
sends it through whichever transport the client holds, and rebuilds the answer with the
framework's own ``from_dict``. There is no second implementation of anything the server
does, so an in-process and a remote client are the same code over different pipes.

Module scope imports only ``mactf.http``, ``mactf.exceptions`` and this package. The
framework classes (``PeerScorecard`` and friends) are imported inside the parse helpers, so
``import mactf.client`` never loads torch or an engine package.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .base import (
    Checkpoint,
    Episode,
    Event,
    Page,
    Report,
    Run,
    RuleEvent,
    RunHandle,
    Transport,
)
from .errors import Conflict, ValidationError, raise_for_response
from .pagination import iter_pages
from .routes import path_for

T = TypeVar("T")

__all__ = [
    "MACTFClient", "ObservabilityAPI", "PeersAPI", "PointsEntry", "RulesAPI", "RunsAPI",
    "SandboxAPI", "ScoresAPI", "ScoringAPI", "TrainersAPI",
]

SCORECARD_V1 = "application/vnd.mactf.v1+json"
SCORECARD_V2 = "application/vnd.mactf.v2+json"


@dataclass(frozen=True)
class PointsEntry:
    """One ``(episode, peer)`` row of ``GET /runs/{id}/points``."""

    episode: int
    peer: str
    final_score: float
    points: Any  # mactf.scoring.points.PointsBreakdown

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PointsEntry":
        from ..scoring.points import PointsBreakdown
        from ..wire import need

        owner = "PointsEntry"
        return cls(
            episode=int(need(owner, payload, "episode")),
            peer=str(need(owner, payload, "peer")),
            final_score=float(need(owner, payload, "final_score")),
            points=PointsBreakdown.from_dict(need(owner, payload, "points")),
        )


def _compact(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _scorecard(payload: Mapping[str, Any]) -> Any:
    from ..scoring.scorecard import PeerScorecard

    return PeerScorecard.from_dict(payload)


def _event_dict(event: Any) -> dict[str, Any]:
    """A ledger event as the API takes it: a mapping, or ``(rule_id, evidence[, magnitude[, severity]])``."""
    if isinstance(event, Mapping):
        return dict(event)
    if isinstance(event, (tuple, list)) and 2 <= len(event) <= 4:
        keys = ("rule_id", "evidence", "magnitude", "severity")
        return {key: value for key, value in zip(keys, event) if value is not None}
    raise TypeError(f"a ledger event is a mapping or a (rule_id, evidence, ...) tuple, not {type(event).__name__}")


class _Namespace:
    def __init__(self, client: "MACTFClient") -> None:
        self._client = client

    @property
    def _transport(self) -> Transport:
        return self._client.transport

    def _send(
        self,
        method: str,
        route: str,
        /,
        *,
        json: Any | None = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        **path_args: Any,
    ) -> Any:
        path = path_for(route, **path_args)
        return raise_for_response(
            self._transport.request(method, path, json=json, params=_compact(params or {}) or None, headers=headers)
        )

    def _json(self, method: str, route: str, /, **kwargs: Any) -> Any:
        return self._send(method, route, **kwargs).json()

    def _page(self, route: str, parse: Callable[[Any], T], /, *, params: Mapping[str, Any], **path_args: Any) -> Page[T]:
        return Page.from_payload(self._json("GET", route, params=params, **path_args), parse)

    def _walk(self, route: str, parse: Callable[[Any], T], /, *, limit: int = 100, **filters: Any) -> Iterator[T]:
        path_args = {k: filters.pop(k) for k in list(filters) if k in ("run_id", "episode", "name")}
        for page in iter_pages(self._transport, path_for(route, **path_args), limit=limit, parse=parse, **filters):
            yield from page

    def _report(self, method: str, route: str, /, **kwargs: Any) -> Report:
        return Report(self._json(method, route, **kwargs))

    def _reports(self, route: str, /, **kwargs: Any) -> tuple[Report, ...]:
        return tuple(Report(item) for item in self._json("GET", route, **kwargs))


class RunsAPI(_Namespace):
    """Start, observe, steer and inspect runs."""

    def create(
        self,
        options: Mapping[str, Any] | None = None,
        /,
        *,
        epochs: int | None = None,
        peers: int | None = None,
        device: str | None = None,
        seed: int | None = None,
        dry_run: bool = False,
        redteam: bool = False,
        resume_from: str | int | None = None,
        run_id: str | None = None,
        task_source: str | None = None,
        tags: Mapping[str, str] | None = None,
        trainer: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> RunHandle:
        """``POST /runs``. A fresh ``Idempotency-Key`` is generated unless one is given, so a
        retried request can never start a second run. ``options`` is the same request as a
        mapping (``create({"epochs": 1, "seed": 1337})``); keywords override it."""
        if options:
            merged = {**dict(options)}
            for name, value in (
                ("epochs", epochs), ("peers", peers), ("device", device), ("seed", seed),
                ("run_id", run_id), ("task_source", task_source), ("trainer", trainer),
                ("resume_from", resume_from), ("tags", tags),
            ):
                if value is not None:
                    merged[name] = value
            unknown = sorted(set(merged) - {
                "epochs", "peers", "device", "seed", "dry_run", "redteam", "resume_from",
                "run_id", "task_source", "tags", "trainer",
            })
            if unknown:
                raise ValidationError(f"unknown run option(s) {unknown}")
            epochs, peers, device, seed = (merged.get(k) for k in ("epochs", "peers", "device", "seed"))
            dry_run = bool(merged.get("dry_run", dry_run))
            redteam = bool(merged.get("redteam", redteam))
            resume_from, run_id = merged.get("resume_from"), merged.get("run_id")
            task_source, tags, trainer = merged.get("task_source"), merged.get("tags"), merged.get("trainer")
        body = _compact({
            "epochs": epochs, "peers": peers, "device": device, "seed": seed,
            "dry_run": dry_run or None, "redteam": redteam or None,
            "resume_from": str(resume_from) if resume_from is not None else None,
            "run_id": run_id, "task_source": task_source, "tags": dict(tags) if tags else None,
            "trainer": dict(trainer) if trainer is not None else None,
        })
        key = idempotency_key or uuid.uuid4().hex
        return RunHandle.from_dict(
            self._json("POST", "run-list", json=body, headers={"Idempotency-Key": key})
        )

    def list(self, *, limit: int = 50, offset: int = 0, status: str | None = None) -> Page[Run]:
        return self._page("run-list", Run.from_dict, params={"limit": limit, "offset": offset, "status": status})

    def iter(self, *, limit: int = 100, status: str | None = None) -> Iterator[Run]:
        """Every run, newest first, across pages."""
        return self._walk("run-list", Run.from_dict, limit=limit, status=status)

    def get(self, run_id: str) -> Run:
        return Run.from_dict(self._json("GET", "run-detail", run_id=run_id))

    def cancel(self, run_id: str) -> Report:
        return self._report("POST", "run-cancel", run_id=run_id)

    def summary(self, run_id: str) -> Report:
        return self._report("GET", "run-summary", run_id=run_id)

    def episodes(self, run_id: str, *, limit: int = 50, offset: int = 0) -> Page[Episode]:
        return self._page(
            "run-episode-list", Episode.from_dict, params={"limit": limit, "offset": offset}, run_id=run_id
        )

    def iter_episodes(self, run_id: str, *, limit: int = 100) -> Iterator[Episode]:
        return self._walk("run-episode-list", Episode.from_dict, limit=limit, run_id=run_id)

    def episode(self, run_id: str, episode: int) -> Episode:
        return Episode.from_dict(self._json("GET", "run-episode-detail", run_id=run_id, episode=episode))

    def scorecard(self, run_id: str, episode: int, peer: str, *, version: int | None = None) -> Any:
        """The stored ``PeerScorecard`` of one peer in one episode."""
        return self._client.scores.scorecard(run_id, episode, peer, version=version)

    def rule_events(
        self,
        run_id: str,
        *,
        peer: str | None = None,
        rule_id: str | None = None,
        episode: int | None = None,
        severity: str | None = None,
        source: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Page[RuleEvent]:
        return self._page(
            "run-rule-event-list",
            RuleEvent.from_dict,
            params={
                "peer": peer, "rule_id": rule_id, "episode": episode, "severity": severity,
                "source": source, "limit": limit, "offset": offset,
            },
            run_id=run_id,
        )

    def rule_counts(self, run_id: str) -> dict[str, int]:
        return dict(self._json("GET", "run-rule-counts", run_id=run_id))

    def aggregations(self, run_id: str, *, limit: int = 50, offset: int = 0) -> Page[Any]:
        """Per-episode ``AggregationReport`` objects; each keeps ``episode`` on the returned pair."""
        from ..coopetition.report import AggregationReport

        def parse(item: Mapping[str, Any]) -> Any:
            report = AggregationReport.from_dict(item["report"])
            return report

        return self._page("run-aggregation-list", parse, params={"limit": limit, "offset": offset}, run_id=run_id)

    def checkpoints(
        self, run_id: str, *, verify: bool | None = None, valid_only: bool | None = None,
        limit: int = 50, offset: int = 0,
    ) -> Page[Checkpoint]:
        return self._page(
            "run-checkpoint-list",
            Checkpoint.from_dict,
            params={"verify": verify, "valid_only": valid_only, "limit": limit, "offset": offset},
            run_id=run_id,
        )

    def checkpoint(self, run_id: str, episode: int) -> Checkpoint:
        return Checkpoint.from_dict(self._json("GET", "run-checkpoint-detail", run_id=run_id, episode=episode))

    def rollback(self, run_id: str, episode: int) -> Report:
        """Roll the run back to the checkpoint of ``episode`` (newer ones are archived, not deleted)."""
        return self._report("POST", "run-rollback", json={"episode": int(episode)}, run_id=run_id)

    def warnings(
        self, run_id: str, *, peer: str | None = None, all: bool | None = None, limit: int = 50, offset: int = 0
    ) -> Page[Report]:
        return self._page(
            "run-warning-list", Report,
            params={"peer": peer, "all": all, "limit": limit, "offset": offset}, run_id=run_id,
        )

    def stream(self, run_id: str, *, last_event_id: int | None = None) -> Iterator[Event]:
        """The run's server-sent events until ``terminal`` (inclusive). Heartbeats are yielded too."""
        headers = {"Last-Event-ID": str(last_event_id)} if last_event_id is not None else None
        return self._transport.stream("GET", path_for("run-events", run_id=run_id), headers=headers)

    def tree(self, run_id: str) -> Any:
        from ..coopetition.tree import PeerTree

        return PeerTree.from_dict(self._json("GET", "run-peer-tree", run_id=run_id))

    def node(self, run_id: str, path: str) -> Report:
        return self._report("GET", "run-peer-node", run_id=run_id, path=path)

    def infer(
        self,
        run_id: str,
        inputs: Mapping[str, Any],
        *,
        node_path: str | None = None,
        max_depth: int | None = None,
        budget: int | None = None,
        parallel: bool = False,
        max_workers: int | None = None,
        seed: int | None = None,
        score: bool = False,
        persist_warnings: bool = False,
    ) -> Report:
        body = _compact({
            "inputs": dict(inputs), "node_path": node_path, "max_depth": max_depth, "budget": budget,
            "parallel": parallel or None, "max_workers": max_workers, "seed": seed,
            "score": score or None, "persist_warnings": persist_warnings or None,
        })
        return self._report("POST", "run-infer", json=body, run_id=run_id)


class ScoringAPI(_Namespace):
    """Score ledgers and record the human decisions that change them."""

    def evaluate(
        self,
        peer: str,
        events: Sequence[Any],
        *,
        episode: int = 0,
        run_id: str | None = None,
    ) -> Any:
        """``POST /evaluations``: a ``PeerScorecard`` for a submitted ledger. Never retried."""
        body = _compact({
            "peer": peer, "episode": episode, "run_id": run_id,
            "events": [_event_dict(event) for event in events],
        })
        return _scorecard(self._json("POST", "evaluation-create", json=body))

    def append_ledger(self, run_id: str, episode: int, peer: str, events: Sequence[Any]) -> Any:
        body = {"peer": peer, "episode": episode, "events": [_event_dict(e) for e in events]}
        return _scorecard(self._json("POST", "run-episode-ledger", json=body, run_id=run_id, episode=episode))

    def adjudicate(
        self,
        run_id: str,
        *,
        peer: str,
        rule_id: str,
        episode: int,
        decision: str,
        rationale: str,
        adjudicator: str,
        severity: str | None = None,
    ) -> Report:
        body = _compact({
            "peer": peer, "rule_id": rule_id, "episode": episode, "decision": decision,
            "severity": severity, "rationale": rationale, "adjudicator": adjudicator,
        })
        return self._report("POST", "run-adjudication-create", json=body, run_id=run_id)

    def adjudications(self, run_id: str, *, limit: int = 50, offset: int = 0) -> Page[Report]:
        return self._page(
            "run-adjudication-list", Report, params={"limit": limit, "offset": offset}, run_id=run_id
        )

    def flags(self, run_id: str) -> Report:
        return self._report("GET", "run-flags", run_id=run_id)


class RulesAPI(_Namespace):
    def list(self, **filters: Any) -> tuple[Report, ...]:
        return self._reports("rule-list", params=filters)

    def get(self, rule_id: str) -> Report:
        return self._report("GET", "rule-detail", rule_id=rule_id)

    def validation(self) -> Report:
        return self._report("GET", "rule-validation")

    def severities(self) -> tuple[Report, ...]:
        return self._reports("severity-list")

    def tiers(self) -> tuple[Report, ...]:
        return self._reports("tier-list")


class PeersAPI(_Namespace):
    def list(self) -> tuple[Report, ...]:
        return self._reports("peer-list")

    def get(self, name: str) -> Report:
        return self._report("GET", "peer-detail", name=name)

    def contract(self, name: str) -> Report:
        return self._report("GET", "peer-contract", name=name)

    def preview_weights(
        self,
        scores: Mapping[str, float],
        *,
        eligible: Mapping[str, bool] | None = None,
        confidence: Mapping[str, float] | None = None,
        advance: bool = False,
    ) -> Any:
        """What the server's weight rule would do with these scores (an ``AggregationReport``)."""
        from ..coopetition.report import AggregationReport

        body = _compact({
            "scores": dict(scores),
            "eligible": dict(eligible) if eligible is not None else None,
            "confidence": dict(confidence) if confidence is not None else None,
            "advance": advance or None,
        })
        return AggregationReport.from_dict(self._json("POST", "weight-preview", json=body))


class SandboxAPI(_Namespace):
    def providers(self) -> tuple[Any, ...]:
        from ..sandbox.base import CapabilityReport

        return tuple(CapabilityReport.from_dict(item) for item in self._json("GET", "sandbox-provider-list"))

    def status(self) -> Report:
        return self._report("GET", "sandbox-status")

    def probe(self) -> Report:
        return self._report("POST", "sandbox-probe")

    def exec(self, peer: str, command: str | Sequence[str], *, timeout: float | None = None) -> Any:
        from ..sandbox.base import ExecResult

        body = _compact({
            "peer": peer, "command": command if isinstance(command, str) else list(command), "timeout": timeout,
        })
        return ExecResult.from_dict(self._json("POST", "sandbox-exec", json=body))


class TrainersAPI(_Namespace):
    def capabilities(self) -> Report:
        return self._report("GET", "trainer-capabilities")

    def scorers(self) -> tuple[Report, ...]:
        return self._reports("scorer-list")

    def preview(self, spec: Mapping[str, Any]) -> Report:
        """Validate and resolve a trainer spec without starting a run."""
        return self._report("POST", "trainer-preview", json=dict(spec))

    def presets(self) -> tuple[Report, ...]:
        return self._reports("trainer-preset-list")


class ScoresAPI(_Namespace):
    def scale(self) -> Any:
        from ..scoring.points import PointsTable

        return PointsTable.from_dict(self._json("GET", "score-scale"))

    def points(
        self, run_id: str, *, peer: str | None = None, episode: int | None = None,
        limit: int = 50, offset: int = 0,
    ) -> Page[PointsEntry]:
        return self._page(
            "run-points", PointsEntry.from_dict,
            params={"peer": peer, "episode": episode, "limit": limit, "offset": offset}, run_id=run_id,
        )

    def normalization(self, run_id: str, episode: int) -> Any:
        from ..scoring.normalize import NormalizationReport

        return NormalizationReport.from_dict(
            self._json("GET", "run-episode-normalization", run_id=run_id, episode=episode)
        )

    def scorecard(self, run_id: str, episode: int, peer: str, *, version: int | None = 2) -> Any:
        """One peer's scorecard. ``version=2`` (the default) must carry ``points``: a run scored
        without them raises :class:`ValidationError` rather than returning a v1 card as a v2."""
        accept = {None: None, 1: SCORECARD_V1, 2: SCORECARD_V2}.get(version)
        if version not in (None, 1, 2):
            raise ValidationError(f"unsupported scorecard version {version!r}", status=406)
        headers = {"Accept": accept} if accept else None
        try:
            payload = self._json(
                "GET", "run-episode-peer", headers=headers, run_id=run_id, episode=episode, peer=peer
            )
        except Conflict as exc:
            if version == 2:
                raise ValidationError(
                    f"run {run_id!r} has no points; request version=1 or enable points",
                    status=exc.status, type_uri=exc.type_uri, extensions=exc.extensions,
                ) from exc
            raise
        if version == 2 and payload.get("points") is None:
            raise ValidationError(f"episode {episode} of run {run_id!r} has no points for {peer!r}")
        return _scorecard(payload)


class ObservabilityAPI(_Namespace):
    def logs(
        self, run_id: str, *, n: int | None = None, node_path: str | None = None,
        kind: str | None = None, source: str | None = None,
    ) -> Report:
        return self._report(
            "GET", "run-log-tail",
            params={"n": n, "node_path": node_path, "kind": kind, "source": source}, run_id=run_id,
        )

    def health(self, run_id: str | None = None, *, limit: int | None = None) -> Report:
        """The server's liveness, or a run's watchdog history when ``run_id`` is given."""
        if run_id is None:
            return self._report("GET", "health")
        return self._report("GET", "run-watchdog", params={"limit": limit}, run_id=run_id)

    def snapshot(self, run_id: str) -> Report:
        return self._report("POST", "run-watchdog-snapshot", run_id=run_id)

    def verify_checkpoint(self, run_id: str, episode: int) -> Report:
        return self._report("POST", "run-checkpoint-verify", run_id=run_id, episode=episode)

    def resume(self, run_id: str, *, epochs: int | None = None) -> Report:
        body = _compact({"epochs": epochs})
        return self._report("POST", "run-resume", json=body or None, run_id=run_id)


class MACTFClient:
    """One entry point over either transport.

    Transport precedence: an explicit ``transport``; else ``base_url`` (HTTP); else the
    in-process application for the active settings. ``settings`` applies to the in-process
    transport only.
    """

    def __init__(
        self,
        *,
        transport: Transport | str | None = None,
        settings: Any | None = None,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        if isinstance(transport, str):
            kind = transport.strip().lower().replace("-", "_")
            if kind not in ("in_process", "http"):
                raise ValueError(f"transport must be 'in_process' or 'http', not {transport!r}")
            if kind == "http" and base_url is None:
                raise ValueError("transport='http' needs base_url")
            if kind == "in_process":
                base_url = None
            transport = None
        if transport is None:
            from .transport import HTTPTransport, InProcessTransport

            if base_url is not None:
                transport = HTTPTransport(base_url, token=token, timeout=timeout)
            else:
                transport = InProcessTransport(settings=settings, token=token)
        self.transport: Transport = transport
        self.runs = RunsAPI(self)
        self.scoring = ScoringAPI(self)
        self.rules = RulesAPI(self)
        self.peers = PeersAPI(self)
        self.sandbox = SandboxAPI(self)
        self.trainers = TrainersAPI(self)
        self.scores = ScoresAPI(self)
        self.observability = ObservabilityAPI(self)

    # -- conveniences -----------------------------------------------------------------
    def health(self) -> Report:
        return self.observability.health()

    def meta(self) -> Report:
        return self.runs._report("GET", "meta")

    def openapi(self) -> dict[str, Any]:
        return dict(self.runs._json("GET", "openapi"))

    def wait_for(
        self,
        run_id: str,
        *,
        timeout: float = 300.0,
        poll: float = 0.25,
        until: Callable[[Run], bool] | None = None,
    ) -> Run:
        """Poll ``GET /runs/{id}`` until the run is terminal (or ``until(run)`` is true)."""
        deadline = time.monotonic() + timeout
        while True:
            run = self.runs.get(run_id)
            if run.terminal or (until is not None and until(run)):
                return run
            if time.monotonic() >= deadline:
                raise TimeoutError(f"run {run_id!r} was still {run.status!r} after {timeout:g}s")
            time.sleep(poll)

    def run_to_completion(self, *, timeout: float = 300.0, poll: float = 0.25, **create: Any) -> Run:
        """Start a run and block until it ends; the returned :class:`Run` carries the final status."""
        handle = self.runs.create(**create)
        return self.wait_for(handle.run_id, timeout=timeout, poll=poll)

    def score(self, peer: str, events: Sequence[Any], *, episode: int = 0, run_id: str | None = None) -> Any:
        """Shorthand for :meth:`ScoringAPI.evaluate`."""
        return self.scoring.evaluate(peer, events, episode=episode, run_id=run_id)

    def close(self) -> None:
        closer = getattr(self.transport, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> "MACTFClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
