"""Evaluations, ledger append, flags and adjudications (document 02 section 3.5)."""

from __future__ import annotations

import time
from typing import Any

from ....exceptions import RuleViolation
from ....http.exceptions import Http404, ValidationError
from ....http.request import Request
from ....http.response import JsonResponse
from ....rules.catalog import get_rule
from ....rules.severity import Severity
from ....scoring.ledger import Ledger
from ....scoring.pipeline import build_scorer
from .. import deps
from ..serializers import (
    AdjudicationPayloadSerializer,
    EvaluationResponseSerializer,
    LedgerPayloadSerializer,
)
from .base import (
    ApiView,
    conflict,
    get_run_or_404,
    page_params,
    paginate_list,
    run_is_live,
)


def ledger_from_payload(payload: dict[str, Any]) -> Ledger:
    """Build and validate a ledger. ``Ledger.record`` rejects unknown rules and negative magnitudes."""
    ledger = Ledger(peer=payload["peer"])
    try:
        for event in payload["events"]:
            ledger.record(
                event["rule_id"],
                magnitude=event.get("magnitude", 1.0),
                severity=event.get("severity"),
                evidence=event["evidence"],
                source=event.get("source", "api"),
                tags=event.get("tags") or (),
            )
    except ValueError as exc:  # an unparseable severity
        raise ValidationError({"events": [str(exc)]}) from None
    return ledger


def scorecard_response(card: Any) -> dict[str, Any]:
    body = dict(card.to_dict())
    body["explain"] = card.explain()
    return body


class EvaluationCreateView(ApiView):
    scopes = {"post": "score:write"}
    summary = "Score a submitted ledger"
    request_serializer = {"post": LedgerPayloadSerializer}
    response_serializer = {"post": EvaluationResponseSerializer}
    description = (
        "Scores a ledger without a run. With run_id the app-scoped scorer is used, after restoring "
        "that run's stored warnings; otherwise a fresh scorer is used so warnings never leak "
        "between unrelated evaluations."
    )

    def post(self, request: Request) -> JsonResponse:
        payload = self.body(request, LedgerPayloadSerializer)
        ledger = ledger_from_payload(payload)
        settings = deps.get_settings_dep(request)
        if payload.get("run_id"):
            get_run_or_404(request, payload["run_id"])
            scorer = deps.get_scorer_dep(request)
            deps.get_registry_dep(request).restore_warnings(
                payload["run_id"], scorer.registry_for(payload["peer"]), peer=payload["peer"]
            )
        else:
            scorer = build_scorer(settings)
        card = scorer.score_peer(ledger, peer=payload["peer"], episode=payload["episode"])
        return JsonResponse(scorecard_response(card))


class EpisodeLedgerView(ApiView):
    scopes = {"post": "score:write"}
    summary = "Append a ledger to a stored episode and rescore that peer"
    request_serializer = {"post": LedgerPayloadSerializer}
    response_serializer = {"post": EvaluationResponseSerializer}

    def post(self, request: Request, run_id: str, episode: int) -> JsonResponse:
        get_run_or_404(request, run_id)
        if run_is_live(request, run_id):
            raise conflict(
                "the run is live; a scorecard cannot be mutated under an active aggregation",
                run_id=run_id,
            )
        registry = deps.get_registry_dep(request)
        row = registry.get_episode(run_id, episode)
        if row is None:
            raise Http404(f"run {run_id!r} has no episode {episode}")
        payload = self.body(request, LedgerPayloadSerializer)
        peer = payload["peer"]
        if peer not in (row["scorecard"].get("peers") or {}):
            raise Http404(f"episode {episode} of run {run_id!r} has no peer {peer!r}")

        # Validate the new events first: nothing is written when one of them is bad.
        appended = ledger_from_payload(payload)
        stored, _ = registry.rule_events(run_id, peer=peer, episode=episode, limit=100_000)
        ledger = Ledger(peer=peer)
        for old in stored:
            if old.get("reversed_at"):
                continue
            ledger.record(
                old["rule_id"],
                magnitude=old["magnitude"],
                severity=old.get("severity"),
                evidence=old.get("evidence") or "",
                source=old.get("source") or "stored",
            )
        ledger.extend(appended.events)
        registry.record_events(
            run_id,
            episode,
            peer,
            [
                {
                    "rule_id": e.rule_id,
                    "magnitude": e.magnitude,
                    "severity": e.resolved_severity().label,
                    "evidence": e.evidence,
                    "source": e.source,
                }
                for e in appended.events
            ],
        )
        scorer = build_scorer(deps.get_settings_dep(request))
        registry.restore_warnings(run_id, scorer.registry_for(peer), peer=peer)
        card = scorer.score_peer(ledger, peer=peer, episode=episode)
        scorecard = dict(row["scorecard"])
        scorecard["peers"] = {**scorecard["peers"], peer: card.to_dict()}
        registry.record_episode(
            run_id, episode, scorecard, started_at=row.get("started_at"), normalization=row.get("normalization")
        )
        return JsonResponse(scorecard_response(card))


class FlagsView(ApiView):
    summary = "Flag precision and P19 state"
    description = (
        "An accepted flag is an unreversed R3 event; a rejected flag is an unreversed P3 event. "
        "p19_active is true for a peer whose precision is below ENGINE.P_MIN."
    )

    def get(self, request: Request, run_id: str) -> JsonResponse:
        get_run_or_404(request, run_id)
        registry = deps.get_registry_dep(request)
        threshold = float((deps.get_settings_dep(request).get("ENGINE") or {}).get("P_MIN", 0.5))
        per_peer: dict[str, dict[str, Any]] = {}
        for rule_id, key in (("R3", "accepted"), ("P3", "rejected")):
            rows, _ = registry.rule_events(run_id, rule_id=rule_id, limit=1_000_000)
            for row in rows:
                if row.get("reversed_at"):
                    continue
                entry = per_peer.setdefault(row["peer"], {"accepted": 0, "rejected": 0})
                entry[key] += 1
        accepted = rejected = 0
        for peer, entry in per_peer.items():
            submitted = entry["accepted"] + entry["rejected"]
            entry["submitted"] = submitted
            entry["precision"] = entry["accepted"] / submitted if submitted else None
            entry["p19_active"] = bool(submitted and entry["precision"] < threshold)
            accepted += entry["accepted"]
            rejected += entry["rejected"]
        submitted = accepted + rejected
        return JsonResponse(
            {
                "submitted": submitted,
                "accepted": accepted,
                "precision": accepted / submitted if submitted else None,
                "threshold": threshold,
                "p19_active": any(entry["p19_active"] for entry in per_peer.values()),
                "per_peer": per_peer,
            }
        )


class AdjudicationView(ApiView):
    """Backs two named routes on one path: ``run-adjudication-create`` and ``-list``."""

    scopes = {"get": "read", "post": "score:write"}
    operation_ids = {"get": "run-adjudication-list", "post": "run-adjudication-create"}
    summary = {"get": "Adjudication history", "post": "Record an adjudication"}
    request_serializer = {"post": AdjudicationPayloadSerializer}

    def get(self, request: Request, run_id: str) -> JsonResponse:
        get_run_or_404(request, run_id)
        limit, offset = page_params(request)
        return JsonResponse(paginate_list(deps.get_registry_dep(request).adjudications(run_id), limit, offset))

    def post(self, request: Request, run_id: str) -> JsonResponse:
        get_run_or_404(request, run_id)
        payload = self.body(request, AdjudicationPayloadSerializer)
        try:
            rule = get_rule(payload["rule_id"])
        except RuleViolation as exc:
            raise ValidationError({"rule_id": [str(exc)]}) from None
        if payload.get("severity"):
            Severity.parse(payload["severity"])
        registry = deps.get_registry_dep(request)
        adjudication_id = registry.record_adjudication(run_id, **payload)
        effects: dict[str, Any] = {"events_reversed": 0, "warnings_expunged": 0}
        if payload["decision"] == "reversed":
            # The audit trail is append-only: events are marked, never deleted (W5).
            effects["events_reversed"] = registry.mark_events_reversed(
                run_id, payload["peer"], rule.id, payload["episode"], by=payload["adjudicator"]
            )
            registry.expire_warning(run_id, payload["peer"], rule.id, payload["episode"], expired_at=time.time())
            scorer = deps.get_scorer_dep(request)
            effects["warnings_expunged"] = scorer.registry_for(payload["peer"]).expunge(payload["peer"], rule.id)
        body = {"id": adjudication_id, "run_id": run_id, **payload, "effects": effects}
        return JsonResponse(body, 201)
