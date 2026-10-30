"""Run detail: episodes, rule events, aggregations, warnings, points, normalization (section 3.3)."""

from __future__ import annotations

import copy
from typing import Any

from ....http.exceptions import Conflict, Http404, HttpError, ValidationError
from ....http.request import Request
from ....http.response import JsonResponse
from .. import deps
from ..serializers import (
    AggregationSerializer,
    EpisodeSerializer,
    NormalizationSerializer,
    PointsSerializer,
    RuleEventSerializer,
    ScorecardSerializer,
)
from .base import ApiView, get_run_or_404, int_param, page_params, paginate_list, paginated

SUPPORTED_SCORECARD_VERSIONS = ("1", "2")
_VENDOR = "application/vnd.mactf.v"


class UnsupportedScorecardVersion(HttpError):
    status = 406
    title = "Unsupported scorecard version"
    type_uri = "https://mactf.dev/problems/unsupported-scorecard-version"


def negotiate_scorecard_version(request: Request) -> tuple[str, dict[str, str]]:
    """``(version, extra_headers)`` per document 02 section 4.3.1.

    The ``Accept`` vendor type wins over ``?scorecard_version``; when both are present
    and disagree the response says so with ``Warning: 299``.
    """
    accept = request.headers.get("Accept") or ""
    header_version = None
    for part in accept.split(","):
        media = part.split(";", 1)[0].strip().lower()
        if media.startswith(_VENDOR) and media.endswith("+json"):
            header_version = media[len(_VENDOR): -len("+json")]
            break
    query_version = request.GET.get("scorecard_version")
    chosen = header_version if header_version is not None else query_version
    headers: dict[str, str] = {}
    if chosen is None:
        return "1", headers
    if chosen not in SUPPORTED_SCORECARD_VERSIONS:
        raise UnsupportedScorecardVersion(
            f"scorecard version {chosen!r} is not supported",
            extensions={"supported": list(SUPPORTED_SCORECARD_VERSIONS), "requested": chosen},
        )
    if header_version is not None and query_version is not None and query_version != header_version:
        headers["Warning"] = '299 - "scorecard_version query ignored; Accept header takes precedence"'
    if header_version is not None:
        headers["Content-Type"] = f"application/vnd.mactf.v{header_version}+json"
    return chosen, headers


def project_card(card: dict[str, Any], version: str, normalization: dict[str, Any] | None) -> dict[str, Any]:
    """The v1 card as stored, or its v2 projection (points mandatory, guards flattened)."""
    if version == "1":
        out = dict(card)
        out.setdefault("points", None)
        return out
    if not card.get("points"):
        raise Conflict(
            "this scorecard has no points record; request scorecard version 1",
            extensions={"peer": card.get("peer"), "episode": card.get("episode")},
        )
    out = copy.deepcopy(card)
    out["normalization"] = normalization
    out["guards_denied"] = list((card["points"] or {}).get("denials") or [])
    return out


def project_episode(row: dict[str, Any], version: str) -> dict[str, Any]:
    out = dict(row)
    scorecard = dict(row["scorecard"])
    scorecard["peers"] = {
        name: project_card(card, version, row.get("normalization")) for name, card in scorecard.get("peers", {}).items()
    }
    out["scorecard"] = scorecard
    return out


def _episode_or_404(request: Request, run_id: str, episode: int) -> dict[str, Any]:
    get_run_or_404(request, run_id)
    row = deps.get_registry_dep(request).get_episode(run_id, episode)
    if row is None:
        raise Http404(f"run {run_id!r} has no episode {episode}")
    return row


def _respond(payload: Any, headers: dict[str, str]) -> JsonResponse:
    response = JsonResponse(payload)
    for name, value in headers.items():
        if name == "Content-Type":
            response.headers["Content-Type"] = value
        else:
            response[name] = value
    return response


class EpisodeListView(ApiView):
    summary = "Episodes of a run"
    response_serializer = EpisodeSerializer

    def get(self, request: Request, run_id: str) -> JsonResponse:
        get_run_or_404(request, run_id)
        version, headers = negotiate_scorecard_version(request)
        limit, offset = page_params(request)
        registry = deps.get_registry_dep(request)
        rows = registry.episodes(run_id, limit=limit, offset=offset)
        items = [project_episode(row, version) for row in rows]
        return _respond(paginated(items, registry.count_episodes(run_id), limit, offset), headers)


class EpisodeDetailView(ApiView):
    summary = "One episode"
    response_serializer = EpisodeSerializer

    def get(self, request: Request, run_id: str, episode: int) -> JsonResponse:
        version, headers = negotiate_scorecard_version(request)
        row = _episode_or_404(request, run_id, episode)
        return _respond(project_episode(row, version), headers)


class EpisodePeerView(ApiView):
    summary = "One peer's scorecard for one episode"
    response_serializer = ScorecardSerializer

    def get(self, request: Request, run_id: str, episode: int, peer: str) -> JsonResponse:
        version, headers = negotiate_scorecard_version(request)
        row = _episode_or_404(request, run_id, episode)
        card = (row["scorecard"].get("peers") or {}).get(peer)
        if card is None:
            raise Http404(f"episode {episode} of run {run_id!r} has no peer {peer!r}")
        return _respond(project_card(card, version, row.get("normalization")), headers)


class RuleEventListView(ApiView):
    summary = "Individual rule events (the audit surface)"
    response_serializer = RuleEventSerializer
    description = "Never aggregated: each row is one stored event. Filters: peer, rule_id, episode, severity, source."

    def get(self, request: Request, run_id: str) -> JsonResponse:
        get_run_or_404(request, run_id)
        limit, offset = page_params(request)
        severity = request.GET.get("severity")
        if severity is not None and severity not in ("S0", "S1", "S2", "S3", "S4"):
            raise ValidationError({"severity": ["must be one of S0..S4."]})
        rows, total = deps.get_registry_dep(request).rule_events(
            run_id,
            peer=request.GET.get("peer"),
            rule_id=request.GET.get("rule_id"),
            episode=int_param(request, "episode", None),
            severity=severity,
            source=request.GET.get("source"),
            limit=limit,
            offset=offset,
        )
        return JsonResponse(paginated(rows, total, limit, offset))


class RuleCountsView(ApiView):
    summary = "Rule event counts by rule"

    def get(self, request: Request, run_id: str) -> JsonResponse:
        get_run_or_404(request, run_id)
        return JsonResponse(deps.get_registry_dep(request).rule_counts(run_id))


class AggregationListView(ApiView):
    summary = "Aggregation reports of a run"
    response_serializer = AggregationSerializer

    def get(self, request: Request, run_id: str) -> JsonResponse:
        get_run_or_404(request, run_id)
        limit, offset = page_params(request)
        rows = deps.get_registry_dep(request).aggregations(run_id)
        return JsonResponse(paginate_list(rows, limit, offset))


class WarningListView(ApiView):
    summary = "Active warnings per peer and rule"

    def get(self, request: Request, run_id: str) -> JsonResponse:
        get_run_or_404(request, run_id)
        limit, offset = page_params(request)
        peer = request.GET.get("peer")
        registry = deps.get_registry_dep(request)
        rows = registry.all_warnings(run_id, peer=peer) if request.GET.get("all") == "true" else registry.active_warnings(run_id, peer=peer)
        return JsonResponse(paginate_list(rows, limit, offset))


class PointsView(ApiView):
    summary = "Per-episode, per-peer points for a whole run"
    response_serializer = PointsSerializer
    description = "A peer with no points record is absent: zero is a score, absence is missing data."

    def get(self, request: Request, run_id: str) -> JsonResponse:
        get_run_or_404(request, run_id)
        limit, offset = page_params(request)
        episode = int_param(request, "episode", None)
        peer = request.GET.get("peer")
        rows = deps.get_registry_dep(request).scorecards(run_id, episode=episode)
        items = [
            {"episode": row["episode"], "peer": row["peer"], "final_score": row["final_score"], "points": row["points"]}
            for row in rows
            if row.get("points") is not None and (peer is None or row["peer"] == peer)
        ]
        return JsonResponse(paginate_list(items, limit, offset))


class NormalizationView(ApiView):
    summary = "The cohort normalization report for one episode"
    response_serializer = NormalizationSerializer

    def get(self, request: Request, run_id: str, episode: int) -> JsonResponse:
        row = _episode_or_404(request, run_id, episode)
        report = row.get("normalization")
        if not report:
            raise Conflict(
                "this episode was scored without normalization",
                extensions={"run_id": run_id, "episode": episode, "reason": "normalization disabled"},
            )
        return JsonResponse(report)
