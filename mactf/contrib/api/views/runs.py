"""Runs: create, list, detail, cancel, summary (document 02 section 3.2)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ....exceptions import ImproperlyConfigured, ResourceNotFound
from ....http.exceptions import Conflict, Http404, ValidationError
from ....http.request import Request
from ....http.response import JsonResponse
from ....http.urls import reverse
from ....orchestration import recovery
from ....orchestration.runview import summarize_run
from ....runtime import resolve_device
from .. import deps
from ..serializers import RunAcceptedSerializer, RunCreateRequestSerializer, RunSerializer
from .base import (
    ApiView,
    get_run_or_404,
    page_params,
    paginated,
    run_payload,
)

_RUN_CONFIG_FIELDS = ("epochs", "peers", "device", "seed", "dry_run", "redteam", "task_source")


def _links(run_id: str) -> dict[str, str]:
    return {
        "self": reverse("api:run-detail", run_id=run_id),
        "events": reverse("api:run-events", run_id=run_id),
    }


def _validate_trainer(spec: dict[str, Any], settings: Any) -> None:
    """Construct (without building) a trainer so a bad spec is a 422 now, not a failure later."""
    from ....training.trainer import MACTFTrainer
    from .scorers import trainer_kwargs

    kwargs = trainer_kwargs(spec, settings)
    if "hierarchy" not in kwargs:
        # Construction validates the spec; nothing is built, so no module is needed yet.
        kwargs["model_factory"] = lambda name: None
    try:
        MACTFTrainer(**kwargs)
    except ImproperlyConfigured as exc:
        raise ValidationError({"trainer": [str(exc)]}) from None


def _branch_checkpoint(request: Request, source_run: str, resume_from: str) -> tuple[Path, int]:
    """The checkpoint file a new run is seeded from: ``latest`` or an episode of ``source_run``."""
    try:
        return recovery.branch_checkpoint(deps.get_registry_dep(request), source_run, resume_from)
    except ResourceNotFound as exc:
        raise Http404(str(exc)) from None


class RunListView(ApiView):
    scopes = {"get": "read", "post": "run:write"}
    operation_ids = {"get": "run-list", "post": "run-list"}
    summary = {"get": "List runs", "post": "Start a run"}
    request_serializer = {"post": RunCreateRequestSerializer}
    response_serializer = {"post": RunAcceptedSerializer}
    description = {
        "post": "Validates the request, then queues a run on a background thread. Replays the "
        "original 202 for a repeated Idempotency-Key. When resume_from is set, run_id names the "
        "SOURCE run whose checkpoint seeds a NEW run."
    }

    def get(self, request: Request) -> JsonResponse:
        limit, offset = page_params(request)
        registry = deps.get_registry_dep(request)
        manager = deps.get_run_manager_dep(request)
        items = []
        for row in registry.list_runs(limit=limit, offset=offset):
            merged = manager.get(row["id"]) or row
            items.append(run_payload(merged))
        return JsonResponse(paginated(items, registry.count_runs(), limit, offset))

    def post(self, request: Request) -> JsonResponse:
        body = self.body(request, RunCreateRequestSerializer)
        settings = deps.get_settings_dep(request)

        if body.get("device"):
            try:
                resolve_device(body["device"])
            except Exception as exc:  # noqa: BLE001 - the library decides which devices are real
                raise ValidationError({"device": [str(exc)]}) from None
        declared = settings.get("PEERS") or []
        if body.get("peers") is not None and declared and body["peers"] > len(declared):
            raise ValidationError({"peers": [f"only {len(declared)} peer(s) are configured."]})
        if body.get("trainer") is not None:
            _validate_trainer(body["trainer"], settings)

        key = request.headers.get("Idempotency-Key")
        cache = deps.get_idempotency_dep(request)
        principal = getattr(getattr(request, "principal", None), "name", "anonymous")
        fingerprint = cache.fingerprint(body)
        if key is not None:
            if not 1 <= len(key) <= 128:
                raise ValidationError({"Idempotency-Key": ["must be 1-128 characters."]})
            hit = cache.get(principal, key)
            if hit is not None:
                if hit["fingerprint"] != fingerprint:
                    raise Conflict(
                        "Idempotency-Key was already used with a different request body",
                        extensions={"idempotency_key": key},
                    )
                return JsonResponse(
                    hit["body"], hit["status"], headers={**hit["headers"], "Idempotent-Replay": "true"}
                )

        config = {name: body[name] for name in _RUN_CONFIG_FIELDS if body.get(name) not in (None, False)}
        resumed_episode = None
        if body.get("resume_from") is not None:
            if not body.get("run_id"):
                raise ValidationError({"run_id": ["run_id must name the source run when resume_from is set."]})
            path, resumed_episode = _branch_checkpoint(request, body["run_id"], body["resume_from"])
            config["resume_from"] = path
        elif body.get("run_id"):
            config["run_id"] = body["run_id"]

        manager = deps.get_run_manager_dep(request)
        handle = manager.start(config)
        if body.get("tags"):
            handle.components["tags"] = dict(body["tags"])
        links = _links(handle.run_id)
        payload = RunAcceptedSerializer(
            {"run_id": handle.run_id, "status": handle.status, "created_at": time.time(), "links": links}
        ).to_representation()
        if resumed_episode is not None:
            payload["resumed_from"] = {"run_id": body["run_id"], "episode": resumed_episode}
        headers = {"Location": links["self"]}
        if key is not None:
            cache.put(principal, key, fingerprint=fingerprint, status=202, body=payload, headers=headers)
        return JsonResponse(payload, 202, headers=headers)


class RunDetailView(ApiView):
    summary = "One run"
    response_serializer = RunSerializer

    def get(self, request: Request, run_id: str) -> JsonResponse:
        return JsonResponse(run_payload(get_run_or_404(request, run_id)))


class RunCancelView(ApiView):
    scopes = {"post": "run:write"}
    summary = "Request cooperative cancellation"
    description = (
        "Sets the cancel flag and waits up to API.CANCEL_GRACE_SECONDS for an epoch boundary. "
        "There is no force option: killing a thread mid-step corrupts optimizer state."
    )

    def post(self, request: Request, run_id: str) -> JsonResponse:
        manager = deps.get_run_manager_dep(request)
        get_run_or_404(request, run_id)
        outcome = manager.cancel(run_id)
        return JsonResponse({"run_id": run_id, "status": outcome}, 202)


class RunSummaryView(ApiView):
    summary = "A run, its latest episode and its latest aggregation"

    def get(self, request: Request, run_id: str) -> JsonResponse:
        row = get_run_or_404(request, run_id)
        registry = deps.get_registry_dep(request)
        handle = deps.get_run_manager_dep(request).get_handle(run_id)
        payload = summarize_run(registry, run_id, row=row)
        payload["tags"] = dict(handle.components.get("tags", {})) if handle is not None else {}
        return JsonResponse(payload)
