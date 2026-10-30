"""Checkpoints, verification, resume and rollback (document 02 sections 3.3 and 3.11)."""

from __future__ import annotations

import time
from typing import Any

from ....http.exceptions import Http404
from ....http.request import Request
from ....http.response import JsonResponse
from ....orchestration import recovery
from ....orchestration.recovery import checkpoint_entry, verify_checkpoint
from .. import deps
from ..serializers import (
    CheckpointSerializer,
    ResumeRequestSerializer,
    RollbackRequestSerializer,
)
from .base import (
    ApiView,
    conflict,
    get_run_or_404,
    page_params,
    paginate_list,
    run_is_live,
)

_FINAL = tuple(sorted(recovery.FINAL_STATUSES))


def _row_or_404(request: Request, run_id: str, episode: int) -> dict[str, Any]:
    get_run_or_404(request, run_id)
    for row in deps.get_registry_dep(request).checkpoints(run_id):
        if int(row["episode"]) == int(episode):
            return row
    raise Http404(f"run {run_id!r} has no checkpoint for episode {episode}")


def perform_rollback(request: Request, run_id: str, episode: int) -> dict[str, Any]:
    """Make ``episode`` the newest checkpoint of a non-live run, so a resume continues from it.

    The run-id gate and the 404 for an unknown run stay here; everything else is
    ``orchestration.recovery.rollback``, which the CLI calls too.
    """
    get_run_or_404(request, run_id)
    return recovery.rollback(
        deps.get_registry_dep(request), run_id, episode,
        live=run_is_live(request, run_id), source="api",
    )


class CheckpointListView(ApiView):
    summary = "Checkpoints of a run"
    response_serializer = CheckpointSerializer

    def get(self, request: Request, run_id: str) -> JsonResponse:
        get_run_or_404(request, run_id)
        limit, offset = page_params(request)
        deep = request.GET.get("verify") == "true"
        valid_only = request.GET.get("valid_only") == "true"
        entries = [checkpoint_entry(row, deep=deep or valid_only) for row in deps.get_registry_dep(request).checkpoints(run_id)]
        if valid_only:
            entries = [entry for entry in entries if entry["verified"]]
        return JsonResponse(paginate_list(entries, limit, offset))


class CheckpointDetailView(ApiView):
    summary = "One checkpoint, described and verified"
    response_serializer = CheckpointSerializer

    def get(self, request: Request, run_id: str, episode: int) -> JsonResponse:
        return JsonResponse(checkpoint_entry(_row_or_404(request, run_id, episode), deep=True))


class CheckpointVerifyView(ApiView):
    scopes = {"post": "read"}
    summary = "Verify a checkpoint file"

    def post(self, request: Request, run_id: str, episode: int) -> JsonResponse:
        row = _row_or_404(request, run_id, episode)
        return JsonResponse(verify_checkpoint(row["path"], run_id))


class CheckpointRollbackView(ApiView):
    scopes = {"post": "run:write"}
    summary = "Roll a run back to a checkpoint"

    def post(self, request: Request, run_id: str, episode: int) -> JsonResponse:
        return JsonResponse(perform_rollback(request, run_id, episode), 202)


class RunRollbackView(ApiView):
    scopes = {"post": "run:write"}
    summary = "Roll a run back to the checkpoint of a given episode"
    request_serializer = {"post": RollbackRequestSerializer}
    response_serializer = {"post": CheckpointSerializer}

    def post(self, request: Request, run_id: str) -> JsonResponse:
        body = self.body(request, RollbackRequestSerializer)
        return JsonResponse(perform_rollback(request, run_id, body["episode"]), 202)


class RunResumeView(ApiView):
    scopes = {"post": "run:write"}
    summary = "Continue a paused or interrupted run under the same id"
    request_serializer = {"post": ResumeRequestSerializer}
    description = (
        "A terminal (completed, failed, cancelled) run answers 409: restarting it would make its "
        "original scorecard a lie. Branch a new run with POST /runs and resume_from instead."
    )

    def post(self, request: Request, run_id: str) -> JsonResponse:
        row = get_run_or_404(request, run_id)
        body = self.optional_body(request, ResumeRequestSerializer)
        if row["status"] in _FINAL:
            raise conflict(
                "run is terminal; create a new run with resume_from",
                reason="run is terminal; create a new run with resume_from",
                current_status=row["status"],
            )
        registry = deps.get_registry_dep(request)
        before = len(registry.resume_reports(run_id))
        handle = deps.get_run_manager_dep(request).resume(run_id, epochs=body.get("epochs"))
        deadline = time.monotonic() + 5.0
        reports = registry.resume_reports(run_id)
        while len(reports) <= before and not handle.done.is_set() and time.monotonic() < deadline:
            time.sleep(0.05)
            reports = registry.resume_reports(run_id)
        report = reports[-1] if len(reports) > before else None
        return JsonResponse(
            {"run_id": run_id, "status": handle.status, "resume": report, "links": _links(run_id)}, 202
        )


def _links(run_id: str) -> dict[str, str]:
    from ....http.urls import reverse

    return {"self": reverse("api:run-detail", run_id=run_id), "events": reverse("api:run-events", run_id=run_id)}
