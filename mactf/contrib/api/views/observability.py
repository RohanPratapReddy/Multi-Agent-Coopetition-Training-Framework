"""Logs, watchdog and operator snapshots (document 02 section 3.10)."""

from __future__ import annotations

from typing import Any

from ....http.exceptions import ValidationError
from ....http.request import Request
from ....http.response import JsonResponse
from ....orchestration.live import jsonable
from ....training.logging import run_log_location, tail_jsonl
from ....training.watchdog import HealthContext
from .. import deps
from ..serializers import (
    CheckpointSerializer,
    HealthReportSerializer,
    LogRecordSerializer,
    SnapshotRequestSerializer,
)
from .base import ApiView, conflict, get_run_or_404, int_param


class LogTailView(ApiView):
    summary = "The last log records of a run"
    response_serializer = LogRecordSerializer
    description = (
        "?source=memory (default) reads the bounded in-memory sink; ?source=file reads the run's "
        "JSONL (see run_log_location). Neither reads the registry, so a tail never contends with training."
    )

    def get(self, request: Request, run_id: str) -> JsonResponse:
        get_run_or_404(request, run_id)
        api = deps.get_settings_dep(request).get("API") or {}
        default = int(api.get("LOG_TAIL_DEFAULT", 100))
        ceiling = int(api.get("LOG_TAIL_MAX", 2000))
        n = int(int_param(request, "n", default, minimum=1, maximum=ceiling) or default)
        node_path = request.GET.get("node_path")
        kind = request.GET.get("kind")
        source = request.GET.get("source", "memory")
        if source not in ("memory", "file"):
            raise ValidationError({"source": ["must be 'memory' or 'file'."]})
        if source == "file":
            location = run_log_location(deps.get_settings_dep(request), run_id)
            records = tail_jsonl(location.path, n, node_path=node_path, kind=kind,
                                 run_id=run_id if location.shared else None)
            return JsonResponse({"run_id": run_id, "source": "file", "path": str(location.path),
                                 "items": records})
        logger = deps.get_run_manager_dep(request).component(run_id, "logger")
        if logger is None:
            return JsonResponse({"run_id": run_id, "source": "memory", "live": False, "items": []})
        records = [record.as_dict() for record in logger.tail(n, node_path=node_path, kind=kind)]
        return JsonResponse({"run_id": run_id, "source": "memory", "live": True, "items": records})


class WatchdogView(ApiView):
    summary = "The last health report and recent history"
    response_serializer = HealthReportSerializer

    def get(self, request: Request, run_id: str) -> JsonResponse:
        get_run_or_404(request, run_id)
        api = deps.get_settings_dep(request).get("API") or {}
        limit = int(
            int_param(request, "limit", int(api.get("HEALTH_HISTORY_LIMIT", 50)), minimum=1, maximum=512) or 50
        )
        watchdog = deps.get_run_manager_dep(request).component(run_id, "watchdog")
        if watchdog is None:
            return JsonResponse({"run_id": run_id, "live": False, "last": None, "history": []})
        history = [jsonable(report.as_dict()) for report in watchdog.health_history(limit)]
        return JsonResponse(
            {"run_id": run_id, "live": True, "last": history[-1] if history else None, "history": history}
        )


class WatchdogSnapshotView(ApiView):
    scopes = {"post": "run:write"}
    summary = "Take an operator snapshot now"
    request_serializer = {"post": SnapshotRequestSerializer}
    response_serializer = {"post": CheckpointSerializer}

    def post(self, request: Request, run_id: str) -> JsonResponse:
        row = get_run_or_404(request, run_id)
        manager = deps.get_run_manager_dep(request)
        body = self.optional_body(request, SnapshotRequestSerializer)
        if row["status"] in ("completed", "failed", "cancelled", "dry_run"):
            raise conflict("the run is terminal; there is nothing to snapshot", current_status=row["status"])
        watchdog = manager.component(run_id, "watchdog")
        if watchdog is None:
            raise conflict(
                "no live handle with a watchdog is attached to this run",
                reason="no live watchdog", run_id=run_id,
            )
        handle = manager.get_handle(run_id)
        context = HealthContext(epoch=int(handle.epochs_completed if handle is not None else 0))
        record = watchdog.snapshot(context, reason=body["reason"])
        checkpoint = manager.component(run_id, "checkpoint") or getattr(watchdog, "checkpoint", None)
        describe = getattr(checkpoint, "describe", None)
        path = getattr(record, "path", None)
        payload = describe(path) if callable(describe) and path is not None else jsonable(record)
        manager._publish(run_id, "checkpoint", {"reason": body["reason"], "episode": context.epoch})
        return JsonResponse(jsonable(payload))
