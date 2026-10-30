"""The six worker routes (document 05 section 6).

They reuse the API app's view base, token table and problem+json, but demand the ``worker``
scope, so a worker credential can neither read scorecards nor start runs. Nothing here
computes anything: each route validates, calls the process's :class:`Coordinator`, and encodes.
"""

from __future__ import annotations

from typing import Any

from ..contrib.api import deps
from ..contrib.api.views.base import ApiView
from ..exceptions import DistributedError, ParameterIntegrityError
from ..http.exceptions import Http404, ValidationError, register_handler
from ..http.request import Request
from ..http.response import JsonResponse
from .coordinator import Coordinator, UnknownWorker, WorkerConflict, get_coordinator
from .protocol import TaskResult, WorkerHeartbeat, WorkerRegistration

MAX_LONG_POLL = 20.0


def _problem(status: int, type_uri: str, title: str):
    def handler(exc: BaseException) -> tuple:
        return (status, type_uri, title, str(exc), {}, {})

    return handler


register_handler(DistributedError, _problem(409, "urn:mactf:error:conflict", "Conflict"))
register_handler(WorkerConflict, _problem(409, "urn:mactf:error:conflict", "Conflict"))
register_handler(UnknownWorker, _problem(404, "urn:mactf:error:not-found", "Unknown worker"))
register_handler(ParameterIntegrityError, _problem(422, "urn:mactf:error:validation", "Parameter integrity"))


def coordinator_for(request: Request) -> Coordinator:
    return get_coordinator(deps.get_settings_dep(request))


def _load(request: Request, cls: Any) -> Any:
    payload = request.json()
    if not isinstance(payload, dict):
        raise ValidationError({"body": ["must be a JSON object."]})
    try:
        return cls.from_dict(payload)
    except KeyError as error:
        raise ValidationError({str(error.args[0]).split(".")[-1]: ["is required."]}) from None
    except (TypeError, ValueError) as error:
        raise ValidationError({"body": [str(error)]}) from None


def _wait(request: Request, payload: dict | None = None) -> float:
    raw = (payload or {}).get("wait", request.GET.get("wait"))
    try:
        value = float(raw) if raw not in (None, "") else MAX_LONG_POLL
    except (TypeError, ValueError):
        raise ValidationError({"wait": ["must be a number."]}) from None
    return max(0.0, min(value, MAX_LONG_POLL))


class _WorkerView(ApiView):
    tags = ("worker",)


class RegisterView(_WorkerView):
    scopes = {"post": "worker"}
    summary = "Register a worker; returns its id and the run's peer layout"

    def post(self, request: Request) -> JsonResponse:
        registration = _load(request, WorkerRegistration)
        return JsonResponse(dict(coordinator_for(request).register(registration)))


class HeartbeatView(_WorkerView):
    scopes = {"post": "worker"}
    summary = "Liveness and load"

    def post(self, request: Request) -> JsonResponse:
        heartbeat = _load(request, WorkerHeartbeat)
        return JsonResponse(dict(coordinator_for(request).heartbeat(heartbeat)))


class TasksView(_WorkerView):
    scopes = {"get": "worker"}
    summary = "Long-poll for an assignment (up to 20 s)"

    def get(self, request: Request) -> JsonResponse:
        worker_id = request.GET.get("worker_id")
        if not worker_id:
            raise ValidationError({"worker_id": ["is required."]})
        task = coordinator_for(request).next_task(worker_id, wait=_wait(request))
        return JsonResponse({"task": task.to_dict() if task is not None else None})


class ResultsView(_WorkerView):
    scopes = {"post": "worker"}
    summary = "Submit a TaskResult"

    def post(self, request: Request) -> JsonResponse:
        result = _load(request, TaskResult)
        status = coordinator_for(request).submit_result(result)
        return JsonResponse({"status": status, "task_id": result.task_id})


class ExchangeView(_WorkerView):
    scopes = {"post": "worker"}
    summary = "Fetch the aggregated vector for an epoch"

    def post(self, request: Request) -> JsonResponse:
        payload = request.json()
        if not isinstance(payload, dict):
            raise ValidationError({"body": ["must be a JSON object."]})
        for key in ("worker_id", "run_id", "epoch"):
            if key not in payload:
                raise ValidationError({key: ["is required."]})
        coordinator = coordinator_for(request)
        if payload["worker_id"] not in {m["worker_id"] for m in coordinator.members()}:
            raise Http404(f"worker {payload['worker_id']!r} is not registered")
        try:
            epoch = int(payload["epoch"])
        except (TypeError, ValueError):
            raise ValidationError({"epoch": ["must be an integer."]}) from None
        found = coordinator.exchange(str(payload["run_id"]), epoch, wait=_wait(request, payload))
        return JsonResponse({"exchange": found.to_dict() if found is not None else None})


class DeregisterView(_WorkerView):
    scopes = {"post": "worker"}
    summary = "Leave cleanly"

    def post(self, request: Request) -> JsonResponse:
        payload = request.json()
        worker_id = payload.get("worker_id") if isinstance(payload, dict) else None
        if not worker_id:
            raise ValidationError({"worker_id": ["is required."]})
        coordinator_for(request).deregister(str(worker_id))
        return JsonResponse({"status": "deregistered", "worker_id": worker_id})


register = RegisterView.as_view()
heartbeat = HeartbeatView.as_view()
tasks = TasksView.as_view()
results = ResultsView.as_view()
exchange = ExchangeView.as_view()
deregister = DeregisterView.as_view()
