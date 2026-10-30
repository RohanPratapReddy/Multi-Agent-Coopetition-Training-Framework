"""Components over HTTP (document 17 section 16.3)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ....components import HUB, ComponentResolver, ComponentSpec
from ....components.chain_validation import finding_of, injected_offer, probe_instance, validate_trainer_spec
from ....exceptions import (
    ComponentContractError,
    ComponentError,
    ComponentImportDenied,
    ComponentMissing,
    ComponentParamError,
    ComponentSpecError,
    ImproperlyConfigured,
    MutationError,
)
from ....http.exceptions import Conflict, Http404, ValidationError
from ....http.request import Request
from ....http.response import JsonResponse
from ....orchestration.live import RunNotFound
from .. import deps
from ..serializers import (
    ComponentPatchSerializer,
    ComponentValidateSerializer,
    MutationSerializer,
    TrainerSpecSerializer,
)
from .base import ApiView, bool_param, get_run_or_404, page_params, paginate_list


def _findings(exc: Exception) -> list[dict[str, Any]]:
    """A component error as the same shape ``mactf components validate`` prints."""
    return [finding_of(exc)]


def _missing(exc: ComponentMissing) -> Http404:
    return Http404(str(exc), extensions={"code": exc.code, "known": list(exc.known)})


def _idempotent(request: Request, scope: str, body: Any, produce: Callable[[], JsonResponse]) -> JsonResponse:
    """Replay the stored answer for a repeated ``Idempotency-Key`` with the same body."""
    key = request.headers.get("Idempotency-Key")
    if key is None:
        return produce()
    if not 1 <= len(key) <= 128:
        raise ValidationError({"Idempotency-Key": ["must be 1-128 characters."]})
    cache = deps.get_idempotency_dep(request)
    principal = getattr(getattr(request, "principal", None), "name", "anonymous")
    fingerprint = cache.fingerprint({"scope": scope, "body": body})
    hit = cache.get(f"{principal}:{scope}", key)
    if hit is not None:
        if hit["fingerprint"] != fingerprint:
            raise Conflict(
                "Idempotency-Key was already used with a different request body",
                extensions={"idempotency_key": key},
            )
        return JsonResponse(hit["body"], hit["status"], headers={**hit["headers"], "Idempotent-Replay": "true"})
    response = produce()
    if response.status_code < 300:
        import json

        cache.put(
            f"{principal}:{scope}", key, fingerprint=fingerprint, status=response.status_code,
            body=json.loads(response.content.decode("utf-8")), headers={},
        )
    return response


class ComponentListView(ApiView):
    summary = "Every visible component"
    description = "Filters: kind, origin, locked. Paginated like every list."

    def get(self, request: Request) -> JsonResponse:
        kind, origin = request.GET.get("kind"), request.GET.get("origin")
        locked = bool_param(request, "locked", False)
        rows: list[dict[str, Any]] = []
        for kind_id, entries in HUB.snapshot().items():
            if kind and kind != kind_id:
                continue
            for name, info in entries.items():
                if origin and info.get("origin") != origin:
                    continue
                if locked and not info.get("locked"):
                    continue
                rows.append({"kind": kind_id, "name": name, **info})
        limit, offset = page_params(request)
        return JsonResponse(paginate_list(rows, limit, offset))


class ComponentDetailView(ApiView):
    summary = "One component"

    def get(self, request: Request, kind: str, name: str) -> JsonResponse:
        try:
            return JsonResponse(HUB.registry(kind).get(name).describe())
        except ComponentMissing as exc:
            raise _missing(exc) from None
        except (ImproperlyConfigured, KeyError, ValueError):       # an unknown kind id
            from ....components import all_kind_ids

            raise Http404(
                f"no component kind {kind!r}; kinds: {sorted(all_kind_ids())}",
                extensions={"code": "E001", "known": sorted(all_kind_ids())},
            ) from None


class ComponentValidateView(ApiView):
    scopes = {"post": "read"}
    summary = "Validate a component spec"
    request_serializer = {"post": ComponentValidateSerializer}
    description = (
        "Untrusted: an `import` spec is refused unless COMPONENTS.API_ALLOW_IMPORT allows it, "
        "and a spec that carries an import is never probed. Never mutates anything."
    )

    def post(self, request: Request) -> JsonResponse:
        body = self.body(request, ComponentValidateSerializer)
        raw = {k: v for k, v in body["spec"].items() if v not in (None, {}, "")}
        findings: list[dict[str, Any]] = []
        try:
            spec = ComponentSpec.parse(body["kind"], raw)
            ComponentResolver().validate(spec, trusted=False, inject=injected_offer(body["kind"]))
            if body["probe"] and not spec.import_path:
                findings.extend(
                    item for item in probe_instance(ComponentResolver(), spec) if item["severity"] == "error"
                )
        except ComponentMissing as exc:
            raise _missing(exc) from None
        except (ComponentParamError, ComponentSpecError, ComponentImportDenied, ComponentContractError) as exc:
            findings = _findings(exc)
        except ComponentError as exc:
            findings = _findings(exc)
        if findings:
            raise ValidationError(
                {"spec": [item["message"] for item in findings]},
                extensions={"code": findings[0]["code"], "findings": findings},
            )
        return JsonResponse({"kind": body["kind"], "ok": True, "findings": [], "spec_hash": spec.hash()})


class TrainerValidateView(ApiView):
    scopes = {"post": "read"}
    summary = "Validate a whole trainer spec, reporting every problem"
    request_serializer = {"post": TrainerSpecSerializer}

    def post(self, request: Request) -> JsonResponse:
        body = self.body(request, TrainerSpecSerializer)
        findings = validate_trainer_spec(body, deps.get_settings_dep(request), trusted=False)
        ok = not any(item["severity"] == "error" for item in findings)
        return JsonResponse({"ok": ok, "findings": findings}, status=200 if ok else 422)


class RunComponentsView(ApiView):
    scopes = {"get": "read", "patch": "run:write"}
    operation_ids = {"get": "run-components", "patch": "run-components"}
    summary = "A run's components: show, or queue a change"
    request_serializer = {"patch": ComponentPatchSerializer}

    def get(self, request: Request, run_id: str) -> JsonResponse:
        get_run_or_404(request, run_id)
        return JsonResponse(deps.get_run_manager_dep(request).components(run_id))

    def patch(self, request: Request, run_id: str) -> JsonResponse:
        get_run_or_404(request, run_id)
        body = self.body(request, ComponentPatchSerializer)
        manager = deps.get_run_manager_dep(request)

        def queue() -> JsonResponse:
            try:
                queued = manager.request_components(
                    run_id, body["slot"], body["spec"], trusted=False, boundary=body["boundary"],
                )
            except ComponentMissing as exc:
                raise _missing(exc) from None
            except RunNotFound:
                raise Http404(f"no run {run_id!r}") from None
            return JsonResponse(queued, status=202)

        return _idempotent(request, f"components:{run_id}", body, queue)


class RunMutationView(ApiView):
    scopes = {"post": "run:write"}
    summary = "Simulate or queue a mutation plan"
    request_serializer = {"post": MutationSerializer}
    description = (
        "`dry_run: true` returns the simulation and queues nothing. "
        "Otherwise the plan is applied at the next epoch boundary."
    )

    def post(self, request: Request, run_id: str) -> JsonResponse:
        get_run_or_404(request, run_id)
        body = self.body(request, MutationSerializer)
        manager = deps.get_run_manager_dep(request)

        def run() -> JsonResponse:
            try:
                if body["dry_run"]:
                    report = manager.simulate_mutation(run_id, body["plan"])
                    return JsonResponse(report.as_dict(), status=200 if report.ok else 422)
                queued = manager.request_mutation(
                    run_id, body["plan"], label=body.get("label") or "",
                    reason=body.get("reason") or "", source="api",
                )
            except MutationError as exc:
                raise ValidationError(
                    {"plan": [str(exc)]}, extensions={"code": exc.code, "findings": list(exc.findings)},
                ) from None
            except RunNotFound:
                raise Http404(f"no run {run_id!r}") from None
            return JsonResponse(queued, status=202)

        if body["dry_run"]:
            return run()
        return _idempotent(request, f"mutations:{run_id}", body, run)
