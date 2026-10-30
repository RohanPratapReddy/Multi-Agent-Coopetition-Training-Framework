"""Sandbox providers, status, probe and exec (document 02 section 3.7)."""

from __future__ import annotations

from typing import Any

from ....http.request import Request
from ....http.response import JsonResponse
from ....sandbox.detect import choose_provider, probe_all
from .. import deps
from ..serializers import CapabilitySerializer, ExecSerializer, SandboxExecRequestSerializer
from .base import ApiView, conflict


def _status(request: Request) -> dict[str, Any]:
    sandbox = dict(deps.get_settings_dep(request).get("SANDBOX") or {})
    choice = choose_provider(
        requested=str(sandbox.get("PROVIDER", "auto")),
        isolation=sandbox.get("ISOLATION", "hardware"),
        settings=sandbox,
    )
    report = choice.report.as_dict()
    return {
        "provider": report.get("provider"),
        "isolation": report.get("isolation"),
        "requested": choice.requested,
        "degraded": choice.degraded,
        "reasons": list(choice.reasons) or list(report.get("reasons") or []),
        "capabilities": dict(report.get("capabilities") or {}),
        "usable": report.get("usable"),
        "report": report,
    }


class SandboxProviderListView(ApiView):
    summary = "Every sandbox provider and what it can do on this host"
    response_serializer = CapabilitySerializer

    def get(self, request: Request) -> JsonResponse:
        return JsonResponse([report.as_dict() for report in probe_all()])


class SandboxStatusView(ApiView):
    summary = "The provider a run would use, and why"

    def get(self, request: Request) -> JsonResponse:
        return JsonResponse(_status(request))


class SandboxProbeView(ApiView):
    scopes = {"post": "read"}
    summary = "Force a fresh probe"

    def post(self, request: Request) -> JsonResponse:
        # probe_all() holds no cache; calling it again is by construction a fresh probe.
        return JsonResponse({"providers": [r.as_dict() for r in probe_all()], "status": _status(request)})


class SandboxExecView(ApiView):
    scopes = {"post": "sandbox:exec"}
    summary = "Run a command inside a peer's sandbox"
    request_serializer = {"post": SandboxExecRequestSerializer}
    response_serializer = {"post": ExecSerializer}
    description = (
        "Requires a live handle registered by a run that provisioned that peer. The exec result "
        "is the response, including exit_code 127 when the guest agent is absent."
    )

    def post(self, request: Request) -> JsonResponse:
        body = self.body(request, SandboxExecRequestSerializer)
        handles = deps.sandbox_handles(request)
        entry = handles.get(body["peer"])
        if entry is None:
            reason = f"no active run has provisioned a sandbox for peer {body['peer']!r}"
            status = _status(request)
            if status.get("provider") == "local":
                reason += "; the local provider (process isolation) supports exec only for the current user"
            elif status.get("provider") == "qemu":
                reason += "; qemu needs a serial socket on a started guest"
            raise conflict(reason, peer=body["peer"], reason=reason, registered=sorted(handles))
        provider, handle = entry
        result = provider.exec(handle, body["command"], timeout=body.get("timeout"))
        return JsonResponse(result.as_dict())
