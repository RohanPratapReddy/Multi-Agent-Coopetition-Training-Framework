"""Admin: capabilities and config reload (document 02 section 3.12)."""

from __future__ import annotations

from typing import Any

from .... import conf
from ....http.request import Request
from ....http.response import JsonResponse
from .. import deps
from ..receivers import signal_tally
from .base import ApiView, conflict
from .meta import meta_payload
from .scorers import available_extras


def _writable(path: Any) -> bool:
    import os

    try:
        return os.access(path, os.W_OK)
    except OSError:
        return False


class AdminCapabilitiesView(ApiView):
    scopes = {"get": "admin"}
    summary = "Everything /meta reports plus extras, settings module and writable paths"

    def get(self, request: Request) -> JsonResponse:
        settings = deps.get_settings_dep(request)
        paths: dict[str, Any] = {}
        for key, value in dict(settings.get("PATHS") or {}).items():
            try:
                resolved = settings.path(key)
            except Exception:  # noqa: BLE001 - report the raw value for an unresolvable path
                resolved = value
            paths[key] = {"path": str(resolved), "writable": _writable(resolved)}
        state = deps.app_state(request)
        return JsonResponse(
            {
                **meta_payload(request),
                "available_extras": available_extras(),
                "paths": paths,
                "signals": signal_tally(),
                "live_runs": deps.get_run_manager_dep(request).live_run_ids(),
                "idempotency_entries": len(state["idempotency"]) if state.get("idempotency") is not None else 0,
            }
        )


class ConfigReloadView(ApiView):
    scopes = {"post": "admin"}
    summary = "Reload settings and rebuild the scorer and run manager"
    description = "Refused with 409 while any run is live: swapping settings under a loop makes provenance ambiguous."

    def post(self, request: Request) -> JsonResponse:
        manager = deps.get_run_manager_dep(request)
        live = manager.live_run_ids()
        if live:
            raise conflict("a run is live; settings cannot be swapped under an active loop", live_runs=live)
        state = deps.app_state(request)
        module = getattr(state["settings"], "module", None)
        conf.configure(module if isinstance(module, str) else None, force=True)
        fresh = conf.get_settings()
        old_manager = state.get("run_manager")
        for key in ("run_manager", "scorer", "events", "registry", "idempotency", "_token_resolver"):
            state.pop(key, None)
        if old_manager is not None:
            old_manager.close()
        state["settings"] = fresh
        # The handler (if any) keeps pointing at this same dict, so views see the swap.
        return JsonResponse(meta_payload(request))
