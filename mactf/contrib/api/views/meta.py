"""Health, metadata, the rule catalog, scales, OpenAPI and system checks (document 02 section 3.1)."""

from __future__ import annotations

import time
from typing import Any

from .... import __version__
from ....checks import run_checks
from ....exceptions import RuleViolation
from ....http.exceptions import Http404, ValidationError
from ....http.openapi import schema as openapi_schema
from ....http.request import Request
from ....http.response import JsonResponse
from ....rules.catalog import CATALOG, TIERS, Rule, get_rule, validate_catalog
from ....rules.severity import Severity, weight_for
from .. import deps
from ..serializers import (
    CatalogValidationSerializer,
    RuleSerializer,
    SeveritySerializer,
    TierSerializer,
)
from .base import ApiView, int_param


def rule_dict(rule: Rule) -> dict[str, Any]:
    return {
        "id": rule.id,
        "name": rule.name,
        "kind": rule.kind,
        "tier": rule.tier,
        "tier_name": rule.tier_name,
        "default_severity": rule.default_severity.label,
        "bucket": rule.bucket,
        "description": rule.description,
        "warning_eligible": rule.warning_eligible,
        "non_offsettable": rule.non_offsettable,
        "omega": rule.omega,
        "tags": sorted(rule.tags),
        "is_reward": rule.is_reward,
        "intent": "reward" if rule.is_reward else "deter",
    }


def sandbox_summary(request: Request) -> dict[str, Any]:
    """The *effective* sandbox: what ``choose_provider`` would actually pick."""
    from ....sandbox.detect import choose_provider

    sandbox = dict(deps.get_settings_dep(request).get("SANDBOX") or {})
    try:
        choice = choose_provider(
            requested=str(sandbox.get("PROVIDER", "auto")),
            isolation=sandbox.get("ISOLATION", "hardware"),
            settings=sandbox,
        )
    except Exception as exc:  # noqa: BLE001 - /meta never fails because a sandbox is missing
        return {"usable": False, "provider": None, "isolation": None, "reasons": [f"{type(exc).__name__}: {exc}"]}
    report = choice.report.as_dict()
    return {
        "provider": report.get("provider"),
        "isolation": report.get("isolation"),
        "usable": report.get("usable"),
        "requested": choice.requested,
        "degraded": choice.degraded,
        "reasons": list(choice.reasons) or list(report.get("reasons") or []),
    }


def meta_payload(request: Request) -> dict[str, Any]:
    from ....runtime import build_runtime

    settings = deps.get_settings_dep(request)
    try:
        runtime = build_runtime(settings).describe()
    except Exception as exc:  # noqa: BLE001
        runtime = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "version": __version__,
        "api_version": "v1",
        "settings_module": getattr(settings, "module", None),
        "runtime": runtime,
        "rule_count": len(CATALOG),
        "sandbox": sandbox_summary(request),
    }


class HealthView(ApiView):
    """Liveness. Loads no settings and opens no database, so a probe can call it freely."""

    public = True
    scopes = {}
    summary = "Liveness probe"

    def get(self, request: Request) -> JsonResponse:
        state = getattr(request, "app_state", None) or {}
        started = float(state.get("started_at") or time.time())
        return JsonResponse({"status": "ok", "version": __version__, "uptime": max(0.0, time.time() - started)})


class MetaView(ApiView):
    summary = "What this server can do"

    def get(self, request: Request) -> JsonResponse:
        return JsonResponse(meta_payload(request))


class RuleListView(ApiView):
    summary = "The rule catalog"
    query_serializer = None

    def get(self, request: Request) -> JsonResponse:
        kind = request.GET.get("kind")
        if kind is not None and kind not in ("reward", "penalty"):
            raise ValidationError({"kind": ["must be 'reward' or 'penalty'."]})
        tier = int_param(request, "tier", None, minimum=0, maximum=max(TIERS))
        rules = [
            rule for rule in CATALOG.values()
            if (kind is None or rule.kind == kind) and (tier is None or rule.tier == tier)
        ]
        return JsonResponse([RuleSerializer(rule_dict(r)).to_representation() for r in rules])


class RuleDetailView(ApiView):
    summary = "One rule"

    def get(self, request: Request, rule_id: str) -> JsonResponse:
        try:
            rule = get_rule(rule_id)
        except RuleViolation:
            raise Http404(f"no rule {rule_id!r}") from None
        return JsonResponse(rule_dict(rule))


class RuleValidationView(ApiView):
    summary = "Self-check of the rule catalog"

    def get(self, request: Request) -> JsonResponse:
        problems = validate_catalog()
        return JsonResponse(CatalogValidationSerializer({"ok": not problems, "problems": problems}).to_representation())


class SeverityListView(ApiView):
    summary = "Severity labels with their configured weights"

    def get(self, request: Request) -> JsonResponse:
        weights = dict(deps.get_settings_dep(request)["SEVERITY"]["WEIGHTS"])
        rows = [
            SeveritySerializer(
                {"label": s.label, "value": int(s), "weight": float(weight_for(s, weights))}
            ).to_representation()
            for s in Severity
        ]
        return JsonResponse(rows)


class TierListView(ApiView):
    summary = "Precedence tiers"

    def get(self, request: Request) -> JsonResponse:
        return JsonResponse([TierSerializer({"tier": t, "name": n}).to_representation() for t, n in TIERS.items()])


class ScoreScaleView(ApiView):
    summary = "The 0-100 points scale"

    def get(self, request: Request) -> JsonResponse:
        from ....scoring.points import PointsTable

        table = PointsTable.from_settings(dict(deps.get_settings_dep(request).as_dict()))
        return JsonResponse(table.to_dict())


class OpenApiView(ApiView):
    summary = "The OpenAPI 3 document for this server"

    def get(self, request: Request) -> JsonResponse:
        return JsonResponse(openapi_schema())


class SystemChecksView(ApiView):
    summary = "Run the system checks"
    scopes = {"post": "read"}

    def post(self, request: Request) -> JsonResponse:
        result = run_checks()
        return JsonResponse(
            {
                "ok": result.ok,
                "summary": result.summary(),
                "messages": [
                    {"level": m.level, "id": m.id, "message": m.message, "hint": m.hint, "app": m.app}
                    for m in result.messages
                ],
            }
        )
