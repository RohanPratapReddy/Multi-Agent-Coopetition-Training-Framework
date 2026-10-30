"""Route paths for a client process that does not install the API app.

Paths normally come from ``reverse("api:<name>")`` against the project's own URL conf, so a
client in a project that mounts the API elsewhere follows the mount. A *remote* client has
no such URL conf; for it, :func:`path_for` falls back to this table. The table is not a
second source of truth: ``tests/client/test_routes.py`` fails if it differs from what the
API app's own ``reverse`` produces.
"""

from __future__ import annotations

import urllib.parse
from typing import Any

BASE_PATH = "/api/v1"

ROUTES: dict[str, str] = {
    "health": "health",
    "meta": "meta",
    "rule-list": "rules",
    "rule-validation": "rules/validation",
    "rule-detail": "rules/{rule_id}",
    "severity-list": "severities",
    "tier-list": "tiers",
    "score-scale": "scores/scale",
    "openapi": "openapi.json",
    "system-checks": "system-checks",
    "run-list": "runs",
    "run-detail": "runs/{run_id}",
    "run-cancel": "runs/{run_id}/cancel",
    "run-summary": "runs/{run_id}/summary",
    "run-episode-list": "runs/{run_id}/episodes",
    "run-episode-detail": "runs/{run_id}/episodes/{episode:int}",
    "run-episode-peer": "runs/{run_id}/episodes/{episode:int}/peers/{peer}",
    "run-episode-normalization": "runs/{run_id}/episodes/{episode:int}/normalization",
    "run-rule-event-list": "runs/{run_id}/rule-events",
    "run-rule-counts": "runs/{run_id}/rule-counts",
    "run-aggregation-list": "runs/{run_id}/aggregations",
    "run-checkpoint-list": "runs/{run_id}/checkpoints",
    "run-rollback": "runs/{run_id}/rollback",
    "run-warning-list": "runs/{run_id}/warnings",
    "run-points": "runs/{run_id}/points",
    "run-events": "runs/{run_id}/events",
    "evaluation-create": "evaluations",
    "run-episode-ledger": "runs/{run_id}/episodes/{episode:int}/ledger",
    "run-flags": "runs/{run_id}/flags",
    "run-adjudication-create": "runs/{run_id}/adjudications",
    "run-adjudication-list": "runs/{run_id}/adjudications",
    "peer-list": "peers",
    "peer-detail": "peers/{name}",
    "peer-contract": "peers/{name}/contract",
    "weight-preview": "weights/preview",
    "sandbox-provider-list": "sandbox/providers",
    "sandbox-status": "sandbox/status",
    "sandbox-probe": "sandbox/probe",
    "sandbox-exec": "sandbox/exec",
    "scorer-list": "scorers",
    "trainer-capabilities": "trainers/capabilities",
    "trainer-preview": "trainers/preview",
    "trainer-preset-list": "trainers/presets",
    "run-peer-tree": "runs/{run_id}/peers/tree",
    "run-peer-node": "runs/{run_id}/peers/{path:path}",
    "run-infer": "runs/{run_id}/infer",
    "peer-infer": "peers/{path:path}/infer",
    "run-log-tail": "runs/{run_id}/logs/tail",
    "run-watchdog": "runs/{run_id}/watchdog",
    "run-watchdog-snapshot": "runs/{run_id}/watchdog/snapshot",
    "run-checkpoint-detail": "runs/{run_id}/checkpoints/{episode:int}",
    "run-checkpoint-verify": "runs/{run_id}/checkpoints/{episode:int}/verify",
    "run-resume": "runs/{run_id}/resume",
    "run-checkpoint-rollback": "runs/{run_id}/checkpoints/{episode:int}/rollback",
    "admin-capabilities": "admin/capabilities",
    "admin-config-reload": "admin/config/reload",
}


def expand(template: str, /, **values: Any) -> str:
    """Fill a ``{name}`` / ``{name:int}`` / ``{name:path}`` template, quoting each value."""
    out: list[str] = []
    rest = template
    while True:
        start = rest.find("{")
        if start < 0:
            out.append(rest)
            break
        end = rest.index("}", start)
        out.append(rest[:start])
        name, _, kind = rest[start + 1 : end].partition(":")
        if name not in values:
            raise KeyError(f"missing path argument {name!r} for {template!r}")
        value = str(values[name])
        out.append(urllib.parse.quote(value, safe="/" if kind == "path" else ""))
        rest = rest[end + 1 :]
    return "".join(out)


def path_for(route: str, /, **values: Any) -> str:
    """``reverse("api:<name>")`` when the API app is mounted locally, else the table."""
    try:
        from ..http.urls import reverse

        return reverse(f"api:{route}", **values)
    except Exception:  # noqa: BLE001 - no local URL conf mounts the API: use the documented base
        template = ROUTES[route]
        return f"{BASE_PATH}/{expand(template, **values)}"
