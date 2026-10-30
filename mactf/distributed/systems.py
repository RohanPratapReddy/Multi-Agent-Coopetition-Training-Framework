"""System checks for ``mactf.distributed`` (registered by ``DistributedConfig.ready``)."""

from __future__ import annotations

import math

from .. import checks
from ..checks import CheckMessage, ERROR, WARNING

_POSITIVE_FLOATS = ("WORKER_HEARTBEAT_TIMEOUT", "WORKER_TASK_TIMEOUT")


def check_distributed_settings() -> list[CheckMessage]:
    """``mactf.distributed.E001``: every ``DISTRIBUTED`` value has the right type and range."""
    from ..conf import get_settings

    block = get_settings().get("DISTRIBUTED")
    problems: list[str] = []
    if not isinstance(block, dict):
        problems.append("DISTRIBUTED must be a mapping")
    else:
        for key in _POSITIVE_FLOATS:
            value = block.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not (math.isfinite(value) and value > 0):
                problems.append(f"DISTRIBUTED.{key} must be a positive number")
        for key in ("MAX_PEERS", "MAX_WORKERS"):
            value = block.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                problems.append(f"DISTRIBUTED.{key} must be a positive int")
        if not isinstance(block.get("COMPRESS"), bool):
            problems.append("DISTRIBUTED.COMPRESS must be a bool")
        url = block.get("COORDINATOR_URL")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append("DISTRIBUTED.COORDINATOR_URL must be an http(s) URL")
        base = block.get("BASE_PATH", "worker/v1")
        if not isinstance(base, str) or base.startswith("/") or base.endswith("/"):
            problems.append("DISTRIBUTED.BASE_PATH must have no leading or trailing slash")
    return [
        CheckMessage(
            level=ERROR, message=problem, hint="See document 05 section 8.",
            id="mactf.distributed.E001", app="distributed",
        )
        for problem in problems
    ]


def check_worker_tokens() -> list[CheckMessage]:
    """``mactf.distributed.W001``: workers authenticate with a ``worker``-scoped API token."""
    from ..conf import get_settings
    from ..contrib.api.auth import entry_scopes

    settings = get_settings()
    api = settings.get("API") or {}
    if any("worker" in entry_scopes(value) for value in (api.get("TOKENS") or {}).values()):
        return []
    return [
        CheckMessage(
            level=WARNING,
            message="no API token grants the 'worker' scope; workers can only register if anonymous access is allowed for it, which it never is",
            hint='Add a token such as API["TOKENS"] = {"<secret>": ["worker"]}.',
            id="mactf.distributed.W001",
            app="distributed",
        )
    ]


for _check in (check_distributed_settings, check_worker_tokens):
    checks.register(_check, app="distributed", requires="mactf.distributed")
