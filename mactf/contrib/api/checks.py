"""``mactf.api.E001``, ``W002``, ``E003`` and ``E004`` (document 13 section 2.3)."""

from __future__ import annotations

from ...checks import ERROR, WARNING, CheckMessage


def _api_settings() -> dict:
    from ...conf import get_settings

    return dict(get_settings().get("API") or {})


def check_unprotected() -> list[CheckMessage]:
    api = _api_settings()
    if not api.get("ALLOW_ANONYMOUS", True) and not (api.get("TOKENS") or api.get("PRINCIPAL_RESOLVER")):
        return [
            CheckMessage(
                level=ERROR,
                message="API.ALLOW_ANONYMOUS is false and API.TOKENS is empty; no request can authenticate",
                hint="Add a token to API.TOKENS, set API.PRINCIPAL_RESOLVER, or allow anonymous reads.",
                id="mactf.api.E001",
                app="mactf_api",
            )
        ]
    return []


def check_mounted() -> list[CheckMessage]:
    from ...http import reverse
    from ...http.urls import resolve

    try:
        resolve(reverse("api:rule-list"))
    except Exception as exc:  # noqa: BLE001 - any failure means "not mounted"
        return [
            CheckMessage(
                level=WARNING,
                message=f"the API app is installed but its routes are not mounted ({type(exc).__name__})",
                hint='Add path("api/v1/", include("mactf.contrib.api.urls")) to ROOT_URLCONF.',
                id="mactf.api.W002",
                app="mactf_api",
            )
        ]
    return []


def check_route_names() -> list[CheckMessage]:
    from . import urls

    seen: set[str] = set()
    duplicates: list[str] = []
    for pattern in urls.urlpatterns:
        name = getattr(pattern, "name", None)
        if not name:
            continue
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    return [
        CheckMessage(
            level=ERROR,
            message=f"duplicate route name {name!r} in mactf.contrib.api.urls",
            hint="reverse() would be ambiguous and the OpenAPI operationId would collide.",
            id="mactf.api.E003",
            app="mactf_api",
        )
        for name in sorted(set(duplicates))
    ]


def check_tokens() -> list[CheckMessage]:
    from .auth import validate_tokens

    return [
        CheckMessage(
            level=ERROR,
            message=problem,
            hint="Scopes are read, run:write, score:write, sandbox:exec, admin, worker.",
            id="mactf.api.E004",
            app="mactf_api",
        )
        for problem in validate_tokens(_api_settings().get("TOKENS"))
    ]
