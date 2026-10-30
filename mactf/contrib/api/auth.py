"""Bearer tokens, the scope vocabulary and principal resolution (document 02 section 4.4).

``mactf.http.auth`` owns ``Principal`` and ``Denial`` so the transport can carry an
identity without knowing what a scope means. This module owns the meaning.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol

from ...exceptions import ImproperlyConfigured
from ...http.auth import ANONYMOUS, Denial, Principal
from ...http.exceptions import NotAuthenticated, PermissionDenied
from ...http.middleware import AuthMiddleware as _CoreAuthMiddleware
from ...http.request import Request

#: The complete scope vocabulary. A token granting anything else is a startup error.
SCOPES: tuple[str, ...] = ("read", "run:write", "score:write", "sandbox:exec", "admin", "worker")

#: Scopes an unauthenticated caller may hold. Never a write scope.
ANONYMOUS_SCOPES: frozenset[str] = frozenset({"read"})

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


class TokenResolver(Protocol):
    def resolve(self, token: str) -> Principal | None: ...


def split_token_key(key: str) -> tuple[str, str]:
    """``("plain", secret)`` or ``("sha256", hexdigest)``. An unprefixed key is plain."""
    key = str(key)
    if key.startswith("sha256:"):
        return "sha256", key[len("sha256:"):].lower()
    if key.startswith("plain:"):
        return "plain", key[len("plain:"):]
    return "plain", key


def hash_token(token: str) -> str:
    """The ``sha256:`` form to put in ``API.TOKENS`` instead of the secret."""
    return "sha256:" + hashlib.sha256(token.encode("utf-8")).hexdigest()


def entry_scopes(value: Any) -> list[str]:
    """Scopes from a token entry: ``["read"]`` or ``{"scopes": ["read"], "name": ...}``."""
    if isinstance(value, Mapping):
        value = value.get("scopes", [])
    if isinstance(value, str):
        return [value]
    return [str(scope) for scope in (value or [])]


def validate_tokens(tokens: Mapping[str, Any] | None) -> list[str]:
    """Problems with an ``API.TOKENS`` mapping, as human-readable strings."""
    problems: list[str] = []
    for key, value in (tokens or {}).items():
        kind, secret = split_token_key(key)
        label = f"{kind}:{'*' * 4}"
        if kind == "sha256" and not _HEX64.match(secret):
            problems.append(f"token {label}: a sha256: key must be 64 hex characters")
        if kind == "plain" and not secret:
            problems.append("a token key is empty")
        unknown = sorted(set(entry_scopes(value)) - set(SCOPES))
        if unknown:
            problems.append(f"token {label} grants unknown scope(s) {unknown}; use {list(SCOPES)}")
    return problems


class StaticTokenResolver:
    """Tokens from ``API.TOKENS``: ``{"<token>": ["read", "run:write"]}``.

    Comparison is constant-time, and every configured entry is compared so the time
    taken does not reveal which entry (if any) matched.
    """

    def __init__(self, tokens: Mapping[str, Any] | None = None) -> None:
        self._entries: list[tuple[str, str, str, frozenset[str]]] = []
        for index, (key, value) in enumerate((tokens or {}).items()):
            kind, secret = split_token_key(key)
            name = value.get("name") if isinstance(value, Mapping) else None
            self._entries.append((kind, secret, str(name or f"token-{index}"), frozenset(entry_scopes(value))))

    def __len__(self) -> int:
        return len(self._entries)

    def resolve(self, token: str) -> Principal | None:
        candidate = token.encode("utf-8")
        digest = hashlib.sha256(candidate).hexdigest().encode("ascii")
        found: Principal | None = None
        for kind, secret, name, scopes in self._entries:
            expected = secret.encode("utf-8") if kind == "plain" else secret.encode("ascii", "ignore")
            actual = candidate if kind == "plain" else digest
            if hmac.compare_digest(expected, actual) and found is None:
                found = Principal.authenticated(name, scopes)
        return found


def api_settings(request: Request | None = None) -> Mapping[str, Any]:
    from ...conf import get_settings

    state = getattr(request, "app_state", None) or {}
    settings = state.get("settings") or get_settings()
    return settings.get("API") or {}


def _token_resolver(request: Request) -> StaticTokenResolver:
    """Cached on the app state, keyed by the token mapping so a reload is noticed."""
    state = getattr(request, "app_state", None)
    tokens = api_settings(request).get("TOKENS") or {}
    if state is None:
        return StaticTokenResolver(tokens)
    cached = state.get("_token_resolver")
    if cached is None or cached[0] != tokens:
        cached = (dict(tokens), StaticTokenResolver(tokens))
        state["_token_resolver"] = cached
    return cached[1]


def _external_resolver(request: Request) -> Callable[[Request], Any] | None:
    state = getattr(request, "app_state", None) or {}
    resolver = state.get("principal_resolver")
    if resolver is not None:
        return resolver
    dotted = api_settings(request).get("PRINCIPAL_RESOLVER")
    if not dotted:
        return None
    from ...http.middleware import import_string

    resolver = import_string(dotted) if isinstance(dotted, str) else dotted
    if not callable(resolver):
        raise ImproperlyConfigured(f"API.PRINCIPAL_RESOLVER {dotted!r} is not callable")
    if isinstance(state, dict):
        state["principal_resolver"] = resolver
    return resolver


def resolve_principal(request: Request) -> Principal:
    """Who is calling. Raises ``NotAuthenticated`` for a presented-but-invalid token."""
    existing = getattr(request, "principal", None)
    if isinstance(existing, Principal) and not existing.anonymous:
        return existing

    resolver = _external_resolver(request)
    if resolver is not None:
        outcome = resolver(request)
        if isinstance(outcome, Denial):
            headers = dict(outcome.headers)
            if outcome.status == 401:
                headers.setdefault("WWW-Authenticate", "Bearer")
                raise NotAuthenticated(outcome.detail, headers=headers)
            raise PermissionDenied(outcome.detail, headers=headers)
        if isinstance(outcome, Principal) and not outcome.anonymous:
            return outcome

    token = request.get_bearer_token()
    if token:
        principal = _token_resolver(request).resolve(token)
        if principal is None:
            raise NotAuthenticated("invalid bearer token", headers={"WWW-Authenticate": 'Bearer error="invalid_token"'})
        return principal

    if bool(api_settings(request).get("ALLOW_ANONYMOUS", True)):
        return Principal(name="anonymous", scopes=ANONYMOUS_SCOPES, anonymous=True)
    return ANONYMOUS


def authorize(request: Request, required: Iterable[str]) -> Principal:
    """Resolve the principal and enforce ``required`` scopes (401 vs 403 per document 02)."""
    principal = resolve_principal(request)
    request.principal = principal
    needed = [scope for scope in required if scope]
    missing = [scope for scope in needed if not principal.has_scope(scope)]
    if not missing:
        return principal
    if principal.anonymous:
        raise NotAuthenticated(
            f"authentication required (scope {missing[0]!r})",
            headers={"WWW-Authenticate": "Bearer"},
            extensions={"required_scope": missing[0]},
        )
    raise PermissionDenied(
        f"scope {missing[0]!r} is required",
        extensions={"required_scope": missing[0], "scopes": sorted(principal.scopes)},
    )


class AuthMiddleware(_CoreAuthMiddleware):
    """Resolves ``request.principal`` from ``API.TOKENS`` / ``ALLOW_ANONYMOUS``.

    Views enforce scopes themselves (``ApiView.dispatch``), so installing this
    middleware is optional; it exists so a project can have ``request.principal`` in
    its own views and middleware too.
    """

    def __call__(self, request: Request):
        request.principal = resolve_principal(request)
        return self.get_response(request)
