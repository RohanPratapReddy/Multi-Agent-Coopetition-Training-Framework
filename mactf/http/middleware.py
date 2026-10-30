"""Middleware: request ids, exception conversion, CORS and authentication.

A middleware is any callable following the protocol in document 12 section 7:
``__init__(get_response)`` and ``__call__(request) -> Response``. Instances are
constructed once per application, so per-request state lives on the ``Request``.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import re
import uuid
from typing import Any, Callable, Protocol

from .auth import ANONYMOUS, Denial, Principal
from .exceptions import (
    HttpError,
    NotAuthenticated,
    PermissionDenied,
    Resolver404,
    translate,
)
from .request import Request
from .response import ProblemResponse, Response

__all__ = [
    "AuthMiddleware",
    "CorsMiddleware",
    "ExceptionMiddleware",
    "Middleware",
    "RequestIdMiddleware",
    "import_string",
]

logger = logging.getLogger("mactf.http")

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,128}$")


class Middleware(Protocol):
    def __init__(self, get_response: Callable[[Request], Response]) -> None: ...

    def __call__(self, request: Request) -> Response: ...


def import_string(dotted_path: str) -> Any:
    """``"pkg.module.attr"`` -> the attribute. Errors name the path that failed."""
    module_path, _, attribute = dotted_path.rpartition(".")
    if not module_path:
        raise ImportError(f"{dotted_path!r} is not a dotted path")
    module = importlib.import_module(module_path)
    try:
        return getattr(module, attribute)
    except AttributeError:
        raise ImportError(f"module {module_path!r} has no attribute {attribute!r}") from None


def _settings() -> Any:
    from ..conf import get_settings

    return get_settings()


class _Base:
    def __init__(self, get_response: Callable[[Request], Response]) -> None:
        self.get_response = get_response


class RequestIdMiddleware(_Base):
    """Assigns ``request.request_id`` and echoes it as ``X-Request-Id`` and ``X-Trace-Id``."""

    def __call__(self, request: Request) -> Response:
        supplied = request.headers.get("X-Request-Id", "") or ""
        request.request_id = supplied if _REQUEST_ID.match(supplied) else uuid.uuid4().hex
        response = self.get_response(request)
        response["X-Request-Id"] = request.request_id
        response["X-Trace-Id"] = request.request_id
        return response


class ExceptionMiddleware(_Base):
    """Converts ``HttpError`` and framework exceptions into problem+json."""

    def __call__(self, request: Request) -> Response:
        try:
            return self.get_response(request)
        except Exception as exc:  # noqa: BLE001 - the funnel for every failure
            return convert_exception(request, exc)


_OBSERVERS: list[Callable[[Request, BaseException, int], None]] = []


@contextlib.contextmanager
def observe_exceptions(callback: Callable[[Request, BaseException, int], None]):
    """Call ``callback(request, exc, status)`` for every exception converted to a response.

    This is how a test client re-raises a 500 (``raise_request_exception=True``)
    without the transport growing a debug mode: the conversion still happens, the
    observer just gets to see what was converted.
    """
    _OBSERVERS.append(callback)
    try:
        yield
    finally:
        try:
            _OBSERVERS.remove(callback)
        except ValueError:  # pragma: no cover - removed twice
            pass


def convert_exception(request: Request, exc: BaseException, *, debug: bool | None = None) -> Response:
    """One conversion path, shared by the middleware and the handler's fallback."""
    if debug is None:
        try:
            debug = bool(_settings().get("DEBUG", False))
        except Exception:  # noqa: BLE001
            debug = False
    if isinstance(exc, Resolver404):
        exc = Resolver404(f"no route matches {request.path!r}", path=request.path)
    status = translate(exc)[0]
    trace = getattr(request, "request_id", "") or ""
    for observer in list(_OBSERVERS):
        observer(request, exc, status)
    if status >= 500:
        logger.error(
            "request %s %s failed with %s [trace %s]",
            request.method,
            request.path,
            type(exc).__name__,
            trace,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    else:
        logger.info(
            "request %s %s -> %s %s [trace %s]",
            request.method,
            request.path,
            status,
            type(exc).__name__,
            trace,
        )
    return ProblemResponse(exc, instance=request.path, trace_id=trace, debug=bool(debug))


class CorsMiddleware(_Base):
    """CORS from ``API.CORS_ORIGINS``. A wildcard is never combined with credentials."""

    DEFAULT_METHODS = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    EXPOSED = "X-Request-Id, X-Trace-Id, Retry-After, Location"

    def _origins(self) -> list[str]:
        api = _settings().get("API") or {}
        return [str(origin) for origin in (api.get("CORS_ORIGINS") or [])]

    def __call__(self, request: Request) -> Response:
        origin = request.headers.get("Origin")
        allowed = self._origins()
        if not origin or not allowed:
            return self.get_response(request)
        wildcard = "*" in allowed
        permitted = wildcard or origin in allowed
        preflight = request.method == "OPTIONS" and request.headers.get(
            "Access-Control-Request-Method"
        ) is not None
        if preflight:
            response = Response(b"", 204, content_type=None)
            if permitted:
                self._decorate(response, origin, wildcard)
                response["Access-Control-Allow-Methods"] = (
                    request.headers.get("Access-Control-Request-Method", "") + ", " + self.DEFAULT_METHODS
                ).strip(", ")
                requested = request.headers.get("Access-Control-Request-Headers")
                response["Access-Control-Allow-Headers"] = requested or "Authorization, Content-Type"
                response["Access-Control-Max-Age"] = "600"
            response["Vary"] = "Origin"
            return response
        response = self.get_response(request)
        if permitted:
            self._decorate(response, origin, wildcard)
            response["Access-Control-Expose-Headers"] = self.EXPOSED
        response["Vary"] = "Origin"
        return response

    @staticmethod
    def _decorate(response: Response, origin: str, wildcard: bool) -> None:
        if wildcard:
            response["Access-Control-Allow-Origin"] = "*"
        else:
            response["Access-Control-Allow-Origin"] = origin
            response["Access-Control-Allow-Credentials"] = "true"


class AuthMiddleware(_Base):
    """Resolves ``request.principal`` through ``API.PRINCIPAL_RESOLVER``.

    The resolver is a dotted path to ``callable(request) -> Principal | Denial | None``.
    With none configured every request is anonymous, which is what ``DEBUG``
    projects want. Removing ``mactf.contrib.api`` leaves this importable and inert.
    """

    def __init__(self, get_response: Callable[[Request], Response]) -> None:
        super().__init__(get_response)
        self._resolver: Callable[[Request], Any] | None = None
        self._resolver_path: str | None = None

    def _resolve_callable(self) -> Callable[[Request], Any] | None:
        api = _settings().get("API") or {}
        dotted = api.get("PRINCIPAL_RESOLVER")
        if not dotted:
            return None
        if dotted != self._resolver_path:
            resolver = import_string(dotted) if isinstance(dotted, str) else dotted
            if not callable(resolver):
                raise HttpError(f"API.PRINCIPAL_RESOLVER {dotted!r} is not callable")
            self._resolver, self._resolver_path = resolver, dotted
        return self._resolver

    def __call__(self, request: Request) -> Response:
        resolver = self._resolve_callable()
        if resolver is None:
            request.principal = ANONYMOUS
            return self.get_response(request)
        outcome = resolver(request)
        if isinstance(outcome, Denial):
            headers = dict(outcome.headers)
            if outcome.status == 401:
                headers.setdefault("WWW-Authenticate", "Bearer")
                raise NotAuthenticated(outcome.detail, headers=headers)
            raise PermissionDenied(outcome.detail, headers=headers)
        request.principal = outcome if isinstance(outcome, Principal) else ANONYMOUS
        return self.get_response(request)
