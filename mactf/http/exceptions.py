"""HTTP errors, the framework-exception translation table, and problem+json.

``mactf.exceptions`` classes are not re-declared here. :func:`register_handler`
maps them (and any app's own exceptions) onto a status and a ``type`` URI, which is
how ``mactf.contrib.api`` maps its own denials without this layer knowing about
tokens.
"""

from __future__ import annotations

import json
import traceback
from typing import Any, Callable, Mapping

from .. import exceptions as core
from . import status as http_status


class HttpError(Exception):
    """An error with a status code and a problem+json rendering."""

    status: int = 500
    title: str = "Internal Server Error"
    type_uri: str = "urn:mactf:error:internal"

    def __init__(
        self,
        detail: str = "",
        *,
        extensions: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.extensions: dict[str, Any] = dict(extensions or {})
        self.headers: dict[str, str] = dict(headers or {})


class BadRequest(HttpError):
    status = 400
    title = "Bad Request"
    type_uri = "urn:mactf:error:bad-request"


class ParseError(BadRequest):
    title = "Malformed Request Body"
    type_uri = "urn:mactf:error:parse"


class NotAuthenticated(HttpError):
    status = 401
    title = "Unauthorized"
    type_uri = "urn:mactf:error:unauthorized"


class PermissionDenied(HttpError):
    status = 403
    title = "Forbidden"
    type_uri = "urn:mactf:error:forbidden"


class Http404(HttpError):
    status = 404
    title = "Not Found"
    type_uri = "urn:mactf:error:not-found"


class MethodNotAllowed(HttpError):
    status = 405
    title = "Method Not Allowed"
    type_uri = "urn:mactf:error:method-not-allowed"

    def __init__(self, detail: str = "", *, allowed: tuple[str, ...] = (), **kwargs: Any) -> None:
        super().__init__(detail, **kwargs)
        self.allowed = tuple(allowed)
        if allowed:
            self.headers.setdefault("Allow", ", ".join(allowed))


class Conflict(HttpError):
    status = 409
    title = "Conflict"
    type_uri = "urn:mactf:error:conflict"


class Gone(HttpError):
    status = 410
    title = "Gone"
    type_uri = "urn:mactf:error:gone"


class RequestDataTooBig(HttpError):
    status = 413
    title = "Payload Too Large"
    type_uri = "urn:mactf:error:too-large"


class UnsupportedMediaType(HttpError):
    status = 415
    title = "Unsupported Media Type"
    type_uri = "urn:mactf:error:unsupported-media-type"


class ValidationError(HttpError):
    """Field errors from a serializer: ``{"field": ["message", ...]}``."""

    status = 422
    title = "Validation Error"
    type_uri = "urn:mactf:error:validation"

    def __init__(
        self,
        errors: Mapping[str, Any] | str | None = None,
        *,
        detail: str = "",
        **kwargs: Any,
    ) -> None:
        if isinstance(errors, str):
            errors = {"non_field_errors": [errors]}
        self.errors: dict[str, Any] = dict(errors or {})
        super().__init__(detail or "request validation failed", **kwargs)
        self.extensions.setdefault("errors", self.errors)


class RuleViolation(HttpError):
    status = 422
    title = "RuleViolation"
    type_uri = "urn:mactf:error:rule-violation"


class RateLimited(HttpError):
    status = 429
    title = "Too Many Requests"
    type_uri = "urn:mactf:error:rate-limited"

    def __init__(self, detail: str = "", *, retry_after: float | None = None, **kwargs: Any) -> None:
        super().__init__(detail, **kwargs)
        self.retry_after = retry_after
        if retry_after is not None:
            self.headers.setdefault("Retry-After", str(int(max(1, round(retry_after)))))
            self.extensions.setdefault("retry_after", retry_after)


class ServerError(HttpError):
    status = 500
    title = "Internal Server Error"
    type_uri = "urn:mactf:error:internal"


class Resolver404(Http404):
    """No URL pattern matched."""

    def __init__(self, detail: str = "", *, path: str = "", tried: tuple[str, ...] = ()) -> None:
        super().__init__(detail or f"no route matches {path!r}")
        self.path = path
        self.tried = tuple(tried)


class NoReverseMatch(Exception):
    """``reverse()`` could not build a URL."""


# ---------------------------------------------------------------------------
# The translation table (document 12 section 10)
# ---------------------------------------------------------------------------
#: ``handler(exc) -> (status, type_uri, title, detail, extensions, headers)``
Handler = Callable[[BaseException], "Translation"]
Translation = tuple[int, str, str, str, dict[str, Any], dict[str, str]]

_HANDLERS: dict[type[BaseException], Handler] = {}


def _static(status: int, type_uri: str, title: str | None = None) -> Handler:
    def handler(exc: BaseException) -> Translation:
        return (status, type_uri, title or type(exc).__name__, _detail(exc), {}, {})

    return handler


def _detail(exc: BaseException) -> str:
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc)


def _from_http_error(exc: BaseException) -> Translation:
    assert isinstance(exc, HttpError)
    return (exc.status, exc.type_uri, exc.title, exc.detail, dict(exc.extensions), dict(exc.headers))


def register_handler(exception_type: type[BaseException], handler: Handler) -> None:
    """Map an exception type to a problem. The most specific registered type wins."""
    if not (isinstance(exception_type, type) and issubclass(exception_type, BaseException)):
        raise TypeError("register_handler needs an exception class")
    _HANDLERS[exception_type] = handler


def unregister_handler(exception_type: type[BaseException]) -> None:
    _HANDLERS.pop(exception_type, None)


def _install_defaults() -> None:
    register_handler(HttpError, _from_http_error)
    register_handler(core.RuleViolation, _static(422, "urn:mactf:error:rule-violation"))
    register_handler(core.ComplianceError, _static(422, "urn:mactf:error:compliance"))
    register_handler(core.ImproperlyConfigured, _static(500, "urn:mactf:error:configuration"))
    register_handler(core.ValidationError, _static(422, "urn:mactf:error:validation"))
    register_handler(core.ContractError, _static(422, "urn:mactf:error:contract"))
    register_handler(core.SandboxError, _static(502, "urn:mactf:error:sandbox"))
    register_handler(LookupError, _static(404, "urn:mactf:error:not-found"))
    register_handler(core.CommandError, _static(400, "urn:mactf:error:command"))
    register_handler(core.DataError, _static(422, "urn:mactf:error:data"))
    register_handler(core.MACTFError, _static(500, "urn:mactf:error:internal"))


_install_defaults()


def translate(exc: BaseException) -> Translation:
    """Resolve ``exc`` through the table, walking the MRO so subclasses inherit."""
    for klass in type(exc).__mro__:
        handler = _HANDLERS.get(klass)
        if handler is not None:
            return handler(exc)
    return (500, "urn:mactf:error:internal", "Internal Server Error", "", {}, {})


def problem_body(
    exc: BaseException,
    *,
    instance: str = "",
    trace_id: str = "",
    debug: bool = False,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    """The RFC 9457 body for an exception, plus its status and extra headers.

    A 500 never carries the exception text unless ``debug`` is set: an unexpected
    error's message is an implementation detail, and for a framework error the
    registered handler decides what is safe to say.
    """
    status, type_uri, title, detail, extensions, headers = translate(exc)
    unexpected = status >= 500 and not isinstance(exc, (HttpError, core.MACTFError))
    if unexpected and not debug:
        detail = ""
    body: dict[str, Any] = {
        "type": type_uri,
        "title": title,
        "status": status,
        "detail": detail,
        "instance": instance,
        "trace_id": trace_id,
        "extensions": dict(extensions),
    }
    if debug:
        body["extensions"]["debug"] = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
    if status not in http_status.STATUS_TEXT:
        body["status"] = 500
        status = 500
    return status, body, headers


def render_problem(body: Mapping[str, Any]) -> bytes:
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
