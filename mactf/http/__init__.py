"""MACTF's HTTP transport layer (layer L3): stdlib only.

``Request`` and ``Response`` over WSGI, URL resolution through ``ROOT_URLCONF``,
views, middleware, serializers, problem+json errors, SSE framing and an OpenAPI
generator. It knows nothing about peers, scorecards or runs.

``mactf.http.asgi`` is deliberately not imported here: an optional adapter must not
become a mandatory dependency of the package.
"""

from __future__ import annotations

from . import auth, converters, exceptions, handler, middleware, openapi, request, response
from . import serializers, status, testing, urls, views
from .auth import ANONYMOUS, Denial, Principal
from .exceptions import (
    BadRequest,
    Conflict,
    Gone,
    Http404,
    HttpError,
    MethodNotAllowed,
    NoReverseMatch,
    NotAuthenticated,
    ParseError,
    PermissionDenied,
    RateLimited,
    RequestDataTooBig,
    Resolver404,
    RuleViolation,
    ServerError,
    UnsupportedMediaType,
    ValidationError,
    register_handler,
)
from .handler import WSGIHandler, clear_application_cache, get_wsgi_application
from .middleware import (
    AuthMiddleware,
    CorsMiddleware,
    ExceptionMiddleware,
    RequestIdMiddleware,
    import_string,
)
from .request import Headers, QueryDict, Request
from .response import (
    FileResponse,
    JsonResponse,
    ProblemResponse,
    Response,
    StreamingResponse,
    sse_event,
    sse_heartbeat,
)
from .serializers import (
    BoolField,
    CharField,
    ChoiceField,
    DictField,
    Field,
    FloatField,
    IntField,
    ListField,
    ModelSerializer,
    NestedField,
    RawField,
    Serializer,
)
from .urls import (
    ResolverMatch,
    URLPattern,
    URLResolver,
    clear_url_caches,
    get_resolver,
    include,
    path,
    re_path,
    resolve,
    reverse,
)
from .views import View, csrf_exempt, require_GET, require_http_methods, require_POST


def clear_caches() -> None:
    """Drop the resolver and application caches (``conf.reset()`` calls this)."""
    clear_url_caches()
    clear_application_cache()


__all__ = [
    "ANONYMOUS",
    "AuthMiddleware",
    "BadRequest",
    "BoolField",
    "CharField",
    "ChoiceField",
    "Conflict",
    "CorsMiddleware",
    "Denial",
    "DictField",
    "ExceptionMiddleware",
    "Field",
    "FileResponse",
    "FloatField",
    "Gone",
    "Headers",
    "Http404",
    "HttpError",
    "IntField",
    "JsonResponse",
    "ListField",
    "MethodNotAllowed",
    "ModelSerializer",
    "NestedField",
    "NoReverseMatch",
    "NotAuthenticated",
    "ParseError",
    "PermissionDenied",
    "Principal",
    "ProblemResponse",
    "QueryDict",
    "RateLimited",
    "RawField",
    "Request",
    "RequestDataTooBig",
    "RequestIdMiddleware",
    "Resolver404",
    "ResolverMatch",
    "Response",
    "RuleViolation",
    "Serializer",
    "ServerError",
    "StreamingResponse",
    "URLPattern",
    "URLResolver",
    "UnsupportedMediaType",
    "ValidationError",
    "View",
    "WSGIHandler",
    "clear_caches",
    "clear_url_caches",
    "csrf_exempt",
    "get_resolver",
    "get_wsgi_application",
    "import_string",
    "include",
    "path",
    "re_path",
    "register_handler",
    "require_GET",
    "require_POST",
    "require_http_methods",
    "resolve",
    "reverse",
    "sse_event",
    "sse_heartbeat",
]
