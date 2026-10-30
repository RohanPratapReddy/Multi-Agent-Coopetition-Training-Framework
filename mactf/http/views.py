"""Views: function views and ``View`` subclasses, with method dispatch."""

from __future__ import annotations

import functools
from typing import Any, Callable, Iterable, Mapping

from ..exceptions import ImproperlyConfigured
from .exceptions import MethodNotAllowed
from .request import Request
from .response import JsonResponse, Response

__all__ = [
    "View",
    "csrf_exempt",
    "ensure_response",
    "require_GET",
    "require_POST",
    "require_http_methods",
]


def ensure_response(result: Any, *, view: Any = None) -> Response:
    """Normalise a view's return value.

    ``dict``/``list`` become ``JsonResponse``. ``None`` is an error: a view that
    forgets ``return`` must not look like an empty success.
    """
    if isinstance(result, Response):
        return result
    if isinstance(result, (dict, list)):
        return JsonResponse(result)
    name = getattr(view, "__name__", repr(view))
    if result is None:
        raise ImproperlyConfigured(
            f"view {name} returned None; return a Response (or a dict/list for JSON)"
        )
    raise ImproperlyConfigured(
        f"view {name} returned {type(result).__name__}; return a Response, dict or list"
    )


class View:
    """Class-based view. Define ``get``/``post``/... methods taking ``(request, **kwargs)``."""

    http_method_names: list[str] = ["get", "post", "put", "patch", "delete", "head", "options"]

    #: Documentation attributes read by ``mactf.http.openapi``.
    summary: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def as_view(cls, **initkwargs: Any) -> Callable[..., Response]:
        for key in initkwargs:
            if key in cls.http_method_names:
                raise TypeError(
                    f"The method name {key!r} is not accepted as a keyword argument to as_view()"
                )
            if not hasattr(cls, key):
                raise TypeError(
                    f"{cls.__name__}() received an invalid keyword {key!r}; as_view only accepts "
                    "arguments that are already attributes of the class"
                )

        def view(request: Request, *args: Any, **kwargs: Any) -> Response:
            self = cls(**initkwargs)
            self.request = request  # type: ignore[attr-defined]
            self.args = args  # type: ignore[attr-defined]
            self.kwargs = kwargs  # type: ignore[attr-defined]
            return self.dispatch(request, *args, **kwargs)

        view.view_class = cls  # type: ignore[attr-defined]
        view.view_initkwargs = dict(initkwargs)  # type: ignore[attr-defined]
        functools.update_wrapper(view, cls, updated=())
        view.__dict__.pop("__wrapped__", None)  # the class is not a wrapped callable
        view.__name__ = cls.__name__
        return view

    @property
    def allowed_methods(self) -> str:
        return ", ".join(self._allowed())

    def _allowed(self) -> list[str]:
        methods = [m.upper() for m in self.http_method_names if hasattr(self, m)]
        if "GET" in methods and "HEAD" not in methods:
            methods.append("HEAD")
        if "OPTIONS" not in methods:
            methods.append("OPTIONS")
        return sorted(set(methods))

    def dispatch(self, request: Request, *args: Any, **kwargs: Any) -> Response:
        method = request.method.lower()
        if method in self.http_method_names:
            handler = getattr(self, method, None)
            if handler is None and method == "head":
                handler = getattr(self, "get", None)  # served by GET; handler drops the body
            if handler is None:
                handler = self.http_method_not_allowed
        else:
            handler = self.http_method_not_allowed
        return ensure_response(handler(request, *args, **kwargs), view=type(self))

    def http_method_not_allowed(self, request: Request, *args: Any, **kwargs: Any) -> Response:
        return Response(
            status=405,
            headers={"Allow": self.allowed_methods},
            content_type="text/plain; charset=utf-8",
            content=b"",
        )

    def options(self, request: Request, *args: Any, **kwargs: Any) -> Response:
        return Response(
            b"",
            200,
            headers={"Allow": self.allowed_methods, "Content-Length": "0", "Cache-Control": "no-store"},
            content_type=None,
        )


# ---------------------------------------------------------------------------
# Function-view decorators
# ---------------------------------------------------------------------------
def require_http_methods(methods: Iterable[str]) -> Callable[[Callable[..., Any]], Callable[..., Response]]:
    allowed = tuple(m.upper() for m in methods)

    def decorator(view: Callable[..., Any]) -> Callable[..., Response]:
        @functools.wraps(view)
        def wrapper(request: Request, *args: Any, **kwargs: Any) -> Response:
            if request.method not in allowed and not (
                request.method == "HEAD" and "GET" in allowed
            ):
                raise MethodNotAllowed(
                    f"{request.method} is not allowed here", allowed=allowed
                )
            return ensure_response(view(request, *args, **kwargs), view=view)

        wrapper.allowed_methods = allowed  # type: ignore[attr-defined]
        return wrapper

    return decorator


require_GET = require_http_methods(["GET"])
require_POST = require_http_methods(["POST"])


def csrf_exempt(view: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a function view. There is no CSRF layer here; the marker keeps ports honest."""
    view.csrf_exempt = True  # type: ignore[attr-defined]
    return view
