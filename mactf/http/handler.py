"""The WSGI handler: ``environ`` in, ``(status, headers, body)`` out.

Nothing here is asynchronous. A streaming response is a generator the server
iterates; the run loop is a thread, so an event-loop request path would have to
bridge back to threads for every write.
"""

from __future__ import annotations

import logging
import time
import uuid
from types import ModuleType
from typing import Any, Callable, Iterable, Iterator
from urllib.parse import quote
from wsgiref.util import is_hop_by_hop

from ..exceptions import ImproperlyConfigured
from .exceptions import Http404, HttpError, MethodNotAllowed, Resolver404
from .middleware import convert_exception, import_string
from .request import Request
from .response import FileResponse, ProblemResponse, Response, StreamingResponse
from .status import status_line
from .urls import get_resolver, reset_script_prefix, set_script_prefix
from .views import ensure_response

__all__ = ["WSGIHandler", "clear_application_cache", "get_wsgi_application"]

logger = logging.getLogger("mactf.http")


class WSGIHandler:
    """The WSGI callable for one settings object and URL conf."""

    def __init__(self, *, urlconf: str | ModuleType | None = None) -> None:
        from ..conf import get_settings

        self.settings = get_settings()
        self.urlconf = urlconf
        #: Per-application state. The core fills only ``settings`` and
        #: ``started_at``; an app adds what it owns lazily (``mactf.contrib.api.state``).
        self.state: dict[str, Any] = {"settings": self.settings, "started_at": time.time()}
        self._chain: Callable[[Request], Response] = self._build_chain()

    # -- construction -----------------------------------------------------
    def _build_chain(self) -> Callable[[Request], Response]:
        handler: Callable[[Request], Response] = self._get_response
        for dotted in reversed(list(self.settings.get("MIDDLEWARE") or [])):
            try:
                factory = import_string(dotted) if isinstance(dotted, str) else dotted
                handler = factory(handler)
            except Exception as exc:  # noqa: BLE001
                raise ImproperlyConfigured(f"could not load middleware {dotted!r}: {exc}") from exc
        return handler

    @property
    def middleware_chain(self) -> Callable[[Request], Response]:
        return self._chain

    # -- the innermost step: resolve and call the view ----------------------
    def _get_response(self, request: Request) -> Response:
        resolver = get_resolver(self.urlconf)
        try:
            match = resolver.resolve(request.path)
        except Resolver404:
            redirect = self._append_slash_redirect(request, resolver)
            if redirect is not None:
                return redirect
            raise
        request.resolver_match = match
        return ensure_response(match.func(request, *match.args, **match.kwargs), view=match.func)

    def _append_slash_redirect(self, request: Request, resolver: Any) -> Response | None:
        if not self.settings.get("APPEND_SLASH", True) or request.path.endswith("/"):
            return None
        try:
            resolver.resolve(request.path + "/")
        except Resolver404:
            return None
        location = quote(request.script_name + request.path + "/")
        if request.query_string:
            location += "?" + request.query_string
        return Response(b"", 301, headers={"Location": location}, content_type=None)

    # -- WSGI ---------------------------------------------------------------
    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> Iterable[bytes]:
        started = time.monotonic()
        try:
            request = Request(environ)
        except HttpError as exc:
            # A malformed environ never reaches middleware.
            response: Response = ProblemResponse(
                exc, instance=str(environ.get("PATH_INFO", "") or ""), trace_id=uuid.uuid4().hex
            )
            return self._finish(environ, start_response, None, response, started)

        request.app_state = self.state
        token = set_script_prefix(request.script_name)
        try:
            try:
                response = self._chain(request)
                if not isinstance(response, Response):
                    response = ensure_response(response, view=self._chain)
            except Exception as exc:  # noqa: BLE001 - nothing may escape a WSGI callable
                if not request.request_id:
                    request.request_id = uuid.uuid4().hex
                # An exception that is not a framework error becomes a bare 500
                # problem: ``problem_body`` withholds its text unless DEBUG is set.
                response = convert_exception(request, exc)
                response["X-Trace-Id"] = request.request_id
                response["X-Request-Id"] = request.request_id
            response = self._check_allow(request, response)
        finally:
            reset_script_prefix(token)
        return self._finish(environ, start_response, request, response, started)

    @staticmethod
    def _check_allow(request: Request, response: Response) -> Response:
        """A ``405`` without ``Allow`` is unrecoverable for a client; replace it."""
        if response.status_code == 405 and "Allow" not in response.headers:
            return ProblemResponse(
                MethodNotAllowed(f"{request.method} is not allowed on {request.path}"),
                instance=request.path,
                trace_id=request.request_id,
                extra_headers={"Allow": "OPTIONS"},
            )
        return response

    def _finish(
        self,
        environ: dict[str, Any],
        start_response: Callable[..., Any],
        request: Request | None,
        response: Response,
        started: float,
    ) -> Iterable[bytes]:
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        head = method == "HEAD"
        if isinstance(response, FileResponse):
            response.file_wrapper = environ.get("wsgi.file_wrapper")
        if not response.streaming and "Content-Length" not in response.headers:
            response.headers["Content-Length"] = str(len(response.content))
        if response.streaming and "Content-Length" not in response.headers:
            if "wsgi.file_wrapper" not in environ and not head:
                response.headers.setdefault("Transfer-Encoding", "chunked")
        headers = response.wsgi_headers()
        if str(environ.get("SERVER_SOFTWARE", "")).startswith("WSGIServer"):
            # wsgiref asserts against hop-by-hop headers and frames the body itself.
            headers = [(k, v) for k, v in headers if not is_hop_by_hop(k)]
        start_response(status_line(response.status_code), headers)
        label = f"{method} {environ.get('PATH_INFO', '')}"
        rid = request.request_id if request is not None else ""

        def done(note: str = "") -> None:
            response.close()
            logger.info(
                "%s -> %s in %.1fms [request %s]%s",
                label,
                response.status_code,
                (time.monotonic() - started) * 1000.0,
                rid,
                note,
            )

        if head:
            done()
            return [b""]
        if response.streaming:
            return _StreamIterable(response, done)
        body = response.content
        done()
        return [body] if body else []


class _StreamIterable:
    """Forwards a streaming response's chunks, then closes it exactly once."""

    def __init__(self, response: Response, done: Callable[..., None]) -> None:
        self._response = response
        self._done = done
        self._iterator: Iterator[bytes] | None = None
        self._finished = False

    def __iter__(self) -> "_StreamIterable":
        return self

    def __next__(self) -> bytes:
        if self._iterator is None:
            self._iterator = iter(self._response.body_chunks())
        try:
            return next(self._iterator)
        except StopIteration:
            self._finish()
            raise
        except (BrokenPipeError, ConnectionResetError):
            # Disconnect surfaces as a write error from the server on the next
            # chunk; the subscription held by the generator must be released.
            self._finish(" (client disconnected)")
            raise StopIteration from None
        except Exception:  # noqa: BLE001
            # The status line is already sent, so it cannot be rewritten: log, close.
            logger.exception("streaming response failed after its headers were sent")
            self._finish(" (stream error)")
            raise StopIteration from None

    def _finish(self, note: str = "") -> None:
        if not self._finished:
            self._finished = True
            self._done(note)

    def close(self) -> None:
        """Called by the server when the connection ends, early or not."""
        self._finish(" (closed early)" if self._iterator is not None else "")


# ---------------------------------------------------------------------------
# Application cache, per settings object identity
# ---------------------------------------------------------------------------
_APPLICATIONS: dict[tuple[int, str | None], tuple[Any, WSGIHandler]] = {}


def get_wsgi_application(*, urlconf: str | ModuleType | None = None) -> WSGIHandler:
    """The WSGI callable for the current settings, built once per settings object."""
    from ..conf import get_settings

    settings = get_settings()
    name = urlconf if isinstance(urlconf, (str, type(None))) else urlconf.__name__
    key = (id(settings), name)
    cached = _APPLICATIONS.get(key)
    if cached is not None and cached[0] is settings:
        return cached[1]
    application = WSGIHandler(urlconf=urlconf)
    _APPLICATIONS[key] = (settings, application)
    return application


def clear_application_cache() -> None:
    _APPLICATIONS.clear()


# Http404 is re-exported for projects that raise it from views.
_ = Http404
