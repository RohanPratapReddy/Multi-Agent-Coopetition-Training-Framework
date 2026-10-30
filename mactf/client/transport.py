"""The two transports (document 03 section 4).

``InProcessTransport`` calls the WSGI application directly; ``HTTPTransport`` speaks the
same protocol to a running ``mactf serve`` over ``urllib``. Both hand back a
:class:`mactf.http.Response` holding the exact bytes the application produced, so the
parsing, the error mapping and the ``from_dict`` calls above them are shared.

Standard library plus ``mactf.http`` and ``mactf.wire``; no torch, no engine package.
"""

from __future__ import annotations

import http.client
import json as _json
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from ..http.response import Response
from ..http.status import STATUS_TEXT
from ..http.testing import active_settings, dispatch_wsgi, parse_sse
from .base import Event
from .errors import MACTFClientError, TransportError, error_from_problem, parse_retry_after

__all__ = ["HTTPTransport", "InProcessTransport", "encode_params", "response_from"]

_RETRY_STATUSES = frozenset({429, 502, 503, 504})
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "DELETE", "PUT"})
_MAX_DELAY = 30.0


def _client_version() -> str:
    from .. import __version__

    return __version__


def encode_params(params: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """Query pairs: ``None`` is dropped, booleans are ``true``/``false``, sequences repeat the key."""
    pairs: list[tuple[str, str]] = []
    for key, value in (params or {}).items():
        if value is None:
            continue
        values = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
        for item in values:
            pairs.append((str(key), ("true" if item else "false") if isinstance(item, bool) else str(item)))
    return pairs


def _query(params: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The params as the in-process client wants them: ``None`` dropped, lists kept as lists."""
    cleaned = {k: v for k, v in (params or {}).items() if v is not None}
    return cleaned or None


def response_from(status: int, headers: Mapping[str, str] | list[tuple[str, str]], body: bytes) -> Response:
    """A :class:`mactf.http.Response` from raw wire parts, whatever the status code."""
    items = headers.items() if isinstance(headers, Mapping) else headers
    content_type: str | None = None
    rest: dict[str, str] = {}
    for key, value in items:
        if key.lower() == "content-type":
            content_type = value
        else:
            rest[key] = value
    if status not in STATUS_TEXT:  # a proxy's 504 or 418: keep the class of the answer
        status = 500 if status >= 500 else 400
    return Response(body, status, headers=rest, content_type=content_type)


def _default_headers(extra: Mapping[str, str] | None) -> dict[str, str]:
    headers = {"Accept": "application/json", "X-MACTF-Client": _client_version()}
    headers.update(extra or {})
    return headers


class InProcessTransport:
    """No socket, the whole path: resolver, middleware, auth, view and serializer.

    ``application`` defaults to ``get_wsgi_application()``, resolved on the first request
    for the *current* settings. ``settings`` is made current only for the duration of a
    call, so the client never mutates global state. ``token`` is sent as a bearer token;
    with none, ``API.ALLOW_ANONYMOUS`` decides, exactly as over a socket.
    """

    def __init__(
        self,
        application: Callable[..., Any] | None = None,
        *,
        settings: Any | None = None,
        token: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._application = application
        self.settings = settings
        self.token = token
        self.headers = dict(headers or {})
        self._lock = threading.Lock()

    def _app(self) -> Callable[..., Any]:
        if self._application is not None:
            return self._application
        from ..http.handler import get_wsgi_application

        return get_wsgi_application()

    def _headers(self, extra: Mapping[str, str] | None) -> dict[str, str]:
        merged = _default_headers(self.headers)
        if self.token:
            merged["Authorization"] = f"Bearer {self.token}"
        merged.update(extra or {})
        return merged

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        kwargs: dict[str, Any] = {"headers": self._headers(headers), "query": _query(params)}
        if json is not None:
            kwargs["json"] = json
        with active_settings(self.settings):
            reply = dispatch_wsgi(self._app(), method.upper(), path, **kwargs)
            body = reply.content
            return response_from(reply.status_code, list(reply.headers.items()), body)

    def stream(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Iterator[Event]:
        """Events as the application yields them; a generator response is consumed chunk by chunk."""
        merged = self._headers(headers)
        merged["Accept"] = "text/event-stream"
        with active_settings(self.settings):
            reply = dispatch_wsgi(
                self._app(), method.upper(), path, headers=merged, query=_query(params)
            )
            try:
                if reply.status_code >= 400:
                    raise _error_from_body(reply.status_code, list(reply.headers.items()), reply.content)
                if not reply.streaming:
                    raise TransportError(
                        f"{path} did not answer with text/event-stream", status=reply.status_code
                    )
                for frame in parse_sse(reply.iter_content()):
                    event = Event.from_frame(frame)
                    _check_lost(event)
                    yield event
                    if event.event == "terminal":
                        return
            finally:
                reply.close()


def _check_lost(event: Event) -> None:
    if event.event == "stream_lost":
        raise TransportError(
            "the event stream lost events the server no longer holds",
            extensions={"oldest": event.data.get("oldest"), **dict(event.data)},
        )


def _error_from_body(status: int, headers: Any, body: bytes) -> MACTFClientError:
    problem: Mapping[str, Any] | None = None
    try:
        decoded = _json.loads(body.decode("utf-8")) if body else None
        problem = decoded if isinstance(decoded, Mapping) else None
    except ValueError:
        problem = None
    lookup = dict(headers.items() if hasattr(headers, "items") else headers)
    return error_from_problem(status, problem, lookup, fallback=body[:200].decode("utf-8", "replace"))


class HTTPTransport:
    """``urllib`` against a running server, with retries, idempotency and SSE reconnect.

    * ``429/502/503/504`` and connection errors are retried ``max_retries`` times with
      ``backoff * 2**attempt`` and full jitter (capped at 30 s), honouring ``Retry-After``.
    * Only requests that are safe to replay are retried: GET/HEAD/OPTIONS/DELETE/PUT, and a
      POST that carries an ``Idempotency-Key``. An ``/evaluations`` POST is never retried.
    * ``timeout`` is per request, not per retry sequence.
    * The server prefix is probed once at construction (``/api/v1`` then bare); pass
      ``probe=False`` to skip it and use ``prefix`` as given.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        token: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff: float = 0.5,
        headers: Mapping[str, str] | None = None,
        probe: bool = True,
        prefix: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.backoff = float(backoff)
        self.headers = dict(headers or {})
        self._sleep = sleep
        self._local_base: str | None = None
        self.prefix = prefix if prefix is not None else "/api/v1"
        if probe and prefix is None:
            self.prefix = self._probe_prefix()

    # -- prefix ---------------------------------------------------------------
    def local_base(self) -> str:
        """The mount point the *local* URL conf uses (``/api/v1`` by default)."""
        if self._local_base is None:
            try:
                from ..http.urls import reverse

                health = reverse("api:health")
                self._local_base = health[: -len("/health")] if health.endswith("/health") else "/api/v1"
            except Exception:  # noqa: BLE001 - no api app locally: fall back to the documented default
                self._local_base = "/api/v1"
        return self._local_base

    def _probe_prefix(self) -> str:
        candidates: list[str] = []
        for candidate in (self.local_base(), "/api/v1", ""):
            if candidate not in candidates:
                candidates.append(candidate)
        tried: list[str] = []
        for candidate in candidates:
            url = f"{self.base_url}{candidate}/health"
            tried.append(url)
            try:
                status, _, _ = self._once("GET", url, None, _default_headers(self.headers))
            except TransportError:
                continue
            if status == 200:
                return candidate
        raise TransportError(f"no MACTF API answered at any of: {', '.join(tried)}")

    def _url(self, path: str, params: Mapping[str, Any] | None) -> str:
        local = self.local_base()
        if local != self.prefix and path.startswith(local + "/"):
            path = self.prefix + path[len(local):]
        url = self.base_url + path
        pairs = encode_params(params)
        if pairs:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(pairs)
        return url

    # -- single exchange --------------------------------------------------------
    def _headers(self, extra: Mapping[str, str] | None) -> dict[str, str]:
        merged = _default_headers(self.headers)
        if self.token:
            merged["Authorization"] = f"Bearer {self.token}"
        merged.update(extra or {})
        return merged

    def _once(
        self, method: str, url: str, body: bytes | None, headers: Mapping[str, str]
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                return reply.status, list(reply.headers.items()), reply.read()
        except urllib.error.HTTPError as error:
            try:
                payload = error.read()
            finally:
                error.close()
            return error.code, list(error.headers.items()) if error.headers else [], payload
        except (urllib.error.URLError, OSError, http.client.HTTPException, TimeoutError) as error:
            raise TransportError(f"{method} {url} failed: {error}") from error

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(_MAX_DELAY, retry_after)
        return min(_MAX_DELAY, random.uniform(0.0, self.backoff * (2 ** attempt)))

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        method = method.upper()
        merged = self._headers(headers)
        body: bytes | None = None
        if json is not None:
            body = _json.dumps(json, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            merged.setdefault("Content-Type", "application/json")
        replayable = method in _SAFE_METHODS or any(k.lower() == "idempotency-key" for k in merged)
        url = self._url(path, params)
        attempt = 0
        while True:
            try:
                status, reply_headers, payload = self._once(method, url, body, merged)
            except TransportError:
                if not replayable or attempt >= self.max_retries:
                    raise
                self._sleep(self._delay(attempt, None))
                attempt += 1
                continue
            if status in _RETRY_STATUSES and replayable and attempt < self.max_retries:
                retry_after = parse_retry_after(dict((k.lower(), v) for k, v in reply_headers).get("retry-after"))
                self._sleep(self._delay(attempt, retry_after))
                attempt += 1
                continue
            return response_from(status, reply_headers, payload)

    # -- streaming --------------------------------------------------------------
    def stream(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Iterator[Event]:
        """Events as they arrive, reconnecting with ``Last-Event-ID`` after a dropped connection."""
        method = method.upper()
        url = self._url(path, params)
        base_headers = self._headers(headers)
        base_headers["Accept"] = "text/event-stream"
        last_id: int | None = None
        for key, value in list(base_headers.items()):
            if key.lower() == "last-event-id":
                try:
                    last_id = int(value)
                except ValueError:
                    last_id = None
        failures = 0
        while True:
            attempt_headers = dict(base_headers)
            if last_id is not None:
                attempt_headers["Last-Event-ID"] = str(last_id)
            request = urllib.request.Request(url, headers=attempt_headers, method=method)
            try:
                reply = urllib.request.urlopen(request, timeout=self.timeout)
            except urllib.error.HTTPError as error:
                try:
                    payload = error.read()
                finally:
                    error.close()
                raise _error_from_body(error.code, error.headers or {}, payload) from None
            except (urllib.error.URLError, OSError, http.client.HTTPException, TimeoutError) as error:
                failures += 1
                if failures > self.max_retries:
                    raise TransportError(f"stream {url} could not be opened: {error}") from error
                self._sleep(self._delay(failures - 1, None))
                continue
            try:
                content_type = reply.headers.get("Content-Type", "")
                if "text/event-stream" not in content_type:
                    raise TransportError(
                        f"{url} answered {content_type or 'without a content type'}, not text/event-stream",
                        status=reply.status,
                    )
                lines = iter(reply.readline, b"")
                for frame in parse_sse(lines):
                    event = Event.from_frame(frame)
                    _check_lost(event)
                    if event.id is not None:
                        if last_id is not None and event.id <= last_id:
                            continue  # a replayed frame we already delivered
                        last_id = event.id
                        failures = 0  # only real progress earns a fresh retry budget
                    yield event
                    if event.event == "terminal":
                        return
                dropped: BaseException | None = None
            except (OSError, http.client.HTTPException, TimeoutError) as error:
                dropped = error
            finally:
                reply.close()
            failures += 1
            if failures > self.max_retries:
                raise TransportError(f"stream {url} was lost after {last_id}") from dropped
            self._sleep(self._delay(failures - 1, None))
