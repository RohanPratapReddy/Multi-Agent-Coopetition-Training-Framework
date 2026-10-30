"""Dispatch a WSGI application without a socket.

Shared by ``mactf.client.InProcessTransport`` and ``mactf.testing.Client`` so there
is exactly one implementation of "build an environ, call the app, read the answer".
"""

from __future__ import annotations

import contextlib
import io
import json as _json
from typing import Any, Callable, Iterable, Iterator, Mapping
from urllib.parse import quote, unquote, urlencode, urlsplit

from .request import Headers

__all__ = ["ClientResponse", "active_settings", "dispatch_wsgi", "make_environ", "parse_sse"]

_UNSET: Any = object()


def make_environ(
    method: str,
    path: str,
    *,
    query: Mapping[str, Any] | str | None = None,
    headers: Mapping[str, str] | None = None,
    body: bytes | str | None = None,
    json: Any = _UNSET,
    data: Mapping[str, Any] | None = None,
    content_type: str | None = None,
    script_name: str = "",
    chunked: bool = False,
    server: tuple[str, int] = ("testserver", 80),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A PEP 3333 ``environ`` for one request."""
    split = urlsplit(path)
    raw_path = unquote(split.path or "/")
    query_string = split.query
    if query:
        extra_query = query if isinstance(query, str) else urlencode(_flatten(query), doseq=True)
        query_string = f"{query_string}&{extra_query}" if query_string else extra_query
    payload: bytes
    if json is not _UNSET:
        payload = _json.dumps(json, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        content_type = content_type or "application/json"
    elif data is not None:
        payload = urlencode(_flatten(data), doseq=True).encode("utf-8")
        content_type = content_type or "application/x-www-form-urlencoded"
    elif body is not None:
        payload = body.encode("utf-8") if isinstance(body, str) else bytes(body)
    else:
        payload = b""
    environ: dict[str, Any] = {
        "REQUEST_METHOD": method.upper(),
        "SCRIPT_NAME": script_name,
        # PEP 3333: PATH_INFO is the decoded path, carried as latin-1 text.
        "PATH_INFO": raw_path.encode("utf-8").decode("latin-1"),
        "QUERY_STRING": query_string,
        "SERVER_NAME": server[0],
        "SERVER_PORT": str(server[1]),
        "SERVER_PROTOCOL": "HTTP/1.1",
        "SERVER_SOFTWARE": "mactf-test",
        "REMOTE_ADDR": "127.0.0.1",
        "wsgi.version": (1, 0),
        "wsgi.url_scheme": "http",
        "wsgi.input": io.BytesIO(payload),
        "wsgi.errors": io.StringIO(),
        "wsgi.multithread": False,
        "wsgi.multiprocess": False,
        "wsgi.run_once": False,
    }
    if chunked:
        environ["HTTP_TRANSFER_ENCODING"] = "chunked"
    elif payload or method.upper() in ("POST", "PUT", "PATCH"):
        environ["CONTENT_LENGTH"] = str(len(payload))
    if content_type:
        environ["CONTENT_TYPE"] = content_type
    for name, value in (headers or {}).items():
        key = name.upper().replace("-", "_")
        if key in ("CONTENT_TYPE", "CONTENT_LENGTH"):
            environ[key] = str(value)
        else:
            environ["HTTP_" + key] = str(value)
    environ.update(extra or {})
    return environ


def _flatten(values: Mapping[str, Any]) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            out.extend((key, item) for item in value)
        elif isinstance(value, bool):
            out.append((key, "true" if value else "false"))
        else:
            out.append((key, value))
    return out


class ClientResponse:
    """What the application returned. Non-streaming bodies are read eagerly."""

    def __init__(
        self,
        status_line: str,
        headers: list[tuple[str, str]],
        body: Iterable[bytes],
        *,
        request_environ: Mapping[str, Any] | None = None,
    ) -> None:
        code, _, reason = status_line.partition(" ")
        self.status_code = int(code)
        self.reason = reason
        self.headers = Headers(list(headers))
        self.request_environ = request_environ
        self._iterable = body
        self._content: bytes | None = None
        self._consumed = False
        self.streaming = "text/event-stream" in (self.headers.get("Content-Type") or "") or (
            "chunked" in (self.headers.get("Transfer-Encoding") or "").lower()
        )
        if not self.streaming:
            self._content = b"".join(self._iterable)
            self.close()

    # -- bodies -----------------------------------------------------------
    @property
    def content(self) -> bytes:
        if self._content is None:
            if self._consumed:
                raise RuntimeError("the streaming body was already consumed")
            self._content = b"".join(self.iter_content())
        return self._content

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return _json.loads(self.content.decode("utf-8"))

    def iter_content(self) -> Iterator[bytes]:
        if self._content is not None:
            if self._content:
                yield self._content
            return
        if self._consumed:
            raise RuntimeError("the streaming body was already consumed")
        self._consumed = True
        try:
            for chunk in self._iterable:
                yield chunk
        finally:
            self.close()

    def iter_sse(self) -> Iterator[dict[str, Any]]:
        """Parse the stream into ``{"event", "id", "data", "comment"}`` dicts."""
        return parse_sse(self.iter_content())

    # -- header conveniences -----------------------------------------------------
    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def problem(self) -> dict[str, Any] | None:
        if "problem+json" in (self.headers.get("Content-Type") or ""):
            return self.json()
        return None

    def get(self, name: str, default: str | None = None) -> str | None:
        return self.headers.get(name, default)

    def __getitem__(self, name: str) -> str:
        return self.headers[name]

    def close(self) -> None:
        closer = getattr(self._iterable, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> "ClientResponse":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<ClientResponse {self.status_code} {self.headers.get('Content-Type')!r}>"


def parse_sse(chunks: Iterable[bytes]) -> Iterator[dict[str, Any]]:
    """Incrementally frame ``text/event-stream`` bytes into events and comments."""
    buffer = ""
    decoder = __import__("codecs").getincrementaldecoder("utf-8")(errors="replace")
    for chunk in chunks:
        buffer += decoder.decode(chunk)
        buffer = buffer.replace("\r\n", "\n")
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            event = _frame(frame)
            if event is not None:
                yield event
    buffer += decoder.decode(b"", final=True)
    if buffer.strip():
        event = _frame(buffer)
        if event is not None:
            yield event


def _frame(frame: str) -> dict[str, Any] | None:
    event: dict[str, Any] = {"event": "message", "id": None, "data": None, "comment": None}
    data_lines: list[str] = []
    for line in frame.split("\n"):
        if not line:
            continue
        if line.startswith(":"):
            event["comment"] = line[1:].strip()
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event["event"] = value
        elif field == "id":
            event["id"] = value
        elif field == "data":
            data_lines.append(value)
    if data_lines:
        raw = "\n".join(data_lines)
        try:
            event["data"] = _json.loads(raw)
        except ValueError:
            event["data"] = raw
    if event["data"] is None and event["comment"] is None:
        return None
    return event


def dispatch_wsgi(
    app: Callable[..., Iterable[bytes]],
    method: str,
    path: str,
    *,
    response_class: type[ClientResponse] | None = None,
    **kwargs: Any,
) -> ClientResponse:
    """Call ``app`` as a WSGI server would, returning a :class:`ClientResponse`."""
    environ = make_environ(method, path, **kwargs)
    captured: dict[str, Any] = {}

    def start_response(status: str, headers: list[tuple[str, str]], exc_info: Any = None) -> Callable:
        captured["status"] = status
        captured["headers"] = headers
        return lambda data: None

    iterable = app(environ, start_response)
    if "status" not in captured:  # a generator app that starts lazily
        iterator = iter(iterable)
        first = next(iterator, b"")
        iterable = _chain(first, iterator, iterable)
    return (response_class or ClientResponse)(
        captured["status"], captured["headers"], iterable, request_environ=environ
    )


class _chain:
    def __init__(self, first: bytes, rest: Iterator[bytes], original: Any) -> None:
        self._first, self._rest, self._original = first, rest, original
        self._started = False

    def __iter__(self) -> Iterator[bytes]:
        if not self._started:
            self._started = True
            if self._first:
                yield self._first
        yield from self._rest

    def close(self) -> None:
        closer = getattr(self._original, "close", None)
        if callable(closer):
            closer()


@contextlib.contextmanager
def active_settings(settings: Any | None):
    """Make ``settings`` the current settings object for one call, then restore.

    Shared by ``mactf.testing.Client`` and ``mactf.client.InProcessTransport`` so a
    caller can hand over a ``Settings`` without it leaking into the process: the
    module-level object and the resolver/application caches are swapped in and out.
    """
    if settings is None:
        yield
        return
    from .. import conf

    previous = conf._settings
    conf._settings = settings
    conf._clear_http_caches()
    try:
        yield
    finally:
        conf._settings = previous
        conf._clear_http_caches()


# ``quote`` is re-exported for callers building paths by hand.
_ = quote
