"""``Response`` and its subclasses, plus server-sent-event framing."""

from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import json
import os
import pathlib
from email.utils import format_datetime
from typing import Any, Callable, Iterable, Iterator, Mapping

from . import status as status  # re-exported: ``response.status.OK``
from .status import STATUS_TEXT, status_line

__all__ = [
    "FileResponse",
    "JsonResponse",
    "ProblemResponse",
    "Response",
    "STATUS_TEXT",
    "StreamingResponse",
    "dumps",
    "sse_event",
    "sse_heartbeat",
    "status",
]

_FILE_CHUNK = 64 * 1024


def _json_default(value: Any) -> Any:
    """Projection for objects a view may reasonably return."""
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, (pathlib.PurePath,)):
        return str(value)
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return as_dict()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def dumps(data: Any) -> str:
    """The one JSON encoding used on the wire."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=_json_default)


class _ResponseHeaders(dict):
    """A dict that treats header names case-insensitively but keeps first spelling."""

    def __init__(self, initial: Mapping[str, str] | None = None) -> None:
        super().__init__()
        self._folded: dict[str, str] = {}
        for key, value in (initial or {}).items():
            self[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        folded = key.lower()
        existing = self._folded.get(folded)
        if existing is not None and existing != key:
            super().__delitem__(existing)
        self._folded[folded] = key
        super().__setitem__(key, str(value))

    def __getitem__(self, key: str) -> str:
        return super().__getitem__(self._folded.get(key.lower(), key))

    def __delitem__(self, key: str) -> None:
        actual = self._folded.pop(key.lower(), key)
        super().__delitem__(actual)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key.lower() in self._folded

    def get(self, key: str, default: Any = None) -> Any:
        actual = self._folded.get(key.lower())
        return super().__getitem__(actual) if actual is not None else default

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key in self:
            return self[key]
        self[key] = default
        return self[key]

    def pop(self, key: str, *default: Any) -> Any:
        actual = self._folded.pop(key.lower(), None)
        if actual is None:
            if default:
                return default[0]
            raise KeyError(key)
        return super().pop(actual)

    def update(self, *args: Any, **kwargs: Any) -> None:
        for key, value in dict(*args, **kwargs).items():
            self[key] = value


class Response:
    """A fully buffered response. ``content`` is always ``bytes``."""

    streaming = False

    def __init__(
        self,
        content: bytes = b"",
        status: int = 200,
        *,
        headers: Mapping[str, str] | None = None,
        content_type: str | None = "text/plain; charset=utf-8",
    ) -> None:
        if not isinstance(content, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"Response content must be bytes, not {type(content).__name__}; "
                "encode it explicitly (implicit encoding turns a lone surrogate into a 500)"
            )
        status_line(int(status))  # validates: a code the framework never returns is an error
        self.status_code = int(status)
        self.content = bytes(content)
        self.headers = _ResponseHeaders(headers)
        self.cookies: list[str] = []
        self._closed = False
        if content_type is not None and "Content-Type" not in self.headers:
            self.headers["Content-Type"] = content_type

    # -- mapping-ish header access ---------------------------------------
    def __setitem__(self, name: str, value: str) -> None:
        self.headers[name] = value

    def __getitem__(self, name: str) -> str:
        return self.headers[name]

    def __delitem__(self, name: str) -> None:
        del self.headers[name]

    def __contains__(self, name: object) -> bool:
        return name in self.headers

    def get(self, name: str, default: Any = None) -> Any:
        return self.headers.get(name, default)

    @property
    def status_text(self) -> str:
        return STATUS_TEXT[self.status_code]

    @property
    def content_type(self) -> str:
        return self.headers.get("Content-Type", "")

    # -- cookies ------------------------------------------------------------
    def set_cookie(
        self,
        key: str,
        value: str = "",
        *,
        max_age: int | None = None,
        path: str = "/",
        domain: str | None = None,
        secure: bool = False,
        httponly: bool = False,
        samesite: str | None = "Lax",
        expires: _dt.datetime | None = None,
    ) -> None:
        from urllib.parse import quote

        parts = [f"{key}={quote(str(value), safe='')}"]
        if max_age is not None:
            parts.append(f"Max-Age={int(max_age)}")
            if expires is None:
                expires = _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(seconds=int(max_age))
        if expires is not None:
            parts.append(f"Expires={format_datetime(expires.astimezone(_dt.timezone.utc), usegmt=True)}")
        if path:
            parts.append(f"Path={path}")
        if domain:
            parts.append(f"Domain={domain}")
        if secure:
            parts.append("Secure")
        if httponly:
            parts.append("HttpOnly")
        if samesite:
            if samesite.lower() not in ("lax", "strict", "none"):
                raise ValueError("samesite must be 'Lax', 'Strict' or 'None'")
            parts.append(f"SameSite={samesite}")
        self.cookies.append("; ".join(parts))

    def delete_cookie(self, key: str, *, path: str = "/", domain: str | None = None) -> None:
        self.set_cookie(
            key,
            "",
            max_age=0,
            path=path,
            domain=domain,
            expires=_dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc),
        )

    # -- body access ----------------------------------------------------------
    def body_chunks(self) -> Iterable[bytes]:
        return (self.content,) if self.content else ()

    def text(self, encoding: str = "utf-8") -> str:
        return self.content.decode(encoding, errors="replace")

    def json(self) -> Any:
        return json.loads(self.content.decode("utf-8"))

    def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def wsgi_headers(self) -> list[tuple[str, str]]:
        """Native-string headers for ``start_response`` (cookies repeat)."""
        pairs = [(key, _latin1_safe(key, value)) for key, value in self.headers.items()]
        pairs.extend(("Set-Cookie", _latin1_safe("Set-Cookie", cookie)) for cookie in self.cookies)
        return pairs

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.status_code} {self.content_type!r}>"


def _latin1_safe(name: str, value: str) -> str:
    """WSGI headers must be Latin-1; percent-encode the rest instead of dropping it."""
    value = str(value).replace("\r", " ").replace("\n", " ")
    try:
        value.encode("latin-1")
        return value
    except UnicodeEncodeError:
        import logging
        from urllib.parse import quote

        logging.getLogger("mactf.http").warning(
            "header %r carries non-Latin-1 text and was percent-encoded", name
        )
        return quote(value, safe=" !#$%&'()*+,/:;=?@[]~-._")


class JsonResponse(Response):
    """UTF-8 JSON, ``application/json``."""

    def __init__(
        self,
        data: Any,
        status: int = 200,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(
            dumps(data).encode("utf-8"),
            status,
            headers=headers,
            content_type="application/json",
        )
        self.data = data


class ProblemResponse(Response):
    """``application/problem+json``, built from an exception (RFC 9457)."""

    def __init__(
        self,
        error: BaseException,
        *,
        instance: str = "",
        trace_id: str = "",
        debug: bool = False,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        from .exceptions import problem_body, render_problem

        status, body, headers = problem_body(
            error, instance=instance, trace_id=trace_id, debug=debug
        )
        merged = {**headers, **(extra_headers or {})}
        super().__init__(
            render_problem(body),
            status,
            headers=merged,
            content_type="application/problem+json",
        )
        self.problem = body
        self.error = error


class StreamingResponse(Response):
    """A response whose body is an iterator of ``bytes``, forwarded chunk by chunk."""

    streaming = True

    def __init__(
        self,
        iterator: Iterable[bytes],
        status: int = 200,
        *,
        headers: Mapping[str, str] | None = None,
        content_type: str = "text/event-stream",
    ) -> None:
        super().__init__(b"", status, headers=headers, content_type=content_type)
        self._iterator: Iterator[bytes] = iter(iterator)
        self._source = iterator
        self._consumed = False

    def body_chunks(self) -> Iterable[bytes]:
        if self._consumed:
            raise RuntimeError("a streaming response can be iterated only once")
        self._consumed = True
        return self._iter_checked()

    def _iter_checked(self) -> Iterator[bytes]:
        for chunk in self._iterator:
            if not isinstance(chunk, (bytes, bytearray)):
                raise TypeError(
                    f"a streaming response must yield bytes, got {type(chunk).__name__}"
                )
            yield bytes(chunk)

    @property
    def content(self) -> bytes:  # type: ignore[override]
        raise AttributeError("a streaming response has no buffered content; iterate it")

    @content.setter
    def content(self, value: bytes) -> None:
        self._buffered = value

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            for target in (self._iterator, self._source):
                closer = getattr(target, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except Exception:  # noqa: BLE001 - closing must never raise into the server
                        pass
                    break


class FileResponse(Response):
    """A file, via ``wsgi.file_wrapper`` when the server offers it."""

    streaming = True

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        content_type: str | None = None,
        filename: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        import mimetypes

        self.path = pathlib.Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(str(self.path))
        guessed = content_type or mimetypes.guess_type(str(self.path))[0] or "application/octet-stream"
        super().__init__(b"", 200, headers=headers, content_type=guessed)
        self.headers["Content-Length"] = str(self.path.stat().st_size)
        if filename:
            from urllib.parse import quote

            self.headers["Content-Disposition"] = (
                f"attachment; filename*=UTF-8''{quote(filename, safe='')}"
            )
        self._handle: Any = None
        self.file_wrapper: Callable[..., Iterable[bytes]] | None = None

    def body_chunks(self) -> Iterable[bytes]:
        self._handle = open(self.path, "rb")  # noqa: SIM115 - closed in close()
        if self.file_wrapper is not None:
            return self.file_wrapper(self._handle, _FILE_CHUNK)
        return self._read_loop()

    def _read_loop(self) -> Iterator[bytes]:
        assert self._handle is not None
        while True:
            chunk = self._handle.read(_FILE_CHUNK)
            if not chunk:
                return
            yield chunk

    @property
    def content(self) -> bytes:  # type: ignore[override]
        return self.path.read_bytes()

    @content.setter
    def content(self, value: bytes) -> None:
        self._ignored = value

    def close(self) -> None:
        self._closed = True
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None


# ---------------------------------------------------------------------------
# Server-sent events (document 12 section 11)
# ---------------------------------------------------------------------------
def sse_event(event: str, data: Mapping[str, Any] | Any, *, event_id: int | str | None = None) -> bytes:
    """One framed SSE event: ``event:``, optional ``id:``, one ``data:`` line, blank line."""
    if "\n" in event or "\r" in event:
        raise ValueError("an SSE event name cannot contain a newline")
    lines = [f"event: {event}"]
    if event_id is not None:
        lines.append(f"id: {event_id}")
    payload = dumps(data)
    for line in payload.split("\n"):  # compact JSON has none; kept for non-JSON-safe text
        lines.append(f"data: {line}")
    return ("\n".join(lines) + "\n\n").encode("utf-8")


def sse_heartbeat(*, comment: str = "keepalive") -> bytes:
    """A comment frame: ignored by clients, keeps proxies from closing the stream."""
    safe = comment.replace("\r", " ").replace("\n", " ")
    return f": {safe}\n\n".encode("utf-8")


def sse_headers() -> dict[str, str]:
    """The three headers every SSE response carries."""
    return {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
