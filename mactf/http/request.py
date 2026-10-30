"""``Request``: a lazily decoded view of a WSGI ``environ``."""

from __future__ import annotations

import io
import json
import time
from email.message import Message
from types import MappingProxyType
from typing import Any, Iterator, Mapping
from urllib.parse import parse_qs, quote, unquote, urlencode

from .exceptions import ParseError, RequestDataTooBig, UnsupportedMediaType

DEFAULT_MAX_BODY_BYTES = 8 * 1024 * 1024
_READ_CHUNK = 64 * 1024


def max_body_bytes() -> int:
    """``HTTP.MAX_BODY_BYTES``, read through ``get_settings`` and never cached."""
    try:
        from ..conf import get_settings

        block = get_settings().get("HTTP") or {}
        return int(block.get("MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES))
    except Exception:  # noqa: BLE001 - settings unavailable: use the documented default
        return DEFAULT_MAX_BODY_BYTES


class Headers:
    """Case-insensitive, multi-value, read-only header mapping.

    Backed by ``email.message.Message``, which already implements RFC 5322 header
    semantics, so ``headers["Content-Type"]`` and ``headers.get("content-type")``
    are the same lookup and repeated headers are preserved for ``getlist``.
    """

    def __init__(self, pairs: list[tuple[str, str]] | None = None) -> None:
        self._message = Message()
        for name, value in pairs or ():
            # ``Message.__setitem__`` would apply policy folding/validation; keep
            # the wire values verbatim so ``getlist`` reproduces them exactly.
            self._message._headers.append((name, value))  # noqa: SLF001

    def __getitem__(self, name: str) -> str:
        value = self.get(name)
        if value is None:
            raise KeyError(name)
        return value

    def get(self, name: str, default: str | None = None) -> str | None:
        lowered = name.lower()
        for key, value in self._message._headers:  # noqa: SLF001
            if key.lower() == lowered:
                return value
        return default

    def getlist(self, name: str) -> list[str]:
        lowered = name.lower()
        return [value for key, value in self._message._headers if key.lower() == lowered]  # noqa: SLF001

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.get(name) is not None

    def __iter__(self) -> Iterator[str]:
        seen: set[str] = set()
        for key, _ in self._message._headers:  # noqa: SLF001
            lowered = key.lower()
            if lowered not in seen:
                seen.add(lowered)
                yield key

    def __len__(self) -> int:
        return len(set(key.lower() for key, _ in self._message._headers))  # noqa: SLF001

    def keys(self) -> list[str]:
        return list(iter(self))

    def items(self) -> list[tuple[str, str]]:
        return [(key, self[key]) for key in self]

    def to_dict(self) -> dict[str, str]:
        return {key: self[key] for key in self}

    def __repr__(self) -> str:
        return f"Headers({self.to_dict()!r})"


class QueryDict(Mapping[str, str]):
    """Query/form parameters. Immutable unless built with ``mutable=True``.

    ``__getitem__`` returns the **last** value of a repeated key (Django's rule), so
    ``?limit=10&limit=20`` is deterministic; ``getlist`` returns all of them.
    """

    def __init__(
        self,
        source: str | Mapping[str, Any] | None = None,
        *,
        mutable: bool = False,
        encoding: str = "utf-8",
    ) -> None:
        self._mutable = bool(mutable)
        self.encoding = encoding
        self._data: dict[str, list[str]] = {}
        if isinstance(source, str):
            for key, values in parse_qs(source, keep_blank_values=True, encoding=encoding).items():
                self._data[key] = list(values)
        elif source is not None:
            for key, value in source.items():
                if isinstance(value, (list, tuple)):
                    self._data[str(key)] = [str(item) for item in value]
                else:
                    self._data[str(key)] = [str(value)]

    # -- Mapping ----------------------------------------------------------
    def __getitem__(self, key: str) -> str:
        values = self._data[key]
        return values[-1] if values else ""

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def getlist(self, key: str, default: list[str] | None = None) -> list[str]:
        return list(self._data.get(key, default if default is not None else []))

    def lists(self) -> list[tuple[str, list[str]]]:
        return [(key, list(values)) for key, values in self._data.items()]

    # -- mutation (only on a copy) ---------------------------------------
    @property
    def mutable(self) -> bool:
        return self._mutable

    def _check_mutable(self) -> None:
        if not self._mutable:
            raise AttributeError("this QueryDict is immutable; use .copy() for a mutable one")

    def __setitem__(self, key: str, value: Any) -> None:
        self._check_mutable()
        self._data[str(key)] = [str(value)]

    def __delitem__(self, key: str) -> None:
        self._check_mutable()
        del self._data[key]

    def setlist(self, key: str, values: list[Any]) -> None:
        self._check_mutable()
        self._data[str(key)] = [str(item) for item in values]

    def appendlist(self, key: str, value: Any) -> None:
        self._check_mutable()
        self._data.setdefault(str(key), []).append(str(value))

    def copy(self) -> "QueryDict":
        clone = QueryDict(mutable=True, encoding=self.encoding)
        clone._data = {key: list(values) for key, values in self._data.items()}
        return clone

    def urlencode(self) -> str:
        return urlencode([(key, value) for key, values in self._data.items() for value in values])

    def to_dict(self) -> dict[str, str]:
        return {key: self[key] for key in self._data}

    def __repr__(self) -> str:
        return f"QueryDict({self._data!r})"


class Request:
    """A WSGI request. Nothing is parsed until a view asks for it."""

    def __init__(self, environ: Mapping[str, Any]) -> None:
        method = environ.get("REQUEST_METHOD")
        if not method:
            raise ParseError("request has no REQUEST_METHOD")
        raw_path = environ.get("PATH_INFO")
        if raw_path is None:
            raise ParseError("request has no PATH_INFO")
        self.method: str = str(method).upper()
        self.path_info: str = str(raw_path)
        self.path: str = _decode_path(self.path_info)
        self.script_name: str = str(environ.get("SCRIPT_NAME", "") or "")
        self.META: Mapping[str, Any] = MappingProxyType(dict(environ))
        self._environ = environ
        query_string = str(environ.get("QUERY_STRING", "") or "")
        self.query_string = query_string
        self.GET: QueryDict = QueryDict(query_string)
        self.headers: Headers = Headers(_headers_from_environ(environ))
        self.content_type: str = str(environ.get("CONTENT_TYPE", "") or "")
        self._body: bytes | None = None
        self._json: Any = _UNSET
        self._post: QueryDict | None = None
        self.resolver_match: Any = None
        self.principal: Any = None
        self.request_id: str = ""
        self.started_at: float = time.monotonic()
        #: Free-form per-request state for middleware.
        self.state: dict[str, Any] = {}

    # -- body -------------------------------------------------------------
    @property
    def body(self) -> bytes:
        """The body, read once and cached, never larger than ``HTTP.MAX_BODY_BYTES``."""
        if self._body is None:
            self._body = self._read_body()
        return self._body

    def _read_body(self) -> bytes:
        limit = max_body_bytes()
        stream = self._environ.get("wsgi.input")
        if stream is None:
            return b""
        length_text = self._environ.get("CONTENT_LENGTH")
        chunked = "chunked" in str(self._environ.get("HTTP_TRANSFER_ENCODING", "")).lower()
        if length_text not in (None, ""):
            try:
                length = int(length_text)
            except ValueError:
                raise ParseError(f"invalid Content-Length {length_text!r}") from None
            if length < 0:
                raise ParseError("negative Content-Length")
            if length > limit:  # refused before a single byte is read
                raise RequestDataTooBig(f"body of {length} bytes exceeds the {limit} byte limit")
            return stream.read(length) if length else b""
        if not chunked:
            return b""
        buffer = io.BytesIO()
        while True:
            chunk = stream.read(_READ_CHUNK)
            if not chunk:
                break
            buffer.write(chunk)
            if buffer.tell() > limit:
                raise RequestDataTooBig(f"streamed body exceeds the {limit} byte limit")
        return buffer.getvalue()

    # -- decoded bodies ---------------------------------------------------
    @property
    def media_type(self) -> str:
        return self.content_type.split(";", 1)[0].strip().lower()

    def is_json(self) -> bool:
        media = self.media_type
        return media == "application/json" or (
            media.startswith("application/") and media.endswith("+json")
        )

    def json(self) -> Any:
        """Parse the body as JSON, at most once."""
        if self._json is not _UNSET:
            return self._json
        if not self.is_json():
            raise UnsupportedMediaType(
                f"expected application/json, got {self.content_type or 'no Content-Type'!r}"
            )
        raw = self.body
        if not raw.strip():
            raise ParseError("empty request body; expected a JSON document")
        try:
            self._json = json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ParseError(f"request body is not valid UTF-8: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ParseError(f"malformed JSON: {exc.msg} at line {exc.lineno} column {exc.colno}") from exc
        return self._json

    @property
    def POST(self) -> QueryDict:  # noqa: N802 - Django-compatible name
        if self._post is None:
            if self.media_type == "application/x-www-form-urlencoded" and self.method in (
                "POST",
                "PUT",
                "PATCH",
            ):
                self._post = QueryDict(self.body.decode("utf-8", errors="replace"))
            else:
                self._post = QueryDict()
        return self._post

    # -- helpers ----------------------------------------------------------
    @property
    def full_path(self) -> str:
        quoted = quote(self.script_name + self.path)
        return f"{quoted}?{self.query_string}" if self.query_string else quoted

    def accepts(self, media_type: str) -> bool:
        """True when ``Accept`` is absent or lists ``media_type`` or a wildcard for it."""
        values = self.headers.getlist("Accept")
        if not values:
            return True
        offered = {
            part.split(";", 1)[0].strip().lower()
            for value in values
            for part in value.split(",")
            if part.strip()
        }
        major = media_type.split("/", 1)[0]
        return bool({media_type.lower(), "*/*", f"{major}/*"} & offered)

    def get_bearer_token(self) -> str | None:
        header = self.headers.get("Authorization")
        if not header:
            return None
        scheme, _, token = header.partition(" ")
        return token.strip() or None if scheme.lower() == "bearer" else None

    def __repr__(self) -> str:
        return f"<Request {self.method} {self.path!r}>"


_UNSET: Any = object()


def _decode_path(path_info: str) -> str:
    """PEP 3333 hands PATH_INFO as latin-1-decoded bytes; recover the UTF-8 text."""
    try:
        return path_info.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return path_info


def _headers_from_environ(environ: Mapping[str, Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for key, value in environ.items():
        if key.startswith("HTTP_"):
            name = "-".join(part.capitalize() for part in key[5:].split("_"))
            pairs.append((name, str(value)))
        elif key in ("CONTENT_TYPE", "CONTENT_LENGTH") and value not in (None, ""):
            name = "-".join(part.capitalize() for part in key.split("_"))
            pairs.append((name, str(value)))
    return pairs


__all__ = ["Headers", "QueryDict", "Request", "max_body_bytes", "unquote"]
