"""Client exceptions (document 03 section 5).

Both transports receive the same ``application/problem+json`` body, so there is one
mapping from a problem to an exception class and the client never imports
``mactf.exceptions``: a ``RuleViolation`` raised in a view is a 422 by the time it
reaches here.
"""

from __future__ import annotations

from collections.abc import Mapping
from email.utils import parsedate_to_datetime
from typing import Any


class MACTFClientError(Exception):
    """Base class. ``status``, ``type_uri``, ``trace_id`` and ``extensions`` come from the problem body."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        type_uri: str | None = None,
        trace_id: str | None = None,
        extensions: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.type_uri = type_uri
        self.trace_id = trace_id
        self.extensions: Mapping[str, Any] = dict(extensions or {})

    def __str__(self) -> str:
        parts = [self.message]
        if self.status is not None:
            parts.append(f"[{self.status}]")
        if self.trace_id:
            parts.append(f"(trace {self.trace_id})")
        return " ".join(parts)


class ValidationError(MACTFClientError):
    """422, and the 413 / 415 body-shape rejections."""


class NotFound(MACTFClientError):
    """404."""


class Conflict(MACTFClientError):
    """409: the resource is in a state that refuses the request (``extensions["current_status"]``)."""


class Unauthorized(MACTFClientError):
    """401."""


class Forbidden(MACTFClientError):
    """403 (``extensions["required_scope"]``)."""


class RateLimited(MACTFClientError):
    """429. ``retry_after`` is seconds, from the ``Retry-After`` header."""

    def __init__(self, message: str, *, retry_after: float | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ServerError(MACTFClientError):
    """Any other 5xx."""


class CapabilityUnavailable(MACTFClientError):
    """502 ``urn:mactf:error:sandbox``: this host cannot do that; ``extensions["reasons"]`` says why."""


class TransportError(MACTFClientError):
    """No connection, a timeout, a bad frame, or a lost stream."""


_BY_TYPE: dict[tuple[int, str], type[MACTFClientError]] = {
    (404, "urn:mactf:error:not-found"): NotFound,
    (409, "urn:mactf:error:conflict"): Conflict,
    (413, "urn:mactf:error:too-large"): ValidationError,
    (415, "urn:mactf:error:unsupported-media-type"): ValidationError,
    (429, "urn:mactf:error:rate-limited"): RateLimited,
    (502, "urn:mactf:error:sandbox"): CapabilityUnavailable,
}
_BY_STATUS: dict[int, type[MACTFClientError]] = {
    400: ValidationError,
    401: Unauthorized,
    406: ValidationError,
    403: Forbidden,
    404: NotFound,
    409: Conflict,
    413: ValidationError,
    415: ValidationError,
    422: ValidationError,
    429: RateLimited,
}


def parse_retry_after(value: str | None) -> float | None:
    """Seconds from a ``Retry-After`` header in either of its two forms."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        import time

        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError, IndexError):
        return None


def error_from_problem(
    status: int,
    problem: Mapping[str, Any] | None,
    headers: Mapping[str, str] | None = None,
    *,
    fallback: str = "",
) -> MACTFClientError:
    """The exception for one non-2xx response. ``problem`` is its decoded body, if it had one."""
    problem = problem or {}
    type_uri = problem.get("type")
    message = str(problem.get("detail") or problem.get("title") or fallback or f"HTTP {status}")
    kwargs: dict[str, Any] = {
        "status": status,
        "type_uri": type_uri,
        "trace_id": problem.get("trace_id"),
        "extensions": problem.get("extensions") or {},
    }
    # Status decides the class; ``type`` only separates the 5xx the client can act on.
    cls = (
        _BY_TYPE.get((status, str(type_uri)))
        or _BY_STATUS.get(status)
        or (ServerError if status >= 500 else MACTFClientError)
    )
    if cls is RateLimited:
        lowered = {str(k).lower(): v for k, v in dict(headers or {}).items()}
        retry = parse_retry_after(lowered.get("retry-after"))
        return RateLimited(message, retry_after=retry, **kwargs)
    return cls(message, **kwargs)


def raise_for_response(response: Any) -> Any:
    """Return ``response`` when it is 2xx/3xx; otherwise raise the mapped exception."""
    if response.status_code < 400:
        return response
    try:
        decoded = response.json()
    except (ValueError, UnicodeDecodeError):
        decoded = None
    raise error_from_problem(
        response.status_code,
        decoded if isinstance(decoded, Mapping) else None,
        response.headers,
        fallback=response.content[:200].decode("utf-8", "replace"),
    )


__all__ = [
    "raise_for_response",
    "CapabilityUnavailable", "Conflict", "Forbidden", "MACTFClientError", "NotFound", "RateLimited",
    "ServerError", "TransportError", "Unauthorized", "ValidationError", "error_from_problem",
    "parse_retry_after",
]
