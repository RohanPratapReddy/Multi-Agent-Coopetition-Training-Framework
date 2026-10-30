"""HTTP status constants: exactly the codes document 02 uses, and no others."""

from __future__ import annotations

OK = 200
CREATED = 201
ACCEPTED = 202
NO_CONTENT = 204
MOVED_PERMANENTLY = 301
NOT_MODIFIED = 304
BAD_REQUEST = 400
UNAUTHORIZED = 401
FORBIDDEN = 403
NOT_FOUND = 404
METHOD_NOT_ALLOWED = 405
NOT_ACCEPTABLE = 406
CONFLICT = 409
GONE = 410
REQUEST_ENTITY_TOO_LARGE = 413
UNSUPPORTED_MEDIA_TYPE = 415
UNPROCESSABLE_ENTITY = 422
TOO_MANY_REQUESTS = 429
INTERNAL_SERVER_ERROR = 500
BAD_GATEWAY = 502
SERVICE_UNAVAILABLE = 503

STATUS_TEXT: dict[int, str] = {
    OK: "OK",
    CREATED: "Created",
    ACCEPTED: "Accepted",
    NO_CONTENT: "No Content",
    MOVED_PERMANENTLY: "Moved Permanently",
    NOT_MODIFIED: "Not Modified",
    BAD_REQUEST: "Bad Request",
    UNAUTHORIZED: "Unauthorized",
    FORBIDDEN: "Forbidden",
    NOT_FOUND: "Not Found",
    METHOD_NOT_ALLOWED: "Method Not Allowed",
    NOT_ACCEPTABLE: "Not Acceptable",
    CONFLICT: "Conflict",
    GONE: "Gone",
    REQUEST_ENTITY_TOO_LARGE: "Payload Too Large",
    UNSUPPORTED_MEDIA_TYPE: "Unsupported Media Type",
    UNPROCESSABLE_ENTITY: "Unprocessable Entity",
    TOO_MANY_REQUESTS: "Too Many Requests",
    INTERNAL_SERVER_ERROR: "Internal Server Error",
    BAD_GATEWAY: "Bad Gateway",
    SERVICE_UNAVAILABLE: "Service Unavailable",
}


def status_line(code: int) -> str:
    """``"404 Not Found"``. An undefined code is a programming error, not a guess."""
    try:
        return f"{code} {STATUS_TEXT[code]}"
    except KeyError:
        raise ValueError(
            f"status {code} is not one the framework returns; "
            f"defined codes are {sorted(STATUS_TEXT)}"
        ) from None


def is_success(code: int) -> bool:
    return 200 <= code < 300


def is_redirect(code: int) -> bool:
    return 300 <= code < 400


def is_client_error(code: int) -> bool:
    return 400 <= code < 500


def is_server_error(code: int) -> bool:
    return 500 <= code < 600
