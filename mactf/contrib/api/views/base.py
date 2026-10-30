"""``ApiView``: scope enforcement, headers, body parsing and pagination helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from ....http.exceptions import (
    BadRequest,
    Conflict,
    Http404,
    HttpError,
    NotAuthenticated,
    RateLimited,
    ServerError,
    ValidationError,
    register_handler,
)
from ....http.middleware import convert_exception
from ....http.request import Request
from ....http.response import JsonResponse, Response
from ....http.serializers import Serializer
from ....http.views import View
from ....exceptions import (
    ComponentBuildError,
    ComponentCollision,
    ComponentContractError,
    ComponentImportDenied,
    ComponentLocked,
    ComponentMigrationError,
    ComponentMissing,
    ComponentParamError,
    ComponentSpecError,
    InvalidArgument,
    MutationError,
    ResourceNotFound,
    SandboxUnavailable,
    StateConflict,
    UnsafePolicyDenied,
)
from ....orchestration.runview import decode_json_columns, decode_run
from ....orchestration.live import CapacityExceeded, RunExists, RunStateError
from .. import deps
from ..auth import authorize

API_VERSION = "v1"

#: Read-only methods need no body; used to decide whether OPTIONS/HEAD skip auth.
_SAFE = {"options"}


# -- domain errors -> problem+json ------------------------------------------
def _translate(exc: BaseException) -> BaseException:
    """Core exceptions -> the HTTP exception classes the API already uses, so every
    problem document is byte-identical to what the inlined code produced."""
    # Document 17 section 16.3.4: a component error below any view becomes the right problem.
    if isinstance(exc, UnsafePolicyDenied):  # a ComponentLocked, but a configuration mistake (17.1.3: 422)
        return ValidationError({"policy": [str(exc)]}, extensions={"code": exc.code})
    if isinstance(exc, ComponentMigrationError):  # 17.1.3: 409, the old component keeps running
        return Conflict(str(exc), extensions={"code": exc.code})
    if isinstance(exc, ComponentBuildError):  # 17.1.3: 500 as problem+json carrying the code
        return ServerError(str(exc), extensions={"code": exc.code})
    if isinstance(exc, (ComponentCollision, ComponentLocked)):
        return Conflict(str(exc), extensions={"code": exc.code, "slot": getattr(exc, "path", None) or exc.name})
    if isinstance(exc, ComponentMissing):
        return Http404(str(exc), extensions={"code": exc.code, "known": list(exc.known)})
    if isinstance(exc, (ComponentParamError, ComponentSpecError, ComponentImportDenied, ComponentContractError)):
        return ValidationError({"spec": [str(exc)]}, extensions={"code": exc.code})
    if isinstance(exc, MutationError):
        return ValidationError(
            {"plan": [str(exc)]}, extensions={"code": exc.code, "findings": list(exc.findings)}
        )
    if isinstance(exc, ResourceNotFound):
        return Http404(str(exc))
    if isinstance(exc, StateConflict):
        return Conflict(str(exc), extensions=dict(exc.details))
    if isinstance(exc, InvalidArgument):
        return ValidationError({exc.field or "detail": [str(exc)]})
    return exc


def _state_error(exc: BaseException) -> tuple:
    assert isinstance(exc, RunStateError)
    return (
        409, "urn:mactf:error:conflict", "Conflict", str(exc),
        {"current_status": exc.current_status, "action": exc.action, "run_id": exc.run_id}, {},
    )


def _exists(exc: BaseException) -> tuple:
    return (409, "urn:mactf:error:conflict", "Conflict", str(exc), {"run_id": getattr(exc, "run_id", "")}, {})


def _capacity(exc: BaseException) -> tuple:
    assert isinstance(exc, CapacityExceeded)
    retry = int(max(1, round(exc.retry_after)))
    return (
        429, "urn:mactf:error:rate-limited", "Too Many Requests", str(exc),
        {"limit": exc.limit, "retry_after": exc.retry_after}, {"Retry-After": str(retry)},
    )


def _sandbox_unavailable(exc: BaseException) -> tuple:
    return (
        503, "urn:mactf:error:sandbox-unavailable", "Sandbox Unavailable", str(exc),
        {"reason": str(exc)}, {},
    )


register_handler(RunStateError, _state_error)
register_handler(RunExists, _exists)
register_handler(CapacityExceeded, _capacity)
register_handler(SandboxUnavailable, _sandbox_unavailable)


class ApiView(View):
    """Base for every API view.

    ``scopes`` maps a lower-case method to the scope it needs; a method that is
    absent needs none. ``public = True`` skips authentication entirely (``/health``).
    ``operation_ids`` maps a method to the route name it serves, for a view that
    backs two named routes on one path.
    """

    scopes: Mapping[str, str] = {"get": "read"}
    public: bool = False
    operation_ids: Mapping[str, str] = {}
    tags: tuple[str, ...] = ("api",)

    # -- dispatch ---------------------------------------------------------------
    def dispatch(self, request: Request, *args: Any, **kwargs: Any) -> Response:
        try:
            method = request.method.lower()
            if not self.public and method not in _SAFE:
                lookup = "get" if method == "head" else method
                scope = self.scopes.get(lookup)
                if scope:
                    authorize(request, [scope])
            response = super().dispatch(request, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - headers must ride on error responses too
            exc = _translate(exc)
            if isinstance(exc, NotAuthenticated):
                exc.headers.setdefault("WWW-Authenticate", "Bearer")
            response = convert_exception(request, exc)
            if getattr(request, "request_id", ""):
                response["X-Trace-Id"] = request.request_id
        return self._decorate(response)

    @staticmethod
    def _decorate(response: Response) -> Response:
        from .... import __version__

        response["X-MACTF-Version"] = str(__version__)
        response["X-API-Version"] = API_VERSION
        return response

    # -- helpers ---------------------------------------------------------------
    def body(self, request: Request, serializer: type[Serializer]) -> dict[str, Any]:
        """The validated JSON body (422 on invalid, 400 on malformed JSON)."""
        payload = request.json()
        instance = serializer(data=payload)
        instance.is_valid(raise_exception=True)
        return instance.validated_data

    def optional_body(self, request: Request, serializer: type[Serializer]) -> dict[str, Any]:
        """Like :meth:`body` but an empty body means "all defaults"."""
        if not request.body.strip():
            instance = serializer(data={})
        else:
            instance = serializer(data=request.json())
        instance.is_valid(raise_exception=True)
        return instance.validated_data


# -- shared helpers -------------------------------------------------------------
def int_param(request: Request, name: str, default: int | None = None, *, minimum: int = 0, maximum: int | None = None) -> int | None:
    raw = request.GET.get(name)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValidationError({name: ["must be an integer."]}) from None
    if value < minimum or (maximum is not None and value > maximum):
        bound = f">= {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        raise ValidationError({name: [f"must be {bound}."]})
    return value


def page_params(request: Request, *, default: int = 50, maximum: int = 500) -> tuple[int, int]:
    limit = int_param(request, "limit", default, minimum=1, maximum=maximum)
    offset = int_param(request, "offset", 0, minimum=0)
    return int(limit or default), int(offset or 0)


def paginated(items: list[Any], total: int, limit: int, offset: int) -> dict[str, Any]:
    return {
        "items": items,
        "total": int(total),
        "limit": limit,
        "offset": offset,
        "has_more": offset + len(items) < total,
    }


def paginate_list(items: list[Any], limit: int, offset: int) -> dict[str, Any]:
    return paginated(items[offset: offset + limit], len(items), limit, offset)


def bool_param(request: Request, name: str, default: bool = False) -> bool:
    raw = request.GET.get(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off", ""):
        return False
    raise ValidationError({name: ["must be a boolean."]})


_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def get_run_or_404(request: Request, run_id: str) -> dict[str, Any]:
    """The stored run row with the live status laid over it, else 404."""
    if not _RUN_ID.match(run_id):
        raise Http404(f"no run {run_id!r}")
    manager = deps.get_run_manager_dep(request)
    row = manager.get(run_id)
    if row is None:
        raise Http404(f"no run {run_id!r}")
    return row


run_payload = decode_run


def run_is_live(request: Request, run_id: str) -> bool:
    manager = deps.get_run_manager_dep(request)
    return run_id in manager.live_run_ids()


def conflict(detail: str, **extensions: Any) -> Conflict:
    return Conflict(detail, extensions=extensions)


__all__ = [
    "API_VERSION", "ApiView", "BadRequest", "HttpError", "JsonResponse", "RateLimited",
    "bool_param", "conflict", "decode_json_columns", "get_run_or_404", "int_param",
    "page_params", "paginate_list", "paginated", "run_is_live", "run_payload",
]
