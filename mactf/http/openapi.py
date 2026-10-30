"""OpenAPI generation from the URL conf and serializer metadata, with no framework.

Inputs: the resolved URL conf, each view's ``view_class``/docstring/attributes, and
the serializer field metadata. It does not validate that the schema matches reality
and does not infer a response model from a return statement; the route table test
remains the completeness gate.

A view class may declare, on the class or per method (``{"get": X, "post": Y}``):

``summary``, ``description``, ``tags``         documentation
``request_serializer``                          body of POST/PUT/PATCH
``query_serializer``                            query parameters
``response_serializer``                         the success body
``responses``                                   ``{status: Serializer | str}`` extras
"""

from __future__ import annotations

import inspect
from typing import Any, Mapping

from .converters import route_template
from .serializers import Serializer
from .status import STATUS_TEXT
from .urls import URLPattern, get_resolver

__all__ = ["schema"]

_BODY_METHODS = {"post", "put", "patch"}
_SUCCESS = {"get": 200, "post": 200, "put": 200, "patch": 200, "delete": 204}

PROBLEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "RFC 9457 problem details.",
    "required": ["type", "title", "status", "detail", "instance", "trace_id", "extensions"],
    "properties": {
        "type": {"type": "string", "description": "A urn:mactf:error:* identifier."},
        "title": {"type": "string"},
        "status": {"type": "integer"},
        "detail": {"type": "string"},
        "instance": {"type": "string", "description": "The request path."},
        "trace_id": {"type": "string", "description": "Also returned as X-Trace-Id."},
        "extensions": {"type": "object", "additionalProperties": True},
    },
}


def schema(
    *, title: str = "MACTF", version: str | None = None, base_path: str = "api/v1", urlconf: Any = None
) -> dict[str, Any]:
    """The OpenAPI 3.0 document for every route reachable from ``ROOT_URLCONF``."""
    if version is None:
        try:
            from importlib.metadata import version as _version

            version = _version("mactf")
        except Exception:  # noqa: BLE001
            version = "0.0.0"
    resolver = get_resolver(urlconf)
    paths: dict[str, dict[str, Any]] = {}
    for route, pattern, namespaces in resolver.patterns():
        template = "/" + _template(route, pattern)
        entry = paths.setdefault(template, {})
        for method in _methods(pattern):
            if method not in entry:  # a second name on the same path adds no second operation
                entry[method] = _operation(pattern, method, namespaces, route)
    return {
        "openapi": "3.0.3",
        "info": {"title": title, "version": version, "x-base-path": base_path},
        "paths": dict(sorted(paths.items())),
        "components": {
            "schemas": {"Problem": PROBLEM_SCHEMA},
            "responses": {
                "Problem": {
                    "description": "An RFC 9457 problem document.",
                    "content": {"application/problem+json": {"schema": {"$ref": "#/components/schemas/Problem"}}},
                }
            },
        },
    }


def _template(route: str, pattern: URLPattern) -> str:
    if not pattern.is_regex:
        return route_template(route)
    from .urls import _regex_segments  # noqa: PLC0415

    parts = []
    for kind, value in _regex_segments(route, pattern.pattern):
        parts.append(value if kind == "literal" else "{" + value + "}")
    return "".join(parts)


def _view_class(pattern: URLPattern) -> Any:
    return getattr(pattern.callback, "view_class", None)


def _methods(pattern: URLPattern) -> list[str]:
    cls = _view_class(pattern)
    if cls is not None:
        found = [m for m in cls.http_method_names if hasattr(cls, m) and m not in ("options", "head")]
        return found or ["get"]
    declared = getattr(pattern.callback, "allowed_methods", None)
    if declared:
        return [m.lower() for m in declared if m.lower() not in ("options", "head")]
    return ["get"]


def _per_method(owner: Any, attribute: str, method: str, default: Any = None) -> Any:
    value = getattr(owner, attribute, default)
    if isinstance(value, Mapping) and value and set(value) <= {
        "get", "post", "put", "patch", "delete",
    }:
        return value.get(method, default)
    return value


def _summary(owner: Any, method: str) -> str:
    text = _per_method(owner, "summary", method, "") or ""
    if text:
        return str(text)
    handler = getattr(owner, method, None) if inspect.isclass(owner) else owner
    doc = inspect.getdoc(handler) or inspect.getdoc(owner) or ""
    return doc.strip().split("\n", 1)[0].strip()


def _operation(
    pattern: URLPattern, method: str, namespaces: tuple[str, ...], route: str
) -> dict[str, Any]:
    cls = _view_class(pattern)
    owner = cls if cls is not None else pattern.callback
    name = pattern.name or route_template(route).strip("/").replace("/", "-") or "root"
    methods = _methods(pattern)
    # One name may serve several methods (``run-list`` is GET and POST): the
    # primary method carries the bare route name, the rest are suffixed.
    primary = "get" if "get" in methods else methods[0]
    operation_id = name if method == primary else f"{name}.{method}"
    # A view that serves two named routes on one path (list + create) says which
    # name belongs to which method; that name is the operationId.
    declared_ids = getattr(owner, "operation_ids", None)
    if isinstance(declared_ids, Mapping) and method in declared_ids:
        operation_id = str(declared_ids[method])
    summary = _summary(owner, method) or name.replace("-", " ").capitalize()
    description = str(_per_method(owner, "description", method, "") or "") or (
        inspect.getdoc(owner) or summary
    )
    tags = _per_method(owner, "tags", method, ()) or (namespaces[-1] if namespaces else "default",)
    operation: dict[str, Any] = {
        "operationId": operation_id,
        "summary": summary,
        "description": description,
        "tags": list(tags if not isinstance(tags, str) else (tags,)),
        "x-route-name": ":".join([*namespaces, name]),
        "parameters": _parameters(pattern, owner, method),
        "responses": _responses(owner, method),
    }
    request_serializer = _per_method(owner, "request_serializer", method)
    if request_serializer is not None and method in _BODY_METHODS:
        operation["requestBody"] = {
            "required": True,
            "content": {"application/json": {"schema": _schema_of(request_serializer)}},
        }
    return operation


def _schema_of(target: Any) -> dict[str, Any]:
    if inspect.isclass(target) and issubclass(target, Serializer):
        return target.schema()
    if isinstance(target, Mapping):
        return dict(target)
    return {"type": "object", "additionalProperties": True}


def _parameters(pattern: URLPattern, owner: Any, method: str) -> list[dict[str, Any]]:
    params: list[dict[str, Any]] = []
    for name, converter in pattern.converters.items():
        params.append({"name": name, "in": "path", "required": True, "schema": dict(converter.schema)})
    if pattern.is_regex:
        for name in pattern.group_names:
            params.append({"name": name, "in": "path", "required": True, "schema": {"type": "string"}})
    query = _per_method(owner, "query_serializer", method)
    if inspect.isclass(query) and issubclass(query, Serializer):
        for name, field in query._get_fields().items():
            entry = {
                "name": name,
                "in": "query",
                "required": field.required,
                "schema": field.schema(),
            }
            if field.help_text:
                entry["description"] = field.help_text
            params.append(entry)
    return params


def _responses(owner: Any, method: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    success = _SUCCESS.get(method, 200)
    body = _per_method(owner, "response_serializer", method)
    declared = _per_method(owner, "responses", method) or {}
    if success == 204 and body is None:
        out["204"] = {"description": STATUS_TEXT[204]}
    else:
        content = {"application/json": {"schema": _schema_of(body)}} if body is not None else {}
        entry: dict[str, Any] = {"description": "Successful response"}
        if content:
            entry["content"] = content
        out[str(success)] = entry
    for code, value in dict(declared).items():
        code = int(code)
        key = str(code)
        if 200 <= code < 300:
            entry = {"description": STATUS_TEXT.get(code, "Success")}
            if inspect.isclass(value) and issubclass(value, Serializer):
                entry["content"] = {"application/json": {"schema": value.schema()}}
            elif isinstance(value, str):
                entry["description"] = value
            out[key] = entry
        else:
            description = value if isinstance(value, str) else STATUS_TEXT.get(code, "Error")
            out[key] = {
                "description": description,
                "content": {"application/problem+json": {"schema": {"$ref": "#/components/schemas/Problem"}}},
            }
    out["default"] = {"$ref": "#/components/responses/Problem"}
    return out
