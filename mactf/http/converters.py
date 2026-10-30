"""Route converters: the ``{name:converter}`` vocabulary of ``path()``."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from ..exceptions import ImproperlyConfigured


@dataclass(frozen=True)
class Converter:
    name: str
    regex: str
    to_python: Callable[[str], Any]
    to_url: Callable[[Any], str]
    #: OpenAPI ``schema`` for the path parameter.
    schema: dict[str, Any]

    def accepts(self, value: Any) -> bool:
        try:
            text = self.to_url(value)
        except (TypeError, ValueError):
            return False
        return re.fullmatch(self.regex, text, re.DOTALL) is not None


def _int_to_url(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("bool is not an int route value")
    number = int(value)
    if number < 0:
        raise ValueError("negative int in an int route")
    return str(number)


def _uuid_to_url(value: Any) -> str:
    return str(value if isinstance(value, uuid.UUID) else uuid.UUID(str(value)))


CONVERTERS: dict[str, Converter] = {
    "str": Converter("str", r"[^/]+", str, str, {"type": "string"}),
    "int": Converter("int", r"[0-9]+", int, _int_to_url, {"type": "integer", "minimum": 0}),
    "slug": Converter(
        "slug", r"[-a-zA-Z0-9_]+", str, str, {"type": "string", "pattern": r"^[-a-zA-Z0-9_]+$"}
    ),
    "uuid": Converter(
        "uuid",
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        uuid.UUID,
        _uuid_to_url,
        {"type": "string", "format": "uuid"},
    ),
    "path": Converter("path", r".*", str, str, {"type": "string"}),
}


def get_converter(name: str) -> Converter:
    try:
        return CONVERTERS[name]
    except KeyError:
        raise ImproperlyConfigured(
            f"unknown route converter {name!r}; known converters are {sorted(CONVERTERS)}"
        ) from None


def register_converter(converter: Converter) -> None:
    """Add a converter. Re-registering a name is refused: routes already compiled use it."""
    if converter.name in CONVERTERS:
        raise ImproperlyConfigured(f"route converter {converter.name!r} is already registered")
    CONVERTERS[converter.name] = converter


_PARAMETER = re.compile(r"\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::(?P<converter>[A-Za-z_][A-Za-z0-9_]*))?\}")


def compile_route(route: str, *, endpoint: bool) -> tuple[re.Pattern[str], dict[str, Converter], list[tuple[str, Any]]]:
    """Compile ``"runs/{run_id}/episodes/{episode:int}"`` to a regex.

    Returns ``(pattern, converters_by_name, segments)`` where ``segments`` is the
    literal/parameter sequence ``reverse()`` replays. A parameter name declared
    twice is an ``ImproperlyConfigured`` at import of the URL conf, not at request
    time.
    """
    converters: dict[str, Converter] = {}
    segments: list[tuple[str, Any]] = []
    regex = ["^"]
    cursor = 0
    for match in _PARAMETER.finditer(route):
        literal = route[cursor : match.start()]
        _reject_stray_braces(route, literal)
        if literal:
            regex.append(re.escape(literal))
            segments.append(("literal", literal))
        name = match.group("name")
        if name in converters:
            raise ImproperlyConfigured(
                f"route {route!r} declares the parameter {name!r} more than once"
            )
        converter = get_converter(match.group("converter") or "str")
        converters[name] = converter
        regex.append(f"(?P<{name}>{converter.regex})")
        segments.append(("param", name))
        cursor = match.end()
    tail = route[cursor:]
    _reject_stray_braces(route, tail)
    if tail:
        regex.append(re.escape(tail))
        segments.append(("literal", tail))
    if endpoint:
        regex.append("$")
    return re.compile("".join(regex), re.DOTALL), converters, segments


def _reject_stray_braces(route: str, literal: str) -> None:
    if "{" in literal or "}" in literal:
        raise ImproperlyConfigured(
            f"route {route!r} has a malformed parameter near {literal!r}; "
            "write {name} or {name:converter}"
        )


def route_template(route: str) -> str:
    """The OpenAPI ``/{param}`` form of a route (converter names dropped)."""
    return _PARAMETER.sub(lambda match: "{" + match.group("name") + "}", route)
