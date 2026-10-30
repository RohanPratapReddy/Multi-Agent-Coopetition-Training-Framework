"""URL configuration: ``path``, ``re_path``, ``include``, ``resolve`` and ``reverse``.

The shape deliberately matches Django's so its mental model transfers. A project's
``urls.py`` is a module with a ``urlpatterns`` list; ``ROOT_URLCONF`` names it.
"""

from __future__ import annotations

import contextvars
import importlib
import re
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import quote

from ..exceptions import ImproperlyConfigured
from .converters import Converter, compile_route, get_converter
from .exceptions import NoReverseMatch, Resolver404

__all__ = [
    "ResolverMatch",
    "URLPattern",
    "URLResolver",
    "clear_url_caches",
    "default_urlpatterns",
    "get_resolver",
    "get_script_prefix",
    "include",
    "path",
    "re_path",
    "resolve",
    "reverse",
    "set_script_prefix",
]

_SCRIPT_PREFIX: contextvars.ContextVar[str] = contextvars.ContextVar("mactf_script_prefix", default="/")


def set_script_prefix(prefix: str) -> contextvars.Token[str]:
    """Set the mount prefix ``reverse()`` prepends (the WSGI ``SCRIPT_NAME``)."""
    normalised = "/" + prefix.strip("/") + "/" if prefix.strip("/") else "/"
    return _SCRIPT_PREFIX.set(normalised)


def reset_script_prefix(token: contextvars.Token[str]) -> None:
    _SCRIPT_PREFIX.reset(token)


def get_script_prefix() -> str:
    return _SCRIPT_PREFIX.get()


@dataclass(frozen=True)
class ResolverMatch:
    func: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    url_name: str | None
    route: str  # the pattern text, for logs and OpenAPI
    namespaces: tuple[str, ...] = ()
    app_name: str | None = None
    app_names: tuple[str, ...] = field(default=(), compare=False)

    @property
    def view_name(self) -> str:
        return ":".join([*self.namespaces, self.url_name or ""]) if self.url_name else ""

    def __iter__(self):
        yield self.func
        yield self.args
        yield self.kwargs


# A reverse segment is ``("literal", text)`` or ``("param", name)``.
Segment = tuple[str, str]


class _Pattern:
    """Shared compilation for patterns and resolvers."""

    def __init__(self, route: str, *, regex: bool, endpoint: bool) -> None:
        self.route = route
        self.is_regex = regex
        self.converters: dict[str, Converter]
        if regex:
            self.pattern = re.compile(route)
            self.converters = {}
            self.segments = _regex_segments(route, self.pattern)
            self.group_names = tuple(self.pattern.groupindex)
        else:
            self.pattern, self.converters, segments = compile_route(route, endpoint=endpoint)
            self.segments = segments
            self.group_names = tuple(self.converters)

    def _match(self, path: str) -> tuple[re.Match[str], tuple[Any, ...], dict[str, Any]] | None:
        found = self.pattern.fullmatch(path) if self._full else self.pattern.match(path)
        if found is None:
            return None
        kwargs: dict[str, Any] = {}
        for name, raw in found.groupdict().items():
            converter = self.converters.get(name)
            kwargs[name] = converter.to_python(raw) if converter and raw is not None else raw
        args: tuple[Any, ...] = ()
        if self.is_regex and not kwargs:
            args = tuple(found.groups())
        return found, args, kwargs

    _full = True


class URLPattern(_Pattern):
    """One route bound to a view."""

    def __init__(
        self,
        route: str,
        callback: Callable[..., Any],
        *,
        name: str | None = None,
        kwargs: Mapping[str, Any] | None = None,
        regex: bool = False,
    ) -> None:
        super().__init__(route, regex=regex, endpoint=True)
        self.callback = callback
        self.name = name
        self.default_kwargs: Mapping[str, Any] = dict(kwargs or {})
        self._full = True

    def match(self, path: str) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        found = self._match(path)
        if found is None:
            return None
        _, args, captured = found
        return args, {**captured, **self.default_kwargs}

    @property
    def view_class(self) -> Any:
        return getattr(self.callback, "view_class", None)

    def __repr__(self) -> str:
        return f"<URLPattern {self.route!r} name={self.name!r}>"


class URLResolver(_Pattern):
    """A prefix plus a list of patterns (and nested resolvers)."""

    def __init__(
        self,
        route: str,
        url_patterns: Sequence[URLPattern | "URLResolver"],
        *,
        namespace: str | None = None,
        app_name: str | None = None,
        regex: bool = False,
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(route, regex=regex, endpoint=False)
        self._full = False
        self.url_patterns = list(url_patterns)
        for entry in self.url_patterns:
            if not isinstance(entry, (URLPattern, URLResolver)):
                raise ImproperlyConfigured(
                    f"urlpatterns may hold path()/re_path()/include() results, got {entry!r}"
                )
        self.namespace = namespace
        self.app_name = app_name
        self.default_kwargs: Mapping[str, Any] = dict(kwargs or {})

    # -- resolve ----------------------------------------------------------
    def resolve(self, path: str) -> ResolverMatch:
        path = path.lstrip("/") if not self.route else path
        tried: list[str] = []
        match = self._resolve(path, (), (), "", (), tried)
        if match is None:
            raise Resolver404(path="/" + path.lstrip("/"), tried=tuple(tried))
        return match

    def _resolve(
        self,
        path: str,
        namespaces: tuple[str, ...],
        app_names: tuple[str, ...],
        prefix_route: str,
        inherited_args: tuple[Any, ...],
        tried: list[str],
        inherited_kwargs: Mapping[str, Any] | None = None,
    ) -> ResolverMatch | None:
        inherited_kwargs = dict(inherited_kwargs or {})
        found = self._match(path)
        if found is None:
            tried.append(prefix_route + self.route)
            return None
        matched, args, captured = found
        remainder = path[matched.end() :]
        merged_kwargs = {**inherited_kwargs, **captured, **self.default_kwargs}
        merged_args = inherited_args + args
        if self.namespace:
            namespaces = (*namespaces, self.namespace)
        if self.app_name:
            app_names = (*app_names, self.app_name)
        here = prefix_route + self.route
        for entry in self.url_patterns:
            if isinstance(entry, URLResolver):
                sub = entry._resolve(
                    remainder, namespaces, app_names, here, merged_args, tried, merged_kwargs
                )
                if sub is not None:
                    return sub
                continue
            result = entry.match(remainder)
            if result is None:
                tried.append(here + entry.route)
                continue
            sub_args, sub_kwargs = result
            if sub_kwargs or merged_kwargs:
                call_args: tuple[Any, ...] = ()
                call_kwargs = {**merged_kwargs, **sub_kwargs}
            else:
                call_args = merged_args + sub_args
                call_kwargs = {}
            return ResolverMatch(
                func=entry.callback,
                args=call_args,
                kwargs=call_kwargs,
                url_name=entry.name,
                route=here + entry.route,
                namespaces=namespaces,
                app_name=app_names[-1] if app_names else None,
                app_names=app_names,
            )
        return None

    # -- reverse ----------------------------------------------------------
    def reverse(self, name: str, /, *args: Any, **kwargs: Any) -> str:
        if args and kwargs:
            raise NoReverseMatch("reverse() takes positional or keyword arguments, not both")
        *namespace_parts, view_name = name.split(":")
        if not view_name:
            raise NoReverseMatch(f"empty route name in {name!r}")
        failures: list[str] = []
        candidates = list(self._candidates(tuple(namespace_parts), view_name, []))
        for segments, converters, label in candidates:
            built = _build(segments, converters, args, kwargs, failures, label)
            if built is not None:
                return get_script_prefix().rstrip("/") + "/" + built.lstrip("/")
        if candidates:
            raise NoReverseMatch(
                f"reverse for {name!r} failed; tried: " + "; ".join(failures)
            )
        known = sorted({n for n in self.names() if n})
        raise NoReverseMatch(
            f"{name!r} is not a registered route name"
            + (
                " (a route inside a namespace needs the qualified form, e.g. 'api:rule-list')"
                if ":" not in name and any(n.endswith(":" + view_name) for n in known)
                else ""
            )
            + f"; known names: {known}"
        )

    def _candidates(
        self,
        namespace_parts: tuple[str, ...],
        view_name: str,
        prefix: list[Any],
    ) -> Iterable[tuple[list[Segment], dict[str, Converter], str]]:
        yield from self._walk(namespace_parts, view_name, [], {}, "")

    def _walk(
        self,
        namespace_parts: tuple[str, ...],
        view_name: str,
        prefix_segments: list[Segment],
        prefix_converters: dict[str, Converter],
        label: str,
    ) -> Iterable[tuple[list[Segment], dict[str, Converter], str]]:
        segments = [*prefix_segments, *self.segments]
        converters = {**prefix_converters, **self.converters}
        here = label if label else self.route
        for entry in self.url_patterns:
            if isinstance(entry, URLResolver):
                if entry.namespace:
                    if namespace_parts and entry.namespace == namespace_parts[0]:
                        yield from entry._walk(
                            namespace_parts[1:], view_name, segments, converters, here + entry.route
                        )
                else:  # a transparent resolver: its names are visible at this level
                    yield from entry._walk(
                        namespace_parts, view_name, segments, converters, here + entry.route
                    )
            elif not namespace_parts and entry.name == view_name:
                yield (
                    [*segments, *entry.segments],
                    {**converters, **entry.converters},
                    here + entry.route,
                )

    def names(self, namespace: tuple[str, ...] = ()) -> list[str]:
        """Every qualified route name, for error messages and checks."""
        found: list[str] = []
        for entry in self.url_patterns:
            if isinstance(entry, URLResolver):
                child = (*namespace, entry.namespace) if entry.namespace else namespace
                found.extend(entry.names(child))
            elif entry.name:
                found.append(":".join([*namespace, entry.name]))
        return found

    def patterns(self) -> Iterable[tuple[str, URLPattern, tuple[str, ...]]]:
        """Flatten to ``(full_route, pattern, namespaces)`` for OpenAPI and checks."""
        yield from self._flatten("", ())

    def _flatten(
        self, prefix: str, namespaces: tuple[str, ...]
    ) -> Iterable[tuple[str, URLPattern, tuple[str, ...]]]:
        here = prefix + self.route
        if self.namespace:
            namespaces = (*namespaces, self.namespace)
        for entry in self.url_patterns:
            if isinstance(entry, URLResolver):
                yield from entry._flatten(here, namespaces)
            else:
                yield here + entry.route, entry, namespaces

    def __repr__(self) -> str:
        return f"<URLResolver {self.route!r} ns={self.namespace!r} n={len(self.url_patterns)}>"


def _build(
    segments: Sequence[Segment],
    converters: Mapping[str, Converter],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    failures: list[str],
    label: str,
) -> str | None:
    params = [value for kind, value in segments if kind == "param"]
    expected = ", ".join(f"{name}:{converters[name].name if name in converters else 'regex'}" for name in params)
    if args:
        if len(args) != len(params):
            failures.append(f"{label!r} takes {len(params)} argument(s) ({expected}), got {len(args)}")
            return None
        values = dict(zip(params, args))
    else:
        if set(kwargs) != set(params):
            missing = [name for name in params if name not in kwargs]
            extra = [name for name in kwargs if name not in params]
            detail = []
            if missing:
                detail.append(f"missing {missing}")
            if extra:
                detail.append(f"unexpected {extra}")
            failures.append(f"{label!r} expects ({expected}); " + ", ".join(detail))
            return None
        values = dict(kwargs)
    out: list[str] = []
    for kind, value in segments:
        if kind == "literal":
            out.append(value)
            continue
        converter = converters.get(value)
        raw = values[value]
        if converter is None:
            text = str(raw)
        else:
            try:
                text = converter.to_url(raw)
            except (TypeError, ValueError):
                failures.append(f"{label!r}: {raw!r} is not a valid {converter.name} for {value!r}")
                return None
            if re.fullmatch(converter.regex, text, re.DOTALL) is None:
                failures.append(f"{label!r}: {raw!r} does not match {converter.name} for {value!r}")
                return None
        out.append(quote(text, safe="/" if converter is not None and converter.name == "path" else ""))
    return "".join(out)


def _regex_segments(route: str, compiled: re.Pattern[str]) -> list[Segment]:
    """Replay a simple regex as literal/param segments for ``reverse()``.

    Supports literals and ``(?P<name>...)`` groups; anything cleverer (alternation
    outside a group, lookaheads) is routable but not reversible, which is the same
    contract Django documents.
    """
    text = route[1:] if route.startswith("^") else route
    text = text[:-1] if text.endswith("$") and not text.endswith("\\$") else text
    segments: list[Segment] = []
    literal: list[str] = []
    index = 0
    while index < len(text):
        if text.startswith("(?P<", index):
            end_name = text.index(">", index)
            name = text[index + 4 : end_name]
            depth = 1
            cursor = end_name + 1
            while cursor < len(text) and depth:
                if text[cursor] == "\\":
                    cursor += 2
                    continue
                if text[cursor] == "(":
                    depth += 1
                elif text[cursor] == ")":
                    depth -= 1
                cursor += 1
            if literal:
                segments.append(("literal", "".join(literal)))
                literal = []
            segments.append(("param", name))
            index = cursor
            continue
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            literal.append(text[index + 1])
            index += 2
            continue
        literal.append(char)
        index += 1
    if literal:
        segments.append(("literal", "".join(literal)))
    return segments


# ---------------------------------------------------------------------------
# Public constructors
# ---------------------------------------------------------------------------
def _make(route: str, view: Any, *, name: str | None, kwargs: Mapping[str, Any] | None, regex: bool):
    if isinstance(view, tuple) and len(view) == 3:
        patterns, app_name, namespace = view
        return URLResolver(
            route, patterns, namespace=namespace, app_name=app_name, regex=regex, kwargs=kwargs
        )
    if isinstance(view, (list, tuple)):
        return URLResolver(route, list(view), regex=regex, kwargs=kwargs)
    if isinstance(view, type):
        raise ImproperlyConfigured(
            f"route {route!r}: pass {view.__name__}.as_view(), not the class itself"
        )
    if not callable(view):
        raise ImproperlyConfigured(f"route {route!r}: view must be callable, got {view!r}")
    return URLPattern(route, view, name=name, kwargs=kwargs, regex=regex)


def path(
    route: str, view: Any, *, name: str | None = None, kwargs: Mapping[str, Any] | None = None
) -> URLPattern | URLResolver:
    return _make(route, view, name=name, kwargs=kwargs, regex=False)


def re_path(
    regex: str, view: Any, *, name: str | None = None, kwargs: Mapping[str, Any] | None = None
) -> URLPattern | URLResolver:
    return _make(regex, view, name=name, kwargs=kwargs, regex=True)


def include(
    arg: Any, *, namespace: str | None = None, app_name: str | None = None
) -> tuple[list[URLPattern | URLResolver], str | None, str | None]:
    """A module, dotted path, ``(patterns, app_name)`` tuple, or list of patterns.

    The namespace defaults to the module's ``app_name``, which is what makes
    ``reverse("api:rule-list")`` work without the project naming anything.
    """
    patterns: Any
    if isinstance(arg, str):
        arg = importlib.import_module(arg)
    if isinstance(arg, ModuleType):
        try:
            patterns = arg.urlpatterns  # type: ignore[attr-defined]
        except AttributeError:
            raise ImproperlyConfigured(f"{arg.__name__} has no urlpatterns") from None
        app_name = app_name or getattr(arg, "app_name", None)
    elif isinstance(arg, tuple) and len(arg) == 2:
        patterns, declared = arg
        app_name = app_name or declared
        if isinstance(patterns, (str, ModuleType)):
            return include(patterns, namespace=namespace, app_name=app_name)
    elif isinstance(arg, (list, tuple)):
        patterns = arg
    else:
        raise ImproperlyConfigured(
            f"include() takes a module, dotted path, (patterns, app_name) or a list, got {arg!r}"
        )
    return list(patterns), app_name, namespace if namespace is not None else app_name


# ---------------------------------------------------------------------------
# Loading and caching
# ---------------------------------------------------------------------------
_RESOLVERS: dict[tuple[int, str | None], tuple[Any, URLResolver]] = {}


def default_urlpatterns() -> list[URLPattern | URLResolver]:
    """What ``ROOT_URLCONF = None`` selects: the API app's routes, if it is installed."""
    from ..conf import get_settings

    settings = get_settings()
    installed = settings.get("INSTALLED_APPS") or []
    if any(str(entry).startswith("mactf.contrib.api") for entry in installed):
        base = str((settings.get("API") or {}).get("BASE_PATH", "api/v1")).strip("/")
        patterns = [path(f"{base}/" if base else "", include("mactf.contrib.api.urls"))]
    else:
        patterns = []
    if any(str(entry).startswith("mactf.distributed") for entry in installed):
        worker = str((settings.get("DISTRIBUTED") or {}).get("BASE_PATH", "worker/v1")).strip("/")
        patterns.append(path(f"{worker}/" if worker else "", include("mactf.distributed.urls")))
    return patterns  # type: ignore[return-value]


def _load_patterns(urlconf: str | ModuleType | None) -> list[Any]:
    if urlconf is None:
        return default_urlpatterns()
    module = importlib.import_module(urlconf) if isinstance(urlconf, str) else urlconf
    try:
        return list(module.urlpatterns)
    except AttributeError:
        raise ImproperlyConfigured(f"URL conf {getattr(module, '__name__', module)!r} has no urlpatterns") from None


def _check_unique_names(resolver: URLResolver) -> None:
    seen: dict[tuple[str, ...], set[str]] = {}

    def walk(node: URLResolver, namespace: tuple[str, ...]) -> None:
        for entry in node.url_patterns:
            if isinstance(entry, URLResolver):
                walk(entry, (*namespace, entry.namespace) if entry.namespace else namespace)
            elif entry.name:
                names = seen.setdefault(namespace, set())
                if entry.name in names:
                    scope = ":".join(namespace) or "<root>"
                    raise ImproperlyConfigured(
                        f"duplicate route name {entry.name!r} in namespace {scope!r}"
                    )
                names.add(entry.name)

    walk(resolver, ())


def get_resolver(urlconf: str | ModuleType | None = None) -> URLResolver:
    """The resolver for ``urlconf`` (default: ``ROOT_URLCONF``), cached per settings object."""
    from ..conf import get_settings

    settings = get_settings()
    name = urlconf if urlconf is not None else settings.get("ROOT_URLCONF")
    key = (id(settings), name if isinstance(name, (str, type(None))) else name.__name__)
    cached = _RESOLVERS.get(key)
    if cached is not None and cached[0] is settings:
        return cached[1]
    resolver = URLResolver("", _load_patterns(name))
    _check_unique_names(resolver)
    _RESOLVERS[key] = (settings, resolver)
    return resolver


def clear_url_caches() -> None:
    """The only supported way to pick up a changed ``ROOT_URLCONF`` in a live process."""
    _RESOLVERS.clear()


def resolve(path_: str, urlconf: str | ModuleType | None = None) -> ResolverMatch:
    """Match ``path_`` (SCRIPT_NAME stripped); ``Resolver404`` when nothing matches."""
    prefix = get_script_prefix()
    if prefix != "/" and path_.startswith(prefix.rstrip("/")):
        path_ = path_[len(prefix.rstrip("/")) :]
    return get_resolver(urlconf).resolve(path_)


def reverse(name: str, /, *args: Any, urlconf: str | ModuleType | None = None, **kwargs: Any) -> str:
    """Build a URL from a route name. ``NoReverseMatch`` when it cannot."""
    return get_resolver(urlconf).reverse(name, *args, **kwargs)
