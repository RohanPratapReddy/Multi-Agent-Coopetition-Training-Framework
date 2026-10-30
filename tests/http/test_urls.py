"""Criterion 40: every named ``path()`` round-trips through ``reverse()`` and ``resolve()``."""

from __future__ import annotations

import pytest
from http_fixtures import make_urlconf

from mactf.http import NoReverseMatch, Resolver404, resolve, reverse
from mactf.http.urls import URLPattern, URLResolver, get_resolver

SAMPLES = {"int": 3, "str": "abc", "slug": "a-b", "path": "x/y", "uuid": "12345678-1234-5678-1234-567812345678"}


def _walk(resolver, namespace=()):
    for entry in resolver.url_patterns:
        if isinstance(entry, URLResolver):
            ns = (*namespace, entry.namespace) if getattr(entry, "namespace", None) else namespace
            yield from _walk(entry, ns)
        elif isinstance(entry, URLPattern) and entry.name:
            yield ":".join((*namespace, entry.name)), entry


def test_the_fixture_conf_has_named_routes():
    names = [name for name, _ in _walk(get_resolver(make_urlconf()))]
    assert {"items:item-list", "items:item-detail", "boom", "plain", "stream"} <= set(names)


def test_every_fixture_route_round_trips(settings):
    conf = make_urlconf()
    for name, _ in _walk(get_resolver(conf)):
        kwargs = {"item_id": 9} if name.endswith("item-detail") else {}
        url = reverse(name, urlconf=conf, **kwargs)
        match = resolve(url, urlconf=conf)
        assert match.view_name == name
        assert match.kwargs == kwargs


def test_every_api_route_round_trips(api_settings_for_urls):
    from mactf.contrib.api.urls import urlpatterns

    checked = 0
    sharing: dict[str, set[str]] = {}
    for entry in urlpatterns:
        name = getattr(entry, "name", None)
        if not name:
            continue
        converters = getattr(entry, "converters", None) or {}
        kwargs = {key: SAMPLES.get(getattr(conv, "name", "str"), "abc") for key, conv in converters.items()}
        if "episode" in kwargs:
            kwargs["episode"] = 2
        url = reverse(f"api:{name}", **kwargs)
        match = resolve(url)
        # one path can carry two names (list/create); resolve() returns the first registered
        assert match.view_name in sharing.get(url, {f"api:{name}"}) | {f"api:{name}"}
        sharing.setdefault(url, set()).add(f"api:{name}")
        assert {k: str(v) for k, v in match.kwargs.items()} == {k: str(v) for k, v in kwargs.items()}
        checked += 1
    assert checked >= 40


def test_a_converter_rejects_a_bad_value_on_both_directions(settings):
    conf = make_urlconf()
    with pytest.raises((NoReverseMatch, ValueError)):
        reverse("items:item-detail", urlconf=conf, item_id="abc")
    with pytest.raises(Resolver404):
        resolve("/api/items/abc/", urlconf=conf)


def test_missing_arguments_do_not_reverse(settings):
    with pytest.raises(NoReverseMatch):
        reverse("items:item-detail", urlconf=make_urlconf())
