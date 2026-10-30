"""The API's URL names are the contract: unique, reversible, resolvable, and relocatable."""

from __future__ import annotations

import pytest

from mactf import conf
from mactf.client.routes import ROUTES, expand
from mactf.contrib.api import urls as api_urls
from mactf.http import NoReverseMatch, Resolver404, resolve, reverse
from mactf.http.handler import clear_application_cache
from mactf.http.urls import clear_url_caches

SAMPLE = {"run_id": "r1", "episode": 3, "rule_id": "P1", "peer": "p0", "name": "p0", "path": "a/b"}


def _named():
    return [entry for entry in api_urls.urlpatterns if getattr(entry, "name", None)]


def _args(entry):
    template = ROUTES[entry.name]
    return {key: value for key, value in SAMPLE.items() if "{" + key in template}


def test_every_route_name_is_unique_per_path(api_settings):
    seen: dict[str, set[str]] = {}
    for entry in _named():
        seen.setdefault(entry.name, set()).add(str(entry.route))
    # run-list is the one documented name that serves two methods on one path
    assert all(len(paths) == 1 for paths in seen.values()), {
        name: paths for name, paths in seen.items() if len(paths) > 1
    }


def test_the_client_route_table_covers_exactly_the_named_routes(api_settings):
    assert {entry.name for entry in _named()} == set(ROUTES)


def test_every_name_reverses_under_the_base_path_and_resolves_back(api_settings):
    for entry in _named():
        args = _args(entry)
        url = reverse(f"api:{entry.name}", **args)
        assert url == "/api/v1/" + expand(ROUTES[entry.name], **args)
        assert resolve(url).view_name.startswith("api:")


def test_a_missing_argument_does_not_reverse(api_settings):
    with pytest.raises(NoReverseMatch):
        reverse("api:run-detail")
    with pytest.raises(NoReverseMatch):
        reverse("api:not-a-route")


def test_a_path_outside_the_api_does_not_resolve(api_settings):
    with pytest.raises(Resolver404):
        resolve("/elsewhere/runs")


def test_an_integer_converter_rejects_text(api_settings):
    with pytest.raises((NoReverseMatch, ValueError)):
        reverse("api:run-episode-detail", run_id="r", episode="three")
    with pytest.raises(Resolver404):
        resolve("/api/v1/runs/r/episodes/three")


def test_base_path_relocates_every_route(api_settings):
    api_settings["API"]["BASE_PATH"] = "svc/v2"
    conf._settings = api_settings
    clear_url_caches()
    clear_application_cache()
    assert reverse("api:health") == "/svc/v2/health"
    assert reverse("api:run-detail", run_id="x") == "/svc/v2/runs/x"
    with pytest.raises(Resolver404):
        resolve("/api/v1/health")
    assert resolve("/svc/v2/health").view_name == "api:health"


def test_relocated_routes_are_served(api_settings):
    from api_support import ApiClient
    from mactf.http.handler import WSGIHandler

    api_settings["API"]["BASE_PATH"] = "svc/v2"
    conf._settings = api_settings
    clear_url_caches()
    clear_application_cache()
    client = ApiClient(WSGIHandler(), prefix="/svc/v2")
    assert client.get("/health").status_code == 200
    assert ApiClient(WSGIHandler(), prefix="/api/v1").get("/health").status_code == 404
