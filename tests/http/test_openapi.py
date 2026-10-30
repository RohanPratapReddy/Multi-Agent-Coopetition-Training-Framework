"""Criterion 45: the OpenAPI document has an operation per route and ``operationId == name``."""

from __future__ import annotations

from http_fixtures import make_urlconf

from mactf.http import openapi


def _operations(doc):
    for template, item in doc["paths"].items():
        for method, operation in item.items():
            yield template, method, operation


def test_fixture_conf_emits_one_operation_per_route_method(settings):
    doc = openapi.schema(urlconf=make_urlconf(), title="T", version="1")
    found = {(t, m) for t, m, _ in _operations(doc)}
    assert {("/api/items/", "get"), ("/api/items/", "post"), ("/api/items/{item_id}/", "get")} <= found
    ids = [op["operationId"] for _, _, op in _operations(doc)]
    owners = {}
    for template, _, op in _operations(doc):
        owners.setdefault(op["operationId"], set()).add(template)
    assert all(len(paths) == 1 for paths in owners.values())


def test_operation_ids_are_the_route_names(settings):
    doc = openapi.schema(urlconf=make_urlconf(), title="T", version="1")
    by_path = {t: item for t, item in doc["paths"].items()}
    assert by_path["/api/items/"]["get"]["operationId"].startswith("item-list")
    assert by_path["/api/items/{item_id}/"]["get"]["operationId"] == "item-detail"
    assert by_path["/plain/"]["get"]["operationId"] == "plain"


def test_path_parameters_are_declared_for_every_templated_route(settings):
    doc = openapi.schema(urlconf=make_urlconf(), title="T", version="1")
    operation = doc["paths"]["/api/items/{item_id}/"]["get"]
    names = {p["name"] for p in operation["parameters"] if p["in"] == "path"}
    assert names == {"item_id"}


def test_the_api_document_covers_every_named_route(api_settings_for_urls):
    from mactf.contrib.api.urls import urlpatterns
    from mactf.http.urls import reverse

    doc = openapi.schema(title="MACTF", version="1")
    ids = {op["operationId"] for _, _, op in _operations(doc)}
    names = {entry.name for entry in urlpatterns if getattr(entry, "name", None)}
    assert names <= ids, sorted(names - ids)
    # document 02 section 5: ``run-list`` is one name for GET and POST on one path, so the only
    # permitted repeat is the same name on the same path under different methods.
    owners: dict[str, set[str]] = {}
    for template, _, op in _operations(doc):
        owners.setdefault(op["operationId"], set()).add(template)
    assert {name for name, paths in owners.items() if len(paths) > 1} == set()


def test_document_shape_is_openapi_3(settings):
    doc = openapi.schema(urlconf=make_urlconf(), title="T", version="1")
    assert doc["openapi"].startswith("3.")
    assert doc["info"]["title"] == "T" and doc["info"]["version"] == "1"
