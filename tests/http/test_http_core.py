from __future__ import annotations

import pytest
from http_fixtures import make_app, make_urlconf

from mactf.http import NoReverseMatch, Resolver404, get_resolver, resolve, reverse
from mactf.http.testing import dispatch_wsgi


@pytest.fixture()
def app(settings):
    return make_app()


def test_get_and_query(app):
    r = dispatch_wsgi(app, "GET", "/api/items/?q=x")
    assert r.status_code == 200
    assert r.json() == {"items": [], "q": "x"}
    assert r["Content-Type"].startswith("application/json")


def test_post_validates_and_creates(app):
    r = dispatch_wsgi(app, "POST", "/api/items/", json={"name": "a"})
    assert r.status_code == 201
    assert r.json() == {"name": "a", "count": 1}


def test_validation_error_is_problem_json(app):
    r = dispatch_wsgi(app, "POST", "/api/items/", json={"count": -1})
    assert r.status_code == 422 or r.status_code == 400
    assert r.problem is not None
    assert r.problem["status"] == r.status_code
    assert "name" in r.problem["extensions"]["errors"]


def test_bad_json_is_400(app):
    r = dispatch_wsgi(app, "POST", "/api/items/", body=b"{nope", content_type="application/json")
    assert r.status_code == 400
    assert r.problem


def test_unknown_path_is_404_problem(app):
    r = dispatch_wsgi(app, "GET", "/nothing/here")
    assert r.status_code == 404
    assert r.problem["status"] == 404


def test_method_not_allowed_lists_allow(app):
    r = dispatch_wsgi(app, "DELETE", "/api/items/")
    assert r.status_code == 405
    assert "GET" in r["Allow"]


def test_conflict_maps_to_409(app):
    r = dispatch_wsgi(app, "GET", "/api/items/0/")
    assert r.status_code == 409


def test_unhandled_exception_is_500_without_leak(app):
    r = dispatch_wsgi(app, "GET", "/boom/")
    assert r.status_code == 500
    assert "kaput" not in r.text


def test_append_slash_redirects(app):
    r = dispatch_wsgi(app, "GET", "/api/items")
    assert r.status_code == 301
    assert r["Location"].endswith("/api/items/")


def test_request_id_header(app):
    r = dispatch_wsgi(app, "GET", "/plain/", headers={"X-Request-Id": "abc"})
    assert r.status_code == 200
    assert r.text == "hello"
    assert r.get("X-Request-Id")


def test_options_and_head(app):
    r = dispatch_wsgi(app, "OPTIONS", "/api/items/")
    assert r.status_code in (200, 204)
    h = dispatch_wsgi(app, "HEAD", "/plain/")
    assert h.status_code == 200
    assert h.content == b""


def test_sse_framing(app):
    r = dispatch_wsgi(app, "GET", "/stream/")
    assert r.streaming
    events = list(r.iter_sse())
    data = [e for e in events if e["data"] is not None]
    assert [e["data"]["n"] for e in data] == [1, 2]
    assert [e["id"] for e in data] == ["1", "2"]
    assert any(e["comment"] for e in events)


def test_reverse_resolve_roundtrip(settings):
    conf = make_urlconf()
    url = reverse("items:item-detail", item_id=7, urlconf=conf)
    assert url == "/api/items/7/"
    match = resolve(url, urlconf=conf)
    assert match.kwargs == {"item_id": 7}
    assert match.view_name == "items:item-detail"
    with pytest.raises(NoReverseMatch):
        reverse("items:missing", urlconf=conf)
    with pytest.raises(Resolver404):
        resolve("/zzz/", urlconf=conf)


def test_resolver_is_cached_per_urlconf(settings):
    conf = make_urlconf()
    assert get_resolver(conf) is get_resolver(conf)
