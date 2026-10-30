"""Criterion 54: ``mactf.testing.Client`` needs no server and leaks no settings."""

from __future__ import annotations

import socket
import sys
import types

import pytest

from mactf import conf
from mactf.http import Conflict, JsonResponse, Response, View, path
from mactf.http.handler import get_wsgi_application
from mactf.http.testing import dispatch_wsgi
from mactf.testing import Client, TestResponse


class Echo(View):
    def get(self, request):
        return JsonResponse({"q": request.GET.get("q"), "ua": request.headers.get("X-Probe")})

    def post(self, request):
        return JsonResponse({"got": request.json()}, status=201)

    def delete(self, request):
        raise Conflict("nope")


def boom(request):
    raise RuntimeError("kaput")


def form(request):
    return Response(request.body, content_type="text/plain")


@pytest.fixture()
def urlconf(settings):
    module = types.ModuleType("testing_client_urls")
    module.urlpatterns = [
        path("echo", Echo.as_view(), name="echo"),
        path("boom", boom, name="boom"),
        path("form", form, name="form"),
    ]
    sys.modules["testing_client_urls"] = module
    conf._clear_http_caches()
    yield "testing_client_urls"
    sys.modules.pop("testing_client_urls", None)
    conf._clear_http_caches()


def test_get_returns_status_headers_and_json(urlconf):
    response = Client(urlconf=urlconf).get("/echo", {"q": "x"}, headers={"X-Probe": "1"})
    assert isinstance(response, TestResponse)
    assert response.status_code == 200
    assert response.json() == {"q": "x", "ua": "1"}
    assert response["Content-Type"].startswith("application/json")
    assert isinstance(response.content, bytes)


def test_post_serializes_data_as_json(urlconf):
    response = Client(urlconf=urlconf).post("/echo", {"a": [1, 2]})
    assert response.status_code == 201
    assert response.json() == {"got": {"a": [1, 2]}}


def test_form_encoded_and_raw_bodies(urlconf):
    client = Client(urlconf=urlconf)
    assert client.post("/form", {"a": "1", "b": "2"}, content_type="application/x-www-form-urlencoded").text == "a=1&b=2"
    assert client.post("/form", b"\x00\x01", content_type="application/octet-stream").content == b"\x00\x01"
    assert client.post("/form", "hello", content_type="text/plain").text == "hello"


def test_http_errors_are_problem_responses(urlconf):
    response = Client(urlconf=urlconf).delete("/echo")
    assert response.status_code == 409
    assert response["Content-Type"].startswith("application/problem+json")


def test_unknown_route_is_404(urlconf):
    assert Client(urlconf=urlconf).get("/missing").status_code == 404


def test_500_is_a_response_by_default_and_raises_on_request(urlconf):
    assert Client(urlconf=urlconf).get("/boom").status_code == 500
    with pytest.raises(RuntimeError, match="kaput"):
        Client(urlconf=urlconf, raise_request_exception=True).get("/boom")
    # a 4xx is expected behaviour and never re-raised
    assert Client(urlconf=urlconf, raise_request_exception=True).delete("/echo").status_code == 409


def test_no_socket_is_opened(urlconf, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the test client opened a socket")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "bind", forbidden)
    assert Client(urlconf=urlconf).get("/echo").status_code == 200


def test_matches_dispatch_wsgi_byte_for_byte(urlconf):
    via_client = Client(urlconf=urlconf).get("/echo", {"q": "same"})
    via_wsgi = dispatch_wsgi(get_wsgi_application(urlconf=urlconf), "GET", "/echo", query={"q": "same"})
    assert via_client.status_code == via_wsgi.status_code
    assert via_client.content == via_wsgi.content
    drop = {"x-request-id", "x-trace-id"}
    first = {k.lower(): v for k, v in via_client.headers.items() if k.lower() not in drop}
    second = {k.lower(): v for k, v in via_wsgi.headers.items() if k.lower() not in drop}
    assert first == second


def test_no_state_is_shared_between_requests(urlconf):
    client = Client(urlconf=urlconf)
    first = client.get("/echo", headers={"Authorization": "Bearer one", "Cookie": "a=b"})
    second = client.get("/echo")
    assert "set-cookie" not in {k.lower() for k in first.headers}
    assert second.json()["ua"] is None


def test_settings_are_swapped_in_and_restored(urlconf, settings):
    before = conf.get_settings()
    custom = conf.load_settings("mactf_test_settings")
    custom["DEBUG"] = True
    seen = {}

    def peek(request):
        seen["debug"] = conf.get_settings()["DEBUG"]
        seen["same"] = conf.get_settings() is custom
        return Response(b"ok")

    module = types.ModuleType("testing_client_peek")
    module.urlpatterns = [path("peek", peek, name="peek")]
    sys.modules["testing_client_peek"] = module
    try:
        # restore ``before`` as current: load_settings above did not install ``custom``
        conf._settings = before
        assert Client(urlconf="testing_client_peek", settings=custom).get("/peek").status_code == 200
    finally:
        sys.modules.pop("testing_client_peek", None)
    assert seen == {"debug": True, "same": True}
    assert conf.get_settings() is before
    assert before["DEBUG"] is False
