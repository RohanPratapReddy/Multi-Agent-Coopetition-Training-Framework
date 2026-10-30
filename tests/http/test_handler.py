"""Criterion 41: the application is a plain WSGI callable that ``wsgiref`` and the test client both drive."""

from __future__ import annotations

import io
import json
import threading
import urllib.request
from wsgiref.simple_server import WSGIRequestHandler, make_server
from wsgiref.validate import validator

from http_fixtures import make_app

from mactf.http.testing import dispatch_wsgi, make_environ


class _Quiet(WSGIRequestHandler):
    def log_message(self, *args):
        pass


def _call(app, environ):
    captured = {}

    def start_response(status, headers, exc_info=None):
        captured["status"], captured["headers"] = status, headers

    result = app(environ, start_response)
    try:
        body = b"".join(result)
    finally:
        close = getattr(result, "close", None)
        if close:
            close()
    return captured["status"], dict(captured["headers"]), body


def test_the_app_satisfies_the_wsgi_validator(settings):
    app = validator(make_app())
    status, headers, body = _call(app, make_environ("GET", "/api/items/?q=v"))
    assert status.startswith("200")
    assert json.loads(body) == {"items": [], "q": "v"}


def test_post_through_the_validator(settings):
    app = validator(make_app())
    environ = make_environ("POST", "/api/items/", json={"name": "k", "count": 4})
    status, _, body = _call(app, environ)
    assert status.startswith("201") and json.loads(body) == {"name": "k", "count": 4}


def test_wsgiref_server_serves_the_same_bytes_as_the_test_client(settings):
    app = make_app()
    server = make_server("127.0.0.1", 0, app, handler_class=_Quiet)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/items/?q=z", timeout=5) as resp:
            wire = resp.read()
            assert resp.status == 200
        assert wire == dispatch_wsgi(app, "GET", "/api/items/?q=z").content
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/items/",
            data=json.dumps({"name": "n"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as resp:
            assert resp.status == 201 and json.loads(resp.read()) == {"name": "n", "count": 1}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_the_cached_application_is_callable_unmodified(settings):
    from mactf.http.handler import clear_application_cache, get_wsgi_application

    clear_application_cache()
    app = get_wsgi_application()
    assert callable(app) and get_wsgi_application() is app
    status, _, _ = _call(app, make_environ("GET", "/definitely/missing/"))
    assert status.startswith("404")
    clear_application_cache()


def test_the_app_reads_the_request_body_from_wsgi_input(settings):
    app = make_app()
    payload = json.dumps({"name": "raw"}).encode()
    environ = make_environ("POST", "/api/items/")
    environ.update(
        {"wsgi.input": io.BytesIO(payload), "CONTENT_LENGTH": str(len(payload)), "CONTENT_TYPE": "application/json"}
    )
    status, _, body = _call(app, environ)
    assert status.startswith("201") and json.loads(body)["name"] == "raw"
