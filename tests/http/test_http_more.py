from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import urllib.request
from wsgiref.simple_server import WSGIRequestHandler, make_server

import pytest
from http_fixtures import ItemIn, make_app, make_urlconf

from mactf.http import (
    BoolField,
    CharField,
    ChoiceField,
    IntField,
    JsonResponse,
    ListField,
    Principal,
    QueryDict,
    Response,
    Serializer,
    ValidationError,
    openapi,
)
from mactf.http.testing import dispatch_wsgi, make_environ, parse_sse


# -- request ---------------------------------------------------------------
def test_query_dict_multi_values():
    q = QueryDict("a=1&a=2&b=x")
    assert q["a"] == "2"
    assert q.getlist("a") == ["1", "2"]
    with pytest.raises(Exception):
        q["c"] = "no"  # immutable


def test_environ_builder_encodes_json_and_query():
    env = make_environ("POST", "/x?a=1", query={"b": [1, 2]}, json={"k": "v"})
    assert env["CONTENT_TYPE"] == "application/json"
    assert env["QUERY_STRING"] == "a=1&b=1&b=2"
    assert env["wsgi.input"].read() == b'{"k":"v"}'


def test_oversized_body_is_413(settings):
    from mactf.http.request import max_body_bytes

    app = make_app()
    big = b"x" * (max_body_bytes() + 1)
    r = dispatch_wsgi(app, "POST", "/api/items/", body=big, content_type="application/json")
    assert r.status_code == 413


def test_bearer_token_parsing(settings):
    from mactf.http.request import Request

    req = Request(make_environ("GET", "/", headers={"Authorization": "Bearer abc.def"}))
    assert req.get_bearer_token() == "abc.def"


# -- response ---------------------------------------------------------------
def test_json_response_and_cookies():
    r = JsonResponse({"a": 1}, status=202)
    assert r.status_code == 202 and r.json() == {"a": 1}
    r.set_cookie("sid", "v", httponly=True, max_age=5)
    cookies = [v for k, v in r.wsgi_headers() if k.lower() == "set-cookie"]
    assert cookies and "HttpOnly" in cookies[0]


def test_response_rejects_str_body():
    with pytest.raises(TypeError):
        Response("text")


def test_response_header_case_insensitive():
    r = Response(b"x")
    r.headers["X-A"] = "1"
    assert r.get("x-a") == "1"


# -- serializers -------------------------------------------------------------
class Opts(Serializer):
    flag = BoolField(default=False)
    kind = ChoiceField(choices=["a", "b"])
    tags = ListField(child=CharField(), required=False)


def test_serializer_valid_and_defaults():
    s = ItemIn(data={"name": "x"})
    assert s.is_valid()
    assert s.validated_data == {"name": "x", "count": 1}


def test_serializer_errors_by_field():
    s = ItemIn(data={"name": "x" * 50, "count": "nan"})
    assert not s.is_valid()
    assert set(s.errors) == {"name", "count"}
    with pytest.raises(ValidationError):
        s.is_valid(raise_exception=True)


def test_serializer_choice_bool_list():
    s = Opts(data={"kind": "c"})
    assert not s.is_valid() and "kind" in s.errors
    ok = Opts(data={"kind": "a", "flag": True, "tags": ["p", "q"]})
    assert ok.is_valid() and ok.validated_data["tags"] == ["p", "q"]


def test_serializer_schema_lists_required():
    sch = ItemIn.schema()
    assert sch["type"] == "object"
    assert "name" in sch["required"] and "count" not in sch["required"]


def test_annotated_serializer():
    class Annotated(Serializer):
        title: str
        n: int = 3

    s = Annotated(data={"title": "t"})
    assert s.is_valid() and s.validated_data == {"title": "t", "n": 3}
    assert not Annotated(data={"n": 1}).is_valid()


# -- principals and middleware ---------------------------------------------
def test_principal_scopes():
    p = Principal.authenticated("u", ["runs:*", "events:read"])
    assert p.has_scope("runs:write") and p.has_scope("events:read")
    assert not p.has_scope("admin")
    assert Principal.authenticated("r", ["*"]).has_scope("anything:x")


def test_cors_preflight(settings):
    settings["MIDDLEWARE"] = list(settings.get("MIDDLEWARE") or []) + []
    app = make_app()
    r = dispatch_wsgi(
        app,
        "OPTIONS",
        "/api/items/",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"},
    )
    assert "Access-Control-Allow-Origin" not in r.headers.to_dict() or r.get(
        "Access-Control-Allow-Origin"
    ) != "http://evil.example"


# -- openapi ------------------------------------------------------------------
def test_openapi_document(settings):
    doc = openapi.schema(urlconf=make_urlconf(), title="T", version="1")
    assert doc["openapi"].startswith("3.")
    assert "/api/items/" in doc["paths"]
    assert "/api/items/{item_id}/" in doc["paths"]
    ops = doc["paths"]["/api/items/"]
    assert set(ops) >= {"get", "post"}
    assert ops["post"]["operationId"]
    ids = [op["operationId"] for p in doc["paths"].values() for op in p.values()]
    assert len(ids) == len(set(ids))


# -- a real socket (the dev server path) --------------------------------------------
class _Quiet(WSGIRequestHandler):
    def log_message(self, *args):  # noqa: D401
        pass


def test_streaming_over_a_real_socket(settings):
    app = make_app()
    server = make_server("127.0.0.1", 0, app, handler_class=_Quiet)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/stream/", timeout=5) as resp:
            assert resp.status == 200
            body = resp.read()
        events = [e for e in parse_sse([body]) if e["data"] is not None]
        assert [e["data"]["n"] for e in events] == [1, 2]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/items/", timeout=5) as resp:
            assert resp.status == 200
    finally:
        server.shutdown()
        server.server_close()


# -- layering ------------------------------------------------------------------------
def _run(code: str) -> str:
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_http_does_not_import_the_engine():
    out = _run(
        """
        import sys
        import mactf.http
        banned = ["mactf.orchestration", "mactf.training", "mactf.scoring", "mactf.coopetition",
                  "mactf.rules", "mactf.sandbox", "mactf.contrib", "mactf.client"]
        print([m for m in sys.modules if any(m == b or m.startswith(b + ".") for b in banned)])
        """
    )
    assert out == "[]"


def test_http_is_stdlib_only():
    out = _run(
        """
        import sys
        import mactf.http
        banned = {"fastapi", "starlette", "pydantic", "httpx", "uvicorn", "flask", "django"}
        print(sorted(banned & {m.split(".")[0] for m in sys.modules}))
        """
    )
    assert out == "[]"
