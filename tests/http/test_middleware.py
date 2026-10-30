"""Criterion 42: a view exception is a 500 problem with a trace id, and no traceback unless DEBUG."""

from __future__ import annotations

import json

from http_fixtures import make_app

from mactf import conf
from mactf.http.testing import dispatch_wsgi


def test_a_500_is_a_problem_with_a_trace_id(settings):
    response = dispatch_wsgi(make_app(), "GET", "/boom/")
    assert response.status_code == 500
    assert response["Content-Type"].startswith("application/problem+json")
    body = response.json()
    assert body["status"] == 500 and body["type"].startswith("urn:mactf:error")
    assert body["trace_id"] and response["X-Trace-Id"] == body["trace_id"]


def test_no_traceback_or_exception_text_without_debug(settings):
    settings["DEBUG"] = False
    conf._settings = settings
    response = dispatch_wsgi(make_app(), "GET", "/boom/")
    text = response.text
    assert "Traceback" not in text and "kaput" not in text and "RuntimeError" not in text
    assert "File \"" not in text


def test_each_failure_gets_its_own_trace_id(settings):
    app = make_app()
    first = dispatch_wsgi(app, "GET", "/boom/").json()["trace_id"]
    second = dispatch_wsgi(app, "GET", "/boom/").json()["trace_id"]
    assert first != second and len(first) >= 16

def test_debug_adds_the_detail_but_keeps_the_problem_shape(settings):
    settings["DEBUG"] = True
    conf._settings = settings
    response = dispatch_wsgi(make_app(), "GET", "/boom/")
    body = json.loads(response.text)
    assert response.status_code == 500 and body["trace_id"]
    assert "kaput" in response.text


def test_the_server_keeps_serving_after_a_500(settings):
    app = make_app()
    assert dispatch_wsgi(app, "GET", "/boom/").status_code == 500
    assert dispatch_wsgi(app, "GET", "/plain/").text == "hello"


def test_domain_errors_are_not_500s(settings):
    response = dispatch_wsgi(make_app(), "GET", "/api/items/0/")
    assert response.status_code == 409
    assert response.json()["trace_id"]
