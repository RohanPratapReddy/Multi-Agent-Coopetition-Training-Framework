from __future__ import annotations

from api_support import body, build_settings


from mactf.http.handler import WSGIHandler


def test_missing_scope_is_403_and_anonymous_is_401(client):
    denied = client.post("/evaluations", json={"peer": "a", "events": []}, token="read-tok")
    assert denied.status_code == 403
    assert body(denied)["extensions"]["required_scope"] == "score:write"
    anonymous = client.post("/evaluations", json={"peer": "a", "events": []}, token=None)
    assert anonymous.status_code == 401
    assert "Bearer" in anonymous.get("WWW-Authenticate")


def test_bad_token_is_401_even_on_a_read_route(client):
    assert client.get("/meta", token="nope").status_code == 401


def test_anonymous_gets_read_only(client):
    assert client.get("/rules", token=None).status_code == 200
    assert client.post("/runs", json={"dry_run": True}, token=None).status_code == 401


def test_anonymous_disabled(settings):
    build_settings(settings, ALLOW_ANONYMOUS=False)
    app = WSGIHandler()
    from api_support import ApiClient

    anon = ApiClient(app)
    anon.token = None
    assert anon.get("/rules").status_code == 401
    assert anon.get("/health").status_code == 200


def test_responses_carry_version_headers(client):
    response = client.get("/rules")
    assert response.get("X-API-Version") == "v1"
    assert response.get("X-MACTF-Version")
    assert response.get("X-Request-ID")
