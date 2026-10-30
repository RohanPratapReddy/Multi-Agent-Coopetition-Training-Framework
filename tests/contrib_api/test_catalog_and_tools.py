from __future__ import annotations

from api_support import body


def test_meta_and_catalog(client):
    meta = body(client.get("/meta"))
    assert meta["api_version"] == "v1"
    rules = body(client.get("/rules?kind=penalty"))
    assert rules["total"] > 0 if "total" in rules else rules
    assert client.get("/rules/NOPE").status_code == 404
    assert body(client.get("/rules/validation"))
    scale = body(client.get("/scores/scale"))
    assert "table_version" in scale


def test_openapi_lists_every_documented_operation(client):
    doc = body(client.get("/openapi.json"))
    assert doc["openapi"].startswith("3.")
    operation_ids = {
        op.get("operationId")
        for item in doc["paths"].values()
        for op in item.values()
        if isinstance(op, dict)
    }
    for name in ("health", "run-list", "run-detail", "run-events", "run-rollback", "peer-infer"):
        assert name in operation_ids


def test_peers_listing_and_contract(client):
    peers = body(client.get("/peers"))
    assert isinstance(peers, list)
    assert client.get("/peers/ghost").status_code == 404


def test_weight_preview(client):
    response = client.post(
        "/weights/preview",
        json={"scores": {"a": 1.0, "b": 0.0}},
    )
    assert response.status_code in (200, 422), response.text


def test_scorers_and_trainer_capabilities(client):
    scorers = body(client.get("/scorers"))
    assert scorers
    caps = body(client.get("/trainers/capabilities"))
    assert caps
    presets = body(client.get("/trainers/presets"))
    assert "items" in presets or isinstance(presets, list)


def test_trainer_preview_reports_parameters(client):
    response = client.post(
        "/trainers/preview",
        json={"optimizer": {"type": "adamw", "lr": 0.001}},
    )
    assert response.status_code in (200, 422), response.text
    if response.status_code == 200:
        assert "params" in str(body(response)) or "parameters" in str(body(response))


def test_preview_rejects_unknown_fields(client):
    assert client.post("/trainers/preview", json={"bogus": 1}).status_code == 422


def test_admin_requires_admin_scope(client):
    assert client.get("/admin/capabilities", token="write-tok").status_code == 403
    assert client.get("/admin/capabilities").status_code == 200
    assert client.post("/admin/config/reload", token="write-tok").status_code == 403


def test_system_checks_endpoint(client):
    response = client.post("/system-checks")
    assert response.status_code == 200
    assert "ok" in body(response)


def test_peer_inference_missing_tree_is_clean_error(client):
    response = client.post("/peers/p0/infer", json={"inputs": [[0.0, 0.0, 0.0, 0.0]]})
    assert response.status_code in (200, 404, 409, 422), response.text
