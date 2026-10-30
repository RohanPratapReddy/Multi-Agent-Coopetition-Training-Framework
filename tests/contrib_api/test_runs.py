from __future__ import annotations

import time


from api_support import body, wait_terminal




def test_dry_run_lifecycle(client):
    created = client.post("/runs", json={"dry_run": True, "epochs": 1, "run_id": "dry1"})
    assert created.status_code == 202, created.text
    assert created.get("Location").endswith("/runs/dry1")
    assert body(created)["links"]["events"].endswith("/runs/dry1/events")
    final = wait_terminal(client, "dry1")
    assert final["status"] in ("dry_run", "completed"), final
    listing = body(client.get("/runs"))
    assert listing["total"] >= 1 and {"items", "limit", "offset", "has_more"} <= set(listing)
    assert client.get("/runs/nope").status_code == 404


def test_duplicate_run_id_is_409(client):
    assert client.post("/runs", json={"dry_run": True, "run_id": "dup"}).status_code == 202
    wait_terminal(client, "dup")
    assert client.post("/runs", json={"dry_run": True, "run_id": "dup"}).status_code == 409


def test_idempotency_key_replays_same_response(client):
    headers = {"Idempotency-Key": "k-1"}
    first = client.post("/runs", json={"dry_run": True, "run_id": "idem"}, headers=headers)
    second = client.post("/runs", json={"dry_run": True, "run_id": "idem"}, headers=headers)
    assert first.status_code == second.status_code == 202
    assert body(first) == body(second)
    conflicting = client.post("/runs", json={"dry_run": True, "run_id": "other"}, headers=headers)
    assert conflicting.status_code in (409, 422)


def test_run_requires_write_scope(client):
    assert client.post("/runs", json={"dry_run": True}, token="read-tok").status_code == 403
    assert client.post("/runs", json={"dry_run": True}, token="write-tok").status_code == 202


def test_invalid_run_body_is_422(client):
    assert client.post("/runs", json={"epochs": 0}).status_code == 422
    assert client.post("/runs", json={"bogus": 1}).status_code == 422


def test_cancel_unknown_and_terminal(client):
    assert client.post("/runs/ghost/cancel").status_code == 404
    client.post("/runs", json={"dry_run": True, "run_id": "c1"})
    wait_terminal(client, "c1")
    response = client.post("/runs/c1/cancel")
    assert response.status_code in (200, 202, 409)


def test_summary_and_episode_lists_exist(client):
    client.post("/runs", json={"dry_run": True, "run_id": "s1"})
    wait_terminal(client, "s1")
    assert client.get("/runs/s1/summary").status_code == 200
    assert client.get("/runs/s1/episodes").status_code == 200
    assert client.get("/runs/s1/rule-counts").status_code == 200
    assert client.get("/runs/s1/rule-events").status_code == 200
    assert client.get("/runs/s1/aggregations").status_code == 200
    assert client.get("/runs/s1/warnings").status_code == 200
    assert client.get("/runs/s1/episodes/9").status_code == 404
