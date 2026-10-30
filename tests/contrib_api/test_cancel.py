"""test cancel (split from test_catalog_and_tools.py)."""

from __future__ import annotations

import time


from api_support import body, wait_terminal


def test_cancel_a_running_run(client, api_settings):
    api_settings["TRAINING"]["EPOCHS"] = 50
    created = client.post("/runs", json={"epochs": 50, "run_id": "long"})
    assert created.status_code == 202
    time.sleep(0.2)
    response = client.post("/runs/long/cancel")
    assert response.status_code in (200, 202), response.text
    final = wait_terminal(client, "long")
    assert final["status"] == "cancelled"
    assert client.post("/runs/long/cancel").status_code == 409


def test_second_concurrent_run_is_429_with_retry_after(client, api_settings):
    api_settings["TRAINING"]["EPOCHS"] = 50
    assert client.post("/runs", json={"epochs": 50, "run_id": "first"}).status_code == 202
    second = client.post("/runs", json={"epochs": 1, "run_id": "second"})
    assert second.status_code == 429
    assert second.get("Retry-After")
    client.post("/runs/first/cancel")
    wait_terminal(client, "first")


def test_cancel_stops_at_an_epoch_boundary_with_no_partial_episode(client, api_settings):
    api_settings["TRAINING"]["EPOCHS"] = 200
    assert client.post("/runs", json={"epochs": 200, "run_id": "boundary"}).status_code == 202
    time.sleep(0.3)
    assert client.post("/runs/boundary/cancel").status_code in (200, 202)
    final = wait_terminal(client, "boundary")
    assert final["status"] == "cancelled"

    episodes = body(client.get("/runs/boundary/episodes"))
    with client.get("/runs/boundary/events") as response:
        events = list(response.iter_sse())
    epoch_events = [event for event in events if event["event"] == "epoch"]
    assert 0 < episodes["total"] < 200
    # every stored episode has its epoch event and nothing else was left half-written
    assert len(epoch_events) == episodes["total"]
    assert events[-1]["event"] == "terminal"
    assert events[-1]["data"]["status"] == "cancelled"
    numbers = sorted(row["episode"] for row in episodes["items"])
    assert numbers == list(range(len(numbers)))
    for row in body(client.get("/runs/boundary/checkpoints"))["items"]:
        verdict = body(client.post(f"/runs/boundary/checkpoints/{row['episode']}/verify"))
        assert verdict.get("valid", verdict.get("ok", True)) is not False


def test_cancel_needs_the_write_scope(client, api_settings):
    api_settings["TRAINING"]["EPOCHS"] = 200
    client.post("/runs", json={"epochs": 200, "run_id": "scoped"})
    assert client.post("/runs/scoped/cancel", token="read-tok").status_code == 403
    assert client.post("/runs/scoped/cancel", token=None).status_code == 401
    client.post("/runs/scoped/cancel")
    wait_terminal(client, "scoped")
