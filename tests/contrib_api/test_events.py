"""test events (split from test_runs.py)."""

from __future__ import annotations

import time


from api_support import body, wait_terminal




def test_events_stream_terminates_with_the_run(client):
    client.post("/runs", json={"dry_run": True, "run_id": "ev1"})
    with client.get("/runs/ev1/events") as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.get("Content-Type")
        events = list(response.iter_sse())
    kinds = [event.get("event") for event in events]
    assert kinds, "no events"
    assert kinds[-1] == "terminal"


def test_events_for_finished_run_replay_then_close(client):
    client.post("/runs", json={"dry_run": True, "run_id": "ev2"})
    wait_terminal(client, "ev2")
    with client.get("/runs/ev2/events") as response:
        assert response.status_code == 200
        assert list(response.iter_sse())


def test_replay_is_gapless_and_resumable_from_any_id(client):
    created = client.post("/runs", json={"epochs": 2, "run_id": "gapless"})
    assert created.status_code == 202, created.text
    wait_terminal(client, "gapless")
    with client.get("/runs/gapless/events") as response:
        events = list(response.iter_sse())
    ids = [int(event["id"]) for event in events]
    assert ids == list(range(ids[0], ids[0] + len(ids))), "event ids are not contiguous"
    assert [event["event"] for event in events].count("epoch") == 2
    assert events[-1]["event"] == "terminal"
    middle = ids[len(ids) // 2]
    with client.get("/runs/gapless/events", headers={"Last-Event-ID": str(middle)}) as response:
        tail = list(response.iter_sse())
    assert [int(event["id"]) for event in tail] == [i for i in ids if i > middle]
    assert [event["event"] for event in tail] == [
        event["event"] for event in events if int(event["id"]) > middle
    ]


def test_resuming_after_the_last_event_yields_nothing_but_a_clean_close(client):
    client.post("/runs", json={"epochs": 1, "run_id": "tail"})
    wait_terminal(client, "tail")
    with client.get("/runs/tail/events") as response:
        last = int(list(response.iter_sse())[-1]["id"])
    with client.get("/runs/tail/events", headers={"Last-Event-ID": str(last)}) as response:
        assert response.status_code == 200
        assert [e for e in response.iter_sse() if e["event"] != "message"] == []


def test_events_of_an_unknown_run_are_a_404_problem(client):
    response = client.get("/runs/never-existed/events")
    assert response.status_code == 404
    assert response.problem["type"] == "urn:mactf:error:not-found"
