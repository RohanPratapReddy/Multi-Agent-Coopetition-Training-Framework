from __future__ import annotations

import pytest


from api_support import body


from api_support import wait_terminal


@pytest.fixture()
def finished(client):
    created = client.post("/runs", json={"epochs": 2, "run_id": "full"})
    assert created.status_code == 202, created.text
    final = wait_terminal(client, "full")
    assert final["status"] == "completed", final
    return "full"


def test_episodes_and_scorecards(client, finished):
    page = body(client.get(f"/runs/{finished}/episodes"))
    assert page["total"] == 2
    detail = client.get(f"/runs/{finished}/episodes/0")
    assert detail.status_code == 200, detail.text
    peer = client.get(f"/runs/{finished}/episodes/0/peers/p0")
    assert peer.status_code == 200, peer.text
    assert client.get(f"/runs/{finished}/episodes/0/peers/zzz").status_code == 404


def test_scorecard_version_negotiation(client, finished):
    v1 = client.get(f"/runs/{finished}/episodes/0/peers/p0", headers={"Accept": "application/vnd.mactf.v1+json"})
    v2 = client.get(f"/runs/{finished}/episodes/0/peers/p0", headers={"Accept": "application/vnd.mactf.v2+json"})
    assert v1.status_code == 200
    # Points are opt-in; a run scored without them has no v2 form and says so.
    assert v2.status_code in (200, 409)
    if v2.status_code == 409:
        assert "version 1" in body(v2)["detail"]
    else:
        assert "points" in body(v2)
    bad = client.get(f"/runs/{finished}/episodes/0/peers/p0", headers={"Accept": "application/vnd.mactf.v9+json"})
    assert bad.status_code == 406


def test_checkpoints_list_detail_verify(client, finished):
    page = body(client.get(f"/runs/{finished}/checkpoints"))
    assert page["total"] >= 1
    episode = page["items"][0]["episode"]
    detail = client.get(f"/runs/{finished}/checkpoints/{episode}")
    assert detail.status_code == 200
    assert body(detail)["verified"] is True
    verify = body(client.post(f"/runs/{finished}/checkpoints/{episode}/verify"))
    assert verify["ok"] is True and verify["content_hash"]
    assert client.get(f"/runs/{finished}/checkpoints/99").status_code == 404


def test_rollback_archives_newer_checkpoints(client, finished):
    rows = body(client.get(f"/runs/{finished}/checkpoints"))["items"]
    episodes = sorted(row["episode"] for row in rows)
    assert len(episodes) >= 2
    target = episodes[0]
    response = client.post(f"/runs/{finished}/rollback", json={"episode": target})
    assert response.status_code == 202, response.text
    remaining = [row["episode"] for row in body(client.get(f"/runs/{finished}/checkpoints"))["items"]]
    assert max(remaining) == target
    assert client.post(f"/runs/{finished}/rollback", json={"episode": 99}).status_code == 404


def test_logs_and_watchdog_for_finished_run_are_not_errors(client, finished):
    logs = client.get(f"/runs/{finished}/logs/tail?n=5")
    assert logs.status_code == 200 and len(body(logs)["items"]) <= 5
    assert client.get(f"/runs/{finished}/logs/tail?n=0").status_code == 422
    assert client.get(f"/runs/{finished}/logs/tail?source=file").status_code == 200
    assert client.get(f"/runs/{finished}/logs/tail?source=bogus").status_code == 422
    dog = body(client.get(f"/runs/{finished}/watchdog"))
    assert dog["history"] == []
    assert client.post(f"/runs/{finished}/watchdog/snapshot").status_code == 409


def test_peer_tree_of_run(client, finished):
    tree = client.get(f"/runs/{finished}/peers/tree")
    assert tree.status_code == 200, tree.text
    node = client.get(f"/runs/{finished}/peers/p0")
    assert node.status_code == 200, node.text


def test_evaluation_ledger_replacement_and_adjudication(client, finished):
    ledger = {
        "peer": "p0",
        "episode": 0,
        "events": [{"rule_id": "P1", "evidence": "tampered"}],
    }
    put = client.request("PUT", f"/runs/{finished}/episodes/0/ledger", json=ledger)
    assert put.status_code in (200, 405), put.text
    post = client.post(f"/runs/{finished}/episodes/0/ledger", json=ledger)
    assert post.status_code == 200, post.text
    events = body(client.get(f"/runs/{finished}/rule-events?peer=p0"))
    assert events["total"] >= 1
    adjudication = client.post(
        f"/runs/{finished}/adjudications",
        json={
            "peer": "p0", "rule_id": "P1", "episode": 0, "decision": "reversed",
            "rationale": "false positive", "adjudicator": "ref-1",
        },
    )
    assert adjudication.status_code in (200, 201), adjudication.text
    listing = body(client.get(f"/runs/{finished}/adjudications"))
    assert listing["total"] == 1
    assert client.get(f"/runs/{finished}/flags").status_code == 200
