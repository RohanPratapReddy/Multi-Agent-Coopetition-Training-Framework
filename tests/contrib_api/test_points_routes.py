"""test points routes (split from test_completed_run.py)."""

from __future__ import annotations

import pytest


from api_support import ApiClient, body, wait_terminal
from mactf.http.handler import WSGIHandler


@pytest.fixture()
def finished(client):
    created = client.post("/runs", json={"epochs": 2, "run_id": "full"})
    assert created.status_code == 202, created.text
    final = wait_terminal(client, "full")
    assert final["status"] == "completed", final
    return "full"


def test_the_scale_has_forty_bindings_that_sum_to_one_hundred(client):
    scale = body(client.get("/scores/scale"))
    assert len(scale["rewards"]) == 17 and len(scale["penalties"]) == 23
    assert len(scale["rewards"]) + len(scale["penalties"]) == 40
    assert sum(scale["rewards"].values()) == 100.0
    assert scale["reward_total"] == 100.0
    assert set(scale["rewards"]) == {f"R{i}" for i in range(1, 18)}
    assert set(scale["penalties"]) == {f"P{i}" for i in range(1, 24)}
    assert all(weight > 0 for weight in {**scale["rewards"], **scale["penalties"]}.values())
    assert set(scale["severity_fractions"]) == {"S0", "S1", "S2", "S3", "S4"}


def test_the_scale_route_is_the_engine_table_not_a_copy(client):
    from mactf.scoring.points import DEFAULT_POINTS_TABLE

    scale = body(client.get("/scores/scale"))
    assert scale["rewards"] == dict(DEFAULT_POINTS_TABLE.rewards)
    assert scale["penalties"] == dict(DEFAULT_POINTS_TABLE.penalties)
    from mactf.scoring.points import TABLE_VERSION

    assert scale["table_version"] == TABLE_VERSION


def test_a_run_without_the_points_signal_has_none_and_says_why(client, finished):
    page = body(client.get(f"/runs/{finished}/points"))
    assert page["total"] == 0 and page["items"] == []
    refused = client.get(f"/runs/{finished}/episodes/0/normalization")
    assert refused.status_code == 409
    assert refused.problem["extensions"]["reason"] == "normalization disabled"


def test_a_points_run_stores_points_for_every_peer_and_episode(points_client):
    page = body(points_client.get("/runs/pts/points"))
    assert page["total"] == 4
    assert {(row["episode"], row["peer"]) for row in page["items"]} == {
        (0, "p0"), (0, "p1"), (1, "p0"), (1, "p1"),
    }
    for row in page["items"]:
        assert 0.0 <= row["points"]["final_points"] <= 100.0
        assert row["points"]["table_version"] == "1.0.0"


def test_points_filters_and_paging(points_client):
    one = body(points_client.get("/runs/pts/points", query={"peer": "p0", "episode": 1}))
    assert one["total"] == 1 and one["items"][0]["peer"] == "p0"
    page = body(points_client.get("/runs/pts/points", query={"limit": 2, "offset": 0}))
    assert len(page["items"]) == 2 and page["has_more"] is True
    assert points_client.get("/runs/pts/points", query={"limit": 0}).status_code == 422
    assert points_client.get("/runs/missing/points").status_code == 404


def test_a_guard_denial_is_recorded_not_dropped(points_client):
    row = body(points_client.get("/runs/pts/points", query={"peer": "p0", "episode": 0}))["items"][0]
    denied = row["points"]["denied"]
    assert denied and denied[0]["rule_id"] == "R2"
    assert denied[0]["missing"] == ["claims"]


def test_normalization_report_is_served_and_consistent(points_client):
    report = body(points_client.get("/runs/pts/episodes/0/normalization"))
    assert report["preserves_order"] is True
    assert {entry["peer"] for entry in report["scores"]} == {"p0", "p1"}
    assert report["policy"]["target_mean"] == 62.0
    assert points_client.get("/runs/pts/episodes/99/normalization").status_code == 404


def test_the_v2_scorecard_carries_points_and_matches_the_points_route(points_client):
    card = body(
        points_client.get(
            "/runs/pts/episodes/0/peers/p0",
            headers={"Accept": "application/vnd.mactf.scorecard.v2+json"},
        )
    )
    listed = body(points_client.get("/runs/pts/points", query={"peer": "p0", "episode": 0}))["items"][0]
    assert card["points"] == listed["points"]
    assert card["final_score"] == listed["final_score"]
