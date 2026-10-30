"""Scorecard version negotiation: v1 by default, v2 with mandatory points (document 02 section 4.3.1)."""

from __future__ import annotations

from api_support import body

V2 = {"Accept": "application/vnd.mactf.v2+json"}
V1 = {"Accept": "application/vnd.mactf.v1+json"}
CARD = "/runs/pts/episodes/0/peers/p0"


def test_default_is_v1_and_points_is_present_but_may_be_null(client, finished):
    card = body(client.get(f"/runs/{finished}/episodes/0/peers/p0"))
    assert "final_score" in card and card["points"] is None
    assert "guards_denied" not in card and "normalization" not in card


def test_v2_on_a_points_run_has_mandatory_points_and_normalization(points_client):
    response = points_client.get(CARD, headers=V2)
    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("application/vnd.mactf.v2+json")
    card = response.json()
    assert card["points"] is not None and "final_points" in card["points"]
    assert "normalization" in card and isinstance(card["guards_denied"], list)


def test_v1_on_a_points_run_keeps_the_docx_fields_and_includes_points(points_client):
    v1 = body(points_client.get(CARD, headers=V1))
    v2 = body(points_client.get(CARD, headers=V2))
    for key, value in v1.items():
        assert v2[key] == value, key
    assert v1["final_score"] == v2["final_score"]


def test_v2_on_a_run_without_points_is_a_409_not_a_silent_v1(client, finished):
    response = client.get(f"/runs/{finished}/episodes/0/peers/p0", headers=V2)
    assert response.status_code == 409
    assert "version 1" in response.problem["detail"]


def test_an_unsupported_version_is_a_406_listing_the_supported_ones(client, finished):
    response = client.get(
        f"/runs/{finished}/episodes/0/peers/p0", headers={"Accept": "application/vnd.mactf.v9+json"}
    )
    assert response.status_code == 406
    assert response.problem["extensions"]["supported"] == ["1", "2"]


def test_the_query_parameter_selects_a_version_and_the_header_wins(points_client):
    by_query = points_client.get(CARD, query={"scorecard_version": "2"})
    assert by_query.status_code == 200 and "guards_denied" in by_query.json()
    both = points_client.get(CARD + "?scorecard_version=1", headers=V2)
    assert both.status_code == 200 and "guards_denied" in both.json()
    assert both.headers["Warning"].startswith("299")


def test_episode_lists_project_every_peer(points_client):
    page = body(points_client.get("/runs/pts/episodes", headers=V2))
    for episode in page["items"]:
        assert all(card["points"] is not None for card in episode["scorecard"]["peers"].values())


def test_the_v2_projection_does_not_mutate_the_stored_card(points_client):
    before = body(points_client.get(CARD, headers=V1))
    points_client.get(CARD, headers=V2)
    assert body(points_client.get(CARD, headers=V1)) == before
