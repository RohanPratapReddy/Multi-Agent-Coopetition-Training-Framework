"""Criteria 37/38: points, the scale and normalization read identically over both transports."""

from __future__ import annotations

import pytest

from mactf.client import Conflict, HTTPTransport, InProcessTransport, MACTFClient, ValidationError
from mactf.scoring.normalize import NormalizationReport
from mactf.scoring.points import PointsBreakdown, PointsTable


@pytest.fixture()
def points_settings(api_settings):
    api_settings["COOPETITION"]["SIGNAL_SOURCE"] = "points"
    return api_settings


@pytest.fixture()
def pair(points_settings, server):
    """One app, one points run, two transports pointed at it."""
    over_http = MACTFClient(transport=HTTPTransport(server, token="all-tok", timeout=15, backoff=0.01))
    in_process = MACTFClient(transport=InProcessTransport(token="all-tok"))
    run = over_http.run_to_completion(run_id="pp", epochs=2, timeout=120)
    assert run.status == "completed", run
    return in_process, over_http


def test_scale_is_a_points_table_and_equal_over_both(pair):
    scales = [client.scores.scale() for client in pair]
    assert all(isinstance(scale, PointsTable) for scale in scales)
    assert scales[0].to_dict() == scales[1].to_dict()
    assert len(scales[0].rewards) + len(scales[0].penalties) == 40
    assert sum(scales[0].rewards.values()) == 100.0


def test_points_pages_are_equal_and_typed(pair):
    pages = [client.scores.points("pp") for client in pair]
    assert pages[0].total == pages[1].total == 4
    for entry in pages[0]:
        assert isinstance(entry.points, PointsBreakdown)
    left = [(e.episode, e.peer, e.final_score, e.points.to_dict()) for e in pages[0]]
    right = [(e.episode, e.peer, e.final_score, e.points.to_dict()) for e in pages[1]]
    assert left == right


def test_points_filters_agree(pair):
    for client in pair:
        one = client.scores.points("pp", peer="p1", episode=0)
        assert one.total == 1 and one[0].peer == "p1" and one[0].episode == 0


def test_normalization_is_typed_and_equal(pair):
    reports = [client.scores.normalization("pp", 0) for client in pair]
    assert all(isinstance(report, NormalizationReport) for report in reports)
    assert reports[0].as_dict() == reports[1].as_dict()
    assert reports[0].preserves_order is True


def test_v2_scorecard_matches_the_points_row(pair):
    for client in pair:
        card = client.scores.scorecard("pp", 0, "p0")
        row = client.scores.points("pp", peer="p0", episode=0)[0]
        assert card.points == row.points.to_dict()


def test_the_points_wire_round_trips_through_the_framework_classes(pair):
    row = pair[0].scores.points("pp", peer="p0", episode=0)[0]
    assert PointsBreakdown.from_dict(row.points.to_dict()).to_dict() == row.points.to_dict()


def test_a_void_ledger_with_a_reward_voids_over_both_transports(points_settings, server):
    """Criterion 38: a void-triggering penalty beside a reward still voids; void points are zero."""
    from mactf.scoring.points import finalize_points

    for client in (
        MACTFClient(transport=InProcessTransport(token="all-tok")),
        MACTFClient(transport=HTTPTransport(server, token="all-tok", timeout=15, backoff=0.01)),
    ):
        card = client.score(
            "a",
            [("R1", "answered correctly", 1.0), ("P1", "fabricated a citation", 1.0)],
        )
        assert card.is_void and "P1" in card.violated_non_offsettable
        assert card.final_score < 0
    assert finalize_points(37.5, void=True) == 0.0
    assert finalize_points(37.5, void=False) == 37.5

def test_a_run_without_points_refuses_v2_and_normalization(api_settings, server):
    client = MACTFClient(transport=HTTPTransport(server, token="all-tok", timeout=15, backoff=0.01))
    run = client.run_to_completion(run_id="np", epochs=2, timeout=120)
    assert run.status == "completed"
    assert client.scores.points("np").total == 0
    with pytest.raises((ValidationError, Conflict)):
        client.scores.scorecard("np", 0, "p0")
    with pytest.raises(Conflict):
        client.scores.normalization("np", 0)
    assert client.scores.scorecard("np", 0, "p0", version=1).peer == "p0"
