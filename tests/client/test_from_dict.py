"""Criterion 52: ``from_dict(to_dict(x)) == x`` per class, and unknown keys are ignored."""

from __future__ import annotations

import dataclasses

import pytest

from mactf.client import Run, specs
from mactf.client.client import PointsEntry
from mactf.scoring import DEFAULT_POINTS_TABLE, EpisodeEvidence, Ledger, ScoringPipeline
from mactf.scoring.normalize import CohortNormalizer, NormalizationReport
from mactf.scoring.points import PointsBreakdown, PointsTable
from mactf.scoring.scorecard import EpisodeScorecard, PeerScorecard


def _card(*rules, evidence=None):
    ledger = Ledger(peer="p")
    for rule_id in rules:
        ledger.record(rule_id, evidence="e" if rule_id.startswith("P") else "")
    return ScoringPipeline().score_peer(ledger, peer="p", episode=2, evidence=evidence)


@pytest.mark.parametrize(
    "rules", [("R1", "R2"), ("R1", "P6"), ("R1", "P1"), ("P11", "P11", "P11", "P11")],
    ids=["clean", "suppressed", "void", "warned"],
)
def test_peer_scorecard_round_trips_and_ignores_unknown_keys(rules):
    payload = _card(*rules).to_dict()
    assert PeerScorecard.from_dict(payload).to_dict() == payload
    assert PeerScorecard.from_dict({**payload, "added_by_a_newer_server": 1}).to_dict() == payload


def test_a_scorecard_with_points_round_trips():
    card = _card("R1", "P12", evidence=EpisodeEvidence())
    payload = card.to_dict()
    assert payload["points"] is not None
    again = PeerScorecard.from_dict(payload)
    assert again.to_dict() == payload
    assert PointsBreakdown.from_dict(payload["points"]).to_dict() == payload["points"]


def test_episode_scorecard_round_trips():
    episode = EpisodeScorecard(episode=1, peers={"a": _card("R1")}, metadata={"k": 1})
    payload = episode.to_dict()
    assert EpisodeScorecard.from_dict({**payload, "future": True}).to_dict() == payload


def test_points_table_round_trips_and_ignores_unknown_keys():
    payload = DEFAULT_POINTS_TABLE.to_dict()
    assert PointsTable.from_dict(payload).to_dict() == payload
    assert PointsTable.from_dict({**payload, "extra": 0}).to_dict() == payload


def test_normalization_report_round_trips():
    report = CohortNormalizer().normalize({"a": 80.0, "b": 55.0, "c": 20.0}, episode=3)
    payload = report.as_dict()
    assert NormalizationReport.from_dict({**payload, "extra": 1}).as_dict() == payload


def _spec_names():
    return sorted(n for n in dir(specs) if dataclasses.is_dataclass(getattr(specs, n, None)))


@pytest.mark.parametrize("name", _spec_names())
def test_every_spec_round_trips_and_ignores_unknown_keys(name):
    cls = getattr(specs, name)
    if not (hasattr(cls, "from_dict") and hasattr(cls, "to_dict")):
        pytest.skip(f"{name} has no wire form")
    try:
        instance = cls()
    except TypeError:
        pytest.skip(f"{name} has required fields")
    payload = instance.to_dict()
    assert cls.from_dict(payload) == instance
    assert cls.from_dict({**payload, "unknown_key": 1}) == instance


def test_run_ignores_unknown_keys():
    payload = {"id": "r", "started_at": 1.0, "finished_at": None, "status": "completed", "settings": {}}
    run = Run.from_dict({**payload, "from_the_future": [1, 2]})
    assert run.id == "r" and run.status == "completed"


def test_a_missing_required_key_is_a_keyerror_naming_the_owner_and_key():
    with pytest.raises(KeyError, match="PointsEntry.peer"):
        PointsEntry.from_dict({"episode": 0})
