"""The rubric never reaches the docx numbers: points are computed beside ``final_score``, not into it."""

from __future__ import annotations

import copy

import pytest
from points_fixtures import EARNED

from mactf.scoring import EpisodeEvidence, Ledger, PointsEngine, ScoringPipeline

RULES = [("R1", "R2"), ("R1", "P6"), ("R1", "P1"), ("P11", "P11", "P11", "P11"), ("R1", "R4", "P12")]


def _ledger(rules):
    ledger = Ledger(peer="p")
    for rule_id in rules:
        ledger.record(rule_id, evidence="e" if rule_id.startswith("P") else "")
    return ledger


def _strip_points(card_dict):
    out = copy.deepcopy(card_dict)
    out.pop("points", None)
    # the points layer may add its own metadata keys; the docx layers must be untouched
    metadata = out.get("metadata", {})
    out["metadata"] = {k: v for k, v in metadata.items() if k in ("compliance", "cooperative_pool")}
    return out


@pytest.mark.parametrize("rules", RULES, ids=["-".join(r) for r in RULES])
def test_evidence_does_not_change_a_single_docx_number(rules):
    pipeline = ScoringPipeline()
    plain = pipeline.score_peer(_ledger(rules), peer="p", episode=1)
    with_points = pipeline.score_peer(_ledger(rules), peer="p", episode=1, evidence=EpisodeEvidence())
    assert with_points.points is not None and plain.points is None
    assert with_points.final_score == plain.final_score
    assert _strip_points(with_points.to_dict()) == _strip_points(plain.to_dict())


@pytest.mark.parametrize("rule", sorted(EARNED))
def test_every_reward_leaves_final_score_alone_whatever_its_evidence(rule):
    pipeline = ScoringPipeline()
    plain = pipeline.score_peer(_ledger((rule,)), peer="p", episode=1)
    earned = pipeline.score_peer(_ledger((rule,)), peer="p", episode=1, evidence=EARNED[rule])
    empty = pipeline.score_peer(_ledger((rule,)), peer="p", episode=1, evidence=EpisodeEvidence())
    assert plain.final_score == earned.final_score == empty.final_score


def test_scoring_does_not_mutate_the_ledger_or_the_evidence():
    ledger = _ledger(("R1", "P6"))
    evidence = EARNED["R2"]
    before_events = [(e.rule_id, e.magnitude, e.evidence) for e in ledger]
    before_evidence = copy.deepcopy(evidence)
    ScoringPipeline().score_peer(ledger, peer="p", episode=1, evidence=evidence)
    assert [(e.rule_id, e.magnitude, e.evidence) for e in ledger] == before_events
    assert evidence == before_evidence


def test_the_engine_is_pure_and_repeatable():
    engine = PointsEngine()
    first = engine.score(_ledger(("R1", "P12")), evidence=EARNED["R1"], peer="p", episode=1)
    second = engine.score(_ledger(("R1", "P12")), evidence=EARNED["R1"], peer="p", episode=1)
    assert first.to_dict() == second.to_dict()


def test_a_void_episode_still_reports_the_same_negative_final_score():
    pipeline = ScoringPipeline()
    plain = pipeline.score_peer(_ledger(("R1", "P1")), peer="p", episode=1)
    pointed = pipeline.score_peer(_ledger(("R1", "P1")), peer="p", episode=1, evidence=EARNED["R1"])
    assert plain.is_void and pointed.is_void
    assert pointed.final_score == plain.final_score < 0
    assert pointed.points["final_points"] == 0.0
