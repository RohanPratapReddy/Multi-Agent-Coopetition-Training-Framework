"""Document 11 section 12.3: monotonicity M1-M4 and the points/final_score divergence."""

from __future__ import annotations

import pytest

from mactf.rules.catalog import PENALTIES, REWARDS
from mactf.scoring import EpisodeEvidence, Ledger, PointsEngine, ScoringPipeline
from mactf.scoring.evidence import (
    ClaimRecord,
    EvidenceLink,
    FlagRecord,
    HelpEvent,
    SuggestionRecord,
    SynthesisRecord,
    UnknownStatement,
    VerificationTrace,
)

from points_fixtures import EARNED

ENGINE = PointsEngine()
FULL = EpisodeEvidence(task_checks_passed=True, claims=(ClaimRecord("c1", supported=True),))


def _points(rules, evidence, **kwargs):
    ledger = Ledger(peer="p")
    for rule_id in rules:
        ledger.record(rule_id)
    return ENGINE.score(ledger, evidence=evidence, peer="p", episode=1, **kwargs)


@pytest.mark.parametrize("rule_id", [r.id for r in REWARDS])
def test_m1_an_earned_reward_never_lowers_points(rule_id):
    base = _points(["R1"], EARNED["R1"]).final_points
    both = EpisodeEvidence(**{**vars(EARNED["R1"]), **{k: v for k, v in vars(EARNED[rule_id]).items()
                                                      if v not in ((), None, False, 0)}})
    assert _points(["R1", rule_id], both).final_points >= base


@pytest.mark.parametrize("rule_id", [p.id for p in PENALTIES])
def test_m2_a_penalty_never_raises_points(rule_id):
    base = _points(["R1"], EARNED["R1"])
    worse = _points(["R1", rule_id], EARNED["R1"])
    assert worse.final_points <= base.final_points


@pytest.mark.parametrize("rule_id", [r.id for r in REWARDS])
def test_m3_a_denied_reward_pays_nothing_and_changes_nothing(rule_id):
    assert _points([rule_id], None).final_points == 0.0
    assert _points([rule_id], None).reward_points == 0.0


def test_m3_removing_evidence_never_increases_points():
    full = _points(["R1", "R2"], FULL)
    stripped = _points(["R1", "R2"], EpisodeEvidence())
    assert stripped.final_points <= full.final_points


def test_m4_a_void_pins_final_points_to_zero_whatever_was_earned():
    breakdown = _points(["R1", "R2"], FULL, void_rules=("P20",))
    assert breakdown.reward_points > 0
    assert breakdown.void and breakdown.final_points == 0.0
    assert breakdown.void_rules == ("P20",)


def test_points_are_bounded_to_0_100():
    heavy = _points(["P17", "P20", "P21", "P22"], EpisodeEvidence())
    assert heavy.penalty_points <= 100.0 and heavy.final_points == 0.0
    top = _points([r.id for r in REWARDS], EpisodeEvidence(**{
        k: v for rule in EARNED.values() for k, v in vars(rule).items() if v not in ((), None, False, 0)
    }))
    assert top.final_points <= 100.0


def test_a_warning_only_rule_costs_nothing():
    ledger = Ledger(peer="p")
    ledger.record("P11")
    card = ENGINE.score(
        ledger, evidence=EpisodeEvidence(progress_made=False), peer="p", warning_only={"P11"}
    )
    assert card.penalty_points == 0.0


def test_precedence_suppressed_rewards_are_reported_not_paid():
    ledger = Ledger(peer="p")
    ledger.record("R1")
    card = ENGINE.score(ledger, evidence=EARNED["R1"], peer="p", suppressed_rewards=["R1"])
    assert card.reward_points == 0.0 and card.suppressed_by_precedence == ("R1",)


def test_final_score_and_final_points_may_disagree_on_a_void_peer():
    pipeline = ScoringPipeline()
    ledger = Ledger(peer="p")
    ledger.record("R1")
    ledger.record("P20")
    card = pipeline.score_peer(ledger, peer="p", episode=1, evidence=EARNED["R1"])
    assert card.points is not None
    assert card.points["void"] is True and card.points["final_points"] == 0.0
    # The legacy score is untouched by the points layer.
    plain = ScoringPipeline().score_peer(
        _rebuild(["R1", "P20"]), peer="p", episode=1
    )
    assert plain.points is None
    assert card.final_score == plain.final_score


def test_pipeline_attaches_points_and_an_evidence_digest():
    ledger = Ledger(peer="p")
    ledger.record("R1")
    card = ScoringPipeline().score_peer(ledger, peer="p", episode=1, evidence=EARNED["R1"])
    assert card.points["final_points"] == pytest.approx(25.0)
    assert card.metadata["evidence_digest"] == EARNED["R1"].digest()
    assert card.metadata["points_table_version"] == card.points["table_version"]


def test_the_pipeline_accepts_evidence_as_a_mapping():
    ledger = Ledger(peer="p")
    ledger.record("R1")
    card = ScoringPipeline().score_peer(
        ledger, peer="p", episode=1, evidence={"task_checks_passed": True}
    )
    assert card.points["final_points"] == pytest.approx(25.0)


def _rebuild(rules):
    ledger = Ledger(peer="p")
    for rule_id in rules:
        ledger.record(rule_id)
    return ledger
