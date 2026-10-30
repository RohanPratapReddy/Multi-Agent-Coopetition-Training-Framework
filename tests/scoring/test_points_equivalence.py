"""Document 11 section 12.6: the points layer never changes final_score."""

from __future__ import annotations

import itertools

import pytest
from points_fixtures import EARNED

from mactf.scoring import EpisodeEvidence, Ledger, PointsEngine, ScoringPipeline

RULE_SETS = [
    [],
    ["R1"],
    ["R1", "R2"],
    ["P4"],
    ["R1", "P12"],
    ["R1", "P1"],
    ["R1", "P20"],
    ["R3", "R3", "R3", "R3"],
    ["P11", "P11", "P11"],
]


def _ledger(rules):
    ledger = Ledger(peer="p")
    for rule_id in rules:
        ledger.record(rule_id)
    return ledger


@pytest.mark.parametrize("rules", RULE_SETS)
def test_final_score_is_identical_with_and_without_evidence(rules):
    plain = ScoringPipeline().score_peer(_ledger(rules), peer="p", episode=1)
    evidence = EpisodeEvidence(**{
        k: v for rule in EARNED.values() for k, v in vars(rule).items() if v not in ((), None, False, 0)
    })
    scored = ScoringPipeline().score_peer(_ledger(rules), peer="p", episode=1, evidence=evidence)
    assert scored.final_score == plain.final_score
    assert scored.floor == plain.floor
    assert scored.violated_non_offsettable == plain.violated_non_offsettable
    assert plain.points is None and scored.points is not None


@pytest.mark.parametrize("rules", RULE_SETS)
def test_void_status_agrees_between_the_two_layers(rules):
    scored = ScoringPipeline().score_peer(
        _ledger(rules), peer="p", episode=1, evidence=EpisodeEvidence()
    )
    assert scored.points["void"] == scored.is_void
    if scored.is_void:
        assert scored.points["final_points"] == 0.0


def test_points_are_reproducible():
    evidence = EARNED["R1"]
    first = ScoringPipeline().score_peer(_ledger(["R1"]), peer="p", episode=1, evidence=evidence)
    second = ScoringPipeline().score_peer(_ledger(["R1"]), peer="p", episode=1, evidence=evidence)
    assert first.points == second.points


def test_every_pair_of_rewards_is_additive_up_to_the_weights():
    # The engine alone, so precedence (which the pipeline applies first) is out of play.
    engine = PointsEngine()
    for left, right in itertools.combinations(["R1", "R2", "R12", "R8"], 2):
        both = EpisodeEvidence(**{
            k: v
            for rule in (EARNED[left], EARNED[right])
            for k, v in vars(rule).items()
            if v not in ((), None, False, 0)
        })
        total = engine.score(_ledger([left, right]), evidence=both).reward_points
        singles = sum(
            engine.score(_ledger([rule]), evidence=EARNED[rule]).reward_points
            for rule in (left, right)
        )
        assert total == pytest.approx(singles)