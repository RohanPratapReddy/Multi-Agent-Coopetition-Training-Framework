"""Continuous docx equations and compliance constraints."""

from __future__ import annotations

import pytest

from mactf.compliance import constrain, is_void
from mactf.scoring.continuous import (
    penalty_from_amounts,
    reward_from_buckets,
    score,
)
from mactf.scoring.ledger import Ledger


def test_reward_equation_matches_docx():
    # R = 1.0*task + 1.5*epi + 1.2*collab + 1.0*align
    value = reward_from_buckets(r_task=1.0, r_epi=1.0, r_collab=1.0, r_align=1.0)
    assert value == pytest.approx(4.7)


def test_penalty_equation_applies_omega():
    raw = penalty_from_amounts(
        {"P20": 1.0}, lambdas={"P20": 2.0}, default_lambda=1.0, omega=1000.0
    )
    assert raw == pytest.approx(2000.0)


def test_penalty_equation_never_shrinks_below_one():
    # Omega is an amplifier, never an attenuator.
    assert penalty_from_amounts(
        {"P6": 1.0}, lambdas={"P6": 1.0}, omega=0.1
    ) == pytest.approx(1.0)


def test_score_builds_buckets_from_ledger(score_config, severity_weights):
    ledger = Ledger(peer="a")
    ledger.record("R1", magnitude=1.0)
    ledger.record("R2", magnitude=2.0)
    result = score(
        ledger, score_config=score_config, severity_weights=severity_weights
    )
    assert result.reward.r_task == pytest.approx(1.0)
    assert result.reward.r_epi == pytest.approx(2.0)
    assert result.reward.total == pytest.approx(1.0 * 1.0 + 1.5 * 2.0)


def test_net_equals_reward_minus_penalty_minus_warning(score_config, severity_weights):
    ledger = Ledger(peer="a")
    ledger.record("R1", magnitude=1.0)
    ledger.record("P11", magnitude=2.0)  # tier 6 penalty, warning eligible
    result = score(
        ledger,
        score_config=score_config,
        severity_weights=severity_weights,
        warning_total=1.5,
    )
    assert result.net == pytest.approx(
        result.reward.total - result.penalty.total - 1.5
    )


def test_non_offsettable_penalty_sets_floor(score_config, severity_weights):
    ledger = Ledger(peer="a")
    ledger.record("R1", magnitude=10.0)
    ledger.record("P1", magnitude=1.0)
    result = score(ledger, score_config=score_config, severity_weights=severity_weights)
    constrained = constrain(result)
    assert constrained.floor == pytest.approx(-5.0)
    assert constrained.value == pytest.approx(-5.0)
    assert constrained.violated


def test_offsettable_penalty_can_be_offset_by_reward(score_config, severity_weights):
    ledger = Ledger(peer="a")
    ledger.record("R1", magnitude=5.0)
    ledger.record("P11", magnitude=1.0)
    result = score(ledger, score_config=score_config, severity_weights=severity_weights)
    constrained = constrain(result)
    assert constrained.floor == 0.0
    assert constrained.value > 0


def test_omega_applies_to_offsettable_penalties(score_config, severity_weights):
    ledger = Ledger(peer="a")
    ledger.record("R1", magnitude=10.0)
    ledger.record("P20", magnitude=1.0)
    ledger.record("P11", magnitude=1.0)
    result = score(ledger, score_config=score_config, severity_weights=severity_weights)
    constrained = constrain(result)
    # P20 is non-offsettable, so Omega must not be multiplied by the offsettable
    # portion on top of it.
    assert constrained.non_offsettable_penalty == pytest.approx(1000.0)
    assert constrained.omega == pytest.approx(1.0)


def test_is_void_only_for_reserved_rules(score_config, severity_weights):
    ledger = Ledger(peer="a")
    ledger.record("P6", magnitude=1.0)
    result = score(ledger, score_config=score_config, severity_weights=severity_weights)
    assert not is_void(result)

    ledger.record("P21", magnitude=1.0)
    result = score(ledger, score_config=score_config, severity_weights=severity_weights)
    assert is_void(result)


def test_compliance_clamps_extreme_values(score_config, severity_weights):
    ledger = Ledger(peer="a")
    ledger.record("P20", magnitude=1000.0)
    result = score(ledger, score_config=score_config, severity_weights=severity_weights)
    constrained = constrain(result, value_cap=1e4)
    assert constrained.value >= -1e4


def test_layered_mode_scores_all_tiers_but_keeps_floor(score_config, severity_weights):
    config = dict(score_config)
    config["PRECEDENCE_MODE"] = "layered"
    ledger = Ledger(peer="a")
    ledger.record("R1", magnitude=1.0)
    ledger.record("P1", magnitude=1.0)
    result = score(ledger, score_config=config, severity_weights=severity_weights)
    assert result.reward.total > 0  # layered keeps the reward component
    constrained = constrain(result)
    assert constrained.floor == pytest.approx(-5.0)
    # framework.md 3.1.2 still wins: the floor is not cancelled by the reward.
    assert constrained.value == pytest.approx(1.0 - 5.0)
