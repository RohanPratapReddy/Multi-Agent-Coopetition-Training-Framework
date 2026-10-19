"""Ledger, warning regime, and scorecard tests."""

from __future__ import annotations

import pytest

from mactf.exceptions import RuleViolation
from mactf.scoring.ledger import Ledger, RuleEvent
from mactf.scoring.warning import WarningPolicy, WarningRegistry
from mactf.rules.severity import Severity


def test_ledger_validates_magnitude():
    ledger = Ledger(peer="a")
    with pytest.raises(RuleViolation):
        ledger.record("R1", magnitude=-1.0)


def test_ledger_rejects_unknown_rule():
    with pytest.raises(RuleViolation):
        Ledger().record("P999")


def test_ledger_bucket_totals():
    ledger = Ledger()
    ledger.record("R1", magnitude=2.0)
    ledger.record("R2", magnitude=3.0)
    ledger.record("R5", magnitude=1.0)
    assert ledger.bucket_total("r_task") == 2.0
    assert ledger.bucket_total("r_epi") == 3.0
    assert ledger.bucket_total("r_collab") == 1.0
    assert ledger.bucket_total("r_align") == 0.0


def test_ledger_penalty_amounts_apply_lambdas():
    ledger = Ledger()
    ledger.record("P1", magnitude=2.0)
    ledger.record("P6", magnitude=1.0)
    amounts = ledger.penalty_amounts(lambdas={"P1": 5.0}, default_lambda=1.0)
    assert amounts["P1"] == 10.0
    assert amounts["P6"] == 1.0


def test_ledger_extend_revalidates():
    ledger = Ledger()
    event = RuleEvent(rule_id="R1", magnitude=1.0)
    ledger.extend([event])
    assert len(ledger) == 1


def test_warning_escalates_after_k_esc(warning_policy):
    registry = WarningRegistry(warning_policy)
    outcomes = [
        registry.issue("peer", "P11", severity=Severity.MINOR, episode=index)
        for index in range(3)
    ]
    assert outcomes[0].is_warning_only
    assert outcomes[1].is_warning_only
    assert outcomes[2].escalated is not None
    assert outcomes[2].penalty == pytest.approx(5.0)
    # W3: escalation raises by exactly one tier.
    assert outcomes[2].severity is Severity.MODERATE


def test_warning_is_scoped_per_rule_id(warning_policy):
    registry = WarningRegistry(warning_policy)
    for index in range(2):
        registry.issue("peer", "P11", severity=Severity.MINOR, episode=index)
        registry.issue("peer", "P12", severity=Severity.MINOR, episode=index)
    # Two warnings each, below K_ESC=3: no escalation across pooled rules.
    assert registry.count("peer", "P11") == 2
    assert registry.count("peer", "P12") == 2


def test_warning_never_applies_to_non_eligible_rule(warning_policy):
    registry = WarningRegistry(warning_policy)
    outcome = registry.issue("peer", "P1", severity=Severity.MAJOR, episode=0)
    assert outcome.escalated is None
    assert outcome.issued == []
    assert outcome.penalty > 0  # immediate penalty instead


def test_warning_decay_after_clean_streak(warning_policy):
    registry = WarningRegistry(warning_policy)
    registry.issue("peer", "P11", severity=Severity.MINOR, episode=0)
    for _ in range(warning_policy.n_decay):
        registry.mark_clean("peer", "P11")
    # Past the N_WIN window and with a full clean streak, it expires.
    expired = registry.begin_episode("peer", episode=warning_policy.n_win + 1)
    assert expired
    assert registry.count("peer", "P11") == 0


def test_warning_does_not_expire_inside_window(warning_policy):
    registry = WarningRegistry(warning_policy)
    registry.issue("peer", "P11", severity=Severity.MINOR, episode=0)
    for _ in range(warning_policy.n_decay):
        registry.mark_clean("peer", "P11")
    expired = registry.begin_episode("peer", episode=1)
    assert expired == []
    assert registry.count("peer", "P11") == 1


def test_warning_cap_retires_oldest(warning_policy):
    registry = WarningRegistry(warning_policy)
    # Escalate is disabled so we can observe the cap independently.
    registry.policy = WarningPolicy(k_esc=5, n_win=5, n_decay=3, w_cap=2, p_warn=5.0)
    for index in range(3):
        registry.issue("peer", "P11", severity=Severity.MINOR, episode=index)
    assert registry.count("peer", "P11") == 2


def test_expunge_clears_warnings(warning_policy):
    registry = WarningRegistry(warning_policy)
    registry.issue("peer", "P11", severity=Severity.MINOR, episode=0)
    removed = registry.expunge("peer", "P11")
    assert removed
    assert registry.count("peer", "P11") == 0


def test_scorecard_void_when_non_offsettable(score_config, severity_weights, warning_policy):
    from mactf.scoring.pipeline import Scorer

    scorer = Scorer(
        score_config=score_config,
        severity_weights=severity_weights,
        warning_policy=warning_policy,
    )
    ledger = Ledger(peer="a")
    ledger.record("R1", magnitude=1.0)
    ledger.record("P1", magnitude=1.0, evidence="fabricated citation")
    card = scorer.score_peer(ledger, peer="a", episode=0)
    assert card.is_void
    assert card.violated_non_offsettable == ("P1",)
    # R1 must not survive: precedence suppresses the task-success tier.
    assert card.continuous.reward.total == 0.0
    assert card.final_score < 0


def test_scorecard_records_suppression(score_config, severity_weights, warning_policy):
    from mactf.scoring.pipeline import Scorer

    scorer = Scorer(
        score_config=score_config,
        severity_weights=severity_weights,
        warning_policy=warning_policy,
    )
    ledger = Ledger(peer="a")
    ledger.record("R1", magnitude=1.0)
    ledger.record("P6", magnitude=1.0)
    card = scorer.score_peer(ledger, peer="a", episode=0)
    assert {item.rule_id for item in card.suppressed} == {"R1"}


def test_scorecard_explain_is_readable(score_config, severity_weights, warning_policy):
    from mactf.scoring.pipeline import Scorer

    scorer = Scorer(
        score_config=score_config,
        severity_weights=severity_weights,
        warning_policy=warning_policy,
    )
    ledger = Ledger(peer="a")
    ledger.record("R1", magnitude=0.8)
    card = scorer.score_peer(ledger, peer="a", episode=0)
    text = card.explain()
    assert "final=" in text and "reward=" in text
