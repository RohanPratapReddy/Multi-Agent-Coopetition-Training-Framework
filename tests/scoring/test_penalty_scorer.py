"""``PenaltyScorer`` must equal ``continuous._penalty_breakdown`` exactly."""

from __future__ import annotations

import pytest

from mactf.exceptions import ComplianceError, ImproperlyConfigured
from mactf.scoring.base import ScoreContext
from mactf.scoring.continuous import _penalty_breakdown
from mactf.scoring.ledger import Ledger
from mactf.scoring.penalty import DEFAULT_LAMBDA, DEFAULT_SANDBOX_MULTIPLIER, PenaltyScorer


def _ctx(ledger, config=None, *, warning_events=(), escalated=frozenset(), extras=None):
    return ScoreContext(
        peer=ledger.peer or "a",
        episode=1,
        ledger=ledger,
        config=dict(config or {}),
        severity_weights={"S0": 0.0, "S1": 0.25, "S2": 1.0, "S3": 4.0, "S4": 25.0},
        warning_events=tuple(warning_events),
        escalated_rules=frozenset(escalated),
        extras=dict(extras or {}),
    )


def test_defaults_match_continuous():
    assert DEFAULT_LAMBDA == 1.0
    assert DEFAULT_SANDBOX_MULTIPLIER == 1000.0


def test_matches_the_continuous_breakdown_for_offsettable_rules(score_config):
    ledger = Ledger(peer="a")
    ledger.record("P4", magnitude=2.0, severity="S2")
    ledger.record("P6", magnitude=1.0, severity="S1")
    config = dict(score_config)
    config["PRECEDENCE_MODE"] = "layered"

    expected = _penalty_breakdown(ledger, config)
    actual = PenaltyScorer(config=config).score(_ctx(ledger, config))

    assert actual.amount == pytest.approx(expected.total, abs=1e-12)
    assert actual.breakdown["amounts"] == pytest.approx(expected.amounts)
    assert actual.breakdown["omega"] == pytest.approx(expected.omega)


def test_warning_only_rule_contributes_nothing(score_config):
    ledger = Ledger(peer="a")
    ledger.record("P6", magnitude=1.0, severity="S1")
    # P6 produced a warning and did not escalate: framework.md W1 -> zero.
    output = PenaltyScorer(config=score_config).score(
        _ctx(ledger, score_config, warning_events=("P6",))
    )
    assert output.amount == pytest.approx(0.0)
    assert output.breakdown["amounts"] == {}
    assert output.metadata["warning_only_excluded"] == ("P6",)


def test_escalated_rule_is_scored_normally(score_config):
    ledger = Ledger(peer="a")
    ledger.record("P6", magnitude=1.0, severity="S1")
    output = PenaltyScorer(config=score_config).score(
        _ctx(ledger, score_config, warning_events=("P6",), escalated=frozenset({"P6"}))
    )
    assert output.amount == pytest.approx(1.0)


def test_escalation_is_counted_exactly_once(score_config):
    ledger = Ledger(peer="a")
    ledger.record("P6", magnitude=1.0, severity="S1")
    output = PenaltyScorer(config=score_config).score(
        _ctx(ledger, score_config, warning_events=("P6",), escalated=frozenset({"P6"}))
    )
    # Not excluded, and not counted twice: one event, one lambda.
    assert output.breakdown["amounts"] == {"P6": pytest.approx(1.0)}


def test_omega_clamps_below_one(score_config):
    ledger = Ledger(peer="a")
    ledger.record("P20", magnitude=1.0, severity="S4")
    config = dict(score_config)
    config["SANDBOX_MULTIPLIER"] = 0.01  # an attenuating multiplier is illegal
    output = PenaltyScorer(config=config).score(_ctx(ledger, config))
    assert output.breakdown["omega"] == pytest.approx(1.0)


def test_omega_is_a_product_of_triggered_rules():
    ledger = Ledger(peer="a")
    ledger.record("P20", magnitude=1.0, severity="S4")
    ledger.record("P21", magnitude=1.0, severity="S4")
    config = {
        "PENALTY_LAMBDAS": {"P20": 1.0, "P21": 1.0},
        "OMEGA_RULES": ["P20", "P21"],
        "SANDBOX_MULTIPLIER": 10.0,
    }
    output = PenaltyScorer(config=config).score(_ctx(ledger, config))
    assert output.breakdown["omega"] == pytest.approx(100.0)
    assert output.breakdown["omega_rules"] == ["P20", "P21"]


def test_omega_rule_override_wins_over_the_default():
    ledger = Ledger(peer="a")
    ledger.record("P20", magnitude=1.0, severity="S4")
    config = {
        "PENALTY_LAMBDAS": {"P20": 1.0},
        "OMEGA_RULES": ["P20"],
        "SANDBOX_MULTIPLIER": 1000.0,
        "OMEGA_RULE_OVERRIDES": {"P20": 3.0},
    }
    output = PenaltyScorer(config=config).score(_ctx(ledger, config))
    assert output.breakdown["omega"] == pytest.approx(3.0)


def test_omega_is_one_when_no_omega_rule_fires(score_config):
    ledger = Ledger(peer="a")
    ledger.record("P6", magnitude=1.0, severity="S1")
    output = PenaltyScorer(config=score_config).score(_ctx(ledger, score_config))
    assert output.breakdown["omega"] == pytest.approx(1.0)
    assert output.breakdown["omega_rules"] == []


def test_strict_mode_honours_the_precedence_allow_list(score_config):
    ledger = Ledger(peer="a")
    ledger.record("P1", magnitude=1.0, severity="S3")   # tier 1
    ledger.record("P11", magnitude=1.0, severity="S1")  # tier 6, suppressed
    output = PenaltyScorer(config=score_config).score(
        _ctx(ledger, score_config, extras={"allowed_rules": frozenset({"P1"})})
    )
    assert set(output.breakdown["amounts"]) == {"P1"}


def test_layered_mode_ignores_the_allow_list(score_config):
    ledger = Ledger(peer="a")
    ledger.record("P1", magnitude=1.0, severity="S3")
    ledger.record("P11", magnitude=1.0, severity="S1")
    config = dict(score_config) | {"PRECEDENCE_MODE": "layered"}
    output = PenaltyScorer(config=config).score(
        _ctx(ledger, config, extras={"allowed_rules": frozenset({"P1"})})
    )
    assert set(output.breakdown["amounts"]) == {"P1", "P11"}


def test_void_rules_are_the_non_offsettable_intersection(score_config):
    ledger = Ledger(peer="a")
    ledger.record("P1", magnitude=1.0, severity="S3")
    ledger.record("P6", magnitude=1.0, severity="S1")
    output = PenaltyScorer(config=score_config).score(
        _ctx(ledger, score_config, extras={"allowed_rules": frozenset({"P1", "P6"})})
    )
    assert output.void_rules == ("P1",)


def test_negative_magnitude_raises_rather_than_paying_a_peer():
    ledger = Ledger(peer="a")
    ledger.record("P6", magnitude=1.0, severity="S1")
    # Simulate a custom ledger that admits a negative magnitude; the constructor
    # rejects it, so bypass the frozen dataclass guard deliberately.
    object.__setattr__(ledger._events[0], "magnitude", -5.0)
    with pytest.raises(ComplianceError, match="cannot be a reward"):
        PenaltyScorer().score(_ctx(ledger))


def test_reads_penalties_only():
    ledger = Ledger(peer="a")
    ledger.record("R1", magnitude=5.0, severity="S2")
    output = PenaltyScorer().score(_ctx(ledger))
    assert output.amount == pytest.approx(0.0)
    assert output.breakdown["rules"] == []


def test_rules_are_sorted_in_the_breakdown(score_config):
    ledger = Ledger(peer="a")
    ledger.record("P11", magnitude=1.0, severity="S1")
    ledger.record("P4", magnitude=1.0, severity="S1")
    config = dict(score_config) | {"PRECEDENCE_MODE": "layered"}
    output = PenaltyScorer(config=config).score(_ctx(ledger, config))
    assert output.breakdown["rules"] == ["P11", "P4"]
    assert output.breakdown["rules"] == sorted(output.breakdown["rules"])


def test_raw_total_is_pre_omega(score_config):
    ledger = Ledger(peer="a")
    ledger.record("P20", magnitude=2.0, severity="S4")
    output = PenaltyScorer(config=score_config).score(
        _ctx(ledger, score_config, extras={"allowed_rules": frozenset({"P20"})})
    )
    assert output.breakdown["raw_total"] == pytest.approx(2000.0)
    assert output.breakdown["total"] == pytest.approx(2000.0 * score_config["SANDBOX_MULTIPLIER"])


def test_validate_rejects_a_negative_lambda():
    diagnostics = PenaltyScorer().validate({"PENALTY_LAMBDAS": {"P1": -1.0}})
    assert [item.code for item in diagnostics] == ["penalties.negative_lambda"]


def test_validate_rejects_a_non_numeric_lambda():
    diagnostics = PenaltyScorer().validate({"PENALTY_LAMBDAS": {"P1": "big"}})
    assert [item.code for item in diagnostics] == ["penalties.non_numeric_lambda"]


def test_validate_rejects_an_attenuating_omega():
    diagnostics = PenaltyScorer().validate({"SANDBOX_MULTIPLIER": 0.5})
    assert [item.code for item in diagnostics] == ["penalties.omega_below_one"]


def test_validate_rejects_a_non_numeric_omega():
    diagnostics = PenaltyScorer().validate({"SANDBOX_MULTIPLIER": "huge"})
    assert [item.code for item in diagnostics] == ["penalties.non_numeric_omega"]


def test_assert_valid_raises_naming_the_problem():
    with pytest.raises(ImproperlyConfigured, match="penalties.negative_lambda"):
        PenaltyScorer().assert_valid({"PENALTY_LAMBDAS": {"P1": -3.0}})


def test_non_offsettable_can_be_overridden_for_a_custom_catalog():
    ledger = Ledger(peer="a")
    ledger.record("P6", magnitude=1.0, severity="S1")
    output = PenaltyScorer(config={"NON_OFFSETTABLE": ["P6"]}).score(
        _ctx(ledger, {"NON_OFFSETTABLE": ["P6"]})
    )
    assert output.void_rules == ("P6",)
