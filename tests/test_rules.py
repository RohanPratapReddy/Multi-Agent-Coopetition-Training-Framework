"""Catalog, severity, and precedence tests against framework.md."""

from __future__ import annotations

import pytest

from mactf.exceptions import RuleViolation
from mactf.rules.catalog import (
    CATALOG,
    NEVER_WARNING_FIRST,
    NON_OFFSETTABLE,
    PENALTIES,
    REWARDS,
    get_rule,
    validate_catalog,
)
from mactf.rules.precedence import apply_no_netting, resolve
from mactf.rules.severity import Proportionality, Severity, escalate


def test_catalog_matches_framework_md():
    assert validate_catalog() == []
    assert {rule.id for rule in REWARDS} == {f"R{i}" for i in range(1, 18)}
    assert {rule.id for rule in PENALTIES} == {f"P{i}" for i in range(1, 24)}
    assert len(CATALOG) == 40


def test_tiers_follow_framework_precedence():
    # T0 safety, T1 honesty, T2 privacy/IP, T3 epistemic, T4 collaboration, T5 task, T6 efficiency
    assert get_rule("P17").tier == 0
    assert get_rule("P20").tier == 0
    assert get_rule("P1").tier == 1
    assert get_rule("P21").tier == 2
    assert get_rule("P10").tier == 3
    assert get_rule("P6").tier == 4
    assert get_rule("R1").tier == 5
    assert get_rule("P11").tier == 6


def test_non_offsettable_rules_are_marked():
    for rule_id in NON_OFFSETTABLE:
        assert get_rule(rule_id).non_offsettable
        assert get_rule(rule_id).is_penalty


def test_never_warning_first_rules_are_not_warning_eligible():
    for rule_id in NEVER_WARNING_FIRST:
        assert not get_rule(rule_id).warning_eligible


def test_unknown_rule_raises_precise_error():
    with pytest.raises(RuleViolation, match="R1-R17"):
        get_rule("R99")


def test_severity_parse_and_order():
    assert Severity.parse("S3") is Severity.MAJOR
    assert Severity.parse(4) is Severity.SEVERE
    assert Severity.parse("minor") is Severity.MINOR
    with pytest.raises(ValueError):
        Severity.parse("S9")


def test_escalate_caps_at_three():
    assert escalate(Severity.MODERATE, max_tier=3) is Severity.MAJOR
    assert escalate(Severity.MAJOR, max_tier=3) is Severity.MAJOR


def test_proportionality_is_monotone_in_impact():
    low = Proportionality(scope=0.1, reversibility=0.9, intent="accidental")
    high = Proportionality(scope=1.0, reversibility=0.0, intent="deliberate")
    assert high.risk > low.risk
    assert low.suggest() < high.suggest()


def test_proportionality_rejects_bad_inputs():
    with pytest.raises(ValueError):
        Proportionality(scope=1.5)
    with pytest.raises(ValueError):
        Proportionality(intent="malevolent")


def test_strict_precedence_only_scores_controlling_tier():
    rules = [get_rule("R1"), get_rule("R2"), get_rule("P1")]
    decision = resolve(rules, mode="strict")
    # P1 and R2 both sit at T1 (honesty); R1 at T5 is suppressed.
    assert decision.controlling_tier == 1
    assert {rule.id for rule in decision.applied} == {"P1", "R2"}
    assert {rule.id for rule in decision.suppressed} == {"R1"}


def test_strict_precedence_suppresses_lower_tier_penalties():
    rules = [get_rule("P1"), get_rule("P11"), get_rule("P12")]
    decision = resolve(rules, mode="strict")
    assert {rule.id for rule in decision.applied} == {"P1"}
    assert {rule.id for rule in decision.suppressed} == {"P11", "P12"}


def test_layered_precedence_scores_everything_but_still_flags():
    rules = [get_rule("R1"), get_rule("P1")]
    decision = resolve(rules, mode="layered")
    assert {rule.id for rule in decision.applied} == {"R1", "P1"}
    assert decision.suppressed == ()


def test_precedence_is_deterministic_regardless_of_input_order():
    rules = [get_rule("P6"), get_rule("P1"), get_rule("R4")]
    first = resolve(rules, mode="strict")
    second = resolve(list(reversed(rules)), mode="strict")
    assert [rule.id for rule in first.applied] == [rule.id for rule in second.applied]


def test_no_netting_separates_floor_from_offsettable_penalties():
    net, floor, reason = apply_no_netting(
        reward_total=10.0,
        penalty_total=3.0,
        applied=[get_rule("P1"), get_rule("P6")],
        non_offsettable={"P1", "P2", "P17", "P20", "P21", "P22"},
        penalty_amounts={"P1": 3.0},
    )
    assert net == 10.0  # P1 is excluded from the offsettable pool
    assert floor == -3.0
    assert reason is not None and "P1" in reason
    assert net + floor == 7.0


def test_no_netting_without_reserved_penalties_is_plain_subtraction():
    net, floor, reason = apply_no_netting(
        reward_total=10.0,
        penalty_total=4.0,
        applied=[get_rule("P6")],
        non_offsettable={"P1"},
        penalty_amounts={"P6": 4.0},
    )
    assert net == 6.0
    assert floor == 0.0
    assert reason is None
