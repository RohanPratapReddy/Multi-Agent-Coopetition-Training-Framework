"""Warning-first eligibility must match the catalog.

The evaluators emit severities; the warning policy decides whether a first
occurrence becomes a warning. Those two must agree with the catalog's
``warning_eligible`` flag and with ``NEVER_WARNING_FIRST``, or a rule that must
penalize immediately could be softened into a warning - which is exactly what
framework.md W9 forbids.
"""

from __future__ import annotations

import pytest

from mactf.rules.catalog import CATALOG, NEVER_WARNING_FIRST
from mactf.scoring.evaluator import CompositeEvaluator
from mactf.scoring.ledger import Ledger
from mactf.scoring.warning import WarningPolicy

from .test_rule_reachability import SUBJECTS, _subject


def test_never_warning_first_rules_are_marked_ineligible():
    for rule_id in NEVER_WARNING_FIRST:
        rule = CATALOG[rule_id]
        assert rule.is_penalty, rule_id
        assert rule.warning_eligible is False, (
            f"{rule_id} is in NEVER_WARNING_FIRST but the catalog marks it warning-eligible"
        )


@pytest.mark.parametrize("rule_id", sorted(NEVER_WARNING_FIRST))
def test_severe_rule_never_becomes_a_warning(rule_id):
    policy = WarningPolicy(k_esc=3, n_win=5, n_decay=3, w_cap=2, p_warn=5.0)
    registry = policy_registry(policy)
    journal = CompositeEvaluator().apply(Ledger(peer="peer_a"), _subject(**SUBJECTS[rule_id]))
    events = [event for event in journal.penalties() if event.rule_id == rule_id]
    assert events, f"{rule_id} was not emitted"

    for event in events:
        outcome = registry.issue(
            "peer_a",
            rule_id,
            severity=event.resolved_severity(),
            episode=0,
            evidence=event.evidence,
        )
        assert outcome.issued == [], f"{rule_id} must not issue a warning"
        assert outcome.escalated is None
        assert outcome.penalty > 0.0, f"{rule_id} must carry a penalty on first occurrence"
        assert registry.count("peer_a", rule_id) == 0


def test_warning_eligible_rule_warns_first():
    """The complement: an eligible rule (P6) earns a warning, not a penalty."""
    policy = WarningPolicy(k_esc=3, n_win=5, n_decay=3, w_cap=2, p_warn=5.0)
    registry = policy_registry(policy)
    outcome = registry.issue("peer_a", "P6", severity="S1", episode=0, evidence="ignored")
    assert len(outcome.issued) == 1
    assert outcome.penalty == 0.0
    assert registry.count("peer_a", "P6") == 1


def policy_registry(policy: WarningPolicy):
    from mactf.scoring.warning import WarningRegistry

    return WarningRegistry(policy)
