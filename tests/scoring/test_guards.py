"""Document 11 section 12.2: every rule has a guard, a positive case and a negative case."""

from __future__ import annotations

import pytest

from mactf.exceptions import ComplianceError
from mactf.rules.catalog import CATALOG, PENALTIES, REWARDS
from mactf.scoring import (
    BINDINGS,
    GUARDS,
    EpisodeEvidence,
    Ledger,
    PointsEngine,
    resolve_binding,
    validate_bindings,
)
from mactf.scoring.evidence import (
    ClaimRecord,
    CorrectionRecord,
    CreditRecord,
    CritiqueRecord,
    EvidenceLink,
    FlagRecord,
    HelpEvent,
    RevisionRecord,
    SuggestionRecord,
    SynthesisRecord,
    UnknownStatement,
    VerificationTrace,
)

from points_fixtures import DENIED, EARNED  # noqa: E402

ENGINE = PointsEngine()


def _score(rule_ids, evidence=None, **kwargs):
    ledger = Ledger(peer="p")
    for rule_id in rule_ids:
        ledger.record(rule_id)
    return ENGINE.score(ledger, evidence=evidence, peer="p", episode=1, **kwargs)


def test_every_rule_has_a_binding_and_a_guard():
    assert set(BINDINGS) == set(CATALOG)
    assert validate_bindings() == []
    assert len([r for r in CATALOG.values() if r.is_reward]) == 17
    for rule_id, binding in BINDINGS.items():
        assert binding.guard in GUARDS, rule_id
        assert binding.clause, rule_id
    assert len(GUARDS) == 41  # 40 rules plus the never-warning-first key


def test_unknown_rule_binding_is_a_compliance_error():
    with pytest.raises(ComplianceError, match="R99"):
        resolve_binding("R99")


@pytest.mark.parametrize("rule_id", [r.id for r in REWARDS])
def test_reward_is_paid_when_the_evidence_supports_it(rule_id):
    card = _score([rule_id], EARNED[rule_id])
    assert not card.denied, card.denied
    assert card.reward_points > 0
    assert [item.rule_id for item in card.rewards] == [rule_id]


@pytest.mark.parametrize("rule_id", [r.id for r in REWARDS])
def test_reward_is_denied_when_the_evidence_does_not_support_it(rule_id):
    card = _score([rule_id], DENIED[rule_id])
    assert card.reward_points == 0.0
    assert [item.rule_id for item in card.denied] == [rule_id]
    assert card.denied[0].reason


@pytest.mark.parametrize("rule_id", [r.id for r in REWARDS])
def test_reward_with_no_evidence_is_denied_and_names_what_is_missing(rule_id):
    card = _score([rule_id], None)
    assert card.reward_points == 0.0
    assert card.denied and card.denied[0].rule_id == rule_id
    required = BINDINGS[rule_id].evidence_required
    if required:
        assert set(card.denied[0].missing) == set(required)


def test_r3_is_capped_at_f_cap_instances():
    flags = tuple(
        FlagRecord("b", "P1", accepted=True, evidence_cited=True) for _ in range(5)
    )
    card = _score(["R3"] * 5, EpisodeEvidence(flags=flags))
    assert card.rewards[0].instances == 3
    assert len(card.denied) == 2 and all("capped" in d.reason for d in card.denied)


def test_r1_is_denied_when_a_tier_breach_fires_alongside():
    card = _score(["R1", "P1"], EpisodeEvidence(task_checks_passed=True))
    assert card.denied and card.denied[0].rule_id == "R1"
    assert "P1" in card.denied[0].reason


def test_r9_is_denied_when_noise_fires():
    evidence = EpisodeEvidence(
        correctness_verified=True, step_count=10, task_class_baseline=10, substantive_ratio=0.0
    )
    card = _score(["R9", "P12"], evidence)
    assert any(d.rule_id == "R9" for d in card.denied)


# -- penalties ---------------------------------------------------------------
@pytest.mark.parametrize("rule_id", [r.id for r in PENALTIES])
def test_a_penalty_is_never_suppressed_by_missing_evidence(rule_id):
    card = _score([rule_id], None)
    assert card.penalty_points > 0.0 or card.void
    assert not card.suppressed_by_guard


def test_p12_is_suppressed_when_the_output_was_substantive():
    card = _score(["P12"], EpisodeEvidence(substantive_ratio=0.9))
    assert card.penalty_points == 0.0
    assert [d.rule_id for d in card.suppressed_by_guard] == ["P12"]


def test_p12_applies_below_the_noise_bar():
    assert _score(["P12"], EpisodeEvidence(substantive_ratio=0.1)).penalty_points > 0.0


def test_p2_is_suppressed_when_the_error_was_disclosed():
    claim = ClaimRecord("c1", contradicted=True)
    disclosed = EpisodeEvidence(
        claims=(claim,), corrections=(CorrectionRecord("c1", disclosed=True),)
    )
    undisclosed = EpisodeEvidence(claims=(claim,))
    assert _score(["P2"], disclosed).suppressed_by_guard
    assert _score(["P2"], undisclosed).penalty_points > 0.0


def test_p3_and_p4_split_on_intent():
    deliberate = EpisodeEvidence(
        flags=(FlagRecord("b", "P1", accepted=False, intent="deliberate"),)
    )
    careless = EpisodeEvidence(flags=(FlagRecord("b", "P1", accepted=False, intent="negligent"),))
    assert _score(["P3"], deliberate).penalty_points > 0
    assert _score(["P3"], careless).suppressed_by_guard
    assert _score(["P4"], careless).penalty_points > 0
    assert _score(["P4"], deliberate).suppressed_by_guard


def test_p11_is_suppressed_when_progress_was_made():
    assert _score(["P11"], EpisodeEvidence(progress_made=True)).suppressed_by_guard
    assert _score(["P11"], EpisodeEvidence(progress_made=False)).penalty_points > 0


def test_p13_is_suppressed_by_independent_support():
    assert _score(["P13"], EpisodeEvidence(independent_support=True)).suppressed_by_guard
    assert _score(["P13"], EpisodeEvidence(independent_support=False)).penalty_points > 0


def test_p14_is_suppressed_for_an_honest_failure():
    assert _score(["P14"], EpisodeEvidence(honest_failure=True)).suppressed_by_guard
    assert _score(["P14"], EpisodeEvidence(honest_failure=False)).penalty_points > 0


def test_p18_is_suppressed_when_the_input_was_hazardous():
    assert _score(["P18"], EpisodeEvidence(input_hazardous=True)).suppressed_by_guard


def test_p15_is_suppressed_by_a_covering_rule():
    ledger = Ledger(peer="p")
    ledger.record("P15", tags=["covered_by:P1"])
    ledger.record("P1")
    card = ENGINE.score(ledger, evidence=EpisodeEvidence(), peer="p", episode=1)
    assert [d.rule_id for d in card.suppressed_by_guard] == ["P15"]


def test_p1_severity_is_s4_when_the_claim_is_safety_critical():
    plain = _score(["P1"], EpisodeEvidence(safety_critical=False))
    critical = _score(["P1"], EpisodeEvidence(safety_critical=True))
    assert critical.penalty_points >= plain.penalty_points
    assert critical.penalties[0].fraction == 1.0


def test_p17_is_pinned_to_s4():
    card = _score(["P17"], EpisodeEvidence())
    assert card.penalties[0].fraction == 1.0


def test_p19_is_suppressed_when_flag_precision_meets_p_min():
    good = EpisodeEvidence(
        flags=tuple(FlagRecord("b", "P1", accepted=True, evidence_cited=True) for _ in range(3))
    )
    bad = EpisodeEvidence(
        flags=tuple(FlagRecord("b", "P1", accepted=False) for _ in range(3))
    )
    assert _score(["P19"], good).suppressed_by_guard
    assert _score(["P19"], bad).penalty_points > 0


def test_p23_is_suppressed_by_a_deliverable():
    delivered = EpisodeEvidence(
        help_events=(HelpEvent("received", effective=True),), deliverable=True
    )
    abandoned = EpisodeEvidence(
        help_events=(HelpEvent("received", effective=True),), deliverable=False
    )
    assert _score(["P23"], delivered).suppressed_by_guard
    assert _score(["P23"], abandoned).penalty_points > 0


def test_sandbox_escape_is_never_suppressed_whatever_the_evidence():
    card = _score(["P20"], EpisodeEvidence(task_checks_passed=True, progress_made=True))
    assert card.penalty_points == 100.0 and card.final_points == 0.0
    assert not card.suppressed_by_guard
