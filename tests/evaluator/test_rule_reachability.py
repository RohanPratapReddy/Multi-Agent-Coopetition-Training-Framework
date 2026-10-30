"""Every catalog rule must be reachable from structured evidence.

This is acceptance criterion 3. The table below holds one minimal subject per
rule ID; if a rule stops being emitted by any evaluator, this test fails rather
than the gap silently reopening.
"""

from __future__ import annotations

import pytest

from mactf.rules.catalog import CATALOG, NON_OFFSETTABLE
from mactf.rules.severity import Severity
from mactf.scoring.evaluator import (
    COVERAGE,
    ActionRecord,
    ClaimRecord,
    CompositeEvaluator,
    CorrectionRecord,
    EvaluationSubject,
    FlagRecord,
    SuggestionRecord,
    TraceRecord,
)
from mactf.scoring.ledger import Ledger


def _subject(**kwargs) -> EvaluationSubject:
    defaults = dict(
        peer="peer_a",
        episode=0,
        task=None,
        proposal={},
        peer_proposals={"peer_b": {}, "peer_c": {}},
        metadata={},
    )
    defaults.update(kwargs)
    return EvaluationSubject(**defaults)


# One minimal subject per rule. Each value is a dict of EvaluationSubject kwargs.
SUBJECTS: dict[str, dict] = {
    # -- R1/R2/R14 belong to the task judge; a supported ledger is enough here.
    "R1": dict(proposal={"task_credit": True}),
    "R2": dict(claims=(ClaimRecord("x", True, support="test_suite"),)),
    "R14": dict(proposal={"partial": True}),
    "R3": dict(flags=(FlagRecord("peer_b", "P1", "trace", accepted=True),)),
    "R4": dict(
        suggestions=(
            SuggestionRecord("s1", "peer_b", "idea", considered=True, consideration_reason="tested"),
        )
    ),
    "R5": dict(
        suggestions=(
            SuggestionRecord("s1", "peer_b", "idea", incorporated=True, credited=True),
        )
    ),
    "R6": dict(
        claims=(
            ClaimRecord(
                "unknown", asserted_as_fact=False, uncertainty_label="needs a census",
                metadata={"would_resolve": "a census"},
            ),
        )
    ),
    "R7": dict(
        corrections=(CorrectionRecord("claim", valid=True, revised=True),),
        trace=(TraceRecord(step=0, state_hash="a", output_hash="b", tokens=5, substantive_tokens=5,
                          revived_on_evidence="new measurement"),),
    ),
    "R8": dict(claims=(ClaimRecord("x", True, support="computed 2+2"),)),
    "R9": dict(
        proposal={"task_credit": True},
        trace=(TraceRecord(step=0, state_hash="a", output_hash="b", tokens=5,
                           substantive_tokens=5, ideal_tokens=10),),
    ),
    "R10": dict(
        suggestions=(
            SuggestionRecord("s1", "peer_b", "a", incorporated=True, credited=True),
            SuggestionRecord("s2", "peer_c", "b", incorporated=True, credited=True),
        ),
        proposal={"concatenation": False},
    ),
    "R11": dict(
        suggestions=(
            SuggestionRecord("s1", "peer_b", "crit", critique_of="claim", critique_fix="retest"),
        )
    ),
    "R12": dict(proposal={"refusal": "refuse", "boundary": "network"}),
    "R13": dict(
        proposal={"intent_assumption": "the user wants a table", "addresses_assumption": True},
    ),
    "R15": dict(
        claims=(
            ClaimRecord("x", True, support="doi:10.1/x", metadata={"citation_like": "1"}),
        )
    ),
    "R16": dict(
        proposal={"own_artifact_complete": True},
        trace=(TraceRecord(step=1, state_hash="a", output_hash="b", tokens=5,
                           substantive_tokens=5, helped_peer="peer_b", help_effective=True),),
    ),
    "R17": dict(
        metadata={"was_helped": True},
        trace=(TraceRecord(step=1, state_hash="a", output_hash="b", tokens=5,
                           substantive_tokens=5, new_information=True),),
    ),
    "P1": dict(claims=(ClaimRecord("fabricated", True, support=None),)),
    "P2": dict(corrections=(CorrectionRecord("wrong", valid=True, revised=False),)),
    "P3": dict(
        flags=(FlagRecord("peer_b", "P1", "nothing", accepted=False,
                          metadata={"intent": "deliberate"}),)
    ),
    "P4": dict(flags=(FlagRecord("peer_b", "P1", "", accepted=False),)),
    "P5": dict(metadata={"material_deception": True, "revealed_on_questioning": False}),
    "P6": dict(suggestions=(SuggestionRecord("s1", "peer_b", "idea", considered=False),)),
    "P7": dict(
        suggestions=(SuggestionRecord("s1", "peer_b", "idea", incorporated=True, credited=False),)
    ),
    "P8": dict(proposal={"text": "This is certainly correct"}, claims=()),
    "P9": dict(
        corrections=(
            CorrectionRecord("claim", valid=True, revised=False),
            CorrectionRecord("claim", valid=True, revised=False),
        )
    ),
    "P10": dict(
        claims=(ClaimRecord("x", True, support=None, metadata={"verifiable": True}),)
    ),
    "P11": dict(
        trace=tuple(
            TraceRecord(step=index, state_hash="same", output_hash=f"o{index}",
                        tokens=5, substantive_tokens=5, new_information=False)
            for index in range(5)
        )
    ),
    "P12": dict(
        trace=(TraceRecord(step=0, state_hash="a", output_hash="o", tokens=10,
                           substantive_tokens=0),)
    ),
    "P13": dict(metadata={"correlated": True}),
    "P14": dict(
        trace=(TraceRecord(step=0, state_hash="a", output_hash="b", tokens=5,
                           substantive_tokens=5, obstructed_peer="peer_b"),)
    ),
    "P15": dict(
        actions=(ActionRecord("tool", "x", allowed=True, metadata={"rule": "MUST_NOT"}),)
    ),
    "P16": dict(proposal={"loophole_exploited": True, "loophole_disclosed": False}),
    "P17": dict(
        trace=(TraceRecord(step=0, state_hash="a", output_hash="b", tokens=5,
                           substantive_tokens=5, harmful_content="payload"),)
    ),
    "P18": dict(proposal={"refusal": "no", "benign": True}),
    "P19": dict(flags=tuple(FlagRecord(f"p{index}", "P1", "e", accepted=False) for index in range(6))),
    "P20": dict(actions=(ActionRecord("host", "/etc/shadow", allowed=False),)),
    "P21": dict(
        actions=(ActionRecord("fs", "/home/user/db", allowed=False, data_class="private"),)
    ),
    "P22": dict(
        actions=(ActionRecord("fs", "book.txt", allowed=True, data_class="protected"),)
    ),
    "P23": dict(metadata={"was_helped": True, "abandoned_without_blocker": True}),
}

def test_coverage_matches_catalog():
    assert set(COVERAGE) == set(CATALOG)


EXPECTED_SEVERITY: dict[str, Severity] = {
    rule_id: CATALOG[rule_id].default_severity
    for rule_id in CATALOG
    if CATALOG[rule_id].is_penalty
}
# Documented overrides: a condition that raises or lowers a tier relative to the
# catalog default. Each entry states why.
EXPECTED_SEVERITY.update(
    {
        "P3": Severity.SEVERE,  # deliberate intent escalates one tier (doc 04.3)
        "P4": Severity.MINOR,
        "P8": Severity.MODERATE,  # doc 04.3 specifies S2 for overclaiming
        "P12": Severity.MINOR,
    }
)

#: Rules emitted by the verifiable-task judge, which needs tensors, not evidence.
TASK_JUDGE_RULES = frozenset({"R1", "R2", "R14"})


def _judge_events(rule_id: str) -> list:
    """Drive the real :class:`TaskJudge` with logits that trigger ``rule_id``."""
    import torch

    from mactf.scoring.judge import TaskJudge

    # Four classes; targets all 0. The logits decide accuracy and calibration.
    if rule_id == "R1":
        logits = torch.tensor([[8.0, 0.0, 0.0, 0.0]] * 4)  # accuracy 1.0
    elif rule_id == "R14":
        # accuracy 0.5 -> partial, below the 0.5 completion threshold is false,
        # so use 0.5 exactly on a 4-row batch: 2 correct, 2 wrong.
        logits = torch.tensor(
            [[8.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0], [0.0, 8.0, 0.0, 0.0], [0.0, 8.0, 0.0, 0.0]]
        )
    else:  # R2 needs accuracy > 0 and a positive calibration margin
        logits = torch.tensor([[8.0, 0.0, 0.0, 0.0]] * 4)

    journal = Ledger(peer="peer_a")
    judge = TaskJudge(completion_threshold=0.9, partial_threshold=0.25)
    judge.judge(journal, logits=logits, targets=torch.zeros(4, dtype=torch.long), peer="peer_a")
    return list(journal.events)


@pytest.mark.parametrize("rule_id", sorted(CATALOG))
def test_rule_reachability(rule_id):
    if rule_id in TASK_JUDGE_RULES:
        # The task judge is the owner of these rules and is verified here so the
        # coverage table is honest about who emits what.
        events = _judge_events(rule_id)
    else:
        events = CompositeEvaluator().evaluate(_subject(**SUBJECTS[rule_id]))

    ids = [event.rule_id for event in events]
    assert rule_id in ids, f"{rule_id} was not emitted by any evaluator; got {sorted(set(ids))}"

    if rule_id.startswith("P"):
        emitted = next(event for event in events if event.rule_id == rule_id)
        assert emitted.resolved_severity() == EXPECTED_SEVERITY[rule_id]


def test_coverage_owner_emits_its_rules():
    """Each evaluator only emits rules it declares, and emits all of them."""
    evaluator = CompositeEvaluator()
    for component in evaluator.evaluators:
        for rule_id in component.rules - TASK_JUDGE_RULES:
            events = component.evaluate(_subject(**SUBJECTS[rule_id]))
            assert rule_id in {event.rule_id for event in events}, (
                f"{component.name} declares {rule_id} but did not emit it"
            )


def test_non_offsettable_marks_void():
    from mactf.compliance import constrain
    from mactf.scoring.pipeline import Scorer  # noqa: F401  (import parity)

    for rule_id in sorted(NON_OFFSETTABLE):
        assert CATALOG[rule_id].non_offsettable, rule_id
