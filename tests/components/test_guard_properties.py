"""Hypothesis attacks on ``ComplianceGuard`` (document 17 section 19.7.1).

The property is the guarantee of I2: whatever a custom stage returns, either the guard
refuses it (``ComplianceGuardViolation``, never a score), or the score it allows still
sits under the reference floor, and a void peer is still void.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mactf.compliance import effective_non_offsettable
from mactf.exceptions import ComplianceGuardViolation
from mactf.rules.catalog import ALL_RULES, NON_OFFSETTABLE
from mactf.scoring.penalty import PenaltyScorer

from scoring_helpers import (
    default_pipeline,
    ledger_of,
    penalty_variant,
    pipeline_with,
    reward_variant,
    settings_copy,
    with_output,
)

PROFILE = settings(
    max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)

REWARD_IDS = [rule.id for rule in ALL_RULES if rule.is_reward]
PENALTY_IDS = [rule.id for rule in ALL_RULES if rule.is_penalty]
NO_IDS = sorted(NON_OFFSETTABLE)

magnitude = st.floats(min_value=0.05, max_value=6.0, allow_nan=False, allow_infinity=False)
ledgers = st.fixed_dictionaries(
    {},
    optional={
        **{rule: magnitude for rule in PENALTY_IDS},
        **{rule: magnitude for rule in REWARD_IDS},
    },
)
# the values a buggy or hostile bucket can take: ordinary, huge, negative, non-finite
numbers = st.one_of(
    st.floats(min_value=0.0, max_value=50.0),
    st.floats(min_value=-1.0e12, max_value=1.0e12),
    st.sampled_from([0.0, -0.0, 1.0e4, 1.0e4 + 1e-6, 1.0e300, float("inf"), float("-inf"), float("nan")]),
)
BUCKET_KEYS = ("r_task", "r_epi", "r_collab", "r_align")
bucket_vectors = st.fixed_dictionaries({key: numbers for key in BUCKET_KEYS})

TOL = 1e-6


def reference_floor(spec) -> float:
    """sum_k m_k * max(lambda_k, MIN_FLOOR_LAMBDA) over the admitted non-offsettable events."""
    reference = default_pipeline()
    card = reference.score_peer(ledger_of(spec), peer="r", episode=1)
    return sum(
        amount for rule, amount in card.continuous.penalty.amounts.items() if rule in NON_OFFSETTABLE
    )


def run(pipeline, spec, **kwargs):
    return pipeline.score_peer(ledger_of(spec), peer="a", episode=1, **kwargs)


@PROFILE
@given(spec=ledgers, buckets=bucket_vectors, layered=st.booleans())
def test_guard_property_compliance_floor(spec, buckets, layered):
    """A reward stage returning any float, inf, nan or huge value cannot weaken a floor."""
    overrides = {"SCORING": {"PRECEDENCE_MODE": "layered"}} if layered else {}
    pipeline = pipeline_with(
        reward=reward_variant(lambda out, ctx: with_output(out, buckets)),
        settings=settings_copy(**overrides),
    )
    try:
        card = run(pipeline, spec)
    except ComplianceGuardViolation as error:
        # refused: only for a reason the guard knows, and the first one is a C-code
        assert error.code in {"C001", "C008", "C009"}
        weights = settings_copy()["SCORING"]["REWARD_WEIGHTS"]
        assert any(not math.isfinite(v) or v < 0 for v in buckets.values()) or sum(
            weights[key] * value for key, value in buckets.items()
        ) > 1.0e4
        return
    assert all(math.isfinite(v) and v >= 0 for v in buckets.values())
    floor = reference_floor(spec)
    assert math.isfinite(card.final_score)
    # I2: the breach costs at least the reference floor, whatever the reward paid
    assert card.final_score <= max(-1.0e4, card.continuous.reward.total - floor) + TOL
    # and the breach is on the card exactly as the default pipeline reports it
    reference_card = default_pipeline(**overrides).score_peer(ledger_of(spec), peer="r", episode=1)
    assert card.violated_non_offsettable == reference_card.violated_non_offsettable


@PROFILE
@given(spec=ledgers, kind=st.sampled_from(["zero", "scale", "drop_void", "drop_one", "honest", "double"]),
       factor=st.floats(min_value=0.0, max_value=3.0), branch=st.sets(st.sampled_from(NO_IDS)))
def test_guard_property_void_stays_void(spec, kind, factor, branch):
    """A void peer is void after every stage: the card says so and the score is at its floor."""

    def attack(output, ctx):
        amounts = dict(output.breakdown["amounts"])
        if kind == "zero":
            amounts = {}
        elif kind == "scale":
            amounts = {rule: value * factor for rule, value in amounts.items()}
        elif kind == "drop_one":
            amounts.pop(next(iter(sorted(amounts)), None), None)
        elif kind == "double":
            amounts = {rule: value * 2.0 for rule, value in amounts.items()}
        void = () if kind == "drop_void" else output.void_rules
        total = sum(amounts.values())
        return with_output(output, {"amounts": amounts, "total": total}, amount=total, void_rules=void)

    pipeline = pipeline_with(penalty=penalty_variant(attack))
    branch = frozenset(branch)
    reference = default_pipeline().score_peer(
        ledger_of(spec), peer="r", episode=1, branch_void=branch
    )
    try:
        card = run(pipeline, spec, branch_void=branch)
    except ComplianceGuardViolation as error:
        # refused: the stage weakened a floor or forgot a void, and says which
        assert error.code in {"C002", "C004", "C003"}
        assert kind in {"zero", "scale", "drop_void", "drop_one"}
        return
    # accepted: the peer is exactly as void as the reference says ...
    assert set(card.violated_non_offsettable) == set(reference.violated_non_offsettable)
    assert card.is_void == reference.is_void
    assert set(branch) <= set(card.violated_non_offsettable)
    # ... and the void value is no kinder than the reference's
    assert card.floor <= reference.floor + TOL
    if reference.is_void:
        assert card.final_score <= card.continuous.reward.total - abs(reference.floor) + TOL \
            or card.final_score <= max(-1.0e4, reference.final_score) + TOL


@PROFILE
@given(spec=ledgers, branch=st.sets(st.sampled_from(NO_IDS)))
def test_builtin_stages_are_never_flagged_when_the_guard_does_not_trust_them(spec, branch):
    """No false positives: the reference *is* the built-in, so inspecting built-ins finds nothing."""
    pipeline = default_pipeline(COMPLIANCE={"TRUST_BUILTIN": False})
    card = run(pipeline, spec, branch_void=frozenset(branch))
    assert "compliance_guard" not in card.metadata


@PROFILE
@given(spec=ledgers, lam=st.floats(min_value=0.0, max_value=20.0), floor=st.floats(min_value=0.5, max_value=30.0))
def test_a_floor_lambda_can_never_be_configured_below_the_minimum(spec, lam, floor):
    config = {
        **settings_copy()["SCORING"],
        "PENALTY_LAMBDAS": {rule: lam for rule in NO_IDS},
        "MIN_FLOOR_LAMBDA": floor,
    }
    stage = PenaltyScorer(config=config)
    pipeline = pipeline_with(penalty=stage, settings=settings_copy(SCORING=config))
    card = run(pipeline, spec)
    for rule, amount in card.continuous.penalty.amounts.items():
        if rule in effective_non_offsettable(config["NON_OFFSETTABLE"]):
            events = [e for e in ledger_of(spec).penalties() if e.rule_id == rule]
            assert amount >= max(lam, floor) * sum(e.magnitude for e in events) - TOL
