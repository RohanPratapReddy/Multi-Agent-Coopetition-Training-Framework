"""``validate_stage_chain`` against ``ScoringPipeline._validate_chain`` (document 17 section 6.7)."""

from __future__ import annotations

import itertools
import re

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mactf.exceptions import ImproperlyConfigured
from mactf.scoring.base import ScoringKind
from mactf.scoring.chain_validation import (
    PROVIDED_CONTEXT_KEYS,
    REQUIRED_KINDS,
    validate_stage_chain,
)
from mactf.scoring.hierarchy import HierarchicalScorer
from mactf.scoring.null_stages import NoWarningScorer, ZeroRewardScorer
from mactf.scoring.penalty import PenaltyScorer
from mactf.scoring.pipeline import ScoringPipeline
from mactf.scoring.reward_scorer import RewardScorer
from mactf.scoring.warning_scorer import WarningScorer

from scoring_helpers import settings_copy

#: the exception ``_validate_chain`` (or ``_add_unchecked``) raises for each chain code
RAISES = {
    "chain.not_a_scorer": r"must be a Scorer subclass",
    "chain.negative_order": r"negative order",
    "chain.hierarchy_nested": r"cannot be a stage of a per-node",
    "chain.pipeline_nested": r"cannot be a stage of another",
    "chain.duplicate_kind": r"two scorers with kind",
    "chain.order_collision": r"both declare order=",
    "chain.missing_stage": r"missing required|at least one stage",
    "chain.warning_order": r"must run before",
}


def _ordered(base, order):
    return type(f"{base.__name__}At{order}".replace("-", "Neg"), (base,), {"order": order})


STAGES = {
    "W": lambda: WarningScorer(),
    "R": lambda: RewardScorer(),
    "P": lambda: PenaltyScorer(),
    "R2": lambda: ZeroRewardScorer(),
    "W2": lambda: NoWarningScorer(),
    "P2": lambda: _ordered(PenaltyScorer, 21)(),
    "R_neg": lambda: _ordered(RewardScorer, -3)(),
    "R_at_5": lambda: _ordered(RewardScorer, 5)(),
    "P_at_10": lambda: _ordered(PenaltyScorer, 10)(),
    "R_early": lambda: _ordered(RewardScorer, 2)(),
    "P_same_as_W": lambda: _ordered(PenaltyScorer, 5)(),
    "W_late": lambda: _ordered(WarningScorer, 15)(),
    "H": lambda: HierarchicalScorer(),
    "PIPE": lambda: ScoringPipeline(),
    "junk": lambda: object(),
}


def first_structural(diagnostics):
    for item in diagnostics:
        if item.level == "error" and item.code.startswith("chain.") and item.code != "chain.floor_lambda":
            return item
    return None


def build(names):
    stages = [STAGES[name]() for name in names]
    try:
        ScoringPipeline(stages)
    except ImproperlyConfigured as error:
        return stages, error
    return stages, None


def assert_agrees(names):
    stages, raised = build(names)
    first = first_structural(validate_stage_chain(stages))
    if raised is None:
        assert first is None, f"{names}: validate says {first.code}, the pipeline accepted it"
    else:
        assert first is not None, f"{names}: the pipeline raised {raised} but validate found nothing"
        assert re.search(RAISES[first.code], str(raised)), (
            f"{names}: validate's first error is {first.code}, the pipeline raised: {raised}"
        )


def test_chain_validation_agrees():
    """For every chain, the first error validate reports is the exception the pipeline raises."""
    names = list(STAGES)
    seen = set()
    chains = [[]] + [list(c) for n in (1, 2, 3) for c in itertools.permutations(names, n)]
    for chain in chains:
        assert_agrees(chain)
        stages, raised = build(chain)
        if raised is not None:
            seen.add(first_structural(validate_stage_chain(stages)).code)
    # the catalogue exercises every code, so the agreement is not vacuous
    assert seen == set(RAISES), set(RAISES) - seen
    assert_agrees(["W", "R", "P"])  # and the one valid chain agrees by both being silent


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.sampled_from(sorted(STAGES)), min_size=0, max_size=6))
def test_chain_validation_agrees_on_longer_random_chains(names):
    assert_agrees(names)


def test_a_valid_chain_has_no_errors():
    stages = [WarningScorer(), RewardScorer(), PenaltyScorer()]
    assert [d for d in validate_stage_chain(stages) if d.level == "error"] == []
    assert [d for d in validate_stage_chain([WarningScorer(), ZeroRewardScorer(), PenaltyScorer()])
            if d.level == "error"] == []
    assert not [d for d in validate_stage_chain([NoWarningScorer(), RewardScorer(), PenaltyScorer()])
                if d.level == "error"]


def test_every_problem_is_reported_not_only_the_first():
    stages = [WarningScorer(), _ordered(RewardScorer, 5)(), RewardScorer(), HierarchicalScorer()]
    codes = [d.code for d in validate_stage_chain(stages)]
    assert "chain.hierarchy_nested" in codes and "chain.duplicate_kind" in codes
    assert "chain.order_collision" in codes and "chain.missing_stage" in codes
    assert "chain.warning_order" in codes
    # an empty chain misses all three kinds
    assert [d.code for d in validate_stage_chain([])] == ["chain.missing_stage"] * 3
    # each carries a human message; the missing-stage hint names the way to turn a kind off
    missing = [d for d in validate_stage_chain([]) if d.code == "chain.missing_stage"]
    assert all("zero_reward" in d.hint for d in missing)


def test_stage_level_diagnostics_follow_the_structural_ones():
    class Fussy(PenaltyScorer):
        def validate(self, settings):
            return [*super().validate(settings), *PenaltyScorer.validate(self, settings)]

    diagnostics = validate_stage_chain(
        [WarningScorer(), RewardScorer(), RewardScorer(), Fussy()],
        {"PENALTY_LAMBDAS": {"P3": -1.0}},
    )
    codes = [d.code for d in diagnostics]
    assert codes[0] == "chain.duplicate_kind"
    assert "penalties.negative_lambda" in codes
    assert codes.index("chain.duplicate_kind") < codes.index("penalties.negative_lambda")


def test_unknown_requirements_are_warnings_not_errors():
    class Needy(RewardScorer):
        requires = frozenset({"ledger", "weather"})

    diagnostics = validate_stage_chain([WarningScorer(), Needy(), PenaltyScorer()])
    needy = [d for d in diagnostics if d.code == "chain.unknown_requirement"]
    assert len(needy) == 1 and needy[0].level == "warning" and "weather" in needy[0].message
    assert "ledger" in PROVIDED_CONTEXT_KEYS and "weather" not in PROVIDED_CONTEXT_KEYS
    assert first_structural(diagnostics) is None


def test_the_lambda_floor_is_checked_over_the_whole_effective_set():
    config = settings_copy()["SCORING"]
    ok = validate_stage_chain([WarningScorer(), RewardScorer(), PenaltyScorer()], {"SCORING": config})
    assert not [d for d in ok if d.code == "chain.floor_lambda"]
    config["PENALTY_LAMBDAS"] = {**config["PENALTY_LAMBDAS"], "P17": 0.5}
    settings_ = {"SCORING": config, "COMPLIANCE": {"MIN_FLOOR_LAMBDA": 1.0}}
    bad = [d for d in validate_stage_chain([WarningScorer(), RewardScorer(), PenaltyScorer()], settings_)
           if d.code == "chain.floor_lambda"]
    assert len(bad) == 1 and "P17" in bad[0].message
    # a widened set is checked too, and narrowing the configured set does not hide a catalog rule
    config["NON_OFFSETTABLE"] = ["P3"]
    config["PENALTY_LAMBDAS"] = {**config["PENALTY_LAMBDAS"], "P3": 0.5}
    messages = " ".join(
        d.message for d in validate_stage_chain([WarningScorer(), RewardScorer(), PenaltyScorer()],
                                                {"SCORING": config})
        if d.code == "chain.floor_lambda"
    )
    assert "P3" in messages and "P17" in messages


def test_required_kinds_are_the_pipelines():
    assert tuple(REQUIRED_KINDS) == ScoringPipeline.REQUIRED_KINDS == (
        ScoringKind.WARNING, ScoringKind.REWARD, ScoringKind.PENALTY,
    )


def test_validate_does_not_raise_on_garbage():
    out = validate_stage_chain([None, 3, "x"], None)
    assert [d.code for d in out][:3] == ["chain.not_a_scorer"] * 3
