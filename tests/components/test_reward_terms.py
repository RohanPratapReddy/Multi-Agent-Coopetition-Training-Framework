"""Reward terms and ``composed_reward`` (document 17 section 6.3)."""

from __future__ import annotations

import math
import random

import pytest

from mactf import conf
from mactf.components import ComponentKind, register
from mactf.exceptions import (
    ComplianceError,
    ComplianceGuardViolation,
    ComponentBuildError,
    ComponentImportDenied,
    ImproperlyConfigured,
    ValidationError,
)
from mactf.rules.catalog import ALL_RULES, REWARDS
from mactf.scoring.base import ScoreContext, ScoringKind
from mactf.scoring.ledger import Ledger
from mactf.scoring.reward_scorer import BUCKETS, RewardScorer
from mactf.scoring.reward_terms import (
    REWARD_IDS,
    CappedTerm,
    ComposedRewardScorer,
    RewardTerm,
    SeverityScaledTerm,
)

from scoring_helpers import default_pipeline, ledger_of, pipeline_with, settings_copy

CONFIG = settings_copy()["SCORING"]
SEVERITY = settings_copy()["SEVERITY"]["WEIGHTS"]


def ctx_for(ledger, **extras):
    return ScoreContext(
        peer="a", episode=1, ledger=ledger, config=CONFIG, severity_weights=SEVERITY, extras=extras
    )


def random_ledger(rng):
    ledger = Ledger(peer="a")
    for rule in rng.sample(list(REWARDS), rng.randint(0, 6)):
        severity = rng.choice([None, "S1", "S2", "S3", "S4"])
        ledger.record(rule.id, magnitude=round(rng.uniform(0.0, 5.0), 4), severity=severity)
    return ledger


# ------------------------------------------------------------------ equivalence
def test_composed_matches_builtin():
    """One ``severity_scaled`` term at weight 1 is the built-in stage, for every ledger."""
    rng = random.Random(11)
    builtin = RewardScorer(config=CONFIG)
    composed = ComposedRewardScorer(config=CONFIG)
    assert [label for label, _w, _t in composed.terms] == ["severity_scaled"]
    for _ in range(200):
        ledger = random_ledger(rng)
        expected = builtin.score(ctx_for(ledger))
        actual = composed.score(ctx_for(ledger))
        assert actual.amount == expected.amount
        assert actual.rules == expected.rules and actual.kind is ScoringKind.REWARD
        extra = dict(actual.breakdown)
        terms = extra.pop("terms")
        assert extra == dict(expected.breakdown)
        assert terms == {
            "severity_scaled": pytest.approx(sum(expected.breakdown[b] for b in BUCKETS))
        }
        assert actual.metadata["ignored_rules"] == expected.metadata["ignored_rules"]


def test_composed_matches_builtin_through_the_whole_pipeline():
    spec = {"R1": 2.0, "R3": 1.5, "R10": 3.0, "P6": 1.0, "P3": 1.0}
    plain = default_pipeline()
    composed = default_pipeline(SCORING={"STAGES": {"reward": "composed_reward"}})
    a = plain.score_peer(ledger_of(spec), peer="a", episode=1)
    b = composed.score_peer(ledger_of(spec), peer="a", episode=1)
    assert (a.final_score, a.floor) == (b.final_score, b.floor)
    assert a.continuous.reward.as_dict() == b.continuous.reward.as_dict()
    assert "compliance_guard" not in b.metadata


# --------------------------------------------------------------------- the terms
def test_capped_limits_the_magnitude_before_scaling():
    ledger = ledger_of({"R1": 10.0})
    capped = ComposedRewardScorer(config=CONFIG, terms=[{"type": "capped", "params": {"limit": 2.0}}])
    base = RewardScorer(config=CONFIG).score(ctx_for(ledger))
    out = capped.score(ctx_for(ledger))
    assert out.amount == pytest.approx(base.amount * 0.2)
    flat = capped.score(ctx_for(ledger_of({"R1": 1.0})))
    assert flat.amount == RewardScorer(config=CONFIG).score(ctx_for(ledger_of({"R1": 1.0}))).amount


def test_terms_are_weighted_and_summed_and_reported_per_term():
    ledger = ledger_of({"R1": 4.0})
    out = ComposedRewardScorer(
        config=CONFIG,
        terms=[
            {"type": "severity_scaled", "weight": 1.0},
            {"type": "capped", "weight": 0.5, "params": {"limit": 1.0}},
        ],
    ).score(ctx_for(ledger))
    base = RewardScorer(config=CONFIG).score(ctx_for(ledger))
    scale = base.amount / (4.0 * base.breakdown["weights"]["r_task"])
    assert out.breakdown["terms"]["severity_scaled"] == pytest.approx(4.0 * scale)
    assert out.breakdown["terms"]["capped"] == pytest.approx(0.5 * 1.0 * scale)
    assert out.breakdown["r_task"] == pytest.approx(4.5 * scale)


def test_a_term_only_applies_to_its_rules():
    ledger = ledger_of({"R1": 2.0, "R10": 2.0})
    only_r10 = ComposedRewardScorer(
        config=CONFIG, terms=[{"type": "severity_scaled", "params": {"rules": ["R10"]}}]
    ).score(ctx_for(ledger))
    assert only_r10.rules == ("R10",) and set(only_r10.breakdown["by_rule"]) == {"R10"}


def test_duplicate_terms_get_distinct_labels():
    stage = ComposedRewardScorer(
        config=CONFIG, terms=[{"type": "capped", "params": {"limit": 1}}, {"type": "capped", "params": {"limit": 2}}]
    )
    assert [label for label, _w, _t in stage.terms] == ["capped", "capped#2"]


def test_a_term_instance_is_accepted_and_needs_not_subclass_reward_term():
    class Half:
        rule_ids = frozenset({"R1"})

        def contribution(self, event, ctx):
            return event.magnitude * 0.5

    stage = ComposedRewardScorer(config=CONFIG, terms=[Half()])
    assert stage.score(ctx_for(ledger_of({"R1": 2.0}))).breakdown["r_task"] == 1.0
    assert stage.describe()["terms"][0]["rule_ids"] == ["R1"]  # a derived description
    with pytest.raises(ImproperlyConfigured, match="neither a spec nor a RewardTerm"):
        ComposedRewardScorer(config=CONFIG, terms=[object()])


def test_admitted_only_skips_events_the_controlling_tier_did_not_admit():
    ledger = ledger_of({"R1": 2.0, "R3": 1.0})
    ctx = ctx_for(ledger, allowed_rules=frozenset({"R1"}))
    default = ComposedRewardScorer(config=CONFIG).score(ctx)
    strict = ComposedRewardScorer(config=CONFIG, admitted_only=True).score(ctx)
    assert set(default.rules) == {"R1", "R3"} and strict.rules == ("R1",)
    assert strict.metadata["skipped_not_admitted"] == ("R3",)
    # no allowed set (layered mode): nothing is skipped
    assert ComposedRewardScorer(config=CONFIG, admitted_only=True).score(ctx_for(ledger)).rules == default.rules


# ------------------------------------------------------------------ refusals
@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf"), "heavy", None])
def test_a_bad_weight_is_refused_at_build(weight):
    with pytest.raises(ImproperlyConfigured, match="weight"):
        ComposedRewardScorer(config=CONFIG, terms=[{"type": "severity_scaled", "weight": weight}])


def test_at_least_one_term_is_required():
    with pytest.raises(ImproperlyConfigured, match="at least one term"):
        ComposedRewardScorer(config=CONFIG, terms=[])


@pytest.mark.parametrize("rules", [["P1"], ["R99"], []])
def test_a_term_may_only_name_reward_rules(rules):
    with pytest.raises(ValidationError):
        SeverityScaledTerm(rules=rules)
    # a ``ValidationError`` raised by a component itself reaches the caller unchanged
    with pytest.raises(ValidationError, match="reward rules|empty"):
        ComposedRewardScorer(config=CONFIG, terms=[{"type": "severity_scaled", "params": {"rules": rules}}])


@pytest.mark.parametrize("limit", [-1.0, float("nan"), float("inf")])
def test_capped_needs_a_finite_non_negative_limit(limit):
    with pytest.raises(ValidationError, match="limit"):
        CappedTerm(limit=limit)


def test_an_unknown_or_misspelt_term_is_refused_naming_the_accepted_parameters():
    from mactf.exceptions import ComponentMissing, ComponentParamError

    with pytest.raises(ComponentMissing, match="no_such"):
        ComposedRewardScorer(config=CONFIG, terms=[{"type": "no_such"}])
    with pytest.raises(ComponentParamError, match="limt"):
        ComposedRewardScorer(config=CONFIG, terms=[{"type": "capped", "params": {"limt": 1}}])


@pytest.mark.parametrize("bad", [-0.5, float("nan"), float("inf")])
def test_a_term_returning_a_bad_number_stops_the_stage_at_the_first_event(bad):
    class Bad:
        rule_ids = frozenset({"R1"})

        def contribution(self, event, ctx):
            return bad

    with pytest.raises(ComplianceError, match="finite and non-negative"):
        ComposedRewardScorer(config=CONFIG, terms=[Bad()]).score(ctx_for(ledger_of({"R1": 1.0})))


def test_a_term_that_cannot_be_imported_on_an_untrusted_surface_is_refused():
    with pytest.raises(ComponentImportDenied):
        ComposedRewardScorer(config=CONFIG, terms=[{"import": "os.path:join"}], trusted=False)


def test_a_registered_term_can_be_used_by_name(component_overlay):
    class Sqrt(RewardTerm):
        name = "sqrt_magnitude"

        def __init__(self, *, scale: float = 1.0):
            if scale < 0:
                raise ValueError("scale must be >= 0")
            self.scale = scale

        def contribution(self, event, ctx):
            return self.scale * math.sqrt(event.magnitude)

    register(ComponentKind.REWARD_TERM, "sqrt_magnitude", Sqrt, scope="run")
    stage = ComposedRewardScorer(
        config=CONFIG, terms=[{"type": "sqrt_magnitude", "params": {"scale": 2.0}, "weight": 0.5}]
    )
    out = stage.score(ctx_for(ledger_of({"R1": 4.0})))
    assert out.breakdown["terms"] == {"sqrt_magnitude": pytest.approx(2.0)}
    with pytest.raises(ComponentBuildError, match="scale must be >= 0"):  # a plain ValueError: E008
        ComposedRewardScorer(config=CONFIG, terms=[{"type": "sqrt_magnitude", "params": {"scale": -1}}])


def test_the_term_contract_is_checked_at_registration(component_overlay):
    from mactf.exceptions import ComponentContractError

    class NoRules:
        def contribution(self, event, ctx):
            return 0.0

    class NoMethod:
        rule_ids = frozenset({"R1"})

    for cls in (NoRules, NoMethod):
        with pytest.raises(ComponentContractError):
            register(ComponentKind.REWARD_TERM, "broken", cls, scope="run")


# ---------------------------------------------------------- diagnostics, guard, pipeline
def test_validate_reports_dead_configurations():
    all_zero = ComposedRewardScorer(config=CONFIG, terms=[{"type": "severity_scaled", "weight": 0.0}])
    diagnostics = all_zero.validate({})
    assert [d.code for d in diagnostics] == ["rewards.all_terms_zero_weight"]
    assert diagnostics[0].level == "warning" and "zero_reward" in diagnostics[0].hint
    assert ComposedRewardScorer(config=CONFIG).validate({}) == []
    # the inherited REWARD_WEIGHTS checks still run
    codes = [d.code for d in ComposedRewardScorer(config=CONFIG).validate({"REWARD_WEIGHTS": {"r_task": -1}})]
    assert codes == ["rewards.negative_weight"]


def test_describe_lists_the_terms():
    payload = ComposedRewardScorer(
        config=CONFIG, terms=[{"type": "capped", "weight": 0.5, "params": {"limit": 2.0, "rules": ["R10"]}}]
    ).describe()
    assert payload["terms"] == [
        {"label": "capped", "weight": 0.5, "name": "capped", "rule_ids": ["R10"], "limit": 2.0}
    ]
    assert payload["admitted_only"] is False and payload["kind"] == "reward"


def test_a_composed_stage_that_pays_too_much_is_the_guards_to_refuse():
    """The terms are non-negative, the weights too; a huge weight still meets the ceiling (C008)."""
    pipeline = default_pipeline(
        SCORING={"STAGES": {"reward": {"type": "composed_reward", "terms": [{"type": "severity_scaled", "weight": 1.0e9}]}}}
    )
    with pytest.raises(ComplianceGuardViolation) as excinfo:
        pipeline.score_peer(ledger_of({"R1": 5.0}), peer="a", episode=1)
    assert excinfo.value.code == "C008"
    fallback = default_pipeline(
        SCORING={"STAGES": {"reward": {"type": "composed_reward", "terms": [{"type": "severity_scaled", "weight": 1.0e9}]}}},
        COMPLIANCE={"GUARD_MODE": "fallback"},
    )
    card = fallback.score_peer(ledger_of({"R1": 5.0}), peer="a", episode=1)
    assert card.metadata["compliance_guard"]["substituted"] == ["reward"]
    assert card.continuous.reward.total == default_pipeline().score_peer(
        ledger_of({"R1": 5.0}), peer="a", episode=1).continuous.reward.total


def test_a_stage_from_a_yaml_style_spec_builds_with_nested_terms():
    pipeline = default_pipeline(SCORING={"STAGES": {"reward": {
        "type": "composed_reward",
        "terms": [
            {"type": "severity_scaled", "weight": 1.0},
            {"type": "capped", "weight": 0.5, "params": {"limit": 2.0, "rules": ["R10", "R11"]}},
        ],
    }}})
    stage = pipeline.get(ScoringKind.REWARD)
    assert isinstance(stage, ComposedRewardScorer) and len(stage.terms) == 2
    card = pipeline.score_peer(ledger_of({"R10": 3.0}), peer="a", episode=1)
    assert card.continuous.reward.total > default_pipeline().score_peer(
        ledger_of({"R10": 3.0}), peer="a", episode=1).continuous.reward.total


def test_reward_ids_are_exactly_the_catalogs_reward_rules():
    assert REWARD_IDS == frozenset(rule.id for rule in ALL_RULES if rule.is_reward)
