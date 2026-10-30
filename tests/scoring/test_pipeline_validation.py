"""Chain validation: duplicate kind/order, missing kind, bad order, from_spec."""

from __future__ import annotations

import pytest

from mactf.exceptions import ImproperlyConfigured, ValidationError
from mactf.scoring.base import ScoreContext, Scorer, ScorerOutput, ScoringKind
from mactf.scoring.ledger import Ledger
from mactf.scoring.penalty import PenaltyScorer
from mactf.scoring.pipeline import ScoringPipeline, build_stages
from mactf.scoring.registry import (
    SCORER_TYPES,
    build_scorer_from_spec,
    register_scorer,
    scorer_capabilities,
)
from mactf.scoring.reward_scorer import RewardScorer
from mactf.scoring.warning import WarningPolicy
from mactf.scoring.warning_scorer import WarningScorer


def _pipeline(stages, score_config, severity_weights):
    return ScoringPipeline(
        stages,
        score_config=score_config,
        severity_weights=severity_weights,
        warning_policy=WarningPolicy(),
    )


def test_default_construction_succeeds(score_config, severity_weights):
    pipeline = _pipeline(None, score_config, severity_weights)
    assert [stage.kind for stage in pipeline.chain] == [
        ScoringKind.WARNING,
        ScoringKind.REWARD,
        ScoringKind.PENALTY,
    ]


def test_chain_is_sorted_by_order(score_config, severity_weights):
    pipeline = _pipeline(
        [
            PenaltyScorer(config=score_config),
            RewardScorer(config=score_config),
            WarningScorer(config=score_config),
        ],
        score_config,
        severity_weights,
    )
    assert [stage.order for stage in pipeline.chain] == [5, 10, 20]


def test_empty_chain_is_rejected(score_config, severity_weights):
    with pytest.raises(ImproperlyConfigured, match="at least one stage"):
        _pipeline([], score_config, severity_weights)


def test_duplicate_kind_is_rejected_naming_both(score_config, severity_weights):
    with pytest.raises(ImproperlyConfigured) as excinfo:
        _pipeline(
            [
                WarningScorer(config=score_config),
                RewardScorer(config=score_config),
                RewardScorer(config=score_config),
            ],
            score_config,
            severity_weights,
        )
    message = str(excinfo.value)
    assert "reward" in message
    assert "RewardScorer" in message


def test_duplicate_order_is_rejected(score_config, severity_weights):
    class Clashing(RewardScorer):
        order = 20

    with pytest.raises(ImproperlyConfigured, match="both declare"):
        _pipeline(
            [
                WarningScorer(config=score_config),
                Clashing(config=score_config),
                PenaltyScorer(config=score_config),
            ],
            score_config,
            severity_weights,
        )


def test_missing_kind_is_named(score_config, severity_weights):
    with pytest.raises(ImproperlyConfigured, match="penalty"):
        _pipeline(
            [WarningScorer(config=score_config), RewardScorer(config=score_config)],
            score_config,
            severity_weights,
        )


def test_negative_order_is_rejected(score_config, severity_weights):
    class Backwards(RewardScorer):
        order = -1

    with pytest.raises(ImproperlyConfigured, match="negative order"):
        _pipeline(
            [
                WarningScorer(config=score_config),
                Backwards(config=score_config),
                PenaltyScorer(config=score_config),
            ],
            score_config,
            severity_weights,
        )


def test_warning_stage_must_precede_reward_and_penalty(score_config, severity_weights):
    class LateWarning(WarningScorer):
        order = 30

    with pytest.raises(ImproperlyConfigured, match="must run before"):
        _pipeline(
            [
                LateWarning(config=score_config),
                RewardScorer(config=score_config),
                PenaltyScorer(config=score_config),
            ],
            score_config,
            severity_weights,
        )


def test_a_non_scorer_is_rejected(score_config, severity_weights):
    with pytest.raises(ImproperlyConfigured, match="must be a Scorer subclass"):
        _pipeline(
            [WarningScorer(config=score_config), object(), PenaltyScorer(config=score_config)],
            score_config,
            severity_weights,
        )


def test_add_validates_and_rolls_back(score_config, severity_weights):
    pipeline = _pipeline(None, score_config, severity_weights)

    class Extra(RewardScorer):
        kind = ScoringKind.PIPELINE

    before = pipeline.chain
    with pytest.raises(ImproperlyConfigured):
        pipeline.add(Extra(config=score_config))
    assert pipeline.chain == before


def test_remove_refuses_to_break_the_chain(score_config, severity_weights):
    pipeline = _pipeline(None, score_config, severity_weights)
    with pytest.raises(ImproperlyConfigured, match="missing required stage"):
        pipeline.remove(ScoringKind.PENALTY)
    # The failed removal must leave the chain intact.
    assert [stage.kind for stage in pipeline.chain] == [
        ScoringKind.WARNING,
        ScoringKind.REWARD,
        ScoringKind.PENALTY,
    ]


def test_remove_of_an_absent_kind_is_reported(score_config, severity_weights):
    class NothingReward(RewardScorer):
        pass

    pipeline = _pipeline(None, score_config, severity_weights)
    with pytest.raises(ImproperlyConfigured, match="no stage of kind"):
        pipeline.remove(ScoringKind.HIERARCHY)


def test_get_returns_the_stage_of_that_kind(score_config, severity_weights):
    pipeline = _pipeline(None, score_config, severity_weights)
    assert isinstance(pipeline.get(ScoringKind.WARNING), WarningScorer)
    assert isinstance(pipeline.get(ScoringKind.REWARD), RewardScorer)
    assert isinstance(pipeline.get(ScoringKind.PENALTY), PenaltyScorer)
    with pytest.raises(ImproperlyConfigured):
        pipeline.get(ScoringKind.HIERARCHY)


def test_build_stages_returns_the_three_defaults():
    stages = build_stages(score_config={}, severity_weights={})
    assert [stage.order for stage in stages] == [5, 10, 20]


def test_registry_lists_the_three_per_node_stages_plus_hierarchy():
    # document 17 sections 6.3 and 6.8 add three built-in stage names
    assert set(SCORER_TYPES) == {
        "reward", "penalty", "warning", "hierarchy", "composed_reward", "zero_reward", "no_warnings",
    }
    assert SCORER_TYPES["penalty"] is PenaltyScorer
    # ``hierarchy`` resolves so ``ScorerSpec(type="hierarchy")`` is valid in the
    # API, but it is root-only: a per-node chain may not include it.
    assert SCORER_TYPES["hierarchy"].kind.value == "hierarchy"


def test_build_scorer_from_spec_round_trips():
    stage = build_scorer_from_spec({"type": "reward", "config": {"REWARD_WEIGHTS": {}}})
    assert isinstance(stage, RewardScorer)
    assert stage.config == {"REWARD_WEIGHTS": {}}


def test_unknown_type_is_rejected_listing_the_accepted_ones():
    with pytest.raises(ValidationError) as excinfo:
        build_scorer_from_spec({"type": "rewards"})
    message = str(excinfo.value)
    assert "rewards" in message
    assert "reward" in message and "penalty" in message and "warning" in message


def test_missing_type_is_rejected():
    with pytest.raises(ValidationError, match="requires a 'type' key"):
        build_scorer_from_spec({"config": {}})


def test_unknown_keyword_is_rejected_listing_the_accepted_ones():
    with pytest.raises(ValidationError) as excinfo:
        build_scorer_from_spec({"type": "penalty", "lamdbas": {}})
    message = str(excinfo.value)
    assert "lamdbas" in message
    assert "config" in message


def test_register_scorer_refuses_a_silent_override(score_config):
    class Custom(RewardScorer):
        pass

    with pytest.raises(ValidationError, match="already registered"):
        register_scorer("reward", Custom)
    try:
        register_scorer("custom_probe", Custom)
        assert SCORER_TYPES["custom_probe"] is Custom
        assert build_scorer_from_spec({"type": "custom_probe"}) is not None
    finally:
        SCORER_TYPES.pop("custom_probe", None)


def test_register_scorer_rejects_a_non_scorer():
    with pytest.raises(ValidationError, match="not a Scorer subclass"):
        register_scorer("bogus", object)  # type: ignore[arg-type]


def test_capabilities_describe_every_registered_stage():
    payload = {item["type"]: item for item in scorer_capabilities()}
    assert set(payload) == set(SCORER_TYPES)
    assert payload["warning"]["order"] == 5
    assert payload["reward"]["kind"] == "reward"
    assert payload["penalty"]["class"] == "PenaltyScorer"
    assert all(item["requires"] for item in payload.values())


def test_spec_validation_is_strict_about_kwargs_for_every_stage():
    for name in ("reward", "penalty", "warning"):
        with pytest.raises(ValidationError, match="does not accept"):
            build_scorer_from_spec({"type": name, "nonexistent": 1})


def test_state_dict_round_trip_preserves_the_chain(score_config, severity_weights):
    pipeline = _pipeline(None, score_config, severity_weights)
    ledger = Ledger(peer="a")
    ledger.record("P6", magnitude=1.0, severity="S1")
    pipeline.score_peer(ledger, peer="a", episode=1)

    payload = pipeline.state_dict()
    assert set(payload["stages"]) == {"warning", "reward", "penalty"}

    restored = _pipeline(None, score_config, severity_weights)
    restored.load_state_dict(payload)
    assert restored.warning_scorer.registry_for("a").count("a", "P6") == 1


def test_describe_includes_the_chain(score_config, severity_weights):
    payload = _pipeline(None, score_config, severity_weights).describe()
    assert payload["kind"] == "pipeline"
    assert [item["kind"] for item in payload["chain"]] == ["warning", "reward", "penalty"]


def test_pipeline_is_itself_a_scorer(score_config, severity_weights):
    pipeline = _pipeline(None, score_config, severity_weights)
    assert isinstance(pipeline, Scorer)


def test_score_reports_every_stage(score_config, severity_weights):
    pipeline = _pipeline(None, score_config, severity_weights)
    ledger = Ledger(peer="a")
    ledger.record("R1", magnitude=1.0)
    ctx = ScoreContext(peer="a", episode=1, ledger=ledger)
    output = pipeline.score(ctx)
    assert isinstance(output, ScorerOutput)
    assert output.kind is ScoringKind.PIPELINE
    assert set(output.breakdown["stages"]) == {"warning", "reward", "penalty"}
    assert output.amount == pytest.approx(output.breakdown["net"])
