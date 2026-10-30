"""The scoring stage protocol: frozen contexts, kinds, ordering, validation."""

from __future__ import annotations

import dataclasses

import pytest

from mactf.exceptions import ImproperlyConfigured
from mactf.scoring.base import (
    ScoreContext,
    Scorer,
    ScorerDiagnostic,
    ScorerOutput,
    ScoringKind,
    WarningMutation,
    severity_fraction,
)
from mactf.scoring.ledger import Ledger
from mactf.scoring.penalty import PenaltyScorer
from mactf.scoring.pipeline import ScoringPipeline
from mactf.scoring.reward_scorer import RewardScorer
from mactf.scoring.warning_scorer import WarningScorer


def _ctx(**changes) -> ScoreContext:
    base = ScoreContext(peer="a", episode=1, ledger=Ledger(peer="a"))
    return base.evolve(**changes) if changes else base


def test_scoring_kinds_are_strings_for_the_wire():
    assert ScoringKind.REWARD.value == "reward"
    assert ScoringKind("penalty") is ScoringKind.PENALTY
    assert {kind.value for kind in ScoringKind} == {
        "reward",
        "penalty",
        "warning",
        "pipeline",
        "hierarchy",
    }


def test_score_context_is_frozen():
    ctx = _ctx()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.peer = "b"  # type: ignore[misc]


def test_evolve_returns_a_new_context_without_touching_the_original():
    ctx = _ctx()
    evolved = ctx.evolve(episode=7, warning_total=2.5)
    assert ctx.episode == 1 and ctx.warning_total == 0.0
    assert evolved.episode == 7 and evolved.warning_total == 2.5
    assert evolved.ledger is ctx.ledger


def test_warning_only_rules_excludes_escalated():
    ctx = _ctx(
        warning_events=("P6", "P7", "P9"),
        escalated_rules=frozenset({"P9"}),
    )
    assert ctx.warning_only_rules == frozenset({"P6", "P7"})


def test_allowed_rules_is_none_when_unset():
    assert _ctx().allowed_rules is None
    assert _ctx(extras={"allowed_rules": frozenset({"R1"})}).allowed_rules == frozenset({"R1"})


def test_warning_mutation_serializes_severity_as_a_label():
    mutation = WarningMutation(
        peer="a", rule_id="P6", episode=2, action="issue", severity="S1"  # type: ignore[arg-type]
    )
    from mactf.rules.severity import Severity

    mutation = dataclasses.replace(mutation, severity=Severity.MINOR)
    assert mutation.as_dict()["severity"] == "S1"


def test_scorer_output_as_dict_is_json_shaped():
    output = ScorerOutput(
        kind=ScoringKind.REWARD,
        amount=1.5,
        breakdown={"r_task": 1.5},
        rules=("R1",),
        void_rules=("P1",),
    )
    payload = output.as_dict()
    assert payload == {
        "kind": "reward",
        "amount": 1.5,
        "breakdown": {"r_task": 1.5},
        "rules": ["R1"],
        "metadata": {},
        "void_rules": ["P1"],
    }


def test_scorer_diagnostic_as_dict():
    diagnostic = ScorerDiagnostic(level="error", code="x", message="bad", hint="fix it")
    assert diagnostic.as_dict()["hint"] == "fix it"


def test_scorer_is_abstract():
    with pytest.raises(TypeError):
        Scorer()  # type: ignore[abstract]


def test_the_three_default_orders_are_warning_reward_penalty():
    assert WarningScorer.order == 5
    assert RewardScorer.order == 10
    assert PenaltyScorer.order == 20
    assert WarningScorer.order < RewardScorer.order < PenaltyScorer.order


def test_pipeline_sorts_its_chain_by_order():
    pipeline = ScoringPipeline()
    orders = [stage.order for stage in pipeline.chain]
    assert orders == sorted(orders)


def test_describe_reports_kind_class_order_and_requires():
    payload = RewardScorer().describe()
    assert payload["kind"] == "reward"
    assert payload["class"] == "RewardScorer"
    assert payload["order"] == 10
    assert payload["requires"] == ["config", "ledger"]


def test_state_dict_round_trips_config():
    stage = RewardScorer(config={"REWARD_WEIGHTS": {"r_task": 2.0}})
    payload = stage.state_dict()
    assert payload["class"] == "RewardScorer"

    restored = RewardScorer()
    restored.load_state_dict(payload)
    assert restored.config == {"REWARD_WEIGHTS": {"r_task": 2.0}}


def test_load_state_dict_rejects_a_foreign_class():
    with pytest.raises(ValueError, match="cannot load state written by"):
        RewardScorer().load_state_dict({"class": "PenaltyScorer", "config": {}})


def test_load_state_dict_accepts_a_missing_class_key():
    stage = RewardScorer()
    stage.load_state_dict({"config": {"REWARD_WEIGHTS": {"r_task": 3.0}}})
    assert stage.config["REWARD_WEIGHTS"] == {"r_task": 3.0}


def test_severity_fraction_defaults_when_a_label_is_absent():
    from mactf.rules.severity import Severity

    assert severity_fraction(Severity.MAJOR, {"S3": 4.0}) == 4.0
    assert severity_fraction(Severity.NONE, {}) == 1.0
    assert severity_fraction(Severity.NONE, {"S0": 0.0}) == 0.0
    assert severity_fraction(Severity.MAJOR, {}, default=0.5) == 0.5


def test_prepare_default_is_identity():
    stage = RewardScorer()
    ctx = _ctx()
    assert stage.prepare(ctx) is ctx


def test_finalize_default_is_identity():
    stage = RewardScorer()
    output = ScorerOutput(kind=ScoringKind.REWARD, amount=1.0)
    assert stage.finalize(_ctx(), output) is output


def test_validate_default_is_empty():
    assert RewardScorer().validate({}) == []


def test_pipeline_rejects_a_negative_order():
    class Early(RewardScorer):
        order = -1
        kind = ScoringKind.REWARD

    with pytest.raises(ImproperlyConfigured, match="negative order"):
        ScoringPipeline([Early(), WarningScorer(), PenaltyScorer()])
