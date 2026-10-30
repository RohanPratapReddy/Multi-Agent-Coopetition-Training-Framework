"""Criterion 21: the weight-update fallback is recorded, never silent."""

from __future__ import annotations

import pytest
import torch

from mactf.coopetition.aggregation import Aggregator
from mactf.coopetition.weights import (
    ExplicitWeightStrategy,
    KappaSoftmaxAverage,
    NoAggregation,
    ReferenceOnlyAverage,
    ScoreWeightedAverage,
    UniformAverage,
    build_weight_update,
)
from mactf.exceptions import ComplianceError, ValidationError

from train_support import make_trainer, tuned  # noqa: F401


def _vectors(*names):
    return {name: torch.zeros(3) for name in names}


def test_uniform_is_one_over_n_over_eligible_peers_and_records_the_gate():
    report = UniformAverage().apply(
        _vectors("a", "b", "c"), {}, kappa=1.0, eligible={"a": True, "b": True, "c": False}
    )
    assert report.weights == {"a": 0.5, "b": 0.5, "c": 0.0}
    assert report.gated_peers == ("c",)
    assert any("participation gate" in line for line in report.adjustments)
    assert not report.fallback


def test_uniform_with_nobody_eligible_reports_fallback():
    report = UniformAverage().apply(
        _vectors("a", "b"), {}, kappa=1.0, eligible={"a": False, "b": False}
    )
    assert report.fallback
    assert sum(report.weights.values()) == 0.0
    assert report.adjustments


def test_reference_only_is_a_recorded_reset():
    report = ReferenceOnlyAverage(reference="b").apply(_vectors("a", "b"), {}, kappa=1.0)
    assert report.weights == {"a": 0.0, "b": 1.0}
    assert report.fallback
    assert "reset to reference peer 'b'" in report.adjustments[0]
    missing = ReferenceOnlyAverage(reference="zz").apply(_vectors("a"), {}, kappa=1.0)
    assert missing.fallback and "no reference" in missing.adjustments[0]


def test_no_aggregation_is_a_zero_report():
    report = NoAggregation().apply(_vectors("a", "b"), {"a": 1.0}, kappa=1.0)
    assert set(report.weights.values()) == {0.0}
    assert report.adjustments


def test_the_default_needs_an_aggregator_and_delegates():
    with pytest.raises(ComplianceError, match="needs an Aggregator"):
        KappaSoftmaxAverage().apply(_vectors("a", "b"), {}, kappa=1.0)
    aggregator = Aggregator.from_settings({}, num_peers=2)
    report = KappaSoftmaxAverage(aggregator).apply(
        _vectors("a", "b"), {"a": 2.0, "b": 0.0}, kappa=aggregator.controller.kappa
    )
    assert sum(report.weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert report.weights["a"] > report.weights["b"]
    assert KappaSoftmaxAverage(aggregator).describe()["delegates_to"] == "Aggregator.compute_weights"


def test_score_weighted_and_explicit_strategies_normalise():
    scored = ScoreWeightedAverage().apply(
        _vectors("a", "b"), {"a": 3.0, "b": 1.0}, kappa=1.0
    )
    assert sum(scored.weights.values()) == pytest.approx(1.0)
    assert scored.weights["a"] > scored.weights["b"]
    explicit = ExplicitWeightStrategy.from_spec({"weights": {"a": 0.75, "b": 0.25}})
    result = explicit.apply(_vectors("a", "b"), {}, kappa=1.0)
    assert result.weights["a"] == pytest.approx(0.75)
    with pytest.raises(ValidationError, match="sum to 1"):
        ExplicitWeightStrategy({"a": 3.0, "b": 1.0})


def test_build_weight_update_resolves_names_specs_and_rejects_unknowns():
    aggregator = Aggregator.from_settings({}, num_peers=2)
    assert isinstance(build_weight_update(None, aggregator=aggregator), KappaSoftmaxAverage)
    assert isinstance(build_weight_update("uniform"), UniformAverage)
    assert isinstance(build_weight_update({"type": "reference_only"}), ReferenceOnlyAverage)
    with pytest.raises(ValidationError, match="accepted"):
        build_weight_update("lottery")
    with pytest.raises(ValidationError, match="accepted"):
        build_weight_update({"type": "lottery"})
    with pytest.raises(ValidationError):
        build_weight_update(12)


class _Exploding:
    name = "exploding"

    def apply(self, vectors, scores, **kwargs):
        raise ComplianceError("this strategy cannot produce weights")

    def describe(self):
        return {"name": self.name}


def test_a_failing_strategy_falls_back_and_says_so(tuned, tmp_path):  # noqa: F811
    trainer = make_trainer(tuned, tmp_path, weight_update=_Exploding(), peers=3)
    result = trainer.fit(epochs=1)
    assert result.status == "completed"
    report = trainer.reports[-1]
    notes = [line for line in report.adjustments if "fell back to kappa_softmax" in line]
    assert notes, report.adjustments
    assert "exploding" in notes[0]
    assert "cannot produce weights" in notes[0]
    assert sum(report.weights.values()) == pytest.approx(1.0, abs=1e-5)


def test_a_strategy_that_reports_fallback_is_replaced_and_recorded(tuned, tmp_path):  # noqa: F811
    trainer = make_trainer(
        tuned, tmp_path, weight_update=ReferenceOnlyAverage(reference="peer-1"), peers=3
    )
    trainer.fit(epochs=1)
    report = trainer.reports[-1]
    assert any("it reported fallback" in line for line in report.adjustments)
    assert any("reference_only" in line for line in report.adjustments)


def test_the_default_strategy_never_falls_back_silently(tuned, tmp_path):  # noqa: F811
    trainer = make_trainer(tuned, tmp_path, peers=3)
    trainer.fit(epochs=1)
    assert not any("fell back" in line for line in trainer.reports[-1].adjustments)
