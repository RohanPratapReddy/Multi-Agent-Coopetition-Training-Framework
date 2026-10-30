"""Weights over peers only, the document 08 fallback rule, and blending (doc 14 section 9)."""

from __future__ import annotations

import pytest
import torch

from distill_support import build_step, shared_batch, signals as peer_signals, teacher_spec
from mactf import signals
from mactf.coopetition.aggregation import Aggregator
from mactf.coopetition.report import AggregationReport
from mactf.coopetition.weights import KappaSoftmaxAverage, UniformAverage
from mactf.exceptions import ComplianceError, ContractError


class Raising:
    name = "raising"

    def apply(self, vectors, scores, *, kappa, level=0, eligible=None, confidence=None):
        raise ComplianceError("this strategy refuses every cohort")

    def describe(self):
        return {"type": "raising"}


class NotImplementedStrategy(Raising):
    name = "not_implemented"

    def apply(self, vectors, scores, *, kappa, level=0, eligible=None, confidence=None):
        raise NotImplementedError("not yet")


class ReportsFallback:
    name = "reports_fallback"

    def apply(self, vectors, scores, *, kappa, level=0, eligible=None, confidence=None):
        names = sorted(vectors)
        return AggregationReport(
            weights={n: 1.0 / len(names) for n in names},
            raw_weights={n: 1.0 / len(names) for n in names},
            kappa=float(kappa),
            entropy=0.0,
            adjustments=[],
            fallback=True,
            order=tuple(names),
        )

    def describe(self):
        return {"type": "reports_fallback"}


class LeaksATeacher:
    name = "leaks"

    def apply(self, vectors, scores, *, kappa, level=0, eligible=None, confidence=None):
        names = sorted(vectors) + ["t1"]
        weights = {n: 1.0 / len(names) for n in names}
        return AggregationReport(
            weights=weights,
            raw_weights=dict(weights),
            kappa=float(kappa),
            entropy=0.0,
            adjustments=[],
            order=tuple(names),
        )

    def describe(self):
        return {"type": "leaks"}


class Collector:
    def __init__(self, signal):
        self.events = []
        self.signal = signal
        signal.connect(self, weak=False)

    def __call__(self, sender=None, **payload):
        self.events.append(payload)

    def close(self):
        self.signal.disconnect(self)


SCORES = {"peer-1": 0.4, "peer-2": -0.1}


def test_weights_are_exactly_the_peer_set_and_sum_to_one(settings, tmp_path):
    step, _ = build_step(settings, tmp_path)
    report = step.update_weights(scores=SCORES)
    assert set(report.weights) == {"peer-1", "peer-2"}
    assert sum(report.weights.values()) == pytest.approx(1.0)
    assert "t1" not in report.weights
    assert report.weights["peer-1"] > report.weights["peer-2"]
    assert step.state.aggregations == 1


def test_default_weights_equal_the_plain_aggregator(settings, tmp_path):
    step, _ = build_step(settings, tmp_path)
    report = step.update_weights(scores=SCORES)
    reference = KappaSoftmaxAverage(step._weights_aggregator()).apply(
        {name: peer.export_parameters() for name, peer in step._peers.items()},
        SCORES,
        kappa=step._kappa_value(),
    )
    for name, weight in reference.weights.items():
        assert report.weights[name] == pytest.approx(weight, abs=1e-12)


def test_scores_default_to_the_last_cohort_signal(settings, tmp_path):
    step, trainer = build_step(settings, tmp_path)
    step.step(trainer.peers, shared_batch(), signal={"peer-1": 1.0, "peer-2": 0.0})
    report = step.update_weights()
    assert report.weights["peer-1"] > report.weights["peer-2"]


def test_a_peer_without_a_score_is_an_error_not_a_silent_zero(settings, tmp_path):
    step, _ = build_step(settings, tmp_path)
    with pytest.raises(ContractError, match="no score for peer"):
        step.update_weights(scores={"peer-1": 0.1})


def test_update_weights_before_setup_is_an_error(settings, tmp_path):
    step, _ = build_step(settings, tmp_path, register=False)
    with pytest.raises(ContractError, match="before setup"):
        step.update_weights(scores=SCORES)


def test_an_ineligible_peer_gets_no_weight(settings, tmp_path):
    # Three peers: the default WEIGHT_CAP of 0.6 cannot normalize a single survivor.
    step, _ = build_step(settings, tmp_path, peers=3)
    scores = {"peer-1": 0.4, "peer-2": -0.1, "peer-3": 0.0}
    report = step.update_weights(
        scores=scores, eligible={"peer-1": False, "peer-2": True, "peer-3": True}
    )
    assert report.weights["peer-1"] == 0.0
    assert report.weights["peer-2"] + report.weights["peer-3"] == pytest.approx(1.0)


@pytest.mark.parametrize("strategy", [Raising(), NotImplementedStrategy(), ReportsFallback()])
def test_a_failing_custom_strategy_falls_back_loudly(settings, tmp_path, strategy):
    step, _ = build_step(settings, tmp_path, weight_update=strategy)
    collector = Collector(signals.distillation_fallback)
    try:
        report = step.update_weights(scores=SCORES)
    finally:
        collector.close()
    assert set(report.weights) == {"peer-1", "peer-2"}
    assert any("fell back to kappa_softmax" in note for note in report.adjustments)
    assert any(strategy.name in note for note in report.adjustments)
    assert len(collector.events) == 1
    assert collector.events[0]["reason"]
    assert report.weights["peer-1"] > report.weights["peer-2"]


def test_the_default_strategy_does_not_swallow_its_own_failure(settings, tmp_path):
    step, _ = build_step(settings, tmp_path)

    class Broken(KappaSoftmaxAverage):
        def apply(self, *args, **kwargs):
            raise ComplianceError("broken default")

    step.weight_update = Broken(step._weights_aggregator())
    with pytest.raises(ComplianceError, match="broken default"):
        step.update_weights(scores=SCORES)


def test_a_strategy_that_names_a_teacher_is_refused(settings, tmp_path):
    """The assertion the whole policy exists to make unreachable."""
    step, _ = build_step(settings, tmp_path, weight_update=LeaksATeacher())
    with pytest.raises(ComplianceError, match="teachers are output-only"):
        step.update_weights(scores=SCORES)


def test_a_uniform_strategy_is_honoured(settings, tmp_path):
    step, _ = build_step(settings, tmp_path, weight_update=UniformAverage())
    report = step.update_weights(scores=SCORES)
    assert report.weights == {"peer-1": pytest.approx(0.5), "peer-2": pytest.approx(0.5)}
    assert not any("fell back" in note for note in report.adjustments)


def test_blend_without_an_aggregator_says_so_and_leaves_parameters_alone(settings, tmp_path):
    step, trainer = build_step(settings, tmp_path)
    step.step(trainer.peers, shared_batch(), signal=SCORES)
    before = {name: peer.export_parameters().clone() for name, peer in step._peers.items()}
    report = step.blend()
    assert any("not blended" in note for note in report.adjustments)
    for name, peer in step._peers.items():
        assert torch.equal(peer.export_parameters(), before[name])
    assert step.state.blends == 0


def test_blend_with_an_aggregator_moves_peers_and_never_touches_teachers(settings, tmp_path):
    aggregator = Aggregator.from_settings(dict(settings["COOPETITION"]), num_peers=2)
    step, _ = build_step(settings, tmp_path, aggregator=aggregator)
    teacher_before = torch.cat(
        [p.detach().reshape(-1).clone() for p in step.teachers["t1"].module.parameters()]
    )
    before = {name: peer.export_parameters().clone() for name, peer in step._peers.items()}
    report = step.blend(step.update_weights(scores=SCORES))
    assert step.state.blends == 1
    assert set(report.weights) == {"peer-1", "peer-2"}
    moved = any(
        not torch.equal(peer.export_parameters(), before[name]) for name, peer in step._peers.items()
    )
    assert moved
    teacher_after = torch.cat(
        [p.detach().reshape(-1) for p in step.teachers["t1"].module.parameters()]
    )
    assert torch.equal(teacher_before, teacher_after)


def test_teacher_signals_never_enter_the_peer_score_mapping(settings, tmp_path):
    step, trainer = build_step(
        settings, tmp_path, teachers=[teacher_spec("t1"), teacher_spec("t2")]
    )
    step.step(trainer.peers, shared_batch(), signal=peer_signals(trainer))
    report = step.update_weights()
    assert set(report.weights) == set(trainer.peers)
