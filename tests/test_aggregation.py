"""Adaptive kappa and constrained aggregation tests (framework.md section 10)."""

from __future__ import annotations

import math

import pytest
import torch

from mactf.coopetition.aggregation import Aggregator, feasible_bounds
from mactf.coopetition.temperature import AdaptiveKappa, entropy_collapsed
from mactf.exceptions import ComplianceError


# -- temperature ---------------------------------------------------------
def test_kappa_softmax_matches_definition():
    controller = AdaptiveKappa(num_peers=3, kappa_init=2.0, kappa_min=0.1, kappa_max=10.0)
    scores = torch.tensor([1.0, 0.0, -1.0])
    weights = controller.softmax_weights(scores)
    expected = torch.softmax(2.0 * scores, dim=0)
    assert torch.allclose(weights, expected)


def test_kappa_entropy_is_shannon():
    controller = AdaptiveKappa(num_peers=2, kappa_init=1.0)
    uniform = torch.tensor([0.5, 0.5])
    assert controller.entropy_of(uniform) == pytest.approx(math.log(2))


def test_kappa_target_entropy_uses_ratio():
    controller = AdaptiveKappa(num_peers=4, target_entropy_ratio=0.5)
    assert controller.target_entropy == pytest.approx(0.5 * math.log(4))


def test_kappa_moves_toward_target():
    controller = AdaptiveKappa(
        num_peers=2, kappa_init=1.0, target_entropy_ratio=0.5, beta=0.9, eta_kappa=0.5
    )
    start = controller.kappa
    # A dominant peer lowers entropy below target, so kappa must decrease.
    controller.step(torch.tensor([10.0, 0.0]))
    assert controller.kappa < start


def test_kappa_respects_bounds_under_extreme_scores():
    controller = AdaptiveKappa(num_peers=2, kappa_init=1.0, kappa_min=0.5, kappa_max=2.0)
    for _ in range(50):
        controller.step(torch.tensor([0.0, 0.0]))
    assert 0.5 <= controller.kappa <= 2.0
    for _ in range(50):
        controller.step(torch.tensor([1000.0, -1000.0]))
    assert 0.5 <= controller.kappa <= 2.0


def test_kappa_rejects_wrong_peer_count():
    controller = AdaptiveKappa(num_peers=3)
    # A shrinking active set (participation gating) is allowed and rebases the
    # entropy target; an empty one is not.
    controller.step(torch.tensor([1.0, 2.0]))
    assert controller.num_peers == 2
    with pytest.raises(ValueError):
        controller.step(torch.tensor([]))


def test_kappa_state_roundtrip():
    controller = AdaptiveKappa(num_peers=2, kappa_init=1.5)
    controller.step(torch.tensor([1.0, 0.0]))
    payload = controller.state_dict()
    other = AdaptiveKappa(num_peers=2)
    other.load_state_dict(payload)
    assert other.kappa == pytest.approx(controller.kappa)


def test_entropy_collapse_detection():
    assert entropy_collapsed(0.01, max_entropy=math.log(4), ratio=0.05)
    assert not entropy_collapsed(1.0, max_entropy=math.log(4), ratio=0.05)


# -- aggregation ---------------------------------------------------------
def make_aggregator(**overrides):
    defaults = dict(
        kappa_init=1.0,
        kappa_min=0.1,
        kappa_max=10.0,
        weight_floor=0.02,
        weight_cap=0.6,
        trim_fraction=0.0,
        disagreement_max=0.75,
    )
    defaults.update(overrides)
    controller = AdaptiveKappa(
        num_peers=defaults.pop("num_peers", 3),
        kappa_init=defaults.pop("kappa_init"),
        kappa_min=defaults.pop("kappa_min"),
        kappa_max=defaults.pop("kappa_max"),
    )
    return Aggregator(controller=controller, **defaults)


def test_weights_sum_to_one():
    aggregator = make_aggregator()
    report = aggregator.compute_weights(["a", "b", "c"], [1.0, 0.5, 0.0])
    assert sum(report.weights.values()) == pytest.approx(1.0, abs=1e-5)


def test_weights_never_go_below_floor():
    aggregator = make_aggregator(weight_floor=0.1)
    report = aggregator.compute_weights(["a", "b", "c"], [100.0, -100.0, -100.0])
    assert all(weight >= 0.1 - 1e-9 for weight in report.weights.values())


def test_weights_never_exceed_cap():
    aggregator = make_aggregator(weight_cap=0.5)
    report = aggregator.compute_weights(["a", "b", "c"], [100.0, 0.0, 0.0])
    assert all(weight <= 0.5 + 1e-9 for weight in report.weights.values())


def test_gated_peers_get_zero_and_no_mass():
    aggregator = make_aggregator()
    report = aggregator.compute_weights(
        ["a", "b", "c"], [1.0, 1.0, 1.0], eligible=[True, False, True]
    )
    assert report.weights["b"] == 0.0
    assert "b" in report.gated_peers
    assert sum(report.weights.values()) == pytest.approx(1.0, abs=1e-5)


def test_all_ineligible_falls_back():
    aggregator = make_aggregator()
    report = aggregator.compute_weights(
        ["a", "b"], [1.0, 1.0], eligible=[False, False]
    )
    assert report.fallback
    assert all(weight == 0.0 for weight in report.weights.values())


def test_confidence_gating_shifts_weight():
    aggregator = make_aggregator(weight_cap=1.0, weight_floor=0.0)
    even = aggregator.compute_weights(
        ["a", "b"], [1.0, 1.0], confidence=[1.0, 1.0], advance=False
    )
    skewed = aggregator.compute_weights(
        ["a", "b"], [1.0, 1.0], confidence=[1.0, 0.1], advance=False
    )
    assert skewed.weights["a"] > even.weights["a"]


def test_trimming_drops_the_dominant_peer():
    aggregator = make_aggregator(trim_fraction=0.2, weight_cap=1.0, weight_floor=0.0)
    report = aggregator.compute_weights(["a", "b", "c"], [10.0, 0.0, 0.0], advance=False)
    assert report.trimmed_peers
    assert report.weights[report.trimmed_peers[0]] == 0.0


def test_material_disagreement_triggers_fallback():
    aggregator = make_aggregator(disagreement_max=0.5)
    # Spread is 1.0 relative to |max| = 1.0, so this is a full contradiction.
    report = aggregator.compute_weights(["a", "b"], [1.0, 0.0])
    assert report.fallback


def test_small_score_spread_does_not_trigger_fallback():
    # Regression: 0.28 vs 0.0 previously produced a false positive.
    aggregator = make_aggregator(disagreement_max=0.75)
    report = aggregator.compute_weights(["a", "b", "c"], [0.2812, 0.0, 0.0])
    assert not report.fallback


def test_aggregation_is_order_invariant():
    aggregator = make_aggregator()
    first = aggregator.compute_weights(["a", "b", "c"], [1.0, 2.0, 3.0], advance=False)
    second = aggregator.compute_weights(["c", "b", "a"], [3.0, 2.0, 1.0], advance=False)
    for name in ("a", "b", "c"):
        assert first.weights[name] == pytest.approx(second.weights[name], abs=1e-6)


def test_aggregate_parameters_is_weighted_sum():
    from mactf.coopetition.adapter import PeerAdapter
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(0.0)

        def forward(self, state):
            return self.linear(state)

    adapters = {}
    for index, name in enumerate(["a", "b"]):
        adapter = PeerAdapter(Tiny(), name=name)
        with torch.no_grad():
            for parameter in adapter.parameters():
                parameter.fill_(float(index + 1))
        adapters[name] = adapter

    aggregator = make_aggregator(num_peers=2, disagreement_max=1.0)
    report = aggregator.compute_weights(["a", "b"], [1.0, 0.0])
    result = aggregator.aggregate(adapters, report)
    expected = report.weights["a"] * 1.0 + report.weights["b"] * 2.0
    assert torch.allclose(result.parameters, torch.full_like(result.parameters, expected))


def test_aggregate_requires_reference_on_fallback():
    aggregator = make_aggregator()
    report = aggregator.compute_weights(["a"], [1.0], eligible=[False])
    from mactf.coopetition.adapter import PeerAdapter
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, state):
            return self.linear(state)

    adapter = PeerAdapter(Tiny(), name="a")
    with pytest.raises(ComplianceError, match="reference"):
        aggregator.aggregate({"a": adapter}, report, reference=None)


def test_feasible_bounds_rejects_impossible_floor_and_cap():
    with pytest.raises(ComplianceError):
        feasible_bounds(4, floor=0.4, cap=0.9)
    with pytest.raises(ComplianceError):
        feasible_bounds(4, floor=0.0, cap=0.1)
    assert feasible_bounds(4, floor=0.05, cap=0.6) == (0.05, 0.6)
