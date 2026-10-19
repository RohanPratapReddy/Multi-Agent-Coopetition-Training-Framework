"""RL primitive tests: policy, advantage, and loss."""

from __future__ import annotations

import pytest
import torch

from mactf.rl.advantage import (
    generalized_advantage,
    penalise_void_advantage,
    relative_to_reference,
    temporal_difference,
)
from mactf.rl.loss import LossConfig, compute_loss
from mactf.rl.policy import make_policy, importance_ratio, sample_action
from mactf.rl.value import normalise_advantages, value_loss


def test_policy_probabilities_sum_to_one():
    policy = make_policy(torch.randn(4, 5))
    assert torch.allclose(policy.probs.sum(dim=-1), torch.ones(4))


def test_policy_gather_matches_log_probs():
    policy = make_policy(torch.tensor([[0.0, 10.0]]))
    actions = torch.tensor([1])
    assert policy.gather(actions).item() == pytest.approx(0.0, abs=1e-3)


def test_sample_action_returns_valid_indices():
    policy = make_policy(torch.randn(8, 4))
    actions, log_probs = sample_action(policy)
    assert actions.shape == (8,)
    assert log_probs.shape == (8,)
    assert all(0 <= int(action) < 4 for action in actions)


def test_importance_ratio_is_one_for_identical_policies():
    policy = make_policy(torch.randn(3, 4))
    actions = torch.tensor([0, 1, 2])
    ratio = importance_ratio(policy, policy, actions)
    assert torch.allclose(ratio, torch.ones(3), atol=1e-5)


def test_temporal_difference_matches_docx_formula():
    advantage = temporal_difference(
        rewards=torch.tensor([1.0]),
        values=torch.tensor([0.5]),
        next_values=torch.tensor([1.0]),
        gamma=0.9,
    )
    assert advantage.item() == pytest.approx(1.0 + 0.9 * 1.0 - 0.5)


def test_gae_reduces_to_td_when_lambda_zero():
    batch = generalized_advantage(
        rewards=[1.0], values=[0.5], gamma=0.9, lam=0.0, next_value=1.0
    )
    assert batch.advantages.item() == pytest.approx(1.0 + 0.9 * 1.0 - 0.5)


def test_gae_returns_sum_advantages_plus_values():
    batch = generalized_advantage(
        rewards=[1.0, 1.0], values=[0.0, 0.0], gamma=1.0, lam=1.0
    )
    assert torch.allclose(batch.returns, batch.advantages)


def test_relative_to_reference_subtracts_baseline():
    peer = torch.tensor([1.0, 2.0])
    reference = torch.tensor([0.5, 0.5])
    assert torch.allclose(relative_to_reference(peer, reference), torch.tensor([0.5, 1.5]))


def test_penalise_void_advantage_forces_non_positive():
    advantages = torch.tensor([5.0, 1.0])
    mask = torch.tensor([True, False])
    result = penalise_void_advantage(advantages, mask, floor=-1.0)
    assert result[0].item() <= -1.0
    assert result[1].item() == pytest.approx(1.0)


def test_normalise_advantages_is_zero_centered():
    advantages = torch.tensor([1.0, 3.0, 5.0])
    result = normalise_advantages(advantages)
    assert result.mean().item() == pytest.approx(0.0, abs=1e-6)


def test_normalise_advantages_handles_zero_variance():
    advantages = torch.tensor([2.0, 2.0, 2.0])
    result = normalise_advantages(advantages)
    assert torch.allclose(result, torch.zeros(3))


def test_value_loss_scales_by_coefficient():
    predicted = torch.tensor([[0.0]])
    target = torch.tensor([[1.0]])
    assert value_loss(predicted, target, coefficient=0.5).item() == pytest.approx(0.5)


def test_compute_loss_components_are_reported():
    policy = make_policy(torch.randn(4, 3))
    reference = make_policy(torch.randn(4, 3))
    actions = torch.tensor([0, 1, 2, 0])
    advantages = torch.tensor([1.0, -1.0, 0.5, 0.0])
    values = torch.zeros(4, 1)
    targets = torch.zeros(4, 1)

    total, breakdown = compute_loss(
        policy, reference, actions, advantages, values, targets, config=LossConfig()
    )
    assert total.item() == pytest.approx(breakdown.total, abs=1e-5)
    assert breakdown.value == pytest.approx(0.0, abs=1e-6)
    assert set(breakdown.as_dict()) >= {"policy", "value", "kl", "total"}


def test_compliance_floor_dominates_loss():
    policy = make_policy(torch.randn(2, 2))
    reference = make_policy(torch.randn(2, 2))
    actions = torch.tensor([0, 1])
    advantages = torch.tensor([0.0, 0.0])
    values = torch.zeros(2, 1)
    targets = torch.zeros(2, 1)

    without, _ = compute_loss(policy, reference, actions, advantages, values, targets)
    with_floor, breakdown = compute_loss(
        policy,
        reference,
        actions,
        advantages,
        values,
        targets,
        compliance_floor=torch.tensor(-1000.0),
    )
    assert with_floor.item() > without.item()
    assert breakdown.compliance == pytest.approx(1000.0)


def test_policy_loss_penalises_worse_actions_more():
    reference = make_policy(torch.tensor([[0.0, 0.0]]))
    good = make_policy(torch.tensor([[3.0, -3.0]]))
    bad = make_policy(torch.tensor([[-3.0, 3.0]]))
    actions = torch.tensor([0])
    advantages = torch.tensor([1.0])

    good_loss, _ = compute_loss(
        good, reference, actions, advantages, torch.zeros(1, 1), torch.zeros(1, 1)
    )
    bad_loss, _ = compute_loss(
        bad, reference, actions, advantages, torch.zeros(1, 1), torch.zeros(1, 1)
    )
    assert good_loss.item() < bad_loss.item()
