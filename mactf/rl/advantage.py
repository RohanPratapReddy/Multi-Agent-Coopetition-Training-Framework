"""Advantage estimation (docx Turn 3, function 7).

    A_t ~= (r_t + gamma * V(s_{t+1})) - V(s_t)

and, for multi-step episodes, generalized advantage estimation. The reference
policy enters as the baseline for cross-peer comparability per framework.md
11.7: an advantage must never use the peer's own score as its baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class AdvantageBatch:
    advantages: torch.Tensor
    returns: torch.Tensor

    def describe(self) -> str:
        return (
            f"advantage(mean={self.advantages.mean().item():+.4f}, "
            f"std={self.advantages.std(unbiased=False).item():.4f})"
        )


def temporal_difference(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    *,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Single-step TD advantage, exactly the docx formula."""
    return rewards + gamma * next_values - values


def generalized_advantage(
    rewards: Sequence[float] | torch.Tensor,
    values: Sequence[float] | torch.Tensor,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
    next_value: float = 0.0,
) -> AdvantageBatch:
    """GAE-lambda over a trajectory.

    The docx only states the one-step form; GAE is the standard multi-step
    generalization and reduces to it when ``lam = 0``.
    """
    reward_tensor = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1)
    value_tensor = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    if reward_tensor.numel() != value_tensor.numel():
        raise ValueError("rewards and values must have the same length")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")

    length = reward_tensor.numel()
    advantages = torch.zeros_like(reward_tensor)
    last = float(next_value)
    running = 0.0
    for index in reversed(range(length)):
        next_val = float(value_tensor[index + 1]) if index + 1 < length else last
        delta = float(reward_tensor[index]) + gamma * next_val - float(value_tensor[index])
        running = delta + gamma * lam * running
        advantages[index] = running
    returns = advantages + value_tensor
    return AdvantageBatch(advantages=advantages, returns=returns)


def relative_to_reference(
    peer_signal: torch.Tensor,
    reference_signal: torch.Tensor,
) -> torch.Tensor:
    """Advantage against a reference baseline, not the peer's own mean."""
    if peer_signal.shape != reference_signal.shape:
        raise ValueError(
            f"shape mismatch: peer {tuple(peer_signal.shape)} vs "
            f"reference {tuple(reference_signal.shape)}"
        )
    return peer_signal - reference_signal


def penalise_void_advantage(
    advantages: torch.Tensor,
    void_mask: torch.Tensor,
    *,
    floor: float = -1.0,
) -> torch.Tensor:
    """Force non-positive advantage where a non-offsettable penalty applies.

    framework.md 3.1 rule 2: a T0/T1/T2 breach cannot be offset by good behaviour
    elsewhere, so the advantage for that trajectory must not be positive.
    """
    if advantages.shape != void_mask.shape:
        raise ValueError("advantages and void_mask must have the same shape")
    return torch.where(void_mask.bool(), torch.minimum(advantages, torch.full_like(advantages, floor)), advantages)
