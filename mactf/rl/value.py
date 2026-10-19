"""Value estimation (docx Turn 3, function 6).

``V_phi(s) = E[ sum_k gamma^k (R - P - W) ]``. The value head may be the peer's
own head or a standalone critic; both are supported because the docx does not
require a shared trunk.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """A small MLP critic over a hidden vector."""

    def __init__(self, hidden_dim: int, *, width: int | None = None) -> None:
        super().__init__()
        width = width or max(8, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


@dataclass
class ValueEstimate:
    values: torch.Tensor

    @property
    def mean(self) -> float:
        return float(self.values.mean().item())

    def squeeze(self) -> torch.Tensor:
        return self.values.squeeze(-1)


def normalise_advantages(
    advantages: torch.Tensor, *, eps: float = 1e-8
) -> torch.Tensor:
    """Zero-center and unit-scale advantages across a batch.

    Required by framework.md 11.7 (comparable across peers, zero-centered in
    expectation). With a single sample there is nothing to normalize, so the
    advantages are returned unchanged rather than becoming all zeros.
    """
    if advantages.numel() <= 1:
        return advantages
    std = advantages.std(unbiased=False)
    if float(std) < eps:
        return advantages - advantages.mean()
    return (advantages - advantages.mean()) / (std + eps)


def value_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    coefficient: float = 0.5,
) -> torch.Tensor:
    """Mean-squared value regression, the docx ``c1`` term."""
    return coefficient * torch.nn.functional.mse_loss(
        predicted.reshape(-1), target.reshape(-1)
    )
