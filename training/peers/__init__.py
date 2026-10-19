"""Peer modules for the training project.

Any ``nn.Module`` works. These three exist so a swarm has heterogeneous members
out of the box, and they demonstrate the two supported output contracts.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPPeer(nn.Module):
    """A small peer returning ``(logits, value)``."""

    def __init__(self, state_dim: int = 16, action_dim: int = 4, hidden: int = 64) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(state)
        return self.policy_head(hidden), self.value_head(hidden)


class DeepPeer(nn.Module):
    """A deeper peer exercising the adapter's hidden-state path."""

    def __init__(self, state_dim: int = 16, action_dim: int = 4, hidden: int = 48) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor):
        hidden = self.body(state)
        return {"logits": self.policy_head(hidden), "value": self.value_head(hidden)}


class WidePeer(nn.Module):
    """A peer that returns logits only, so the adapter attaches a value head."""

    def __init__(self, state_dim: int = 16, action_dim: int = 4) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.policy_head(self.trunk(state))


PEERS = [MLPPeer, DeepPeer, WidePeer]
