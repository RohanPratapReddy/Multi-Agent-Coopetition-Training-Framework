"""Peer modules for the {{ project_name }} swarm.

Any ``torch.nn.Module`` works. MACTF wraps it with
:class:`mactf.coopetition.adapter.PeerAdapter`, which accepts:

* ``(logits, value)`` - preferred, and the value head is trained for you
* ``{"logits": ..., "value": ...}``
* ``logits`` only - the adapter attaches a value head

``logits`` must have shape ``(batch, action_dim)``; ``value`` is ``(batch, 1)``.
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


# Declared so app discovery registers this peer automatically.
PEERS = [MLPPeer]
