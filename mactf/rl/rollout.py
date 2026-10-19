"""Rollout collection: states, actions, values, and rewards for one step set."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ..runtime import Runtime, autocast
from ..scoring.ledger import Ledger


@dataclass
class Transition:
    """One peer's forward pass and the resulting scored signal."""

    state: torch.Tensor
    actions: torch.Tensor
    logits: torch.Tensor
    values: torch.Tensor
    net_signal: float
    ledger: Ledger | None = None
    void: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.state.shape[0]) if self.state.dim() else 1


@dataclass
class Rollout:
    """A batch of transitions from one episode, plus reference baselines."""

    transitions: list[Transition] = field(default_factory=list)
    reference_logits: list[torch.Tensor] = field(default_factory=list)
    reference_values: list[torch.Tensor] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.transitions)

    def signals(self) -> torch.Tensor:
        return torch.tensor(
            [transition.net_signal for transition in self.transitions], dtype=torch.float32
        )

    def describe(self) -> str:
        if not self.transitions:
            return "rollout(empty)"
        signals = self.signals()
        return (
            f"rollout(n={len(self.transitions)}, "
            f"signal_mean={signals.mean().item():+.4f}, "
            f"signal_std={signals.std(unbiased=False).item():.4f})"
        )


def collect(
    peers: dict[str, Any],
    states: torch.Tensor,
    *,
    runtime: Runtime,
    sample: bool = True,
    generator: torch.Generator | None = None,
    reference: Any | None = None,
) -> Rollout:
    """Run every peer forward on the same states and collect transitions.

    Scoring is deliberately *not* done here: the caller attaches net signals from
    the scorer so that rollouts stay free of scoring policy.
    """
    rollout = Rollout()
    for name, peer in peers.items():
        adapter = peer.adapter if hasattr(peer, "adapter") else peer
        adapter.train()
        with torch.enable_grad(), autocast(runtime):
            logits, values = adapter(states)
        if sample:
            probs = torch.softmax(logits, dim=-1)
            if generator is not None:
                actions = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            else:
                actions = torch.multinomial(probs, 1).squeeze(-1)
        else:
            actions = logits.argmax(dim=-1)

        rollout.transitions.append(
            Transition(
                state=states.detach(),
                actions=actions.detach(),
                logits=logits.detach(),
                values=values.detach(),
                net_signal=0.0,
                metadata={"peer": name},
            )
        )
        if reference is not None:
            ref_adapter = reference.adapter if hasattr(reference, "adapter") else reference
            with torch.no_grad():
                ref_logits, ref_values = ref_adapter(states)
            rollout.reference_logits.append(ref_logits.detach())
            rollout.reference_values.append(ref_values.detach())
    return rollout


def attach_signals(rollout: Rollout, signals: dict[str, float]) -> Rollout:
    """Attach scored net signals to transitions by peer name."""
    for transition in rollout.transitions:
        peer = transition.metadata.get("peer")
        if peer in signals:
            transition.net_signal = float(signals[peer])
    return rollout
