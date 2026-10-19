"""Policy utilities (docx Turn 3, function 4).

``pi_theta(a|s) = softmax(f_theta(s))``. These helpers keep the policy math in
one place so the loss module can stay readable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PolicyOutput:
    """Logits plus the derived probability distribution and log-probabilities."""

    logits: torch.Tensor
    probs: torch.Tensor
    log_probs: torch.Tensor

    @property
    def entropy(self) -> torch.Tensor:
        """Per-sample entropy in nats."""
        return -(self.probs * self.log_probs).sum(dim=-1)

    def gather(self, actions: torch.Tensor) -> torch.Tensor:
        """Log-probability of the chosen actions, shape ``(batch,)``."""
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        return self.log_probs.gather(-1, actions).squeeze(-1)


def make_policy(logits: torch.Tensor) -> PolicyOutput:
    """Normalize raw logits into a policy distribution."""
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    log_probs = F.log_softmax(logits, dim=-1)
    return PolicyOutput(logits=logits, probs=log_probs.exp(), log_probs=log_probs)


def sample_action(
    policy: PolicyOutput, *, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample actions and return them with their log-probabilities."""
    probs = policy.probs
    if generator is not None:
        actions = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    else:
        actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return actions, policy.gather(actions)


def greedy_action(policy: PolicyOutput) -> torch.Tensor:
    return policy.probs.argmax(dim=-1)


def behaviour_probabilities(policy: PolicyOutput, actions: torch.Tensor) -> torch.Tensor:
    """Probability the policy assigns to the actions that were actually taken."""
    if actions.ndim == 1:
        actions = actions.unsqueeze(-1)
    return policy.probs.gather(-1, actions).squeeze(-1)


def importance_ratio(
    policy: PolicyOutput,
    reference: PolicyOutput,
    actions: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """``pi_theta(a|s) / pi_ref(a|s)`` from the docx policy-loss term."""
    return behaviour_probabilities(policy, actions) / (
        behaviour_probabilities(reference, actions) + eps
    )


def clip_ratio(ratio: torch.Tensor, *, epsilon: float) -> torch.Tensor:
    return torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)


def reverse_kl(policy: PolicyOutput, reference: PolicyOutput) -> torch.Tensor:
    """``KL(pi_theta || pi_ref)`` as a per-sample tensor."""
    return F.kl_div(policy.log_probs, reference.probs.detach(), reduction="none").sum(dim=-1)
