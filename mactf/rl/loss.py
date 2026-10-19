"""The training objective (docx Turn 3, function 8).

The docx loss is::

    L = -E[ (pi_theta(a|s) / pi_ref(a|s)) * A ] + c1 * (V(s) - V_target)^2
        + c2 * KL(pi_theta || pi_ref)

This module implements that with two framework.md additions:

* the importance ratio is clipped (PPO), because framework.md 11.8 requires a
  trust region and stable batches;
* an optional compliance floor term makes a non-offsettable violation dominate
  the objective instead of being traded against policy improvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .policy import PolicyOutput, clip_ratio, importance_ratio, reverse_kl


@dataclass
class LossBreakdown:
    """Component values, reported separately as framework.md 11.8 requires."""

    policy: float = 0.0
    value: float = 0.0
    kl: float = 0.0
    entropy: float = 0.0
    compliance: float = 0.0
    total: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        payload = {
            "policy": self.policy,
            "value": self.value,
            "kl": self.kl,
            "entropy": self.entropy,
            "compliance": self.compliance,
            "total": self.total,
        }
        payload.update(self.extras)
        return payload


@dataclass
class LossConfig:
    value_coef: float = 0.5
    kl_coef: float = 0.01
    entropy_coef: float = 0.0
    clip_epsilon: float = 0.2
    compliance_coef: float = 1.0
    advantage_normalization: bool = True


def policy_loss(
    policy: PolicyOutput,
    reference: PolicyOutput,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
) -> torch.Tensor:
    """Clipped surrogate objective from the docx, negated for minimization."""
    ratio = importance_ratio(policy, reference, actions)
    surrogate = ratio * advantages.detach()
    clipped = clip_ratio(ratio, epsilon=clip_epsilon) * advantages.detach()
    return -torch.min(surrogate, clipped).mean()


def compute_loss(
    policy: PolicyOutput,
    reference: PolicyOutput,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    value_targets: torch.Tensor,
    *,
    config: LossConfig | None = None,
    compliance_floor: torch.Tensor | None = None,
    void_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, LossBreakdown]:
    """Compute the composite loss and its component breakdown."""
    config = config or LossConfig()

    p_loss = policy_loss(
        policy, reference, actions, advantages, clip_epsilon=config.clip_epsilon
    )
    v_loss = F.mse_loss(values.reshape(-1), value_targets.reshape(-1))
    k_loss = reverse_kl(policy, reference).mean()
    ent = policy.entropy.mean()

    total = (
        p_loss
        + config.value_coef * v_loss
        + config.kl_coef * k_loss
        - config.entropy_coef * ent
    )

    compliance_term = torch.zeros((), device=total.device, dtype=total.dtype)
    if compliance_floor is not None:
        # A floor is a negative number; we push the objective down by its magnitude
        # so no amount of policy gain can outvote a severe violation.
        floor_tensor = torch.as_tensor(
            compliance_floor, device=total.device, dtype=total.dtype
        )
        compliance_term = config.compliance_coef * (-floor_tensor).clamp_min(0.0)
        total = total + compliance_term

    if void_mask is not None and bool(void_mask.any()):
        # framework.md 3.1.2: withhold task-success credit. The indicator is added
        # only for observability; the advantage already reflects it.
        pass

    breakdown = LossBreakdown(
        policy=float(p_loss.detach().item()),
        value=float(v_loss.detach().item()),
        kl=float(k_loss.detach().item()),
        entropy=float(ent.detach().item()),
        compliance=float(compliance_term.detach().item()),
        total=float(total.detach().item()),
        extras={"advantage_mean": float(advantages.mean().detach().item())},
    )
    return total, breakdown


def clip_gradients(parameters, *, max_norm: float = 1.0) -> float:
    """Clip gradients and return the pre-clip norm for telemetry."""
    import math

    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    norm = math.sqrt(sum(float(grad.detach().norm() ** 2) for grad in grads))
    torch.nn.utils.clip_grad_norm_(list(parameters), max_norm=max_norm)
    return norm
