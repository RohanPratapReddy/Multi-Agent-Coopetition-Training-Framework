"""18.7: a training policy (the module of 7.10): the default step plus an anchor to the reference."""

from __future__ import annotations

from typing import ClassVar

import torch

from mactf.components import register_policy
from mactf.contrib.components.examples import registration_scope
from mactf.rl.loss import LossBreakdown
from mactf.training.protocol import StepResult


@register_policy("anchored", version="1", scope=registration_scope())
class AnchoredPolicy:
    """A policy that runs the default-style step plus an anchor to the reference (7.10)."""

    name: ClassVar[str] = "anchored"
    unsafe: ClassVar[bool] = False
    self_stepping: ClassVar[bool | None] = False
    requires_terms: ClassVar[frozenset[str]] = frozenset({"compliance_floor"})
    composes_distillation: ClassVar[bool] = False

    def __init__(self, *, anchor_coefficient: float = 0.05) -> None:
        if anchor_coefficient < 0:
            raise ValueError("anchor_coefficient must be >= 0")
        self.anchor_coefficient = float(anchor_coefficient)
        self._host = None

    def attach(self, host):
        self._host = host

    def train_step(self, node, batch):
        host = self._host
        assert host is not None
        _, peer = host.resolve_node(node)
        states = batch["states"]
        logits, values = peer.adapter(states)
        with torch.no_grad():
            ref_logits, _ = host.reference.adapter(states)
        anchor = self.anchor_coefficient * torch.nn.functional.mse_loss(logits, ref_logits)
        floor = float(batch.get("floor", 0.0))
        compliance = torch.tensor(max(-floor, 0.0), device=logits.device)
        signal = float(batch.get("signal", 0.0))
        task = -signal * torch.log_softmax(logits, dim=-1).mean()
        total = task + anchor + compliance
        breakdown = LossBreakdown(policy=float(task.detach()), compliance=float(compliance.detach()),
                                  total=float(total.detach()), extras={"anchor": float(anchor.detach())})
        return StepResult(loss=total, breakdown=breakdown, stepped=True)


class Forgetful(AnchoredPolicy):
    """The misbehaviour: it declares ``compliance_floor`` and reports none (S004)."""

    name = "forgetful"

    def train_step(self, node, batch):
        result = super().train_step(node, batch)
        result.breakdown.compliance = 0.0
        return result


class Greedy(AnchoredPolicy):
    """``unsafe``: ``StepGuard`` does not wrap it, so it needs ``TRAINER.ALLOW_UNSAFE_POLICY`` (E009)."""

    name = "greedy"
    unsafe: ClassVar[bool] = True
