"""18.6: an RL function: a Huber value loss.

A ``loss`` function is ``compute(batch) -> (Tensor, LossBreakdown)``; the compliance term is
added by the framework (R3), so this function cannot leave it out.
"""

from __future__ import annotations

import torch.nn.functional as F

from mactf.components import register_function
from mactf.contrib.components.examples import registration_scope
from mactf.rl.loss import LossBreakdown


@register_function("loss", "huber_value", version="1", scope=registration_scope())
class HuberValueLoss:
    """``delta``-Huber between predicted values and value targets."""

    required_fields = frozenset({"values", "value_targets"})
    probe = {"state_dim": 8}

    def __init__(self, *, delta: float = 1.0) -> None:
        if delta <= 0.0:
            raise ValueError("delta must be > 0")
        self.delta = float(delta)

    def compute(self, batch):
        values, targets = batch["values"], batch["value_targets"]
        loss = F.huber_loss(values, targets.detach(), delta=self.delta)
        return loss, LossBreakdown(total=float(loss.detach()), extras={"huber": float(loss.detach())})


class NonFiniteHuber(HuberValueLoss):
    """The misbehaviour the probe catches before the slot is committed: a non-finite loss."""

    def compute(self, batch):
        loss, parts = super().compute(batch)
        return loss * float("nan"), parts
