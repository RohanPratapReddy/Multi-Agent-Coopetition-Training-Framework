"""18.5: an optimizer and a scheduler: Lion with a step-down."""

from __future__ import annotations

import torch

from mactf.components import register_optimizer, register_scheduler
from mactf.contrib.components.examples import registration_scope


@register_optimizer("Lion", version="1.0", description="Sign-of-momentum optimizer (Chen et al., 2023)",
                    scope=registration_scope())
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-4, betas=(0.9, 0.99), weight_decay: float = 0.0) -> None:
        if lr < 0.0:
            raise ValueError("lr must be >= 0")
        super().__init__(params, dict(lr=lr, betas=tuple(betas), weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p)
                p.mul_(1.0 - group["lr"] * group["weight_decay"])
                update = state["exp_avg"].mul(beta1).add(p.grad, alpha=1.0 - beta1).sign_()
                p.add_(update, alpha=-group["lr"])
                state["exp_avg"].mul_(beta2).add_(p.grad, alpha=1.0 - beta2)
        return loss


@register_scheduler("step_then_flat", version="1", scope=registration_scope())
def step_then_flat(optimizer, *, drop_at: int, factor: float = 0.1):
    if drop_at < 1 or not 0.0 < factor <= 1.0:
        raise ValueError("drop_at must be >= 1 and factor in (0, 1]")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 if step < drop_at else factor)
