"""Shared pieces of the phase-2 tests: a tiny model, a trainer factory and the worked examples.

``component_overlay`` compares registry snapshots, so the built-in ``trainer_policy`` names
(registered by importing ``mactf.training.policy``, which needs torch and therefore is not the
registry's ``builtins_module``) must be present before the first test runs.

The worked examples of document 17 (sections 8.9, 18.5 and 7.10) are defined here as plain
classes, not registered: a test registers them inside ``component_overlay`` so the process
registry is never touched.
"""

from __future__ import annotations

import copy
from typing import Any, ClassVar, Mapping

import torch
import torch.nn as nn

from mactf.coopetition.tree import HierarchySpec, LevelSpec
from mactf.exceptions import ValidationError
from mactf.rl.advantage import AdvantageBatch
from mactf.rl.functions.advantage import GAEAdvantage
from mactf.rl.loss import LossBreakdown
import mactf.training.policy  # noqa: F401  (registers the built-in policies: see the note below)
from mactf.training import MACTFTrainer
from mactf.training.protocol import StepResult


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(4, 16), nn.Tanh())
        self.policy = nn.Linear(16, 3)
        self.value = nn.Linear(16, 1)

    def forward(self, state):
        hidden = self.body(state)
        return self.policy(hidden), self.value(hidden)


def tune(settings, **training: Any):
    settings["TRAINING"].update(
        {"EPOCHS": 2, "BATCH_SIZE": 8, "STATE_DIM": 4, "ACTION_DIM": 3, "BOOTSTRAP": False,
         **training}
    )
    settings["SANDBOX"]["PROVIDER"] = "local"
    return settings


def configure(settings, **blocks: Mapping[str, Any]):
    """Set (deep-merge) whole blocks on a settings object, the way a ``conf/`` file would."""
    authored = {key: set(value) for key, value in dict(settings.get("_AUTHORED") or {}).items()}
    for name, value in blocks.items():
        current = settings.get(name)
        if isinstance(current, dict) and isinstance(value, Mapping):
            current.update(copy.deepcopy(dict(value)))
        else:
            settings[name] = copy.deepcopy(value)
        authored.setdefault(name, set()).update(
            str(key) for key in (value if isinstance(value, Mapping) else ())
        )
    settings["_AUTHORED"] = {key: sorted(value) for key, value in authored.items()}
    return settings


def make_trainer(settings, tmp_path, *, seed: int = 0, peers: int = 2, **kwargs) -> MACTFTrainer:
    torch.manual_seed(seed)
    return MACTFTrainer(
        model=Tiny(),
        num_peers=peers,
        settings=settings,
        checkpoint_dir=tmp_path / "ckpt",
        **kwargs,
    )


def hierarchy_trainer(settings, tmp_path, *, seed: int = 0, roots: int = 2, fanout: int = 2, **kwargs):
    torch.manual_seed(seed)
    spec = HierarchySpec(
        levels=(LevelSpec("sub", fanout, model=Tiny()),),
        roots=roots,
        root_models={f"peer-{index + 1}": Tiny() for index in range(roots)},
    )
    return MACTFTrainer(
        hierarchy=spec, settings=settings, checkpoint_dir=tmp_path / "ckpt", **kwargs
    )


def snapshot_parameters(trainer: MACTFTrainer) -> dict[str, torch.Tensor]:
    return {
        f"{path}:{name}": tensor.detach().clone()
        for path, peer in trainer.peers.items()
        for name, tensor in peer.adapter.module.named_parameters()
    }


# -- worked examples (section 8.9) ----------------------------------------------------
class ClippedAdvantage:
    """GAE advantages clipped to ``[-clip, clip]`` (8.9.1)."""

    spec_family: ClassVar[str] = "advantage"
    spec_type: ClassVar[str] = "clipped"

    def __init__(self, *, clip: float = 5.0) -> None:
        if float(clip) <= 0.0:
            raise ValidationError(f"clip must be positive, got {clip!r}")
        self.clip = float(clip)
        self._inner = GAEAdvantage()

    def __call__(
        self, rewards, values, *, next_values=None, masks=None, reference=None,
        gamma: float = 0.99, lam: float = 0.95,
    ) -> AdvantageBatch:
        base = self._inner(
            rewards, values, next_values=next_values, masks=masks, reference=reference,
            gamma=gamma, lam=lam,
        )
        advantages = base.advantages.clamp(min=-self.clip, max=self.clip)
        returns = advantages + torch.as_tensor(values).float().reshape_as(advantages)
        return AdvantageBatch(advantages=advantages, returns=returns)

    def describe(self) -> dict[str, Any]:
        return {"type": self.spec_type, "class": type(self).__name__, "clip": self.clip}


class L1AnnealRegularization:
    """``coefficient * (1 - t/steps) * sum(|p|)``, with the call counter as real state (8.9.2)."""

    spec_family: ClassVar[str] = "regularization"
    spec_type: ClassVar[str] = "l1_anneal"

    def __init__(self, *, coefficient: float = 1e-4, steps: int = 1000) -> None:
        if float(coefficient) < 0.0:
            raise ValidationError(f"coefficient must be >= 0, got {coefficient!r}")
        if int(steps) < 1:
            raise ValidationError(f"steps must be >= 1, got {steps!r}")
        self.coefficient = float(coefficient)
        self.steps = int(steps)
        self._calls = 0

    def _scale(self) -> float:
        return max(0.0, 1.0 - self._calls / self.steps)

    def __call__(self, module: torch.nn.Module) -> torch.Tensor:
        total = None
        for parameter in module.parameters():
            if not parameter.requires_grad:
                continue
            term = parameter.abs().sum()
            total = term if total is None else total + term
        scale = self._scale()
        self._calls += 1
        if total is None:
            return torch.zeros(())
        return self.coefficient * scale * total

    def reset(self) -> None:
        self._calls = 0

    def state_dict(self) -> dict[str, Any]:
        return {"type": self.spec_type, "class": type(self).__name__,
                "coefficient": self.coefficient, "steps": self.steps, "calls": self._calls}

    def load_state_dict(self, payload) -> None:
        self._calls = int(payload.get("calls", 0))

    def describe(self) -> dict[str, Any]:
        return {"type": self.spec_type, "class": type(self).__name__,
                "coefficient": self.coefficient, "steps": self.steps, "calls": self._calls}


class MedianSynthesis:
    """Tensor-level element-wise median (8.9.4)."""

    spec_family: ClassVar[str] = "synthesis"
    spec_type: ClassVar[str] = "median"

    def __call__(
        self, proposals: Mapping[str, torch.Tensor], *, weights: Mapping[str, float] | None = None,
    ) -> torch.Tensor:
        stacked = torch.stack([proposals[key] for key in sorted(proposals)], dim=0)
        return stacked.median(dim=0).values

    def describe(self) -> dict[str, Any]:
        return {"type": self.spec_type, "class": type(self).__name__}


class HuberValueLoss:
    """18.6: ``delta``-Huber between values and value targets."""

    required_fields = frozenset({"values", "value_targets"})
    probe = {"state_dim": 8}

    def __init__(self, *, delta: float = 1.0) -> None:
        if delta <= 0.0:
            raise ValueError("delta must be > 0")
        self.delta = float(delta)

    def compute(self, batch):
        import torch.nn.functional as F

        values, targets = batch["values"], batch["value_targets"]
        loss = F.huber_loss(values, targets.detach(), delta=self.delta)
        return loss, LossBreakdown(total=float(loss.detach()), extras={"huber": float(loss.detach())})


# -- worked example (section 18.5) --------------------------------------------------------
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


def step_then_flat(optimizer, *, drop_at: int, factor: float = 0.1):
    if drop_at < 1 or not 0.0 < factor <= 1.0:
        raise ValueError("drop_at must be >= 1 and factor in (0, 1]")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 if step < drop_at else factor)


# -- policies ----------------------------------------------------------------------------
def dummy_result(loss: torch.Tensor, **extras: float) -> StepResult:
    return StepResult(
        loss=loss, breakdown=LossBreakdown(total=float(loss.detach().sum()), extras=dict(extras)),
        stepped=True,
    )
