"""Scheduler factories and the one place a scheduler is built (document 17 section 9.3)."""

from __future__ import annotations

import math
import warnings
from typing import Any, Mapping

import torch
from torch.optim.lr_scheduler import LambdaLR

from ..components.registry import HUB
from ..exceptions import ComponentBuildError, ComponentMissing, ValidationError

#: ``ReduceLROnPlateau`` and friends take a metric in ``step``; the facade calls ``step()``.
UNSUPPORTED: frozenset[str] = frozenset({"ReduceLROnPlateau", "LRScheduler", "_LRScheduler", "ChainedScheduler", "SequentialLR"})

#: Replaying more steps than this to reach a carried position is refused (about a second).
MAX_FAST_FORWARD = 1_000_000

_TORCH_REGISTERED: tuple[str, ...] = (
    "StepLR", "MultiStepLR", "ConstantLR", "LinearLR", "ExponentialLR", "PolynomialLR",
    "CosineAnnealingLR", "CosineAnnealingWarmRestarts", "CyclicLR", "OneCycleLR",
)


def horizon(epochs: int, steps_per_epoch: int, grad_accumulation: int) -> int:
    """Optimizer steps one node takes in a whole run, an upper bound (a node that sits out an
    episode takes fewer). ``MACTFOptimizer.step`` calls ``scheduler.step()`` once per
    *effective* step, so accumulation divides the count."""
    micro = max(1, int(epochs)) * max(1, int(steps_per_epoch))
    return max(1, math.ceil(micro / max(1, int(grad_accumulation))))


def _check_horizon(name: str, warmup_steps: int, total_steps: int | None, min_lr_ratio: float = 0.0) -> None:
    if total_steps is None or int(total_steps) < 1:
        raise ValueError(f"{name}: total_steps must be >= 1, got {total_steps!r}")
    if int(warmup_steps) < 0 or int(warmup_steps) >= int(total_steps):
        raise ValueError(
            f"{name}: warmup_steps must be in [0, total_steps), got {warmup_steps!r} with "
            f"total_steps={total_steps!r}"
        )
    if not 0.0 <= float(min_lr_ratio) <= 1.0:
        raise ValueError(f"{name}: min_lr_ratio must be in [0, 1], got {min_lr_ratio!r}")


def _warmup(step: int, warmup_steps: int) -> float | None:
    """The warmup factor while warming up, else ``None``. Never 0, so step 0 trains."""
    if step < warmup_steps:
        return (step + 1) / (warmup_steps + 1)
    return None


def warmup_cosine(
    optimizer: torch.optim.Optimizer, *, warmup_steps: int = 0, total_steps: int,
    min_lr_ratio: float = 0.0, cycles: float = 0.5,
) -> LambdaLR:
    """Linear warmup, then cosine decay to ``min_lr_ratio`` of the base rate."""
    _check_horizon("warmup_cosine", warmup_steps, total_steps, min_lr_ratio)
    warmup_steps, total_steps = int(warmup_steps), int(total_steps)

    def factor(step: int) -> float:
        warm = _warmup(step, warmup_steps)
        if warm is not None:
            return warm
        span = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / span)
        cosine = 0.5 * (1.0 + math.cos(math.pi * 2.0 * float(cycles) * progress))
        return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * cosine

    return LambdaLR(optimizer, factor)


def warmup_linear(
    optimizer: torch.optim.Optimizer, *, warmup_steps: int = 0, total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Linear warmup, then linear decay to ``min_lr_ratio`` of the base rate."""
    _check_horizon("warmup_linear", warmup_steps, total_steps, min_lr_ratio)
    warmup_steps, total_steps = int(warmup_steps), int(total_steps)

    def factor(step: int) -> float:
        warm = _warmup(step, warmup_steps)
        if warm is not None:
            return warm
        span = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / span)
        return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * (1.0 - progress)

    return LambdaLR(optimizer, factor)


def constant_with_warmup(optimizer: torch.optim.Optimizer, *, warmup_steps: int) -> LambdaLR:
    """Linear warmup, then the base rate."""
    if int(warmup_steps) < 0:
        raise ValueError(f"constant_with_warmup: warmup_steps must be >= 0, got {warmup_steps!r}")
    warmup_steps = int(warmup_steps)

    def factor(step: int) -> float:
        warm = _warmup(step, warmup_steps)
        return 1.0 if warm is None else warm

    return LambdaLR(optimizer, factor)


def inverse_sqrt(optimizer: torch.optim.Optimizer, *, warmup_steps: int = 1) -> LambdaLR:
    """Linear warmup, then decay as the inverse square root of the step (Vaswani et al.)."""
    if int(warmup_steps) < 0:
        raise ValueError(f"inverse_sqrt: warmup_steps must be >= 0, got {warmup_steps!r}")
    warmup_steps = int(warmup_steps)

    def factor(step: int) -> float:
        warm = _warmup(step, warmup_steps)
        if warm is not None:
            return warm
        return math.sqrt((warmup_steps + 1) / (step + 1))

    return LambdaLR(optimizer, factor)


def register_builtins() -> None:
    """Register the built-in schedulers. Idempotent; called once at import."""
    registry = HUB.registry("scheduler")
    table: dict[str, Any] = {
        "warmup_cosine": warmup_cosine,
        "warmup_linear": warmup_linear,
        "constant_with_warmup": constant_with_warmup,
        "inverse_sqrt": inverse_sqrt,
    }
    for name in _TORCH_REGISTERED:
        factory = getattr(torch.optim.lr_scheduler, name, None)
        if isinstance(factory, type):
            table[name] = factory
    for name, factory in table.items():
        if registry.find(name) is None:
            registry.register_builtin(name, factory)


def accepts_total_steps(name: str) -> bool:
    """True when the scheduler named ``name`` takes a ``total_steps`` argument."""
    registry = HUB.registry("scheduler")
    component = registry.find(name)
    if component is not None:
        return "total_steps" in component.accepted_params()
    factory = getattr(torch.optim.lr_scheduler, name, None)
    if isinstance(factory, type):
        from ..components.reflect import constructor_keys

        return "total_steps" in constructor_keys(factory)
    return False


def build_scheduler(name: str, native: torch.optim.Optimizer, params: Mapping[str, Any] | None = None) -> Any:
    """Build the scheduler ``name`` over the torch optimizer ``native``.

    Lookup order: the scheduler registry, then ``torch.optim.lr_scheduler`` (the pre-17
    behaviour, so a class that is not registered still works).
    """
    if name in UNSUPPORTED:
        raise ValidationError(
            f"E052: scheduler {name!r} cannot be used: it needs a metric or another scheduler, "
            "and MACTFOptimizer calls step() with no arguments once per optimizer step"
        )
    values = dict(params or {})
    registry = HUB.registry("scheduler")
    if registry.find(name) is not None:
        return registry.build(name, values, inject={"optimizer": native})
    torch_class = getattr(torch.optim.lr_scheduler, str(name), None)
    if isinstance(torch_class, type):
        try:
            return torch_class(native, **values)
        except (TypeError, ValueError) as error:
            raise ComponentBuildError("scheduler", str(name), error) from error
    raise ComponentMissing("scheduler", str(name), registry.names())


def position_of(scheduler: Any) -> int | None:
    """How many ``step()`` calls a scheduler has seen, or ``None`` when it will not say."""
    value = getattr(scheduler, "last_epoch", None)
    if isinstance(value, int) and not isinstance(value, bool):
        return max(0, value)
    getter = getattr(scheduler, "state_dict", None)
    if callable(getter):
        try:
            stored = getter().get("last_epoch")
        except Exception:  # noqa: BLE001 - a plugin scheduler's state_dict may be strict
            return None
        if isinstance(stored, int) and not isinstance(stored, bool):
            return max(0, stored)
    return None


def fast_forward(scheduler: Any, steps: int) -> None:
    """Replay ``steps`` ``step()`` calls so a fresh scheduler stands where an old one stood.

    Replay rather than ``load_state_dict`` because a state dict also carries ``base_lrs``,
    and loading it would silently put the *old* base learning rate back over the new spec's.
    """
    if steps < 0:
        raise ValueError(f"cannot fast-forward a scheduler by {steps} steps")
    if steps > MAX_FAST_FORWARD:
        raise ValueError(
            f"refusing to replay {steps} scheduler steps (limit {MAX_FAST_FORWARD}); "
            "swap with migrate: reset instead"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch warns "step() before optimizer.step()"
        for _ in range(steps):
            scheduler.step()


register_builtins()
