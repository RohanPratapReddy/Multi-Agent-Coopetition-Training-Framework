"""Advantage-function wrappers (document 07 section 8.2).

Each wrapper delegates to the free function in :mod:`mactf.rl.advantage` or
:mod:`mactf.rl.value`; none of them relax a validation the free function
performs, and none of them re-implement an equation.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Mapping, Protocol, Sequence, runtime_checkable

import torch

from ...exceptions import ContractError, ValidationError
from ..advantage import (
    AdvantageBatch,
    generalized_advantage,
    penalise_void_advantage,
    relative_to_reference,
    temporal_difference,
)
from ..value import normalise_advantages
from ._spec import SpecMixin, build_from_kwargs


@runtime_checkable
class AdvantageFunction(Protocol):
    """Turns rewards and values into an :class:`AdvantageBatch`."""

    def __call__(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        *,
        next_values: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        reference: torch.Tensor | None = None,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> AdvantageBatch: ...

    def describe(self) -> dict[str, Any]: ...


def _check_scalars(gamma: float, lam: float) -> None:
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= float(lam) <= 1.0:
        raise ValueError("lam must be in [0, 1]")


class _AdvantageBase(SpecMixin):
    spec_family = "advantage"
    spec_type: ClassVar[str] = ""

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "_AdvantageBase":
        payload = dict(spec or {})
        kind = payload.pop("type", None)
        if kind is not None and kind != cls.spec_type:
            raise ValidationError(
                f"{cls.__name__}.from_spec cannot build type {kind!r}; it builds {cls.spec_type!r}"
            )
        return build_from_kwargs(cls, payload)

    def describe(self) -> dict[str, Any]:
        return {"type": self.spec_type, "class": type(self).__name__}

    def __call__(self, rewards, values, **kwargs) -> AdvantageBatch:  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"


class TDAdvantage(_AdvantageBase):
    """Single-step TD, exactly ``rewards + gamma * next - values``."""

    spec_type = "td"

    def __init__(self, *, gamma: float = 0.99) -> None:
        _check_scalars(gamma, 0.0)
        self.gamma = float(gamma)

    def __call__(
        self,
        rewards,
        values,
        *,
        next_values=None,
        masks=None,
        reference=None,
        gamma=None,
        lam=None,
    ):
        if next_values is None:
            raise ValueError("TDAdvantage requires next_values")
        rate = self.gamma if gamma is None else float(gamma)
        _check_scalars(rate, 0.0)
        if torch.is_tensor(rewards) and torch.is_tensor(values):
            if rewards.shape != values.shape or values.shape != next_values.shape:
                raise ValueError(
                    "rewards, values and next_values must have the same shape"
                )
        advantages = temporal_difference(
            torch.as_tensor(rewards).float(),
            torch.as_tensor(values).float(),
            torch.as_tensor(next_values).float(),
            gamma=rate,
        )
        if masks is not None:
            advantages = advantages * torch.as_tensor(masks).bool()
        return AdvantageBatch(
            advantages=advantages, returns=advantages + torch.as_tensor(values).float()
        )

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "gamma": self.gamma}


class GAEAdvantage(_AdvantageBase):
    """GAE-lambda, delegating to :func:`mactf.rl.advantage.generalized_advantage`."""

    spec_type = "gae"

    def __init__(self, *, gamma: float = 0.99, lam: float = 0.95, next_value: float = 0.0) -> None:
        _check_scalars(gamma, lam)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.next_value = float(next_value)

    def __call__(
        self,
        rewards,
        values,
        *,
        next_values=None,
        masks=None,
        reference=None,
        gamma=None,
        lam=None,
    ):
        rate = self.gamma if gamma is None else float(gamma)
        lam_value = self.lam if lam is None else float(lam)
        _check_scalars(rate, lam_value)
        if torch.is_tensor(rewards) and torch.is_tensor(values):
            if rewards.reshape(-1).numel() != values.reshape(-1).numel():
                raise ValueError("rewards and values must have the same length")
        next_value = self.next_value
        if next_values is not None:
            tensor = torch.as_tensor(next_values).float().reshape(-1)
            if tensor.numel() == 1:
                next_value = float(tensor.item())
            elif tensor.numel() > 1:
                raise ValueError(
                    "GAEAdvantage takes a single next value through 'next_values'; "
                    "pass the bootstrap value as one element"
                )
        batch = generalized_advantage(
            rewards, values, gamma=rate, lam=lam_value, next_value=next_value
        )
        if masks is not None:
            mask = torch.as_tensor(masks).bool()
            if mask.shape != batch.advantages.shape:
                raise ValueError("advantages and masks must have the same shape")
            batch = AdvantageBatch(
                advantages=torch.where(mask, batch.advantages, torch.zeros_like(batch.advantages)),
                returns=batch.returns,
            )
        return batch

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "gamma": self.gamma,
            "lam": self.lam,
            "next_value": self.next_value,
        }


class ReferenceBaselineAdvantage(_AdvantageBase):
    """Subtracts a reference signal rather than the peer's own mean (11.7)."""

    spec_type = "reference"

    def __call__(
        self,
        rewards,
        values,
        *,
        next_values=None,
        masks=None,
        reference=None,
        gamma=0.99,
        lam=0.95,
    ):
        if reference is None:
            raise ValueError(
                "ReferenceBaselineAdvantage requires 'reference'; a peer's own mean is "
                "not a legal baseline (framework.md 11.7)"
            )
        peer = torch.as_tensor(rewards).float()
        baseline = torch.as_tensor(reference).float()
        advantages = relative_to_reference(peer, baseline)
        # ``masks`` is deliberately not applied here. Masking is the void policy's
        # job and belongs to :class:`VoidMaskedAdvantage`; applying it here too
        # would zero a masked sample before the chain decided what masking means.
        base_values = torch.as_tensor(values).float() if values is not None else None
        returns = advantages if base_values is None else advantages + base_values.reshape_as(
            advantages
        )
        return AdvantageBatch(advantages=advantages, returns=returns)


class VoidMaskedAdvantage(_AdvantageBase):
    """Forces non-positive advantages where a non-offsettable breach applies."""

    spec_type = "void_masked"

    def __init__(self, *, floor: float = -1.0) -> None:
        if float(floor) > 0.0:
            raise ValidationError(
                f"floor={floor!r} would make a voided advantage positive; use <= 0"
            )
        self.floor = float(floor)

    def __call__(
        self,
        rewards,
        values,
        *,
        next_values=None,
        masks=None,
        reference=None,
        gamma=0.99,
        lam=0.95,
    ):
        if masks is None:
            raise ValueError("VoidMaskedAdvantage requires 'masks' (the void mask)")
        advantages = torch.as_tensor(rewards).float()
        if reference is not None:
            advantages = relative_to_reference(advantages, torch.as_tensor(reference).float())
        masked = penalise_void_advantage(
            advantages, torch.as_tensor(masks).bool(), floor=self.floor
        )
        base_values = torch.as_tensor(values).float() if values is not None else None
        returns = masked if base_values is None else masked + base_values.reshape_as(masked)
        return AdvantageBatch(advantages=masked, returns=returns)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "floor": self.floor}


class NormalisedAdvantage(_AdvantageBase):
    """Post-processes with :func:`mactf.rl.value.normalise_advantages`."""

    spec_type = "normalised"

    def __init__(self, *, eps: float = 1e-8) -> None:
        if float(eps) <= 0.0:
            raise ValidationError(f"eps must be positive, got {eps!r}")
        self.eps = float(eps)

    def __call__(
        self,
        rewards,
        values,
        *,
        next_values=None,
        masks=None,
        reference=None,
        gamma=0.99,
        lam=0.95,
    ):
        advantages = normalise_advantages(torch.as_tensor(rewards).float(), eps=self.eps)
        if masks is not None:
            mask = torch.as_tensor(masks).bool()
            if mask.shape != advantages.shape:
                raise ValueError("advantages and masks must have the same shape")
            advantages = torch.where(mask, advantages, torch.zeros_like(advantages))
        base_values = torch.as_tensor(values).float() if values is not None else None
        returns = advantages if base_values is None else advantages + base_values.reshape_as(
            advantages
        )
        return AdvantageBatch(advantages=advantages, returns=returns)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "eps": self.eps}


class ChainAdvantage(_AdvantageBase):
    """Applies a list of advantage transforms in order.

    Semantics, stated because they are the one thing a reader cannot infer: each
    step receives the previous step's advantages as its ``rewards`` argument and
    the chain's original ``values`` unchanged, so a step is a pure transform of an
    advantage tensor. The returned ``returns`` are ``final_advantages + values``.

    That is exactly how ``local_update`` expresses the current behaviour:
    ``ChainAdvantage([ReferenceBaselineAdvantage(), VoidMaskedAdvantage()])``
    called with ``reference=zeros`` produces ``penalise_void_advantage(signal,
    void)`` — the advantages the existing loop computes (``phases.py:239-247``).
    """

    spec_type = "chain"

    def __init__(self, steps: Sequence[AdvantageFunction] | None = None, *, strict: bool = True) -> None:
        self.steps: list[Any] = list(steps or [])
        self.strict = bool(strict)
        for step in self.steps:
            if not callable(step):
                raise ValidationError(
                    f"ChainAdvantage step is not callable: {step!r}"
                )
        if self.strict:
            for step in self.steps:
                if not hasattr(step, "describe"):
                    raise ValidationError(
                        f"ChainAdvantage step {type(step).__name__} has no describe(); "
                        "every wrapper implements it"
                    )

    def add(self, step: AdvantageFunction) -> "ChainAdvantage":
        self.steps.append(step)
        return self

    def __call__(
        self,
        rewards,
        values,
        *,
        next_values=None,
        masks=None,
        reference=None,
        gamma=0.99,
        lam=0.95,
    ):
        if not self.steps:
            advantages = torch.as_tensor(rewards).float()
            base = torch.as_tensor(values).float() if values is not None else None
            return AdvantageBatch(
                advantages=advantages,
                returns=advantages if base is None else advantages + base.reshape_as(advantages),
            )
        current = torch.as_tensor(rewards).float()
        shape = current.shape
        for step in self.steps:
            batch = step(
                current,
                values,
                next_values=next_values,
                masks=masks,
                reference=reference,
                gamma=gamma,
                lam=lam,
            )
            if self.strict and batch.advantages.shape != shape:
                raise ContractError(
                    f"ChainAdvantage step {type(step).__name__} changed the advantage "
                    f"shape from {tuple(shape)} to {tuple(batch.advantages.shape)}"
                )
            current = batch.advantages
        base = torch.as_tensor(values).float() if values is not None else None
        returns = current if base is None else current + base.reshape_as(current)
        return AdvantageBatch(advantages=current, returns=returns)

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "steps": [
                step.describe() if hasattr(step, "describe") else type(step).__name__
                for step in self.steps
            ],
        }

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "ChainAdvantage":
        from .registry import build_function

        payload = dict(spec or {})
        kind = payload.pop("type", None)
        if kind is not None and kind != cls.spec_type:
            raise ValidationError(
                f"ChainAdvantage.from_spec cannot build type {kind!r}; it builds 'chain'"
            )
        raw_steps = payload.pop("steps", None)
        if raw_steps is None:
            raise ValidationError("ChainAdvantage spec requires a 'steps' list")
        if not isinstance(raw_steps, (list, tuple)):
            raise ValidationError("ChainAdvantage 'steps' must be a list of specs")
        steps = [build_function(step, family="advantage") for step in raw_steps]
        return build_from_kwargs(cls, {**payload, "steps": steps})


class CallableAdvantage(_AdvantageBase):
    """Adapts ``fn(rewards, values, **kwargs) -> AdvantageBatch | Tensor``."""

    spec_type = "callable"

    def __init__(self, fn: Callable[..., Any] | None = None, *, name: str = "") -> None:
        if fn is None:
            raise ValidationError("CallableAdvantage requires an 'fn' callable")
        if not callable(fn):
            raise ValidationError(f"CallableAdvantage fn is not callable: {fn!r}")
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "")

    def __call__(
        self,
        rewards,
        values,
        *,
        next_values=None,
        masks=None,
        reference=None,
        gamma=0.99,
        lam=0.95,
    ):
        result = self.fn(
            rewards,
            values,
            next_values=next_values,
            masks=masks,
            reference=reference,
            gamma=gamma,
            lam=lam,
        )
        if isinstance(result, AdvantageBatch):
            return result
        if torch.is_tensor(result):
            base = torch.as_tensor(values).float() if values is not None else None
            return AdvantageBatch(
                advantages=result,
                returns=result if base is None else result + base.reshape_as(result),
            )
        raise ContractError(
            f"CallableAdvantage returned {type(result).__name__}, "
            "expected AdvantageBatch or Tensor"
        )

    def describe(self) -> dict[str, Any]:
        payload = {**super().describe(), "fn": self.name}
        return payload


ADVANTAGE_FUNCTION_TYPES: dict[str, type] = {
    "td": TDAdvantage,
    "gae": GAEAdvantage,
    "reference": ReferenceBaselineAdvantage,
    "void_masked": VoidMaskedAdvantage,
    "normalised": NormalisedAdvantage,
    "chain": ChainAdvantage,
    "callable": CallableAdvantage,
}

ADVANTAGE_CLASSES = tuple(ADVANTAGE_FUNCTION_TYPES.values())


__all__ = [
    "ADVANTAGE_CLASSES",
    "ADVANTAGE_FUNCTION_TYPES",
    "AdvantageFunction",
    "CallableAdvantage",
    "ChainAdvantage",
    "GAEAdvantage",
    "NormalisedAdvantage",
    "ReferenceBaselineAdvantage",
    "TDAdvantage",
    "VoidMaskedAdvantage",
]
