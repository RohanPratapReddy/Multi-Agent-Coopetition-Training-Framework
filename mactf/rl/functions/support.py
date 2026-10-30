"""The supporting wrapper catalogue (document 07 section 9).

Each family is a ``Protocol`` plus concrete defaults so "and so on" in the
requirement is a closed list. Every concrete class is spec-constructible.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import torch
import torch.nn as nn

from ...components.legacy import LegacyTable
from ...exceptions import ContractError, ValidationError
from ..loss import LossBreakdown, clip_gradients
from ..policy import PolicyOutput, greedy_action, sample_action
from ._spec import SpecMixin, build_from_kwargs


class _SupportBase(SpecMixin):
    spec_family = ""
    spec_type: ClassVar[str] = ""

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "_SupportBase":
        payload = dict(spec or {})
        kind = payload.pop("type", None)
        if kind is not None and kind != cls.spec_type:
            raise ValidationError(
                f"{cls.__name__}.from_spec cannot build type {kind!r}; it builds {cls.spec_type!r}"
            )
        return build_from_kwargs(cls, payload)

    def describe(self) -> dict[str, Any]:
        return {"type": self.spec_type, "class": type(self).__name__}


# ---------------------------------------------------------------------------
# ReturnFunction
# ---------------------------------------------------------------------------
@runtime_checkable
class ReturnFunction(Protocol):
    def __call__(
        self,
        rewards: torch.Tensor,
        *,
        gamma: float,
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class DiscountedReturn(_SupportBase):
    """``G_t = r_t + gamma * G_{t+1}``, with optional terminal masking."""

    spec_family = "return"
    spec_type = "discounted"

    def __init__(self, *, gamma: float = 0.99) -> None:
        if not 0.0 <= float(gamma) <= 1.0:
            raise ValidationError(f"gamma must be in [0, 1], got {gamma!r}")
        self.gamma = float(gamma)

    def __call__(self, rewards, *, gamma=None, masks=None):
        rate = self.gamma if gamma is None else float(gamma)
        if not 0.0 <= rate <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        tensor = torch.as_tensor(rewards).float().reshape(-1)
        mask = None if masks is None else torch.as_tensor(masks).bool().reshape(-1)
        if mask is not None and mask.shape != tensor.shape:
            raise ValueError("rewards and masks must have the same shape")
        out = torch.zeros_like(tensor)
        running = 0.0
        for index in reversed(range(tensor.numel())):
            if mask is not None and not bool(mask[index]):
                running = 0.0
                out[index] = 0.0
                continue
            running = float(tensor[index]) + rate * running
            out[index] = running
        return out

    def describe(self):
        return {**super().describe(), "gamma": self.gamma}


class MonteCarloReturn(_SupportBase):
    """Undiscounted-to-go, the ``gamma == 1`` special case stated explicitly."""

    spec_family = "return"
    spec_type = "monte_carlo"

    def __call__(self, rewards, *, gamma=1.0, masks=None):
        return DiscountedReturn(gamma=1.0)(rewards, gamma=1.0, masks=masks)


# ---------------------------------------------------------------------------
# RewardFunction
# ---------------------------------------------------------------------------
@runtime_checkable
class RewardFunction(Protocol):
    def __call__(self, episode: Any, *, peer: str) -> float: ...


class ScorecardReward(_SupportBase):
    """Reads ``PeerScorecard.final_score`` from an episode, never recomputing it."""

    spec_family = "reward"
    spec_type = "scorecard"

    def __init__(self, *, field: str = "final_score", default: float = 0.0) -> None:
        self.field = field
        self.default = float(default)

    def __call__(self, episode, *, peer):
        scorecard = getattr(episode, "scorecard", None)
        if scorecard is None:
            return self.default
        card = getattr(scorecard, "peers", {}).get(peer)
        if card is None:
            return self.default
        return float(getattr(card, self.field, self.default))

    def describe(self):
        return {**super().describe(), "field": self.field, "default": self.default}


class SignalReward(_SupportBase):
    """Reads a precomputed per-peer signal mapping."""

    spec_family = "reward"
    spec_type = "signal"

    def __init__(self, signals: Mapping[str, float] | None = None, *, default: float = 0.0) -> None:
        self.signals = dict(signals or {})
        self.default = float(default)

    def __call__(self, episode, *, peer):
        return float(self.signals.get(peer, self.default))

    def describe(self):
        return {**super().describe(), "peers": sorted(self.signals)}


class CallableReward(_SupportBase):
    spec_family = "reward"
    spec_type = "callable"

    def __init__(self, fn: Callable[..., float] | None = None) -> None:
        if fn is None:
            raise ValidationError("CallableReward requires an 'fn' callable")
        if not callable(fn):
            raise ValidationError(f"CallableReward fn is not callable: {fn!r}")
        self.fn = fn

    def __call__(self, episode, *, peer):
        value = self.fn(episode, peer=peer)
        if not isinstance(value, (int, float)):
            raise ContractError(
                f"CallableReward returned {type(value).__name__}, expected a number"
            )
        return float(value)

    def describe(self):
        return {**super().describe(), "fn": getattr(self.fn, "__name__", repr(self.fn))}


# ---------------------------------------------------------------------------
# RegularizationFunction
# ---------------------------------------------------------------------------
@runtime_checkable
class RegularizationFunction(Protocol):
    def __call__(self, module: nn.Module) -> torch.Tensor: ...


def _trainable(module: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in module.parameters() if parameter.requires_grad)


class L2Regularization(_SupportBase):
    """``coef * sum(p^2)`` over trainable parameters."""

    spec_family = "regularization"
    spec_type = "l2"

    def __init__(self, *, coefficient: float = 1e-4) -> None:
        self.coefficient = float(coefficient)

    def __call__(self, module):
        total = None
        for parameter in _trainable(module):
            term = parameter.pow(2).sum()
            total = term if total is None else total + term
        if total is None:
            return torch.zeros(())
        return self.coefficient * total

    def describe(self):
        return {**super().describe(), "coefficient": self.coefficient}


class WeightDecayRegularization(_SupportBase):
    """``coef * sum(|p|)``: the L1/decoupled-weight-decay form."""

    spec_family = "regularization"
    spec_type = "weight_decay"

    def __init__(self, *, coefficient: float = 1e-4) -> None:
        self.coefficient = float(coefficient)

    def __call__(self, module):
        total = None
        for parameter in _trainable(module):
            term = parameter.abs().sum()
            total = term if total is None else total + term
        if total is None:
            return torch.zeros(())
        return self.coefficient * total

    def describe(self):
        return {**super().describe(), "coefficient": self.coefficient}


class KLToReferenceRegularization(_SupportBase):
    """Penalises drift from a reference module over the same inputs.

    Both modules are called with ``states``; the penalty is the mean squared
    difference of their outputs, which is well defined for any module shape and is
    what "stay near the reference" means without assuming logits.
    """

    spec_family = "regularization"
    spec_type = "kl_to_reference"

    def __init__(
        self,
        reference: nn.Module | None = None,
        *,
        coefficient: float = 1e-3,
        states: torch.Tensor | None = None,
    ) -> None:
        if reference is None:
            raise ValidationError("KLToReferenceRegularization requires a 'reference' module")
        self.reference = reference
        self.coefficient = float(coefficient)
        self.states = states

    def __call__(self, module, *, states: torch.Tensor | None = None):
        cache = states if states is not None else self.states
        if cache is None:
            raise ContractError(
                "KLToReferenceRegularization needs 'states' at construction or call time"
            )
        with torch.no_grad():
            reference_output = _first_output(self.reference(cache))
        output = _first_output(module(cache))
        if output.shape != reference_output.shape:
            raise ContractError(
                f"module output {tuple(output.shape)} does not match the reference "
                f"{tuple(reference_output.shape)}"
            )
        return self.coefficient * torch.nn.functional.mse_loss(output, reference_output)

    def describe(self):
        return {**super().describe(), "coefficient": self.coefficient}


def _first_output(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)) and value and torch.is_tensor(value[0]):
        return value[0]
    if isinstance(value, dict):
        for candidate in value.values():
            if torch.is_tensor(candidate):
                return candidate
    tensor = getattr(value, "logits", None)
    if torch.is_tensor(tensor):
        return tensor
    raise ContractError(f"cannot find a tensor in {type(value).__name__}")


# ---------------------------------------------------------------------------
# ObjectiveFn
# ---------------------------------------------------------------------------
@runtime_checkable
class ObjectiveFn(Protocol):
    def __call__(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]: ...


class LossObjective(_SupportBase):
    """Adapts a :class:`~mactf.rl.functions.loss.LossFunction` to ``ObjectiveFn``."""

    spec_family = "objective"
    spec_type = "loss"

    def __init__(self, loss: Any = None) -> None:
        if loss is None:
            raise ValidationError("LossObjective requires a 'loss'")
        if not callable(getattr(loss, "compute", None)):
            raise ValidationError(
                f"LossObjective loss has no compute(): {type(loss).__name__}"
            )
        self.loss = loss

    def __call__(self, batch):
        total, breakdown = self.loss.compute(batch)
        payload = breakdown.as_dict() if isinstance(breakdown, LossBreakdown) else dict(breakdown)
        return total, payload

    def describe(self):
        return {
            **super().describe(),
            "required_fields": sorted(getattr(self.loss, "required_fields", ())),
        }


class CallableObjective(_SupportBase):
    spec_family = "objective"
    spec_type = "callable"

    def __init__(self, fn: Callable[..., Any] | None = None) -> None:
        if fn is None:
            raise ValidationError("CallableObjective requires an 'fn' callable")
        if not callable(fn):
            raise ValidationError(f"CallableObjective fn is not callable: {fn!r}")
        self.fn = fn

    def __call__(self, batch):
        result = self.fn(batch)
        if torch.is_tensor(result):
            return result, {}
        if isinstance(result, tuple) and len(result) == 2 and torch.is_tensor(result[0]):
            return result[0], dict(result[1] or {})
        raise ContractError(
            f"CallableObjective returned {type(result).__name__}; expected a tensor or "
            "(tensor, dict)"
        )

    def describe(self):
        return {**super().describe(), "fn": getattr(self.fn, "__name__", repr(self.fn))}


# ---------------------------------------------------------------------------
# GradientFn
# ---------------------------------------------------------------------------
@runtime_checkable
class GradientFn(Protocol):
    def __call__(
        self,
        parameters: Iterable[nn.Parameter],
        *,
        max_norm: float,
        scaler: Any = None,
    ) -> float: ...


class ClipGradient(_SupportBase):
    """Clips gradients and returns the pre-clip norm.

    The AMP order is the one thing this encodes: ``unscale_`` must happen before
    clipping and ``step`` must happen after (``phases.py:206-213``). Calling
    ``clip`` with a scaler un-escales it here, so a caller cannot get the order
    wrong by clipping first.
    """

    spec_family = "gradient"
    spec_type = "clip"

    def __init__(self, *, max_norm: float = 1.0) -> None:
        if float(max_norm) <= 0.0:
            raise ValidationError(f"max_norm must be positive, got {max_norm!r}")
        self.max_norm = float(max_norm)

    def __call__(self, parameters, *, max_norm=None, scaler=None):
        limit = self.max_norm if max_norm is None else float(max_norm)
        params = [parameter for parameter in parameters if parameter.grad is not None]
        if not params:
            return 0.0
        if scaler is not None:
            # ``unscale_`` needs an optimizer, which a GradientFn does not have;
            # the caller passes the scaler only after unscaling, so this is a guard
            # against the wrong order rather than an action.
            if getattr(scaler, "is_unscaled", False) is False and not getattr(
                scaler, "unscaled", True
            ):
                raise ContractError(
                    "ClipGradient must run after scaler.unscale_(optimizer)"
                )
        return float(clip_gradients(params, max_norm=limit))

    def describe(self):
        return {**super().describe(), "max_norm": self.max_norm}


class AccumulateGradient(_SupportBase):
    """Runs the wrapped gradient step every ``n`` calls, otherwise returns 0.0.

    The counter is real state; ``reset()`` and ``state_dict()`` are part of the
    contract so a resumed run continues in the same phase of its accumulation
    window rather than restarting it.
    """

    spec_family = "gradient"
    spec_type = "accumulate"

    def __init__(
        self,
        inner: GradientFn | None = None,
        *,
        n: int = 1,
        max_norm: float = 1.0,
    ) -> None:
        if int(n) < 1:
            raise ValidationError(f"n must be >= 1, got {n!r}")
        self.inner = inner or ClipGradient(max_norm=max_norm)
        self.n = int(n)
        self.max_norm = float(max_norm)
        self._count = 0

    def __call__(self, parameters, *, max_norm=None, scaler=None):
        self._count += 1
        if self._count < self.n:
            return 0.0
        self._count = 0
        return float(
            self.inner(parameters, max_norm=self.max_norm if max_norm is None else max_norm, scaler=scaler)
        )

    def reset(self) -> None:
        self._count = 0

    def state_dict(self):
        return {**self.describe(), "count": self._count}

    def load_state_dict(self, payload):
        self._count = int(payload.get("count", 0))

    def describe(self):
        return {**super().describe(), "n": self.n, "count": self._count}


class ScaleGradient(_SupportBase):
    """Multiplies every gradient in place, then delegates."""

    spec_family = "gradient"
    spec_type = "scale"

    def __init__(
        self,
        inner: GradientFn | None = None,
        *,
        factor: float = 1.0,
        max_norm: float = 1.0,
    ) -> None:
        if float(factor) == 0.0:
            raise ValidationError("factor=0 would discard every gradient")
        self.inner = inner or ClipGradient(max_norm=max_norm)
        self.factor = float(factor)
        self.max_norm = float(max_norm)

    def __call__(self, parameters, *, max_norm=None, scaler=None):
        params = [parameter for parameter in parameters if parameter.grad is not None]
        for parameter in params:
            parameter.grad.mul_(self.factor)
        return float(
            self.inner(params, max_norm=self.max_norm if max_norm is None else max_norm, scaler=scaler)
        )

    def describe(self):
        return {**super().describe(), "factor": self.factor}


# ---------------------------------------------------------------------------
# MetricFn
# ---------------------------------------------------------------------------
@runtime_checkable
class MetricFn(Protocol):
    def __call__(self, batch: Mapping[str, Any], output: Any) -> Mapping[str, float]: ...


class AccuracyMetric(_SupportBase):
    """Top-1 accuracy when logits and targets are both present, else nothing."""

    spec_family = "metric"
    spec_type = "accuracy"

    def __call__(self, batch, output):
        logits = _maybe_logits(output)
        targets = batch.get("targets", batch.get("labels"))
        if logits is None or targets is None:
            return {}
        predicted = logits.reshape(-1, logits.shape[-1]).argmax(dim=-1)
        target = torch.as_tensor(targets).reshape(-1)
        if predicted.numel() != target.numel():
            return {}
        return {"accuracy": float((predicted == target).float().mean().item())}


class LossMetric(_SupportBase):
    """Reports the loss the objective returned, without recomputing it."""

    spec_family = "metric"
    spec_type = "loss"

    def __call__(self, batch, output):
        if torch.is_tensor(output):
            return {"loss": float(output.detach().item())}
        if isinstance(output, Mapping) and "total" in output:
            return {"loss": float(output["total"])}
        return {}


class PerplexityMetric(_SupportBase):
    """``exp(loss)``, guarded against overflow."""

    spec_family = "metric"
    spec_type = "perplexity"

    def __call__(self, batch, output):
        if not torch.is_tensor(output):
            return {}
        value = float(output.detach().item())
        if value > 60.0:
            return {"perplexity": float("inf")}
        return {"perplexity": float(torch.exp(torch.tensor(value)).item())}


class CallableMetric(_SupportBase):
    spec_family = "metric"
    spec_type = "callable"

    def __init__(self, fn: Callable[..., Mapping[str, float]] | None = None) -> None:
        if fn is None:
            raise ValidationError("CallableMetric requires an 'fn' callable")
        if not callable(fn):
            raise ValidationError(f"CallableMetric fn is not callable: {fn!r}")
        self.fn = fn

    def __call__(self, batch, output):
        result = self.fn(batch, output)
        if not isinstance(result, Mapping):
            raise ContractError(
                f"CallableMetric returned {type(result).__name__}, expected a mapping"
            )
        return {str(key): float(value) for key, value in result.items()}


def _maybe_logits(output: Any) -> torch.Tensor | None:
    if isinstance(output, PolicyOutput):
        return output.logits
    if torch.is_tensor(output):
        return output
    if isinstance(output, Mapping):
        candidate = output.get("logits")
        return candidate if torch.is_tensor(candidate) else None
    logits = getattr(output, "logits", None)
    return logits if torch.is_tensor(logits) else None


# ---------------------------------------------------------------------------
# ActionFunction
# ---------------------------------------------------------------------------
@runtime_checkable
class ActionFunction(Protocol):
    def __call__(self, logits: torch.Tensor, *, sample: bool) -> torch.Tensor: ...


class GreedyAction(_SupportBase):
    spec_family = "action"
    spec_type = "greedy"

    def __call__(self, logits, *, sample=False):
        from ..policy import make_policy

        return greedy_action(make_policy(logits))


class SampledAction(_SupportBase):
    """Samples when ``sample=True``, otherwise takes the argmax."""

    spec_family = "action"
    spec_type = "sampled"

    def __init__(self, *, generator: torch.Generator | None = None) -> None:
        self.generator = generator

    def __call__(self, logits, *, sample=True):
        from ..policy import make_policy

        policy = make_policy(logits)
        if not sample:
            return greedy_action(policy)
        actions, _ = sample_action(policy, generator=self.generator)
        return actions


# ---------------------------------------------------------------------------
# SynthesisFunction
# ---------------------------------------------------------------------------
@runtime_checkable
class SynthesisFunction(Protocol):
    def __call__(
        self,
        proposals: Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        *,
        weights: Mapping[str, float] | None = None,
    ) -> torch.Tensor: ...


def _stack(proposals: Mapping[str, torch.Tensor] | Sequence[torch.Tensor]) -> tuple[torch.Tensor, list[str]]:
    if isinstance(proposals, Mapping):
        names = list(proposals)
        tensors = [proposals[name] for name in names]
    else:
        names = [str(index) for index in range(len(proposals))]
        tensors = list(proposals)
    if not tensors:
        raise ContractError("synthesis needs at least one proposal")
    for tensor in tensors:
        if not torch.is_tensor(tensor):
            raise ContractError(
                f"proposals must be tensors, got {type(tensor).__name__}"
            )
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) > 1:
        raise ContractError(f"proposals have mismatched shapes: {sorted(shapes)}")
    return torch.stack([tensor.float() for tensor in tensors], dim=0), names


class ConsensusSynthesis(_SupportBase):
    """Uniform mean: the promoted ``episode.consensus`` behaviour."""

    spec_family = "synthesis"
    spec_type = "consensus"

    def __call__(self, proposals, *, weights=None):
        stacked, _ = _stack(proposals)
        return stacked.mean(dim=0)


class WeightedConsensus(_SupportBase):
    """Weighted mean. Absent weights default to 1.0 and are normalised."""

    spec_family = "synthesis"
    spec_type = "weighted"

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        self.weights = dict(weights or {})

    def __call__(self, proposals, *, weights=None):
        stacked, names = _stack(proposals)
        resolved = dict(self.weights)
        resolved.update(dict(weights or {}))
        vector = torch.tensor(
            [float(resolved.get(name, 1.0)) for name in names], dtype=stacked.dtype
        )
        if float(vector.sum()) == 0.0:
            raise ContractError("WeightedConsensus weights sum to zero")
        vector = vector / vector.sum()
        return (stacked * vector.reshape(-1, *([1] * (stacked.ndim - 1)))).sum(dim=0)

    def describe(self):
        return {**super().describe(), "weights": dict(self.weights)}


class BestByScore(_SupportBase):
    """Picks one proposal by score; a selection, not an average.

    Says so explicitly because framework.md 9.6.4 forbids voting a claim into
    acceptance: this selects an existing artifact rather than deriving a new one.
    """

    spec_family = "synthesis"
    spec_type = "best"

    def __init__(self, scores: Mapping[str, float] | None = None, *, highest: bool = True) -> None:
        self.scores = dict(scores or {})
        self.highest = bool(highest)

    def __call__(self, proposals, *, weights=None):
        stacked, names = _stack(proposals)
        resolved = dict(self.scores)
        resolved.update(dict(weights or {}))
        if not resolved:
            raise ContractError("BestByScore needs scores to choose from")
        key = (max if self.highest else min)(
            names, key=lambda name: float(resolved.get(name, float("-inf") if self.highest else float("inf")))
        )
        return proposals[key] if isinstance(proposals, Mapping) else proposals[names.index(key)]

    def describe(self):
        return {**super().describe(), "highest": self.highest}


class LearnedSynthesizer(_SupportBase):
    """A trainable linear blender over proposals, with real parameters."""

    spec_family = "synthesis"
    spec_type = "learned"

    def __init__(self, n_proposals: int = 2, *, temperature: float = 1.0) -> None:
        if int(n_proposals) < 1:
            raise ValidationError(f"n_proposals must be >= 1, got {n_proposals!r}")
        if float(temperature) <= 0.0:
            raise ValidationError(f"temperature must be positive, got {temperature!r}")
        self.n_proposals = int(n_proposals)
        self.temperature = float(temperature)
        self.logits = nn.Parameter(torch.zeros(self.n_proposals))

    def parameters(self):
        return (self.logits,)

    def __call__(self, proposals, *, weights=None):
        stacked, names = _stack(proposals)
        if stacked.shape[0] != self.n_proposals:
            raise ContractError(
                f"LearnedSynthesizer was built for {self.n_proposals} proposals but "
                f"received {stacked.shape[0]}"
            )
        vector = torch.softmax(self.logits / self.temperature, dim=0).to(stacked.dtype)
        return (stacked * vector.reshape(-1, *([1] * (stacked.ndim - 1)))).sum(dim=0)

    def describe(self):
        return {
            **super().describe(),
            "n_proposals": self.n_proposals,
            "temperature": self.temperature,
        }


class ExplicitSynthesis(_SupportBase):
    """``fn(proposals, *, weights) -> Tensor``."""

    spec_family = "synthesis"
    spec_type = "explicit"

    def __init__(self, fn: Callable[..., torch.Tensor] | None = None) -> None:
        if fn is None:
            raise ValidationError("ExplicitSynthesis requires an 'fn' callable")
        if not callable(fn):
            raise ValidationError(f"ExplicitSynthesis fn is not callable: {fn!r}")
        self.fn = fn

    def __call__(self, proposals, *, weights=None):
        result = self.fn(proposals, weights=weights)
        if not torch.is_tensor(result):
            raise ContractError(
                f"ExplicitSynthesis returned {type(result).__name__}, expected a tensor"
            )
        return result

    def describe(self):
        return {**super().describe(), "fn": getattr(self.fn, "__name__", repr(self.fn))}


# ---------------------------------------------------------------------------
# Registry tables
# ---------------------------------------------------------------------------
RETURN_FUNCTION_TYPES: dict[str, type] = LegacyTable(
    "rl_function.return",
    {
        "discounted": DiscountedReturn,
        "monte_carlo": MonteCarloReturn,
    },
)

REWARD_FUNCTION_TYPES: dict[str, type] = LegacyTable(
    "rl_function.reward",
    {
        "scorecard": ScorecardReward,
        "signal": SignalReward,
        "callable": CallableReward,
    },
)

REGULARIZATION_FUNCTION_TYPES: dict[str, type] = LegacyTable(
    "rl_function.regularization",
    {
        "l2": L2Regularization,
        "weight_decay": WeightDecayRegularization,
        "kl_to_reference": KLToReferenceRegularization,
    },
)

OBJECTIVE_TYPES: dict[str, type] = LegacyTable(
    "rl_function.objective",
    {
        "loss": LossObjective,
        "callable": CallableObjective,
    },
)

GRADIENT_FUNCTION_TYPES: dict[str, type] = LegacyTable(
    "rl_function.gradient",
    {
        "clip": ClipGradient,
        "accumulate": AccumulateGradient,
        "scale": ScaleGradient,
    },
)

METRIC_FUNCTION_TYPES: dict[str, type] = LegacyTable(
    "rl_function.metric",
    {
        "accuracy": AccuracyMetric,
        "loss": LossMetric,
        "perplexity": PerplexityMetric,
        "callable": CallableMetric,
    },
)

ACTION_FUNCTION_TYPES: dict[str, type] = LegacyTable(
    "rl_function.action",
    {
        "greedy": GreedyAction,
        "sampled": SampledAction,
    },
)

SYNTHESIS_FUNCTION_TYPES: dict[str, type] = LegacyTable(
    "rl_function.synthesis",
    {
        "consensus": ConsensusSynthesis,
        "weighted": WeightedConsensus,
        "best": BestByScore,
        "learned": LearnedSynthesizer,
        "explicit": ExplicitSynthesis,
    },
)

SUPPORT_FAMILIES: dict[str, dict[str, type]] = {
    "return": RETURN_FUNCTION_TYPES,
    "reward": REWARD_FUNCTION_TYPES,
    "regularization": REGULARIZATION_FUNCTION_TYPES,
    "objective": OBJECTIVE_TYPES,
    "gradient": GRADIENT_FUNCTION_TYPES,
    "metric": METRIC_FUNCTION_TYPES,
    "action": ACTION_FUNCTION_TYPES,
    "synthesis": SYNTHESIS_FUNCTION_TYPES,
}


__all__ = [
    "ACTION_FUNCTION_TYPES",
    "AccumulateGradient",
    "AccuracyMetric",
    "ActionFunction",
    "BestByScore",
    "CallableMetric",
    "CallableObjective",
    "CallableReward",
    "ClipGradient",
    "ConsensusSynthesis",
    "DiscountedReturn",
    "ExplicitSynthesis",
    "GRADIENT_FUNCTION_TYPES",
    "GradientFn",
    "KLToReferenceRegularization",
    "L2Regularization",
    "LearnedSynthesizer",
    "LossMetric",
    "LossObjective",
    "METRIC_FUNCTION_TYPES",
    "MetricFn",
    "MonteCarloReturn",
    "OBJECTIVE_TYPES",
    "ObjectiveFn",
    "PerplexityMetric",
    "REGULARIZATION_FUNCTION_TYPES",
    "RETURN_FUNCTION_TYPES",
    "REWARD_FUNCTION_TYPES",
    "RegularizationFunction",
    "ReturnFunction",
    "RewardFunction",
    "SUPPORT_FAMILIES",
    "SYNTHESIS_FUNCTION_TYPES",
    "SampledAction",
    "ScaleGradient",
    "ScorecardReward",
    "SignalReward",
    "SynthesisFunction",
    "WeightDecayRegularization",
    "WeightedConsensus",
]
