"""Loss-function wrappers (document 07 section 8.3).

Every term returns ``(tensor, LossBreakdown)`` and declares ``required_fields`` so
``MACTFTrainer`` can reject a batch before the first forward pass instead of
raising ``KeyError`` at step 4000. :class:`CompositeLoss` reproduces
:func:`mactf.rl.loss.compute_loss` exactly, including the ``advantage_mean`` entry
in ``extras``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, ClassVar, Mapping, Protocol, Sequence, runtime_checkable

import torch
import torch.nn.functional as F

from ...components.legacy import LegacyTable
from ...exceptions import ContractError, ValidationError
from ..loss import LossBreakdown, LossConfig, policy_loss
from ..policy import PolicyOutput, reverse_kl
from ..value import value_loss
from ._spec import SpecMixin, build_from_kwargs


@runtime_checkable
class LossFunction(Protocol):
    """A trainable objective term."""

    required_fields: frozenset[str]

    def compute(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, LossBreakdown]: ...

    def describe(self) -> dict[str, Any]: ...


def _require(batch: Mapping[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in batch]
    if missing:
        raise ContractError(
            f"batch is missing required field(s): {', '.join(missing)}"
        )


def _empty_breakdown(total: torch.Tensor, **extras: float) -> LossBreakdown:
    return LossBreakdown(total=float(total.detach().item()), extras=dict(extras))


class _LossBase(SpecMixin):
    spec_family = "loss"
    spec_type: ClassVar[str] = ""
    required_fields: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "_LossBase":
        payload = dict(spec or {})
        kind = payload.pop("type", None)
        if kind is not None and kind != cls.spec_type:
            raise ValidationError(
                f"{cls.__name__}.from_spec cannot build type {kind!r}; it builds {cls.spec_type!r}"
            )
        return build_from_kwargs(cls, payload)

    def compute(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, LossBreakdown]:
        missing = sorted(
            key for key in self.required_fields if key not in batch
        )
        if missing:
            raise ContractError(
                f"{type(self).__name__} is missing required field(s): "
                + ", ".join(missing)
            )
        return self._compute(batch)

    def _compute(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, LossBreakdown]:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.spec_type,
            "class": type(self).__name__,
            "required_fields": sorted(self.required_fields),
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} required={sorted(self.required_fields)}>"


class PolicyGradientLoss(_LossBase):
    """The clipped surrogate from the docx, delegating to :func:`policy_loss`."""

    spec_type = "policy_gradient"
    required_fields = frozenset({"policy", "reference", "actions", "advantages"})

    def __init__(self, *, clip_epsilon: float = 0.2, coefficient: float = 1.0) -> None:
        if not 0.0 <= float(clip_epsilon) < 1.0:
            raise ValidationError(f"clip_epsilon must be in [0, 1), got {clip_epsilon!r}")
        self.clip_epsilon = float(clip_epsilon)
        self.coefficient = float(coefficient)

    def _compute(self, batch):
        value = policy_loss(
            batch["policy"],
            batch["reference"],
            batch["actions"],
            batch["advantages"],
            clip_epsilon=self.clip_epsilon,
        )
        total = self.coefficient * value
        breakdown = LossBreakdown(policy=float(value.detach().item()), total=float(total.detach().item()))
        return total, breakdown

    def describe(self):
        return {
            **super().describe(),
            "clip_epsilon": self.clip_epsilon,
            "coefficient": self.coefficient,
        }


class ValueRegressionLoss(_LossBase):
    """MSE value regression at the configured coefficient (``value.py:61``)."""

    spec_type = "value_regression"
    required_fields = frozenset({"values", "value_targets"})

    def __init__(self, *, coefficient: float = 0.5) -> None:
        self.coefficient = float(coefficient)

    def _compute(self, batch):
        values = batch["values"]
        targets = batch["value_targets"]
        if torch.is_tensor(values) and torch.is_tensor(targets):
            if values.reshape(-1).numel() != targets.reshape(-1).numel():
                raise ContractError(
                    f"ValueRegressionLoss: values has {values.reshape(-1).numel()} elements, "
                    f"value_targets has {targets.reshape(-1).numel()}"
                )
        # The breakdown reports the *unscaled* MSE because that is what
        # ``compute_loss`` records (``loss.py:94``); ``total`` carries the
        # coefficient so a composite sum stays correct.
        raw = value_loss(values, targets, coefficient=1.0)
        loss = self.coefficient * raw
        breakdown = LossBreakdown(
            value=float(raw.detach().item()), total=float(loss.detach().item())
        )
        return loss, breakdown

    def describe(self):
        return {**super().describe(), "coefficient": self.coefficient}


class KLLoss(_LossBase):
    """``KL(pi_theta || pi_ref).mean()`` with the configured coefficient."""

    spec_type = "kl"
    required_fields = frozenset({"policy", "reference"})

    def __init__(self, *, coefficient: float = 0.01) -> None:
        self.coefficient = float(coefficient)

    def _compute(self, batch):
        raw = reverse_kl(batch["policy"], batch["reference"]).mean()
        loss = self.coefficient * raw
        breakdown = LossBreakdown(kl=float(raw.detach().item()), total=float(loss.detach().item()))
        return loss, breakdown

    def describe(self):
        return {**super().describe(), "coefficient": self.coefficient}


class EntropyBonus(_LossBase):
    """``-entropy_coef * mean(H)``: the sign and the mean both match ``loss.py:103``."""

    spec_type = "entropy"
    required_fields = frozenset({"policy"})

    def __init__(self, *, coefficient: float = 0.0) -> None:
        self.coefficient = float(coefficient)

    def _compute(self, batch):
        policy: PolicyOutput = batch["policy"]
        mean_entropy = policy.entropy.mean()
        loss = -self.coefficient * mean_entropy
        breakdown = LossBreakdown(
            entropy=float(mean_entropy.detach().item()), total=float(loss.detach().item())
        )
        return loss, breakdown

    def describe(self):
        return {**super().describe(), "coefficient": self.coefficient}


class ComplianceFloorLoss(_LossBase):
    """``compliance_coef * clamp(-floor, min=0)``: a breach outranks any gain."""

    spec_type = "compliance_floor"
    required_fields = frozenset({"compliance_floor"})

    def __init__(self, *, coefficient: float = 1.0) -> None:
        self.coefficient = float(coefficient)

    def _compute(self, batch):
        reference = _reference_tensor(batch)
        floor_tensor = torch.as_tensor(
            batch["compliance_floor"], device=reference.device, dtype=reference.dtype
        )
        loss = self.coefficient * (-floor_tensor).clamp_min(0.0)
        breakdown = LossBreakdown(
            compliance=float(loss.detach().item()), total=float(loss.detach().item())
        )
        return loss, breakdown

    def describe(self):
        return {**super().describe(), "coefficient": self.coefficient}


class VoidMaskLoss(_LossBase):
    """Zeroes task credit where ``void_mask`` is asserted: observability only.

    ``loss.py:116-119`` deliberately does not change the total, because the
    advantage already reflects the void. This term therefore contributes a zero
    tensor and reports how many samples were masked, so a reader can see the mask
    was honoured rather than wondering whether it was applied.
    """

    spec_type = "void_mask"
    required_fields = frozenset({"void_mask"})

    def _compute(self, batch):
        mask = torch.as_tensor(batch["void_mask"]).bool()
        reference = torch.zeros(
            (), device=mask.device, dtype=torch.float32
        )
        breakdown = LossBreakdown(
            total=0.0, extras={"void_masked": float(mask.sum().item())}
        )
        return reference, breakdown


class CompositeLoss(_LossBase):
    """Weighted sum reproducing :func:`compute_loss` exactly.

    The default coefficients are ``LossConfig``'s, and the term list is the docx
    one, so ``CompositeLoss().compute(batch)`` equals ``compute_loss(...)`` to
    ``1e-12`` for the same inputs. Adding or removing a term is how a caller
    changes the objective; the arithmetic is not re-derived here.
    """

    spec_type = "composite"
    required_fields = frozenset(
        {"policy", "reference", "actions", "advantages", "values", "value_targets"}
    )

    def __init__(
        self,
        *,
        config: LossConfig | None = None,
        policy_coefficient: float = 1.0,
    ) -> None:
        self.config = config or LossConfig()
        self.policy_coefficient = float(policy_coefficient)

    # -- terms -----------------------------------------------------------
    def _terms(self):
        return (
            PolicyGradientLoss(
                clip_epsilon=self.config.clip_epsilon,
                coefficient=self.policy_coefficient,
            ),
            ValueRegressionLoss(coefficient=self.config.value_coef),
            KLLoss(coefficient=self.config.kl_coef),
            EntropyBonus(coefficient=self.config.entropy_coef),
        )

    def _compute(self, batch):
        breakdown = LossBreakdown()
        total = None
        weighted: dict[str, float] = {}

        for term in self._terms():
            component, part = term.compute(batch)
            total = component if total is None else total + component
            breakdown.policy += part.policy
            breakdown.value += part.value
            breakdown.kl += part.kl
            breakdown.entropy += part.entropy
            weighted[term.spec_type] = float(part.total)

        if "compliance_floor" in batch:
            component, part = ComplianceFloorLoss(
                coefficient=self.config.compliance_coef
            ).compute(batch)
            total = component if total is None else total + component
            breakdown.compliance += part.compliance
            weighted["compliance_floor"] = float(part.total)

        if "void_mask" in batch:
            _, part = VoidMaskLoss().compute(batch)
            breakdown.extras.update(part.extras)

        if "advantages" in batch:
            advantages = torch.as_tensor(batch["advantages"]).float()
            breakdown.extras["advantage_mean"] = float(advantages.mean().detach().item())

        assert total is not None  # always at least one term
        breakdown.total = float(total.detach().item())
        breakdown.extras.update({f"term:{name}": value for name, value in weighted.items()})
        return total, breakdown

    def describe(self):
        return {
            **super().describe(),
            "config": {
                "value_coef": self.config.value_coef,
                "kl_coef": self.config.kl_coef,
                "entropy_coef": self.config.entropy_coef,
                "clip_epsilon": self.config.clip_epsilon,
                "compliance_coef": self.config.compliance_coef,
            },
            "terms": [term.describe() for term in self._terms()],
        }


class CausalLMLoss(_LossBase):
    """Language-model loss for fine-tuning (document 08).

    Uses ``model(**batch).loss`` when the model computes one, and otherwise
    ``cross_entropy(logits, labels, ignore_index=-100)``. Both paths are real; the
    fallback exists because a bare ``nn.Module`` returning logits is common and
    requiring a ``ModelOutput`` would rule it out.
    """

    spec_type = "causal_lm"
    required_fields = frozenset({"input_ids", "labels"})

    def __init__(self, *, ignore_index: int = -100, model: Any = None) -> None:
        self.ignore_index = int(ignore_index)
        self.model = model

    def compute(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, LossBreakdown]:
        """Guard the fields the selected path actually reads.

        ``input_ids`` is only required when the model is asked to run: the
        logits path takes precomputed logits and must stay usable for a bare
        tensor, while the class-level ``required_fields`` documents the
        fine-tuning contract for ``MACTFTrainer``.
        """
        if "labels" not in batch:
            raise ContractError(
                "CausalLMLoss is missing required field(s): labels"
            )
        model = self.model or batch.get("model")
        if model is not None and "input_ids" not in batch:
            raise ContractError(
                "CausalLMLoss is missing required field(s): input_ids"
            )
        return self._compute(batch)

    def _compute(self, batch):
        model = self.model or batch.get("model")
        if model is not None:
            kwargs = {
                key: value
                for key, value in batch.items()
                if key not in {"model", "logits", "labels"}
            }
            output = model(**kwargs)
            native = getattr(output, "loss", None)
            if native is not None:
                breakdown = LossBreakdown(total=float(native.detach().item()))
                return native, breakdown
            logits = getattr(output, "logits", None)
            if logits is None and torch.is_tensor(output):
                logits = output
            if logits is None:
                raise ContractError(
                    "CausalLMLoss: model output has neither .loss nor .logits"
                )
        else:
            logits = batch.get("logits")
            if logits is None:
                raise ContractError(
                    "CausalLMLoss: provide either 'model' or 'logits' in the batch"
                )
        labels = batch["labels"]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=self.ignore_index,
        )
        breakdown = LossBreakdown(total=float(loss.detach().item()))
        return loss, breakdown

    def describe(self):
        return {**super().describe(), "ignore_index": self.ignore_index}


class CallableLoss(_LossBase):
    """Adapts ``fn(batch) -> Tensor`` or ``fn(batch) -> (Tensor, dict)``."""

    spec_type = "callable"

    def __init__(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        batch_keys: Sequence[str] | None = None,
    ) -> None:
        if fn is None:
            raise ValidationError("CallableLoss requires an 'fn' callable")
        if not callable(fn):
            raise ValidationError(f"CallableLoss fn is not callable: {fn!r}")
        self.fn = fn
        self.required_fields = frozenset(batch_keys or ())

    def _compute(self, batch):
        result = self.fn(batch)
        if torch.is_tensor(result):
            return result, LossBreakdown(total=float(result.detach().item()))
        if isinstance(result, tuple) and len(result) == 2:
            tensor, extras = result
            if not torch.is_tensor(tensor):
                raise ContractError(
                    f"CallableLoss returned a {type(tensor).__name__} where a tensor was expected"
                )
            breakdown = LossBreakdown(
                total=float(tensor.detach().item()),
                extras={key: float(value) for key, value in dict(extras or {}).items()},
            )
            return tensor, breakdown
        if isinstance(result, LossBreakdown):
            raise ContractError("CallableLoss must return a tensor, not a LossBreakdown")
        raise ContractError(
            f"CallableLoss returned {type(result).__name__}; expected a tensor or "
            "(tensor, dict)"
        )

    def describe(self):
        return {
            **super().describe(),
            "fn": getattr(self.fn, "__name__", repr(self.fn)),
            "batch_keys": sorted(self.required_fields),
        }


class ChainedLoss(_LossBase):
    """Sequential application, for multi-objective runs where each term scales itself."""

    spec_type = "chained"

    def __init__(self, terms: Sequence[LossFunction] | None = None) -> None:
        self.terms: list[Any] = list(terms or [])
        for term in self.terms:
            if not callable(getattr(term, "compute", None)):
                raise ValidationError(
                    f"ChainedLoss term {type(term).__name__} has no compute()"
                )
        self.required_fields = frozenset(
            key for term in self.terms for key in term.required_fields
        )

    def add(self, term: LossFunction) -> "ChainedLoss":
        self.terms.append(term)
        self.required_fields = self.required_fields | frozenset(term.required_fields)
        return self

    def _compute(self, batch):
        if not self.terms:
            raise ContractError("ChainedLoss has no terms to apply")
        breakdown = LossBreakdown()
        total = None
        for term in self.terms:
            component, part = term.compute(batch)
            total = component if total is None else total + component
            breakdown.policy += part.policy
            breakdown.value += part.value
            breakdown.kl += part.kl
            breakdown.entropy += part.entropy
            breakdown.compliance += part.compliance
            breakdown.extras.update(
                {f"{term.spec_type}.{key}": value for key, value in part.extras.items()}
            )
        assert total is not None
        breakdown.total = float(total.detach().item())
        return total, breakdown

    def describe(self):
        return {
            **super().describe(),
            "terms": [term.describe() for term in self.terms],
        }

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "ChainedLoss":
        from .registry import build_function

        payload = dict(spec or {})
        kind = payload.pop("type", None)
        if kind is not None and kind != cls.spec_type:
            raise ValidationError(
                f"ChainedLoss.from_spec cannot build type {kind!r}; it builds 'chained'"
            )
        raw_terms = payload.pop("terms", None)
        if raw_terms is None:
            raise ValidationError("ChainedLoss spec requires a 'terms' list")
        if not isinstance(raw_terms, (list, tuple)):
            raise ValidationError("ChainedLoss 'terms' must be a list of specs")
        terms = [build_function(term, family="loss") for term in raw_terms]
        return build_from_kwargs(cls, {**payload, "terms": terms})


class WeightedSumLoss(_LossBase):
    """``sum_i weight_i * term_i``, each term a registered loss.

    ``ChainedLoss`` adds terms with weight 1; a term that needs a coefficient has to scale
    itself. This takes the coefficients out of the terms so the same ``kl`` term can be
    used at 0.01 in one run and 0.1 in another by changing a spec, not a class.
    """

    spec_type = "weighted_sum"

    def __init__(
        self,
        terms: Sequence[Any] | None = None,
        *,
        weights: Sequence[float] | None = None,
    ) -> None:
        raw = list(terms or [])
        if not raw:
            raise ValidationError("WeightedSumLoss needs at least one term")
        resolved = list(weights) if weights is not None else [1.0] * len(raw)
        if len(resolved) != len(raw):
            raise ValidationError(
                f"WeightedSumLoss has {len(raw)} terms but {len(resolved)} weights"
            )
        for index, weight in enumerate(resolved):
            if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                raise ValidationError(f"WeightedSumLoss weight {index} is not a number: {weight!r}")
            if not math.isfinite(float(weight)):
                raise ValidationError(f"WeightedSumLoss weight {index} is not finite: {weight!r}")
        built = []
        for term in raw:
            if isinstance(term, Mapping):
                from .registry import build_function

                term = build_function(term, family="loss")
            if not callable(getattr(term, "compute", None)):
                raise ValidationError(f"WeightedSumLoss term {type(term).__name__} has no compute()")
            built.append(term)
        self.terms: list[Any] = built
        self.weights: list[float] = [float(weight) for weight in resolved]
        self.required_fields = frozenset(
            key for term in built for key in getattr(term, "required_fields", ())
        )

    def _compute(self, batch):
        breakdown = LossBreakdown()
        total = None
        for term, weight in zip(self.terms, self.weights):
            component, part = term.compute(batch)
            scaled = component * weight
            total = scaled if total is None else total + scaled
            breakdown.policy += weight * part.policy
            breakdown.value += weight * part.value
            breakdown.kl += weight * part.kl
            breakdown.entropy += weight * part.entropy
            breakdown.compliance += weight * part.compliance
            label = getattr(term, "spec_type", type(term).__name__)
            breakdown.extras[f"term:{label}"] = float(weight * part.total)
            for key, value in part.extras.items():
                breakdown.extras[f"{label}.{key}"] = value
                if key == "advantage_mean":
                    # The one number a reader (the StepGuard's void probe) looks for unprefixed.
                    breakdown.extras.setdefault("advantage_mean", value)
        assert total is not None  # the constructor rejects an empty term list
        breakdown.total = float(total.detach().item())
        return total, breakdown

    def describe(self):
        return {
            **super().describe(),
            "weights": list(self.weights),
            "terms": [term.describe() for term in self.terms],
        }

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "WeightedSumLoss":
        payload = dict(spec or {})
        kind = payload.pop("type", None)
        if kind is not None and kind != cls.spec_type:
            raise ValidationError(
                f"WeightedSumLoss.from_spec cannot build type {kind!r}; it builds 'weighted_sum'"
            )
        return build_from_kwargs(cls, payload)


def _reference_tensor(batch: Mapping[str, Any]) -> torch.Tensor:
    """Find a tensor to take device/dtype from, for scalar terms."""
    for key in ("policy", "values", "advantages", "rewards", "void_mask"):
        candidate = batch.get(key)
        if torch.is_tensor(candidate):
            return candidate
        if isinstance(candidate, PolicyOutput):
            return candidate.logits
    floor = batch.get("compliance_floor")
    if torch.is_tensor(floor):
        return floor
    return torch.zeros(())


LOSS_FUNCTION_TYPES: dict[str, type] = LegacyTable(
    "rl_function.loss",
    {
        "policy_gradient": PolicyGradientLoss,
        "value_regression": ValueRegressionLoss,
        "kl": KLLoss,
        "entropy": EntropyBonus,
        "compliance_floor": ComplianceFloorLoss,
        "void_mask": VoidMaskLoss,
        "composite": CompositeLoss,
        "causal_lm": CausalLMLoss,
        "callable": CallableLoss,
        "chained": ChainedLoss,
        "weighted_sum": WeightedSumLoss,
    },
)

LOSS_CLASSES = tuple(LOSS_FUNCTION_TYPES.values())


__all__ = [
    "CallableLoss",
    "CausalLMLoss",
    "ChainedLoss",
    "CompositeLoss",
    "ComplianceFloorLoss",
    "EntropyBonus",
    "KLLoss",
    "LOSS_CLASSES",
    "LOSS_FUNCTION_TYPES",
    "LossFunction",
    "PolicyGradientLoss",
    "ValueRegressionLoss",
    "VoidMaskLoss",
    "WeightedSumLoss",
]
