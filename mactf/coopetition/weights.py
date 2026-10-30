"""Weight-update strategies (document 08 section 6.2).

``KappaSoftmaxAverage`` is the default *and* the fallback. It delegates to
:meth:`mactf.coopetition.aggregation.Aggregator.compute_weights`, which already
implements participation gating, confidence gating, softmax over ``kappa * score``,
trimmed-mean outlier removal, floor-then-cap projection, and the disagreement
fallback. Reimplementing any of that here would be a fork, not a wrapper.

Every strategy must be agnostic to the range of ``scores``: the points path feeds
a calibrated signal in ``[-1, 1]`` and the legacy path feeds ``final_score`` in
roughly ``[-50, +60]``. ``softmax`` is shift-invariant, so adding a constant is
safe; dividing by a hardcoded ``100.0`` is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Protocol, Sequence, runtime_checkable

import torch

from ..exceptions import ComplianceError, ValidationError
from .aggregation import AggregationReport, Aggregator

__all__ = [
    "ExplicitWeightStrategy",
    "KappaSoftmaxAverage",
    "NoAggregation",
    "ReferenceOnlyAverage",
    "ScoreWeightedAverage",
    "UniformAverage",
    "WEIGHT_STRATEGIES",
    "WeightUpdateStrategy",
    "build_weight_update",
]


@runtime_checkable
class WeightUpdateStrategy(Protocol):
    """How peer scores become weights."""

    name: ClassVar[str]

    def apply(
        self,
        vectors: Mapping[str, torch.Tensor],
        scores: Mapping[str, float],
        *,
        kappa: float,
        level: int = 0,
        eligible: Mapping[str, bool] | None = None,
        confidence: Mapping[str, float] | None = None,
    ) -> AggregationReport: ...

    def describe(self) -> dict[str, Any]: ...

    @classmethod
    def from_spec(cls, spec: "Mapping[str, Any] | str") -> "WeightUpdateStrategy": ...


def _ordered(names: Sequence[str]) -> list[str]:
    """Lexicographic, so two processes agree on weight order."""
    return sorted(names)


def _zero_report(
    names: Sequence[str],
    *,
    kappa: float,
    adjustments: Sequence[str],
    fallback: bool = False,
) -> AggregationReport:
    """A report with no participation, used by ``NoAggregation`` and all-void groups."""
    ordered = _ordered(names)
    return AggregationReport(
        weights={name: 0.0 for name in ordered},
        raw_weights={name: 0.0 for name in ordered},
        kappa=float(kappa),
        entropy=0.0,
        adjustments=list(adjustments),
        fallback=fallback,
        order=tuple(ordered),
    )


class KappaSoftmaxAverage:
    """The default: delegate to ``Aggregator.compute_weights`` unchanged."""

    name = "kappa_softmax"

    def __init__(self, aggregator: Aggregator | None = None) -> None:
        self.aggregator = aggregator

    def _resolve(self) -> Aggregator:
        if self.aggregator is None:
            raise ComplianceError(
                "KappaSoftmaxAverage needs an Aggregator; the trainer supplies its own "
                "when none is configured"
            )
        return self.aggregator

    def apply(
        self,
        vectors: Mapping[str, torch.Tensor],
        scores: Mapping[str, float],
        *,
        kappa: float,
        level: int = 0,
        eligible: Mapping[str, bool] | None = None,
        confidence: Mapping[str, float] | None = None,
    ) -> AggregationReport:
        names = _ordered(vectors)
        if not names:
            raise ComplianceError("cannot aggregate an empty peer set")
        aggregator = self._resolve()
        report = aggregator.compute_weights(
            names,
            [float(scores.get(name, 0.0)) for name in names],
            eligible=[bool(eligible.get(name, True)) for name in names] if eligible else None,
            confidence=[float(confidence.get(name, 1.0)) for name in names] if confidence else None,
        )
        if level:
            report.adjustments.append(f"level {level}")
        return report

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "class": type(self).__name__,
            "delegates_to": "Aggregator.compute_weights",
            "floor": getattr(self.aggregator, "weight_floor", None),
            "cap": getattr(self.aggregator, "weight_cap", None),
        }

    @classmethod
    def from_spec(cls, spec: Any) -> "WeightUpdateStrategy":
        if isinstance(spec, Aggregator):
            return cls(spec)
        if isinstance(spec, Mapping):
            aggregator = spec.get("aggregator")
            return cls(aggregator if isinstance(aggregator, Aggregator) else None)
        return cls()


class UniformAverage:
    """``1/n`` over eligible peers."""

    name = "uniform"

    def apply(
        self,
        vectors: Mapping[str, torch.Tensor],
        scores: Mapping[str, float],
        *,
        kappa: float,
        level: int = 0,
        eligible: Mapping[str, bool] | None = None,
        confidence: Mapping[str, float] | None = None,
    ) -> AggregationReport:
        del scores, confidence
        names = _ordered(vectors)
        active = [name for name in names if (eligible or {}).get(name, True)]
        adjustments: list[str] = []
        if not active:
            return _zero_report(
                names,
                kappa=kappa,
                adjustments=["no eligible peers; nothing to aggregate"],
                fallback=True,
            )
        if len(active) != len(names):
            gated = [name for name in names if name not in active]
            adjustments.append(f"participation gate zeroed {', '.join(gated)}")
        share = 1.0 / len(active)
        weights = {name: (share if name in active else 0.0) for name in names}
        return AggregationReport(
            weights=weights,
            raw_weights=dict(weights),
            kappa=float(kappa),
            entropy=_entropy([share] * len(active)),
            adjustments=adjustments,
            gated_peers=tuple(name for name in names if name not in active),
            order=tuple(names),
        )

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "class": type(self).__name__}

    @classmethod
    def from_spec(cls, spec: Any) -> "WeightUpdateStrategy":
        return cls()


class ScoreWeightedAverage:
    """``score_i / sum(scores)``, shifted so negative scores stay meaningful.

    The shift is ``-min(scores) + epsilon`` applied to every score, which leaves
    the *ordering* and the *ratios* intact for a non-negative cohort and makes a
    cohort with negatives still produce weights. A shift of a constant is always
    safe under a range change; a division by a hardcoded scale would not be.
    """

    name = "score_weighted"

    def __init__(self, *, epsilon: float = 1e-9) -> None:
        if epsilon <= 0:
            raise ValidationError("ScoreWeightedAverage.epsilon must be > 0")
        self.epsilon = float(epsilon)

    def apply(
        self,
        vectors: Mapping[str, torch.Tensor],
        scores: Mapping[str, float],
        *,
        kappa: float,
        level: int = 0,
        eligible: Mapping[str, bool] | None = None,
        confidence: Mapping[str, float] | None = None,
    ) -> AggregationReport:
        names = _ordered(vectors)
        active = [name for name in names if (eligible or {}).get(name, True)]
        adjustments: list[str] = []
        if not active:
            return _zero_report(
                names,
                kappa=kappa,
                adjustments=["no eligible peers; nothing to aggregate"],
                fallback=True,
            )
        raw_values = {name: float(scores.get(name, 0.0)) for name in active}
        floor = min(raw_values.values())
        shift = -floor if floor < 0 else 0.0
        shifted = {name: value + shift for name, value in raw_values.items()}
        total = sum(shifted.values())
        if total <= 0:
            share = 1.0 / len(active)
            weights = {name: (share if name in active else 0.0) for name in names}
            adjustments.append("degenerate scores; fell back to uniform weights")
        else:
            weights = {
                name: (shifted.get(name, 0.0) / total if name in active else 0.0)
                for name in names
            }
            if shift:
                adjustments.append(f"scores shifted by {-floor:g} to make them non-negative")
        if confidence:
            for name in active:
                weights[name] *= max(0.0, min(1.0, float(confidence.get(name, 1.0))))
            subtotal = sum(weights.values()) or 1.0
            for name in active:
                weights[name] /= subtotal
            adjustments.append("confidence gate scaled participants")
        entropy = _entropy([weights[name] for name in active])
        return AggregationReport(
            weights=weights,
            raw_weights=dict(weights),
            kappa=float(kappa),
            entropy=entropy,
            adjustments=adjustments,
            gated_peers=tuple(name for name in names if name not in active),
            order=tuple(names),
        )

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "class": type(self).__name__, "epsilon": self.epsilon}

    @classmethod
    def from_spec(cls, spec: Any) -> "WeightUpdateStrategy":
        payload = dict(spec) if isinstance(spec, Mapping) else {}
        return cls(epsilon=float(payload.get("epsilon", 1e-9)))


class ReferenceOnlyAverage:
    """Weight 1.0 on the reference, 0 elsewhere: a deliberate reset."""

    name = "reference_only"

    def __init__(self, *, reference: str | None = None) -> None:
        self.reference = reference

    def apply(
        self,
        vectors: Mapping[str, torch.Tensor],
        scores: Mapping[str, float],
        *,
        kappa: float,
        level: int = 0,
        eligible: Mapping[str, bool] | None = None,
        confidence: Mapping[str, float] | None = None,
    ) -> AggregationReport:
        del scores, confidence
        names = _ordered(vectors)
        target = self.reference
        if target is None:
            # Convention: the reference peer is whichever entry is named
            # ``*__reference``, or the first lexicographic node when none is.
            matches = [name for name in names if name.endswith("__reference")]
            target = matches[0] if matches else (names[0] if names else None)
        if target is None or target not in names:
            return _zero_report(
                names,
                kappa=kappa,
                adjustments=["no reference peer available; nothing to aggregate"],
                fallback=True,
            )
        weights = {name: (1.0 if name == target else 0.0) for name in names}
        return AggregationReport(
            weights=weights,
            raw_weights=dict(weights),
            kappa=float(kappa),
            entropy=0.0,
            adjustments=[f"reset to reference peer {target!r}"],
            gated_peers=tuple(name for name in names if name != target),
            fallback=True,
            order=tuple(names),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "class": type(self).__name__,
            "reference": self.reference,
        }

    @classmethod
    def from_spec(cls, spec: Any) -> "WeightUpdateStrategy":
        payload = dict(spec) if isinstance(spec, Mapping) else {}
        reference = payload.get("reference")
        return cls(reference=str(reference) if reference is not None else None)


class NoAggregation:
    """Nothing is written. The report records why rather than being empty."""

    name = "none"

    def apply(
        self,
        vectors: Mapping[str, torch.Tensor],
        scores: Mapping[str, float],
        *,
        kappa: float,
        level: int = 0,
        eligible: Mapping[str, bool] | None = None,
        confidence: Mapping[str, float] | None = None,
    ) -> AggregationReport:
        del scores, eligible, confidence
        return _zero_report(
            list(vectors),
            kappa=kappa,
            adjustments=["aggregation disabled"],
        )

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "class": type(self).__name__, "aggregates": False}

    @classmethod
    def from_spec(cls, spec: Any) -> "WeightUpdateStrategy":
        return cls()


class ExplicitWeightStrategy:
    """Caller-supplied fixed weights, validated to sum to 1 within ``1e-9``."""

    name = "explicit"

    def __init__(self, weights: Mapping[str, float], *, tolerance: float = 1e-9) -> None:
        if not weights:
            raise ValidationError("ExplicitWeightStrategy requires at least one weight")
        cleaned = {str(name): float(value) for name, value in weights.items()}
        if any(value < 0 for value in cleaned.values()):
            raise ValidationError(
                "ExplicitWeightStrategy weights must be non-negative, got "
                + ", ".join(f"{name}={value:g}" for name, value in cleaned.items() if value < 0)
            )
        total = sum(cleaned.values())
        if abs(total - 1.0) > tolerance:
            raise ValidationError(
                f"ExplicitWeightStrategy weights must sum to 1 within {tolerance:g}, "
                f"got {total!r}"
            )
        self.weights = cleaned
        self.tolerance = float(tolerance)

    def apply(
        self,
        vectors: Mapping[str, torch.Tensor],
        scores: Mapping[str, float],
        *,
        kappa: float,
        level: int = 0,
        eligible: Mapping[str, bool] | None = None,
        confidence: Mapping[str, float] | None = None,
    ) -> AggregationReport:
        del scores, confidence
        names = _ordered(vectors)
        adjustments: list[str] = []
        missing = sorted(set(self.weights) - set(names))
        if missing:
            # Silently dropping a configured weight would renormalise the rest and
            # change every number, so it is an explicit error.
            raise ComplianceError(
                "ExplicitWeightStrategy names peers absent from this group: "
                + ", ".join(missing)
            )
        weights = {name: float(self.weights.get(name, 0.0)) for name in names}
        active = [name for name in names if (eligible or {}).get(name, True)]
        if len(active) != len(names):
            gated = [name for name in names if name not in active]
            adjustments.append(f"participation gate zeroed {', '.join(gated)}")
            subtotal = sum(weights[name] for name in active)
            if subtotal <= 0:
                return _zero_report(
                    names,
                    kappa=kappa,
                    adjustments=adjustments + ["no eligible peer carries weight"],
                    fallback=True,
                )
            for name in names:
                weights[name] = weights[name] / subtotal if name in active else 0.0
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ComplianceError(
                f"explicit weights sum to {total!r} after gating; they cannot be applied"
            )
        entropy = _entropy([weights[name] for name in names if weights[name] > 0])
        return AggregationReport(
            weights=weights,
            raw_weights={name: float(self.weights.get(name, 0.0)) for name in names},
            kappa=float(kappa),
            entropy=entropy,
            adjustments=adjustments,
            gated_peers=tuple(name for name in names if name not in active),
            order=tuple(names),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "class": type(self).__name__,
            "weights": dict(self.weights),
        }

    @classmethod
    def from_spec(cls, spec: Any) -> "WeightUpdateStrategy":
        payload = dict(spec) if isinstance(spec, Mapping) else {}
        weights = payload.get("weights")
        if not isinstance(weights, Mapping):
            raise ValidationError(
                "ExplicitWeightStrategy.from_spec requires a 'weights' mapping"
            )
        return cls(weights, tolerance=float(payload.get("tolerance", 1e-9)))


def _entropy(values: Sequence[float]) -> float:
    total = 0.0
    for value in values:
        if value > 0:
            total -= value * torch.log(torch.tensor(value)).item()
    return float(total)


#: Every strategy addressable by name, in the order the plan lists them.
WEIGHT_STRATEGIES: dict[str, type] = {
    "kappa_softmax": KappaSoftmaxAverage,
    "uniform": UniformAverage,
    "score_weighted": ScoreWeightedAverage,
    "reference_only": ReferenceOnlyAverage,
    "none": NoAggregation,
    "explicit": ExplicitWeightStrategy,
}


def build_weight_update(
    spec: "WeightUpdateStrategy | str | Mapping[str, Any] | None",
    *,
    aggregator: Aggregator | None = None,
) -> WeightUpdateStrategy:
    """Resolve a strategy from an instance, a name, a spec, or ``None``.

    ``None`` returns ``KappaSoftmaxAverage`` bound to ``aggregator``, which is the
    documented default. An unknown name raises at construction with the accepted
    list, never at epoch 4.
    """
    if spec is None:
        return KappaSoftmaxAverage(aggregator)
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key not in WEIGHT_STRATEGIES:
            raise ValidationError(
                f"unknown weight_update {spec!r}; accepted: "
                + ", ".join(sorted(WEIGHT_STRATEGIES))
            )
        if key == "kappa_softmax":
            return KappaSoftmaxAverage(aggregator)
        return WEIGHT_STRATEGIES[key]()
    if isinstance(spec, Mapping):
        payload = dict(spec)
        key = str(payload.pop("type", payload.pop("name", "kappa_softmax"))).lower()
        if key not in WEIGHT_STRATEGIES:
            raise ValidationError(
                f"unknown weight_update type {key!r}; accepted: "
                + ", ".join(sorted(WEIGHT_STRATEGIES))
            )
        if key == "kappa_softmax":
            candidate = payload.get("aggregator")
            return KappaSoftmaxAverage(
                candidate if isinstance(candidate, Aggregator) else aggregator
            )
        if key == "explicit":
            return ExplicitWeightStrategy.from_spec(payload)
        target = WEIGHT_STRATEGIES[key]
        return target.from_spec(payload)
    if isinstance(spec, (KappaSoftmaxAverage, UniformAverage, ScoreWeightedAverage,
                         ReferenceOnlyAverage, NoAggregation, ExplicitWeightStrategy)):
        if isinstance(spec, KappaSoftmaxAverage) and spec.aggregator is None and aggregator:
            return KappaSoftmaxAverage(aggregator)
        return spec
    if hasattr(spec, "apply") and hasattr(spec, "describe"):
        return spec  # a structural custom strategy
    raise ValidationError(
        f"weight_update must be a strategy, a name, or a mapping; got {type(spec).__name__}"
    )


@dataclass
class WeightResolution:
    """The resolved strategy plus the fallback that would replace it."""

    strategy: WeightUpdateStrategy
    fallback: WeightUpdateStrategy
    requested: str
    adjustments: list[str] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "resolved": getattr(self.strategy, "name", type(self.strategy).__name__),
            "fallback": getattr(self.fallback, "name", type(self.fallback).__name__),
            "adjustments": list(self.adjustments),
        }
