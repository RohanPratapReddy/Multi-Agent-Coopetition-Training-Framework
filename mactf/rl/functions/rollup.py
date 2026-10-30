"""Rollup strategies for hierarchical peers (document 10 section 5.2).

A rollup turns a node's children ``ScorerOutput`` objects into the contribution
that node reports to its own parent. Every strategy except
:class:`ExplicitRollup` unions the children's ``void_rules`` into the parent
output regardless of the numeric strategy: averaging a voided child with healthy
siblings produces a number, and that number must not launder the void.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Mapping, Protocol, Sequence, runtime_checkable

from ...exceptions import ContractError, ValidationError
from ...scoring.base import ScorerOutput, ScoringKind
from ._spec import SpecMixin, build_from_kwargs


@runtime_checkable
class RollupStrategy(Protocol):
    """A rule for combining child outputs into a parent contribution."""

    name: ClassVar[str]

    def rollup(
        self,
        parent: str,
        children: Sequence[ScorerOutput],
        *,
        weights: Mapping[str, float] | None = None,
        scores: Mapping[str, float] | None = None,
    ) -> ScorerOutput: ...


def _child_weights(
    children: Sequence[ScorerOutput],
    weights: Mapping[str, float] | None,
    scores: Mapping[str, float] | None,
) -> list[float]:
    """Resolve one weight per child.

    Priority: an explicit ``weights`` mapping (keyed by the rule ids the child
    reported, or by index if it is a plain sequence), then ``scores``, then 1.0.
    An explicit weight of zero is honoured; a missing one falls back rather than
    being treated as zero, because "not supplied" is not "silenced".
    """
    if not children:
        return []
    if weights:
        resolved: list[float] = []
        for index, child in enumerate(children):
            key = child.rules[0] if child.rules else str(index)
            value = weights.get(key, weights.get(str(index), 1.0))
            resolved.append(float(value))
        return resolved
    if scores:
        resolved = []
        for index, child in enumerate(children):
            key = child.rules[0] if child.rules else str(index)
            resolved.append(float(scores.get(key, 1.0)))
        return resolved
    return [1.0] * len(children)


def _void_union(children: Sequence[ScorerOutput]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for child in children:
        for rule_id in child.void_rules:
            seen.setdefault(rule_id, None)
    return tuple(sorted(seen))


class _RollupBase(SpecMixin):
    """Common plumbing: names, validation, and the void union."""

    spec_family = "rollup"
    spec_type: ClassVar[str] = ""
    name: ClassVar[str] = ""

    def __init__(self, *, kind: ScoringKind = ScoringKind.HIERARCHY, strict: bool = True) -> None:
        self.kind = kind
        self.strict = strict

    # -- construction ----------------------------------------------------
    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "_RollupBase":
        payload = dict(spec or {})
        kind = payload.pop("type", None)
        if kind is not None and kind != cls.spec_type:
            raise ValidationError(
                f"{cls.__name__}.from_spec cannot build type {kind!r}; "
                f"it builds {cls.spec_type!r}"
            )
        return build_from_kwargs(cls, payload)

    # -- internals -------------------------------------------------------
    def _prepare(
        self,
        parent: str,
        children: Sequence[ScorerOutput],
        weights: Mapping[str, float] | None,
    ) -> tuple[list[float], dict[str, Any]]:
        if self.strict:
            for child in children:
                if not isinstance(child, ScorerOutput):
                    raise ContractError(
                        f"{self.name}: child outputs must be ScorerOutput, "
                        f"got {type(child).__name__}"
                    )
        resolved = _child_weights(children, weights, None)
        void = _void_union(children)
        return resolved, {"parent": parent, "void_rules": void}

    def _output(
        self,
        parent: str,
        amount: float,
        children: Sequence[ScorerOutput],
        *,
        weights: Sequence[float],
        extras: Mapping[str, Any],
    ) -> ScorerOutput:
        breakdown: dict[str, Any] = {
            "strategy": self.name,
            "parent": parent,
            "children": len(children),
            "child_amounts": [float(child.amount) for child in children],
            "weights": list(float(weight) for weight in weights),
            "total": float(amount),
        }
        breakdown.update(dict(extras))
        return ScorerOutput(
            kind=self.kind,
            amount=float(amount),
            breakdown=breakdown,
            rules=tuple(
                dict.fromkeys(rule_id for child in children for rule_id in child.rules)
            ),
            metadata={"rollup": self.name, "parent": parent},
            void_rules=_void_union(children),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.spec_type,
            "class": type(self).__name__,
            "name": self.name,
            "kind": self.kind.value,
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name}>"

    # -- the protocol method --------------------------------------------
    def rollup(
        self,
        parent: str,
        children: Sequence[ScorerOutput],
        *,
        weights: Mapping[str, float] | None = None,
        scores: Mapping[str, float] | None = None,
    ) -> ScorerOutput:  # pragma: no cover - abstract
        raise NotImplementedError


class MeanRollup(_RollupBase):
    """Arithmetic mean of the child amounts."""

    spec_type = "mean"
    name = "mean"

    def rollup(self, parent, children, *, weights=None, scores=None):
        if not children:
            return self._output(parent, 0.0, children, weights=[], extras={})
        total = sum(float(child.amount) for child in children)
        return self._output(
            parent,
            total / len(children),
            children,
            weights=[1.0] * len(children),
            extras={"raw_total": total},
        )


class SumRollup(_RollupBase):
    """Additive rollup, for pools whose children contribute disjoint credit."""

    spec_type = "sum"
    name = "sum"

    def rollup(self, parent, children, *, weights=None, scores=None):
        total = sum(float(child.amount) for child in children)
        return self._output(
            parent, total, children, weights=[1.0] * len(children), extras={}
        )


class MinimumRollup(_RollupBase):
    """The worst child value, for a conservative parent."""

    spec_type = "minimum"
    name = "minimum"

    def rollup(self, parent, children, *, weights=None, scores=None):
        if not children:
            return self._output(parent, 0.0, children, weights=[], extras={})
        worst = min(float(child.amount) for child in children)
        index = min(
            range(len(children)), key=lambda position: float(children[position].amount)
        )
        return self._output(
            parent,
            worst,
            children,
            weights=[1.0] * len(children),
            extras={"worst_index": index},
        )


class WorstCaseRollup(MinimumRollup):
    """The minimum, with every child's ``void_rules`` unioned in.

    Numerically identical to :class:`MinimumRollup`; named separately because the
    intent differs and a reader of a scorecard should be able to tell which one
    produced it. The union is already automatic, so the class exists to record
    intent rather than to change arithmetic.
    """

    spec_type = "worst_case"
    name = "worst_case"


class KappaWeightedRollup(_RollupBase):
    """Weighted sum using kappa weights; the default (document 10 section 5.2).

    Missing weights fall back to a uniform 1/n, which is the arithmetic mean. The
    weights are normalised so a caller passing raw confidences rather than
    softmax output cannot inflate the parent.
    """

    spec_type = "kappa"
    name = "kappa"

    def __init__(
        self,
        *,
        kind: ScoringKind = ScoringKind.HIERARCHY,
        strict: bool = True,
        normalise: bool = True,
        fallback: str = "mean",
    ) -> None:
        super().__init__(kind=kind, strict=strict)
        if fallback not in {"mean", "zero", "error"}:
            raise ValidationError(
                f"unknown fallback {fallback!r}; use 'mean', 'zero', or 'error'"
            )
        self.normalise = bool(normalise)
        self.fallback = fallback

    def rollup(self, parent, children, *, weights=None, scores=None):
        if not children:
            return self._output(parent, 0.0, children, weights=[], extras={})
        resolved = _child_weights(
            children, weights, scores if self.fallback == "mean" else scores
        )
        total_weight = sum(resolved)
        used_fallback = False
        if total_weight == 0.0:
            if self.fallback == "error":
                raise ContractError(
                    f"KappaWeightedRollup({parent}): child weights sum to zero"
                )
            used_fallback = True
            resolved = [1.0 / len(children)] * len(children)
            total_weight = 1.0
        elif self.normalise:
            resolved = [weight / total_weight for weight in resolved]
            total_weight = 1.0
        amount = sum(weight * float(child.amount) for weight, child in zip(resolved, children))
        return self._output(
            parent,
            amount,
            children,
            weights=resolved,
            extras={"fallback_used": used_fallback, "normalised": self.normalise},
        )

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update({"normalise": self.normalise, "fallback": self.fallback})
        return payload


class ExplicitRollup(_RollupBase):
    """``fn(parent, children, *, weights, scores) -> ScorerOutput``.

    The one strategy that does *not* union the children's voids, because the
    supplied function owns the whole output. That is the documented exception,
    and it is why the class is named differently: choosing it is an explicit
    statement that the caller wants full control.
    """

    spec_type = "explicit"
    name = "explicit"

    def __init__(
        self,
        fn: Callable[..., ScorerOutput] | None = None,
        *,
        kind: ScoringKind = ScoringKind.HIERARCHY,
        strict: bool = True,
    ) -> None:
        super().__init__(kind=kind, strict=strict)
        if fn is None:
            raise ValidationError("ExplicitRollup requires an 'fn' callable")
        if not callable(fn):
            raise ValidationError(f"ExplicitRollup fn is not callable: {fn!r}")
        self.fn = fn

    def rollup(self, parent, children, *, weights=None, scores=None):
        output = self.fn(parent, list(children), weights=weights, scores=scores)
        if not isinstance(output, ScorerOutput):
            raise ContractError(
                f"ExplicitRollup fn returned {type(output).__name__}, not ScorerOutput"
            )
        return output

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload["fn"] = getattr(self.fn, "__name__", repr(self.fn))
        return payload


__all__ = [
    "ExplicitRollup",
    "KappaWeightedRollup",
    "MeanRollup",
    "MinimumRollup",
    "RollupStrategy",
    "SumRollup",
    "WorstCaseRollup",
]
