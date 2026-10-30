"""Synthesis functions: how a parent turns child proposals into its own.

The two-layer split is deliberate. :mod:`mactf.rl.functions.support` owns the
node-free arithmetic (``ConsensusSynthesis``, ``WeightedConsensus``, ...) because
it is also used outside a hierarchy, while this module owns the node-aware
contract from document 10 section 4.2: which children proposed, what they scored,
and how a parent's proposal is assembled from them.

Shape rules are enforced before returning (document 10 section 4.2):

* every proposal shares ``shape[0]`` (batch) and the same trailing shape, unless
  the function declares otherwise (``LearnedSynthesizer`` and
  ``ConcatSynthesis`` validate their own contract);
* an empty proposal map raises rather than returning zeros, because zeros let a
  dead subtree look like a confident neutral vote.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Mapping, Sequence

import torch

from ..exceptions import ContractError, ValidationError
from .tree import PeerNode

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..orchestration.episode import Task

__all__ = [
    "BestByScoreSynthesis",
    "ConcatSynthesis",
    "ConsensusSynthesis",
    "ExplicitSynthesis",
    "LearnedSynthesizer",
    "SYNTHESIS_TYPES",
    "SynthesisFunction",
    "WeightedConsensus",
    "build_synthesis",
    "check_aligned",
    "synthesis_capabilities",
]


def check_aligned(proposals: Mapping[str, torch.Tensor], *, node: str = "") -> None:
    """Validate batch and trailing shape agreement, and that every value is a tensor."""
    if not proposals:
        raise ContractError(
            f"no proposals to synthesize for {node or '<unknown>'}"
        )
    trailing: tuple[int, ...] | None = None
    batch: int | None = None
    for path, tensor in proposals.items():
        if not torch.is_tensor(tensor):
            raise ContractError(
                f"synthesis at {node or '<unknown>'}: proposal from {path!r} is a "
                f"{type(tensor).__name__}, not a tensor"
            )
        if batch is None:
            batch = int(tensor.shape[0]) if tensor.dim() else -1
        elif int(tensor.shape[0]) != batch:
            raise ContractError(
                f"synthesis at {node or '<unknown>'}: proposal from {path!r} has batch "
                f"{int(tensor.shape[0])}, expected {batch}"
            )
        candidate = tuple(tensor.shape[1:])
        if trailing is None:
            trailing = candidate
        elif candidate != trailing:
            raise ContractError(
                f"synthesis at {node or '<unknown>'}: proposal from {path!r} has trailing "
                f"shape {candidate}, expected {trailing}"
            )


def _ordered(proposals: Mapping[str, torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
    names = list(proposals)
    return names, [proposals[name] for name in names]


class SynthesisFunction(ABC):
    """Base class for every node-aware synthesis function."""

    name: ClassVar[str] = ""

    @abstractmethod
    def synthesize(
        self,
        node: PeerNode,
        proposals: Mapping[str, torch.Tensor],
        *,
        scores: Mapping[str, float] | None = None,
        weights: Mapping[str, float] | None = None,
        task: "Task | None" = None,
    ) -> torch.Tensor:
        """Return the parent's proposal built from its children's."""

    def describe(self) -> dict[str, Any]:
        return {"type": self.name, "class": type(self).__name__}

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "SynthesisFunction":
        return build_synthesis(spec)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name}>"


class ConsensusSynthesis(SynthesisFunction):
    """Uniform mean: a promotion of :func:`mactf.orchestration.episode.consensus`.

    Bit-identical to the existing helper, including its ``float()`` cast on each
    proposal, which is what makes the flat-equivalence gate meaningful (document
    10 section 4.2).
    """

    name = "consensus"

    def synthesize(self, node, proposals, *, scores=None, weights=None, task=None):
        from ..orchestration.episode import consensus

        if not proposals:
            raise ContractError(f"no proposals to synthesize for {node.path}")
        check_aligned(proposals, node=node.path)
        # ``consensus`` returns ``None`` only for an empty map, which is already
        # rejected above; the assertion is for readers, not for control flow.
        result = consensus(dict(proposals))
        assert result is not None
        return result


class WeightedConsensus(SynthesisFunction):
    """Weighted mean using kappa weights; missing weights fall back to uniform."""

    name = "weighted"

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        self.weights = dict(weights or {})

    def synthesize(self, node, proposals, *, scores=None, weights=None, task=None):
        from ..rl.functions.support import WeightedConsensus as _Weighted

        if not proposals:
            raise ContractError(f"no proposals to synthesize for {node.path}")
        check_aligned(proposals, node=node.path)
        resolved = dict(self.weights)
        resolved.update(dict(weights or {}))
        resolved.update(dict(scores or {}))
        return _Weighted(resolved)(dict(proposals))

    def describe(self):
        return {**super().describe(), "weights": dict(self.weights)}


class BestByScore(SynthesisFunction):
    """The proposal from the highest-scoring child; ties broken by path."""

    name = "best"

    def __init__(self, highest: bool = True) -> None:
        self.highest = bool(highest)

    def synthesize(self, node, proposals, *, scores=None, weights=None, task=None):
        if not proposals:
            raise ContractError(f"no proposals to synthesize for {node.path}")
        check_aligned(proposals, node=node.path)
        table = dict(scores or {})
        table.update(dict(weights or {}))
        if self.highest:
            chosen = min(
                proposals,
                key=lambda path: (-float(table.get(path, float("-inf"))), path),
            )
        else:
            chosen = min(
                proposals,
                key=lambda path: (float(table.get(path, float("inf"))), path),
            )
        return proposals[chosen]

    def describe(self):
        return {**super().describe(), "highest": self.highest}


class LearnedSynthesizer(SynthesisFunction):
    """The parent's own trainable module consumes the stacked child proposals.

    Two forms, both real: a ``torch.nn.Module`` that maps
    ``(n_children, ...)`` to the parent's proposal, or a learned simplex over
    children when no module is supplied. The module form is what makes a parent a
    trainable node rather than a switchboard.
    """

    name = "learned"

    def __init__(
        self,
        module: Any = None,
        *,
        n_proposals: int | None = None,
        temperature: float = 1.0,
    ) -> None:
        self.module = module
        if module is None:
            if n_proposals is None:
                raise ValidationError(
                    "LearnedSynthesizer requires either a 'module' or 'n_proposals'"
                )
            from ..rl.functions.support import LearnedSynthesizer as _Linear

            self._linear: Any = _Linear(int(n_proposals), temperature=temperature)
        else:
            if not callable(getattr(module, "forward", None)):
                raise ValidationError(
                    f"LearnedSynthesizer module has no forward(): {module!r}"
                )
            self._linear = None
        self.n_proposals = n_proposals
        self.temperature = float(temperature)

    def parameters(self):
        if self.module is not None:
            return tuple(self.module.parameters())
        return tuple(self._linear.parameters())

    def train(self, mode: bool = True):
        target = self.module if self.module is not None else self._linear
        train = getattr(target, "train", None)
        if callable(train):
            train(mode)
        return self

    def synthesize(self, node, proposals, *, scores=None, weights=None, task=None):
        if not proposals:
            raise ContractError(f"no proposals to synthesize for {node.path}")
        check_aligned(proposals, node=node.path)
        names, tensors = _ordered(proposals)
        stacked = torch.stack([tensor.float() for tensor in tensors], dim=0)
        if self.module is not None:
            output = self.module(stacked)
            if isinstance(output, tuple):
                output = output[0]
            if not torch.is_tensor(output):
                raise ContractError(
                    f"LearnedSynthesizer module returned {type(output).__name__}, "
                    "expected a tensor"
                )
            if output.dim() and int(output.shape[0]) != int(stacked.shape[1]):
                raise ContractError(
                    f"LearnedSynthesizer at {node.path} returned batch "
                    f"{int(output.shape[0])}, expected {int(stacked.shape[1])}"
                )
            return output
        return self._linear(stacked)

    def describe(self):
        return {
            **super().describe(),
            "module": type(self.module).__name__ if self.module is not None else None,
            "n_proposals": self.n_proposals,
            "temperature": self.temperature,
        }


class ExplicitSynthesis(SynthesisFunction):
    """``fn(node, proposals, *, scores, weights, task) -> Tensor``."""

    name = "explicit"

    def __init__(self, fn: Callable[..., torch.Tensor] | None = None) -> None:
        if fn is None:
            raise ValidationError("ExplicitSynthesis requires an 'fn' callable")
        if not callable(fn):
            raise ValidationError(f"ExplicitSynthesis fn is not callable: {fn!r}")
        self.fn = fn

    def synthesize(self, node, proposals, *, scores=None, weights=None, task=None):
        result = self.fn(
            node, dict(proposals), scores=scores, weights=weights, task=task
        )
        if not torch.is_tensor(result):
            raise ContractError(
                f"ExplicitSynthesis returned {type(result).__name__}, expected a tensor"
            )
        return result

    def describe(self):
        return {**super().describe(), "fn": getattr(self.fn, "__name__", repr(self.fn))}


class ConcatSynthesis(SynthesisFunction):
    """Concatenate along the feature axis, for models expecting a joint representation.

    Declares its own contract (document 10 section 4.2), so it does not require
    trailing shapes to agree; it does require batch size to agree, because
    concatenation along the feature axis is meaningless otherwise.
    """

    name = "concat"

    def __init__(self, *, dim: int = -1) -> None:
        self.dim = int(dim)

    def synthesize(self, node, proposals, *, scores=None, weights=None, task=None):
        if not proposals:
            raise ContractError(f"no proposals to synthesize for {node.path}")
        for path, tensor in proposals.items():
            if not torch.is_tensor(tensor):
                raise ContractError(
                    f"synthesis at {node.path}: proposal from {path!r} is a "
                    f"{type(tensor).__name__}, not a tensor"
                )
        names, tensors = _ordered(proposals)
        batch = int(tensors[0].shape[0])
        for path, tensor in zip(names, tensors):
            if int(tensor.shape[0]) != batch:
                raise ContractError(
                    f"synthesis at {node.path}: proposal from {path!r} has batch "
                    f"{int(tensor.shape[0])}, expected {batch}"
                )
        return torch.cat([tensor.float() for tensor in tensors], dim=self.dim)

    def describe(self):
        return {**super().describe(), "dim": self.dim}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
SYNTHESIS_TYPES: dict[str, type[SynthesisFunction]] = {
    "consensus": ConsensusSynthesis,
    "weighted": WeightedConsensus,
    "best": BestByScore,
    "learned": LearnedSynthesizer,
    "explicit": ExplicitSynthesis,
    "concat": ConcatSynthesis,
}

_ALIASES = {
    "weighted_consensus": "weighted",
    "best_by_score": "best",
    "mean": "consensus",
}


def build_synthesis(spec: Mapping[str, Any] | str | None) -> SynthesisFunction:
    """Resolve a function from a name or a ``{"type": ..., **kwargs}`` spec."""
    if spec is None:
        return ConsensusSynthesis()
    if isinstance(spec, str):
        payload: dict[str, Any] = {"type": spec}
    elif isinstance(spec, SynthesisFunction):
        return spec
    else:
        payload = dict(spec)
    type_name = payload.pop("type", None)
    if type_name is None:
        raise ValidationError(
            "a synthesis spec requires a 'type' key; accepted types: "
            + ", ".join(sorted(SYNTHESIS_TYPES))
        )
    type_name = _ALIASES.get(str(type_name), str(type_name))
    if type_name not in SYNTHESIS_TYPES:
        raise ValidationError(
            f"unknown synthesis type {type_name!r}; accepted types: "
            + ", ".join(sorted(SYNTHESIS_TYPES))
        )
    cls = SYNTHESIS_TYPES[type_name]
    accepted = _constructor_keys(cls)
    unknown = sorted(set(payload) - accepted)
    if unknown:
        raise ValidationError(
            f"{cls.__name__} does not accept "
            + ", ".join(repr(key) for key in unknown)
            + f"; accepted keys: {', '.join(sorted(accepted)) or '(none)'}"
        )
    return cls(**payload)


def synthesis_capabilities() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for name in sorted(SYNTHESIS_TYPES):
        cls = SYNTHESIS_TYPES[name]
        entries.append(
            {
                "type": name,
                "class": cls.__name__,
                "kwargs": sorted(_constructor_keys(cls)),
            }
        )
    return entries


def _constructor_keys(cls: type) -> frozenset[str]:
    from inspect import Parameter, signature

    accepted: set[str] = set()
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None or init is object.__init__:
            continue
        for name, parameter in signature(init).parameters.items():
            if name in ("self", "node", "proposals", "task"):
                continue
            if parameter.kind in (Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL):
                continue
            accepted.add(name)
        break
    return frozenset(accepted)
