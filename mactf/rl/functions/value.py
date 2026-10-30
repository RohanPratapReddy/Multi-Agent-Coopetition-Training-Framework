"""Value-function wrappers (document 07 section 8.1).

``ValueEstimate`` is reused unchanged: the wrappers do not introduce a second
estimate type, because ``mean`` and ``squeeze()`` are already what callers want.
Each wrapper holds real parameters and reports them so an optimizer or an
aggregator can act on it; :class:`ReferenceValueFunction` and
:class:`ZeroValueFunction` report none, deliberately and explicitly.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Iterable, Mapping, Protocol, runtime_checkable

import torch
import torch.nn as nn

from ...coopetition.adapter import PeerAdapter
from ...exceptions import ContractError, ValidationError
from ..value import ValueEstimate, ValueHead
from ._spec import SpecMixin, build_from_kwargs


@runtime_checkable
class ValueFunction(Protocol):
    """``V(s)`` with the lifecycle an optimizer needs."""

    def __call__(self, states: torch.Tensor, **kwargs: Any) -> ValueEstimate: ...

    def parameters(self) -> Iterable[nn.Parameter]: ...

    def train(self, mode: bool = True) -> "ValueFunction": ...

    def eval(self) -> "ValueFunction": ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, payload: Mapping[str, Any]) -> None: ...

    def describe(self) -> dict[str, Any]: ...


class _ValueBase(SpecMixin):
    """Shared parameter/train plumbing so no wrapper forgets half of the lifecycle."""

    spec_family = "value"
    spec_type: ClassVar[str] = ""

    def __init__(self) -> None:
        self._training = True

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "_ValueBase":
        return _resolve(cls, spec)

    # -- lifecycle -------------------------------------------------------
    def parameters(self) -> Iterable[nn.Parameter]:
        return ()

    def train(self, mode: bool = True) -> "_ValueBase":
        self._training = bool(mode)
        for parameter in self.parameters():
            parameter.requires_grad_(bool(mode))
        return self

    def eval(self) -> "_ValueBase":
        return self.train(False)

    @property
    def training(self) -> bool:
        return self._training

    # -- state -----------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {"class": type(self).__name__, "spec_type": self.spec_type, "training": self._training}

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if payload.get("class") not in (None, type(self).__name__):
            raise ValueError(
                f"{type(self).__name__} cannot load state written by {payload.get('class')!r}"
            )
        self._training = bool(payload.get("training", self._training))

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.spec_type,
            "class": type(self).__name__,
            "training": self._training,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
        }

    def __call__(self, states: torch.Tensor, **kwargs: Any) -> ValueEstimate:  # pragma: no cover
        raise NotImplementedError


def _resolve(cls: type, spec: Mapping[str, Any]) -> Any:
    payload = dict(spec or {})
    kind = payload.pop("type", None)
    if kind is not None and kind != cls.spec_type:
        raise ValidationError(
            f"{cls.__name__}.from_spec cannot build type {kind!r}; it builds {cls.spec_type!r}"
        )
    return build_from_kwargs(cls, payload)


class ModuleValueFunction(_ValueBase):
    """The value head owned by a :class:`~mactf.coopetition.adapter.PeerAdapter`.

    Behaviour matches ``PeerAdapter._ensure_value`` exactly: with
    ``value_strategy="none"`` the estimate is zeros of the right shape and no head
    is ever attached, which is the explicit pure-policy-gradient case.
    """

    spec_type = "module"

    def __init__(self, adapter: PeerAdapter, *, use_native: bool = True) -> None:
        super().__init__()
        if not isinstance(adapter, PeerAdapter):
            raise ValidationError(
                f"ModuleValueFunction needs a PeerAdapter, got {type(adapter).__name__}"
            )
        self.adapter = adapter
        self.use_native = bool(use_native)

    def parameters(self) -> Iterable[nn.Parameter]:
        head = self.adapter.value_head
        if head is None:
            return ()
        return head.parameters()

    def __call__(self, states: torch.Tensor, **kwargs: Any) -> ValueEstimate:
        logits, value = self.adapter(states, **kwargs)
        if not self.use_native:
            value = self.adapter._ensure_value(logits, None)
        return ValueEstimate(values=value)

    def train(self, mode: bool = True) -> "ModuleValueFunction":
        # The head lives on the adapter, so the adapter's own mode is what
        # decides dropout and batch norm. Delegate rather than shadow it.
        self.adapter.train(bool(mode))
        return super().train(mode)  # type: ignore[return-value]

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update(
            {
                "peer": self.adapter.peer_name,
                "value_strategy": self.adapter.value_strategy,
                "uses_native_value": bool(self.adapter.contract.outputs_value),
            }
        )
        return payload


class CriticValueFunction(_ValueBase):
    """A standalone critic over a hidden vector, with its own parameters."""

    spec_type = "critic"

    def __init__(
        self,
        hidden_dim: int | None = None,
        *,
        head: ValueHead | None = None,
        width: int | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if head is not None:
            self.head = head
        elif hidden_dim is not None:
            self.head = ValueHead(int(hidden_dim), width=width)
        else:
            raise ValidationError("CriticValueFunction needs 'hidden_dim' or an existing 'head'")
        if device is not None or dtype is not None:
            self.head = self.head.to(
                device=device if device is not None else None,
                dtype=dtype if dtype is not None else None,
            )

    def parameters(self) -> Iterable[nn.Parameter]:
        return self.head.parameters()

    def __call__(self, states: torch.Tensor, **kwargs: Any) -> ValueEstimate:
        return ValueEstimate(values=self.head(states))

    def train(self, mode: bool = True) -> "CriticValueFunction":
        self.head.train(bool(mode))
        return super().train(mode)  # type: ignore[return-value]

    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), "head": self.head.state_dict()}

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        super().load_state_dict(payload)
        if payload.get("head"):
            self.head.load_state_dict(payload["head"])

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload["hidden_dim"] = (
            self.head[0].in_features if isinstance(self.head, nn.Sequential) else None
        )
        return payload


class ReferenceValueFunction(_ValueBase):
    """A frozen peer adapter used as a baseline.

    ``parameters()`` is empty and ``train()`` never re-enables gradients, because
    the reference is the thing the trained policy must not drift from
    (framework.md 11.5). Freezing here is what makes that structural: a caller
    cannot resume training on the baseline by accident.
    """

    spec_type = "reference"

    def __init__(self, adapter: PeerAdapter) -> None:
        super().__init__()
        if not isinstance(adapter, PeerAdapter):
            raise ValidationError(
                f"ReferenceValueFunction needs a PeerAdapter, got {type(adapter).__name__}"
            )
        self.adapter = adapter
        adapter.eval()
        for parameter in adapter.parameters():
            parameter.requires_grad_(False)

    def parameters(self) -> Iterable[nn.Parameter]:
        return ()

    @torch.no_grad()
    def __call__(self, states: torch.Tensor, **kwargs: Any) -> ValueEstimate:
        _, value = self.adapter(states, **kwargs)
        return ValueEstimate(values=value.detach())

    def train(self, mode: bool = True) -> "ReferenceValueFunction":
        # Deliberately ignores the request: the reference is always in eval mode.
        self._training = False
        self.adapter.eval()
        return self

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload["peer"] = self.adapter.peer_name
        payload["frozen"] = True
        return payload


class ZeroValueFunction(_ValueBase):
    """Zeros with the right shape: pure policy gradient, stated explicitly."""

    spec_type = "zero"

    def __init__(self, *, trailing_dim: bool = True) -> None:
        super().__init__()
        self.trailing_dim = bool(trailing_dim)

    def __call__(self, states: torch.Tensor, **kwargs: Any) -> ValueEstimate:
        shape = states.shape[:-1] + ((1,) if self.trailing_dim else ())
        return ValueEstimate(
            values=torch.zeros(shape, device=states.device, dtype=states.dtype)
        )


class ImageValueFunction(_ValueBase):
    """A critic over a pooled feature map, for vision peers.

    Included because "and so on" must be a closed list: a vision adapter produces
    ``(batch, channels, height, width)`` and a plain ``ValueHead`` over the last
    axis would be a silent shape bug rather than an error.
    """

    spec_type = "image"

    def __init__(
        self,
        hidden_dim: int | None = None,
        *,
        head: ValueHead | None = None,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        if pool not in {"mean", "max", "flatten"}:
            raise ValidationError(f"unknown pool {pool!r}; use 'mean', 'max', or 'flatten'")
        self.pool = pool
        if head is not None:
            self.head = head
        elif hidden_dim is not None:
            self.head = ValueHead(int(hidden_dim))
        else:
            raise ValidationError("ImageValueFunction needs 'hidden_dim' or an existing 'head'")

    def parameters(self) -> Iterable[nn.Parameter]:
        return self.head.parameters()

    def __call__(self, states: torch.Tensor, **kwargs: Any) -> ValueEstimate:
        if states.ndim < 3:
            raise ContractError(
                f"ImageValueFunction expects (batch, ..., features), got {tuple(states.shape)}"
            )
        if self.pool == "mean":
            pooled = states.flatten(start_dim=1, end_dim=-2).mean(dim=1)
        elif self.pool == "max":
            pooled = states.flatten(start_dim=1, end_dim=-2).amax(dim=1)
        else:
            pooled = states.flatten(start_dim=1)
            expected = (
                self.head[0].in_features if isinstance(self.head, nn.Sequential) else pooled.shape[-1]
            )
            if pooled.shape[-1] != expected:
                raise ContractError(
                    f"ImageValueFunction flatten produced {pooled.shape[-1]} features but "
                    f"the head expects {expected}; pass hidden_dim={pooled.shape[-1]} or "
                    "use pool='mean'"
                )
        return ValueEstimate(values=self.head(pooled))

    def train(self, mode: bool = True) -> "ImageValueFunction":
        self.head.train(bool(mode))
        return super().train(mode)  # type: ignore[return-value]


class NestedValueFunction(_ValueBase):
    """Composes child estimates into a parent estimate through a ``RollupStrategy``.

    The children are themselves ``ValueFunction`` objects, so a tree of any depth
    is expressible. The rollup consumes ``ScorerOutput`` objects in scoring, so the
    estimates are wrapped in a minimal adapter that carries the child's value and
    its void rules (a child whose estimate is void still voids the parent).
    """

    spec_type = "nested"

    def __init__(
        self,
        children: Mapping[str, ValueFunction] | None = None,
        *,
        rollup: Any = None,
        weights: Mapping[str, float] | None = None,
        parent: str = "root",
    ) -> None:
        super().__init__()
        self.children: dict[str, ValueFunction] = dict(children or {})
        if rollup is None:
            from .rollup import KappaWeightedRollup

            rollup = KappaWeightedRollup()
        if not hasattr(rollup, "rollup"):
            raise ValidationError(
                f"NestedValueFunction rollup must expose .rollup(), got {type(rollup).__name__}"
            )
        self.rollup = rollup
        self.weights = dict(weights or {})
        self.parent = parent

    def add(self, name: str, child: ValueFunction) -> "NestedValueFunction":
        self.children[name] = child
        return self

    def parameters(self) -> Iterable[nn.Parameter]:
        seen: set[int] = set()
        for child in self.children.values():
            for parameter in child.parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                yield parameter

    def __call__(self, states: torch.Tensor, **kwargs: Any) -> ValueEstimate:
        from ...scoring.base import ScorerOutput, ScoringKind

        if not self.children:
            return ValueEstimate(
                values=torch.zeros(
                    states.shape[:-1] + (1,), device=states.device, dtype=states.dtype
                )
            )
        outputs = []
        for name, child in self.children.items():
            estimate = child(states, **kwargs)
            outputs.append(
                ScorerOutput(
                    kind=ScoringKind.HIERARCHY,
                    amount=float(estimate.values.mean().item()),
                    breakdown={"child": name, "values_shape": list(estimate.values.shape)},
                    rules=(name,),
                )
            )
        rolled = self.rollup.rollup(
            self.parent, outputs, weights=self.weights or None
        )
        # The parent estimate broadcasts the rolled amount across the batch; that
        # is what a nested critic means when children are per-batch scalars.
        shape = states.shape[:-1] + (1,)
        return ValueEstimate(
            values=torch.full(
                shape, float(rolled.amount), device=states.device, dtype=states.dtype
            )
        )

    def train(self, mode: bool = True) -> "NestedValueFunction":
        for child in self.children.values():
            child.train(mode)
        self._training = bool(mode)
        return self

    def state_dict(self) -> dict[str, Any]:
        return {
            **super().state_dict(),
            "children": {name: child.state_dict() for name, child in self.children.items()},
            "weights": dict(self.weights),
            "parent": self.parent,
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        super().load_state_dict(payload)
        for name, state in (payload.get("children") or {}).items():
            if name in self.children:
                self.children[name].load_state_dict(state)
        self.weights = dict(payload.get("weights", self.weights))
        self.parent = str(payload.get("parent", self.parent))

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update(
            {
                "children": sorted(self.children),
                "rollup": getattr(self.rollup, "name", type(self.rollup).__name__),
                "parent": self.parent,
            }
        )
        return payload


class CallableValueFunction(_ValueBase):
    """Adapts ``fn(states, **kwargs) -> Tensor`` (or a ``ValueEstimate``)."""

    spec_type = "callable"

    def __init__(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        parameters_fn: Callable[[], Iterable[nn.Parameter]] | None = None,
        state_fns: tuple[Callable[[], dict[str, Any]], Callable[[Mapping[str, Any]], None]] | None = None,
    ) -> None:
        super().__init__()
        if fn is None:
            raise ValidationError("CallableValueFunction requires an 'fn' callable")
        if not callable(fn):
            raise ValidationError(f"CallableValueFunction fn is not callable: {fn!r}")
        self.fn = fn
        self.parameters_fn = parameters_fn
        self.state_fns = state_fns

    def parameters(self) -> Iterable[nn.Parameter]:
        if self.parameters_fn is None:
            return ()
        return self.parameters_fn()

    def __call__(self, states: torch.Tensor, **kwargs: Any) -> ValueEstimate:
        result = self.fn(states, **kwargs)
        if isinstance(result, ValueEstimate):
            return result
        if not torch.is_tensor(result):
            raise ContractError(
                f"CallableValueFunction returned {type(result).__name__}, "
                "expected a tensor or ValueEstimate"
            )
        return ValueEstimate(values=result)

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        if self.state_fns is not None:
            payload["fn_state"] = dict(self.state_fns[0]())
        return payload

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        super().load_state_dict(payload)
        if self.state_fns is not None and "fn_state" in payload:
            self.state_fns[1](payload["fn_state"])

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload["fn"] = getattr(self.fn, "__name__", repr(self.fn))
        payload["has_parameters"] = self.parameters_fn is not None
        return payload


VALUE_FUNCTION_TYPES: dict[str, type] = {
    "module": ModuleValueFunction,
    "critic": CriticValueFunction,
    "reference": ReferenceValueFunction,
    "zero": ZeroValueFunction,
    "image": ImageValueFunction,
    "nested": NestedValueFunction,
    "callable": CallableValueFunction,
}

#: The classes the registry imports. Kept as a tuple so ``registry`` does not need
#: to know the mapping above.
VALUE_CLASSES = tuple(VALUE_FUNCTION_TYPES.values())


__all__ = [
    "CallableValueFunction",
    "CriticValueFunction",
    "ImageValueFunction",
    "ModuleValueFunction",
    "NestedValueFunction",
    "ReferenceValueFunction",
    "VALUE_CLASSES",
    "VALUE_FUNCTION_TYPES",
    "ValueFunction",
    "ZeroValueFunction",
]
