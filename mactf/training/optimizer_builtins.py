"""Built-in optimizer registrations and the two data-described backends (document 17 section 9.2)."""

from __future__ import annotations

from typing import Any, Callable, Iterator, Mapping, Sequence

import torch

from ..components.registry import HUB
from ..exceptions import ContractError, ValidationError
from .optimizer import (
    BackendBase,
    CallableOptimizer,
    MACTFOptimizer,
    ParamGroupRouter,
    Route,
    ScipyOptimizer,
)

#: torch classes that cannot work behind ``MACTFOptimizer.step`` (it passes no closure
#: to a non-scipy backend and the loss is dense), so they are not offered.
_TORCH_EXCLUDED = frozenset({"Optimizer", "LBFGS", "SparseAdam"})


class NullOptimizer(BackendBase):
    """No update rule. The node whose optimizer is ``null`` does not learn.

    ``bind`` keeps the trainable parameters so ``zero_grad`` can clear gradients a stray
    backward pass left; ``step`` is a no-op. Freezing the *node* (so it is skipped by the
    training loop and by write-back aggregation) is the freeze set's job (11.4); this
    backend only guarantees that nothing here moves a weight.
    """

    name = "null"
    requires_parameters = False

    def __init__(self) -> None:
        super().__init__()

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> Any:
        return None

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state:
            raise ContractError(
                "NullOptimizer has no state to restore; the checkpoint was written by a real "
                "optimizer (see E055 and COMPONENTS.STRICT_RESTORE)"
            )

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update({"stateful": False, "frozen": True, "lr": 0.0})
        return payload


def _nested_backend(spec: Any) -> BackendBase:
    """One route's optimizer, from a name, a mapping, or an existing backend."""
    if isinstance(spec, BackendBase):
        return spec
    if isinstance(spec, str):
        spec = {"type": spec}
    if not isinstance(spec, Mapping):
        raise ValidationError(
            f"a router optimizer is a name, a mapping or a backend, got {type(spec).__name__}"
        )
    data = {str(key).lower(): value for key, value in dict(spec).items() if value is not None}
    if str(data.get("type", "torch")).lower() == "router":
        raise ValidationError("a router cannot contain another router")
    return MACTFOptimizer.from_spec(data).backend


class RouterOptimizer(ParamGroupRouter):
    """``type: router``: a :class:`ParamGroupRouter` whose routes are data.

    .. code-block:: yaml

        OPTIMIZER:
          TYPE: router
          DEFAULT: {TYPE: AdamW, LR: 1.0e-4}
          ROUTES:
            - {PATTERN: ".*lora_.*", TYPE: Lion, LR: 3.0e-4, PRIORITY: 10}

    ``default`` is mandatory (a router with none silently drops a parameter no route claims),
    a pattern may appear once (``state_dict`` is keyed by pattern), and a router cannot hold
    a router.
    """

    name = "router"

    def __init__(self, *, default: Any, routes: Sequence[Mapping[str, Any]] = ()) -> None:
        built: list[Route] = []
        seen: set[str] = set()
        for index, entry in enumerate(routes):
            if not isinstance(entry, Mapping):
                raise ValidationError(
                    f"router route {index} must be a mapping, got {type(entry).__name__}"
                )
            data = {str(key).lower(): value for key, value in dict(entry).items()}
            pattern = data.pop("pattern", None)
            if not isinstance(pattern, str) or not pattern:
                raise ValidationError(f"router route {index} needs a non-empty 'pattern'")
            if pattern in seen:
                raise ValidationError(
                    f"router pattern {pattern!r} appears twice; the router's state is keyed by pattern"
                )
            seen.add(pattern)
            priority = int(data.pop("priority", 0))
            built.append(Route(pattern=pattern, optimizer=_nested_backend(data), priority=priority))
        super().__init__(built, default=_nested_backend(default))


def _torch_classes() -> Iterator[tuple[str, type]]:
    for name in sorted(dir(torch.optim)):
        candidate = getattr(torch.optim, name)
        if (
            isinstance(candidate, type)
            and issubclass(candidate, torch.optim.Optimizer)
            and name not in _TORCH_EXCLUDED
        ):
            yield name, candidate


def register_builtins() -> None:
    """Register every built-in optimizer. Idempotent; called once at the end of ``optimizer.py``."""
    registry = HUB.registry("optimizer")
    table: dict[str, type] = {
        "scipy": ScipyOptimizer,
        "callable": CallableOptimizer,
        "null": NullOptimizer,
        "router": RouterOptimizer,
    }
    table.update(dict(_torch_classes()))
    for name, factory in table.items():
        if registry.find(name) is None:
            registry.register_builtin(name, factory)
