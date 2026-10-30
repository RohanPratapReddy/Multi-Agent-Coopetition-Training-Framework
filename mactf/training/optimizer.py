"""The optimizer layer (document 08 section 2).

``build_peer`` hardcodes ``torch.optim.Adam``, and ``Peer.to()`` rebuilds it as
Adam after a device move, which silently discards a user's optimizer choice. This
module replaces that with a facade that keeps the choice:

* :class:`TorchOptimizer` wraps a torch optimizer class, a ``functools.partial``,
  or an already-built instance.
* :class:`ScipyOptimizer` runs a full-batch scipy method over a closure.
* :class:`CallableOptimizer` supports imperative and functional custom updates.
* :class:`ParamGroupRouter` routes parameters to different optimizers by name.
* :class:`MACTFOptimizer` is the facade the trainer holds: it owns gradient
  accumulation, clipping, AMP ordering, the scheduler, and serialization.

AMP ordering lives here rather than in the trainer, because
``unscale_ -> clip -> step`` is the order whose violation changes gradient norms
(``orchestration/phases.py:206-212``).
"""

from __future__ import annotations

import copy
import functools
import inspect
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import torch
import torch.nn as nn

from ..exceptions import ContractError, ImproperlyConfigured, ValidationError

__all__ = [
    "CallableOptimizer",
    "MACTFOptimizer",
    "OptimizerBackend",
    "ParamGroupRouter",
    "Route",
    "SCIPY_METHODS",
    "ScipyOptimizer",
    "StepOutcome",
    "TorchOptimizer",
    "build_optimizer",
    "describe_optimizer_spec",
]


#: The scipy methods this optimizer supports, and what each needs.
#:
#: ``jac`` means "an analytic gradient can be supplied"; ``hess`` means the method
#: additionally requires a Hessian or a Hessian-vector product, which is rejected
#: explicitly rather than failing inside scipy with a confusing message.
SCIPY_METHODS: dict[str, dict[str, bool]] = {
    "L-BFGS-B": {"jac": True, "hess": False},
    "BFGS": {"jac": True, "hess": False},
    "CG": {"jac": True, "hess": False},
    "Newton-CG": {"jac": True, "hess": True},
    "Nelder-Mead": {"jac": False, "hess": False},
    "Powell": {"jac": False, "hess": False},
    "SLSQP": {"jac": True, "hess": False},
}


@runtime_checkable
class OptimizerBackend(Protocol):
    """The structural contract a backend must satisfy.

    Satisfied by ``torch.optim.Optimizer`` for everything but ``describe``, which
    is why :class:`TorchOptimizer` is a thin adapter rather than a rewrite.
    """

    @property
    def param_groups(self) -> list[dict[str, Any]]: ...

    @property
    def learning_rate(self) -> float: ...

    @learning_rate.setter
    def learning_rate(self, value: float) -> None: ...

    def zero_grad(self, set_to_none: bool = True) -> None: ...

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> Any: ...

    def add_param_group(self, group: dict[str, Any]) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...

    def describe(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class StepOutcome:
    """What one ``MACTFOptimizer.step()`` call did."""

    stepped: bool
    grad_norm: float | None = None
    lr: float = 0.0
    skipped_reason: str | None = None
    accumulated: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "stepped": self.stepped,
            "grad_norm": self.grad_norm,
            "lr": self.lr,
            "skipped_reason": self.skipped_reason,
            "accumulated": self.accumulated,
        }


def _trainable(parameters: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    """Frozen parameters are filtered out before binding, always.

    ``build_peer`` already does this (``peer.py:104``), and a PEFT model is mostly
    frozen; handing frozen tensors to Adam allocates state that will never be
    touched, which for a 7B model is gigabytes.
    """
    seen: set[int] = set()
    selected: list[nn.Parameter] = []
    for parameter in parameters:
        if not isinstance(parameter, nn.Parameter):
            raise ContractError(
                f"optimizer parameters must be nn.Parameter, got {type(parameter).__name__}"
            )
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            # A tied parameter appearing twice would receive a doubled update.
            continue
        seen.add(id(parameter))
        selected.append(parameter)
    return selected


class _Backend:
    """Shared plumbing: binding, learning-rate access, and honest ``describe``."""

    name: str = "backend"

    #: ``CallableOptimizer`` may legitimately operate on a frozen model, so an
    #: empty parameter list is not an error everywhere.
    requires_parameters: bool = True

    def __init__(self) -> None:
        self._parameters: list[nn.Parameter] = []
        self._bound = False
        self._learning_rate = 0.0
        self._names: dict[int, str] = {}

    # -- binding ---------------------------------------------------------
    @property
    def bound(self) -> bool:
        return self._bound

    @property
    def parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self._parameters)

    def bind(self, parameters: Iterable[nn.Parameter]) -> "_Backend":
        self._parameters = _trainable(parameters)
        if not self._parameters and self.requires_parameters:
            raise ContractError(
                f"{type(self).__name__} was bound to no trainable parameters"
            )
        self._bound = True
        return self

    def name_parameters(self, module: nn.Module, *, prefix: str = "") -> "_Backend":
        """Record fully qualified names, which :class:`ParamGroupRouter` matches on."""
        self._names = {
            id(parameter): f"{prefix}{name}" for name, parameter in module.named_parameters()
        }
        return self

    @property
    def parameter_names(self) -> dict[int, str]:
        return dict(self._names)

    # -- contract --------------------------------------------------------
    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self._learning_rate = float(value)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return []

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self._parameters:
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.detach_()
                    parameter.grad.zero_()

    def add_param_group(self, group: dict[str, Any]) -> None:
        raise ContractError(f"{type(self).__name__} does not support parameter groups")

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state:
            raise ContractError(f"{type(self).__name__} cannot restore optimizer state")

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "class": type(self).__name__,
            "lr": self._learning_rate,
            "parameters": sum(p.numel() for p in self._parameters),
            "stateful": True,
        }


class TorchOptimizer(_Backend):
    """A thin adapter over a torch optimizer.

    Three accepted forms, each real::

        TorchOptimizer(torch.optim.AdamW, lr=3e-4, weight_decay=0.01)
        TorchOptimizer(functools.partial(torch.optim.SGD, momentum=0.9), lr=0.1)
        TorchOptimizer(torch.optim.AdamW(params, lr=1e-4))

    Construction is lazy for the first two so the trainer can bind the final
    parameter list after freezing, PEFT wrapping, and device movement.
    """

    name = "torch"

    def __init__(
        self,
        candidate: Any,
        *,
        lr: float | None = None,
        weight_decay: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.kwargs: dict[str, Any] = dict(kwargs)
        self.weight_decay = float(weight_decay)
        self._instance: torch.optim.Optimizer | None = None
        self._factory: Callable[..., torch.optim.Optimizer] | None = None
        self._class_name = ""

        if isinstance(candidate, torch.optim.Optimizer):
            self._instance = candidate
            self._class_name = type(candidate).__name__
            groups = candidate.param_groups
            if groups:
                self._learning_rate = float(groups[0].get("lr", 0.0))
                self._parameters = [p for group in groups for p in group["params"]]
            self._bound = True
        elif isinstance(candidate, functools.partial):
            self._factory = candidate
            self._class_name = getattr(candidate.func, "__name__", "partial")
            keywords = dict(candidate.keywords or {})
            if lr is None and "lr" in keywords:
                self._learning_rate = float(keywords["lr"])
        elif isinstance(candidate, type) and issubclass(candidate, torch.optim.Optimizer):
            self._factory = candidate
            self._class_name = candidate.__name__
        elif callable(candidate):
            self._factory = candidate
            self._class_name = getattr(candidate, "__name__", type(candidate).__name__)
        else:
            raise ContractError(
                "TorchOptimizer accepts a torch optimizer class, a functools.partial, "
                f"or a built torch.optim.Optimizer; got {type(candidate).__name__}"
            )

        if lr is not None:
            self._learning_rate = float(lr)
        if self._factory is not None:
            if self._learning_rate <= 0.0:
                raise ValidationError(
                    f"TorchOptimizer({self._class_name}) requires lr > 0, "
                    f"got {self._learning_rate!r}"
                )

    # -- binding ---------------------------------------------------------
    def bind(self, parameters: Iterable[nn.Parameter]) -> "TorchOptimizer":
        if self._instance is not None:
            # An already-built optimizer owns its parameters; rebinding would
            # either duplicate them or drop the caller's grouping.
            self._parameters = [
                p for group in self._instance.param_groups for p in group["params"]
            ]
            self._bound = True
            return self
        super().bind(parameters)
        assert self._factory is not None
        kwargs = dict(self.kwargs)
        kwargs.setdefault("lr", self._learning_rate)
        if self.weight_decay:
            kwargs.setdefault("weight_decay", self.weight_decay)
        self._instance = self._factory(self._parameters, **kwargs)
        self._learning_rate = float(
            self._instance.param_groups[0].get("lr", self._learning_rate)
        )
        return self

    # -- contract --------------------------------------------------------
    @property
    def optimizer(self) -> torch.optim.Optimizer:
        if self._instance is None:
            raise ContractError(
                f"TorchOptimizer({self._class_name}) is not bound; call bind(parameters) first"
            )
        return self._instance

    @property
    def class_name(self) -> str:
        return self._class_name

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return list(self._instance.param_groups) if self._instance is not None else []

    @property
    def learning_rate(self) -> float:
        if self._instance is not None and self._instance.param_groups:
            return float(self._instance.param_groups[0].get("lr", self._learning_rate))
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self._learning_rate = float(value)
        if self._instance is not None:
            for group in self._instance.param_groups:
                group["lr"] = self._learning_rate

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self._instance is not None:
            self._instance.zero_grad(set_to_none=set_to_none)
            return
        super().zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> Any:
        return self.optimizer.step(closure) if closure is not None else self.optimizer.step()

    def add_param_group(self, group: dict[str, Any]) -> None:
        self.optimizer.add_param_group(group)

    def state_dict(self) -> dict[str, Any]:
        if self._instance is None:
            return {}
        # A snapshot, not live references: torch's own state_dict aliases the moment and
        # step tensors, so a later in-place step would silently rewrite a saved state.
        return copy.deepcopy(self._instance.state_dict())

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not state:
            return
        if self._instance is None:
            raise ContractError(
                f"TorchOptimizer({self._class_name}) must be bound before restoring state"
            )
        self._instance.load_state_dict(copy.deepcopy(dict(state)))
        self._learning_rate = float(self._instance.param_groups[0].get("lr", self._learning_rate))

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update(
            {
                "class": self._class_name,
                "weight_decay": self.weight_decay,
                "kwargs": sorted(self.kwargs),
                "built": self._instance is not None,
            }
        )
        return payload


def _require_scipy():
    """Import ``scipy.optimize`` or say exactly which extra provides it."""
    try:
        import scipy.optimize as optimize  # noqa: PLC0415 - optional dependency
    except ImportError as exc:
        raise ImproperlyConfigured(
            "ScipyOptimizer requires scipy and numpy; install them with "
            '`pip install "mactf[scipy]"`'
        ) from exc
    return optimize


class ScipyOptimizer(_Backend):
    """A full-batch scipy optimizer driven by a closure.

    Honest limitation, stated in ``describe()``: scipy methods are full-batch and
    cannot consume a minibatch stream, so the trainer runs them at epoch
    granularity after the epoch's data is assembled.
    """

    name = "scipy"

    def __init__(
        self,
        method: str = "L-BFGS-B",
        *,
        closure_required: bool = True,
        max_iter: int = 20,
        tol: float = 1e-6,
        options: Mapping[str, Any] | None = None,
        dtype: torch.dtype = torch.float64,
        learning_rate: float = 0.0,
    ) -> None:
        super().__init__()
        _require_scipy()
        if method not in SCIPY_METHODS:
            raise ValidationError(
                f"unknown scipy method {method!r}; accepted: "
                + ", ".join(sorted(SCIPY_METHODS))
            )
        self.method = method
        self.closure_required = bool(closure_required)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.options: dict[str, Any] = dict(options or {})
        self.dtype = dtype
        self._learning_rate = float(learning_rate)
        self.last_result: dict[str, Any] | None = None
        self._offsets: list[tuple[int, int]] = []
        self._total = 0

    # -- binding ---------------------------------------------------------
    def bind(self, parameters: Iterable[nn.Parameter]) -> "ScipyOptimizer":
        super().bind(parameters)
        self._offsets = []
        cursor = 0
        for parameter in self._parameters:
            count = parameter.numel()
            self._offsets.append((cursor, cursor + count))
            cursor += count
        self._total = cursor
        return self

    # -- helpers ---------------------------------------------------------
    def _flatten(self) -> torch.Tensor:
        """The parameter vector in ``float64`` on CPU, which scipy requires.

        ``float32`` accumulation inside L-BFGS-B loses the line-search precision
        that makes it useful, so the vector is always float64.
        """
        pieces = [
            parameter.detach().reshape(-1).to(device="cpu", dtype=self.dtype)
            for parameter in self._parameters
        ]
        return torch.cat(pieces) if pieces else torch.zeros(0, dtype=self.dtype)

    def _assign(self, vector: torch.Tensor) -> None:
        """Write a candidate vector back, restoring each parameter's own dtype."""
        with torch.no_grad():
            for parameter, (start, end) in zip(self._parameters, self._offsets):
                chunk = vector[start:end].to(device=parameter.device, dtype=parameter.dtype)
                parameter.copy_(chunk.reshape(parameter.shape))

    def _gradient(self) -> torch.Tensor:
        pieces = []
        for parameter in self._parameters:
            if parameter.grad is None:
                pieces.append(torch.zeros(parameter.numel(), dtype=self.dtype))
            else:
                pieces.append(
                    parameter.grad.detach().reshape(-1).to(device="cpu", dtype=self.dtype)
                )
        return torch.cat(pieces) if pieces else torch.zeros(0, dtype=self.dtype)

    # -- stepping --------------------------------------------------------
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> Any:
        if closure is None:
            raise ContractError(
                f"ScipyOptimizer({self.method}) requires a closure; there is no silent "
                "fallback to a gradient step because a scipy optimizer without a "
                "closure is a configuration bug"
            )
        optimize = _require_scipy()

        spec = SCIPY_METHODS[self.method]
        if spec["hess"]:
            raise ImproperlyConfigured(
                f"{self.method} requires a Hessian or Hessian-vector product, which "
                "this optimizer does not supply; use L-BFGS-B, BFGS, or CG"
            )
        uses_jacobian = spec["jac"]
        initial = self._flatten()

        def objective(vector):
            tensor = torch.as_tensor(vector, dtype=self.dtype)
            self._assign(tensor)
            try:
                for parameter in self._parameters:
                    parameter.grad = None
                loss = closure()
                if not torch.is_tensor(loss):
                    loss = torch.as_tensor(float(loss), dtype=self.dtype)
                # ``torch.autograd.grad`` rather than ``loss.backward()``: scipy
                # calls this dozens of times and ``backward`` would accumulate
                # gradients across every evaluation.
                grads = torch.autograd.grad(
                    loss, self._parameters, allow_unused=True, retain_graph=False
                )
                for parameter, grad in zip(self._parameters, grads):
                    parameter.grad = (
                        torch.zeros_like(parameter) if grad is None else grad.detach()
                    )
                jacobian = self._gradient().numpy() if uses_jacobian else None
                return float(loss.detach().item()), jacobian
            finally:
                pass

        result = optimize.minimize(
            objective,
            initial.numpy(),
            method=self.method,
            jac=uses_jacobian,
            tol=self.tol,
            options={"maxiter": self.max_iter, **self.options},
        )
        self._assign(torch.as_tensor(result.x, dtype=self.dtype))
        self.last_result = {
            "nit": int(getattr(result, "nit", 0) or 0),
            "nfev": int(getattr(result, "nfev", 0) or 0),
            "fun": float(getattr(result, "fun", float("nan"))),
            "success": bool(getattr(result, "success", False)),
            "message": str(getattr(result, "message", "")),
        }
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "vector": self._flatten().cpu(),
            "last_result": dict(self.last_result or {}),
            "max_iter": self.max_iter,
            "tol": self.tol,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        vector = state.get("vector")
        if vector is not None:
            self._assign(torch.as_tensor(vector, dtype=self.dtype))
        self.last_result = dict(state.get("last_result") or {})
        if "max_iter" in state:
            self.max_iter = int(state["max_iter"])
        if "tol" in state:
            self.tol = float(state["tol"])

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update(
            {
                "method": self.method,
                "granularity": "epoch",
                "full_batch": True,
                "dtype": str(self.dtype).replace("torch.", ""),
                "last_result": dict(self.last_result or {}),
            }
        )
        return payload


class CallableOptimizer(_Backend):
    """A user function as the update rule.

    ``CallableOptimizer(fn)`` calls ``fn(parameters)`` imperatively.
    ``CallableOptimizer(fn, functional=True)`` calls ``fn(parameter) -> Tensor`` and
    copies the returned value into each parameter, under ``no_grad``. It exists for
    evolutionary strategies and hand-written second-order code.

    It is stateless, and :meth:`describe` says so rather than pretending a resume
    can restore state that does not exist.
    """

    name = "callable"
    requires_parameters = False

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        functional: bool = False,
        learning_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if not callable(fn):
            raise ContractError(
                f"CallableOptimizer needs a callable, got {type(fn).__name__}"
            )
        self.fn = fn
        self.functional = bool(functional)
        self._learning_rate = float(learning_rate)
        self.calls = 0

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> Any:
        self.calls += 1
        if self.functional:
            with torch.no_grad():
                for parameter in self._parameters:
                    updated = self.fn(parameter)
                    if updated is None:
                        continue
                    if not torch.is_tensor(updated):
                        raise ContractError(
                            "a functional CallableOptimizer must return a tensor per "
                            f"parameter, got {type(updated).__name__}"
                        )
                    if tuple(updated.shape) != tuple(parameter.shape):
                        raise ContractError(
                            "a functional CallableOptimizer returned a tensor of shape "
                            f"{tuple(updated.shape)} for a parameter of shape "
                            f"{tuple(parameter.shape)}"
                        )
                    parameter.data.copy_(updated.detach())
            return None
        return self.fn(list(self._parameters))

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state:
            raise ContractError(
                "CallableOptimizer is stateless; there is no optimizer state to restore"
            )

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update(
            {
                "functional": self.functional,
                "stateful": False,
                "calls": self.calls,
            }
        )
        return payload


@dataclass(frozen=True)
class Route:
    """One parameter-name route in a :class:`ParamGroupRouter`."""

    pattern: str
    optimizer: Any
    priority: int = 0

    def matches(self, name: str) -> bool:
        try:
            return re.fullmatch(self.pattern, name) is not None
        except re.error as error:  # pragma: no cover - validated on construction
            raise ValidationError(
                f"Route pattern {self.pattern!r} is not a valid regular expression: {error}"
            ) from error

    def describe(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "priority": self.priority,
            "optimizer": describe_optimizer_spec(self.optimizer),
        }


class ParamGroupRouter(_Backend):
    """Routes each trainable parameter to the first matching route, then default.

    ``pattern`` is matched with ``re.fullmatch`` against the fully qualified
    parameter name, so ``.*lora_.*`` and, in a hierarchy,
    ``peer-a/sub-1/.*lora_.*`` are both expressible.

    ``default`` is mandatory. A router without one would silently drop parameters
    when a new module appears, which is the realistic LoRA failure: an adapter
    that never trains.
    """

    name = "router"

    def __init__(
        self,
        routes: Sequence[Route] = (),
        *,
        default: Any = None,
    ) -> None:
        super().__init__()
        self.routes: list[Route] = list(routes)
        for route in self.routes:
            if not isinstance(route, Route):
                raise ValidationError(
                    f"ParamGroupRouter routes must be Route instances, got "
                    f"{type(route).__name__}"
                )
            route.matches("probe")  # validates the regex now, not at epoch 4
        if default is None:
            raise ValidationError(
                "ParamGroupRouter requires a default optimizer; routing without a "
                "default silently drops any parameter no route claims"
            )
        self.default: Any = default
        self._backends: dict[int, _Backend] = {}
        self._assignment: dict[str, str] = {}

    @staticmethod
    def _as_backend(candidate: Any) -> _Backend:
        if isinstance(candidate, _Backend):
            return candidate
        if (
            isinstance(candidate, torch.optim.Optimizer)
            or isinstance(candidate, (type, functools.partial))
            or callable(candidate)
        ):
            return TorchOptimizer(candidate)
        raise ValidationError(
            "a route optimizer must be a backend or a torch optimizer spec, got "
            f"{type(candidate).__name__}"
        )

    def bind(self, parameters: Iterable[nn.Parameter]) -> "ParamGroupRouter":
        self._parameters = _trainable(parameters)
        assignment: dict[int, list[nn.Parameter]] = {}
        owner = {id(parameter): name for name, parameter in self._named_source}

        ordered_routes = sorted(
            self.routes, key=lambda route: (-route.priority, route.pattern)
        )
        for parameter in self._parameters:
            name = owner.get(id(parameter), "")
            chosen = None
            for route in ordered_routes:
                if route.matches(name):
                    chosen = route.optimizer
                    break
            if chosen is None:
                chosen = self.default
            backend = self._as_backend(chosen)
            self._backends.setdefault(id(chosen), backend)
            assignment.setdefault(id(chosen), []).append(parameter)
            if name:
                self._assignment[name] = describe_optimizer_spec(chosen)

        for key, backend in self._backends.items():
            backend.bind(assignment[key])

        claimed = sum(len(group) for group in assignment.values())
        if claimed != len(self._parameters):
            missing = [
                name for name in owner.values() if name and name not in self._assignment
            ]
            raise ContractError(
                "parameter "
                + ", ".join(repr(name) for name in missing)
                + " is trainable but was not claimed by any ParamGroupRouter route or default"
            )
        self._bound = True
        return self

    #: The ``(name, parameter)`` list a caller can supply so routes match on real
    #: names. ``bind`` falls back to empty names, which routes to the default.
    _named_source: list[tuple[str, nn.Parameter]] = []

    def bind_named(
        self, named_parameters: Iterable[tuple[str, nn.Parameter]]
    ) -> "ParamGroupRouter":
        """Bind using fully qualified names, which is what routes match on."""
        pairs = [(str(name), parameter) for name, parameter in named_parameters]
        self._named_source = pairs
        try:
            return self.bind([parameter for _, parameter in pairs])
        finally:
            self._named_source = []

    @property
    def assignment(self) -> dict[str, str]:
        return dict(self._assignment)

    @property
    def learning_rate(self) -> float:
        backend = self._backends.get(id(self.default))
        return backend.learning_rate if backend is not None else 0.0

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        for backend in self._backends.values():
            backend.learning_rate = float(value)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for backend in self._backends.values():
            groups.extend(backend.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        for backend in self._backends.values():
            backend.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> Any:
        return [backend.step(closure) for backend in self._backends.values()]

    def state_dict(self) -> dict[str, Any]:
        return {
            "routes": {
                route.pattern: self._backends[id(route.optimizer)].state_dict()
                for route in self.routes
                if id(route.optimizer) in self._backends
            },
            "default": (
                self._backends[id(self.default)].state_dict()
                if id(self.default) in self._backends
                else {}
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        payload = dict(state or {})
        by_pattern = payload.get("routes", {})
        for route in self.routes:
            backend = self._backends.get(id(route.optimizer))
            if backend is not None and route.pattern in by_pattern:
                backend.load_state_dict(by_pattern[route.pattern])
        default_state = payload.get("default")
        default_backend = self._backends.get(id(self.default))
        if default_backend is not None and default_state:
            default_backend.load_state_dict(default_state)

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "class": "ParamGroupRouter",
            "routes": [route.describe() for route in self.routes],
            "default": describe_optimizer_spec(self.default),
            "learning_rate": self.learning_rate,
            "assigned": len(self._assignment),
            "stateful": True,
        }


class MACTFOptimizer:
    """The facade the trainer holds: backend, accumulation, clipping, AMP, state."""

    def __init__(
        self,
        backend: Any,
        *,
        lr: float | None = None,
        weight_decay: float = 0.0,
        param_groups: Sequence[Mapping[str, Any]] | None = None,
        router: ParamGroupRouter | None = None,
        grad_accumulation: int = 1,
        max_grad_norm: float | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        node_path: str = "",
    ) -> None:
        if grad_accumulation < 1:
            raise ValidationError(
                f"grad_accumulation must be >= 1, got {grad_accumulation!r}"
            )
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValidationError(
                f"max_grad_norm must be > 0 when set, got {max_grad_norm!r}"
            )
        if router is not None and backend is not None:
            raise ValidationError(
                "MACTFOptimizer accepts either a backend or a router, not both; the "
                "router owns parameter assignment"
            )
        self.router = router
        if router is not None:
            self.backend: Any = router
        elif backend is None:
            self.backend = TorchOptimizer(torch.optim.Adam, lr=lr or 1e-3)
        else:
            self.backend = _resolve_backend(
                backend, lr=lr, weight_decay=weight_decay
            )
        if lr is not None and self.backend.bound:
            self.backend.learning_rate = float(lr)
        if isinstance(self.backend, ScipyOptimizer) and grad_accumulation > 1:
            raise ValidationError(
                "ScipyOptimizer is full-batch and cannot be combined with "
                f"grad_accumulation={grad_accumulation}; use grad_accumulation=1"
            )
        self.weight_decay = float(weight_decay)
        self.param_groups = [dict(group) for group in (param_groups or ())]
        self.grad_accumulation = int(grad_accumulation)
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None
        self.scheduler = scheduler
        self.scaler = scaler
        self.node_path = str(node_path)
        self._bound = bool(getattr(self.backend, "bound", False))
        self._accumulated = 0
        self.history: list[dict[str, Any]] = []
        #: ``(name, kwargs)`` of a ``torch.optim.lr_scheduler`` class to build once
        #: ``bind`` has produced the torch optimizer it wraps. Set by ``from_block``.
        self._pending_scheduler: tuple[str, dict[str, Any]] | None = None

    # -- binding ---------------------------------------------------------
    def bind(
        self,
        parameters: Iterable[nn.Parameter],
        *,
        named: Iterable[tuple[str, nn.Parameter]] | None = None,
    ) -> "MACTFOptimizer":
        """Bind the final parameter list.

        ``named`` lets a :class:`ParamGroupRouter` match fully qualified names; a
        plain list still binds, with every parameter routed by the default.
        """
        if named is not None and isinstance(self.backend, ParamGroupRouter):
            self.backend.bind_named(named)
        else:
            self.backend.bind(parameters)
        self._bound = True
        for group in self.param_groups:
            self.backend.add_param_group(dict(group))
        if self._pending_scheduler is not None and self.scheduler is None:
            name, kwargs = self._pending_scheduler
            self.attach_scheduler(name, **kwargs)
        return self

    def attach_scheduler(self, name: str, **kwargs: Any) -> Any:
        """Build ``torch.optim.lr_scheduler.<name>`` over the bound torch optimizer.

        A scheduler is named, not constructed from YAML, because its parameters
        differ per class and a mapping cannot be validated without importing torch.
        """
        instance = getattr(self.backend, "_instance", None)
        if instance is None:
            raise ValidationError(
                f"scheduler {name!r} needs a bound torch optimizer; the "
                f"{type(self.backend).__name__} backend has none"
            )
        factory = getattr(torch.optim.lr_scheduler, str(name), None)
        if factory is None or not isinstance(factory, type):
            raise ValidationError(
                f"unknown scheduler {name!r}; torch.optim.lr_scheduler has no such class"
            )
        self.scheduler = factory(instance, **kwargs)
        self._pending_scheduler = None
        return self.scheduler

    @property
    def bound(self) -> bool:
        return self._bound

    @property
    def parameters(self) -> tuple[nn.Parameter, ...]:
        return self.backend.parameters

    @property
    def closure_based(self) -> bool:
        """True for a full-batch backend that re-evaluates the loss itself (scipy)."""
        return isinstance(self.backend, ScipyOptimizer)

    # -- the update ------------------------------------------------------
    def zero_grad(self, set_to_none: bool = True) -> None:
        self.backend.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> StepOutcome:
        """One optimizer step, or one accumulation tick.

        AMP ordering is ``unscale_ -> clip -> step -> update``. A scale reduction
        means the scaler refused the step (inf/NaN gradients), which is reported as
        ``skipped_reason`` rather than guessed at.
        """
        if self.closure_based:
            if closure is None:
                raise ContractError(
                    f"{type(self.backend).__name__} is closure-driven; MACTFOptimizer.step "
                    "needs a closure that recomputes the loss"
                )
            result = self.backend.step(closure)
            success = bool(getattr(result, "success", True))
            outcome = StepOutcome(
                stepped=True,
                grad_norm=None,
                lr=self.backend.learning_rate,
                skipped_reason=None if success else "scipy reported an unsuccessful solve",
                accumulated=1,
            )
            self.history.append(outcome.as_dict())
            return outcome
        self._accumulated += 1
        if self.grad_accumulation > 1 and self._accumulated % self.grad_accumulation != 0:
            outcome = StepOutcome(
                stepped=False,
                lr=self.backend.learning_rate,
                skipped_reason="gradient accumulation",
                accumulated=self._accumulated,
            )
            self.history.append(outcome.as_dict())
            return outcome

        grad_norm: float | None = None
        skipped: str | None = None
        if self.scaler is not None:
            # ``unscale_`` before clipping, because clipping scaled gradients
            # changes the ratio the clip is meant to enforce.
            native = getattr(self.backend, "optimizer", self.backend)
            self.scaler.unscale_(native)
            grad_norm = self._clip()
            previous = self.scaler.get_scale()
            self.scaler.step(native)
            self.scaler.update()
            if self.scaler.get_scale() < previous:
                skipped = "AMP scaler skipped the step (inf or NaN gradients)"
        else:
            grad_norm = self._clip()
            self.backend.step()
            if closure is not None:
                closure()

        if self.scheduler is not None and skipped is None:
            self.scheduler.step()
        outcome = StepOutcome(
            stepped=skipped is None,
            grad_norm=grad_norm,
            lr=self.backend.learning_rate,
            skipped_reason=skipped,
            accumulated=self._accumulated,
        )
        if skipped is None:
            self._accumulated = 0
        self.history.append(outcome.as_dict())
        return outcome

    def _clip(self) -> float | None:
        if self.max_grad_norm is None:
            return None
        return _clip_grad_norm_(self.backend.parameters, self.max_grad_norm)

    # -- learning rate ---------------------------------------------------
    @property
    def learning_rate(self) -> float:
        return self.backend.learning_rate

    def set_lr(self, value: float, *, group: int | None = None) -> None:
        target = float(value)
        if target <= 0:
            raise ValidationError(f"learning rate must be > 0, got {value!r}")
        groups = self.backend.param_groups
        if group is None:
            self.backend.learning_rate = target
            for entry in groups:
                entry["lr"] = target
            return
        if group < 0 or group >= len(groups):
            raise ContractError(
                f"no parameter group {group}; the optimizer has {len(groups)}"
            )
        groups[group]["lr"] = target

    # -- device moves ----------------------------------------------------
    def move_to(self, runtime: Any) -> "MACTFOptimizer":
        """Rebind onto the moved parameters, preserving the backend's class.

        ``Peer.to()`` rebuilds Adam unconditionally, which discards the user's
        choice; this is the substitute that keeps an SGD(momentum=0.9) an SGD.
        """
        del runtime
        state = self.backend.state_dict()
        if isinstance(self.backend, (TorchOptimizer, ParamGroupRouter)):
            self.backend.bind(self.backend.parameters)
            self.backend.load_state_dict(_cpu_state(state))
        return self

    # -- persistence -----------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "node_path": self.node_path,
            "backend": getattr(self.backend, "name", type(self.backend).__name__),
            "class": self.backend.describe().get("class", type(self.backend).__name__),
            "lr": self.backend.learning_rate,
            "param_groups": _serialise_groups(self.backend.param_groups),
            "state": self.backend.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "grad_accumulation": self.grad_accumulation,
            "max_grad_norm": self.max_grad_norm,
        }
        if self.router is not None:
            payload["router"] = self.router.describe()
        return payload

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        state = dict(payload or {})
        if not self._bound:
            raise ContractError(
                "MACTFOptimizer.load_state_dict requires a bound optimizer; call "
                "bind(parameters) first"
            )
        if state.get("state"):
            self.backend.load_state_dict(_cpu_state(state["state"]))
        if "lr" in state:
            self.set_lr(float(state["lr"]))
        if self.scheduler is not None and state.get("scheduler"):
            self.scheduler.load_state_dict(state["scheduler"])
        if self.scaler is not None and state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])
        self.grad_accumulation = int(state.get("grad_accumulation", self.grad_accumulation))
        self.max_grad_norm = state.get("max_grad_norm", self.max_grad_norm)
        self.node_path = str(state.get("node_path", self.node_path))

    def describe(self) -> dict[str, Any]:
        payload = dict(self.backend.describe())
        payload.update(
            {
                "node_path": self.node_path,
                "lr": self.backend.learning_rate,
                "grad_accumulation": self.grad_accumulation,
                "max_grad_norm": self.max_grad_norm,
                "scheduler": type(self.scheduler).__name__ if self.scheduler else None,
                "scaler": type(self.scaler).__name__ if self.scaler else None,
                "router": self.router.describe() if self.router is not None else None,
            }
        )
        return payload

    # -- construction ----------------------------------------------------
    @classmethod
    def from_spec(cls, spec: Mapping[str, Any] | Any) -> "MACTFOptimizer":
        if spec is None:
            return cls(TorchOptimizer(torch.optim.Adam, lr=1e-3))
        if isinstance(spec, MACTFOptimizer):
            return spec
        if not isinstance(spec, Mapping):
            return cls(spec)

        # An unset field (``None``) means "the backend's own default", never the literal None.
        payload = {key: value for key, value in dict(spec).items() if value is not None}
        kind = str(payload.pop("type", payload.pop("backend", "torch"))).lower()
        lr = payload.pop("lr", payload.pop("learning_rate", None))
        weight_decay = float(payload.pop("weight_decay", 0.0))
        grad_accumulation = int(payload.pop("grad_accumulation", 1))
        max_grad_norm = payload.pop("max_grad_norm", None)
        node_path = str(payload.pop("node_path", ""))
        common = {
            "lr": lr,
            "weight_decay": weight_decay,
            "grad_accumulation": grad_accumulation,
            "max_grad_norm": max_grad_norm,
            "node_path": node_path,
        }

        if kind == "torch":
            name = str(
                payload.pop("class", payload.pop("class_name", payload.pop("name", "Adam")))
            )
            for stale in ("method", "max_iter", "tol"):  # scipy-only fields carried by the API schema
                payload.pop(stale, None)
            factory = getattr(torch.optim, name, None)
            if factory is None:
                raise ValidationError(
                    f"unknown torch optimizer {name!r}; torch.optim has no such class"
                )
            return cls(
                TorchOptimizer(factory, lr=lr, weight_decay=weight_decay, **payload),
                **common,
            )
        if kind == "scipy":
            method = str(payload.pop("method", "L-BFGS-B"))
            return cls(ScipyOptimizer(method, **payload), **common)
        if kind == "callable":
            fn = payload.pop("fn", payload.pop("function", None))
            if fn is None:
                raise ValidationError("a callable optimizer spec requires 'fn'")
            return cls(CallableOptimizer(fn, **payload), **common)
        raise ValidationError(
            f"unknown optimizer backend {kind!r}; accepted: torch, scipy, callable"
        )

    @classmethod
    def from_block(cls, block: Mapping[str, Any], *, node_path: str = "") -> "MACTFOptimizer":
        """``OPTIMIZER`` (upper-case YAML) to an optimizer facade.

        ``node_path`` is per-peer, so it is an argument here and never a key in a
        file shared by every peer. A scheduler named by ``SCHEDULER`` is built when
        the optimizer is bound, because it wraps the torch optimizer ``bind`` creates.
        """
        payload = optimizer_spec(block)
        scheduler = _scheduler_of(block)
        payload["node_path"] = str(node_path)
        optimizer = cls.from_spec(payload)
        optimizer._pending_scheduler = scheduler
        return optimizer


#: Defaults ``OPTIMIZER`` ships for the keys a non-Adam torch optimizer rejects.
_ADAM_ONLY_DEFAULTS: dict[str, Any] = {"betas": (0.9, 0.999), "eps": 1e-8}


def _scheduler_of(block: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """``(name, kwargs)`` for ``SCHEDULER`` / ``SCHEDULER_KWARGS``, or ``None``."""
    lowered = {str(key).lower(): value for key, value in dict(block).items()}
    name = lowered.get("scheduler")
    if not name:
        return None
    kwargs = lowered.get("scheduler_kwargs") or {}
    if not isinstance(kwargs, Mapping):
        raise ValidationError(f"OPTIMIZER.SCHEDULER_KWARGS must be a mapping, got {kwargs!r}")
    return str(name), dict(kwargs)


def optimizer_spec(block: Mapping[str, Any]) -> dict[str, Any]:
    """``OPTIMIZER`` (upper case, YAML) to ``from_spec``'s lower-case payload.

    * keys are lowered, and a ``None`` is dropped so the backend's own default applies;
    * ``SCHEDULER`` and ``SCHEDULER_KWARGS`` are removed, because a torch optimizer
      rejects an unknown keyword (see :meth:`MACTFOptimizer.from_block`);
    * ``BETAS`` becomes a tuple of floats, or is built from ``BETA0`` and ``BETA1``;
    * ``BETAS`` and ``EPS`` that the chosen torch class does not accept are dropped
      when they still hold the shipped default, so ``CLASS: SGD`` works in the file
      that also lists Adam's defaults. A customised value is passed on and torch
      reports it.
    """
    payload = {
        str(key).lower(): value
        for key, value in dict(block).items()
        if value is not None
    }
    payload.pop("scheduler", None)
    payload.pop("scheduler_kwargs", None)
    kind = str(payload.pop("type", payload.pop("backend", "torch"))).lower()
    payload["type"] = kind

    beta0, beta1 = payload.pop("beta0", None), payload.pop("beta1", None)
    betas = payload.pop("betas", None)
    if betas is None and beta0 is not None and beta1 is not None:
        betas = [beta0, beta1]

    if kind != "torch":
        # Adam's knobs mean nothing to a scipy or callable backend, which would
        # receive them as unexpected keywords.
        for key in ("class", "class_name", "name", "eps"):
            payload.pop(key, None)
        return payload

    name = str(payload.pop("class", payload.pop("class_name", payload.pop("name", "Adam"))))
    payload["class"] = name
    if betas is not None:
        payload["betas"] = tuple(float(item) for item in betas)

    factory = getattr(torch.optim, name, None)
    if factory is not None:
        try:
            accepted = set(inspect.signature(factory.__init__).parameters)
            takes_any = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in inspect.signature(factory.__init__).parameters.values()
            )
        except (TypeError, ValueError):  # pragma: no cover - builtin without a signature
            accepted, takes_any = set(), True
        if not takes_any:
            for key, default in _ADAM_ONLY_DEFAULTS.items():
                if key not in accepted and key in payload and payload[key] == default:
                    payload.pop(key)
    return payload


def _resolve_backend(candidate: Any, *, lr: float | None = None, weight_decay: float = 0.0) -> Any:
    if isinstance(candidate, _Backend):
        return candidate
    if isinstance(candidate, MACTFOptimizer):
        return candidate.backend
    if isinstance(candidate, torch.optim.Optimizer):
        return TorchOptimizer(candidate)
    if isinstance(candidate, (functools.partial, type)) or callable(candidate):
        return TorchOptimizer(candidate, lr=lr, weight_decay=weight_decay)
    raise ValidationError(
        "an optimizer must be a MACTFOptimizer, an OptimizerBackend, a torch optimizer "
        f"class, a functools.partial, or a built optimizer; got {type(candidate).__name__}"
    )


def _clip_grad_norm_(parameters: Sequence[nn.Parameter], max_norm: float) -> float:
    values = [parameter for parameter in parameters if parameter.grad is not None]
    if not values:
        return 0.0
    norm = torch.nn.utils.clip_grad_norm_(values, max_norm)
    return float(norm.detach().item()) if torch.is_tensor(norm) else float(norm)


def _cpu_state(state: Any) -> Any:
    """Map a state dict's tensors to CPU so a checkpoint is device-portable."""
    if torch.is_tensor(state):
        return state.detach().cpu()
    if isinstance(state, Mapping):
        return {key: _cpu_state(value) for key, value in state.items()}
    if isinstance(state, (list, tuple)):
        return type(state)(_cpu_state(value) for value in state)
    return state


def _serialise_groups(groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    serialised: list[dict[str, Any]] = []
    for group in groups:
        entry = {key: value for key, value in group.items() if key != "params"}
        entry["parameter_count"] = len(group.get("params", ()))
        serialised.append(entry)
    return serialised


def describe_optimizer_spec(candidate: Any) -> str:
    """A short, stable name for a backend spec, for checkpoints and logs."""
    if candidate is None:
        return "none"
    if isinstance(candidate, _Backend):
        return candidate.describe().get("class", type(candidate).__name__)
    if isinstance(candidate, functools.partial):
        return getattr(candidate.func, "__name__", "partial")
    if isinstance(candidate, type):
        return candidate.__name__
    if callable(candidate):
        return getattr(candidate, "__name__", type(candidate).__name__)
    return type(candidate).__name__


def build_optimizer(
    module: nn.Module,
    spec: Mapping[str, Any] | Any = None,
    *,
    lr: float | None = None,
) -> MACTFOptimizer:
    """Build and bind an optimizer over a module's trainable parameters."""
    optimizer = MACTFOptimizer.from_spec(spec)
    if lr is not None:
        optimizer.set_lr(lr)
    optimizer.bind(
        list(module.parameters()),
        named=list(module.named_parameters()),
    )
    return optimizer
