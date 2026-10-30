"""One contract per kind (document 17 section 3.7).

A contract has two functions: ``contract(factory)`` at registration and
``instance_contract(instance)`` after construction. Both return a list of problem strings
(empty = fine). Framework classes are imported lazily, inside the function, so this module
(and ``mactf.components``) pulls in neither torch nor the scoring, coopetition or training
packages, and those packages can import ``mactf.components`` without a cycle.

Contracts for kinds whose later-phase classes do not exist yet (``reward_term``, the
trainer policy, the optimizer base class) are written against the structure the document
fixes and guard the class they will eventually check: see ``optimizer_contract``.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from .kinds import (
    KIND_SPECS,
    RL_FAMILIES,
    ComponentKind,
    KindSpec,
    Mutability,
    all_kind_ids,
)
from .reflect import reflect


def _is_class(factory: Any) -> bool:
    return inspect.isclass(factory)


def _callable_problems(factory: Any, what: str) -> list[str]:
    if not callable(factory):
        return [f"{what} must be a class or a callable factory, got {type(factory).__name__}"]
    return []


def _method_problems(cls: Any, methods: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    for method in methods:
        attr = getattr(cls, method, None)
        if not callable(attr):
            found.append(f"missing method {method}()")
        elif getattr(attr, "__isabstractmethod__", False):
            found.append(f"abstract method {method}() is not implemented")
    return found


# -- scorer -----------------------------------------------------------------
def scorer_contract(factory: Any) -> list[str]:
    from ..scoring.base import Scorer, ScoringKind

    if not _is_class(factory) or not issubclass(factory, Scorer):
        return [f"{factory!r} is not a Scorer subclass (mactf.scoring.base.Scorer)"]
    problems: list[str] = []
    if inspect.isabstract(factory):
        problems.append(f"abstract methods not implemented: {sorted(factory.__abstractmethods__)}")
    kind = getattr(factory, "kind", None)
    if not isinstance(kind, ScoringKind):
        problems.append("class attribute 'kind' must be a ScoringKind")
    elif kind is ScoringKind.PIPELINE:
        problems.append(
            "kind PIPELINE is reserved for the built-in pipeline; a custom stage is "
            "REWARD, PENALTY or WARNING (HIERARCHY is accepted but is root-only, "
            "as `ScoringPipeline._validate_chain` already enforces)"
        )
    order = getattr(factory, "order", None)
    if not isinstance(order, int) or isinstance(order, bool) or order < 0:
        problems.append("class attribute 'order' must be a non-negative int")
    if kind is ScoringKind.WARNING:
        from ..scoring.warning_scorer import WarningScorer

        if not issubclass(factory, WarningScorer):
            problems.append(
                "a WARNING stage subclasses WarningScorer: the pipeline, the engine and "
                "checkpoints reach warning state through its registries"
            )
    return problems


def scorer_instance_contract(instance: Any) -> list[str]:
    from ..scoring.base import Scorer

    return (
        [] if isinstance(instance, Scorer) else [f"built {type(instance).__name__}, not a Scorer"]
    )


# -- reward term -------------------------------------------------------------
def reward_term_contract(factory: Any) -> list[str]:
    if not _is_class(factory):
        return ["a reward term must be a class"]
    problems = _method_problems(factory, ("contribution",))
    rules = getattr(factory, "rule_ids", None)
    if not isinstance(rules, (frozenset, set, tuple, list)) or not rules:
        problems.append("class attribute 'rule_ids' must be a non-empty collection of rule ids")
    return problems


def reward_term_instance_contract(instance: Any) -> list[str]:
    return [] if callable(getattr(instance, "contribution", None)) else ["no contribution() method"]


# -- rl functions -------------------------------------------------------------
#: The entry method each family's protocol defines (verified against
#: ``rl/functions/{value,advantage,loss,rollup,support}.py``).
RL_ENTRY: dict[str, str] = {
    "value": "__call__",
    "advantage": "__call__",
    "loss": "compute",
    "rollup": "rollup",
    "return": "__call__",
    "reward": "__call__",
    "regularization": "__call__",
    "objective": "__call__",
    "gradient": "__call__",
    "metric": "__call__",
    "action": "__call__",
    "synthesis": "__call__",
}


def _defines(cls: Any, name: str) -> bool:
    """True if ``cls`` or a base other than ``object`` defines ``name`` in its own namespace.

    ``getattr(cls, "__call__")`` is *always* callable (it is ``type.__call__``), so asking the
    attribute proves nothing; the class dictionaries are what say whether the author wrote one.
    """
    return any(name in vars(klass) for klass in cls.__mro__ if klass is not object)


def synthesis_contract(factory: Any) -> list[str]:
    if not _is_class(factory):
        return [f"{factory!r} is not a class"]
    problems: list[str] = []
    if inspect.isabstract(factory):
        problems.append(f"abstract methods not implemented: {sorted(factory.__abstractmethods__)}")
    node_level = _defines(factory, "synthesize")
    tensor_level = _defines(factory, "__call__")
    if not node_level and not tensor_level:
        problems.append(
            "define synthesize(node, proposals, *, scores, weights, task) (node-aware) or "
            "__call__(proposals, *, weights=None) (tensor-level)"
        )
    if node_level:
        try:
            names = [
                name for name, parameter in inspect.signature(factory.synthesize).parameters.items()
                if name != "self" and parameter.kind in (
                    inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
        except (TypeError, ValueError):
            names = []
        if names[:2] != ["node", "proposals"]:
            problems.append(
                f"synthesize() must take (node, proposals, ...), found {names[:2] or 'no positional parameters'}"
            )
    rule = getattr(factory, "output_shape_rule", "same")
    if rule not in ("same", "free"):
        problems.append("class attribute 'output_shape_rule' must be 'same' or 'free'")
    return problems


def synthesis_instance_contract(instance: Any) -> list[str]:
    if callable(getattr(instance, "synthesize", None)) or callable(instance):
        return []
    return ["no synthesize() method and the instance is not callable"]


def rl_function_contract(family: str) -> Callable[[Any], list[str]]:
    if family == "synthesis":
        return synthesis_contract

    def contract(factory: Any) -> list[str]:
        if not _is_class(factory):
            return [f"{factory!r} is not a class"]
        entry = RL_ENTRY[family]
        problems = _method_problems(factory, (entry,))
        if family == "loss":
            required = getattr(factory, "required_fields", None)
            if not isinstance(required, (frozenset, set)):
                problems.append("a loss declares required_fields as a frozenset of batch keys")
        return problems

    return contract


def rl_function_legacy_contract(factory: Any) -> list[str]:
    """What ``register_function`` always checked: the wrapper is a class.

    The entry method is checked when the wrapper is *built* through the registry
    (``rl_function_instance_contract``), not when it is named, because the old function
    registry accepted any class and built it by keyword reflection.
    """
    return [] if _is_class(factory) else [f"{factory!r} is not a class"]


def rl_function_instance_contract(family: str) -> Callable[[Any], list[str]]:
    if family == "synthesis":
        return synthesis_instance_contract
    entry = RL_ENTRY[family]

    def contract(instance: Any) -> list[str]:
        return [] if callable(getattr(instance, entry, None)) else [f"no {entry}() method"]

    return contract


# -- optimizer / scheduler -----------------------------------------------------
def _backend_base() -> Any:
    """The shared base of the optimizer backends.

    Section 9.1 (phase P2) exports it as ``BackendBase``; until then the same class is the
    private ``_Backend`` of ``mactf.training.optimizer``. Either name is accepted so a
    plugin written against the final name registers today.
    """
    from ..training import optimizer

    return getattr(optimizer, "BackendBase", None) or getattr(optimizer, "_Backend")


def optimizer_contract(factory: Any) -> list[str]:
    """Two registrable forms, both classes: a ``BackendBase`` subclass (the shared base of
    ``TorchOptimizer``, ``ScipyOptimizer``, ``CallableOptimizer`` and ``ParamGroupRouter``)
    or a ``torch.optim.Optimizer`` subclass, which the builder wraps in a ``TorchOptimizer``.
    ``OptimizerBackend`` is a structural ``Protocol`` with property members, so
    ``issubclass`` cannot be used against it."""
    import torch

    if not _is_class(factory):
        return [f"{factory!r} is not a class"]
    if issubclass(factory, _backend_base()) or issubclass(factory, torch.optim.Optimizer):
        return []
    return [
        f"{factory!r} is neither a subclass of mactf.training.optimizer.BackendBase "
        "nor of torch.optim.Optimizer"
    ]


#: Names under which a torch optimizer receives its parameters. ``MACTFOptimizer`` binds
#: parameters after construction, so the resolver offers a placeholder under these names.
BIND_KEYS: frozenset[str] = frozenset({"params", "parameters"})


class _Unbound:
    """Placeholder offered to a torch optimizer's ``params`` argument (never reaches the class)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<unbound parameters>"


UNBOUND: Any = _Unbound()


def optimizer_builder(component: Any, values: dict[str, Any], injected: dict[str, Any]) -> Any:
    """Instantiate a registered optimizer factory into a backend."""
    import torch

    from ..training.optimizer import TorchOptimizer

    factory = component.factory
    if issubclass(factory, torch.optim.Optimizer):
        # A torch optimizer takes its parameters as the first argument, named ``params``
        # (or ``parameters``). The facade binds them later, so the resolver offers a
        # placeholder under both names only to let constructor reflection treat the
        # argument as framework-supplied; it never reaches the class here.
        injected = {k: v for k, v in injected.items() if k not in BIND_KEYS}
        values = dict(values)
        lr = values.pop("lr", None)
        weight_decay = float(values.pop("weight_decay", 0.0))
        return TorchOptimizer(factory, lr=lr, weight_decay=weight_decay, **values, **injected)
    return factory(**values, **{k: v for k, v in injected.items() if k not in BIND_KEYS})


def optimizer_instance_contract(instance: Any) -> list[str]:
    return (
        []
        if isinstance(instance, _backend_base())
        else [f"built {type(instance).__name__}"]
    )


def scheduler_contract(factory: Any) -> list[str]:
    problems = _callable_problems(factory, "a scheduler factory")
    if problems:
        return problems
    keys = reflect(factory)
    if "optimizer" not in keys.keys and not keys.var_keyword:
        problems.append("a scheduler factory takes the torch optimizer as its 'optimizer' argument")
    return problems


def scheduler_instance_contract(instance: Any) -> list[str]:
    return [] if callable(getattr(instance, "step", None)) else ["no step() method"]


# -- strategies, weights, aggregator, policy, peer factory ----------------------
def peer_strategy_contract(factory: Any) -> list[str]:
    from ..coopetition.strategy import PeerStrategy

    if not _is_class(factory) or not issubclass(factory, PeerStrategy):
        return [f"{factory!r} is not a subclass of mactf.coopetition.strategy.PeerStrategy"]
    problems: list[str] = []
    if inspect.isabstract(factory):
        problems.append(f"abstract methods not implemented: {sorted(factory.__abstractmethods__)}")
    if not isinstance(getattr(factory, "name", None), str):
        problems.append("class attribute 'name' must be a str")
    return problems


def peer_strategy_instance_contract(instance: Any) -> list[str]:
    from ..coopetition.strategy import PeerStrategy

    return [] if isinstance(instance, PeerStrategy) else [f"built {type(instance).__name__}"]


def weight_update_contract(factory: Any) -> list[str]:
    if not _is_class(factory):
        return [f"{factory!r} is not a class"]
    problems = _method_problems(factory, ("apply", "describe", "from_spec"))
    if not isinstance(getattr(factory, "name", None), str):
        problems.append("class attribute 'name' must be a str")
    return problems


def weight_update_instance_contract(instance: Any) -> list[str]:
    return [] if callable(getattr(instance, "apply", None)) else ["no apply() method"]


def aggregator_contract(factory: Any) -> list[str]:
    from ..coopetition.aggregation import Aggregator

    if not _is_class(factory) or not issubclass(factory, Aggregator):
        return [f"{factory!r} is not a subclass of mactf.coopetition.aggregation.Aggregator"]
    return []


def aggregator_instance_contract(instance: Any) -> list[str]:
    from ..coopetition.aggregation import Aggregator

    return [] if isinstance(instance, Aggregator) else [f"built {type(instance).__name__}"]


def trainer_policy_contract(factory: Any) -> list[str]:
    if not _is_class(factory):
        return ["a trainer policy must be a class"]
    return _method_problems(factory, ("train_step",))


def trainer_policy_instance_contract(instance: Any) -> list[str]:
    return [] if callable(getattr(instance, "train_step", None)) else ["no train_step() method"]


def peer_factory_contract(factory: Any) -> list[str]:
    problems = _callable_problems(factory, "a peer factory")
    if problems:
        return problems
    keys = reflect(factory)
    if "spec" not in keys.keys and not keys.var_keyword:
        problems.append("a peer factory takes a 'spec' (PeerSpec) argument")
    return problems


def peer_factory_instance_contract(instance: Any) -> list[str]:
    import torch.nn as nn

    return (
        []
        if isinstance(instance, nn.Module)
        else [f"a peer factory returns an nn.Module, not {type(instance).__name__}"]
    )


#: The module that defines each legacy table and registers its built-ins.
_BUILTINS_MODULE: dict[str, str] = {
    "scorer": "mactf.scoring.registry",
    "reward_term": "mactf.scoring.reward_terms",
    "peer_strategy": "mactf.coopetition.strategy",
    "weight_update": "mactf.coopetition.weights",
    "aggregator": "mactf.coopetition.aggregation",
    "optimizer": "mactf.training.optimizer",
    "scheduler": "mactf.training.schedulers",
    "peer_factory": "mactf.training.peer_factories",
}


def _spec(
    kind: ComponentKind,
    family: str | None,
    contract: Callable[[Any], list[str]],
    instance_contract: Callable[[Any], list[str]],
    mutability: Mutability,
    legacy: str | None,
    description: str,
    builder: Callable[[Any, dict[str, Any], dict[str, Any]], Any] | None = None,
    legacy_contract: Callable[[Any], list[str]] | None = None,
) -> KindSpec:
    kind_id = kind.kind_id(family)
    return KindSpec(
        kind_id=kind_id,
        kind=kind,
        family=family,
        slot=kind.value if family is None else f"rl_function.{family}",
        default_mutability=mutability,
        contract=contract,
        instance_contract=instance_contract,
        legacy_table=legacy,
        description=description,
        builder=builder,
        legacy_contract=legacy_contract,
        builtins_module=(
            "mactf.rl.functions.registry" if family else _BUILTINS_MODULE.get(kind_id)
        ),
    )


_STEP_FAMILIES = frozenset({"loss", "regularization", "gradient", "metric"})

#: Default mutability per kind, as section 3.2 tabulates it.
KIND_DEFAULTS: dict[str, Mutability] = {}


def install() -> None:
    """Populate ``KIND_SPECS``. Called once, from ``mactf/components/__init__.py``."""
    table = [
        _spec(ComponentKind.SCORER, None, scorer_contract, scorer_instance_contract,
              Mutability.EPISODE, "mactf.scoring.registry.SCORER_TYPES", "A scoring stage."),
        _spec(ComponentKind.REWARD_TERM, None, reward_term_contract, reward_term_instance_contract,
              Mutability.EPISODE, None, "A per-rule reward reshaping term."),
        _spec(ComponentKind.OPTIMIZER, None, optimizer_contract, optimizer_instance_contract,
              Mutability.EPISODE, None, "An optimizer backend.", builder=optimizer_builder),
        _spec(ComponentKind.SCHEDULER, None, scheduler_contract, scheduler_instance_contract,
              Mutability.EPISODE, None, "A learning-rate scheduler factory."),
        _spec(ComponentKind.PEER_STRATEGY, None, peer_strategy_contract,
              peer_strategy_instance_contract, Mutability.EPISODE,
              "mactf.coopetition.strategy.PEER_STRATEGIES", "A coordination policy."),
        _spec(ComponentKind.WEIGHT_UPDATE, None, weight_update_contract,
              weight_update_instance_contract, Mutability.EPISODE,
              "mactf.coopetition.weights.WEIGHT_STRATEGIES", "A weight-update rule."),
        _spec(ComponentKind.AGGREGATOR, None, aggregator_contract, aggregator_instance_contract,
              Mutability.STATIC, None, "A parameter aggregator."),
        _spec(ComponentKind.TRAINER_POLICY, None, trainer_policy_contract,
              trainer_policy_instance_contract, Mutability.STATIC, None, "A training step policy."),
        _spec(ComponentKind.PEER_FACTORY, None, peer_factory_contract,
              peer_factory_instance_contract, Mutability.EPISODE, None, "A peer module factory."),
    ]
    for family in RL_FAMILIES:
        table.append(
            _spec(
                ComponentKind.RL_FUNCTION,
                family,
                rl_function_contract(family),
                rl_function_instance_contract(family),
                Mutability.STEP if family in _STEP_FAMILIES else Mutability.EPISODE,
                f"mactf.rl.functions.registry.FUNCTION_TYPES[{family!r}]",
                f"An RL {family} function.",
                legacy_contract=rl_function_legacy_contract,
            )
        )
    for entry in table:
        KIND_SPECS[entry.kind_id] = entry
        KIND_DEFAULTS[entry.kind_id] = entry.default_mutability
    missing = set(all_kind_ids()) - set(KIND_SPECS)
    if missing:  # pragma: no cover - guards a future kind added without a contract
        raise RuntimeError(f"kinds without a contract: {sorted(missing)}")
