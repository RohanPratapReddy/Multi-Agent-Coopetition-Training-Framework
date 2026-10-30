"""Kinds, origins and mutability classes (document 17 section 3.2).

Pure data: this module imports nothing but ``mactf.exceptions`` and the standard library,
so ``import mactf.components`` stays light (no torch, no scoring, no trainer).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Callable

from ..exceptions import ValidationError

#: The twelve RL function families, in the order ``CAPABILITY_ORDER`` fixes
#: (``mactf/rl/functions/registry.py``).
RL_FAMILIES: tuple[str, ...] = (
    "value",
    "advantage",
    "loss",
    "rollup",
    "return",
    "reward",
    "regularization",
    "objective",
    "gradient",
    "metric",
    "action",
    "synthesis",
)


class Origin(str, enum.Enum):
    """Where a component came from. The rank decides who may replace whom."""

    BUILTIN = "builtin"
    PLUGIN = "plugin"
    PROJECT = "project"
    RUNTIME = "runtime"

    @property
    def rank(self) -> int:
        return _ORIGIN_RANK[self]


_ORIGIN_RANK: dict[Origin, int] = {
    Origin.BUILTIN: 0,
    Origin.PLUGIN: 1,
    Origin.PROJECT: 2,
    Origin.RUNTIME: 3,
}


class Mutability(str, enum.Enum):
    """How often the slot holding this component may be swapped."""

    STATIC = "static"  # fixed for the life of a run
    EPISODE = "episode"  # may change at an episode boundary
    STEP = "step"  # may change between optimizer steps

    @property
    def rank(self) -> int:
        return _MUTABILITY_RANK[self]


_MUTABILITY_RANK: dict[Mutability, int] = {
    Mutability.STATIC: 0,
    Mutability.EPISODE: 1,
    Mutability.STEP: 2,
}


class Boundary(str, enum.Enum):
    """The moment at which a swap is attempted. Same ordering as ``Mutability``."""

    RUN_START = "run_start"
    EPISODE = "episode"
    STEP = "step"

    @property
    def rank(self) -> int:
        return {"run_start": 0, "episode": 1, "step": 2}[self.value]


class ComponentKind(str, enum.Enum):
    SCORER = "scorer"
    REWARD_TERM = "reward_term"
    RL_FUNCTION = "rl_function"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"
    PEER_STRATEGY = "peer_strategy"
    WEIGHT_UPDATE = "weight_update"
    AGGREGATOR = "aggregator"
    TRAINER_POLICY = "trainer_policy"
    PEER_FACTORY = "peer_factory"

    def kind_id(self, family: str | None = None) -> str:
        """``"scorer"``, ``"rl_function.loss"``, ... The registry key."""
        if self is ComponentKind.RL_FUNCTION:
            if family is None:
                raise ValidationError(
                    "rl_function needs a family, one of: " + ", ".join(RL_FAMILIES)
                )
            if family not in RL_FAMILIES:
                raise ValidationError(
                    f"unknown function family {family!r}; accepted families: "
                    + ", ".join(sorted(RL_FAMILIES))
                )
            return f"rl_function.{family}"
        if family is not None:
            raise ValidationError(f"kind {self.value!r} takes no family; got {family!r}")
        return self.value


def parse_kind_id(text: str) -> tuple[ComponentKind, str | None]:
    """Inverse of :meth:`ComponentKind.kind_id`."""
    head, _, family = str(text).partition(".")
    try:
        kind = ComponentKind(head)
    except ValueError:
        raise ValidationError(
            f"unknown component kind {text!r}; kinds: " + ", ".join(sorted(all_kind_ids()))
        ) from None
    if kind is ComponentKind.RL_FUNCTION:
        kind.kind_id(family or None)  # raises ValidationError for a missing or unknown family
        return kind, family
    if family:
        raise ValidationError(f"kind {head!r} takes no family; got {text!r}")
    return kind, None


def resolve_kind(kind: "ComponentKind | str", family: str | None = None) -> str:
    """The registry key for ``kind`` given as an enum, a kind name or a full kind id."""
    if isinstance(kind, ComponentKind):
        return kind.kind_id(family)
    if family is None and "." in str(kind):
        enum_kind, fam = parse_kind_id(str(kind))
        return enum_kind.kind_id(fam)
    try:
        return ComponentKind(str(kind)).kind_id(family)
    except ValueError:
        raise ValidationError(
            f"unknown component kind {kind!r}; kinds: " + ", ".join(sorted(all_kind_ids()))
        ) from None


@dataclass(frozen=True)
class KindSpec:
    """Static facts about one kind. ``contract`` and ``instance_contract`` return
    a list of problems (empty = fine); they live in :mod:`mactf.components.contracts`.
    """

    kind_id: str
    kind: ComponentKind
    family: str | None
    #: The root name used by ``COMPONENTS.LOCKED``: ``scorer``, ``weight_update``, ...
    slot: str
    default_mutability: Mutability
    contract: Callable[[Any], list[str]]
    instance_contract: Callable[[Any], list[str]]
    #: Dotted path of the legacy dict this kind mirrors, informational only.
    legacy_table: str | None = None
    description: str = ""
    #: ``builder(component, values, injected) -> instance``. ``None`` means "call the
    #: factory with ``**values, **injected``". A kind whose registered factories are not
    #: all directly callable (an optimizer may be a ``torch.optim.Optimizer`` subclass that
    #: needs parameters first) supplies one. Resolved in ``mactf.components.contracts``.
    builder: Callable[[Any, dict[str, Any], dict[str, Any]], Any] | None = None
    #: The registration-time check applied to a *legacy* write (a direct write to the old
    #: dict or the positional ``register_scorer`` / ``register_function`` form). ``None``
    #: means the same as ``contract``. The RL families use the check the old
    #: ``register_function`` always applied (a class), because its callers registered
    #: wrappers whose entry method is checked when they are built, not when they are named.
    legacy_contract: Callable[[Any], list[str]] | None = None
    #: The module that defines the legacy table and registers the built-ins of this kind.
    #: The registry imports it before the first write so a plugin cannot register a name
    #: before the built-ins exist (a plugin loaded early would otherwise collide with them).
    builtins_module: str | None = None


# Filled by ``mactf.components.contracts`` at import; see section 3.7.
KIND_SPECS: dict[str, KindSpec] = {}


def all_kind_ids() -> tuple[str, ...]:
    ids = [k.value for k in ComponentKind if k is not ComponentKind.RL_FUNCTION]
    ids.extend(f"rl_function.{family}" for family in RL_FAMILIES)
    return tuple(ids)
