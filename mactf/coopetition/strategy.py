"""Peer coordination strategies (document 08 section 6.1).

A strategy is the coordination *policy*: it decides whether peers mix parameters at
all, in what order, and which nodes are excluded. It never does arithmetic —
that is :mod:`mactf.coopetition.weights`, which delegates to the existing
:class:`~mactf.coopetition.aggregation.Aggregator`.

The five built-ins cover the documented matrix:

=======================  ================  ===================================
Strategy                 Aggregates        Effect at epoch end
=======================  ================  ===================================
``IndependentStrategy``  no                nothing; peers diverge
``CooperativeStrategy``  no                shares signals only, no mixing
``KappaSoftmaxStrategy`` yes               configured weight update
``FrozenPeerStrategy``   yes               one node excluded, still scored
``HierarchicalStrategy`` yes               level by level, deepest first
=======================  ================  ===================================

``HierarchicalStrategy`` is the default when a hierarchy is present and the only
strategy that walks more than one aggregation group.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Mapping, Sequence

import torch

from ..components.legacy import LegacyTable
from ..exceptions import ContractError, ValidationError
from .tree import PeerNode, PeerTree, aggregation_groups

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..training.coordination import CoordinationContext
    from ..training.protocol import EpochResult, StepResult
    from .aggregation import AggregationReport, Aggregator
    from .weights import WeightUpdateStrategy

__all__ = [
    "BUILTIN_STRATEGY_TYPES",
    "CooperativeStrategy",
    "FrozenPeerStrategy",
    "GroupRef",
    "HierarchicalStrategy",
    "IndependentStrategy",
    "KappaSoftmaxStrategy",
    "PEER_STRATEGIES",
    "PeerStrategy",
    "StrategyTraits",
    "TreeView",
    "build_peer_strategy",
    "coordinates_via_context",
    "defines_nothing",
    "legacy_dispatch",
    "traits_of",
    "tree_view",
]

WALK_MODES = ("root", "tree")
ROOT_LABEL = "root"

#: What a strategy may read from a trainer. Duck-typed so a custom trainer that
#: only implements ``PeerStrategy``'s surface still works.
_TRAINER_ATTRS = ("tree", "peers", "weight_update", "aggregator", "kappa")


@dataclass(frozen=True)
class StrategyTraits:
    """What a strategy promises. Read by ``StrategyPlan`` and ``StrategyGuard``."""

    #: The strategy mixes parameters (``should_aggregate`` for a legacy strategy).
    aggregates: bool = False
    #: The strategy adds entries to the freeze set; it must then release them in ``detach``.
    freezes: bool = False
    #: How far the strategy reaches when it is the *default*: ``"root"`` serves the root group
    #: only (the five built-ins' behaviour today, except ``hierarchical``); ``"tree"`` serves
    #: every group that no override claims. An *override* always serves exactly the groups
    #: its glob matches, whatever ``walk`` says.
    walk: str = "root"
    #: The root group's weights are computed before any child group is written, as
    #: ``_coordinate`` does for ``hierarchical``. This fixes the order of ``trainer.reports``
    #: and of the controllers' updates.
    precompute_root: bool = False
    #: One instance per group instead of one per slot. State then never crosses groups.
    per_group: bool = False

    def __post_init__(self) -> None:
        if self.walk not in WALK_MODES:
            raise ValidationError(
                f"StrategyTraits.walk must be one of {list(WALK_MODES)}, got {self.walk!r}"
            )


@dataclass(frozen=True)
class GroupRef:
    """One aggregation group. ``label`` is ``"root"`` or the parent node's path."""

    label: str
    parent: str | None
    members: tuple[str, ...]
    level: int

    @property
    def is_root(self) -> bool:
        return self.parent is None


@dataclass(frozen=True)
class TreeView:
    """The tree as a strategy's ``validate`` sees it: groups in ``aggregation_groups`` order."""

    groups: tuple[GroupRef, ...]
    paths: tuple[str, ...]
    hierarchy: bool

    def group(self, label: str) -> GroupRef:
        for item in self.groups:
            if item.label == label:
                return item
        raise ValidationError(
            f"no aggregation group {label!r}; groups are {[g.label for g in self.groups]}"
        )

    def labels(self) -> tuple[str, ...]:
        return tuple(item.label for item in self.groups)


def tree_view(tree: PeerTree, *, hierarchy: bool) -> TreeView:
    groups = tuple(
        GroupRef(
            label=ROOT_LABEL if parent is None else parent,
            parent=parent,
            members=tuple(members),
            level=0 if parent is None else int(tree[parent].depth),
        )
        for parent, members in aggregation_groups(tree)
    )
    return TreeView(
        groups=groups,
        paths=tuple(node.path for node in tree.walk("root-first")),
        hierarchy=bool(hierarchy),
    )


class PeerStrategy(ABC):
    """Coordination policy for a set of peers.

    A strategy is driven through ``coordinate(ctx)`` (the context API of document 17
    section 11.6) when it defines that method at least as deep in its class hierarchy as
    ``on_epoch_end``; otherwise it is *legacy-dispatched* with exactly the calls the trainer
    always made. ``on_epoch_end`` is no longer abstract: a new-style strategy implements
    ``coordinate`` and nothing else.
    """

    name: ClassVar[str]
    traits: ClassVar[StrategyTraits] = StrategyTraits()

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Called at the top of every epoch. The default is a no-op."""

    def on_peer_step(self, trainer: Any, node: PeerNode, result: "StepResult") -> None:
        """Called after each node's training step. The default is a no-op."""

    # -- new -------------------------------------------------------------
    def coordinate(self, ctx: "CoordinationContext") -> None:
        """New-style entry point. Called once per group the strategy serves, deepest first.

        ``ctx.groups`` holds the group(s) of *this* call. The default is a no-op.
        """

    def attach(self, host: Any) -> None:
        """Called once when the strategy becomes live (build, swap-in, restore). No-op."""

    def detach(self, host: Any) -> None:
        """Called once when the strategy stops being live (swap-out, removal). A strategy
        that froze peers releases them here. No-op."""

    def validate(self, view: TreeView) -> Sequence[str]:
        """Problems with serving ``view`` (a missing frozen path, a hierarchy it needs and the
        tree lacks). Reported as ``E074`` at build and after every tree mutation."""
        return ()

    # -- legacy entry point (no longer abstract) ----------------------------
    def on_epoch_end(
        self, trainer: Any, epoch: int, report: "AggregationReport | None"
    ) -> None:
        """Called after every node has stepped, before logging. The default is a no-op."""

    def should_aggregate(self) -> bool:
        """Whether this strategy mixes parameters at all."""
        return self.traits.aggregates

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "class": type(self).__name__}

    def state_dict(self) -> dict[str, Any]:
        return {"class": type(self).__name__, "name": self.name}

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if payload.get("class") not in (None, type(self).__name__):
            raise ValidationError(
                f"{type(self).__name__} cannot load state written by "
                f"{payload.get('class')!r}"
            )

    @classmethod
    def from_spec(cls, spec: "PeerStrategy | str | Mapping[str, Any] | None") -> "PeerStrategy":
        return build_peer_strategy(spec)


def traits_of(strategy: Any) -> StrategyTraits:
    """The declared traits; for an object that declares none, derived from ``should_aggregate``."""
    found = getattr(strategy, "traits", None)
    if isinstance(found, StrategyTraits):
        return found
    probe = getattr(strategy, "should_aggregate", None)
    aggregates = False
    if callable(probe):
        try:
            aggregates = bool(probe())
        except Exception:  # noqa: BLE001 - a structural strategy that cannot answer is "no"
            aggregates = False
    return StrategyTraits(aggregates=aggregates)


def _owner(cls: type, attribute: str) -> type:
    for klass in cls.__mro__:
        if attribute in vars(klass):
            return klass
    return PeerStrategy  # pragma: no cover - PeerStrategy defines both


def coordinates_via_context(strategy: Any) -> bool:
    """Whether ``strategy`` is driven through ``coordinate(ctx)`` (else: legacy dispatch)."""
    if not isinstance(strategy, PeerStrategy):
        return False
    cls = type(strategy)
    by_context, by_epoch_end = _owner(cls, "coordinate"), _owner(cls, "on_epoch_end")
    if by_context is PeerStrategy:
        return False
    if by_epoch_end is PeerStrategy:
        return True
    mro = cls.__mro__
    return mro.index(by_context) <= mro.index(by_epoch_end)


def defines_nothing(strategy: Any) -> bool:
    """A ``PeerStrategy`` subclass that defines neither entry point (it would do nothing)."""
    if not isinstance(strategy, PeerStrategy):
        return False
    cls = type(strategy)
    return _owner(cls, "coordinate") is PeerStrategy and _owner(cls, "on_epoch_end") is PeerStrategy


def legacy_dispatch(trainer: Any, epoch: int, strategy: Any) -> None:
    """The calls ``MACTFTrainer._coordinate`` always made, unchanged.

    Used for a strategy that does not define ``coordinate``: a structural object, a
    subclass that overrides only ``on_epoch_end``, or a pre-existing custom class.
    """
    if isinstance(strategy, HierarchicalStrategy):
        report = trainer.update_weights()
        strategy.on_epoch_end(trainer, epoch, report)
    elif isinstance(strategy, (KappaSoftmaxStrategy, FrozenPeerStrategy)):
        strategy.on_epoch_end(trainer, epoch, None)
    elif strategy.should_aggregate():
        report = trainer.update_weights()
        trainer.aggregate(report)
        strategy.on_epoch_end(trainer, epoch, report)
    else:
        strategy.on_epoch_end(trainer, epoch, None)


# ---------------------------------------------------------------------------
# Non-aggregating strategies
# ---------------------------------------------------------------------------
class IndependentStrategy(PeerStrategy):
    """Peers never share parameters. The pre-coopetition behaviour."""

    name = "independent"
    traits = StrategyTraits(aggregates=False, walk="root")

    def coordinate(self, ctx: "CoordinationContext") -> None:
        for group in ctx.groups:
            ctx.skip(group, "peers train independently and are never mixed")

    def on_epoch_end(
        self, trainer: Any, epoch: int, report: "AggregationReport | None"
    ) -> None:
        # Deliberately empty: divergence is the point.
        return None

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload["aggregates"] = False
        payload["reason"] = "peers train independently and are never mixed"
        return payload


class CooperativeStrategy(PeerStrategy):
    """Shares episode *signals*, not parameters.

    The signals are what ``PhaseContext.signal_by_peer`` computes: each peer's mean
    score across the epoch. Writing them onto ``trainer.signal_by_peer`` is the
    non-parameter equivalent of aggregation, and a peer's own signal never
    includes a value it did not earn, so no credit transfer happens.
    """

    name = "cooperative"
    traits = StrategyTraits(aggregates=False, walk="root")

    def coordinate(self, ctx: "CoordinationContext") -> None:
        for group in ctx.groups:
            signals = ctx.signals
            if not group.is_root:
                signals = {path: signals[path] for path in group.members if path in signals}
            ctx.record_signal_sharing(signals)

    def on_epoch_end(
        self, trainer: Any, epoch: int, report: "AggregationReport | None"
    ) -> None:
        signals = getattr(trainer, "signal_by_peer", None)
        if signals is None:
            return None
        # ``signal_by_peer`` is a mapping the trainer exposes; the sharing step is
        # recording that every peer saw the whole cohort's signals this epoch.
        setter = getattr(trainer, "record_signal_sharing", None)
        if callable(setter):
            setter(dict(signals), epoch=epoch)
        return None

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload["aggregates"] = False
        payload["reason"] = "signals are shared; parameters are not"
        return payload


# ---------------------------------------------------------------------------
# Aggregating strategies
# ---------------------------------------------------------------------------
class KappaSoftmaxStrategy(PeerStrategy):
    """Delegates to the trainer's configured :class:`WeightUpdateStrategy`."""

    name = "kappa_softmax"
    traits = StrategyTraits(aggregates=True, walk="root")

    def coordinate(self, ctx: "CoordinationContext") -> None:
        for group in ctx.groups:
            if group.is_root:
                ctx.aggregate_root(ctx.update_root())
            else:
                ctx.aggregate_group(group)

    def on_epoch_end(
        self, trainer: Any, epoch: int, report: "AggregationReport | None"
    ) -> None:
        update = getattr(trainer, "update_weights", None)
        if not callable(update):
            raise ContractError(
                "KappaSoftmaxStrategy needs a trainer exposing update_weights()"
            )
        produced = update()
        aggregate = getattr(trainer, "aggregate", None)
        if callable(aggregate):
            aggregate(produced)

    def should_aggregate(self) -> bool:
        return True

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload["aggregates"] = True
        payload["delegates_to"] = "trainer.update_weights + trainer.aggregate"
        return payload


class FrozenPeerStrategy(PeerStrategy):
    """One node is frozen and excluded from weight updates; it still scores.

    Exclusion is done through ``eligible=False``, not by removing the node, so the
    frozen peer's score still reaches the report and the checkpoint. This is the
    peer a user wants as a fixed baseline.
    """

    name = "frozen_peer"
    traits = StrategyTraits(aggregates=True, freezes=True, walk="root")
    REASON = "strategy:frozen_peer"

    def __init__(
        self,
        frozen: str | Sequence[str] = (),
        *,
        still_scores: bool = True,
        path: str | Sequence[str] | None = None,     # legacy alias of ``frozen``
    ) -> None:
        if path is not None and frozen:
            raise ValidationError("FrozenPeerStrategy takes 'frozen' or its alias 'path', not both")
        chosen = path if path is not None else frozen
        if isinstance(chosen, str):
            chosen = (chosen,)
        self.frozen: tuple[str, ...] = tuple(str(item) for item in chosen)
        self.still_scores = bool(still_scores)
        self._marked: list[str] = []

    # The freeze set is the one place that records the freeze, and it mirrors the status
    # onto the tree (document 17 section 11.4).
    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        tree = getattr(trainer, "tree", None)
        if tree is None or not self.frozen:
            return
        freeze = getattr(trainer, "freeze", None)
        if freeze is not None:
            freeze.clear_reason(self.REASON)          # the set equals self.frozen, always
        self._marked = []
        for path in self.frozen:
            node = tree.get(path) if isinstance(tree, PeerTree) else None
            if node is None:
                raise ContractError(
                    f"FrozenPeerStrategy names {path!r}, which is not a node in this tree"
                )
            if freeze is not None:
                freeze.add(path, self.REASON)
            else:                                      # a duck-typed trainer without a freeze set
                tree.mark(path, "frozen")
            self._marked.append(path)

    def detach(self, host: Any) -> None:
        freeze = getattr(host, "freeze", None)
        if freeze is not None:
            freeze.clear_reason(self.REASON)

    def validate(self, view: TreeView) -> list[str]:
        known = set(view.paths)
        return [f"frozen path {path!r} is not a node in the tree" for path in self.frozen
                if path not in known]

    def coordinate(self, ctx: "CoordinationContext") -> None:
        for group in ctx.groups:
            eligible = ctx.eligible_for(group.members) if self.frozen else None
            if eligible is not None and not any(eligible.values()):
                ctx.skip(group, "every member is frozen")
                continue
            if group.is_root:
                ctx.aggregate_root(ctx.update_root(eligible=eligible))
            else:
                ctx.aggregate_group(group, eligible=eligible)

    def eligible(self, names: Iterable[str]) -> dict[str, bool]:
        """The eligibility map the weight update receives."""
        frozen = set(self.frozen)
        return {name: name not in frozen for name in names}

    def on_epoch_end(
        self, trainer: Any, epoch: int, report: "AggregationReport | None"
    ) -> None:
        update = getattr(trainer, "update_weights", None)
        if not callable(update):
            raise ContractError(
                "FrozenPeerStrategy needs a trainer exposing update_weights()"
            )
        produced = update(eligible=None if not self.frozen else self._eligibility(trainer))
        aggregate = getattr(trainer, "aggregate", None)
        if callable(aggregate):
            aggregate(produced)

    def _eligibility(self, trainer: Any) -> Mapping[str, bool]:
        names = getattr(trainer, "peer_names", None)
        if names is None:
            peers = getattr(trainer, "peers", None) or {}
            names = list(peers)
        return self.eligible(list(names))

    def should_aggregate(self) -> bool:
        return True

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update(
            {
                "aggregates": True,
                "frozen": list(self.frozen),
                "still_scores": self.still_scores,
            }
        )
        return payload

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload.update({"frozen": list(self.frozen), "still_scores": self.still_scores})
        return payload

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        super().load_state_dict(payload)
        raw = payload.get("frozen")
        if isinstance(raw, (list, tuple)):
            self.frozen = tuple(str(path) for path in raw)
        if "still_scores" in payload:
            self.still_scores = bool(payload["still_scores"])

    @classmethod
    def from_spec(cls, spec: Any) -> "PeerStrategy":
        if isinstance(spec, Mapping):
            frozen = spec.get("frozen", spec.get("path", ()))
            return cls(frozen, still_scores=bool(spec.get("still_scores", True)))
        return cls()


class HierarchicalStrategy(PeerStrategy):
    """Aggregates level by level, deepest group first, then the root group.

    Without this, a parent would be averaged before its children, and the parent's
    parameter vector would then be overwritten by a blend of its own stale values.
    The order is ``reversed(aggregation_groups(tree))``: deepest parents first, the
    root group last, which is what makes a parent aggregate already-aggregated
    children (document 10 section 8.1).

    A level whose peers are all void is skipped rather than averaged to zero. That
    rule is source-agnostic: voidness is a compliance concept, not a numeric one.
    """

    name = "hierarchical"
    traits = StrategyTraits(aggregates=True, walk="tree", precompute_root=True)

    def __init__(
        self,
        *,
        update_reference: bool = True,
        skip_void_levels: bool = True,
    ) -> None:
        self.update_reference = bool(update_reference)
        self.skip_void_levels = bool(skip_void_levels)
        self.last_reports: dict[str | None, "AggregationReport"] = {}

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        self.last_reports = {}

    def coordinate(self, ctx: "CoordinationContext") -> None:
        for group in ctx.groups:
            if group.is_root:
                report = ctx.update_root()
                ctx.aggregate_root(report, update_reference=self.update_reference)
                self.last_reports[None] = report
                continue
            if not group.members:
                continue
            if self.skip_void_levels and ctx.peers_void(group.members):
                ctx.skip(group, "every member was voided this epoch")
                continue
            self.last_reports[group.parent] = ctx.aggregate_group(group)

    def on_epoch_end(
        self, trainer: Any, epoch: int, report: "AggregationReport | None"
    ) -> None:
        tree = getattr(trainer, "tree", None)
        if tree is None:
            raise ContractError(
                "HierarchicalStrategy requires a trainer built with a PeerTree"
            )
        # The root group's report is the one the trainer already computed, so it is
        # reused rather than recomputed: recomputing would advance AdaptiveKappa a
        # second time and change every downstream number.
        deepest_first = list(reversed(aggregation_groups(tree)))
        self.last_reports = {}
        aggregate_group = getattr(trainer, "aggregate_group", None)
        for index, (parent, members) in enumerate(deepest_first):
            if parent is None:
                # Root group last.
                produced = report
                if produced is None and callable(getattr(trainer, "update_weights", None)):
                    produced = trainer.update_weights()
                if produced is not None and callable(getattr(trainer, "aggregate", None)):
                    trainer.aggregate(produced)
                if produced is not None:
                    self.last_reports[None] = produced
                continue
            if not members:
                continue
            if self.skip_void_levels and self._all_void(trainer, members):
                continue
            if callable(aggregate_group):
                produced = aggregate_group(parent, members)
                if produced is not None:
                    self.last_reports[parent] = produced

    @staticmethod
    def _all_void(trainer: Any, members: Sequence[str]) -> bool:
        checker = getattr(trainer, "peers_void", None)
        if callable(checker):
            return bool(checker(list(members)))
        return False

    def should_aggregate(self) -> bool:
        return True

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update(
            {
                "aggregates": True,
                "order": "deepest-first, root group last",
                "skip_void_levels": self.skip_void_levels,
                "update_reference": self.update_reference,
            }
        )
        return payload

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload.update(
            {
                "update_reference": self.update_reference,
                "skip_void_levels": self.skip_void_levels,
            }
        )
        return payload

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        super().load_state_dict(payload)
        if "update_reference" in payload:
            self.update_reference = bool(payload["update_reference"])
        if "skip_void_levels" in payload:
            self.skip_void_levels = bool(payload["skip_void_levels"])

    @classmethod
    def from_spec(cls, spec: Any) -> "PeerStrategy":
        payload = dict(spec) if isinstance(spec, Mapping) else {}
        return cls(
            update_reference=bool(payload.get("update_reference", True)),
            skip_void_levels=bool(payload.get("skip_void_levels", True)),
        )


#: Every built-in, addressable by name; a view of ``HUB.registry("peer_strategy")``.
PEER_STRATEGIES: dict[str, type] = LegacyTable(
    "peer_strategy",
    {
        "independent": IndependentStrategy,
        "cooperative": CooperativeStrategy,
        "kappa_softmax": KappaSoftmaxStrategy,
        "frozen_peer": FrozenPeerStrategy,
        "hierarchical": HierarchicalStrategy,
    },
)

#: The framework's own classes: what ``StrategyGuard`` leaves alone under ``SCOPE: custom``.
BUILTIN_STRATEGY_TYPES: frozenset[type] = frozenset(dict.values(PEER_STRATEGIES))


def build_peer_strategy(
    spec: "PeerStrategy | str | Mapping[str, Any] | None",
    *,
    hierarchy: bool = False,
    resolver: Any | None = None,
    trusted: bool = True,
) -> PeerStrategy:
    """Resolve a strategy from an instance, a name, a spec, or ``None``.

    ``None`` is ``hierarchical`` with a hierarchy and ``independent`` without, as before. A name
    or mapping goes through the component registry, so it may name a built-in, a project
    component or a plugin; an unknown name or unaccepted key raises here, with the accepted
    list (``ComponentMissing`` / ``ComponentParamError`` are ``ValidationError``, 17.1).
    """
    from ..components.resolver import ComponentResolver, ComponentSpec

    if spec is None:
        return HierarchicalStrategy() if hierarchy else IndependentStrategy()
    if isinstance(spec, PeerStrategy):
        return spec
    if isinstance(spec, (str, Mapping, ComponentSpec)):
        raw: Any = spec
        if isinstance(spec, str):
            raw = spec.strip().lower()
        elif isinstance(spec, Mapping):
            payload = dict(spec)
            written = payload.pop("name", None)
            if "type" not in payload and "import" not in payload and written is not None:
                payload["type"] = written
            if "type" in payload:
                payload["type"] = str(payload["type"]).strip().lower()
            raw = payload
        parsed = ComponentSpec.parse("peer_strategy", raw)
        return (resolver or ComponentResolver()).resolve(parsed, trusted=trusted).instance
    if hasattr(spec, "on_epoch_end") and hasattr(spec, "should_aggregate"):
        return spec  # structural custom strategy: legacy-dispatched
    raise ValidationError(
        f"peer_strategy must be a PeerStrategy, a name, or a mapping; got "
        f"{type(spec).__name__}"
    )


def group_layouts(
    peers: Mapping[str, Any],
    members: Sequence[str],
    *,
    require_same_layout: Any = None,
) -> None:
    """Apply ``require_same_layout`` per aggregation group, not per run.

    Sibling layouts must match because they are averaged. Different branches may
    differ, because they are never averaged together (document 10 section 8.2).
    """
    from .peer import require_same_layout as default_check

    check = require_same_layout or default_check
    subset = {name: peers[name] for name in members if name in peers}
    if len(subset) < 2:
        return
    check(subset)


def deduplicate_tied(
    payloads: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Drop tied parameters that would be averaged twice.

    The dedup key is ``(id(tensor), shape)``. A tie that appears in two groups is a
    ``ContractError`` naming both nodes, because a parameter cannot legitimately
    receive two different blended values.
    """
    seen: dict[tuple[int, tuple[int, ...]], str] = {}
    cleaned: dict[str, dict[str, torch.Tensor]] = {}
    for node, tensors in payloads.items():
        kept: dict[str, torch.Tensor] = {}
        for name, tensor in tensors.items():
            key = (id(tensor), tuple(tensor.shape))
            owner = seen.get(key)
            if owner is None:
                seen[key] = node
                kept[name] = tensor
                continue
            raise ContractError(
                f"parameter {name!r} is tied between {owner!r} and {node!r}; a tied "
                "parameter cannot receive two different blended values"
            )
        cleaned[node] = kept
    return cleaned
