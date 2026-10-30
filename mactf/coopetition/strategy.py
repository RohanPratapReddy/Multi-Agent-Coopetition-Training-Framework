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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Mapping, Sequence

import torch

from ..exceptions import ContractError, ValidationError
from .tree import PeerNode, PeerTree, aggregation_groups

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..training.protocol import EpochResult, StepResult
    from .aggregation import AggregationReport, Aggregator
    from .weights import WeightUpdateStrategy

__all__ = [
    "CooperativeStrategy",
    "FrozenPeerStrategy",
    "HierarchicalStrategy",
    "IndependentStrategy",
    "KappaSoftmaxStrategy",
    "PEER_STRATEGIES",
    "PeerStrategy",
    "build_peer_strategy",
]

#: What a strategy may read from a trainer. Duck-typed so a custom trainer that
#: only implements ``PeerStrategy``'s surface still works.
_TRAINER_ATTRS = ("tree", "peers", "weight_update", "aggregator", "kappa")


class PeerStrategy(ABC):
    """Coordination policy for a set of peers."""

    name: ClassVar[str]

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Called at the top of every epoch. The default is a no-op."""

    def on_peer_step(self, trainer: Any, node: PeerNode, result: "StepResult") -> None:
        """Called after each node's training step. The default is a no-op."""

    @abstractmethod
    def on_epoch_end(
        self, trainer: Any, epoch: int, report: "AggregationReport | None"
    ) -> None:
        """Called after every node has stepped, before logging."""

    def should_aggregate(self) -> bool:
        """Whether this strategy mixes parameters at all."""
        return False

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


# ---------------------------------------------------------------------------
# Non-aggregating strategies
# ---------------------------------------------------------------------------
class IndependentStrategy(PeerStrategy):
    """Peers never share parameters. The pre-coopetition behaviour."""

    name = "independent"

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

    def __init__(self, frozen: str | Sequence[str] = (), *, still_scores: bool = True) -> None:
        if isinstance(frozen, str):
            frozen = (frozen,)
        self.frozen: tuple[str, ...] = tuple(str(path) for path in frozen)
        self.still_scores = bool(still_scores)
        self._marked: list[str] = []

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        tree = getattr(trainer, "tree", None)
        if tree is None or not self.frozen:
            return
        self._marked = []
        for path in self.frozen:
            node = tree.get(path) if isinstance(tree, PeerTree) else None
            if node is None:
                raise ContractError(
                    f"FrozenPeerStrategy names {path!r}, which is not a node in this tree"
                )
            tree.mark(path, "frozen")
            self._marked.append(path)

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

    def __init__(
        self,
        *,
        update_reference: bool = True,
        skip_void_levels: bool = True,
    ) -> None:
        self.update_reference = bool(update_reference)
        self.skip_void_levels = bool(skip_void_levels)
        self.last_reports: dict[str | None, "AggregationReport"] = {}

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


#: Every built-in, addressable by name.
PEER_STRATEGIES: dict[str, type] = {
    "independent": IndependentStrategy,
    "cooperative": CooperativeStrategy,
    "kappa_softmax": KappaSoftmaxStrategy,
    "frozen_peer": FrozenPeerStrategy,
    "hierarchical": HierarchicalStrategy,
}


def build_peer_strategy(
    spec: "PeerStrategy | str | Mapping[str, Any] | None",
    *,
    hierarchy: bool = False,
) -> PeerStrategy:
    """Resolve a strategy from an instance, a name, a spec, or ``None``.

    ``None`` resolves to :class:`HierarchicalStrategy` when ``hierarchy`` is true and
    :class:`IndependentStrategy` otherwise, which is the documented default in both
    cases.
    """
    if spec is None:
        return HierarchicalStrategy() if hierarchy else IndependentStrategy()
    if isinstance(spec, PeerStrategy):
        return spec
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key not in PEER_STRATEGIES:
            raise ValidationError(
                f"unknown peer_strategy {spec!r}; accepted: "
                + ", ".join(sorted(PEER_STRATEGIES))
            )
        return PEER_STRATEGIES[key]()
    if isinstance(spec, Mapping):
        payload = dict(spec)
        key = str(payload.pop("type", payload.pop("name", ""))).lower()
        if key not in PEER_STRATEGIES:
            raise ValidationError(
                f"unknown peer_strategy type {key!r}; accepted: "
                + ", ".join(sorted(PEER_STRATEGIES))
            )
        return PEER_STRATEGIES[key].from_spec(payload)
    if hasattr(spec, "on_epoch_end") and hasattr(spec, "should_aggregate"):
        return spec  # structural custom strategy
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
