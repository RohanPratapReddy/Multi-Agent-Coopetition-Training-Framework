"""Delegation policies: which children a node asks for a proposal.

Delegation is recorded per node (document 10 section 4.1) so a child that is
never chosen is distinguishable from a child that is chosen and performs badly.
Without that record the two look identical in a scorecard, and they need
different fixes.

Every policy's selection is validated by :func:`validate_selection` before it is
returned: a policy cannot route work to a non-child, a missing path, or a failed
node, and a non-leaf cannot silently delegate nothing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Mapping, Sequence

from ..exceptions import ContractError, ValidationError
from .tree import PeerNode, PeerTree

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..orchestration.episode import Task

__all__ = [
    "BroadcastDelegation",
    "ConfidenceDelegation",
    "DELEGATION_TYPES",
    "DelegationPolicy",
    "ExplicitDelegation",
    "RouteByRoleDelegation",
    "SelectKDelegation",
    "build_delegation",
    "delegation_capabilities",
    "validate_selection",
]


def _ordered_children(node: PeerNode) -> tuple[str, ...]:
    """Children in lexicographic order, which is the tree's stable ordering."""
    return tuple(sorted(node.children))


def _weight_for(path: str, weights: Mapping[str, float] | None) -> float:
    if not weights:
        return 1.0
    return float(weights.get(path, 1.0))


def _score_for(path: str, scores: Mapping[str, float] | None) -> float | None:
    if not scores or path not in scores:
        return None
    return float(scores[path])


def validate_selection(
    node: PeerNode,
    selected: Sequence[str],
    *,
    tree: PeerTree,
) -> tuple[str, ...]:
    """Enforce the delegation contract on a policy's result.

    Raises :class:`ContractError` naming both paths when a selection is not a
    child, does not exist, or is failed. A non-leaf selecting nothing raises:
    silently producing no proposal is the failure mode delegation exists to
    prevent.
    """
    names = tuple(str(path) for path in selected)
    for path in names:
        if path not in tree:
            raise ContractError(
                f"delegation from {node.path!r} selected {path!r}, which is not in the tree"
            )
        parent = tree.parent(path)
        if parent != node.path:
            raise ContractError(
                f"delegation from {node.path!r} selected {path!r}, whose parent is {parent!r}"
            )
        if tree[path].status == "failed":
            raise ContractError(
                f"delegation from {node.path!r} selected {path!r}, which has failed this epoch"
            )
    if not names and node.children:
        raise ContractError(
            f"delegation from {node.path!r} selected no children; a non-leaf cannot "
            "delegate nothing"
        )
    # Stable, de-duplicated order: the same selection iterates identically in
    # every process regardless of how a policy built its list.
    return tuple(dict.fromkeys(names))


class DelegationPolicy(ABC):
    """Base class for every delegation policy."""

    name: ClassVar[str] = ""

    @abstractmethod
    def select(
        self,
        node: PeerNode,
        task: "Task | None" = None,
        *,
        tree: PeerTree,
        scores: Mapping[str, float] | None = None,
        weights: Mapping[str, float] | None = None,
        epoch: int = 0,
    ) -> tuple[str, ...]:
        """Return the child paths that participate in this request."""

    def describe(self) -> dict[str, Any]:
        return {"type": self.name, "class": type(self).__name__}

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "DelegationPolicy":
        return build_delegation(spec)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name}>"


class BroadcastDelegation(DelegationPolicy):
    """Every child. The default for small trees."""

    name = "broadcast"

    def select(self, node, task=None, *, tree, scores=None, weights=None, epoch=0):
        return validate_selection(node, _ordered_children(node), tree=tree)


class SelectKDelegation(DelegationPolicy):
    """The top ``k`` children by kappa weight, ties broken by path."""

    name = "select_k"

    def __init__(self, k: int = 1) -> None:
        if int(k) < 1:
            raise ValidationError(f"SelectKDelegation requires k >= 1, got {k!r}")
        self.k = int(k)

    def select(self, node, task=None, *, tree, scores=None, weights=None, epoch=0):
        candidates = _ordered_children(node)
        ranked = sorted(
            candidates,
            key=lambda path: (-_weight_for(path, weights), -(
                _score_for(path, scores) if _score_for(path, scores) is not None else float("-inf")
            ), path),
        )
        return validate_selection(node, ranked[: self.k], tree=tree)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "k": self.k}


class RouteByRoleDelegation(DelegationPolicy):
    """Map ``task.metadata["role"]`` to the child that owns that role.

    An unmapped role raises rather than falling back to a default: routing a
    legal-review task to whoever happened to be first is worse than failing.
    """

    name = "route_by_role"

    def __init__(self, mapping: Mapping[str, str] | None = None) -> None:
        self.mapping: dict[str, str] = dict(mapping or {})

    def select(self, node, task=None, *, tree, scores=None, weights=None, epoch=0):
        role = None
        if task is not None:
            role = (getattr(task, "metadata", {}) or {}).get("role")
        if role is None:
            # No role on the task: broadcast is the only defensible behaviour,
            # because no routing decision was asked for.
            return validate_selection(node, _ordered_children(node), tree=tree)
        target = self.mapping.get(str(role))
        if target is None:
            raise ContractError(
                f"RouteByRoleDelegation has no child mapped for role {role!r} at "
                f"{node.path!r}; mapped roles: "
                + (", ".join(sorted(self.mapping)) or "(none)")
            )
        child = target if target in node.children else f"{node.path}/{target}"
        if child not in node.children:
            raise ContractError(
                f"RouteByRoleDelegation mapped role {role!r} to {target!r}, which is not "
                f"a child of {node.path!r}; children: " + ", ".join(node.children)
            )
        return validate_selection(node, (child,), tree=tree)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "mapping": dict(self.mapping)}


class ConfidenceDelegation(DelegationPolicy):
    """Children scoring above ``threshold``; otherwise the best one.

    A subtree never starves itself: with no child above the threshold the
    highest-scoring child is still selected, because producing nothing is not a
    valid outcome for a non-leaf.
    """

    name = "confidence"

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)

    def select(self, node, task=None, *, tree, scores=None, weights=None, epoch=0):
        candidates = _ordered_children(node)
        if not candidates:
            return ()
        qualifying = [
            path
            for path in candidates
            if (_score_for(path, scores) or float("-inf")) >= self.threshold
        ]
        if qualifying:
            return validate_selection(node, qualifying, tree=tree)
        best = max(
            candidates,
            key=lambda path: (
                _score_for(path, scores) if _score_for(path, scores) is not None else float("-inf"),
                # Ties resolve to the lexicographically smallest path.
                tuple(-ord(char) for char in path),
            ),
        )
        return validate_selection(node, (best,), tree=tree)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "threshold": self.threshold}


class ExplicitDelegation(DelegationPolicy):
    """``fn(node, task, *, tree, scores, weights, epoch) -> Sequence[str]``."""

    name = "explicit"

    def __init__(self, fn: Callable[..., Sequence[str]] | None = None) -> None:
        if fn is None:
            raise ValidationError("ExplicitDelegation requires an 'fn' callable")
        if not callable(fn):
            raise ValidationError(f"ExplicitDelegation fn is not callable: {fn!r}")
        self.fn = fn

    def select(self, node, task=None, *, tree, scores=None, weights=None, epoch=0):
        chosen = self.fn(
            node, task, tree=tree, scores=scores, weights=weights, epoch=epoch
        )
        return validate_selection(node, tuple(chosen or ()), tree=tree)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "fn": getattr(self.fn, "__name__", repr(self.fn))}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
DELEGATION_TYPES: dict[str, type[DelegationPolicy]] = {
    "broadcast": BroadcastDelegation,
    "select_k": SelectKDelegation,
    "route_by_role": RouteByRoleDelegation,
    "confidence": ConfidenceDelegation,
    "explicit": ExplicitDelegation,
}


def build_delegation(spec: Mapping[str, Any] | str | None) -> DelegationPolicy:
    """Resolve a policy from a name or a ``{"type": ..., **kwargs}`` spec."""
    if spec is None:
        return BroadcastDelegation()
    if isinstance(spec, str):
        payload: dict[str, Any] = {"type": spec}
    else:
        payload = dict(spec)
    type_name = payload.pop("type", None)
    if type_name is None:
        raise ValidationError(
            "a delegation spec requires a 'type' key; accepted types: "
            + ", ".join(sorted(DELEGATION_TYPES))
        )
    if type_name not in DELEGATION_TYPES:
        raise ValidationError(
            f"unknown delegation type {type_name!r}; accepted types: "
            + ", ".join(sorted(DELEGATION_TYPES))
        )
    cls = DELEGATION_TYPES[type_name]
    accepted = _constructor_keys(cls)
    unknown = sorted(set(payload) - accepted)
    if unknown:
        raise ValidationError(
            f"{cls.__name__} does not accept "
            + ", ".join(repr(key) for key in unknown)
            + f"; accepted keys: {', '.join(sorted(accepted)) or '(none)'}"
        )
    return cls(**payload)


def delegation_capabilities() -> list[dict[str, Any]]:
    """The catalog backing the API's capability listing."""
    entries: list[dict[str, Any]] = []
    for name in sorted(DELEGATION_TYPES):
        cls = DELEGATION_TYPES[name]
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
            if name in ("self", "node", "task"):
                continue
            if parameter.kind in (Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL):
                continue
            accepted.add(name)
        break
    return frozenset(accepted)
