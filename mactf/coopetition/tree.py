"""The peer tree: the one topology every layer operates on (document 10).

Hierarchy is a property of the orchestrator the user designs, so the framework
does not have a hierarchical mode. A flat swarm is a :class:`PeerTree` of depth
one, built by :meth:`PeerTree.flat`, and a depth-7 tree is the same code path.

Two rules in this module are load-bearing:

* **Paths are the identity.** A node's path appears in the ledger, the scorecard,
  the checkpoint, and the API. Paths are validated against an explicit grammar
  and ``.``/``..`` segments are rejected, because a path is interpolated into
  filesystem locations for per-node artifacts.
* **Ordering is lexicographic, never insertion order.** Two processes that build
  the same topology in a different order iterate it identically, which is what
  makes aggregation and checkpointing reproducible.

This module must not import :mod:`mactf.training` (document 10 section 2). The
tree is a data structure the trainer consumes; a checkpoint reader needs
``PeerTree.from_dict`` without dragging torch optimizers into the import graph.
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Sequence

from ..exceptions import ContractError, ValidationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..runtime import Runtime
    from .peer import Peer

__all__ = [
    "DuplicateNodeError",
    "HierarchySpec",
    "LevelSpec",
    "NodeKind",
    "OrphanNodeError",
    "PeerNode",
    "PeerTree",
    "TreeCycleError",
    "TreeInconsistencyError",
    "TreeTooLargeError",
    "TreeValidationError",
    "node_seed",
    "validate_path",
]

#: ``segment := [A-Za-z0-9_.:-]+`` and no empty segments.
_SEGMENT = re.compile(r"^[A-Za-z0-9_.:-]+$")

#: The statuses the framework writes. Stored as plain strings so a reader that
#: predates a new status can still display it (document 10 section 3.2).
NODE_STATUSES = ("active", "failed", "frozen", "pruned")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class TreeValidationError(ContractError):
    """A topology the framework refuses to run.

    A ``ContractError`` subclass so the HTTP layer's problem+json mapping turns
    it into a 422 carrying the offending path without a new error branch.
    """


class DuplicateNodeError(TreeValidationError):
    """A path was added twice."""


class OrphanNodeError(TreeValidationError):
    """A node's parent does not exist."""


class TreeCycleError(TreeValidationError):
    """A node is its own ancestor."""


class TreeInconsistencyError(TreeValidationError):
    """A node's stored children disagree with the nodes that name it parent."""


class TreeTooLargeError(TreeValidationError):
    """A node, depth, or arity cap was exceeded."""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def validate_path(path: str) -> str:
    """Validate a node path and return it unchanged.

    Rejections, each naming the offender (document 10 section 3.3): empty path,
    leading or trailing separator, empty segment, a character outside the allowed
    set, and a segment that is exactly ``.`` or ``..``.
    """
    if not isinstance(path, str):
        raise TreeValidationError(f"a node path must be a string, got {type(path).__name__}")
    if not path:
        raise TreeValidationError("a node path cannot be empty")
    if path.startswith("/"):
        raise TreeValidationError(f"invalid path {path!r}: leading separator")
    if path.endswith("/"):
        raise TreeValidationError(f"invalid path {path!r}: trailing separator")
    if path == "." or path == "..":
        raise TreeValidationError(
            f"invalid path {path!r}: a path may not be the current- or parent-directory marker"
        )
    for segment in path.split("/"):
        if not segment:
            raise TreeValidationError(f"invalid path {path!r}: empty segment")
        if segment in (".", ".."):
            raise TreeValidationError(
                f"invalid path {path!r}: segment {segment!r} is a directory traversal; "
                "paths are used in filesystem locations"
            )
        if not _SEGMENT.match(segment):
            raise TreeValidationError(
                f"invalid path {path!r}: segment {segment!r} contains a character outside "
                "[A-Za-z0-9_.:-]"
            )
    return path


def _parent_of(path: str) -> str | None:
    head, separator, _ = path.rpartition("/")
    return head if separator else None


def _name_of(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def node_seed(run_seed: int, path: str, counter: int = 0) -> int:
    """Deterministic per-node seed derived from ``blake2b``.

    Builtin ``hash()`` on a tuple containing a ``str`` is salted by
    ``PYTHONHASHSEED``, so a hierarchy that seeded from ``hash((seed, path))``
    would produce different tasks in different processes (document 10 section
    7.4). ``blake2b`` is stable across processes, machines, and interpreter
    versions, which is what a reproducible run requires.
    """
    digest = hashlib.blake2b(
        f"{int(run_seed)}|{path}|{int(counter)}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


# ---------------------------------------------------------------------------
# Node kinds
# ---------------------------------------------------------------------------
class NodeKind(str, Enum):
    """The display vocabulary for depth. Labels are not limits."""

    PEER = "peer"
    SUBAGENT = "subagent"
    MINIAGENT = "miniagent"
    MICROAGENT = "microagent"
    NANOAGENT = "nanoagent"
    GENERIC = "generic"

    @classmethod
    def for_depth(cls, depth: int) -> "NodeKind":
        """The kind for a 1-based depth; every level past the named ones is ``GENERIC``."""
        if depth < 1:
            raise TreeValidationError(f"depth must be >= 1, got {depth!r}")
        table = {
            1: cls.PEER,
            2: cls.SUBAGENT,
            3: cls.MINIAGENT,
            4: cls.MICROAGENT,
            5: cls.NANOAGENT,
        }
        return table.get(depth, cls.GENERIC)

    def label_plural(self) -> str:
        """``subagents``, not ``subagentss``: pluralization lives in one place."""
        return {
            NodeKind.PEER: "peers",
            NodeKind.SUBAGENT: "subagents",
            NodeKind.MINIAGENT: "miniagents",
            NodeKind.MICROAGENT: "microagents",
            NodeKind.NANOAGENT: "nanoagents",
            NodeKind.GENERIC: "nodes",
        }[self]

    def label(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
@dataclass
class PeerNode:
    """One node in the tree: a path, a kind, and optionally a live :class:`Peer`."""

    path: str
    name: str
    kind: NodeKind
    depth: int
    parent: str | None
    children: tuple[str, ...] = ()
    peer: "Peer | None" = None
    role: str = "worker"
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    def describe(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "kind": self.kind.value,
            "depth": self.depth,
            "parent": self.parent,
            "children": list(self.children),
            "role": self.role,
            "status": self.status,
            "has_peer": self.peer is not None,
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.describe()


# ---------------------------------------------------------------------------
# Hierarchy spec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LevelSpec:
    """One level of a generated hierarchy: how many children each parent gets.

    ``count`` is the uniform arity and ``n_count`` the irregular one. Supplying
    both is rejected rather than resolved by precedence, because a silent
    precedence rule trains a topology nobody chose (document 10 section 3.4).
    """

    label: str
    count: int = 0
    model: Any = None
    strategy: str | None = None
    roles: tuple[str, ...] = ()
    n_count: int | tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.label:
            raise ValidationError("a LevelSpec requires a label")
        if self.count and self.n_count is not None:
            raise ValidationError(
                f"LevelSpec {self.label!r} sets both count and n_count; choose one"
            )
        if self.count < 0:
            raise ValidationError(f"LevelSpec {self.label!r} count must be >= 0")
        if isinstance(self.n_count, int) and self.n_count < 0:
            raise ValidationError(f"LevelSpec {self.label!r} n_count must be >= 0")
        if isinstance(self.n_count, tuple) and any(value < 0 for value in self.n_count):
            raise ValidationError(f"LevelSpec {self.label!r} n_count entries must be >= 0")

    def arity_for(self, parent_index: int, *, rng: Any = None, policy: str = "uniform") -> int:
        """How many children the ``parent_index``-th parent at this level gets."""
        if self.n_count is not None:
            if isinstance(self.n_count, int):
                return self.n_count
            if not self.n_count:
                return 0
            # Shorter than the parent count is a spec bug, not a licence to
            # reuse index 0: repeat the last entry so the shape is defined.
            return self.n_count[min(parent_index, len(self.n_count) - 1)]
        if policy == "random" and rng is not None:
            return int(rng.integers(0, self.count + 1)) if self.count else 0
        return self.count

    def to_spec(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"label": self.label, "count": self.count}
        if self.n_count is not None:
            payload["n_count"] = (
                list(self.n_count) if isinstance(self.n_count, tuple) else self.n_count
            )
        if self.strategy is not None:
            payload["strategy"] = self.strategy
        if self.roles:
            payload["roles"] = list(self.roles)
        return payload

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "LevelSpec":
        payload = dict(spec or {})
        label = payload.pop("label", payload.pop("name", None))
        if label is None:
            raise ValidationError("a LevelSpec spec requires a 'label'")
        n_count = payload.pop("n_count", None)
        if isinstance(n_count, list):
            n_count = tuple(int(value) for value in n_count)
        roles = payload.pop("roles", ())
        unknown = sorted(set(payload) - {"count", "model", "strategy"})
        if unknown:
            raise ValidationError(
                "LevelSpec does not accept "
                + ", ".join(repr(key) for key in unknown)
                + "; accepted keys: label, count, n_count, model, strategy, roles"
            )
        return cls(
            label=str(label),
            count=int(payload.get("count", 0)),
            model=payload.get("model"),
            strategy=payload.get("strategy"),
            roles=tuple(roles),
            n_count=n_count,
        )


@dataclass(frozen=True)
class HierarchySpec:
    """A declarative topology: levels, roots, caps, sharing, and attribution."""

    levels: tuple[LevelSpec, ...] = ()
    roots: int = 1
    root_models: Mapping[str, Any] | None = None
    root_names: tuple[str, ...] = ()
    max_depth: int = 8
    max_nodes: int = 256
    max_children_per_node: int = 16
    fanout_policy: str = "uniform"
    sharing: str = "independent"
    attribution: str = "rollup"
    cross_level_grad: bool = False
    allow_double_count: bool = False
    seed: int = 0
    delegation: str = "broadcast"
    synthesis: str = "consensus"
    rollup: str = "kappa_weighted"

    def __post_init__(self) -> None:
        if self.roots < 1:
            raise ValidationError("HierarchySpec.roots must be >= 1")
        if self.max_depth < 1:
            raise ValidationError("HierarchySpec.max_depth must be >= 1")
        if self.max_nodes < 1:
            raise ValidationError("HierarchySpec.max_nodes must be >= 1")
        if self.max_children_per_node < 1:
            raise ValidationError("HierarchySpec.max_children_per_node must be >= 1")
        if self.fanout_policy not in ("uniform", "random", "explicit"):
            raise ValidationError(
                f"unknown fanout_policy {self.fanout_policy!r}; "
                "use 'uniform', 'random', or 'explicit'"
            )
        if self.sharing not in ("independent", "shared_trunk", "tied_weights"):
            raise ValidationError(
                f"unknown sharing {self.sharing!r}; "
                "use 'independent', 'shared_trunk', or 'tied_weights'"
            )
        if self.attribution not in ("rollup", "own"):
            raise ValidationError(
                f"unknown attribution {self.attribution!r}; use 'rollup' or 'own'"
            )
        if self.cross_level_grad and self.sharing == "shared_trunk" and not self.allow_double_count:
            raise ValidationError(
                "cross_level_grad=True with sharing='shared_trunk' double-counts trunk "
                "gradients, once per child path; set allow_double_count=True to accept it"
            )
        # Roots occupy depth 1, so ``levels`` reach depth ``len(levels) + 1``.
        if len(self.levels) + 1 > self.max_depth:
            deepest = self.levels[-1].label if self.levels else ""
            raise ValidationError(
                f"{len(self.levels)} level(s) would reach depth {len(self.levels) + 1} "
                f"(deepest level {deepest!r}), exceeding max_depth={self.max_depth}"
            )

    # -- construction ----------------------------------------------------
    @classmethod
    def flat(cls, names: Sequence[str], *, model: Any = None, **kwargs: Any) -> "HierarchySpec":
        """The degenerate tree: one level, one node per name.

        ``HierarchySpec.flat(PEERS)`` is exactly a run with no hierarchy, which is
        why the trainer can use one code path for both (document 10 section 3.4).
        """
        names = tuple(str(name) for name in names)
        if not names:
            raise ValidationError("HierarchySpec.flat requires at least one peer name")
        for name in names:
            validate_path(name)
        if len(set(names)) != len(names):
            raise ValidationError("HierarchySpec.flat peer names must be unique")
        return cls(levels=(), roots=len(names), root_names=names, root_models={"": model} if model else None, **kwargs)

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "HierarchySpec":
        payload = dict(spec or {})
        raw_levels = payload.pop("levels", ())
        levels = tuple(
            LevelSpec.from_spec(level) if not isinstance(level, LevelSpec) else level
            for level in raw_levels
        )
        roots = payload.pop("roots", 1)
        root_names = payload.pop("root_names", ())
        root_models = payload.pop("root_models", None)
        accepted = {
            "max_depth",
            "max_nodes",
            "max_children_per_node",
            "fanout_policy",
            "sharing",
            "attribution",
            "cross_level_grad",
            "allow_double_count",
            "seed",
            "delegation",
            "synthesis",
            "rollup",
        }
        unknown = sorted(set(payload) - accepted)
        if unknown:
            raise ValidationError(
                "HierarchySpec does not accept "
                + ", ".join(repr(key) for key in unknown)
                + "; accepted keys: "
                + ", ".join(sorted(accepted | {"levels", "roots", "root_names", "root_models"}))
            )
        return cls(
            levels=levels,
            roots=int(roots),
            root_models=root_models,
            root_names=tuple(str(name) for name in root_names),
            **payload,
        )

    def to_spec(self) -> dict[str, Any]:
        return {
            "levels": [level.to_spec() for level in self.levels],
            "roots": self.roots,
            "root_names": list(self.root_names),
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
            "max_children_per_node": self.max_children_per_node,
            "fanout_policy": self.fanout_policy,
            "sharing": self.sharing,
            "attribution": self.attribution,
            "cross_level_grad": self.cross_level_grad,
            "allow_double_count": self.allow_double_count,
            "seed": self.seed,
            "delegation": self.delegation,
            "synthesis": self.synthesis,
            "rollup": self.rollup,
        }

    # -- the shape -------------------------------------------------------
    def labels(self) -> tuple[str, ...]:
        return tuple(level.label for level in self.levels)

    def node_count(self) -> int:
        """The node count this spec would build, without building it."""
        if not self.levels:
            return self.roots
        total = self.roots
        frontier = self.roots
        for level in self.levels:
            arity = level.count if level.n_count is None else (
                level.n_count if isinstance(level.n_count, int) else sum(level.n_count)
            )
            frontier = frontier * arity
            total += frontier
        return total

    def build(self, *, runtime: "Runtime | None" = None, settings: Any = None,
              peer_factory: Callable[[str, int, "LevelSpec | None"], "Peer | None"] | None = None) -> "PeerTree":
        """Create the tree this spec describes, with ``peer=None`` unless a factory is given.

        ``runtime`` and ``settings`` are accepted so the trainer can pass the same
        arguments as the document's signature; the tree itself needs neither, and
        the peer factory is what turns a path into a live :class:`Peer` (the
        trainer supplies it in step 2 of document 10 section 6.1).
        """
        tree = PeerTree(max_depth=self.max_depth, max_nodes=self.max_nodes)
        root_names = list(self.root_names)
        if not root_names:
            if not self.levels and self.roots != 1:
                root_names = [f"peer-{index + 1}" for index in range(self.roots)]
            elif not self.levels:
                root_names = ["peer-1"]
            else:
                root_names = [f"peer-{index + 1}" for index in range(self.roots)]
        if len(root_names) != self.roots:
            raise ValidationError(
                f"root_names has {len(root_names)} entries but roots={self.roots}"
            )
        for name in root_names:
            node = tree.add_node(name, role="root")
            if peer_factory is not None:
                node.peer = peer_factory(node.path, node.depth, None)
        if not self.levels:
            return tree

        rng = None
        if self.fanout_policy == "random":
            import numpy as np  # local import: numpy is optional at tree-only use

            rng = np.random.default_rng(node_seed(self.seed, "fanout"))

        frontier = list(tree.root_paths)
        for level_index, level in enumerate(self.levels):
            depth = level_index + 2
            if depth > self.max_depth:
                raise TreeTooLargeError(
                    f"level {level.label!r} would be at depth {depth}, "
                    f"exceeding max_depth={self.max_depth}"
                )
            next_frontier: list[str] = []
            for parent_index, parent_path in enumerate(frontier):
                arity = level.arity_for(parent_index, rng=rng, policy=self.fanout_policy)
                if arity > self.max_children_per_node:
                    raise TreeTooLargeError(
                        f"{parent_path}: {arity} children exceeds "
                        f"max_children_per_node={self.max_children_per_node}"
                    )
                for child_index in range(arity):
                    name = f"{level.label}-{child_index + 1}"
                    child_path = f"{parent_path}/{name}"
                    role = level.roles[child_index % len(level.roles)] if level.roles else "worker"
                    node = tree.add_node(child_path, role=role)
                    if peer_factory is not None:
                        node.peer = peer_factory(node.path, node.depth, level)
                    next_frontier.append(child_path)
            frontier = next_frontier
            if not frontier:
                break
        tree.validate()
        return tree


# ---------------------------------------------------------------------------
# The tree
# ---------------------------------------------------------------------------
class PeerTree:
    """A validated, lexicographically ordered peer topology."""

    def __init__(self, *, max_depth: int = 8, max_nodes: int = 256) -> None:
        if max_depth < 1:
            raise TreeValidationError(f"max_depth must be >= 1, got {max_depth!r}")
        if max_nodes < 1:
            raise TreeValidationError(f"max_nodes must be >= 1, got {max_nodes!r}")
        self.max_depth = int(max_depth)
        self.max_nodes = int(max_nodes)
        self._nodes: dict[str, PeerNode] = {}

    # -- construction ----------------------------------------------------
    @classmethod
    def flat(cls, names: Sequence[str], *, max_nodes: int = 256) -> "PeerTree":
        tree = cls(max_nodes=max_nodes)
        for name in names:
            tree.add_node(str(name))
        return tree

    def add_node(
        self,
        path: str,
        *,
        peer: "Peer | None" = None,
        role: str = "worker",
        kind: NodeKind | None = None,
        metadata: Mapping[str, Any] | None = None,
        status: str = "active",
    ) -> PeerNode:
        """Add a node, validating the path, its parent, and the caps."""
        validate_path(path)
        if path in self._nodes:
            raise DuplicateNodeError(f"duplicate node path {path!r}")
        parent_path = _parent_of(path)
        if parent_path is not None and parent_path not in self._nodes:
            raise OrphanNodeError(f"orphan node {path!r}: parent {parent_path!r} does not exist")
        depth = path.count("/") + 1
        if depth > self.max_depth:
            raise TreeTooLargeError(
                f"{path!r} is at depth {depth}, exceeding max_depth={self.max_depth}"
            )
        if len(self._nodes) >= self.max_nodes:
            raise TreeTooLargeError(
                f"adding {path!r} would exceed max_nodes={self.max_nodes}"
            )
        resolved_kind = kind or NodeKind.for_depth(depth)
        if status not in NODE_STATUSES:
            raise TreeValidationError(
                f"{path!r}: unknown status {status!r}; known: " + ", ".join(NODE_STATUSES)
            )
        node = PeerNode(
            path=path,
            name=_name_of(path),
            kind=resolved_kind,
            depth=depth,
            parent=parent_path,
            peer=peer,
            role=role,
            status=status,
            metadata=dict(metadata or {}),
        )
        self._nodes[path] = node
        if parent_path is not None:
            parent = self._nodes[parent_path]
            parent.children = tuple(sorted(parent.children + (path,)))
        return node

    def remove_node(self, path: str, *, recursive: bool = False) -> int:
        """Remove a node and, when ``recursive``, its descendants.

        Removing a node with children non-recursively is rejected: orphaning
        children is never a valid state, so it is not offered (document 10
        section 9.3).
        """
        if path not in self._nodes:
            raise TreeValidationError(f"cannot remove {path!r}: no such node")
        node = self._nodes[path]
        if node.children and not recursive:
            raise TreeValidationError(
                f"cannot remove {path!r}: it has {len(node.children)} children"
            )
        removed = [path, *self.descendants(path)]
        for target in reversed(removed):
            self._nodes.pop(target, None)
        if node.parent is not None and node.parent in self._nodes:
            parent = self._nodes[node.parent]
            parent.children = tuple(child for child in parent.children if child != path)
        return len(removed)

    # -- navigation ------------------------------------------------------
    @property
    def root_paths(self) -> tuple[str, ...]:
        return tuple(sorted(node.path for node in self._nodes.values() if node.parent is None))

    @property
    def leaves(self) -> tuple[str, ...]:
        return tuple(sorted(node.path for node in self._nodes.values() if not node.children))

    @property
    def depth(self) -> int:
        """The deepest occupied level; 0 for an empty tree."""
        return max((node.depth for node in self._nodes.values()), default=0)

    @property
    def levels(self) -> Mapping[int, tuple[str, ...]]:
        levels: dict[int, list[str]] = {}
        for node in self._nodes.values():
            levels.setdefault(node.depth, []).append(node.path)
        return {depth: tuple(sorted(paths)) for depth, paths in sorted(levels.items())}

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, path: object) -> bool:
        return path in self._nodes

    def __iter__(self) -> Iterator[PeerNode]:
        return self.walk("breadth")

    def __getitem__(self, path: str) -> PeerNode:
        try:
            return self._nodes[path]
        except KeyError:
            raise KeyError(f"no node at path {path!r}") from None

    def get(self, path: str) -> PeerNode | None:
        return self._nodes.get(path)

    def children(self, path: str) -> tuple[str, ...]:
        return self[path].children

    def parent(self, path: str) -> str | None:
        return self[path].parent

    def ancestors(self, path: str) -> tuple[str, ...]:
        """Every ancestor, nearest first. Excludes ``path`` itself."""
        node = self[path]
        chain: list[str] = []
        current = node.parent
        while current is not None:
            chain.append(current)
            current = self._nodes[current].parent
        return tuple(chain)

    def descendants(self, path: str) -> tuple[str, ...]:
        """Every descendant, parents before children, lexicographic within a level."""
        collected: list[str] = []
        frontier = list(self[path].children)
        while frontier:
            next_frontier: list[str] = []
            for child in frontier:
                collected.append(child)
                next_frontier.extend(self._nodes[child].children)
            frontier = next_frontier
        return tuple(collected)

    def siblings(self, path: str) -> tuple[str, ...]:
        """The other children of this node's parent; roots are siblings of each other."""
        node = self[path]
        if node.parent is None:
            return tuple(other for other in self.root_paths if other != path)
        return tuple(child for child in self._nodes[node.parent].children if child != path)

    def arity(self, path: str) -> int:
        return len(self[path].children)

    def level(self, depth: int) -> tuple[str, ...]:
        return self.levels.get(int(depth), ())

    def subtree(self, path: str) -> "PeerTree":
        """A new tree rooted at ``path``, with paths re-based and nodes shared by reference."""
        source = self[path]
        tree = PeerTree(max_depth=self.max_depth, max_nodes=self.max_nodes)
        mapping: dict[str, str] = {}

        def clone_node(node: PeerNode, new_path: str) -> None:
            mapping[node.path] = new_path
            tree.add_node(
                new_path,
                peer=node.peer,
                role=node.role,
                kind=NodeKind.for_depth(new_path.count("/") + 1),
                metadata=node.metadata,
                status=node.status,
            )

        clone_node(source, source.name)
        for descendant in self.descendants(path):
            node = self._nodes[descendant]
            relative = descendant[len(path) + 1 :]
            clone_node(node, f"{source.name}/{relative}")
        return tree

    def walk(self, order: str = "leaves-first") -> Iterator[PeerNode]:
        """Iterate nodes in one of the four documented orders."""
        nodes = self._nodes
        if order == "root-first":
            ordered = _walk_post_order(self.root_paths, nodes)
            yield from (nodes[path] for path in ordered)
        elif order == "leaves-first":
            ordered = _walk_leaves_first(self.root_paths, nodes)
            yield from (nodes[path] for path in ordered)
        elif order == "breadth":
            for depth in sorted(self.levels):
                for path in self.levels[depth]:
                    yield nodes[path]
        elif order == "depth":
            # Leaves first but level by level: deepest level first, lexicographic
            # inside a level. Aggregation runs deepest level up.
            for depth in sorted(self.levels, reverse=True):
                for path in self.levels[depth]:
                    yield nodes[path]
        else:
            raise TreeValidationError(
                f"unknown walk order {order!r}; use 'leaves-first', 'root-first', "
                "'breadth', or 'depth'"
            )

    # -- validation ------------------------------------------------------
    def validate(self) -> None:
        """Check every invariant, raising the specific error subclass.

        Two passes, so the error names the node that is actually wrong. A single
        pass would report a parent's stale ``children`` tuple before reaching the
        descendant whose ``parent`` was corrupted, and the second message is the
        one that explains the corruption.
        """
        if len(self._nodes) > self.max_nodes:
            raise TreeTooLargeError(
                f"tree has {len(self._nodes)} nodes, exceeding max_nodes={self.max_nodes}"
            )
        for path in sorted(self._nodes):
            node = self._nodes[path]
            validate_path(path)
            if node.path != path:
                raise TreeInconsistencyError(
                    f"node registered at {path!r} reports path {node.path!r}"
                )
            if node.depth != path.count("/") + 1:
                raise TreeInconsistencyError(
                    f"{path!r}: depth {node.depth} does not equal "
                    f"path.count('/') + 1 = {path.count('/') + 1}"
                )
            if node.depth > self.max_depth:
                raise TreeTooLargeError(
                    f"{path!r} is at depth {node.depth}, exceeding max_depth={self.max_depth}"
                )
            if node.name != _name_of(path):
                raise TreeInconsistencyError(
                    f"{path!r}: name {node.name!r} does not match the last segment"
                )
            expected_parent = _parent_of(path)
            if node.parent != expected_parent:
                raise TreeInconsistencyError(
                    f"{path!r}: parent {node.parent!r} does not match "
                    f"{expected_parent!r} implied by the path"
                )
            if expected_parent is not None and expected_parent not in self._nodes:
                raise OrphanNodeError(
                    f"{path!r}: parent {expected_parent!r} does not exist"
                )
            if node.parent is not None and node.parent == path:
                raise TreeCycleError(f"{path!r} is its own parent")
            ancestor = node.parent
            seen = {path}
            while ancestor is not None:
                if ancestor in seen:
                    raise TreeCycleError(f"{path!r} is an ancestor of itself")
                seen.add(ancestor)
                ancestor = self._nodes[ancestor].parent if ancestor in self._nodes else None
            if node.status not in NODE_STATUSES:
                raise TreeValidationError(
                    f"{path!r}: unknown status {node.status!r}; known: "
                    + ", ".join(NODE_STATUSES)
                )

        for path in sorted(self._nodes):
            node = self._nodes[path]
            actual_children = tuple(
                sorted(child for child, candidate in self._nodes.items() if candidate.parent == path)
            )
            if tuple(sorted(node.children)) != actual_children:
                raise TreeInconsistencyError(
                    f"{path!r}: children {sorted(node.children)} disagree with the nodes "
                    f"that name it parent {list(actual_children)}"
                )

    # -- serialization ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Full topology including node status and metadata."""
        return {
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
            "root_paths": list(self.root_paths),
            "nodes": [
                {
                    **node.describe(),
                    "kind": node.kind.value,
                }
                for node in self.walk("breadth")
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PeerTree":
        """Restore a tree, tolerating a schema-version-1 ``peers``-only payload.

        An older checkpoint has no ``nodes`` entry: its ``peers`` dict keys are
        root paths, and a flat tree is the correct reading (document 10 section
        3.6).
        """
        if not isinstance(payload, Mapping):
            raise TreeValidationError(
                f"PeerTree.from_dict needs a mapping, got {type(payload).__name__}"
            )
        tree = cls(
            max_depth=int(payload.get("max_depth", 8) or 8),
            max_nodes=int(payload.get("max_nodes", 256) or 256),
        )
        raw_nodes = payload.get("nodes")
        if not raw_nodes:
            peers = payload.get("peers") or {}
            if not peers:
                raise TreeValidationError(
                    "PeerTree.from_dict needs either a 'nodes' list or a 'peers' mapping"
                )
            for name in peers:
                tree.add_node(str(name))
            return tree
        entries = sorted(raw_nodes, key=lambda entry: (entry.get("depth", 0), entry.get("path", "")))
        for entry in entries:
            path = str(entry["path"])
            kind_value = entry.get("kind")
            kind = NodeKind(kind_value) if kind_value else None
            tree.add_node(
                path,
                role=str(entry.get("role", "worker")),
                kind=kind,
                metadata=entry.get("metadata") or {},
                status=str(entry.get("status", "active")),
            )
        tree.validate()
        return tree

    def state_dict(self) -> dict[str, Any]:
        """Topology only: no parameters, no peers, no metadata payload.

        Checkpoint section boundaries follow this: parameters live in the peer
        section, topology here, so a topology diff is readable without loading
        gigabytes of tensors.
        """
        return {
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
            "nodes": [
                {
                    "path": node.path,
                    "parent": node.parent,
                    "kind": node.kind.value,
                    "depth": node.depth,
                    "role": node.role,
                    "status": node.status,
                }
                for node in self.walk("breadth")
            ],
        }

    def describe(self) -> str:
        if not self._nodes:
            return "PeerTree(empty)"
        lines = [
            f"PeerTree({len(self._nodes)} nodes, depth={self.depth}, "
            f"roots={len(self.root_paths)})"
        ]
        for node in self.walk("breadth"):
            indent = "  " * (node.depth - 1)
            marker = "" if node.status == "active" else f" [{node.status}]"
            lines.append(f"{indent}- {node.name} ({node.kind.label()}){marker}")
        return "\n".join(lines)

    # -- peers -----------------------------------------------------------
    def peers(self) -> dict[str, "Peer"]:
        """Live peers keyed by path, in walk order; nodes without a peer are skipped."""
        return {
            node.path: node.peer
            for node in self.walk("leaves-first")
            if node.peer is not None
        }

    def clone(self) -> "PeerTree":
        """A structural copy; peers are shared by reference, not deep-copied."""
        copy_tree = PeerTree(max_depth=self.max_depth, max_nodes=self.max_nodes)
        for node in self.walk("root-first"):
            copy_tree.add_node(
                node.path,
                peer=node.peer,
                role=node.role,
                kind=node.kind,
                metadata=node.metadata,
                status=node.status,
            )
        return copy_tree

    def mark(self, path: str, status: str) -> None:
        """Set a node's status, validating the value."""
        if status not in NODE_STATUSES:
            raise TreeValidationError(
                f"{path!r}: unknown status {status!r}; known: " + ", ".join(NODE_STATUSES)
            )
        self[path].status = status

    def __repr__(self) -> str:
        return f"<PeerTree nodes={len(self._nodes)} depth={self.depth}>"


def _walk_post_order(root_paths: Sequence[str], nodes: Mapping[str, PeerNode]) -> list[str]:
    """Parents before children, lexicographic at each level."""
    ordered: list[str] = []
    frontier = list(root_paths)
    while frontier:
        next_frontier: list[str] = []
        for path in sorted(frontier):
            ordered.append(path)
            next_frontier.extend(nodes[path].children)
        frontier = next_frontier
    return ordered


def _walk_leaves_first(root_paths: Sequence[str], nodes: Mapping[str, PeerNode]) -> list[str]:
    """Children before parents, depth-first, siblings in lexicographic order.

    A post-order walk. This is the order scoring and the local update use: a
    parent is scored after every child's scorecard exists, and within a branch the
    children appear in the same lexicographic order every process would produce.
    ``walk("depth")`` is the different order aggregation needs (deepest level
    first, whole levels at a time).
    """
    ordered: list[str] = []

    def visit(path: str) -> None:
        for child in nodes[path].children:
            visit(child)
        ordered.append(path)

    for root in root_paths:
        visit(root)
    return ordered


# ---------------------------------------------------------------------------
# Aggregation groups (document 10 section 8.1)
# ---------------------------------------------------------------------------
def aggregation_groups(tree: PeerTree) -> list[tuple[str | None, tuple[str, ...]]]:
    """The root group first, then one group per parent, in root-first order."""
    groups: list[tuple[str | None, tuple[str, ...]]] = [(None, tree.root_paths)]
    for node in tree.walk("root-first"):
        if node.children:
            groups.append((node.path, node.children))
    return groups


def branch_void(
    tree: PeerTree, per_node: Mapping[str, frozenset[str]]
) -> dict[str, frozenset[str]]:
    """Union each node's own void rules into every ancestor (document 10 section 5.3).

    A non-offsettable violation at ``peer-a/nano-1`` voids ``peer-a`` even if the
    root never touched the rule. Averaging never launders a T0/T1/T2 breach, so
    this cannot be a numeric operation on the parent's own events.
    """
    result: dict[str, set[str]] = {
        node.path: set(per_node.get(node.path, ()))
        for node in tree.walk("leaves-first")
    }
    for node in tree.walk("leaves-first"):
        if node.parent is not None:
            result.setdefault(node.parent, set()).update(result.get(node.path, set()))
    return {path: frozenset(rules) for path, rules in result.items()}


def clone_into_parent(tree: PeerTree, parent: str, child: str) -> None:
    """Attach an existing node's subtree under a new parent path.

    Used by ``HierarchySpec.build`` when a caller supplies an explicit topology
    rather than a level description.
    """
    if parent not in tree:
        raise OrphanNodeError(f"cannot attach to {parent!r}: no such node")
    if child not in tree:
        raise TreeValidationError(f"cannot clone {child!r}: no such node")
    source = tree[child]
    for node in [source, *(tree[path] for path in tree.descendants(child))]:
        relative = node.path[len(child) :].lstrip("/")
        target = f"{parent}/{source.name}" + (f"/{relative}" if relative else "")
        tree.add_node(
            target,
            peer=node.peer,
            role=node.role,
            metadata=copy.deepcopy(node.metadata),
            status=node.status,
        )
