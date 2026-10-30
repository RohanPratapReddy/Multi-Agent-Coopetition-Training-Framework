"""Helpers for testing tree mutation without a model (document 17 section 12.11.4)."""
#                          mactf/testing/__init__.py, which resolves them lazily, as it does today)
from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..coopetition.tree import PeerTree
from ..training.mutation import MutationLimits, MutationPlan, TopologySimulator


def tree_of(*paths: str) -> PeerTree:
    """A tree from paths, in the order given: ``"root", "root/team-a", "root/team-a/x"``."""
    tree = PeerTree()
    for path in paths:
        parent = path.rsplit("/", 1)[0] if "/" in path else None
        tree.add_node(path, role="root" if parent is None else "worker")
    return tree


def simulator(
    tree: PeerTree,
    *,
    retired: Iterable[str] = (),
    tombstones: Iterable[str] = (),
    limits: MutationLimits | None = None,
    min_peers: int = 1,
) -> TopologySimulator:
    """The simulator over ``tree`` with a permissive feasibility rule (``min_peers`` active)."""
    def feasible(count: int) -> None:
        if count < min_peers:
            raise ValueError(f"{count} active members; at least {min_peers} needed")

    return TopologySimulator(
        tree,
        retired=retired,
        tombstones=tombstones,
        limits=limits or MutationLimits(enabled=True, min_active_roots=1),
        feasible=feasible,
    )


def findings(plan: Mapping[str, Any] | MutationPlan, tree: PeerTree, **kwargs: Any) -> list[str]:
    """The rendered findings of simulating ``plan``: empty means the plan would commit."""
    simulation = simulator(tree, **kwargs).run(MutationPlan.parse(plan))
    return [item.render() for item in simulation.findings]
