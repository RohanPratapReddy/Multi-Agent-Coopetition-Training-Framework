"""test aggregation groups (split from test_tree.py)."""

from __future__ import annotations

from mactf.coopetition.tree import (
    DuplicateNodeError,
    HierarchySpec,
    LevelSpec,
    NodeKind,
    OrphanNodeError,
    PeerTree,
    TreeInconsistencyError,
    TreeTooLargeError,
    TreeValidationError,
    aggregation_groups,
    branch_void,
    node_seed,
)


def test_aggregation_groups_put_roots_first():
    spec = HierarchySpec(levels=(LevelSpec("sub", 2),))
    groups = aggregation_groups(spec.build())
    assert groups[0] == (None, ("peer-1",))
    assert groups[1] == ("peer-1", ("peer-1/sub-1", "peer-1/sub-2"))


def test_branch_void_propagates_to_every_ancestor():
    spec = HierarchySpec(levels=(LevelSpec("sub", 2), LevelSpec("mini", 1)))
    tree = spec.build()
    result = branch_void(tree, {"peer-1/sub-1/mini-1": frozenset({"P1"})})
    assert result["peer-1/sub-1/mini-1"] == {"P1"}
    assert result["peer-1/sub-1"] == {"P1"}
    assert result["peer-1"] == {"P1"}
    assert result["peer-1/sub-2"] == frozenset()
