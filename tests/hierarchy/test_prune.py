"""Pruning (document 10 section 9.3).

Pruning removes a subtree and returns the count. Non-recursive removal of a
parent is refused, because an orphan is never a valid state. The reference peer
is never pruned: removing parameters from the baseline would change every
advantage.
"""

from __future__ import annotations

import pytest

from mactf.coopetition.tree import HierarchySpec, LevelSpec, PeerTree, TreeValidationError


def _tree() -> PeerTree:
    return HierarchySpec(levels=(LevelSpec("sub", 2), LevelSpec("mini", 2))).build()


def test_recursive_removal_removes_the_subtree_and_returns_the_count():
    tree = _tree()
    assert len(tree) == 7
    removed = tree.remove_node("peer-1/sub-1", recursive=True)
    assert removed == 3
    assert len(tree) == 4
    assert "peer-1/sub-1" not in tree
    assert "peer-1/sub-1/mini-1" not in tree


def test_recursive_removal_updates_the_parents_children():
    tree = _tree()
    tree.remove_node("peer-1/sub-1", recursive=True)
    assert tree["peer-1"].children == ("peer-1/sub-2",)


def test_non_recursive_removal_of_a_parent_is_refused_with_the_child_count():
    tree = _tree()
    with pytest.raises(TreeValidationError, match="it has 2 children"):
        tree.remove_node("peer-1", recursive=False)
    with pytest.raises(TreeValidationError, match="it has 2 children"):
        tree.remove_node("peer-1/sub-1", recursive=False)


def test_non_recursive_removal_of_a_leaf_is_allowed():
    tree = _tree()
    assert tree.remove_node("peer-1/sub-2/mini-1") == 1
    assert "peer-1/sub-2/mini-1" not in tree


def test_removing_a_missing_path_is_rejected():
    tree = _tree()
    with pytest.raises(TreeValidationError, match="no such node"):
        tree.remove_node("peer-1/ghost", recursive=True)


def test_a_pruned_tree_still_validates():
    tree = _tree()
    tree.remove_node("peer-1/sub-1", recursive=True)
    tree.validate()
    assert tree.leaves == ("peer-1/sub-2/mini-1", "peer-1/sub-2/mini-2")


def test_pruning_every_root_leaves_an_empty_valid_tree():
    tree = PeerTree.flat(["peer-a", "peer-b"])
    assert tree.remove_node("peer-a", recursive=True) == 1
    assert tree.remove_node("peer-b", recursive=True) == 1
    tree.validate()
    assert len(tree) == 0


def test_pruning_a_branch_does_not_touch_a_sibling_branch():
    tree = _tree()
    before = tree["peer-1/sub-2"].children
    tree.remove_node("peer-1/sub-1", recursive=True)
    assert tree["peer-1/sub-2"].children == before


def test_the_topology_after_pruning_round_trips():
    tree = _tree()
    tree.remove_node("peer-1/sub-1", recursive=True)
    restored = PeerTree.from_dict(tree.to_dict())
    assert restored.state_dict() == tree.state_dict()


def test_a_pruned_state_dict_has_no_dangling_parent_reference():
    tree = _tree()
    tree.remove_node("peer-1/sub-1", recursive=True)
    paths = {node["path"] for node in tree.state_dict()["nodes"]}
    for node in tree.state_dict()["nodes"]:
        assert node["parent"] is None or node["parent"] in paths
