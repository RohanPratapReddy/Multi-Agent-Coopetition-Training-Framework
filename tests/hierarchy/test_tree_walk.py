"""Tree traversal (document 10 section 4).

Four orders, all lexicographic, none dependent on insertion order. Ordering is
load-bearing: scoring needs children before parents, aggregation needs whole
levels deepest-first, and the API lists nodes breadth-first.
"""

from __future__ import annotations

import pytest

from mactf.coopetition.tree import (
    PeerTree,
    TreeValidationError,
    aggregation_groups,
)


def _tree() -> PeerTree:
    tree = PeerTree()
    for path in ("peer-b", "peer-a", "peer-a/sub-2", "peer-a/sub-1", "peer-a/sub-1/mini-1"):
        tree.add_node(path)
    return tree


def test_breadth_order_visits_whole_levels():
    tree = _tree()
    assert [node.path for node in tree.walk("breadth")] == [
        "peer-a",
        "peer-b",
        "peer-a/sub-1",
        "peer-a/sub-2",
        "peer-a/sub-1/mini-1",
    ]


def test_root_first_visits_parents_before_children():
    tree = _tree()
    assert [node.path for node in tree.walk("root-first")] == [
        "peer-a",
        "peer-b",
        "peer-a/sub-1",
        "peer-a/sub-2",
        "peer-a/sub-1/mini-1",
    ]


def test_leaves_first_visits_children_before_parents():
    tree = _tree()
    order = [node.path for node in tree.walk("leaves-first")]
    assert order.index("peer-a/sub-1/mini-1") < order.index("peer-a/sub-1")
    assert order.index("peer-a/sub-1") < order.index("peer-a")
    assert order.index("peer-b") < len(order)


def test_depth_order_goes_deepest_level_first():
    tree = _tree()
    assert [node.path for node in tree.walk("depth")] == [
        "peer-a/sub-1/mini-1",
        "peer-a/sub-1",
        "peer-a/sub-2",
        "peer-a",
        "peer-b",
    ]


def test_iteration_is_breadth():
    tree = _tree()
    assert [node.path for node in tree] == [node.path for node in tree.walk("breadth")]


def test_every_order_contains_every_node_exactly_once():
    tree = _tree()
    expected = set(tree.to_dict()["nodes"][i]["path"] for i in range(len(tree.to_dict()["nodes"])))
    for order in ("breadth", "root-first", "leaves-first", "depth"):
        paths = [node.path for node in tree.walk(order)]
        assert len(paths) == len(expected)
        assert set(paths) == expected


def test_order_does_not_depend_on_insertion_order():
    forward = PeerTree()
    backward = PeerTree()
    # Parents must exist before their children, but sibling order is free, which
    # is the only insertion-order variable the tree permits.
    forward_order = ["peer-a", "peer-b", "peer-a/sub-1", "peer-a/sub-2", "peer-a/sub-1/mini-1"]
    backward_order = ["peer-b", "peer-a", "peer-a/sub-2", "peer-a/sub-1", "peer-a/sub-1/mini-1"]
    for path in forward_order:
        forward.add_node(path)
    for path in backward_order:
        backward.add_node(path)
    for order in ("breadth", "root-first", "leaves-first", "depth"):
        assert [node.path for node in forward.walk(order)] == [
            node.path for node in backward.walk(order)
        ]


def test_an_unknown_order_is_rejected():
    with pytest.raises(TreeValidationError, match="unknown walk order"):
        list(_tree().walk("sideways"))


def test_depth_limits_can_be_applied_by_a_caller_filtering_the_walk():
    tree = _tree()
    shallow = [node.path for node in tree.walk("breadth") if node.depth <= 2]
    assert "peer-a/sub-1/mini-1" not in shallow
    assert "peer-a/sub-1" in shallow


def test_subtree_rebases_and_keeps_descendants():
    tree = _tree()
    sub = tree.subtree("peer-a/sub-1")
    assert set(sub.root_paths) == {"sub-1"}
    assert "sub-1/mini-1" in sub
    assert sub.depth == 2


def test_ancestors_are_nearest_first():
    tree = _tree()
    assert tree.ancestors("peer-a/sub-1/mini-1") == ("peer-a/sub-1", "peer-a")
    assert tree.ancestors("peer-a") == ()


def test_descendants_are_parents_before_children():
    tree = _tree()
    assert tree.descendants("peer-a") == ("peer-a/sub-1", "peer-a/sub-2", "peer-a/sub-1/mini-1")


def test_siblings_exclude_the_node_itself():
    tree = _tree()
    assert tree.siblings("peer-a/sub-1") == ("peer-a/sub-2",)
    assert tree.siblings("peer-a") == ("peer-b",)


def test_leaves_and_levels():
    tree = _tree()
    assert tree.leaves == ("peer-a/sub-1/mini-1", "peer-a/sub-2", "peer-b")
    assert tree.level(1) == ("peer-a", "peer-b")
    assert tree.level(2) == ("peer-a/sub-1", "peer-a/sub-2")
    assert tree.level(9) == ()


def test_arity_reports_the_child_count():
    tree = _tree()
    assert tree.arity("peer-a") == 2
    assert tree.arity("peer-b") == 0


def test_aggregation_groups_are_root_first_then_each_parent():
    tree = _tree()
    groups = aggregation_groups(tree)
    assert groups[0] == (None, ("peer-a", "peer-b"))
    assert ("peer-a", ("peer-a/sub-1", "peer-a/sub-2")) in groups
    assert ("peer-a/sub-1", ("peer-a/sub-1/mini-1",)) in groups
    # A leaf contributes no group: it has no children to average.
    assert all(parent != "peer-b" for parent, _ in groups)


def test_aggregation_groups_for_a_flat_tree_are_one_group():
    tree = PeerTree.flat(["peer-a", "peer-b", "peer-c"])
    assert aggregation_groups(tree) == [(None, ("peer-a", "peer-b", "peer-c"))]


def test_reverse_group_order_is_deepest_first():
    tree = _tree()
    groups = aggregation_groups(tree)
    reversed_groups = list(reversed(groups))
    assert reversed_groups[-1][0] is None
    for parent, children in reversed_groups:
        if parent is None:
            continue
        for child in children:
            # Every child's own group (if any) was already aggregated.
            assert all(entry[0] != child for entry in reversed_groups[reversed_groups.index((parent, children)):])
