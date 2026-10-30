"""Tree construction, validation, navigation, and serialization (document 10 §3)."""

from __future__ import annotations

import pytest


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


from mactf.exceptions import ValidationError


def test_node_kind_maps_depth_and_pluralizes():
    assert NodeKind.for_depth(1) is NodeKind.PEER
    assert NodeKind.for_depth(2) is NodeKind.SUBAGENT
    assert NodeKind.for_depth(3) is NodeKind.MINIAGENT
    assert NodeKind.for_depth(4) is NodeKind.MICROAGENT
    assert NodeKind.for_depth(5) is NodeKind.NANOAGENT
    # Labels are not limits: a seventh level is GENERIC, not an error.
    assert NodeKind.for_depth(6) is NodeKind.GENERIC
    assert NodeKind.for_depth(9) is NodeKind.GENERIC
    assert NodeKind.SUBAGENT.label_plural() == "subagents"
    assert NodeKind.MINIAGENT.label_plural() == "miniagents"
    assert NodeKind.GENERIC.label_plural() == "nodes"
    with pytest.raises(TreeValidationError, match="depth must be"):
        NodeKind.for_depth(0)


def test_flat_tree_reproduces_a_peer_list():
    tree = PeerTree.flat(["peer-a", "peer-b"])
    assert tree.root_paths == ("peer-a", "peer-b")
    assert len(tree) == 2
    assert tree.depth == 1
    # The flat-equivalence gate: these keys are today's PhaseContext.peers keys.
    assert {node.path for node in tree.walk()} == {"peer-a", "peer-b"}
    assert tree["peer-a"].kind is NodeKind.PEER
    assert tree.leaves == ("peer-a", "peer-b")


def test_flat_tree_rejects_duplicates():
    with pytest.raises(DuplicateNodeError):
        PeerTree.flat(["peer-a", "peer-a"])


def test_hierarchy_spec_flat_matches_the_peer_list():
    spec = HierarchySpec.flat(["peer-a", "peer-b"])
    tree = spec.build()
    assert tree.root_paths == ("peer-a", "peer-b")
    assert len(tree) == 2


def test_hierarchy_spec_flat_rejects_an_empty_list():
    with pytest.raises(ValidationError, match="at least one peer"):
        HierarchySpec.flat([])


def test_add_node_sets_kind_depth_and_parent():
    tree = PeerTree.flat(["peer-a"])
    node = tree.add_node("peer-a/sub-1")
    assert node.kind is NodeKind.SUBAGENT
    assert node.depth == 2
    assert node.parent == "peer-a"
    assert tree["peer-a"].children == ("peer-a/sub-1",)


def test_add_node_sorts_children_lexicographically():
    tree = PeerTree.flat(["peer-a"])
    for name in ("sub-3", "sub-1", "sub-2"):
        tree.add_node(f"peer-a/{name}")
    assert tree["peer-a"].children == (
        "peer-a/sub-1",
        "peer-a/sub-2",
        "peer-a/sub-3",
    )


def test_add_node_accepts_a_custom_kind_and_metadata():
    tree = PeerTree()
    node = tree.add_node("peer-a", kind=NodeKind.GENERIC, metadata={"team": "red"})
    assert node.kind is NodeKind.GENERIC
    assert node.metadata == {"team": "red"}


def test_orphan_is_rejected():
    tree = PeerTree()
    with pytest.raises(OrphanNodeError, match="parent"):
        tree.add_node("peer-a/sub-1")


def test_duplicate_is_rejected():
    tree = PeerTree.flat(["peer-a"])
    with pytest.raises(DuplicateNodeError):
        tree.add_node("peer-a")


def test_unknown_status_is_rejected():
    tree = PeerTree()
    with pytest.raises(TreeValidationError, match="unknown status"):
        tree.add_node("peer-a", status="exploded")


def test_depth_cap_is_enforced_at_add_time():
    tree = PeerTree(max_depth=2)
    tree.add_node("peer-a")
    tree.add_node("peer-a/sub-1")
    with pytest.raises(TreeTooLargeError, match="max_depth"):
        tree.add_node("peer-a/sub-1/mini-1")


def test_node_cap_is_enforced_at_add_time():
    tree = PeerTree(max_nodes=2)
    tree.add_node("peer-a")
    tree.add_node("peer-b")
    with pytest.raises(TreeTooLargeError, match="max_nodes"):
        tree.add_node("peer-c")


def test_a_bad_cap_is_rejected_at_construction():
    with pytest.raises(TreeValidationError, match="max_depth"):
        PeerTree(max_depth=0)
    with pytest.raises(TreeValidationError, match="max_nodes"):
        PeerTree(max_nodes=0)


def test_navigation_helpers():
    spec = HierarchySpec(levels=(LevelSpec("sub", 2), LevelSpec("mini", 2)))
    tree = spec.build()
    assert tree.ancestors("peer-1/sub-1/mini-1") == ("peer-1/sub-1", "peer-1")
    assert tree.siblings("peer-1/sub-1/mini-1") == ("peer-1/sub-1/mini-2",)
    assert tree.siblings("peer-1") == ()
    assert tree.arity("peer-1") == 2
    assert tree.level(2) == ("peer-1/sub-1", "peer-1/sub-2")
    assert tree.descendants("peer-1/sub-1") == (
        "peer-1/sub-1/mini-1",
        "peer-1/sub-1/mini-2",
    )
    assert "peer-1/sub-1/mini-1" in tree
    assert tree.get("nope") is None
    with pytest.raises(KeyError):
        tree["nope"]


def test_parent_reports_none_for_a_root():
    tree = PeerTree.flat(["peer-a"])
    assert tree.parent("peer-a") is None


def test_levels_groups_by_depth():
    spec = HierarchySpec(levels=(LevelSpec("sub", 2),))
    levels = spec.build().levels
    assert levels == {1: ("peer-1",), 2: ("peer-1/sub-1", "peer-1/sub-2")}


def test_walk_orders_are_documented_and_stable():
    spec = HierarchySpec(levels=(LevelSpec("sub", 2), LevelSpec("mini", 2)))
    tree = spec.build()

    leaves_first = [node.path for node in tree.walk("leaves-first")]
    assert leaves_first[-1] == "peer-1"
    assert leaves_first.index("peer-1/sub-1/mini-1") < leaves_first.index("peer-1/sub-1")

    root_first = [node.path for node in tree.walk("root-first")]
    assert root_first[0] == "peer-1"
    assert root_first.index("peer-1/sub-1") < root_first.index("peer-1/sub-1/mini-1")

    breadth = [node.path for node in tree.walk("breadth")]
    assert breadth == sorted(breadth, key=lambda path: (path.count("/"), path))

    depth_order = [node.path for node in tree.walk("depth")]
    assert depth_order[0].count("/") == 2
    assert depth_order[-1] == "peer-1"


def test_leaves_first_and_depth_orders_differ():
    spec = HierarchySpec(levels=(LevelSpec("sub", 2), LevelSpec("mini", 2)))
    tree = spec.build()
    assert [n.path for n in tree.walk("leaves-first")] != [
        n.path for n in tree.walk("depth")
    ]


def test_iteration_uses_breadth_order():
    spec = HierarchySpec(levels=(LevelSpec("sub", 2),))
    tree = spec.build()
    assert [node.path for node in tree] == [node.path for node in tree.walk("breadth")]


def test_unknown_walk_order_is_rejected():
    tree = PeerTree.flat(["peer-a"])
    with pytest.raises(TreeValidationError, match="unknown walk order"):
        list(tree.walk("sideways"))


def test_walk_is_reproducible_regardless_of_insertion_order():
    forward = PeerTree()
    for name in ("peer-a", "peer-b"):
        forward.add_node(name)
    reverse = PeerTree()
    for name in ("peer-b", "peer-a"):
        reverse.add_node(name)
    assert [n.path for n in forward.walk("breadth")] == [
        n.path for n in reverse.walk("breadth")
    ]


def test_subtree_rebases_paths():
    spec = HierarchySpec(levels=(LevelSpec("sub", 2), LevelSpec("mini", 1)))
    tree = spec.build()
    sub = tree.subtree("peer-1/sub-1")
    assert sub.root_paths == ("sub-1",)
    assert "sub-1/mini-1" in sub
    assert len(sub) == 2


def test_remove_node_requires_recursion_for_parents():
    spec = HierarchySpec(levels=(LevelSpec("sub", 2),))
    tree = spec.build()
    with pytest.raises(TreeValidationError, match="2 children"):
        tree.remove_node("peer-1")
    removed = tree.remove_node("peer-1", recursive=True)
    # The root plus its two children.
    assert removed == 3
    assert len(tree) == 0


def test_remove_node_of_a_missing_path_is_rejected():
    with pytest.raises(TreeValidationError, match="no such node"):
        PeerTree().remove_node("nope")


def test_remove_leaf_updates_its_parent():
    spec = HierarchySpec(levels=(LevelSpec("sub", 2),))
    tree = spec.build()
    assert tree.remove_node("peer-1/sub-1") == 1
    assert tree["peer-1"].children == ("peer-1/sub-2",)


def test_clone_is_structural_and_keeps_status():
    tree = PeerTree.flat(["peer-a", "peer-b"])
    tree.mark("peer-b", "frozen")
    clone = tree.clone()
    assert clone.describe() == tree.describe()
    assert clone["peer-b"].status == "frozen"


def test_mark_rejects_an_unknown_status():
    tree = PeerTree.flat(["peer-a"])
    with pytest.raises(TreeValidationError, match="unknown status"):
        tree.mark("peer-a", "exploded")


def test_peers_returns_only_nodes_with_a_peer():
    tree = PeerTree.flat(["peer-a", "peer-b"])
    assert tree.peers() == {}


def test_validate_accepts_a_generated_tree():
    spec = HierarchySpec(levels=(LevelSpec("sub", 2), LevelSpec("mini", 2)))
    spec.build().validate()


def test_validate_detects_a_children_inconsistency():
    tree = PeerTree.flat(["peer-a"])
    tree.add_node("peer-a/sub-1")
    tree["peer-a"].children = ()
    with pytest.raises(TreeInconsistencyError, match="children"):
        tree.validate()


def test_validate_detects_a_depth_mismatch():
    tree = PeerTree.flat(["peer-a"])
    tree["peer-a"].depth = 3
    with pytest.raises(TreeInconsistencyError, match="depth"):
        tree.validate()


def test_validate_detects_a_name_mismatch():
    tree = PeerTree.flat(["peer-a"])
    tree["peer-a"].name = "peer-z"
    with pytest.raises(TreeInconsistencyError, match="name"):
        tree.validate()


def test_validate_detects_a_parent_mismatch():
    tree = PeerTree.flat(["peer-a"])
    tree.add_node("peer-a/sub-1")
    tree["peer-a/sub-1"].parent = None
    with pytest.raises(TreeInconsistencyError, match="parent"):
        tree.validate()


def test_validate_detects_a_cycle():
    tree = PeerTree.flat(["peer-a"])
    tree.add_node("peer-a/sub-1")
    tree["peer-a"].parent = "peer-a/sub-1"
    with pytest.raises((TreeInconsistencyError, TreeValidationError)):
        tree.validate()


def test_round_trip_through_to_dict():
    spec = HierarchySpec(levels=(LevelSpec("sub", 2), LevelSpec("mini", 2)))
    tree = spec.build()
    restored = PeerTree.from_dict(tree.to_dict())
    assert restored.describe() == tree.describe()
    assert restored.root_paths == tree.root_paths


def test_state_dict_holds_topology_only():
    tree = PeerTree.flat(["peer-a", "peer-b"])
    payload = tree.state_dict()
    assert set(payload) == {"max_depth", "max_nodes", "nodes"}
    assert payload["nodes"][0] == {
        "path": "peer-a",
        "parent": None,
        "kind": "peer",
        "depth": 1,
        "role": "worker",
        "status": "active",
    }


def test_to_dict_round_trips_status_and_metadata():
    tree = PeerTree.flat(["peer-a"])
    tree.add_node("peer-a/sub-1", metadata={"team": "red"})
    tree.mark("peer-a/sub-1", "frozen")
    restored = PeerTree.from_dict(tree.to_dict())
    assert restored["peer-a/sub-1"].status == "frozen"
    assert restored["peer-a/sub-1"].metadata == {"team": "red"}


def test_from_dict_accepts_a_version_one_peers_payload():
    tree = PeerTree.from_dict({"peers": {"peer-a": {}, "peer-b": {}}})
    assert tree.root_paths == ("peer-a", "peer-b")
    assert len(tree) == 2


def test_from_dict_needs_nodes_or_peers():
    with pytest.raises(TreeValidationError, match="nodes"):
        PeerTree.from_dict({})


def test_from_dict_rejects_a_non_mapping():
    with pytest.raises(TreeValidationError, match="mapping"):
        PeerTree.from_dict([])  # type: ignore[arg-type]


def test_node_seed_is_stable_and_path_sensitive():
    assert node_seed(7, "peer-a", 0) == node_seed(7, "peer-a", 0)
    assert node_seed(7, "peer-a", 0) != node_seed(7, "peer-b", 0)
    assert node_seed(7, "peer-a", 0) != node_seed(7, "peer-a", 1)
    assert 0 <= node_seed(7, "peer-a", 0) < 2**31 - 1


def test_describe_is_readable():
    spec = HierarchySpec(levels=(LevelSpec("sub", 1),))
    text = spec.build().describe()
    assert "PeerTree(2 nodes" in text
    assert "sub-1 (subagent)" in text


def test_describe_marks_a_non_active_status():
    tree = PeerTree.flat(["peer-a"])
    tree.mark("peer-a", "failed")
    assert "[failed]" in tree.describe()
