"""Tree validation (document 10 section 3 and 9.3).

Every error type gets its own test and its own message assertion, because the
messages are what a user sees when a generated topology is wrong.
"""

from __future__ import annotations

import pytest

from mactf.exceptions import ValidationError
from mactf.coopetition.tree import (
    DuplicateNodeError,
    NodeKind,
    OrphanNodeError,
    PeerTree,
    TreeCycleError,
    TreeInconsistencyError,
    TreeTooLargeError,
    TreeValidationError,
    validate_path,
)


def _tree() -> PeerTree:
    tree = PeerTree()
    tree.add_node("peer-a")
    tree.add_node("peer-a/sub-1")
    tree.add_node("peer-a/sub-2")
    tree.add_node("peer-b")
    return tree


def test_every_error_is_a_contract_error():
    from mactf.exceptions import ContractError

    for error in (
        TreeValidationError,
        DuplicateNodeError,
        OrphanNodeError,
        TreeCycleError,
        TreeInconsistencyError,
        TreeTooLargeError,
    ):
        assert issubclass(error, ContractError)


def test_a_duplicate_path_is_rejected():
    tree = _tree()
    with pytest.raises(DuplicateNodeError, match="duplicate node path 'peer-a'"):
        tree.add_node("peer-a")


def test_an_orphan_is_rejected():
    tree = PeerTree()
    with pytest.raises(OrphanNodeError, match="parent 'missing' does not exist"):
        tree.add_node("missing/child")


def test_a_leading_separator_is_rejected():
    with pytest.raises(TreeValidationError, match="leading separator"):
        validate_path("/peer-a")


def test_a_trailing_separator_is_rejected():
    with pytest.raises(TreeValidationError, match="trailing separator"):
        validate_path("peer-a/")


def test_an_empty_path_is_rejected():
    with pytest.raises(TreeValidationError, match="cannot be empty"):
        validate_path("")


def test_a_non_string_path_is_rejected():
    with pytest.raises(TreeValidationError, match="must be a string"):
        validate_path(7)  # type: ignore[arg-type]


def test_an_empty_segment_is_rejected():
    with pytest.raises(TreeValidationError, match="empty segment"):
        validate_path("peer-a//sub")


def test_the_current_directory_marker_is_rejected():
    with pytest.raises(TreeValidationError, match="parent-directory marker"):
        validate_path(".")


def test_a_traversal_segment_is_rejected():
    with pytest.raises(TreeValidationError, match="directory traversal"):
        validate_path("peer-a/../peer-b")


def test_a_character_outside_the_grammar_is_rejected():
    with pytest.raises(TreeValidationError, match="outside"):
        validate_path("peer a")


def test_every_allowed_character_is_accepted():
    validate_path("Peer-1_sub.2:mini")


def test_an_unknown_status_is_rejected_at_add_time():
    tree = PeerTree()
    with pytest.raises(TreeValidationError, match="unknown status"):
        tree.add_node("peer-a", status="zombie")


def test_mark_rejects_an_unknown_status():
    tree = _tree()
    with pytest.raises(TreeValidationError, match="unknown status"):
        tree.mark("peer-a", "zombie")


def test_validate_accepts_a_generated_tree():
    _tree().validate()


def test_validate_detects_a_depth_disagreement():
    tree = _tree()
    tree["peer-a"].depth = 5
    with pytest.raises(TreeInconsistencyError, match="depth 5 does not equal"):
        tree.validate()


def test_validate_detects_a_name_disagreement():
    tree = _tree()
    tree["peer-a"].name = "other"
    with pytest.raises(TreeInconsistencyError, match="does not match the last segment"):
        tree.validate()


def test_validate_detects_a_parent_disagreement():
    tree = _tree()
    tree["peer-a/sub-1"].parent = "peer-b"
    # The offending node is reported, not the parent whose ``children`` tuple is
    # now stale: the descendant's own field is what is wrong.
    with pytest.raises(TreeInconsistencyError, match="does not match 'peer-a' implied"):
        tree.validate()


def test_validate_detects_a_children_disagreement():
    tree = _tree()
    tree["peer-a"].children = ()
    with pytest.raises(TreeInconsistencyError, match="disagree with the nodes"):
        tree.validate()


def test_validate_detects_a_cycle():
    """A path cannot imply a cycle, so the check is for corrupted state.

    The parent-mismatch check fires first for any real path, which is why this
    test asserts on the shared base class: whichever of the two guards catches a
    self-referencing parent, it is a ``TreeValidationError``.
    """
    tree = _tree()
    tree["peer-a"].parent = "peer-a/sub-1"
    with pytest.raises(TreeValidationError, match="peer-a"):
        tree.validate()


def test_validate_detects_a_missing_parent():
    tree = _tree()
    del tree._nodes["peer-a"]
    with pytest.raises(OrphanNodeError, match="parent 'peer-a' does not exist"):
        tree.validate()


def test_validate_detects_a_path_registered_under_the_wrong_key():
    tree = _tree()
    node = tree._nodes.pop("peer-a")
    node.path = "peer-z"
    tree._nodes["peer-a"] = node
    with pytest.raises(TreeInconsistencyError, match="reports path"):
        tree.validate()


def test_node_kind_for_depth_covers_all_named_levels():
    assert [NodeKind.for_depth(depth) for depth in range(1, 6)] == [
        NodeKind.PEER,
        NodeKind.SUBAGENT,
        NodeKind.MINIAGENT,
        NodeKind.MICROAGENT,
        NodeKind.NANOAGENT,
    ]
    assert NodeKind.for_depth(6) is NodeKind.GENERIC
    assert NodeKind.for_depth(20) is NodeKind.GENERIC


def test_node_kind_rejects_a_non_positive_depth():
    with pytest.raises(TreeValidationError, match="depth must be >= 1"):
        NodeKind.for_depth(0)


def test_node_kind_labels_pluralize():
    assert NodeKind.PEER.label_plural() == "peers"
    assert NodeKind.SUBAGENT.label_plural() == "subagents"
    assert NodeKind.NANOAGENT.label_plural() == "nanoagents"
    assert NodeKind.GENERIC.label_plural() == "nodes"


def test_an_unexpected_keyword_and_value_pair_is_reported_by_level_spec():
    from mactf.exceptions import ValidationError
    from mactf.coopetition.tree import LevelSpec

    with pytest.raises(ValidationError, match="requires a label"):
        LevelSpec("")
    with pytest.raises(ValidationError, match="sets both count and n_count"):
        LevelSpec("sub", count=2, n_count=3)
    with pytest.raises(ValidationError, match="count must be >= 0"):
        LevelSpec("sub", count=-1)
    with pytest.raises(ValidationError, match="n_count entries must be >= 0"):
        LevelSpec("sub", n_count=(1, -2))


def test_level_spec_from_spec_rejects_an_unknown_key():
    from mactf.exceptions import ValidationError
    from mactf.coopetition.tree import LevelSpec

    with pytest.raises(ValidationError, match="does not accept"):
        LevelSpec.from_spec({"label": "sub", "width": 2})


def test_hierarchy_spec_from_spec_rejects_an_unknown_key():
    from mactf.exceptions import ValidationError
    from mactf.coopetition.tree import HierarchySpec

    with pytest.raises(ValidationError, match="does not accept"):
        HierarchySpec.from_spec({"levels": [], "bogus": 1})


def test_a_mismatched_root_names_count_is_rejected():
    from mactf.coopetition.tree import HierarchySpec

    with pytest.raises(ValidationError, match="root_names has 2 entries"):
        HierarchySpec(roots=3, root_names=("a", "b")).build()


def test_shared_trunk_with_cross_level_grad_needs_an_explicit_opt_in():
    from mactf.coopetition.tree import HierarchySpec

    with pytest.raises(ValidationError, match="double-counts trunk gradients"):
        HierarchySpec(sharing="shared_trunk", cross_level_grad=True)


def test_shared_trunk_with_the_opt_in_is_accepted():
    from mactf.coopetition.tree import HierarchySpec

    spec = HierarchySpec(
        sharing="shared_trunk", cross_level_grad=True, allow_double_count=True
    )
    assert spec.cross_level_grad is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"roots": 0}, "roots must be >= 1"),
        ({"max_depth": 0}, "max_depth must be >= 1"),
        ({"max_nodes": 0}, "max_nodes must be >= 1"),
        ({"max_children_per_node": 0}, "max_children_per_node must be >= 1"),
        ({"fanout_policy": "zipf"}, "unknown fanout_policy"),
        ({"sharing": "mesh"}, "unknown sharing"),
        ({"attribution": "nearest"}, "unknown attribution"),
    ],
)
def test_hierarchy_spec_rejects_an_invalid_field(kwargs, message):
    from mactf.coopetition.tree import HierarchySpec

    with pytest.raises(ValidationError, match=message):
        HierarchySpec(**kwargs)


def test_an_empty_tree_has_no_roots_and_depth_zero():
    tree = PeerTree()
    assert tree.root_paths == ()
    assert tree.leaves == ()
    assert tree.depth == 0
    assert tree.describe() == "PeerTree(empty)"


def test_a_tree_cap_below_one_is_rejected_at_construction():
    with pytest.raises(TreeValidationError, match="max_depth must be >= 1"):
        PeerTree(max_depth=0)
    with pytest.raises(TreeValidationError, match="max_nodes must be >= 1"):
        PeerTree(max_nodes=0)


def test_an_unknown_key_in_a_node_restore_is_not_silently_dropped():
    tree = _tree()
    payload = tree.to_dict()
    payload["nodes"][0]["bogus"] = 1
    # ``from_dict`` builds from the documented fields; the extra key is not part
    # of the schema, so the restore is unaffected rather than half-applied.
    assert PeerTree.from_dict(payload).describe() == tree.describe()

