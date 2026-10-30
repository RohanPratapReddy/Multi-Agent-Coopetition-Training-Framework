"""Caps (document 10 section 10).

The caps are defaults, not suggestions: exceeding one raises at build time and
names both the offending path and the configured limit, so a user reading the
traceback knows which knob to turn.
"""

from __future__ import annotations

import pytest

from mactf.coopetition.tree import (
    HierarchySpec,
    LevelSpec,
    PeerTree,
    TreeTooLargeError,
)
from mactf.exceptions import ValidationError


def test_the_documented_defaults():
    spec = HierarchySpec()
    assert spec.max_depth == 8
    assert spec.max_nodes == 256
    assert spec.max_children_per_node == 16


def test_the_default_caps_are_configurable():
    spec = HierarchySpec(max_depth=3, max_nodes=10, max_children_per_node=2)
    assert (spec.max_depth, spec.max_nodes, spec.max_children_per_node) == (3, 10, 2)


@pytest.mark.parametrize("field", ["max_depth", "max_nodes", "max_children_per_node"])
@pytest.mark.parametrize("value", [0, -1])
def test_a_non_positive_cap_is_rejected(field, value):
    with pytest.raises(ValidationError, match=field):
        HierarchySpec(**{field: value})


def test_a_depth_cap_raises_at_build_with_the_offending_level():
    spec = HierarchySpec(
        levels=(LevelSpec("sub", 2), LevelSpec("mini", 2)),
        max_depth=3,
    )
    # Tightening the cap after construction is the state a caller reaches by
    # building a legal spec and then clamping it (as a settings override would).
    object.__setattr__(spec, "max_depth", 2)
    with pytest.raises(TreeTooLargeError) as info:
        spec.build()
    message = str(info.value)
    assert "would be at depth 3" in message
    assert "max_depth=2" in message


def test_a_depth_cap_is_checked_before_the_level_is_built():
    spec = HierarchySpec(levels=(LevelSpec("sub", 2),), max_depth=2)
    object.__setattr__(spec, "max_depth", 1)
    with pytest.raises(TreeTooLargeError) as info:
        spec.build()
    assert "max_depth=1" in str(info.value)


def test_a_node_cap_raises_at_build_with_the_offending_path():
    spec = HierarchySpec(levels=(LevelSpec("sub", 3),), max_nodes=3)
    with pytest.raises(TreeTooLargeError) as info:
        spec.build()
    message = str(info.value)
    assert "would exceed max_nodes=3" in message
    # The path that broke the cap is named, not just the count.
    assert "peer-1/sub-" in message


def test_an_arity_cap_raises_at_build_with_the_offending_parent():
    spec = HierarchySpec(levels=(LevelSpec("sub", 5),), max_children_per_node=4)
    with pytest.raises(TreeTooLargeError) as info:
        spec.build()
    message = str(info.value)
    assert "peer-1" in message
    assert "5 children" in message
    assert "max_children_per_node=4" in message


def test_an_arity_cap_is_reported_per_parent_not_globally():
    spec = HierarchySpec(levels=(LevelSpec("sub", 1), LevelSpec("mini", 3)), max_children_per_node=2)
    with pytest.raises(TreeTooLargeError, match=r"peer-1/sub-1: 3 children"):
        spec.build()


def test_the_node_count_prediction_is_exact_at_the_boundary():
    spec = HierarchySpec(levels=(LevelSpec("sub", 2),))
    assert spec.node_count() == 3
    exact = HierarchySpec(levels=(LevelSpec("sub", 2),), max_nodes=3)
    assert len(exact.build()) == 3


def test_the_node_count_prediction_matches_a_built_tree():
    spec = HierarchySpec(levels=(LevelSpec("sub", 2), LevelSpec("mini", 3)), roots=2)
    assert len(spec.build()) == spec.node_count()


def test_a_flat_spec_respects_the_node_cap():
    spec = HierarchySpec.flat(["a", "b", "c"], max_nodes=2)
    with pytest.raises(TreeTooLargeError, match="max_nodes=2"):
        spec.build()


def test_a_direct_tree_respects_every_cap():
    tree = PeerTree(max_depth=2, max_nodes=2)
    tree.add_node("a")
    tree.add_node("a/b")
    with pytest.raises(TreeTooLargeError, match="max_nodes=2"):
        tree.add_node("c")


def test_validate_catches_a_raised_cap_after_construction(monkeypatch):
    tree = PeerTree.flat(["a"])
    tree.max_nodes = 0
    with pytest.raises(TreeTooLargeError, match="tree has 1 nodes"):
        tree.validate()


def test_a_cap_of_one_still_builds_the_root():
    spec = HierarchySpec(levels=(), roots=1, max_nodes=1)
    assert len(spec.build()) == 1
