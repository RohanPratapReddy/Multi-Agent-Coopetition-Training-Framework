"""HierarchySpec construction: uniform, random, and ``n_count`` arity (document 10 §3.4)."""

from __future__ import annotations

import pytest

from mactf.coopetition.tree import (
    HierarchySpec,
    LevelSpec,
    TreeTooLargeError,
    TreeValidationError,
)
from mactf.exceptions import ValidationError


def test_uniform_fanout_builds_the_expected_shape():
    spec = HierarchySpec(levels=(LevelSpec("sub", 2), LevelSpec("mini", 3)))
    tree = spec.build()
    # 1 root + 2 children + 6 grandchildren
    assert len(tree) == 9
    assert tree.depth == 3
    assert tree.arity("peer-1") == 2
    assert tree.arity("peer-1/sub-1") == 3


def test_multiple_roots_are_numbered():
    spec = HierarchySpec(levels=(LevelSpec("sub", 2),), roots=3)
    tree = spec.build()
    assert tree.root_paths == ("peer-1", "peer-2", "peer-3")
    assert len(tree) == 9


def test_zero_arity_stops_the_branch():
    spec = HierarchySpec(levels=(LevelSpec("sub", 0), LevelSpec("mini", 2)))
    tree = spec.build()
    assert tree.root_paths == ("peer-1",)
    assert len(tree) == 1


def test_n_count_as_an_int_gives_every_parent_that_many_children():
    spec = HierarchySpec(levels=(LevelSpec("sub", n_count=3),))
    tree = spec.build()
    assert tree.arity("peer-1") == 3


def test_n_count_as_a_tuple_gives_each_parent_its_own_arity():
    spec = HierarchySpec(levels=(LevelSpec("sub", n_count=(1, 4, 2)),), roots=3)
    tree = spec.build()
    assert tree.arity("peer-1") == 1
    assert tree.arity("peer-2") == 4
    assert tree.arity("peer-3") == 2


def test_count_and_n_count_together_are_rejected():
    with pytest.raises(ValidationError, match="both count and n_count"):
        LevelSpec("sub", count=2, n_count=3)


def test_a_negative_count_is_rejected():
    with pytest.raises(ValidationError, match="count must be"):
        LevelSpec("sub", count=-1)
    with pytest.raises(ValidationError, match="n_count"):
        LevelSpec("sub", n_count=(-1, 2))


def test_a_level_needs_a_label():
    with pytest.raises(ValidationError, match="label"):
        LevelSpec("")


def test_random_fanout_is_seeded_and_reproducible():
    spec = HierarchySpec(
        levels=(LevelSpec("sub", 4),), fanout_policy="random", seed=11
    )
    first = spec.build()
    second = spec.build()
    assert first.describe() == second.describe()


def test_random_fanout_differs_across_seeds():
    def shape(seed: int) -> tuple[int, ...]:
        spec = HierarchySpec(
            levels=(LevelSpec("sub", 4), LevelSpec("mini", 4)),
            roots=3,
            fanout_policy="random",
            seed=seed,
        )
        tree = spec.build()
        return tuple(tree.arity(node.path) for node in tree.walk("root-first"))

    # With four possible arities and nine parents, one collision is possible but
    # three identical draws are not evidence of a working seed.
    assert len({shape(seed) for seed in (0, 1, 2)}) > 1


def test_random_fanout_respects_the_arity_cap():
    spec = HierarchySpec(
        levels=(LevelSpec("sub", 6),),
        roots=4,
        max_children_per_node=6,
        fanout_policy="random",
        seed=3,
    )
    tree = spec.build()
    for root in tree.root_paths:
        assert tree.arity(root) <= 6


def test_an_unknown_fanout_policy_is_rejected():
    with pytest.raises(ValidationError, match="unknown fanout_policy"):
        HierarchySpec(fanout_policy="pareto")


def test_an_unknown_sharing_mode_is_rejected():
    with pytest.raises(ValidationError, match="unknown sharing"):
        HierarchySpec(sharing="shared_everything")


def test_an_unknown_attribution_is_rejected():
    with pytest.raises(ValidationError, match="unknown attribution"):
        HierarchySpec(attribution="none")


def test_cross_level_grad_with_shared_trunk_is_rejected_without_the_opt_in():
    with pytest.raises(ValidationError, match="double-count"):
        HierarchySpec(cross_level_grad=True, sharing="shared_trunk")
    # The opt-in makes it legal, with the cost stated in the message.
    spec = HierarchySpec(
        cross_level_grad=True, sharing="shared_trunk", allow_double_count=True
    )
    assert spec.cross_level_grad is True


def test_levels_beyond_max_depth_are_rejected_at_spec_time():
    with pytest.raises(ValidationError, match="exceeding max_depth"):
        HierarchySpec(levels=(LevelSpec("sub", 1),) * 4, max_depth=3)


def test_a_level_beyond_max_depth_raises_at_build_with_the_level():
    """The build-time guard, independent of the spec-time check.

    ``__post_init__`` rejects too many levels, so the build guard is exercised by
    constructing a legal spec and then tightening ``max_depth`` — the same state a
    spec restored from an older checkpoint could reach.
    """
    tight = HierarchySpec(levels=(LevelSpec("sub", 2),) * 4, max_depth=8)
    object.__setattr__(tight, "max_depth", 2)
    with pytest.raises(TreeTooLargeError, match="max_depth"):
        tight.build()


def test_the_arity_cap_raises_at_build_with_the_parent_path():
    spec = HierarchySpec(
        levels=(LevelSpec("sub", 20),), max_children_per_node=5
    )
    with pytest.raises(TreeTooLargeError, match="peer-1") as info:
        spec.build()
    assert "max_children_per_node" in str(info.value)


def test_node_count_projection_matches_the_built_tree():
    spec = HierarchySpec(levels=(LevelSpec("sub", 3), LevelSpec("mini", 2)))
    assert spec.node_count() == len(spec.build())


def test_root_names_override_the_generated_names():
    spec = HierarchySpec(
        levels=(LevelSpec("sub", 1),), roots=2, root_names=("alpha", "beta")
    )
    tree = spec.build()
    assert tree.root_paths == ("alpha", "beta")
    assert "alpha/sub-1" in tree


def test_a_root_names_length_mismatch_is_rejected():
    spec = HierarchySpec(roots=2, root_names=("alpha",))
    with pytest.raises(ValidationError, match="root_names"):
        spec.build()


def test_a_peer_factory_is_called_once_per_node():
    calls: list[tuple[str, int]] = []
    spec = HierarchySpec(levels=(LevelSpec("sub", 2),))
    spec.build(peer_factory=lambda path, depth, level: calls.append((path, depth)))
    assert calls == [("peer-1", 1), ("peer-1/sub-1", 2), ("peer-1/sub-2", 2)]


def test_spec_round_trips_through_spec_mappings():
    spec = HierarchySpec(
        levels=(LevelSpec("sub", 2), LevelSpec("mini", 1)),
        roots=2,
        max_depth=6,
        seed=5,
        attribution="own",
    )
    restored = HierarchySpec.from_spec(spec.to_spec())
    assert restored == spec


def test_level_spec_round_trips_with_n_count():
    level = LevelSpec("sub", n_count=(1, 3), roles=("red", "blue"))
    restored = LevelSpec.from_spec(level.to_spec())
    assert restored == level


def test_from_spec_rejects_an_unknown_key():
    with pytest.raises(ValidationError, match="does not accept"):
        HierarchySpec.from_spec({"levels": [], "branching": 3})


def test_a_level_spec_from_spec_needs_a_label():
    with pytest.raises(ValidationError, match="label"):
        LevelSpec.from_spec({"count": 2})


def test_roles_cycle_across_children():
    spec = HierarchySpec(levels=(LevelSpec("sub", 3, roles=("red", "blue")),))
    tree = spec.build()
    roles = [tree[path].role for path in tree.children("peer-1")]
    assert roles == ["red", "blue", "red"]
