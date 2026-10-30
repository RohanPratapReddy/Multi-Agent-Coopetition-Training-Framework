"""Delegation policies (document 10 section 4.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from mactf.coopetition.delegation import (
    BroadcastDelegation,
    ConfidenceDelegation,
    DELEGATION_TYPES,
    ExplicitDelegation,
    RouteByRoleDelegation,
    SelectKDelegation,
    build_delegation,
    delegation_capabilities,
    validate_selection,
)
from mactf.coopetition.tree import HierarchySpec, LevelSpec, PeerTree
from mactf.exceptions import ContractError, ValidationError


@dataclass
class FakeTask:
    id: str = "task-1"
    metadata: dict[str, Any] = field(default_factory=dict)


def _tree():
    return HierarchySpec(levels=(LevelSpec("sub", 3),)).build()


ROOT = "peer-1"
CHILDREN = ("peer-1/sub-1", "peer-1/sub-2", "peer-1/sub-3")


def test_broadcast_selects_every_child_in_path_order():
    tree = _tree()
    selected = BroadcastDelegation().select(tree[ROOT], tree=tree)
    assert selected == CHILDREN


def test_broadcast_on_a_leaf_selects_nothing():
    tree = _tree()
    assert BroadcastDelegation().select(tree["peer-1/sub-1"], tree=tree) == ()


def test_select_k_ranks_by_weight_then_path():
    tree = _tree()
    weights = {
        "peer-1/sub-1": 0.1,
        "peer-1/sub-2": 0.5,
        "peer-1/sub-3": 0.4,
    }
    selected = SelectKDelegation(k=2).select(tree[ROOT], tree=tree, weights=weights)
    assert selected == ("peer-1/sub-2", "peer-1/sub-3")


def test_select_k_breaks_ties_by_path():
    tree = _tree()
    selected = SelectKDelegation(k=2).select(
        tree[ROOT], tree=tree, weights={path: 1.0 for path in CHILDREN}
    )
    assert selected == ("peer-1/sub-1", "peer-1/sub-2")


def test_select_k_rejects_a_bad_k():
    with pytest.raises(ValidationError, match="k >= 1"):
        SelectKDelegation(k=0)


def test_route_by_role_picks_the_mapped_child():
    tree = _tree()
    policy = RouteByRoleDelegation({"review": "sub-2"})
    task = FakeTask(metadata={"role": "review"})
    assert policy.select(tree[ROOT], task, tree=tree) == ("peer-1/sub-2",)


def test_route_by_role_accepts_a_full_path_mapping():
    tree = _tree()
    policy = RouteByRoleDelegation({"review": "peer-1/sub-3"})
    task = FakeTask(metadata={"role": "review"})
    assert policy.select(tree[ROOT], task, tree=tree) == ("peer-1/sub-3",)


def test_route_by_role_raises_for_an_unmapped_role():
    tree = _tree()
    policy = RouteByRoleDelegation({"review": "sub-2"})
    task = FakeTask(metadata={"role": "legal"})
    with pytest.raises(ContractError, match="no child mapped for role 'legal'"):
        policy.select(tree[ROOT], task, tree=tree)


def test_route_by_role_raises_for_a_mapping_to_a_non_child():
    tree = _tree()
    policy = RouteByRoleDelegation({"review": "sub-9"})
    task = FakeTask(metadata={"role": "review"})
    with pytest.raises(ContractError, match="not a child"):
        policy.select(tree[ROOT], task, tree=tree)


def test_route_by_role_broadcasts_when_the_task_has_no_role():
    tree = _tree()
    policy = RouteByRoleDelegation({"review": "sub-2"})
    assert policy.select(tree[ROOT], FakeTask(), tree=tree) == CHILDREN


def test_confidence_selects_every_child_above_the_threshold():
    tree = _tree()
    scores = {"peer-1/sub-1": 0.9, "peer-1/sub-2": 0.7, "peer-1/sub-3": 0.1}
    selected = ConfidenceDelegation(threshold=0.5).select(
        tree[ROOT], tree=tree, scores=scores
    )
    assert selected == ("peer-1/sub-1", "peer-1/sub-2")


def test_confidence_falls_back_to_the_best_child_so_a_subtree_never_starves():
    tree = _tree()
    scores = {"peer-1/sub-1": 0.2, "peer-1/sub-2": 0.4, "peer-1/sub-3": 0.3}
    selected = ConfidenceDelegation(threshold=0.9).select(
        tree[ROOT], tree=tree, scores=scores
    )
    assert selected == ("peer-1/sub-2",)


def test_confidence_on_a_leaf_selects_nothing():
    tree = _tree()
    assert ConfidenceDelegation().select(tree["peer-1/sub-1"], tree=tree) == ()


def test_explicit_delegation_calls_the_function():
    tree = _tree()
    seen: dict[str, Any] = {}

    def choose(node, task, *, tree, scores, weights, epoch):
        seen["node"] = node.path
        seen["epoch"] = epoch
        return ["peer-1/sub-2"]

    selected = ExplicitDelegation(choose).select(tree[ROOT], tree=tree, epoch=4)
    assert selected == ("peer-1/sub-2",)
    assert seen == {"node": ROOT, "epoch": 4}


def test_explicit_delegation_requires_a_callable():
    with pytest.raises(ValidationError, match="requires an 'fn'"):
        ExplicitDelegation()


def test_a_non_child_selection_names_both_paths():
    tree = _tree()
    with pytest.raises(ContractError) as info:
        validate_selection(tree[ROOT], ("peer-1/sub-9",), tree=tree)
    message = str(info.value)
    assert "peer-1/sub-9" in message and "'peer-1'" in message
    assert "not in the tree" in message


def test_a_selection_from_another_branch_is_rejected():
    tree = HierarchySpec(levels=(LevelSpec("sub", 2),), roots=2).build()
    with pytest.raises(ContractError, match="whose parent is"):
        validate_selection(tree["peer-1"], ("peer-2/sub-1",), tree=tree)


def test_a_non_leaf_selecting_nothing_is_rejected():
    tree = _tree()
    with pytest.raises(ContractError, match="cannot delegate nothing"):
        validate_selection(tree[ROOT], (), tree=tree)


def test_a_selection_is_deduplicated_and_order_stable():
    tree = _tree()
    selected = validate_selection(
        tree[ROOT], ("peer-1/sub-3", "peer-1/sub-1", "peer-1/sub-3"), tree=tree
    )
    assert selected == ("peer-1/sub-3", "peer-1/sub-1")


def test_a_failed_child_cannot_be_selected():
    tree = _tree()
    tree.mark("peer-1/sub-2", "failed")
    with pytest.raises(ContractError, match="failed this epoch"):
        validate_selection(tree[ROOT], ("peer-1/sub-2",), tree=tree)


def test_registry_builds_every_policy():
    assert build_delegation(None).name == "broadcast"
    assert build_delegation("confidence").name == "confidence"
    assert build_delegation({"type": "select_k", "k": 2}).k == 2
    assert isinstance(
        build_delegation({"type": "explicit", "fn": lambda *args, **kwargs: ()}),
        ExplicitDelegation,
    )


def test_registry_rejects_an_unknown_type_listing_the_known_ones():
    with pytest.raises(ValidationError, match="unknown delegation type") as info:
        build_delegation("round_robin")
    assert "broadcast" in str(info.value)


def test_registry_rejects_a_missing_type():
    with pytest.raises(ValidationError, match="requires a 'type' key"):
        build_delegation({})


def test_registry_rejects_an_unknown_kwarg():
    with pytest.raises(ValidationError, match="does not accept"):
        build_delegation({"type": "select_k", "k": 2, "kmax": 3})


def test_capabilities_lists_every_policy_with_its_kwargs():
    entries = {entry["type"]: entry for entry in delegation_capabilities()}
    assert set(entries) == set(DELEGATION_TYPES)
    assert entries["select_k"]["kwargs"] == ["k"]
    assert entries["broadcast"]["kwargs"] == []


def test_describe_is_serializable():
    payload = SelectKDelegation(k=3).describe()
    assert payload["type"] == "select_k"
    assert payload["k"] == 3


def test_starvation_is_observable_from_the_selection():
    """A child never chosen must be computable from the tree and selection."""
    tree = _tree()
    selected = SelectKDelegation(k=1).select(
        tree[ROOT], tree=tree, weights={path: 1.0 for path in CHILDREN}
    )
    starved = tuple(child for child in tree.children(ROOT) if child not in selected)
    assert starved == ("peer-1/sub-2", "peer-1/sub-3")
