"""The structural rules of a mutation, with no model (document 17 section 12.5.1, 12.11.5 rows 7-20).

Every test runs the pure ``TopologySimulator`` over a tree built from paths; ``findings`` is the
rendered list a CLI or the API would show.
"""

import dataclasses
import re

import pytest

from mactf.coopetition.tree import PeerTree, TreeCycleError
from mactf.exceptions import ComplianceError, MutationError
from mactf.testing.tree import findings, simulator, tree_of
from mactf.training.mutation import (
    MutationLimits, MutationPlan, TopologySimulator, structure_digest, tree_state_digest,
)


def plan(*ops):
    return MutationPlan.parse({"ops": list(ops)})


FLAT = ("a", "b", "c")
HIER = ("a", "b", "a/x", "a/y", "b/x", "b/y")


def codes(items):
    return [re.match(r"[EWF]\d+", item).group(0) for item in items]


def test_a_valid_plan_has_no_findings_and_changes_one_group():
    tree = tree_of(*FLAT)
    sim = simulator(tree).run(plan({"op": "add", "path": "d", "like": "a"}))
    assert sim.ok and sim.added == ("d",) and sim.changed_labels() == ("root",)
    assert sim.before_digest != sim.after_digest
    assert "d" in sim.tree and "d" not in tree                      # the clone, not the tree itself


def test_the_simulator_never_touches_the_tree_it_was_given():
    tree = tree_of(*HIER)
    before = (structure_digest(tree, ()), [(n.path, n.status, n.children) for n in tree.walk("root-first")])
    simulator(tree).run(plan(
        {"op": "retire", "path": "a", "recursive": True}, {"op": "add", "path": "c"},
        {"op": "purge", "path": "b", "recursive": True, "force": True},
    ))
    assert before == (structure_digest(tree, ()), [(n.path, n.status, n.children) for n in tree.walk("root-first")])


def test_add_under_a_missing_parent():
    out = findings(plan({"op": "add", "path": "peer-9/x"}), tree_of(*FLAT))
    assert codes(out) == ["E081"] and "[peer-9/x]" in out[0] and "does not exist" in out[0]


def test_add_under_a_retired_parent():
    out = findings(plan({"op": "add", "path": "a/z"}), tree_of(*HIER), retired=["a", "a/x", "a/y"])
    assert codes(out) == ["E081"] and "revive it first" in out[0]


def test_add_a_tombstoned_path():
    out = findings(plan({"op": "add", "path": "d"}), tree_of(*FLAT), tombstones=["d"])
    assert codes(out) == ["E081"] and "cannot be reused" in out[0]


def test_add_an_existing_or_malformed_path():
    assert codes(findings(plan({"op": "add", "path": "a"}), tree_of(*FLAT))) == ["E081"]
    for bad in ("a/", "/a", "a//b", "a/../b", "..", "a b", "a\\b"):
        out = findings(plan({"op": "add", "path": bad}), tree_of(*FLAT))
        assert codes(out) == ["E081"], bad


def test_a_duplicate_name_inside_one_plan():
    out = findings(plan({"op": "add", "path": "d"}, {"op": "add", "path": "d"}), tree_of(*FLAT))
    assert codes(out) == ["E081"] and "already exists" in out[0]


def test_retire_with_active_children_needs_recursive():
    out = findings(plan({"op": "retire", "path": "a"}), tree_of(*HIER))
    assert codes(out) == ["E082"] and "a/x" in out[0] and "recursive" in out[0]
    assert findings(plan({"op": "retire", "path": "a", "recursive": True}), tree_of(*HIER)) == []


def test_protected_globs():
    limits = MutationLimits(enabled=True, min_active_roots=1, protected=("a", "b/*"))
    tree = tree_of(*HIER)
    assert codes(findings(plan({"op": "retire", "path": "a", "recursive": True}), tree, limits=limits)) == ["E082"]
    assert codes(findings(plan({"op": "retire", "path": "b/x"}), tree, limits=limits)) == ["E082"]
    assert codes(findings(plan({"op": "purge", "path": "b/y", "force": True}), tree, limits=limits)) == ["E082"]
    rebuild = plan({"op": "modify", "path": "b/y", "set": {"rebuild": {"factory": "@build"}}})
    assert codes(findings(rebuild, tree, limits=limits)) == ["E082"]
    # metadata is not an architecture change: still allowed
    assert findings(plan({"op": "modify", "path": "b/y", "set": {"metadata": {"k": 1}}}), tree, limits=limits) == []


def test_min_active_roots_is_enforced():
    limits = MutationLimits(enabled=True, min_active_roots=2)
    out = findings(plan({"op": "retire", "path": "a"}, {"op": "retire", "path": "b"}), tree_of(*FLAT), limits=limits)
    assert codes(out) == ["E082"] and "1 active root" in out[0] and "MIN_ACTIVE_ROOTS" in out[0]


def test_removing_every_root_is_refused_even_by_purge():
    limits = MutationLimits(enabled=True, min_active_roots=1)
    out = findings(
        plan({"op": "purge", "path": "a", "force": True}, {"op": "purge", "path": "b", "force": True},
             {"op": "purge", "path": "c", "force": True}),
        tree_of(*FLAT), limits=limits,
    )
    assert codes(out) == ["E082"] and "0 active root" in out[0]


def test_the_reference_is_not_a_node_and_cannot_be_addressed():
    for verb in ({"op": "retire", "path": "@reference"}, {"op": "purge", "path": "@reference", "force": True},
                 {"op": "revive", "path": "@reference"}, {"op": "modify", "path": "@reference", "set": {"role": "x"}}):
        assert codes(findings(plan(verb), tree_of(*FLAT))) == ["E081"], verb
    # ...and a node that is merely *named* like a reference is an ordinary node
    assert findings(plan({"op": "retire", "path": "a__reference"}), tree_of("a", "b", "a__reference")) == []


def test_every_child_retired_is_refused():
    out = findings(plan({"op": "retire", "path": "a/x"}, {"op": "retire", "path": "a/y"}), tree_of(*HIER))
    assert codes(out) == ["E082"] and "cannot be averaged" in out[0]


def test_an_active_node_under_a_retired_parent_is_refused():
    out = findings(plan({"op": "revive", "path": "a/x"}), tree_of(*HIER), retired=["a", "a/x", "a/y"])
    assert codes(out) == ["E081"] and "revive it too" in out[0]
    # reviving the branch recursively is fine
    assert findings(plan({"op": "revive", "path": "a", "recursive": True}), tree_of(*HIER),
                    retired=["a", "a/x", "a/y"]) == []


def test_like_and_init_from_must_name_a_live_node():
    tree = tree_of(*FLAT)
    assert codes(findings(plan({"op": "add", "path": "d", "like": "ghost"}), tree)) == ["E083"]
    assert codes(findings(plan({"op": "add", "path": "d", "like": "d"}), tree)) == ["E083"]      # itself
    assert codes(findings(plan({"op": "add", "path": "d", "init_from": "ghost"}), tree)) == ["E083"]
    purged_first = plan({"op": "purge", "path": "c", "force": True}, {"op": "add", "path": "d", "like": "c"})
    assert codes(findings(purged_first, tree)) == ["E083"]                                        # a purged node
    assert findings(plan({"op": "add", "path": "d", "init_from": "@reference"}), tree) == []
    assert codes(findings(plan({"op": "modify", "path": "a", "set": {"init_from": "a"}}), tree)) == ["E083"]
    rebuild = plan({"op": "modify", "path": "a", "set": {"rebuild": {"like": "ghost"}}})
    assert codes(findings(rebuild, tree)) == ["E083"]


def test_too_many_operations():
    limits = MutationLimits(enabled=True, min_active_roots=1, max_ops_per_plan=2)
    out = findings(plan(*({"op": "add", "path": f"n{i}"} for i in range(3))), tree_of(*FLAT), limits=limits)
    assert codes(out) == ["E084"] and "MAX_OPS_PER_PLAN" in out[0] and "2" in out[0]


def test_a_plan_longer_than_the_tree_refuses():
    tree = PeerTree(max_nodes=4)
    for name in FLAT:
        tree.add_node(name)
    out = findings(plan({"op": "add", "path": "d"}, {"op": "add", "path": "e"}), tree)
    assert codes(out) == ["E084"] and "e" in out[0]


def test_a_plan_deeper_than_the_tree_allows_refuses():
    tree = PeerTree(max_depth=2)
    for name in ("a", "b", "a/x"):
        tree.add_node(name)
    assert codes(findings(plan({"op": "add", "path": "a/x/deep"}), tree)) == ["E084"]


def test_infeasible_group_is_refused():
    def feasible(count):
        if count < 2:
            raise ComplianceError(f"{count} active members admit no legal weight vector")

    tree = tree_of(*FLAT)
    sim = TopologySimulator(tree, retired=(), tombstones=(), limits=MutationLimits(enabled=True, min_active_roots=1),
                            feasible=feasible)
    out = sim.run(plan({"op": "retire", "path": "a"}, {"op": "retire", "path": "b"}))
    assert [f.code for f in out.hard] == ["E086"] and "'root'" in out.hard[0].message
    # one new child under a leaf makes a group of one: refused; two make a group of two: accepted
    assert [f.code for f in sim.run(plan({"op": "add", "path": "a/x"})).hard] == ["E086"]
    assert sim.run(plan({"op": "add", "path": "a/x"}, {"op": "add", "path": "a/y"})).ok


def test_already_retired_is_a_warning():
    sim = simulator(tree_of(*FLAT)).run(plan({"op": "retire", "path": "a"}, {"op": "retire", "path": "a"}))
    assert sim.hard == () and [f.code for f in sim.soft] == ["W040"]


def test_revive_when_not_retired_is_a_warning():
    sim = simulator(tree_of(*FLAT)).run(plan({"op": "revive", "path": "a"}))
    assert sim.hard == () and [f.code for f in sim.soft] == ["W041"] and sim.revived_now == ()


def test_findings_stop_at_the_first_hard_one():
    tree = tree_of(*FLAT)
    sim = simulator(tree).run(plan(
        {"op": "add", "path": "ghost/x"},              # E081: later operations are undefined
        {"op": "add", "path": "d"}, {"op": "retire", "path": "zzz"},
    ))
    assert [f.code for f in sim.findings] == ["E081"] and sim.added == ()
    with pytest.raises(MutationError) as caught:
        sim.raise_for_findings()
    assert caught.value.code == "E081" and len(caught.value.findings) == 1


def test_operations_see_the_tree_the_earlier_ones_made():
    tree = tree_of(*FLAT)
    sim = simulator(tree).run(plan({"op": "add", "path": "d"}, {"op": "add", "path": "d/x"}, {"op": "retire", "path": "d/x"}))
    assert sim.ok is False                               # d now has one child and it is retired: E082
    sim = simulator(tree).run(plan({"op": "add", "path": "d"}, {"op": "add", "path": "d/x"}, {"op": "add", "path": "d/y"}))
    assert sim.ok and set(sim.added) == {"d", "d/x", "d/y"}


def test_purge_needs_force_for_an_active_node_and_recursive_for_a_branch():
    tree = tree_of(*HIER)
    assert codes(findings(plan({"op": "purge", "path": "c"}), tree_of(*FLAT))) == ["E082"]        # active, no force
    assert findings(plan({"op": "retire", "path": "c"}, {"op": "purge", "path": "c"}), tree_of(*FLAT)) == []
    assert codes(findings(plan({"op": "purge", "path": "a", "force": True}), tree)) == ["E082"]   # has children
    assert findings(plan({"op": "purge", "path": "a", "recursive": True, "force": True}), tree) == []


def test_allow_purge_false_makes_retire_the_only_way_out():
    limits = MutationLimits(enabled=True, min_active_roots=1, allow_purge=False)
    out = findings(plan({"op": "purge", "path": "c", "force": True}), tree_of(*FLAT), limits=limits)
    assert codes(out) == ["E082"] and "ALLOW_PURGE" in out[0]


def test_purge_records_tombstones_and_a_node_added_then_purged_leaves_none():
    sim = simulator(tree_of(*FLAT)).run(plan({"op": "purge", "path": "c", "force": True}))
    assert sim.purged == ("c",) and sim.tombstones == {"c"}
    sim = simulator(tree_of(*FLAT)).run(plan({"op": "add", "path": "d"}, {"op": "purge", "path": "d", "force": True}))
    assert sim.ok and sim.purged == () and sim.added == () and sim.tombstones == frozenset()
    off = MutationLimits(enabled=True, min_active_roots=1, tombstone="off")
    sim = simulator(tree_of(*FLAT), limits=off).run(plan({"op": "purge", "path": "c", "force": True}))
    assert sim.tombstones == frozenset()                 # TOMBSTONE: off restores the freedom


def test_a_retire_then_revive_in_one_plan_is_a_no_op_for_the_digest():
    tree = tree_of(*FLAT)
    sim = simulator(tree).run(plan({"op": "retire", "path": "a"}, {"op": "revive", "path": "a"}))
    assert sim.ok and sim.before_digest == sim.after_digest and sim.retired_now == () and sim.revived_now == ()


def test_modify_a_retired_node_needs_a_revive_first():
    out = findings(plan({"op": "modify", "path": "a", "set": {"role": "x"}}), tree_of(*FLAT), retired=["a"])
    assert codes(out) == ["E081"] and "revive it first" in out[0]
    assert findings(plan({"op": "revive", "path": "a"}, {"op": "modify", "path": "a", "set": {"role": "x"}}),
                    tree_of(*FLAT), retired=["a"]) == []


def test_modify_merges_metadata_and_a_null_deletes_a_key():
    tree = tree_of(*FLAT)
    tree["a"].metadata = {"keep": 1, "drop": 2}
    sim = simulator(tree).run(plan({"op": "modify", "path": "a", "set": {"metadata": {"drop": None, "new": 3}, "role": "lead"}}))
    assert sim.tree["a"].metadata == {"keep": 1, "new": 3} and sim.tree["a"].role == "lead"
    assert tree["a"].metadata == {"keep": 1, "drop": 2}


def test_a_stored_tree_that_does_not_parse_is_a_corrupt_checkpoint():
    """A path names its parent, so a cycle cannot even be written; an orphan can, and is E089."""
    state = tree_of(*HIER).state_dict()
    state["nodes"] = [node for node in state["nodes"] if node["path"] != "a"]       # a/x loses its parent
    with pytest.raises(MutationError) as caught:
        tree_state_digest(state)
    assert caught.value.code == "E089"
    state = tree_of(*HIER).state_dict()
    state["nodes"][0]["status"] = "exploded"
    with pytest.raises(MutationError) as caught:
        tree_state_digest(state)
    assert caught.value.code == "E089"


def test_a_path_cannot_be_its_own_ancestor():
    """Paths are the identity, so a node cannot be re-parented under its own subtree: there is no verb for it."""
    tree = tree_of(*HIER)
    assert codes(findings(plan({"op": "add", "path": "a"}), tree)) == ["E081"]
    # the nearest thing: adding a child of a descendant is a *new* node, never a cycle
    sim = simulator(tree).run(plan({"op": "add", "path": "a/x/a"}))
    assert sim.ok and [n.path for n in sim.tree.walk("root-first")].count("a/x/a") == 1


def test_the_structure_digest_names_retired_nodes():
    tree = tree_of(*FLAT)
    assert structure_digest(tree, ()) != structure_digest(tree, ["a"])
    assert structure_digest(tree, ["a"]) == structure_digest(tree, ["a"])
    assert dataclasses.replace(MutationLimits(), enabled=True).enabled
