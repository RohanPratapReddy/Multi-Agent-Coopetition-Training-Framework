"""test branch void (split from test_hierarchical_scoring.py)."""

from __future__ import annotations

from mactf.coopetition.tree import HierarchySpec, LevelSpec, PeerTree


from mactf.scoring.hierarchy import HierarchicalScorer, build_rollup


from mactf.scoring.ledger import Ledger


from mactf.scoring.pipeline import ScoringPipeline


SCORE_CONFIG = {"REWARD_WEIGHTS": {"r_task": 1.0, "r_epi": 1.5, "r_collab": 1.2, "r_align": 1.0}}


def _pipeline(**kwargs):
    return ScoringPipeline(score_config=SCORE_CONFIG, **kwargs)


def _ledger(peer: str, *events) -> Ledger:
    ledger = Ledger(peer=peer)
    for rule_id, magnitude in events:
        ledger.record(rule_id, magnitude=magnitude)
    return ledger


def test_the_worked_example_voids_every_ancestor():
    """Document 10 section 5.3, verbatim.

    ``peer-a/nano-1`` breaches P1; ``micro-1`` earns large rewards; ``peer-a``
    earns R5. Every node in the branch must take the floor.
    """
    tree = PeerTree()
    tree.add_node("peer-a")
    tree.add_node("peer-a/micro-1")
    tree.add_node("peer-a/micro-1/nano-1")
    ledgers = {
        "peer-a/micro-1/nano-1": _ledger("peer-a/micro-1/nano-1", ("P1", 1.0)),
        "peer-a/micro-1": _ledger("peer-a/micro-1", ("R1", 1.0), ("R2", 1.0)),
        "peer-a": _ledger("peer-a", ("R5", 1.0)),
    }
    card = HierarchicalScorer(_pipeline()).score_tree(tree, ledgers)
    for path in ("peer-a/micro-1/nano-1", "peer-a/micro-1", "peer-a"):
        assert card.peers[path].final_score < 0.0, path
        assert card.peers[path].violated_non_offsettable == ("P1",), path


def test_rewards_do_not_offset_an_inherited_void():
    """Ten rewards at ``micro-1`` change nothing about the root's void status."""
    tree = PeerTree()
    tree.add_node("peer-a")
    tree.add_node("peer-a/micro-1")
    tree.add_node("peer-a/micro-1/nano-1")
    rich = _ledger(
        "peer-a/micro-1",
        *[("R1", 1.0) for _ in range(5)],
        *[("R2", 1.0) for _ in range(5)],
    )
    ledgers = {
        "peer-a/micro-1/nano-1": _ledger("peer-a/micro-1/nano-1", ("P1", 1.0)),
        "peer-a/micro-1": rich,
    }
    card = HierarchicalScorer(_pipeline()).score_tree(tree, ledgers)
    peer = card.peers["peer-a"]
    assert peer.is_void
    # The inherited void contributes the standard non-offsettable floor. Ten
    # rewards move the number by ~11 points against a floor of -1000, which is
    # exactly the "averaging never launders a breach" property.
    assert peer.floor <= -1000.0
    assert peer.final_score < -900.0


def test_a_clean_branch_is_not_voided_by_a_sibling():
    tree = PeerTree.flat(["peer-a", "peer-b"])
    ledgers = {
        "peer-a": _ledger("peer-a", ("P1", 1.0)),
        "peer-b": _ledger("peer-b", ("R1", 0.8)),
    }
    card = HierarchicalScorer(_pipeline()).score_tree(tree, ledgers)
    assert card.peers["peer-a"].is_void
    assert not card.peers["peer-b"].is_void
    assert card.peers["peer-b"].final_score > 0.0


def test_branch_void_is_reported_on_the_card():
    tree = PeerTree()
    tree.add_node("peer-a")
    tree.add_node("peer-a/sub-1")
    ledgers = {"peer-a/sub-1": _ledger("peer-a/sub-1", ("P1", 1.0))}
    card = HierarchicalScorer(_pipeline()).score_tree(tree, ledgers)
    assert card.peers["peer-a"].branch_void == ("P1",)
