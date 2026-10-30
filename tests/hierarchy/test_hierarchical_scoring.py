"""Hierarchical scoring: the flat-equivalence gate and the worked void example.

Document 10 section 9.1 is the acceptance gate: a depth-1 tree must reproduce
today's flat scoring to ``1e-12`` on every score. Section 5.3 is the property that
makes hierarchy worth having: a nano-agent's P1 floors its root regardless of
sibling rewards.
"""

from __future__ import annotations

import pytest


from mactf.coopetition.tree import HierarchySpec, LevelSpec, PeerTree


from mactf.exceptions import ContractError, ImproperlyConfigured, ValidationError


from mactf.scoring.hierarchy import HierarchicalScorer, build_rollup


from mactf.scoring.ledger import Ledger


from mactf.scoring.pipeline import ScoringPipeline


from mactf.scoring.reward_scorer import RewardScorer


from mactf.scoring.warning_scorer import WarningScorer


SCORE_CONFIG = {"REWARD_WEIGHTS": {"r_task": 1.0, "r_epi": 1.5, "r_collab": 1.2, "r_align": 1.0}}


def _pipeline(**kwargs):
    return ScoringPipeline(score_config=SCORE_CONFIG, **kwargs)


def _ledger(peer: str, *events) -> Ledger:
    ledger = Ledger(peer=peer)
    for rule_id, magnitude in events:
        ledger.record(rule_id, magnitude=magnitude)
    return ledger


def test_a_depth_one_tree_reproduces_flat_scoring_at_1e_12():
    ledgers = {
        "peer-a": _ledger("peer-a", ("R1", 0.9), ("R6", 0.4)),
        "peer-b": _ledger("peer-b", ("R1", 0.5), ("P4", 1.0)),
    }
    flat = _pipeline().score_episode(ledgers, episode=3)
    tree = PeerTree.flat(list(ledgers))
    hierarchical = HierarchicalScorer(_pipeline()).score_tree(tree, ledgers, episode=3)

    assert set(hierarchical.peers) == set(flat.peers)
    for peer, card in flat.peers.items():
        other = hierarchical.peers[peer]
        assert other.final_score == pytest.approx(card.final_score, abs=1e-12)
        assert other.floor == pytest.approx(card.floor, abs=1e-12)
        assert other.continuous.reward.total == pytest.approx(
            card.continuous.reward.total, abs=1e-12
        )
        assert other.continuous.penalty.total == pytest.approx(
            card.continuous.penalty.total, abs=1e-12
        )
        assert other.continuous.warning == pytest.approx(card.continuous.warning, abs=1e-12)
        assert other.violated_non_offsettable == card.violated_non_offsettable


def test_a_depth_one_tree_preserves_the_void_set():
    ledgers = {"peer-a": _ledger("peer-a", ("P1", 1.0))}
    flat = _pipeline().score_episode(ledgers)
    hierarchical = HierarchicalScorer(_pipeline()).score_tree(
        PeerTree.flat(["peer-a"]), ledgers
    )
    assert hierarchical.peers["peer-a"].is_void
    assert (
        hierarchical.peers["peer-a"].violated_non_offsettable
        == flat.peers["peer-a"].violated_non_offsettable
    )


def test_rollup_attribution_inherits_a_childs_rewards():
    tree = PeerTree()
    tree.add_node("peer-a")
    tree.add_node("peer-a/sub-1")
    ledgers = {"peer-a/sub-1": _ledger("peer-a/sub-1", ("R1", 0.9))}
    card = HierarchicalScorer(_pipeline(), attribution="rollup").score_tree(tree, ledgers)
    assert card.peers["peer-a"].continuous.reward.total > 0.0
    assert card.metadata["attribution"] == "rollup"


def test_own_attribution_ignores_a_childs_rewards():
    tree = PeerTree()
    tree.add_node("peer-a")
    tree.add_node("peer-a/sub-1")
    ledgers = {"peer-a/sub-1": _ledger("peer-a/sub-1", ("R1", 0.9))}
    card = HierarchicalScorer(_pipeline(), attribution="own").score_tree(tree, ledgers)
    assert card.peers["peer-a"].continuous.reward.total == pytest.approx(0.0)
    # Non-offsettable propagation is unaffected by attribution.
    ledgers["peer-a/sub-1"].record("P1", magnitude=1.0)
    voided = HierarchicalScorer(_pipeline(), attribution="own").score_tree(tree, ledgers)
    assert voided.peers["peer-a"].is_void


def test_an_unknown_attribution_is_rejected():
    with pytest.raises(ValidationError, match="attribution"):
        HierarchicalScorer(_pipeline(), attribution="none")


def test_ledger_events_are_tagged_with_their_origin_in_rollup_mode():
    """Attribution must be auditable: the parent's ledger names the child path."""
    tree = PeerTree()
    tree.add_node("peer-a")
    tree.add_node("peer-a/sub-1")
    ledgers = {"peer-a/sub-1": _ledger("peer-a/sub-1", ("R1", 0.9))}
    scorer = HierarchicalScorer(_pipeline())
    merged = scorer._ledger_for_node(tree["peer-a"], tree, ledgers)
    tags = {tag for event in merged.events for tag in event.tags}
    assert "derived_from:peer-a/sub-1" in tags


def test_a_pipeline_factory_is_used_per_node():
    seen: list[str] = []

    def factory(path: str) -> ScoringPipeline:
        seen.append(path)
        return _pipeline()

    tree = HierarchySpec(levels=(LevelSpec("sub", 2),)).build()
    HierarchicalScorer(factory).score_tree(
        tree, {node.path: Ledger(peer=node.path) for node in tree.walk()}
    )
    assert set(seen) == {node.path for node in tree.walk()}


def test_a_factory_returning_something_else_is_rejected():
    with pytest.raises(ValidationError, match="not a ScoringPipeline"):
        HierarchicalScorer(lambda path: object()).score_tree(
            PeerTree.flat(["peer-a"]), {"peer-a": Ledger(peer="peer-a")}
        )


def test_an_unknown_pipeline_argument_is_rejected():
    with pytest.raises(ValidationError, match="needs a ScoringPipeline"):
        HierarchicalScorer("nope")


def test_a_ledger_for_an_unknown_path_is_rejected():
    tree = PeerTree.flat(["peer-a"])
    with pytest.raises(ContractError, match="not a node in this tree"):
        HierarchicalScorer(_pipeline()).score_tree(
            tree, {"peer-z": Ledger(peer="peer-z")}
        )


def test_a_non_tree_is_rejected():
    with pytest.raises(ContractError, match="needs a PeerTree"):
        HierarchicalScorer(_pipeline()).score_tree({"peer-a": Ledger()}, {})


def test_the_hierarchy_stage_cannot_join_a_per_node_chain():
    """``hierarchy`` is root-only: nesting it would run the tree once per node."""
    with pytest.raises(ImproperlyConfigured, match="per-node"):
        ScoringPipeline([WarningScorer(), RewardScorer(), HierarchicalScorer(_pipeline())])


def test_describe_reports_the_configuration():
    payload = HierarchicalScorer(
        _pipeline(), rollup="mean", attribution="own", non_offsettable=("P1",)
    ).describe()
    assert payload["attribution"] == "own"
    assert payload["rollup"] == "mean"
    # document 17 section 6.6 (H1): a configured set widens the catalog set, never narrows it
    assert payload["non_offsettable"] == ["P1", "P17", "P2", "P20", "P21", "P22"]
    assert payload["shared_pipeline"] is True


def test_the_stage_interface_rolls_children_up():
    from mactf.scoring.base import ScoreContext, ScorerOutput, ScoringKind

    scorer = HierarchicalScorer(_pipeline(), rollup="sum")
    ctx = ScoreContext(
        peer="peer-a",
        episode=0,
        ledger=Ledger(peer="peer-a"),
        node_path="peer-a",
        children=(
            ScorerOutput(kind=ScoringKind.HIERARCHY, amount=1.0),
            ScorerOutput(kind=ScoringKind.HIERARCHY, amount=2.0),
        ),
    )
    assert scorer.score(ctx).amount == pytest.approx(3.0)


def test_build_rollup_defaults_to_kappa():
    assert build_rollup(None).name == "kappa"
