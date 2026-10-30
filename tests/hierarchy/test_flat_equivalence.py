"""The flat-equivalence gates (document 10 section 9.1).

If any of these fails, the tree abstraction has changed a number and the
implementation is wrong. The gates:

* depth-1 scoring equals flat scoring at ``1e-12``;
* depth-1 synthesis equals ``consensus`` at ``1e-12``;
* ``KappaWeightedRollup`` with one group equals ``Aggregator.compute_weights`` at
  ``1e-12``;
* ``Episode.participants`` keys equal ``tree.root_paths`` as sets.
"""

from __future__ import annotations

import pytest
import torch

from mactf.coopetition.aggregation import Aggregator
from mactf.coopetition.peer import build_peer
from mactf.coopetition.synthesis import ConsensusSynthesis
from mactf.coopetition.temperature import AdaptiveKappa
from mactf.coopetition.tree import HierarchySpec, PeerTree
from mactf.orchestration.episode import Episode, consensus
from mactf.runtime import Runtime
from mactf.rl.functions.rollup import KappaWeightedRollup
from mactf.scoring.hierarchy import HierarchicalScorer
from mactf.scoring.ledger import Ledger
from mactf.scoring.pipeline import ScoringPipeline


def _runtime() -> Runtime:
    return Runtime(
        device=torch.device("cpu"),
        dtype=torch.float32,
        amp=False,
        seed=0,
        deterministic=True,
    )


def _peer(name: str) -> object:
    module = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Tanh(), torch.nn.Linear(4, 3))
    return build_peer(module, name=name, runtime=_runtime(), action_dim=3, state_dim=4)


def _ledgers(names) -> dict[str, Ledger]:
    """Deliberately different ledgers per peer, so equality is not trivial."""
    plan = {
        "peer-a": ["R1", "R2"],
        "peer-b": ["R1", "P4"],
        "peer-c": ["R3"],
    }
    ledgers: dict[str, Ledger] = {}
    for index, name in enumerate(names):
        ledger = Ledger(peer=name)
        for rule_id in plan[name]:
            ledger.record(rule_id, magnitude=0.5 + 0.25 * index, evidence=f"{name}:{rule_id}")
        ledgers[name] = ledger
    return ledgers


def test_depth_one_scoring_equals_flat_scoring_at_1e_12():
    names = ["peer-a", "peer-b", "peer-c"]
    ledgers = _ledgers(names)
    flat = ScoringPipeline()
    flat_cards = {
        name: flat.score_peer(ledger, peer=name, episode=1) for name, ledger in ledgers.items()
    }

    tree = PeerTree.flat(names)
    for name in names:
        tree[name].peer = _peer(name)
    hierarchical = HierarchicalScorer(ScoringPipeline()).score_tree(tree, ledgers, episode=1)

    for name in names:
        assert abs(hierarchical[name].final_score - flat_cards[name].final_score) <= 1e-12, name
        assert abs(
            hierarchical[name].continuous.reward.total - flat_cards[name].continuous.reward.total
        ) <= 1e-12
        assert abs(
            hierarchical[name].continuous.penalty.total - flat_cards[name].continuous.penalty.total
        ) <= 1e-12
        assert abs(hierarchical[name].continuous.warning - flat_cards[name].continuous.warning) <= 1e-12


def test_depth_one_tree_preserves_the_flat_void_set():
    names = ["peer-a", "peer-b"]
    ledgers = _ledgers(names)
    # P1 is non-offsettable; peer-b has none, so only peer-a is void.
    ledgers["peer-a"].record("P1", evidence="fabricated")
    flat = ScoringPipeline()
    flat_cards = {name: flat.score_peer(ledger, peer=name) for name, ledger in ledgers.items()}

    tree = PeerTree.flat(names)
    hierarchical = HierarchicalScorer(ScoringPipeline()).score_tree(tree, ledgers)
    for name in names:
        assert set(hierarchical[name].violated_non_offsettable) == set(
            flat_cards[name].violated_non_offsettable
        ), name


def test_depth_one_synthesis_equals_consensus_at_1e_12():
    tree = PeerTree.flat(["peer-a"])
    proposals = {
        "peer-a": torch.tensor([[1.0, 2.0, 3.0]]),
    }
    node = tree["peer-a"]
    direct = consensus(dict(proposals))
    synthesized = ConsensusSynthesis().synthesize(node, dict(proposals))
    assert direct is not None
    assert torch.allclose(synthesized, direct, atol=1e-12, rtol=0.0)


def test_multi_child_consensus_equals_consensus_at_1e_12():
    tree = PeerTree.flat(["peer-a"])
    tree.add_node("peer-a/sub-1")
    tree.add_node("peer-a/sub-2")
    proposals = {
        "peer-a/sub-1": torch.tensor([[0.25, -1.5, 2.0]]),
        "peer-a/sub-2": torch.tensor([[1.75, 0.5, -0.25]]),
    }
    direct = consensus(dict(proposals))
    synthesized = ConsensusSynthesis().synthesize(tree["peer-a"], dict(proposals))
    assert direct is not None
    assert torch.allclose(synthesized, direct, atol=1e-12, rtol=0.0)


def test_kappa_weighted_rollup_with_one_group_equals_the_aggregator():
    from mactf.scoring.base import ScorerOutput, ScoringKind

    names = ["peer-a", "peer-b", "peer-c"]
    scores = [0.9, 0.4, -0.2]

    aggregator = Aggregator(controller=AdaptiveKappa(num_peers=3, kappa_init=1.7))
    report = aggregator.compute_weights(names, scores, advance=False)

    rollup = KappaWeightedRollup()
    children = tuple(
        ScorerOutput(kind=ScoringKind.HIERARCHY, amount=score, rules=(name,))
        for name, score in zip(names, scores)
    )
    output = rollup.rollup("", children, weights=dict(report.weights))
    expected = sum(report.weights[name] * score for name, score in zip(names, scores))
    assert abs(output.amount - expected) <= 1e-12


def test_episode_participants_equal_the_tree_roots():
    from mactf.orchestration.episode import Task

    tree = PeerTree.flat(["peer-a", "peer-b"])
    task = Task(id="t", states=torch.zeros(1, 3), targets=torch.zeros(1, dtype=torch.long))
    episode = Episode(index=1, task=task)
    for name in tree.root_paths:
        episode.ledger_for(name)
    assert set(episode.participants) == set(tree.root_paths)


def test_hierarchy_spec_flat_matches_the_peer_list_topology():
    names = ["peer-a", "peer-b"]
    spec_tree = HierarchySpec.flat(names).build()
    list_tree = PeerTree.flat(names)
    # The root role is the one intentional difference: ``HierarchySpec.build``
    # labels roots ``"root"`` so a delegation policy can route by role, while a
    # bare ``PeerTree.flat`` leaves every node at the default ``"worker"``.
    assert spec_tree.root_paths == list_tree.root_paths
    assert spec_tree.depth == list_tree.depth == 1
    spec_nodes = [(node["path"], node["kind"], node["parent"]) for node in spec_tree.state_dict()["nodes"]]
    list_nodes = [(node["path"], node["kind"], node["parent"]) for node in list_tree.state_dict()["nodes"]]
    assert spec_nodes == list_nodes


def test_a_flat_tree_has_one_aggregation_group():
    from mactf.coopetition.tree import aggregation_groups

    tree = PeerTree.flat(["a", "b", "c"])
    assert aggregation_groups(tree) == [(None, ("a", "b", "c"))]
