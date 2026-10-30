"""Failure isolation in a tree (document 10 section 9.2).

One broken nano-agent cannot abort a run: the node is marked failed, its parent
synthesizes from whatever proposals survived, and a branch with nothing left
fails upward. The watchdog's ``DeadBranchProbe`` patience is covered in
``tests/training/test_watchdog.py``, where the watchdog lives.
"""

from __future__ import annotations

import pytest
import torch

from mactf.coopetition.peer import build_peer
from mactf.coopetition.synthesis import ConsensusSynthesis, SynthesisFunction
from mactf.coopetition.tree import HierarchySpec, LevelSpec, PeerTree
from mactf.exceptions import ContractError
from mactf.runtime import Runtime
from mactf.scoring.hierarchy import run_inference


class Constant(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.full((3,), float(value)))

    def forward(self, state):
        return self.bias.expand(state.shape[0], 3).clone()


class Exploding(torch.nn.Module):
    def __init__(self, message: str = "boom") -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(3))
        self.message = message

    def forward(self, state):
        raise RuntimeError(self.message)


def _runtime() -> Runtime:
    return Runtime(
        device=torch.device("cpu"),
        dtype=torch.float32,
        amp=False,
        seed=0,
        deterministic=True,
    )


def _build(broken: set[str]) -> PeerTree:
    counter = {"n": 0}

    def factory(path: str, depth: int, level) -> object:
        counter["n"] += 1
        module = Exploding(f"broken {path}") if path in broken else Constant(counter["n"])
        return build_peer(module, name=path, runtime=_runtime(), action_dim=3, state_dim=3)

    return HierarchySpec(levels=(LevelSpec("sub", 2),), max_depth=3).build(
        peer_factory=factory
    )


def test_a_broken_leaf_does_not_abort_the_request():
    tree = _build({"peer-1/sub-1"})
    result = run_inference(tree, {"state": torch.randn(1, 3)})
    assert "peer-1/sub-2" in result.outputs
    assert "peer-1" in result.outputs


def test_the_broken_leaf_is_reported_with_its_error():
    tree = _build({"peer-1/sub-1"})
    result = run_inference(tree, {"state": torch.randn(1, 3)})
    record = next(node for node in result.per_node if node.path == "peer-1/sub-1")
    assert not record.ok
    assert "RuntimeError" in record.error
    assert "broken peer-1/sub-1" in record.error


def test_the_parent_synthesizes_from_the_surviving_proposal():
    tree = _build({"peer-1/sub-1"})
    result = run_inference(tree, {"state": torch.randn(1, 3)})
    survivor = result.outputs["peer-1/sub-2"]
    # Consensus over one proposal is that proposal, so the parent's answer is
    # exactly the surviving child's answer rather than a padded average.
    assert torch.equal(result.outputs["peer-1"], survivor)


def test_every_selected_child_failing_fails_the_parent_too():
    tree = _build({"peer-1/sub-1", "peer-1/sub-2"})
    result = run_inference(tree, {"state": torch.randn(1, 3)})
    assert "peer-1" not in result.outputs
    parent = next(node for node in result.per_node if node.path == "peer-1")
    assert not parent.ok
    assert "no proposals to synthesize" in parent.error


def test_the_failure_propagates_to_the_root_and_leaves_it_empty():
    tree = _build({"peer-1/sub-1", "peer-1/sub-2"})
    result = run_inference(tree, {"state": torch.randn(1, 3)})
    assert result.root_outputs["peer-1"] is None


def test_isolation_can_be_disabled_to_surface_the_first_failure():
    tree = _build({"peer-1/sub-1"})
    with pytest.raises(RuntimeError, match="broken peer-1/sub-1"):
        run_inference(tree, {"state": torch.randn(1, 3)}, isolate_failures=False)


def test_a_deep_failure_only_affects_its_branch():
    spec = HierarchySpec(
        levels=(LevelSpec("sub", 2), LevelSpec("mini", 2)),
        max_depth=4,
    )

    def factory(path: str, depth: int, level) -> object:
        module = Exploding("deep") if path == "peer-1/sub-1/mini-2" else Constant(1.0)
        return build_peer(module, name=path, runtime=_runtime(), action_dim=3, state_dim=3)

    tree = spec.build(peer_factory=factory)
    result = run_inference(tree, {"state": torch.randn(1, 3)})
    assert "peer-1/sub-2/mini-1" in result.outputs
    assert "peer-1/sub-1/mini-2" not in result.outputs
    # ``sub-1`` still has ``mini-1``, so the branch survives.
    assert "peer-1/sub-1" in result.outputs
    assert "peer-1" in result.outputs


def test_inference_does_not_mutate_the_tree_status():
    """Inference is observational: it must not mark production nodes failed."""
    tree = _build({"peer-1/sub-1"})
    run_inference(tree, {"state": torch.randn(1, 3)})
    assert tree["peer-1/sub-1"].status == "active"


def test_a_synthesis_function_that_rejects_an_empty_map_fails_the_parent():
    class Strict(SynthesisFunction):
        name = "strict"

        def synthesize(self, node, proposals, *, scores=None, weights=None, task=None):
            if not proposals:
                raise ContractError(f"no proposals for {node.path}")
            return next(iter(proposals.values()))

    tree = _build({"peer-1/sub-1", "peer-1/sub-2"})
    result = run_inference(tree, {"state": torch.randn(1, 3)}, synthesis=Strict())
    assert "peer-1" not in result.outputs


def test_a_failing_synthesis_fails_only_its_own_node():
    class Picky(SynthesisFunction):
        name = "picky"

        def synthesize(self, node, proposals, *, weights=None, scores=None, task=None):
            if node.path == "peer-1/sub-1":
                raise ContractError("nope")
            return ConsensusSynthesis().synthesize(
                node, proposals, scores=scores, weights=weights, task=task
            )

    spec = HierarchySpec(levels=(LevelSpec("sub", 2), LevelSpec("mini", 2)), max_depth=4)

    def factory(path: str, depth: int, level) -> object:
        return build_peer(
            Constant(1.0), name=path, runtime=_runtime(), action_dim=3, state_dim=3
        )

    tree = spec.build(peer_factory=factory)
    result = run_inference(tree, {"state": torch.randn(1, 3)}, synthesis=Picky())
    assert "peer-1/sub-1" not in result.outputs
    assert "peer-1/sub-2" in result.outputs


def test_an_inactive_subtree_is_excluded_from_aggregation():
    """A failed subtree is ineligible, which is how aggregation skips it."""
    from mactf.coopetition.aggregation import Aggregator
    from mactf.coopetition.temperature import AdaptiveKappa

    tree = _build(set())
    tree.mark("peer-1/sub-1", "failed")
    names = ["peer-1/sub-1", "peer-1/sub-2"]
    eligible = [tree[name].status == "active" for name in names]
    assert eligible == [False, True]
    aggregator = Aggregator(controller=AdaptiveKappa(num_peers=2), weight_cap=1.0)
    report = aggregator.compute_weights(
        names, [1.0, 1.0], eligible=eligible, advance=False
    )
    assert report.weights["peer-1/sub-1"] == 0.0
    assert report.weights["peer-1/sub-2"] == pytest.approx(1.0)
    assert report.gated_peers == ("peer-1/sub-1",)


def test_a_failed_node_is_still_in_the_tree():
    tree = _build(set())
    tree.mark("peer-1/sub-1", "failed")
    assert "peer-1/sub-1" in tree
    assert tree["peer-1/sub-1"].status == "failed"
    assert tree.depth == 2


def test_parallel_and_serial_isolate_identically():
    serial = run_inference(_build({"peer-1/sub-1"}), {"state": torch.randn(2, 3)})
    parallel = run_inference(
        _build({"peer-1/sub-1"}), {"state": torch.randn(2, 3)}, parallel=True
    )
    serial_errors = {node.path: node.error for node in serial.per_node}
    parallel_errors = {node.path: node.error for node in parallel.per_node}
    assert serial_errors == parallel_errors
    assert set(serial.outputs) == set(parallel.outputs)
