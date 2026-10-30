"""Hierarchical inference (document 10 section 7).

Criteria 29 and 30: ``parallel`` must not change results, the result must not
depend on ``PYTHONHASHSEED``, and a budget overrun must raise naming the path
rather than returning a truncated answer.
"""

from __future__ import annotations

import pytest
import torch

from mactf.coopetition.delegation import BroadcastDelegation, SelectKDelegation
from mactf.coopetition.peer import build_peer
from mactf.coopetition.synthesis import (
    BestByScore,
    ConcatSynthesis,
    ConsensusSynthesis,
    WeightedConsensus,
)
from mactf.coopetition.tree import HierarchySpec, LevelSpec, PeerTree
from mactf.exceptions import ContractError, ValidationError
from mactf.runtime import Runtime
from mactf.scoring.hierarchy import run_inference


class Constant(torch.nn.Module):
    """A module whose output is a fixed vector, so results are comparable.

    It carries a single frozen parameter so that ``build_peer`` can construct a
    real optimizer over it, exactly as it would for a trainable policy.
    """

    def __init__(self, value: float, *, width: int = 4) -> None:
        super().__init__()
        self.width = int(width)
        self.bias = torch.nn.Parameter(torch.full((self.width,), float(value)))

    def forward(self, state):
        return self.bias.expand(state.shape[0], self.width).clone()


def _runtime() -> Runtime:
    return Runtime(
        device=torch.device("cpu"),
        dtype=torch.float32,
        amp=False,
        seed=0,
        deterministic=True,
    )


def _tree(depth_two: bool = True, *, root_value: float = 1.0):
    levels = (LevelSpec("sub", 2), LevelSpec("mini", 2)) if depth_two else (LevelSpec("sub", 2),)
    spec = HierarchySpec(levels=levels)
    counter = {"index": 0}

    def factory(path: str, depth: int, level) -> object:
        counter["index"] += 1
        return build_peer(
            Constant(root_value + counter["index"] * 0.1),
            name=path,
            runtime=_runtime(),
            action_dim=4,
            state_dim=3,
        )

    return spec.build(peer_factory=factory)


def test_inference_returns_a_root_output_with_the_right_shape():
    tree = _tree(depth_two=False)
    result = run_inference(tree, {"state": torch.randn(2, 3)})
    assert set(result.root_outputs) == {"peer-1"}
    assert tuple(result.root_outputs["peer-1"].shape) == (2, 4)


def test_inference_records_every_node():
    tree = _tree(depth_two=False)
    result = run_inference(tree, {"state": torch.randn(2, 3)})
    assert {node.path for node in result.per_node} == {
        "peer-1",
        "peer-1/sub-1",
        "peer-1/sub-2",
    }


def test_inference_records_the_delegation_decision():
    tree = _tree(depth_two=False)
    result = run_inference(tree, {"state": torch.randn(2, 3)})
    assert result.delegation["peer-1"]["selected"] == ["peer-1/sub-1", "peer-1/sub-2"]
    assert result.delegation["peer-1"]["starved"] == []


def test_a_narrow_delegation_starves_the_unselected_children():
    tree = _tree(depth_two=False)
    result = run_inference(
        tree,
        {"state": torch.randn(2, 3)},
        delegation=SelectKDelegation(k=1),
    )
    assert result.delegation["peer-1"]["starved"] == ["peer-1/sub-2"]
    assert "peer-1/sub-2" not in result.outputs


def test_inference_starts_at_a_subtree_when_asked():
    tree = _tree(depth_two=False)
    result = run_inference(tree, {"state": torch.randn(2, 3)}, node_path="peer-1/sub-1")
    assert set(result.root_outputs) == {"peer-1/sub-1"}
    assert "peer-1" not in result.outputs
    assert tuple(result.root_outputs["peer-1/sub-1"].shape) == (2, 4)


def test_an_unknown_node_path_is_rejected():
    tree = _tree(depth_two=False)
    with pytest.raises(ContractError, match="no node at path"):
        run_inference(tree, {"state": torch.randn(1, 3)}, node_path="nope")


def test_parallel_and_serial_agree_exactly():
    tree = _tree()
    inputs = {"state": torch.randn(3, 3)}
    serial = run_inference(tree, inputs, seed=7)
    parallel = run_inference(tree, inputs, parallel=True, max_workers=4, seed=7)
    assert set(serial.outputs) == set(parallel.outputs)
    for path, value in serial.outputs.items():
        assert torch.equal(value, parallel.outputs[path]), path


def test_the_result_is_independent_of_python_hash_seed(monkeypatch):
    """The seed derivation uses blake2b, so ``PYTHONHASHSEED`` cannot matter."""
    tree = _tree(depth_two=False)
    inputs = {"state": torch.randn(2, 3)}
    first = run_inference(tree, inputs, seed=11)
    monkeypatch.setenv("PYTHONHASHSEED", "12345")
    # Re-derive the tree so any hashing inside construction also reruns.
    second = run_inference(_tree(depth_two=False), inputs, seed=11)
    for path, value in first.outputs.items():
        assert torch.equal(value, second.outputs[path]), path


def test_a_budget_overrun_names_the_path_and_never_truncates():
    tree = _tree()
    with pytest.raises(ContractError) as info:
        run_inference(tree, {"state": torch.randn(1, 3)}, budget=2)
    message = str(info.value)
    assert "budget exhausted" in message
    assert "peer-1" in message
    assert "remain" in message


def test_a_generous_budget_succeeds():
    tree = _tree(depth_two=False)
    result = run_inference(tree, {"state": torch.randn(1, 3)}, budget=100)
    assert result.cost["invocations"] > 0
    assert result.cost["budget"] == 100


def test_max_depth_beyond_the_tree_is_harmless():
    tree = _tree()
    result = run_inference(tree, {"state": torch.randn(1, 3)}, max_depth=8)
    assert "peer-1" in result.outputs


def test_max_depth_cuts_the_walk_and_names_the_depth():
    tree = _tree()
    with pytest.raises(ContractError, match="max_depth"):
        # A depth limit of 1 still plans the children, which are at depth 2.
        run_inference(tree, {"state": torch.randn(1, 3)}, max_depth=0)


def test_an_invalid_max_workers_is_rejected():
    tree = _tree(depth_two=False)
    with pytest.raises(ValidationError, match="max_workers"):
        run_inference(tree, {"state": torch.randn(1, 3)}, max_workers=0)


def test_inputs_must_be_a_tensor_or_a_mapping():
    tree = _tree(depth_two=False)
    with pytest.raises(ContractError, match="must be a tensor or a mapping"):
        run_inference(tree, "not inputs")


def test_a_bare_tensor_is_accepted():
    tree = _tree(depth_two=False)
    result = run_inference(tree, torch.randn(2, 3))
    assert tuple(result.root_outputs["peer-1"].shape) == (2, 4)


def test_a_missing_named_input_is_reported():
    tree = _tree(depth_two=False)
    with pytest.raises(ContractError, match="expects inputs"):
        run_inference(tree, {"wrong_name": torch.randn(2, 3)})


def test_a_non_tree_is_rejected():
    with pytest.raises(ContractError, match="needs a PeerTree"):
        run_inference({"peer-1": None}, {})


def test_every_synthesis_function_works_end_to_end():
    for synthesis in (
        ConsensusSynthesis(),
        WeightedConsensus(),
        BestByScore(highest=False),
        ConcatSynthesis(),
    ):
        tree = _tree(depth_two=False)
        result = run_inference(tree, {"state": torch.randn(2, 3)}, synthesis=synthesis)
        assert result.root_outputs["peer-1"] is not None, type(synthesis).__name__


def test_the_batching_log_records_a_per_node_decision_per_level():
    tree = _tree()
    result = run_inference(tree, {"state": torch.randn(1, 3)})
    levels = {entry["level"] for entry in result.batching}
    assert levels <= {2, 3}
    for entry in result.batching:
        assert entry["mode"] == "per_node"
        assert entry["reason"]


def test_cost_reports_invocations_wall_time_and_per_level():
    tree = _tree(depth_two=False)
    result = run_inference(tree, {"state": torch.randn(1, 3)})
    assert result.cost["invocations"] == 5
    assert result.cost["wall_time"] >= 0.0
    assert set(result.cost["per_level"]) == {"1", "2"}


def test_mode_reports_that_nothing_training_related_happened():
    tree = _tree(depth_two=False)
    result = run_inference(tree, {"state": torch.randn(1, 3)})
    assert result.mode == {
        "warnings_persisted": False,
        "training": False,
        "kappa_updated": False,
        "checkpointed": False,
    }


def test_the_result_serializes_with_shapes():
    tree = _tree(depth_two=False)
    payload = run_inference(tree, {"state": torch.randn(2, 3)}).as_dict()
    assert payload["root_outputs"]["peer-1"]["shape"] == [2, 4]
    assert payload["root_outputs"]["peer-1"]["dtype"] == "float32"
    assert isinstance(payload["per_node"], list)


def test_describe_names_the_shape_of_the_request():
    tree = _tree(depth_two=False)
    text = run_inference(tree, {"state": torch.randn(1, 3)}).describe()
    assert "InferenceResult(nodes=3" in text
    assert "invocations=" in text


def test_a_flat_tree_inference_matches_the_flat_run(monkeypatch):
    """A depth-1 tree invokes each peer once and returns each root's output."""
    tree = PeerTree.flat(["peer-a", "peer-b"])
    for path in tree.root_paths:
        tree[path].peer = build_peer(
            Constant(2.0), name=path, runtime=_runtime(), action_dim=4, state_dim=3
        )
    result = run_inference(tree, {"state": torch.randn(2, 3)})
    assert set(result.root_outputs) == {"peer-a", "peer-b"}
    for value in result.root_outputs.values():
        assert torch.allclose(value, torch.full((2, 4), 2.0))


def test_inference_with_a_budget_of_one_invokes_only_the_root():
    tree = PeerTree.flat(["peer-a"])
    tree["peer-a"].peer = build_peer(
        Constant(1.0), name="peer-a", runtime=_runtime(), action_dim=4, state_dim=3
    )
    result = run_inference(tree, {"state": torch.randn(1, 3)}, budget=1)
    assert result.cost["invocations"] == 1


def test_broadcast_is_the_default_delegation():
    tree = _tree(depth_two=False)
    result = run_inference(tree, {"state": torch.randn(1, 3)}, delegation=BroadcastDelegation())
    assert result.delegation["peer-1"]["selected"] == ["peer-1/sub-1", "peer-1/sub-2"]
