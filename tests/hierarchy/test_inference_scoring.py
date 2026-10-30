"""Scoring an inference request (document 10 section 7.5).

An inference request is observational: it may be audited with the same rules and
the same non-offsettable floor as training, but it must not advance a training
warning registry unless the caller explicitly asks for that.
"""

from __future__ import annotations

import pytest
import torch

from mactf.coopetition.peer import build_peer
from mactf.coopetition.tree import HierarchySpec, LevelSpec, PeerTree
from mactf.exceptions import ContractError
from mactf.orchestration.episode import Task
from mactf.runtime import Runtime
from mactf.scoring.hierarchy import HierarchicalScorer, run_inference
from mactf.scoring.pipeline import ScoringPipeline


class Fixed(torch.nn.Module):
    """Emits a one-hot distribution, so the judge's accuracy is predictable."""

    def __init__(self, correct: bool = True) -> None:
        super().__init__()
        self.correct = bool(correct)
        self.bias = torch.nn.Parameter(torch.zeros(3))

    def forward(self, state):
        logits = torch.full((state.shape[0], 3), -6.0)
        column = 1 if self.correct else 0
        logits[:, column] = 6.0
        return logits + self.bias


def _runtime() -> Runtime:
    return Runtime(
        device=torch.device("cpu"),
        dtype=torch.float32,
        amp=False,
        seed=0,
        deterministic=True,
    )


def _flat_tree(correct: bool = True) -> PeerTree:
    tree = PeerTree.flat(["peer-a"])
    tree["peer-a"].peer = build_peer(
        Fixed(correct), name="peer-a", runtime=_runtime(), action_dim=3, state_dim=3
    )
    return tree


def _task() -> Task:
    return Task(
        id="t1",
        states=torch.randn(2, 3),
        targets=torch.ones(2, dtype=torch.long),
    )


def test_score_false_returns_no_scorecard():
    result = run_inference(_flat_tree(), {"state": _task().states}, task=_task())
    assert result.scorecard is None


def test_score_true_without_a_task_still_scores_the_tree():
    result = run_inference(_flat_tree(), {"state": _task().states}, score=True)
    assert result.scorecard is not None
    assert set(result.scorecard.peers) == {"peer-a"}


def test_a_correct_answer_records_the_completion_reward():
    task = _task()
    result = run_inference(
        _flat_tree(correct=True), {"state": task.states}, task=task, score=True
    )
    card = result.scorecard.peers["peer-a"]
    assert "R1" in card.continuous.reward.rules
    assert card.continuous.reward.r_task > 0.0


def test_an_incorrect_answer_does_not_record_the_completion_reward():
    task = _task()
    result = run_inference(
        _flat_tree(correct=False), {"state": task.states}, task=task, score=True
    )
    card = result.scorecard.peers["peer-a"]
    assert "R1" not in card.continuous.reward.rules


def test_the_scorecard_is_marked_as_inference():
    result = run_inference(_flat_tree(), {"state": _task().states}, score=True)
    assert result.scorecard.metadata["inference"] is True
    assert result.scorecard.metadata["persist_warnings"] is False
    assert result.scorecard.metadata["roots"] == ["peer-a"]


def test_a_hierarchical_inference_scorecard_covers_every_node():
    spec = HierarchySpec(levels=(LevelSpec("sub", 2),), max_depth=3)
    tree = spec.build(peer_factory=lambda path, depth, level: build_peer(
        Fixed(), name=path, runtime=_runtime(), action_dim=3, state_dim=3
    ))
    result = run_inference(tree, {"state": torch.randn(2, 3)}, score=True)
    assert set(result.scorecard.peers) == {
        "peer-1",
        "peer-1/sub-1",
        "peer-1/sub-2",
    }


def test_warnings_are_not_persisted_by_default():
    tree = _flat_tree()
    scorer = HierarchicalScorer(ScoringPipeline())
    run_inference(tree, {"state": torch.randn(1, 3)}, score=True, scorer=scorer)
    # The caller's pipeline was not used at all, so its registries are untouched.
    assert scorer.pipeline.warning_scorer.registries == {}


def test_the_mode_reports_warnings_as_not_persisted():
    result = run_inference(_flat_tree(), {"state": torch.randn(1, 3)}, score=True)
    assert result.mode["warnings_persisted"] is False
    assert result.mode["training"] is False
    assert result.mode["kappa_updated"] is False
    assert result.mode["checkpointed"] is False


def test_persist_warnings_is_recorded_when_asked_for():
    result = run_inference(
        _flat_tree(), {"state": torch.randn(1, 3)}, score=True, persist_warnings=True
    )
    assert result.mode["warnings_persisted"] is True
    assert result.scorecard.metadata["persist_warnings"] is True


def test_a_supplied_pipeline_is_used_when_it_is_a_pipeline():
    pipeline = ScoringPipeline()
    result = run_inference(
        _flat_tree(), {"state": torch.randn(1, 3)}, score=True, scorer=pipeline
    )
    assert result.scorecard is not None


def test_nothing_else_about_the_result_changes_when_scoring():
    tree = _flat_tree()
    inputs = {"state": torch.randn(2, 3)}
    plain = run_inference(tree, inputs)
    scored = run_inference(tree, inputs, score=True)
    assert torch.equal(plain.outputs["peer-a"], scored.outputs["peer-a"])


def test_a_ledger_is_built_for_every_node_even_without_a_task():
    spec = HierarchySpec(levels=(LevelSpec("sub", 2),), max_depth=3)
    tree = spec.build(peer_factory=lambda path, depth, level: build_peer(
        Fixed(), name=path, runtime=_runtime(), action_dim=3, state_dim=3
    ))
    result = run_inference(tree, {"state": torch.randn(2, 3)}, score=True)
    assert len(result.scorecard.peers) == 3


def test_a_failed_node_is_recorded_as_penalty_evidence_when_scoring():
    tree = _flat_tree()

    class Boom(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(3))

        def forward(self, state):
            raise RuntimeError("kaboom")

    tree["peer-a"].peer = build_peer(
        Boom(), name="peer-a", runtime=_runtime(), action_dim=3, state_dim=3
    )
    result = run_inference(tree, {"state": torch.randn(1, 3)}, score=True)
    card = result.scorecard.peers["peer-a"]
    assert "P23" in card.continuous.penalty.amounts
    assert card.continuous.penalty.amounts["P23"] > 0.0
    assert card.final_score < 0.0


def test_a_failed_node_does_not_produce_an_output():
    tree = _flat_tree()

    class Boom(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(3))

        def forward(self, state):
            raise RuntimeError("nope")

    tree["peer-a"].peer = build_peer(
        Boom(), name="peer-a", runtime=_runtime(), action_dim=3, state_dim=3
    )
    result = run_inference(tree, {"state": torch.randn(1, 3)})
    assert "peer-a" not in result.outputs
    assert result.root_outputs["peer-a"] is None
    assert result.per_node[0].error.endswith("nope")


def test_scoring_a_request_with_no_outputs_still_produces_a_card():
    tree = _flat_tree()

    class Boom(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(3))

        def forward(self, state):
            raise RuntimeError("nope")

    tree["peer-a"].peer = build_peer(
        Boom(), name="peer-a", runtime=_runtime(), action_dim=3, state_dim=3
    )
    result = run_inference(tree, {"state": torch.randn(1, 3)}, score=True)
    assert result.scorecard is not None
    assert len(result.scorecard.peers) == 1


def test_the_scorecard_survives_serialization():
    result = run_inference(_flat_tree(), {"state": _task().states}, task=_task(), score=True)
    payload = result.as_dict()
    assert payload["scorecard"] is not None
    assert "peers" in payload["scorecard"] or "peer_scores" in payload["scorecard"]


def test_an_empty_tree_cannot_be_scored():
    tree = PeerTree()
    with pytest.raises(ContractError):
        run_inference(tree, {"state": torch.randn(1, 3)}, score=True)

