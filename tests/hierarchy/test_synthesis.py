"""Synthesis functions (document 10 section 4.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

from mactf.coopetition.synthesis import (
    SYNTHESIS_TYPES,
    BestByScore,
    ConcatSynthesis,
    ConsensusSynthesis,
    ExplicitSynthesis,
    LearnedSynthesizer,
    WeightedConsensus,
    build_synthesis,
    check_aligned,
    synthesis_capabilities,
)
from mactf.coopetition.tree import HierarchySpec, LevelSpec
from mactf.exceptions import ContractError, ValidationError
from mactf.orchestration.episode import consensus


@dataclass
class FakeTask:
    id: str = "task-1"
    metadata: dict[str, Any] = field(default_factory=dict)


def _node():
    tree = HierarchySpec(levels=(LevelSpec("sub", 2),)).build()
    return tree, tree["peer-1"]


def _proposals(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return {
        "peer-1/sub-1": torch.randn(3, 4, generator=generator),
        "peer-1/sub-2": torch.randn(3, 4, generator=generator),
    }


def test_consensus_equals_the_existing_helper_at_1e_12():
    _, node = _node()
    proposals = _proposals()
    actual = ConsensusSynthesis().synthesize(node, proposals)
    expected = consensus(proposals)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=0.0)


def test_consensus_is_the_default():
    assert build_synthesis(None).name == "consensus"


def test_weighted_consensus_respects_the_weights():
    _, node = _node()
    proposals = {
        "a": torch.ones(2, 2),
        "b": torch.zeros(2, 2),
    }
    result = WeightedConsensus().synthesize(
        node, proposals, weights={"a": 0.75, "b": 0.25}
    )
    assert torch.allclose(result, torch.full((2, 2), 0.75))


def test_weighted_consensus_falls_back_to_uniform():
    _, node = _node()
    proposals = {"a": torch.ones(2), "b": torch.zeros(2)}
    result = WeightedConsensus().synthesize(node, proposals)
    assert torch.allclose(result, torch.full((2,), 0.5))


def test_best_by_score_selects_one_proposal():
    _, node = _node()
    proposals = {"a": torch.ones(2), "b": torch.zeros(2)}
    result = BestByScore().synthesize(node, proposals, scores={"a": 0.1, "b": 0.9})
    assert torch.equal(result, proposals["b"])


def test_best_by_score_breaks_ties_by_path():
    _, node = _node()
    proposals = {"b": torch.ones(2), "a": torch.zeros(2)}
    result = BestByScore().synthesize(node, proposals, scores={"a": 0.5, "b": 0.5})
    assert torch.equal(result, proposals["a"])


def test_learned_synthesizer_with_n_proposals_is_trainable():
    _, node = _node()
    proposals = _proposals()
    synthesizer = LearnedSynthesizer(n_proposals=2)
    assert len(synthesizer.parameters()) == 1
    result = synthesizer.synthesize(node, proposals)
    stacked = torch.stack(list(proposals.values()), dim=0)
    assert torch.allclose(result, stacked.mean(dim=0), atol=1e-7)


def test_learned_synthesizer_accepts_a_module():
    class Blender(torch.nn.Module):
        def forward(self, stacked):
            return stacked.mean(dim=0)

    _, node = _node()
    proposals = _proposals()
    synthesizer = LearnedSynthesizer(Blender())
    assert torch.allclose(synthesizer.synthesize(node, proposals), consensus(proposals))


def test_learned_synthesizer_requires_a_module_or_a_count():
    with pytest.raises(ValidationError, match="either a 'module' or 'n_proposals'"):
        LearnedSynthesizer()


def test_learned_synthesizer_rejects_a_bad_module():
    with pytest.raises(ValidationError, match="no forward"):
        LearnedSynthesizer(object())


def test_learned_synthesizer_rejects_the_wrong_proposal_count():
    _, node = _node()
    synthesizer = LearnedSynthesizer(n_proposals=3)
    with pytest.raises(ContractError, match="built for 3"):
        synthesizer.synthesize(node, _proposals())


def test_concat_synthesizer_joins_the_feature_axis():
    _, node = _node()
    proposals = {"a": torch.ones(2, 3), "b": torch.zeros(2, 5)}
    result = ConcatSynthesis().synthesize(node, proposals)
    assert tuple(result.shape) == (2, 8)


def test_concat_synthesizer_still_requires_matching_batches():
    _, node = _node()
    proposals = {"a": torch.ones(2, 3), "b": torch.zeros(3, 3)}
    with pytest.raises(ContractError, match="batch"):
        ConcatSynthesis().synthesize(node, proposals)


def test_explicit_synthesizer_calls_the_function():
    seen: dict[str, Any] = {}

    def merge(node, proposals, *, scores, weights, task):
        seen["node"] = node.path
        seen["task"] = task
        return torch.stack(list(proposals.values())).sum(dim=0)

    _, node = _node()
    result = ExplicitSynthesis(merge).synthesize(node, _proposals(), task=FakeTask())
    assert seen["node"] == "peer-1"
    assert seen["task"].id == "task-1"
    assert tuple(result.shape) == (3, 4)


def test_explicit_synthesizer_requires_a_callable():
    with pytest.raises(ValidationError, match="requires an 'fn'"):
        ExplicitSynthesis()


def test_explicit_synthesizer_rejects_a_non_tensor_result():
    _, node = _node()
    with pytest.raises(ContractError, match="expected a tensor"):
        ExplicitSynthesis(lambda *args, **kwargs: "nope").synthesize(node, _proposals())


def test_empty_proposals_raise_rather_than_returning_zeros():
    _, node = _node()
    for function in (
        ConsensusSynthesis(),
        WeightedConsensus(),
        BestByScore(),
        ConcatSynthesis(),
    ):
        with pytest.raises(ContractError, match="no proposals to synthesize for peer-1"):
            function.synthesize(node, {})


def test_shape_mismatch_is_named():
    _, node = _node()
    proposals = {"a": torch.zeros(2, 3), "b": torch.zeros(2, 4)}
    with pytest.raises(ContractError, match="trailing shape"):
        check_aligned(proposals, node=node.path)


def test_batch_mismatch_is_named():
    _, node = _node()
    proposals = {"a": torch.zeros(2, 3), "b": torch.zeros(5, 3)}
    with pytest.raises(ContractError, match="batch"):
        check_aligned(proposals, node=node.path)


def test_a_non_tensor_proposal_is_named():
    with pytest.raises(ContractError, match="not a tensor"):
        check_aligned({"a": [[1.0]]}, node="peer-1")


def test_registry_aliases_resolve():
    assert build_synthesis("weighted_consensus").name == "weighted"
    assert build_synthesis("best_by_score").name == "best"
    assert build_synthesis("mean").name == "consensus"


def test_registry_rejects_an_unknown_type():
    with pytest.raises(ValidationError, match="unknown synthesis type") as info:
        build_synthesis("divination")
    assert "consensus" in str(info.value)


def test_registry_rejects_an_unknown_kwarg():
    with pytest.raises(ValidationError, match="does not accept"):
        build_synthesis({"type": "concat", "axis": 0})


def test_registry_passes_every_type():
    for name in SYNTHESIS_TYPES:
        spec: dict[str, Any] = {"type": name}
        if name == "explicit":
            spec["fn"] = lambda *args, **kwargs: torch.zeros(())
        if name == "learned":
            spec["n_proposals"] = 2
        built = build_synthesis(spec)
        assert type(built).__name__ == SYNTHESIS_TYPES[name].__name__


def test_capabilities_lists_every_function():
    entries = {entry["type"]: entry for entry in synthesis_capabilities()}
    assert set(entries) == set(SYNTHESIS_TYPES)
    assert "weights" in entries["weighted"]["kwargs"]
    assert "dim" in entries["concat"]["kwargs"]


def test_describe_is_serializable():
    payload = ConcatSynthesis(dim=1).describe()
    assert payload["type"] == "concat"
    assert payload["dim"] == 1
