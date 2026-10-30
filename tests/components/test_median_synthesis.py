"""The worked synthesis of section 8.9.4 and the tensor/node bridge of 8.7."""

from __future__ import annotations

import pytest
import torch

from mactf.components import register_function
from mactf.components.contracts import synthesis_contract, synthesis_instance_contract
from mactf.coopetition.synthesis import (
    ConcatSynthesis,
    ConsensusSynthesis,
    TensorSynthesisAdapter,
    adapt_synthesis,
    build_synthesis,
    synthesis_capabilities,
)
from mactf.exceptions import ContractError, ValidationError

from p2_support import MedianSynthesis, configure, hierarchy_trainer, tune


class _Node:
    path = "root"


def test_adapter_runs_the_node_level_call() -> None:
    adapter = TensorSynthesisAdapter(MedianSynthesis())
    proposals = {
        "a": torch.tensor([[1.0, 5.0]]),
        "b": torch.tensor([[2.0, 4.0]]),
        "c": torch.tensor([[9.0, 0.0]]),
    }
    out = adapter.synthesize(_Node(), proposals, scores={"a": 1.0, "b": 1.0, "c": 1.0}, weights=None)
    assert torch.equal(out, torch.tensor([[2.0, 4.0]]))


def test_registered_name_resolves_through_build_synthesis(component_overlay) -> None:
    register_function("synthesis", "median", MedianSynthesis, scope="run")
    built = build_synthesis("median")
    assert isinstance(built, TensorSynthesisAdapter)
    assert built.describe()["function"] == "MedianSynthesis"


def test_median_synthesis(settings, tmp_path, component_overlay) -> None:
    """The exit criterion: ``RL.SYNTHESIS`` reaches the trainer and a parent uses the median."""
    register_function("synthesis", "median", MedianSynthesis, scope="run")
    tune(settings)
    configure(settings, RL={"SYNTHESIS": {"type": "median"}})
    trainer = hierarchy_trainer(settings, tmp_path).build()
    parent = trainer.node_paths[0]
    assert isinstance(trainer.synthesis_for(parent), TensorSynthesisAdapter)
    assert isinstance(trainer.synthesis_function, ConsensusSynthesis)          # the in-code default stays
    seen = {}
    original = TensorSynthesisAdapter.synthesize

    def spy(self, node, proposals, **kwargs):
        out = original(self, node, proposals, **kwargs)
        seen[node.path] = (dict(proposals), out)
        return out

    TensorSynthesisAdapter.synthesize = spy
    try:
        trainer.run_epoch(0)
    finally:
        TensorSynthesisAdapter.synthesize = original
    assert seen, "the registered synthesis was never called"
    for proposals, out in seen.values():
        expected = torch.stack([proposals[key] for key in sorted(proposals)]).median(dim=0).values
        assert torch.equal(out, expected)


def test_a_per_parent_override_beats_the_default(settings, tmp_path, component_overlay) -> None:
    register_function("synthesis", "median", MedianSynthesis, scope="run")
    tune(settings)
    configure(settings, RL={"SYNTHESIS": {"type": "median"},
                            "OVERRIDES": {"SYNTHESIS": {"peer-1": {"type": "weighted"}}}})
    trainer = hierarchy_trainer(settings, tmp_path).build()
    assert type(trainer.synthesis_for("peer-1")).__name__ == "WeightedConsensus"
    assert isinstance(trainer.synthesis_for("peer-2"), TensorSynthesisAdapter)


def test_synthesis_failure_is_node_local(settings, tmp_path, component_overlay) -> None:
    """R6: a synthesis that raises fails that parent, with a penalty, and not the episode."""

    class Boom:
        def __call__(self, proposals, *, weights=None):
            raise RuntimeError("synthesis blew up")

    register_function("synthesis", "boom", Boom, scope="run")
    tune(settings)
    configure(settings, RL={"VERIFY": "off", "OVERRIDES": {"SYNTHESIS": {"peer-1": {"type": "boom"}}}})
    trainer = hierarchy_trainer(settings, tmp_path).build()
    result = trainer.run_epoch(0)
    assert result.completed
    assert any("peer-1" in failure and "blew up" in failure for failure in trainer.failures)
    ledger = trainer.episodes[-1].ledgers["peer-1"]
    assert "P23" in ledger.triggered_ids()
    assert not any(failure.startswith("peer-2") for failure in trainer.failures)


# -- build_synthesis -------------------------------------------------------------------------------
def test_e046_a_custom_name_may_not_shadow_a_node_level_builtin(component_overlay) -> None:
    register_function("synthesis", "concat", MedianSynthesis, scope="run")
    with pytest.raises(ValidationError, match="E046"):
        build_synthesis("concat")


def test_unknown_names_list_both_layers(component_overlay) -> None:
    register_function("synthesis", "median", MedianSynthesis, scope="run")
    with pytest.raises(ValidationError, match=r"unknown synthesis type 'mode'.*median.*weighted"):
        build_synthesis("mode")


def test_builtin_names_and_unknown_keys_keep_their_messages() -> None:
    assert isinstance(build_synthesis(None), ConsensusSynthesis)
    assert isinstance(build_synthesis("mean"), ConsensusSynthesis)           # the alias
    with pytest.raises(ValidationError, match="does not accept 'nope'"):
        build_synthesis({"type": "weighted", "nope": 1})
    with pytest.raises(ValidationError, match="requires a 'type' key"):
        build_synthesis({})


def test_an_instance_is_adapted_and_a_bare_function_is_refused() -> None:
    assert isinstance(build_synthesis(MedianSynthesis()), TensorSynthesisAdapter)
    node_level = ConsensusSynthesis()
    assert build_synthesis(node_level) is node_level
    with pytest.raises(ValidationError, match="ambiguous"):
        adapt_synthesis(lambda proposals, *, weights=None: proposals)
    with pytest.raises(ContractError, match="not a synthesis function"):
        adapt_synthesis(42)


def test_an_import_spec_is_adapted(component_overlay) -> None:
    built = build_synthesis({"import": "p2_support:MedianSynthesis"})
    assert isinstance(built, TensorSynthesisAdapter)


def test_capabilities_report_level_and_origin(component_overlay) -> None:
    register_function("synthesis", "median", MedianSynthesis, scope="run")
    entries = {entry["type"]: entry for entry in synthesis_capabilities()}
    assert entries["weighted"]["level"] == "node" and entries["weighted"]["origin"] == "builtin"
    assert entries["median"]["level"] == "tensor" and entries["median"]["origin"] == "project"
    assert {"type", "class", "kwargs"} <= set(entries["median"])


# -- the adapter's rules -----------------------------------------------------------------------------
def test_the_adapter_offers_only_the_keywords_the_function_declares() -> None:
    received = {}

    class Picky:
        def __call__(self, proposals, *, scores, node, task=None):
            received.update(scores=scores, node=node, task=task)
            return next(iter(proposals.values()))

    adapter = TensorSynthesisAdapter(Picky())
    adapter.synthesize(_Node(), {"a": torch.zeros(1, 2)}, scores={"a": 1.0}, weights={"a": 0.5}, task="t")
    assert received == {"scores": {"a": 1.0}, "node": received["node"], "task": "t"}


def test_weights_fall_back_to_scores_when_the_function_takes_weights() -> None:
    seen = {}

    class W:
        def __call__(self, proposals, *, weights=None):
            seen["weights"] = weights
            return next(iter(proposals.values()))

    TensorSynthesisAdapter(W()).synthesize(_Node(), {"a": torch.zeros(1, 2)}, scores={"a": 0.9}, weights=None)
    assert seen["weights"] == {"a": 0.9}


def test_the_adapter_checks_the_result() -> None:
    class Wrong:
        def __call__(self, proposals, *, weights=None):
            return torch.zeros(1, 7)

    with pytest.raises(ContractError, match=r"expected \(1, 2\)"):
        TensorSynthesisAdapter(Wrong()).synthesize(_Node(), {"a": torch.zeros(1, 2)})

    class NotATensor:
        def __call__(self, proposals, *, weights=None):
            return [1, 2]

    with pytest.raises(ContractError, match="expected a tensor"):
        TensorSynthesisAdapter(NotATensor()).synthesize(_Node(), {"a": torch.zeros(1, 2)})

    class Inf:
        def __call__(self, proposals, *, weights=None):
            return torch.full((1, 2), float("inf"))

    with pytest.raises(ContractError, match="non-finite"):
        TensorSynthesisAdapter(Inf()).synthesize(_Node(), {"a": torch.zeros(1, 2)})
    with pytest.raises(ContractError, match="no proposals"):
        TensorSynthesisAdapter(MedianSynthesis()).synthesize(_Node(), {})


def test_a_free_output_shape_is_allowed_only_when_declared() -> None:
    class Joiner:
        output_shape_rule = "free"

        def __call__(self, proposals, *, weights=None):
            return torch.cat(list(proposals.values()), dim=-1)

    out = TensorSynthesisAdapter(Joiner()).synthesize(
        _Node(), {"a": torch.zeros(2, 3), "b": torch.zeros(2, 3)})
    assert out.shape == (2, 6)
    with pytest.raises(ValidationError, match="output_shape_rule"):
        TensorSynthesisAdapter(type("Bad", (), {"output_shape_rule": "sometimes",
                                                "__call__": lambda self, p, *, weights=None: p})())
    assert ConcatSynthesis.output_shape_rule == "free"


# -- the contract accepts either entry method ---------------------------------------------------------------
def test_synthesis_contract_accepts_either_entry_method_and_nothing_else() -> None:
    class NodeLevel:
        def synthesize(self, node, proposals, *, scores=None, weights=None, task=None):
            return proposals

    class Neither:
        pass

    class BadSignature:
        def synthesize(self, proposals):
            return proposals

    assert synthesis_contract(MedianSynthesis) == []
    assert synthesis_contract(NodeLevel) == []
    assert "define synthesize" in synthesis_contract(Neither)[0]
    assert "synthesize() must take (node, proposals" in synthesis_contract(BadSignature)[0]
    assert synthesis_instance_contract(MedianSynthesis()) == [] and synthesis_instance_contract(object())
