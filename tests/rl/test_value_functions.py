"""Value-function wrappers (document 07 section 8.1)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mactf.exceptions import ContractError, ValidationError
from mactf.rl.functions.registry import build_function
from mactf.rl.functions.rollup import KappaWeightedRollup, MeanRollup
from mactf.rl.functions.value import (
    CallableValueFunction,
    CriticValueFunction,
    ImageValueFunction,
    ModuleValueFunction,
    NestedValueFunction,
    ReferenceValueFunction,
    ValueFunction,
    ZeroValueFunction,
)
from mactf.rl.value import ValueEstimate


class _Stateful(nn.Module):
    """Returns logits only, so the adapter synthesizes a value head."""

    def __init__(self, state_dim: int = 4, action_dim: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(state_dim, action_dim)

    def forward(self, state):
        return self.linear(state)


@pytest.fixture()
def adapter():
    from mactf.coopetition.adapter import PeerAdapter

    return PeerAdapter(_Stateful(), name="alpha")


def test_estimates_are_value_estimates(adapter):
    function = ModuleValueFunction(adapter)
    estimate = function(torch.randn(5, 4))
    assert isinstance(estimate, ValueEstimate)
    assert estimate.values.shape == (5, 1)


def test_module_value_function_matches_the_adapter_head(adapter):
    states = torch.randn(6, 4)
    function = ModuleValueFunction(adapter)
    _, expected = adapter(states)
    assert torch.allclose(function(states).values, expected, atol=1e-12)


def test_module_value_function_reports_the_adapter_head(adapter):
    function = ModuleValueFunction(adapter)
    function(torch.randn(2, 4))
    reported = list(function.parameters())
    assert reported == list(adapter.value_head.parameters())


def test_zero_strategy_adapter_yields_zeros():
    from mactf.coopetition.adapter import PeerAdapter

    frozen = PeerAdapter(_Stateful(), name="gamma", value_strategy="none")
    states = torch.randn(4, 4)
    estimate = ModuleValueFunction(frozen)(states)
    assert torch.allclose(estimate.values, torch.zeros(4, 1))
    assert list(ModuleValueFunction(frozen).parameters()) == []


def test_module_value_function_needs_an_adapter():
    with pytest.raises(ValidationError, match="PeerAdapter"):
        ModuleValueFunction(object())  # type: ignore[arg-type]


def test_critic_value_function_owns_its_parameters():
    critic = CriticValueFunction(hidden_dim=8)
    assert len(list(critic.parameters())) == 4
    estimate = critic(torch.randn(3, 8))
    assert estimate.values.shape == (3, 1)


def test_critic_value_function_rejects_a_missing_dimension():
    with pytest.raises(ValidationError, match="hidden_dim"):
        CriticValueFunction()


def test_critic_value_function_state_round_trip():
    critic = CriticValueFunction(hidden_dim=4)
    payload = critic.state_dict()
    other = CriticValueFunction(hidden_dim=4)
    other.load_state_dict(payload)
    states = torch.randn(2, 4)
    assert torch.allclose(critic(states).values, other(states).values)


def test_critic_load_state_rejects_a_foreign_class():
    with pytest.raises(ValueError, match="cannot load state"):
        CriticValueFunction(hidden_dim=4).load_state_dict({"class": "ZeroValueFunction"})


def test_reference_value_function_reports_no_parameters(adapter):
    function = ReferenceValueFunction(adapter)
    assert list(function.parameters()) == []


def test_reference_value_function_stays_frozen_even_if_asked_to_train(adapter):
    function = ReferenceValueFunction(adapter)
    function.train(True)
    assert function.training is False
    assert all(not parameter.requires_grad for parameter in adapter.parameters())


def test_reference_value_function_detaches(adapter):
    estimate = ReferenceValueFunction(adapter)(torch.randn(2, 4))
    assert estimate.values.requires_grad is False


def test_zero_value_function_shape_and_device():
    states = torch.zeros(3, 5)
    estimate = ZeroValueFunction()(states)
    assert estimate.values.shape == (3, 1)
    assert float(estimate.values.abs().sum()) == 0.0


def test_zero_value_function_without_trailing_dim():
    estimate = ZeroValueFunction(trailing_dim=False)(torch.zeros(3, 5))
    assert estimate.values.shape == (3,)


def test_image_value_function_pools_a_feature_map():
    function = ImageValueFunction(hidden_dim=7, pool="mean")
    estimate = function(torch.randn(2, 3, 4, 7))
    assert estimate.values.shape == (2, 1)


def test_image_value_function_rejects_a_flat_tensor():
    with pytest.raises(ContractError, match="expects"):
        ImageValueFunction(hidden_dim=4)(torch.randn(2, 4))


def test_image_value_function_rejects_an_unknown_pool():
    with pytest.raises(ValidationError, match="unknown pool"):
        ImageValueFunction(hidden_dim=4, pool="median")


def test_nested_value_function_rolls_children_up():
    children = {
        "a": CriticValueFunction(hidden_dim=4),
        "b": CriticValueFunction(hidden_dim=4),
    }
    function = NestedValueFunction(children, rollup=MeanRollup(), parent="root")
    estimate = function(torch.randn(3, 4))
    assert estimate.values.shape == (3, 1)


def test_nested_value_function_collects_child_parameters():
    children = {"a": CriticValueFunction(hidden_dim=4)}
    function = NestedValueFunction(children)
    assert list(function.parameters()) == list(children["a"].parameters())


def test_nested_value_function_without_children_is_zero():
    function = NestedValueFunction({})
    estimate = function(torch.randn(2, 4))
    assert float(estimate.values.abs().sum()) == 0.0


def test_nested_value_function_state_round_trip():
    function = NestedValueFunction({"a": CriticValueFunction(hidden_dim=4)}, parent="p")
    payload = function.state_dict()
    restored = NestedValueFunction({"a": CriticValueFunction(hidden_dim=4)}, parent="p")
    restored.load_state_dict(payload)
    assert restored.parent == "p"
    assert restored.describe()["children"] == ["a"]


def test_nested_value_function_rejects_a_bad_rollup():
    with pytest.raises(ValidationError, match="rollup"):
        NestedValueFunction({}, rollup=object())


def test_callable_value_function_adapts_a_tensor():
    function = CallableValueFunction(lambda states: states.sum(dim=-1, keepdim=True))
    estimate = function(torch.ones(2, 3))
    assert torch.allclose(estimate.values, torch.full((2, 1), 3.0))


def test_callable_value_function_accepts_an_estimate():
    function = CallableValueFunction(lambda states: ValueEstimate(values=states[:, :1]))
    assert function(torch.ones(2, 3)).values.shape == (2, 1)


def test_callable_value_function_rejects_a_non_tensor():
    function = CallableValueFunction(lambda states: "nope")
    with pytest.raises(ContractError, match="expected a tensor"):
        function(torch.ones(2, 3))


def test_callable_value_function_requires_a_callable():
    with pytest.raises(ValidationError, match="requires an 'fn' callable"):
        CallableValueFunction()


def test_every_wrapper_satisfies_the_protocol(adapter):
    for function in (
        ModuleValueFunction(adapter),
        CriticValueFunction(hidden_dim=4),
        ReferenceValueFunction(adapter),
        ZeroValueFunction(),
        ImageValueFunction(hidden_dim=3),
        NestedValueFunction({}),
        CallableValueFunction(lambda states: states),
    ):
        assert isinstance(function, ValueFunction), type(function).__name__


def test_train_and_eval_propagate(adapter):
    critic = CriticValueFunction(hidden_dim=4)
    assert critic.train(True).training is True
    assert critic.eval().training is False
    assert not critic.head.training


def test_value_wrappers_build_from_spec(adapter):
    stage = build_function({"type": "zero"}, family="value")
    assert isinstance(stage, ZeroValueFunction)

    stage = build_function(
        {"type": "nested", "children": {"a": ZeroValueFunction()}, "parent": "x"},
        family="value",
    )
    assert isinstance(stage, NestedValueFunction)

    stage = ZeroValueFunction.from_spec({"type": "zero"})
    assert isinstance(stage, ZeroValueFunction)


def test_value_spec_cross_family_type_is_rejected():
    with pytest.raises(ValidationError, match="cannot build type"):
        ZeroValueFunction.from_spec({"type": "critic"})


def test_describe_reports_parameters():
    payload = CriticValueFunction(hidden_dim=8).describe()
    assert payload["type"] == "critic"
    assert payload["parameters"] > 0
