"""Advantage-function wrappers (document 07 section 8.2), at ``1e-12``."""

from __future__ import annotations

import pytest
import torch

from mactf.exceptions import ContractError, ValidationError
from mactf.rl.advantage import (
    generalized_advantage,
    penalise_void_advantage,
    relative_to_reference,
    temporal_difference,
)
from mactf.rl.functions.advantage import (
    ADVANTAGE_FUNCTION_TYPES,
    AdvantageFunction,
    CallableAdvantage,
    ChainAdvantage,
    GAEAdvantage,
    NormalisedAdvantage,
    ReferenceBaselineAdvantage,
    TDAdvantage,
    VoidMaskedAdvantage,
)
from mactf.rl.functions.registry import build_function
from mactf.rl.value import normalise_advantages


def test_td_matches_the_free_function():
    rewards = torch.tensor([1.0, -0.5, 2.0])
    values = torch.tensor([0.1, 0.2, -0.3])
    next_values = torch.tensor([0.4, 0.5, 0.6])
    batch = TDAdvantage(gamma=0.9)(rewards, values, next_values=next_values)
    expected = temporal_difference(rewards, values, next_values, gamma=0.9)
    assert torch.allclose(batch.advantages, expected, atol=1e-12)


def test_td_requires_next_values():
    with pytest.raises(ValueError, match="next_values"):
        TDAdvantage()(torch.zeros(2), torch.zeros(2))


def test_td_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        TDAdvantage()(torch.zeros(2), torch.zeros(3), next_values=torch.zeros(2))


def test_td_zeroes_masked_positions():
    batch = TDAdvantage()(
        torch.tensor([1.0, 1.0]),
        torch.zeros(2),
        next_values=torch.zeros(2),
        masks=torch.tensor([True, False]),
    )
    assert batch.advantages[1].item() == pytest.approx(0.0)


def test_gae_matches_the_free_function_at_1e_12():
    rewards = [1.0, 0.5, -1.0, 2.0]
    values = [0.2, 0.1, 0.0, -0.4]
    batch = GAEAdvantage(gamma=0.99, lam=0.95)(rewards, values)
    expected = generalized_advantage(rewards, values, gamma=0.99, lam=0.95)
    assert torch.allclose(batch.advantages, expected.advantages, atol=1e-12)
    assert torch.allclose(batch.returns, expected.returns, atol=1e-12)


def test_gae_honours_a_bootstrap_value():
    batch = GAEAdvantage()([1.0], [0.5], next_values=torch.tensor([2.0]))
    expected = generalized_advantage([1.0], [0.5], next_value=2.0)
    assert torch.allclose(batch.advantages, expected.advantages, atol=1e-12)


def test_gae_rejects_a_vector_bootstrap():
    with pytest.raises(ValueError, match="single next value"):
        GAEAdvantage()([1.0], [0.5], next_values=torch.tensor([1.0, 2.0]))


def test_gae_validates_gamma_and_lambda():
    with pytest.raises(ValueError, match="gamma"):
        GAEAdvantage()([1.0], [0.5], gamma=1.5)
    with pytest.raises(ValueError, match="lam"):
        GAEAdvantage()([1.0], [0.5], lam=-0.1)


def test_gae_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        GAEAdvantage()([1.0, 2.0], [0.5])


def test_gae_masks_advantages_to_zero():
    batch = GAEAdvantage()(
        [1.0, 1.0], [0.0, 0.0], masks=torch.tensor([True, False])
    )
    assert batch.advantages[1].item() == pytest.approx(0.0)


def test_reference_baseline_matches_relative_to_reference():
    peer = torch.tensor([1.0, 2.0])
    reference = torch.tensor([0.5, 0.5])
    batch = ReferenceBaselineAdvantage()(peer, torch.zeros(2), reference=reference)
    expected = relative_to_reference(peer, reference)
    assert torch.allclose(batch.advantages, expected, atol=1e-12)


def test_reference_baseline_requires_a_reference():
    with pytest.raises(ValueError, match="reference"):
        ReferenceBaselineAdvantage()(torch.zeros(2), torch.zeros(2))


def test_void_masked_matches_penalise_void_advantage():
    advantages = torch.tensor([5.0, 1.0, -3.0])
    mask = torch.tensor([True, False, True])
    batch = VoidMaskedAdvantage(floor=-1.0)(advantages, torch.zeros(3), masks=mask)
    expected = penalise_void_advantage(advantages, mask, floor=-1.0)
    assert torch.allclose(batch.advantages, expected, atol=1e-12)


def test_void_masked_requires_a_mask():
    with pytest.raises(ValueError, match="masks"):
        VoidMaskedAdvantage()(torch.zeros(2), torch.zeros(2))


def test_void_masked_rejects_a_positive_floor():
    with pytest.raises(ValidationError, match="positive"):
        VoidMaskedAdvantage(floor=1.0)


def test_normalised_matches_the_free_function():
    advantages = torch.tensor([1.0, 3.0, 5.0, 7.0])
    batch = NormalisedAdvantage()(advantages, torch.zeros(4))
    expected = normalise_advantages(advantages)
    assert torch.allclose(batch.advantages, expected, atol=1e-12)


def test_normalised_rejects_a_bad_eps():
    with pytest.raises(ValidationError, match="eps"):
        NormalisedAdvantage(eps=0.0)


def test_chain_of_reference_then_void_matches_the_local_update_path():
    """The equivalence gate from doc 07 section 11."""
    signal = torch.tensor([2.0, 2.0, 2.0, 2.0])
    void = torch.tensor([False, True, False, False])

    # What phases.local_update does:
    expected = penalise_void_advantage(signal, void, floor=-1.0)

    chain = ChainAdvantage(
        [ReferenceBaselineAdvantage(), VoidMaskedAdvantage(floor=-1.0)]
    )
    batch = chain(signal, torch.zeros(4), reference=torch.zeros(4), masks=void)
    assert torch.allclose(batch.advantages, expected, atol=1e-12)


def test_chain_applies_steps_in_order():
    chain = ChainAdvantage([NormalisedAdvantage(), VoidMaskedAdvantage()])
    batch = chain(
        torch.tensor([1.0, 3.0, 5.0]), torch.zeros(3), masks=torch.tensor([True, False, True])
    )
    assert batch.advantages[1].item() == pytest.approx(0.0)


def test_chain_rejects_a_shape_change():
    class Widening:
        def __call__(self, rewards, values, **kwargs):
            from mactf.rl.advantage import AdvantageBatch

            return AdvantageBatch(
                advantages=torch.zeros(rewards.numel() + 1), returns=torch.zeros(rewards.numel() + 1)
            )

        def describe(self):
            return {"type": "widening"}

    with pytest.raises(ContractError, match="changed the advantage shape"):
        ChainAdvantage([Widening()])(torch.zeros(3), torch.zeros(3))


def test_chain_without_steps_is_the_identity():
    batch = ChainAdvantage([])(torch.tensor([1.0, 2.0]), torch.zeros(2))
    assert torch.allclose(batch.advantages, torch.tensor([1.0, 2.0]))


def test_chain_requires_describe_in_strict_mode():
    with pytest.raises(ValidationError, match="describe"):
        ChainAdvantage([lambda rewards, values, **kwargs: None])


def test_chain_from_spec_builds_nested_specs():
    chain = ChainAdvantage.from_spec(
        {
            "type": "chain",
            "steps": [
                {"type": "reference"},
                {"type": "void_masked", "floor": -2.0},
            ],
        }
    )
    assert isinstance(chain, ChainAdvantage)
    assert len(chain.steps) == 2
    assert chain.steps[1].floor == pytest.approx(-2.0)


def test_chain_from_spec_requires_steps():
    with pytest.raises(ValidationError, match="steps"):
        ChainAdvantage.from_spec({"type": "chain"})


def test_callable_advantage_adapts_a_tensor():
    function = CallableAdvantage(lambda rewards, values, **kwargs: rewards * 2.0)
    batch = function(torch.tensor([1.0, 2.0]), torch.zeros(2))
    assert torch.allclose(batch.advantages, torch.tensor([2.0, 4.0]))


def test_callable_advantage_requires_a_callable():
    with pytest.raises(ValidationError, match="requires an 'fn'"):
        CallableAdvantage()


def test_callable_advantage_rejects_a_non_tensor():
    function = CallableAdvantage(lambda *args, **kwargs: "nope")
    with pytest.raises(ContractError, match="expected AdvantageBatch or Tensor"):
        function(torch.zeros(2), torch.zeros(2))


def test_every_wrapper_satisfies_the_protocol():
    for function in (
        TDAdvantage(),
        GAEAdvantage(),
        ReferenceBaselineAdvantage(),
        VoidMaskedAdvantage(),
        NormalisedAdvantage(),
        ChainAdvantage([]),
        CallableAdvantage(lambda *args, **kwargs: torch.zeros(1)),
    ):
        assert isinstance(function, AdvantageFunction), type(function).__name__


def test_each_wrapper_builds_from_spec():
    specs = [
        {"type": "td", "gamma": 0.5},
        {"type": "gae", "gamma": 0.99, "lam": 0.9},
        {"type": "reference"},
        {"type": "void_masked", "floor": -3.0},
        {"type": "normalised", "eps": 1e-6},
        {"type": "chain", "steps": []},
    ]
    for spec in specs:
        built = build_function(spec, family="advantage")
        assert type(built).__name__ == ADVANTAGE_FUNCTION_TYPES[spec["type"]].__name__


def test_unknown_advantage_type_is_rejected():
    with pytest.raises(ValidationError, match="unknown advantage type"):
        build_function({"type": "nope"}, family="advantage")


def test_describe_reports_hyperparameters():
    payload = GAEAdvantage(gamma=0.5, lam=0.25).describe()
    assert payload["gamma"] == pytest.approx(0.5)
    assert payload["lam"] == pytest.approx(0.25)
    assert payload["type"] == "gae"


def test_chain_describe_lists_steps():
    payload = ChainAdvantage([ReferenceBaselineAdvantage(), VoidMaskedAdvantage()]).describe()
    assert [step["type"] for step in payload["steps"]] == ["reference", "void_masked"]
