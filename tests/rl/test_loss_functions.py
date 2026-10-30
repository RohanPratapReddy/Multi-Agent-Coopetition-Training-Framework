"""Loss-function wrappers (document 07 section 8.3).

The gate: ``CompositeLoss`` must equal ``compute_loss`` to ``1e-12`` on the total
and on every breakdown field, including the ``advantage_mean`` extra.
"""

from __future__ import annotations

import pytest
import torch

from mactf.exceptions import ContractError, ValidationError
from mactf.rl.functions.loss import (
    LOSS_FUNCTION_TYPES,
    CallableLoss,
    CausalLMLoss,
    ChainedLoss,
    CompositeLoss,
    ComplianceFloorLoss,
    EntropyBonus,
    KLLoss,
    PolicyGradientLoss,
    ValueRegressionLoss,
    VoidMaskLoss,
)
from mactf.rl.functions.registry import build_function
from mactf.rl.loss import LossConfig, compute_loss, policy_loss
from mactf.rl.policy import make_policy, reverse_kl


def _batch(seed: int = 0, **overrides):
    generator = torch.Generator().manual_seed(seed)
    policy = make_policy(torch.randn(6, 4, generator=generator))
    reference = make_policy(torch.randn(6, 4, generator=generator))
    batch = {
        "policy": policy,
        "reference": reference,
        "actions": torch.randint(0, 4, (6,), generator=generator),
        "advantages": torch.randn(6, generator=generator),
        "values": torch.randn(6, 1, generator=generator),
        "value_targets": torch.randn(6, 1, generator=generator),
    }
    batch.update(overrides)
    return batch


def _oracle(batch, config):
    return compute_loss(
        batch["policy"],
        batch["reference"],
        batch["actions"],
        batch["advantages"],
        batch["values"],
        batch["value_targets"],
        config=config,
        compliance_floor=batch.get("compliance_floor"),
        void_mask=batch.get("void_mask"),
    )


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_composite_loss_equals_compute_loss_at_1e_12(seed):
    batch = _batch(seed)
    config = LossConfig()
    expected_total, expected = _oracle(batch, config)
    actual_total, actual = CompositeLoss(config=config).compute(batch)

    assert actual_total.item() == pytest.approx(expected_total.item(), abs=1e-12)
    for field in ("policy", "value", "kl", "entropy", "compliance", "total"):
        assert getattr(actual, field) == pytest.approx(
            getattr(expected, field), abs=1e-12
        ), field
    assert actual.extras["advantage_mean"] == pytest.approx(
        expected.extras["advantage_mean"], abs=1e-12
    )


def test_composite_loss_equals_compute_loss_with_every_coefficient():
    batch = _batch(3)
    config = LossConfig(
        value_coef=0.7, kl_coef=0.05, entropy_coef=0.02, clip_epsilon=0.1, compliance_coef=2.0
    )
    expected_total, expected = _oracle(batch, config)
    actual_total, actual = CompositeLoss(config=config).compute(batch)
    assert actual_total.item() == pytest.approx(expected_total.item(), abs=1e-12)
    assert actual.kl == pytest.approx(expected.kl, abs=1e-12)
    assert actual.entropy == pytest.approx(expected.entropy, abs=1e-12)


def test_composite_loss_equals_compute_loss_with_a_floor_and_void_mask():
    batch = _batch(5, compliance_floor=torch.tensor(-1000.0), void_mask=torch.tensor([True, False, False, True, False, False]))
    config = LossConfig()
    expected_total, expected = _oracle(batch, config)
    actual_total, actual = CompositeLoss(config=config).compute(batch)
    assert actual_total.item() == pytest.approx(expected_total.item(), abs=1e-12)
    assert actual.compliance == pytest.approx(expected.compliance, abs=1e-12)
    assert actual.compliance == pytest.approx(1000.0)


def test_policy_gradient_loss_delegates_to_policy_loss():
    batch = _batch(2)
    expected = policy_loss(
        batch["policy"], batch["reference"], batch["actions"], batch["advantages"]
    )
    loss, breakdown = PolicyGradientLoss().compute(batch)
    assert loss.item() == pytest.approx(expected.item(), abs=1e-12)
    assert breakdown.policy == pytest.approx(expected.item(), abs=1e-12)


def test_policy_gradient_loss_scales_by_its_coefficient():
    batch = _batch(2)
    base, _ = PolicyGradientLoss().compute(batch)
    scaled, _ = PolicyGradientLoss(coefficient=3.0).compute(batch)
    # Compare float32-to-float32: scaling a float32 tensor by 3.0 rounds once,
    # so a double-precision product would be off by ~1 ulp.
    assert scaled.item() == pytest.approx((base * 3.0).item(), abs=0.0)


def test_policy_gradient_loss_declares_its_fields():
    assert PolicyGradientLoss.required_fields == {
        "policy",
        "reference",
        "actions",
        "advantages",
    }


def test_value_regression_matches_value_loss():
    from mactf.rl.value import value_loss

    batch = _batch(4)
    expected = value_loss(batch["values"], batch["value_targets"], coefficient=0.5)
    loss, breakdown = ValueRegressionLoss(coefficient=0.5).compute(batch)
    assert loss.item() == pytest.approx(expected.item(), abs=1e-12)
    # The breakdown field is the unscaled MSE, matching ``compute_loss``.
    assert breakdown.value == pytest.approx(2.0 * expected.item(), abs=1e-12)


def test_value_regression_reports_a_length_mismatch():
    batch = _batch(4, value_targets=torch.zeros(2, 1))
    with pytest.raises(ContractError, match="value_targets"):
        ValueRegressionLoss().compute(batch)


def test_kl_loss_matches_reverse_kl():
    batch = _batch(6)
    expected = reverse_kl(batch["policy"], batch["reference"]).mean()
    loss, breakdown = KLLoss(coefficient=0.01).compute(batch)
    assert loss.item() == pytest.approx((0.01 * expected).item(), abs=0.0)
    assert breakdown.kl == pytest.approx(expected.item(), abs=1e-12)


def test_entropy_bonus_has_the_negated_sign():
    batch = _batch(8)
    mean_entropy = batch["policy"].entropy.mean()
    loss, breakdown = EntropyBonus(coefficient=0.5).compute(batch)
    assert loss.item() == pytest.approx(-0.5 * mean_entropy.item(), abs=1e-12)
    assert breakdown.entropy == pytest.approx(mean_entropy.item(), abs=1e-12)


def test_entropy_bonus_with_zero_coefficient_is_zero():
    loss, _ = EntropyBonus().compute(_batch(8))
    assert float(loss) == 0.0


def test_compliance_floor_sign_and_clamp():
    loss, breakdown = ComplianceFloorLoss(coefficient=2.0).compute(
        {"compliance_floor": torch.tensor(-10.0)}
    )
    assert loss.item() == pytest.approx(20.0)
    assert breakdown.compliance == pytest.approx(20.0)


def test_compliance_floor_is_zero_for_a_non_negative_floor():
    loss, _ = ComplianceFloorLoss().compute({"compliance_floor": torch.tensor(5.0)})
    assert float(loss) == 0.0


def test_void_mask_loss_contributes_nothing_but_reports_the_count():
    loss, breakdown = VoidMaskLoss().compute(
        {"void_mask": torch.tensor([True, False, True])}
    )
    assert float(loss) == 0.0
    assert breakdown.extras["void_masked"] == pytest.approx(2.0)


def test_missing_fields_are_named_before_the_forward_pass():
    with pytest.raises(ContractError, match="missing required field"):
        PolicyGradientLoss().compute({"policy": make_policy(torch.zeros(1, 2))})


def test_missing_field_message_names_the_class_and_every_key():
    try:
        KLLoss().compute({})
    except ContractError as error:
        message = str(error)
        assert "KLLoss" in message
        assert "policy" in message and "reference" in message
    else:  # pragma: no cover
        pytest.fail("expected ContractError")


def test_causal_lm_uses_the_native_model_loss():
    class Model(torch.nn.Module):
        def forward(self, input_ids=None):
            class Output:
                loss = torch.tensor(1.25)

            return Output()

    loss, breakdown = CausalLMLoss(model=Model()).compute(
        {"input_ids": torch.zeros(2, 3, dtype=torch.long), "labels": torch.zeros(2, 3, dtype=torch.long)}
    )
    assert loss.item() == pytest.approx(1.25)
    assert breakdown.total == pytest.approx(1.25)


def test_causal_lm_falls_back_to_cross_entropy():
    oracle = torch.nn.functional.cross_entropy
    logits = torch.randn(2, 3, 5)
    labels = torch.tensor([[0, 1, 2], [3, 4, -100]])
    loss, _ = CausalLMLoss().compute({"logits": logits, "labels": labels})
    expected = oracle(logits.reshape(-1, 5), labels.reshape(-1), ignore_index=-100)
    assert loss.item() == pytest.approx(expected.item(), abs=1e-12)


def test_causal_lm_requires_logits_or_a_model():
    with pytest.raises(ContractError, match="either 'model' or 'logits'"):
        CausalLMLoss().compute(
            {"input_ids": torch.zeros(1, dtype=torch.long), "labels": torch.zeros(1, dtype=torch.long)}
        )


def test_callable_loss_adapts_a_tensor():
    loss, breakdown = CallableLoss(lambda batch: batch["x"].sum(), batch_keys=["x"]).compute(
        {"x": torch.tensor([1.0, 2.0])}
    )
    assert loss.item() == pytest.approx(3.0)
    assert breakdown.total == pytest.approx(3.0)


def test_callable_loss_adapts_a_tensor_and_extras():
    loss, breakdown = CallableLoss(
        lambda batch: (batch["x"].sum(), {"custom": 2.5}), batch_keys=["x"]
    ).compute({"x": torch.tensor([1.0])})
    assert breakdown.extras["custom"] == pytest.approx(2.5)


def test_callable_loss_declares_its_batch_keys():
    assert CallableLoss(lambda batch: torch.zeros(()), batch_keys=["a", "b"]).required_fields == {
        "a",
        "b",
    }


def test_callable_loss_requires_a_callable():
    with pytest.raises(ValidationError, match="requires an 'fn'"):
        CallableLoss()


def test_callable_loss_rejects_a_bad_return_type():
    with pytest.raises(ContractError, match="expected a tensor"):
        CallableLoss(lambda batch: "nope").compute({})


def test_chained_loss_sums_its_terms():
    batch = _batch(9)
    chained = ChainedLoss([PolicyGradientLoss(), ValueRegressionLoss()])
    composite_policy = PolicyGradientLoss().compute(batch)[0]
    composite_value = ValueRegressionLoss().compute(batch)[0]
    total, breakdown = chained.compute(batch)
    assert total.item() == pytest.approx(
        (composite_policy + composite_value).item(), abs=1e-12
    )
    assert breakdown.policy == pytest.approx(composite_policy.item(), abs=1e-12)
    # ``value`` is the unscaled MSE, as in ``compute_loss``.
    assert breakdown.value == pytest.approx(2.0 * composite_value.item(), abs=1e-12)


def test_chained_loss_requires_fields_from_every_term():
    chained = ChainedLoss([PolicyGradientLoss(), ValueRegressionLoss()])
    assert {"policy", "values"} <= chained.required_fields


def test_chained_loss_rejects_a_bad_term():
    with pytest.raises(ValidationError, match="no compute"):
        ChainedLoss([object()])


def test_chained_loss_from_spec_builds_nested_specs():
    chained = ChainedLoss.from_spec(
        {"type": "chained", "terms": [{"type": "kl", "coefficient": 0.1}]}
    )
    assert len(chained.terms) == 1
    assert chained.terms[0].coefficient == pytest.approx(0.1)


def test_chained_loss_from_spec_requires_terms():
    with pytest.raises(ValidationError, match="terms"):
        ChainedLoss.from_spec({"type": "chained"})


def test_composite_loss_describes_its_terms():
    payload = CompositeLoss(config=LossConfig(value_coef=0.3)).describe()
    assert payload["config"]["value_coef"] == pytest.approx(0.3)
    assert [term["type"] for term in payload["terms"]] == [
        "policy_gradient",
        "value_regression",
        "kl",
        "entropy",
    ]


def test_each_loss_builds_from_spec():
    specs = [
        {"type": "policy_gradient"},
        {"type": "value_regression"},
        {"type": "kl"},
        {"type": "entropy"},
        {"type": "compliance_floor"},
        {"type": "void_mask"},
        {"type": "composite"},
        {"type": "causal_lm"},
        {"type": "callable", "batch_keys": []},
    ]
    for spec in specs:
        payload = dict(spec)
        if payload["type"] == "callable":
            payload["fn"] = lambda batch: torch.zeros(())
        built = build_function(payload, family="loss")
        assert type(built).__name__ == LOSS_FUNCTION_TYPES[spec["type"]].__name__


def test_unknown_loss_type_is_rejected():
    with pytest.raises(ValidationError, match="unknown loss type"):
        build_function({"type": "policy"}, family="loss")


def test_policy_gradient_rejects_a_bad_clip_epsilon():
    with pytest.raises(ValidationError, match="clip_epsilon"):
        PolicyGradientLoss(clip_epsilon=1.5)
