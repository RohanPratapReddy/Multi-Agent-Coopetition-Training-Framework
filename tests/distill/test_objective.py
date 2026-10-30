"""Compound objective: cohort signal, L_coop, ramp, assembly, gradient claims."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from mactf.distill.objective import (
    ObjectiveConfig,
    cohort_signal,
    compound_loss,
    cooperative_reward_term,
    cooperative_term,
    individual_term,
    ramp_factor,
    regularization_term,
)
from mactf.exceptions import ComplianceError, ContractError, ValidationError


def _scalar(value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32)


def test_cohort_signal_is_zero_sum():
    out = cohort_signal({"a": 3.0, "b": 1.0, "c": 5.0})
    assert math.fsum(out.values()) == pytest.approx(0.0, abs=1e-12)
    assert out["c"] > 0 > out["b"]


def test_cohort_signal_pins_void_peers_to_zero():
    out = cohort_signal({"a": 3.0, "b": 1.0, "c": 100.0}, eligible={"a": True, "b": True, "c": False})
    assert out["c"] == 0.0
    assert out["a"] == pytest.approx(1.0)
    assert out["b"] == pytest.approx(-1.0)
    assert math.fsum(out.values()) == pytest.approx(0.0, abs=1e-12)


def test_cohort_signal_edge_cases():
    assert cohort_signal({}) == {}
    assert cohort_signal({"a": 1.0}, eligible={"a": False}) == {"a": 0.0}


def test_ramp_factor_phases():
    kwargs = {"warmup_steps": 4, "ramp_steps": 4}
    assert [ramp_factor(s, **kwargs) for s in (0, 3)] == [0.0, 0.0]
    assert ramp_factor(4, **kwargs) == 0.0
    assert ramp_factor(6, **kwargs) == pytest.approx(0.5)
    assert ramp_factor(8, **kwargs) == 1.0
    assert ramp_factor(100, **kwargs) == 1.0
    assert ramp_factor(0, warmup_steps=0, ramp_steps=0) == 1.0
    with pytest.raises(ValidationError):
        ramp_factor(-1, warmup_steps=0, ramp_steps=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lambda_cd": -1.0},
        {"lambda_ind": float("nan")},
        {"lambda_ind": 0.0, "lambda_cd": 0.0, "lambda_coop": 0.0},
        {"peer_margin": 1.5},
        {"tau": 0.0},
        {"tau_min": 0.5, "tau_max": 0.1},
        {"beta": -1.0},
        {"warmup_steps": -1},
    ],
)
def test_objective_config_validation(kwargs):
    with pytest.raises(ValidationError):
        ObjectiveConfig(**kwargs)


def _assemble(config: ObjectiveConfig, *, beta: float = 1.0, sigma=None, contrastive=None):
    contrastive = contrastive if contrastive is not None else {"a": _scalar(2.0), "b": _scalar(4.0)}
    sigma = sigma if sigma is not None else {"a": 0.25, "b": 0.75}
    return compound_loss(
        peer="p",
        config=config,
        individual=_scalar(1.5),
        cooperative=_scalar(0.5),
        regularization=_scalar(0.2),
        contrastive=contrastive,
        sigma=sigma,
        margin=_scalar(0.1),
        beta=beta,
        step=0,
    )


def test_compound_loss_matches_the_closed_form():
    config = ObjectiveConfig(lambda_coop=2.0, lambda_ind=1.0, lambda_reg=3.0, lambda_cd=0.5, peer_margin=0.4)
    result = _assemble(config, beta=0.5)
    weighted = 0.25 * 2.0 + 0.75 * 4.0
    expected = 2.0 * 0.5 + 1.0 * 1.5 + 3.0 * 0.2 + 0.5 * 0.5 * weighted + 0.4 * 0.1
    assert float(result.total) == pytest.approx(expected, rel=1e-6)
    assert result.lambda_cd_effective == pytest.approx(0.25)
    assert result.breakdown.extras["cd.a"] == pytest.approx(2.0)
    assert result.breakdown.extras["sigma.b"] == pytest.approx(0.75)
    assert result.as_dict()["peer"] == "p"


def test_ramp_scales_only_the_distillation_term():
    config = ObjectiveConfig(lambda_cd=1.0, peer_margin=0.0)
    full = _assemble(config, beta=1.0)
    zero = _assemble(config, beta=0.0)
    assert float(full.total) - float(zero.total) == pytest.approx(float(full.cd), rel=1e-6)
    assert float(zero.total) == pytest.approx(1.5)


def test_sigma_must_match_the_contrastive_teachers_and_sum_to_one():
    config = ObjectiveConfig()
    with pytest.raises(ContractError):
        _assemble(config, sigma={"a": 1.0})
    with pytest.raises(ContractError):
        _assemble(config, sigma={"a": 0.2, "b": 0.2})


def test_no_teachers_reduces_to_the_individual_objective():
    config = ObjectiveConfig(lambda_coop=0.0, peer_margin=0.0)
    result = _assemble(config, sigma={}, contrastive={})
    assert float(result.total) == pytest.approx(1.5)


def test_non_finite_total_raises():
    config = ObjectiveConfig()
    with pytest.raises(ComplianceError):
        compound_loss(
            peer="p",
            config=config,
            individual=_scalar(float("nan")),
            cooperative=_scalar(0.0),
            regularization=_scalar(0.0),
            contrastive={},
            sigma={},
            margin=_scalar(0.0),
            beta=1.0,
            step=3,
        )


def _policy(batch: int = 4, actions: int = 3):
    logits = torch.randn(batch, actions, dtype=torch.float64, requires_grad=True)
    log_probs = torch.log_softmax(logits, dim=-1)
    policy = SimpleNamespace(
        log_probs=log_probs,
        gather=lambda a: log_probs.gather(-1, a.reshape(-1, 1)).squeeze(-1),
    )
    return logits, policy


def test_cooperative_term_is_exactly_zero_with_zero_gradient_when_disabled():
    torch.manual_seed(0)
    logits, policy = _policy()
    advantage = torch.randn(4, dtype=torch.float64)
    for coefficient, centring in ((0.0, 1.0), (1.0, 0.0)):
        term = cooperative_term(policy, advantage=advantage, centring=centring, coefficient=coefficient)
        assert float(term) == 0.0
        term.backward(retain_graph=True)
        assert float(logits.grad.abs().sum()) == 0.0
        logits.grad = None


def test_cooperative_term_value_and_detached_advantage():
    torch.manual_seed(1)
    logits, policy = _policy()
    advantage = torch.randn(4, dtype=torch.float64, requires_grad=True)
    actions = torch.tensor([0, 1, 2, 1])
    term = cooperative_term(policy, advantage=advantage, centring=0.5, coefficient=2.0, actions=actions)
    chosen = policy.log_probs.gather(-1, actions.reshape(-1, 1)).squeeze(-1)
    expected = -2.0 * 0.5 * float((chosen * advantage.detach()).mean())
    assert float(term) == pytest.approx(expected, abs=1e-12)
    term.backward()
    assert advantage.grad is None
    assert logits.grad is not None


def test_cooperative_term_row_mismatch_raises():
    _, policy = _policy()
    with pytest.raises(ContractError):
        cooperative_term(policy, advantage=torch.zeros(3), centring=1.0, coefficient=1.0)


def test_cooperative_reward_term_is_a_constant():
    term = cooperative_reward_term(awarded=2.0, coefficient=0.5)
    assert float(term) == pytest.approx(-1.0)
    assert not term.requires_grad


def test_individual_term_reports_missing_fields_and_non_tensor_results():
    class _Loss:
        required_fields = ("policy", "advantages")

        def compute(self, batch):
            return 1.0, None

    with pytest.raises(ContractError, match="advantages"):
        individual_term(_Loss(), {"policy": 1})
    with pytest.raises(ContractError, match="expected a tensor"):
        individual_term(_Loss(), {"policy": 1, "advantages": 2})


def test_regularization_term_sums_scalar_regularizers():
    module = torch.nn.Linear(2, 2)

    class _L2:
        def __call__(self, m):
            return sum((p**2).sum() for p in m.parameters())

    total, values = regularization_term([_L2(), _L2()], module)
    assert float(total) == pytest.approx(2.0 * float(_L2()(module)), rel=1e-6)
    assert len(values) == 2
    empty, none = regularization_term([], module)
    assert float(empty) == 0.0 and none == {}

    class _Bad:
        def __call__(self, m):
            return torch.ones(3)

    with pytest.raises(ContractError):
        regularization_term([_Bad()], module)
