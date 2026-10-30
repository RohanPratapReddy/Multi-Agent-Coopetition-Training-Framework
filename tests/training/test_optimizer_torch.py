"""The torch backend, the facade, accumulation, clipping and state."""

from __future__ import annotations

import functools

import pytest
import torch

from mactf.exceptions import ContractError, ValidationError
from mactf.training.optimizer import MACTFOptimizer, TorchOptimizer, build_optimizer

from train_support import Quadratic, least_squares_problem


def _loss(module, features, targets):
    return torch.mean((module.linear(features) - targets) ** 2)


def test_three_accepted_forms_all_train():
    features, targets, _ = least_squares_problem()
    forms = [
        TorchOptimizer(torch.optim.Adam, lr=0.05),
        TorchOptimizer(functools.partial(torch.optim.SGD, momentum=0.9), lr=0.05),
    ]
    for backend in forms:
        module = Quadratic().double()
        optimizer = MACTFOptimizer(backend).bind(list(module.parameters()))
        start = _loss(module, features, targets).item()
        for _ in range(60):
            optimizer.zero_grad()
            _loss(module, features, targets).backward()
            optimizer.step()
        assert _loss(module, features, targets).item() < start * 0.5

    module = Quadratic().double()
    built = torch.optim.AdamW(module.parameters(), lr=0.01)
    wrapped = MACTFOptimizer(TorchOptimizer(built))
    assert wrapped.bound
    assert wrapped.backend.class_name == "AdamW"


def test_a_factory_needs_a_positive_lr_and_binding():
    with pytest.raises(ValidationError, match="lr > 0"):
        TorchOptimizer(torch.optim.Adam, lr=0.0)
    backend = TorchOptimizer(torch.optim.Adam, lr=0.1)
    with pytest.raises(ContractError, match="not bound"):
        backend.optimizer
    with pytest.raises(ContractError):
        TorchOptimizer(42)


def test_gradient_accumulation_steps_every_n_calls():
    features, targets, _ = least_squares_problem()
    module = Quadratic().double()
    optimizer = MACTFOptimizer(
        TorchOptimizer(torch.optim.SGD, lr=0.01), grad_accumulation=3
    ).bind(list(module.parameters()))
    outcomes = []
    for _ in range(6):
        (_loss(module, features, targets) / 3).backward()
        outcomes.append(optimizer.step())
    assert [o.stepped for o in outcomes] == [False, False, True, False, False, True]
    assert outcomes[0].skipped_reason == "gradient accumulation"


def test_clipping_reports_the_pre_clip_norm_and_bounds_the_update():
    features, targets, _ = least_squares_problem()
    module = Quadratic().double()
    optimizer = MACTFOptimizer(
        TorchOptimizer(torch.optim.SGD, lr=1.0), max_grad_norm=0.1
    ).bind(list(module.parameters()))
    _loss(module, features, targets).backward()
    norm = torch.sqrt(sum((p.grad**2).sum() for p in module.parameters())).item()
    outcome = optimizer.step()
    assert outcome.grad_norm == pytest.approx(norm, rel=1e-6)
    clipped = torch.sqrt(sum((p.grad**2).sum() for p in module.parameters())).item()
    assert clipped == pytest.approx(0.1, rel=1e-6)


def test_set_lr_and_validation():
    module = Quadratic()
    optimizer = build_optimizer(module, {"type": "torch", "class": "SGD", "lr": 0.1})
    optimizer.set_lr(0.02)
    assert optimizer.learning_rate == pytest.approx(0.02)
    assert all(group["lr"] == pytest.approx(0.02) for group in optimizer.backend.param_groups)
    with pytest.raises(ValidationError):
        optimizer.set_lr(0.0)
    with pytest.raises(ValidationError):
        MACTFOptimizer(None, grad_accumulation=0)
    with pytest.raises(ValidationError):
        MACTFOptimizer(None, max_grad_norm=-1.0)
    with pytest.raises(ValidationError, match="unknown torch optimizer"):
        MACTFOptimizer.from_spec({"type": "torch", "class": "NoSuchOptimizer"})
    with pytest.raises(ValidationError, match="unknown optimizer backend"):
        MACTFOptimizer.from_spec({"type": "magic"})


def test_state_round_trip_preserves_moments():
    features, targets, _ = least_squares_problem()
    module = Quadratic().double()
    first = build_optimizer(module, {"type": "torch", "class": "Adam", "lr": 0.05})
    for _ in range(5):
        first.zero_grad()
        _loss(module, features, targets).backward()
        first.step()
    state = first.state_dict()

    twin = Quadratic().double()
    twin.load_state_dict(module.state_dict())
    second = build_optimizer(twin, {"type": "torch", "class": "Adam", "lr": 0.05})
    second.load_state_dict(state)
    for optimizer, subject in ((first, module), (second, twin)):
        optimizer.zero_grad()
        _loss(subject, features, targets).backward()
        optimizer.step()
    assert torch.allclose(module.linear.weight, twin.linear.weight)
    assert state["backend"] == "torch" and state["class"] == "Adam"


def test_load_state_requires_binding():
    optimizer = MACTFOptimizer(TorchOptimizer(torch.optim.Adam, lr=0.1))
    with pytest.raises(ContractError, match="bound"):
        optimizer.load_state_dict({"state": {"x": 1}})


def test_empty_binding_is_a_contract_error():
    with pytest.raises(ContractError, match="no trainable"):
        MACTFOptimizer(TorchOptimizer(torch.optim.Adam, lr=0.1)).bind([])
