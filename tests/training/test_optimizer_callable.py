"""CallableOptimizer: imperative, functional and stateless."""

from __future__ import annotations

import pytest
import torch

from mactf.exceptions import ContractError
from mactf.training.optimizer import CallableOptimizer, MACTFOptimizer, build_optimizer

from train_support import Quadratic


def test_imperative_callable_receives_the_parameter_list():
    seen = {}

    def update(parameters):
        seen["count"] = len(parameters)
        with torch.no_grad():
            for parameter in parameters:
                parameter.add_(1.0)

    module = Quadratic()
    before = module.linear.weight.detach().clone()
    optimizer = MACTFOptimizer(CallableOptimizer(update)).bind(list(module.parameters()))
    outcome = optimizer.step()
    assert outcome.stepped and seen["count"] == 2
    assert torch.allclose(module.linear.weight, before + 1.0)
    assert optimizer.backend.calls == 1


def test_functional_callable_returns_new_values():
    module = Quadratic()
    optimizer = MACTFOptimizer(
        CallableOptimizer(lambda parameter: parameter * 0.0, functional=True)
    ).bind(list(module.parameters()))
    optimizer.step()
    assert float(module.linear.weight.abs().sum()) == 0.0


def test_functional_shape_and_type_are_checked():
    module = Quadratic()
    wrong_shape = CallableOptimizer(lambda p: torch.zeros(1), functional=True)
    wrong_shape.bind(list(module.parameters()))
    with pytest.raises(ContractError, match="shape"):
        wrong_shape.step()
    wrong_type = CallableOptimizer(lambda p: 3.0, functional=True)
    wrong_type.bind(list(module.parameters()))
    with pytest.raises(ContractError, match="tensor"):
        wrong_type.step()


def test_stateless_is_stated_and_enforced():
    backend = CallableOptimizer(lambda parameters: None)
    assert backend.state_dict() == {}
    assert backend.describe()["stateful"] is False
    with pytest.raises(ContractError, match="stateless"):
        backend.load_state_dict({"anything": 1})


def test_non_callable_is_rejected_and_spec_requires_fn():
    with pytest.raises(ContractError, match="callable"):
        CallableOptimizer(3)
    from mactf.exceptions import ValidationError

    with pytest.raises(ValidationError, match="fn"):
        MACTFOptimizer.from_spec({"type": "callable"})
    module = Quadratic()
    built = build_optimizer(module, {"type": "callable", "fn": lambda p: None})
    assert built.describe()["backend"] == "callable"
