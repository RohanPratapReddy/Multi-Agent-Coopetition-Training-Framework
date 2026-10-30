"""Criterion 20: scipy L-BFGS-B converges, and never with gradient accumulation."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("scipy")

from mactf.exceptions import ContractError, ValidationError
from mactf.training.optimizer import MACTFOptimizer, ScipyOptimizer, build_optimizer

from train_support import Quadratic, least_squares_problem, make_trainer, tuned  # noqa: F401


def _module():
    torch.manual_seed(3)
    module = Quadratic().double()
    return module


def test_lbfgsb_converges_on_a_convex_problem():
    features, targets, truth = least_squares_problem()
    module = _module()
    optimizer = MACTFOptimizer(ScipyOptimizer("L-BFGS-B", max_iter=200, tol=1e-12))
    optimizer.bind(list(module.parameters()))

    def closure():
        return torch.mean((module.linear(features) - targets) ** 2)

    before = float(closure())
    outcome = optimizer.step(closure)
    after = float(closure())
    assert outcome.stepped
    assert after < before * 1e-6
    assert torch.allclose(module.linear.weight.T, truth, atol=1e-4)
    assert module.linear.bias.item() == pytest.approx(0.25, abs=1e-4)
    result = optimizer.backend.last_result
    assert result["nit"] > 0 and result["nfev"] >= result["nit"]


def test_scipy_with_grad_accumulation_is_rejected():
    with pytest.raises(ValidationError, match="grad_accumulation"):
        MACTFOptimizer(ScipyOptimizer("L-BFGS-B"), grad_accumulation=4)
    with pytest.raises(ValidationError, match="grad_accumulation"):
        MACTFOptimizer.from_spec({"type": "scipy", "grad_accumulation": 2})


def test_a_closure_is_mandatory_with_no_silent_fallback():
    module = _module()
    optimizer = build_optimizer(module, {"type": "scipy", "method": "BFGS"})
    with pytest.raises(ContractError, match="closure"):
        optimizer.step()
    with pytest.raises(ContractError, match="closure"):
        optimizer.backend.step()


def test_unknown_and_hessian_methods_are_refused():
    with pytest.raises(ValidationError, match="unknown scipy method"):
        ScipyOptimizer("Nelder-Mead-Pro")
    from mactf.exceptions import ImproperlyConfigured

    module = _module()
    backend = ScipyOptimizer("Newton-CG").bind(list(module.parameters()))
    with pytest.raises(ImproperlyConfigured, match="Hessian"):
        backend.step(lambda: module.linear.weight.sum())


def test_parameters_keep_their_own_dtype():
    features, targets, _ = least_squares_problem()
    module = Quadratic().float()
    optimizer = MACTFOptimizer(ScipyOptimizer("L-BFGS-B", max_iter=50)).bind(
        list(module.parameters())
    )
    optimizer.step(lambda: torch.mean((module.linear(features.float()) - targets.float()) ** 2))
    assert module.linear.weight.dtype == torch.float32
    assert optimizer.backend.describe()["granularity"] == "epoch"
    assert optimizer.backend.describe()["full_batch"] is True


def test_state_round_trip_restores_the_vector():
    module = _module()
    backend = ScipyOptimizer("CG", max_iter=5).bind(list(module.parameters()))
    state = backend.state_dict()
    with torch.no_grad():
        module.linear.weight.add_(10.0)
    backend.load_state_dict(state)
    assert torch.allclose(backend._flatten(), state["vector"])


def test_the_trainer_runs_a_scipy_epoch(tuned, tmp_path):  # noqa: F811
    trainer = make_trainer(
        tuned,
        tmp_path,
        optimizer={"type": "scipy", "method": "L-BFGS-B", "max_iter": 3},
    )
    result = trainer.fit(epochs=1)
    assert result.status == "completed"
    assert all(isinstance(loss, float) for loss in trainer.losses)
    description = trainer.optimizers["peer-1"].describe()
    assert description["backend"] == "scipy"
