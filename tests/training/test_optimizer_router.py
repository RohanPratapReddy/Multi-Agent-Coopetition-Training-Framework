"""Criterion 19 (router half): every trainable parameter is claimed, or raise."""

from __future__ import annotations

import pytest
import torch

from mactf.exceptions import ValidationError
from mactf.training.optimizer import (
    MACTFOptimizer,
    ParamGroupRouter,
    Route,
    TorchOptimizer,
)

from train_support import Tiny, make_trainer, tuned  # noqa: F401


def _router():
    return ParamGroupRouter(
        [
            Route(r"policy\..*", TorchOptimizer(torch.optim.SGD, lr=0.5)),
            Route(r"value\..*", TorchOptimizer(torch.optim.SGD, lr=0.25), priority=1),
        ],
        default=TorchOptimizer(torch.optim.Adam, lr=0.001),
    )


def test_a_router_without_a_default_is_rejected():
    with pytest.raises(ValidationError, match="default"):
        ParamGroupRouter([Route(r".*", TorchOptimizer(torch.optim.SGD, lr=0.1))])


def test_routes_must_be_route_instances_and_valid_regexes():
    with pytest.raises(ValidationError, match="Route"):
        ParamGroupRouter(["policy"], default=TorchOptimizer(torch.optim.Adam, lr=0.1))
    with pytest.raises(Exception):
        ParamGroupRouter(
            [Route("(", TorchOptimizer(torch.optim.SGD, lr=0.1))],
            default=TorchOptimizer(torch.optim.Adam, lr=0.1),
        )


def test_every_parameter_is_assigned_by_first_match_then_default():
    module = Tiny()
    router = _router()
    optimizer = MACTFOptimizer(None, router=router)
    optimizer.bind(list(module.parameters()), named=list(module.named_parameters()))
    names = {name for name, _ in module.named_parameters()}
    assert set(router.assignment) == names
    assert router.assignment["policy.weight"] != router.assignment["body.0.weight"]
    claimed = sum(len(group["params"]) for group in router.param_groups)
    assert claimed == len(names)


def test_groups_step_with_their_own_learning_rates():
    module = Tiny()
    router = _router()
    optimizer = MACTFOptimizer(None, router=router)
    optimizer.bind(list(module.parameters()), named=list(module.named_parameters()))
    rates = sorted(group["lr"] for group in router.param_groups)
    assert rates == [0.001, 0.25, 0.5]
    before_policy = module.policy.weight.detach().clone()
    before_body = module.body[0].weight.detach().clone()
    optimizer.zero_grad()
    policy, value = module(torch.randn(5, 4))
    (policy.sum() + value.sum()).backward()
    optimizer.step()
    assert not torch.equal(module.policy.weight, before_policy)
    delta_policy = (module.policy.weight - before_policy).abs().max()
    delta_body = (module.body[0].weight - before_body).abs().max()
    assert delta_policy > delta_body  # lr 0.5 vs 0.001


def test_a_backend_and_a_router_together_are_rejected():
    with pytest.raises(ValidationError, match="either a backend or a router"):
        MACTFOptimizer(TorchOptimizer(torch.optim.Adam, lr=0.1), router=_router())


def test_router_state_round_trips():
    module = Tiny()
    router = _router()
    optimizer = MACTFOptimizer(None, router=router)
    optimizer.bind(list(module.parameters()), named=list(module.named_parameters()))
    optimizer.zero_grad()
    policy, value = module(torch.randn(5, 4))
    (policy.sum() + value.sum()).backward()
    optimizer.step()
    state = router.state_dict()
    assert set(state) == {"routes", "default"}
    router.load_state_dict(state)
    description = router.describe()
    assert description["assigned"] == len(list(module.named_parameters()))
    assert len(description["routes"]) == 2


def test_frozen_parameters_are_not_claimed_or_required():
    module = Tiny()
    for parameter in module.body.parameters():
        parameter.requires_grad_(False)
    router = _router()
    MACTFOptimizer(None, router=router).bind(
        list(module.parameters()), named=list(module.named_parameters())
    )
    assert not any(name.startswith("body.") for name in router.assignment)


def test_the_trainer_accepts_a_router(tuned, tmp_path):  # noqa: F811
    trainer = make_trainer(tuned, tmp_path, optimizer=_router_factory())
    result = trainer.fit(epochs=1)
    assert result.status == "completed"


def _router_factory():
    return MACTFOptimizer(None, router=_router())
