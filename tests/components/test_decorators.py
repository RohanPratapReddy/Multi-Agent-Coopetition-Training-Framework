"""Decorators and the generic API (document 17 section 3.8)."""

from __future__ import annotations

import pytest

from component_helpers import OtherPolicy, Policy, Term, make_scheduler
from mactf import components as mc
from mactf.components import (
    HUB,
    Component,
    ComponentKind,
    Mutability,
    Origin,
    ParamSpec,
    get_component,
    register,
    unregister,
)
from mactf.exceptions import ComponentCollision, ComponentContractError, ComponentLocked, ValidationError


@pytest.fixture(autouse=True)
def _scope(component_overlay):
    yield component_overlay


def test_register_with_an_object_returns_the_component():
    component = register(ComponentKind.TRAINER_POLICY, "d_p", Policy, scope="run")
    assert isinstance(component, Component) and component.factory is Policy
    assert get_component(ComponentKind.TRAINER_POLICY, "d_p") is component


def test_register_as_a_decorator_returns_the_class_unchanged():
    @register(ComponentKind.TRAINER_POLICY, "d_deco", scope="run")
    class Mine(Policy):
        pass

    assert Mine.__name__ == "Mine"
    assert get_component("trainer_policy", "d_deco").factory is Mine


@pytest.mark.parametrize(
    "decorator, kind_id, target",
    [
        (mc.register_reward_term, "reward_term", Term),
        (mc.register_scheduler, "scheduler", make_scheduler),
        (mc.register_policy, "trainer_policy", Policy),
        (mc.register_peer_factory, "peer_factory", lambda spec: None),
    ],
)
def test_each_kind_decorator_targets_its_registry(decorator, kind_id, target):
    returned = decorator("d_kind", scope="run")(target)
    assert returned is target
    assert HUB.registry(kind_id).get("d_kind").factory is target
    assert decorator.__name__.startswith("register_")


def test_the_heavy_kinds_have_decorators_too():
    from mactf.coopetition.aggregation import Aggregator
    from mactf.coopetition.strategy import IndependentStrategy
    from mactf.coopetition.weights import UniformAverage
    import torch

    @mc.register_peer_strategy("d_strategy", scope="run")
    class S(IndependentStrategy):
        name = "d_strategy"

    @mc.register_weight_update("d_weights", scope="run")
    class W(UniformAverage):
        name = "d_weights"

    @mc.register_aggregator("d_agg", scope="run")
    class A(Aggregator):
        pass

    @mc.register_optimizer("d_opt", scope="run")
    class O(torch.optim.SGD):
        pass

    assert HUB.registry("peer_strategy").get("d_strategy").factory is S
    assert HUB.registry("weight_update").get("d_weights").factory is W
    assert HUB.registry("aggregator").get("d_agg").factory is A
    assert HUB.registry("optimizer").get("d_opt").factory is O


def test_every_option_is_passed_through():
    component = register(
        ComponentKind.TRAINER_POLICY, "d_opts", Policy, scope="run", version="3",
        description="desc", params_schema={"rate": ParamSpec(type=float)},
        capabilities={"x": 1}, requires=("scorer:penalty",), origin=Origin.PLUGIN,
        mutability=Mutability.STATIC,
    )
    assert (component.version, component.description, component.origin) == ("3", "desc", Origin.PLUGIN)
    assert component.params_schema == {"rate": ParamSpec(type=float)}
    assert component.capabilities == {"x": 1} and component.requires == {"scorer:penalty"}


def test_kind_may_be_a_name_a_kind_id_or_an_enum():
    register("trainer_policy", "d_a", Policy, scope="run")
    register(ComponentKind.RL_FUNCTION, "d_b", _Loss, family="loss", scope="run")
    register("rl_function.loss", "d_c", _Loss, scope="run")
    register("rl_function", "d_d", _Loss, family="loss", scope="run")
    names = HUB.registry("rl_function.loss").names()
    assert {"d_b", "d_c", "d_d"} <= set(names)


class _Loss:
    required_fields = frozenset({"policy"})

    def compute(self, batch):  # pragma: no cover
        return 0


@pytest.mark.parametrize(
    "call",
    [
        lambda: register("nonsense", "x", Policy),
        lambda: register(ComponentKind.RL_FUNCTION, "x", _Loss),  # no family
        lambda: register(ComponentKind.RL_FUNCTION, "x", _Loss, family="nonsense"),
        lambda: register(ComponentKind.SCORER, "x", Policy, family="loss"),
        lambda: get_component("rl_function.nonsense", "x"),
    ],
)
def test_bad_kinds_raise_validation_errors(call):
    with pytest.raises(ValidationError):
        call()


def test_a_contract_failure_through_the_decorator_leaves_nothing_registered():
    with pytest.raises(ComponentContractError):
        register(ComponentKind.TRAINER_POLICY, "d_bad", object, scope="run")
    assert not HUB.registry("trainer_policy").has("d_bad")


def test_unregister_and_scope_run():
    register(ComponentKind.TRAINER_POLICY, "d_gone", Policy, scope="run")
    removed = unregister(ComponentKind.TRAINER_POLICY, "d_gone", scope="run")
    assert removed.name == "d_gone" and not HUB.registry("trainer_policy").has("d_gone")


def test_collision_and_replace_through_the_api():
    register(ComponentKind.TRAINER_POLICY, "d_r", Policy, scope="run")
    with pytest.raises(ComponentCollision):
        register(ComponentKind.TRAINER_POLICY, "d_r", OtherPolicy, scope="run")
    register(ComponentKind.TRAINER_POLICY, "d_r", OtherPolicy, scope="run", replace=True)
    assert get_component("trainer_policy", "d_r").factory is OtherPolicy


def test_scope_run_with_no_overlay_is_refused():
    # the autouse fixture opened one; leave it for a moment by using a fresh context
    import contextvars

    def attempt():
        with pytest.raises(ComponentLocked, match="no run scope is active"):
            register(ComponentKind.TRAINER_POLICY, "d_x", Policy, scope="run")

    contextvars.Context().run(attempt)


# -- the two legacy forms ------------------------------------------------------------
def test_register_scorer_has_both_forms():
    from mactf.scoring.reward_scorer import RewardScorer

    class Positional(RewardScorer):
        pass

    class Decorated(RewardScorer):
        pass

    assert mc.register_scorer("d_pos", Positional, scope="run") is None  # legacy: returns None
    assert mc.register_scorer("d_deco", scope="run")(Decorated) is Decorated
    assert HUB.registry("scorer").get("d_pos").factory is Positional
    assert HUB.registry("scorer").get("d_deco").factory is Decorated


def test_register_scorer_legacy_form_keeps_the_old_refusals():
    from mactf.scoring.reward_scorer import RewardScorer

    class Custom(RewardScorer):
        pass

    with pytest.raises(ValidationError, match="already registered"):
        mc.register_scorer("reward", Custom, scope="run")
    with pytest.raises(ValidationError, match="not a Scorer subclass"):
        mc.register_scorer("bogus", object, scope="run")
    with pytest.raises(ValidationError, match="invalid scorer type name"):
        mc.register_scorer("not a name", Custom, scope="run")


def test_register_function_has_both_forms_and_the_old_refusals():
    assert mc.register_function("loss", "d_fn", _Loss, scope="run") is None
    assert mc.register_function("loss", "d_fn2", scope="run")(_Loss) is _Loss
    with pytest.raises(ValidationError, match="already registered"):
        mc.register_function("loss", "kl", _Loss, scope="run")
    with pytest.raises(ValidationError, match="unknown function family"):
        mc.register_function("nonsense", "x", _Loss, scope="run")
    with pytest.raises(ValidationError, match="invalid function type name"):
        mc.register_function("loss", "not a name", _Loss, scope="run")
    with pytest.raises(ValidationError, match="is not a class"):
        mc.register_function("loss", "d_obj", object(), scope="run")
    # the legacy form never required the entry method (it is checked at build)
    mc.register_function("rollup", "d_no_entry", type("X", (), {}), scope="run")
    # the decorator form is the strict, new one
    with pytest.raises(ComponentContractError, match="missing method rollup"):
        mc.register_function("rollup", "d_strict", scope="run")(type("Y", (), {}))
