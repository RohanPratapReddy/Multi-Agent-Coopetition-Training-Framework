"""One contract per kind (document 17 section 3.7), positive and negative."""

from __future__ import annotations

import abc

import pytest
import torch
import torch.nn as nn

from component_helpers import Policy, Term, make_scheduler
from mactf.components import KIND_SPECS, all_kind_ids
from mactf.components import contracts as c
from mactf.coopetition.aggregation import Aggregator
from mactf.coopetition.strategy import IndependentStrategy, PeerStrategy
from mactf.coopetition.weights import UniformAverage
from mactf.scoring.base import Scorer, ScoringKind
from mactf.scoring.penalty import PenaltyScorer
from mactf.scoring.reward_scorer import RewardScorer


def test_every_kind_has_both_contracts_and_a_unique_spec():
    assert set(KIND_SPECS) == set(all_kind_ids())
    for spec in KIND_SPECS.values():
        assert callable(spec.contract) and callable(spec.instance_contract)
        assert spec.kind_id == spec.kind.kind_id(spec.family)


def test_install_is_idempotent():
    before = dict(KIND_SPECS)
    c.install()
    assert set(KIND_SPECS) == set(before)


# -- scorer ---------------------------------------------------------------------------
class GoodStage(RewardScorer):
    pass


class NoKind(Scorer):
    def score(self, ctx):  # pragma: no cover
        raise NotImplementedError


class Abstract(Scorer):
    kind = ScoringKind.REWARD
    order = 3


class PipelineKind(RewardScorer):
    kind = ScoringKind.PIPELINE


class BadOrder(RewardScorer):
    order = -1


class BoolOrder(RewardScorer):
    order = True


class HierarchyStage(RewardScorer):
    kind = ScoringKind.HIERARCHY


@pytest.mark.parametrize(
    "factory, fragment",
    [
        (GoodStage, None),
        (PenaltyScorer, None),
        (HierarchyStage, None),  # accepted: root-only is enforced by the pipeline
        (object, "is not a Scorer subclass"),
        (GoodStage(), "is not a Scorer subclass"),
        (lambda: None, "is not a Scorer subclass"),
        (NoKind, "class attribute 'kind' must be a ScoringKind"),
        (Abstract, "abstract methods not implemented"),
        (PipelineKind, "reserved for the built-in pipeline"),
        (BadOrder, "'order' must be a non-negative int"),
        (BoolOrder, "'order' must be a non-negative int"),
    ],
)
def test_scorer_contract(factory, fragment):
    problems = c.scorer_contract(factory)
    if fragment is None:
        assert problems == []
    else:
        assert any(fragment in p for p in problems), problems


def test_scorer_instance_contract():
    assert c.scorer_instance_contract(GoodStage()) == []
    assert "not a Scorer" in c.scorer_instance_contract(object())[0]


# -- reward term -----------------------------------------------------------------------
class NoRules:
    def contribution(self):  # pragma: no cover
        pass


class EmptyRules(NoRules):
    rule_ids = frozenset()


class NoMethod:
    rule_ids = frozenset({"R1"})


@pytest.mark.parametrize(
    "factory, fragment",
    [
        (Term, None),
        (5, "must be a class"),
        (NoRules, "'rule_ids' must be a non-empty collection"),
        (EmptyRules, "'rule_ids' must be a non-empty collection"),
        (NoMethod, "missing method contribution()"),
    ],
)
def test_reward_term_contract(factory, fragment):
    problems = c.reward_term_contract(factory)
    assert (problems == []) if fragment is None else any(fragment in p for p in problems)


def test_reward_term_instance_contract():
    assert c.reward_term_instance_contract(Term()) == []
    assert c.reward_term_instance_contract(object()) == ["no contribution() method"]


# -- rl functions ----------------------------------------------------------------------
class WithCompute:
    required_fields = frozenset({"policy"})

    def compute(self, batch):  # pragma: no cover
        pass


class LossNoFields:
    def compute(self, batch):  # pragma: no cover
        pass


class LossListFields(LossNoFields):
    required_fields = ["policy"]


class WithRollup:
    def rollup(self, parent, children):  # pragma: no cover
        pass


class Callable_:
    def __call__(self, *args):  # pragma: no cover
        pass


def test_rl_entry_methods_cover_every_family():
    from mactf.components import RL_FAMILIES

    assert set(c.RL_ENTRY) == set(RL_FAMILIES)
    assert c.RL_ENTRY["loss"] == "compute" and c.RL_ENTRY["rollup"] == "rollup"
    assert {f for f, m in c.RL_ENTRY.items() if m == "__call__"} == set(RL_FAMILIES) - {"loss", "rollup"}


@pytest.mark.parametrize(
    "family, factory, fragment",
    [
        ("loss", WithCompute, None),
        ("loss", LossNoFields, "required_fields as a frozenset"),
        ("loss", LossListFields, "required_fields as a frozenset"),
        ("loss", WithRollup, "missing method compute()"),
        ("loss", 5, "is not a class"),
        ("rollup", WithRollup, None),
        ("rollup", WithCompute, "missing method rollup()"),
        ("advantage", Callable_, None),
        ("metric", Callable_, None),
        ("value", int, None),  # every class has __call__ through type; see the note in 8.7.3
        ("advantage", lambda: None, "is not a class"),
    ],
)
def test_rl_function_contract(family, factory, fragment):
    problems = c.rl_function_contract(family)(factory)
    assert (problems == []) if fragment is None else any(fragment in p for p in problems), problems


def test_the_legacy_rl_contract_only_requires_a_class():
    assert c.rl_function_legacy_contract(WithRollup) == []
    assert c.rl_function_legacy_contract(type("Marker", (), {})) == []
    assert "is not a class" in c.rl_function_legacy_contract(object())[0]
    assert KIND_SPECS["rl_function.rollup"].legacy_contract is c.rl_function_legacy_contract
    assert KIND_SPECS["scorer"].legacy_contract is None


def test_rl_function_instance_contract():
    assert c.rl_function_instance_contract("loss")(WithCompute()) == []
    assert c.rl_function_instance_contract("loss")(WithRollup()) == ["no compute() method"]
    assert c.rl_function_instance_contract("advantage")(Callable_()) == []


# -- optimizer -----------------------------------------------------------------------
class TorchLike(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        super().__init__(params, {"lr": lr})

    def step(self, closure=None):  # pragma: no cover
        return None


def test_optimizer_contract_accepts_both_documented_forms():
    from mactf.training.optimizer import CallableOptimizer, ParamGroupRouter, ScipyOptimizer, TorchOptimizer

    for backend in (TorchOptimizer, ScipyOptimizer, CallableOptimizer, ParamGroupRouter):
        assert c.optimizer_contract(backend) == []
    assert c.optimizer_contract(TorchLike) == []
    assert c.optimizer_contract(torch.optim.AdamW) == []
    for bad in (object, nn.Linear, 3, lambda: None):
        problems = c.optimizer_contract(bad)
        assert problems and ("not a class" in problems[0] or "neither a subclass" in problems[0])


def test_optimizer_contract_names_the_backend_base_by_either_name():
    from mactf.training import optimizer

    assert c._backend_base() is getattr(optimizer, "BackendBase", None) or c._backend_base() is optimizer._Backend


def test_optimizer_builder_wraps_a_torch_optimizer_in_the_facade():
    from mactf.components.registry import Component
    from mactf.components.kinds import Mutability, Origin
    from mactf.training.optimizer import TorchOptimizer

    component = Component("optimizer", "adamw", torch.optim.AdamW, Origin.PROJECT, Mutability.EPISODE)
    built = c.optimizer_builder(component, {"lr": 3e-4, "weight_decay": 0.01, "betas": (0.8, 0.9)},
                                {"params": object(), "parameters": object()})
    assert isinstance(built, TorchOptimizer) and built.weight_decay == 0.01
    assert built.kwargs == {"betas": (0.8, 0.9)}
    assert c.optimizer_instance_contract(built) == []
    assert "built object" in c.optimizer_instance_contract(object())[0]


def test_optimizer_builder_does_not_mutate_its_arguments():
    from mactf.components.registry import Component
    from mactf.components.kinds import Mutability, Origin

    component = Component("optimizer", "adamw", torch.optim.AdamW, Origin.PROJECT, Mutability.EPISODE)
    values = {"lr": 1e-3, "weight_decay": 0.1}
    c.optimizer_builder(component, values, {})
    assert values == {"lr": 1e-3, "weight_decay": 0.1}


def test_optimizer_builder_calls_a_backend_class_directly():
    from mactf.components.registry import Component
    from mactf.components.kinds import Mutability, Origin
    from mactf.training.optimizer import CallableOptimizer

    seen = {}

    class Mine(CallableOptimizer):
        def __init__(self, **kwargs):
            seen.update(kwargs)
            super().__init__(lambda *a, **k: None)

    component = Component("optimizer", "mine", Mine, Origin.PROJECT, Mutability.EPISODE)
    c.optimizer_builder(component, {"x": 1}, {"params": 1, "other": 2})
    assert seen == {"x": 1, "other": 2}


# -- scheduler / peer factory ------------------------------------------------------------
@pytest.mark.parametrize(
    "factory, fragment",
    [
        (make_scheduler, None),
        (lambda optimizer: None, None),
        (lambda **kw: None, None),
        (lambda gamma: None, "'optimizer' argument"),
        (7, "must be a class or a callable factory"),
    ],
)
def test_scheduler_contract(factory, fragment):
    problems = c.scheduler_contract(factory)
    assert (problems == []) if fragment is None else any(fragment in p for p in problems)


def test_scheduler_instance_contract():
    assert c.scheduler_instance_contract(make_scheduler(None)) == []
    assert c.scheduler_instance_contract(object()) == ["no step() method"]


@pytest.mark.parametrize(
    "factory, fragment",
    [(lambda spec: nn.Linear(1, 1), None), (lambda **kw: None, None),
     (lambda other: None, "'spec' (PeerSpec) argument"), ("x", "must be a class or a callable factory")],
)
def test_peer_factory_contract(factory, fragment):
    problems = c.peer_factory_contract(factory)
    assert (problems == []) if fragment is None else any(fragment in p for p in problems)


def test_peer_factory_instance_contract():
    assert c.peer_factory_instance_contract(nn.Linear(1, 1)) == []
    assert "returns an nn.Module" in c.peer_factory_instance_contract(object())[0]


# -- peer strategy / weight update / aggregator / policy --------------------------------------
class GoodStrategy(IndependentStrategy):
    name = "good"


class NamelessStrategy(PeerStrategy):
    def on_epoch_end(self, *a, **k):  # pragma: no cover
        pass

    def should_aggregate(self, *a, **k):  # pragma: no cover
        pass


class AbstractStrategy(PeerStrategy, abc.ABC):
    """``on_epoch_end`` is no longer abstract (document 17 section 11.3), so a class that is
    still abstract declares its own abstract method."""

    name = "abstract"

    @abc.abstractmethod
    def decide(self) -> None: ...


def test_peer_strategy_contract():
    assert c.peer_strategy_contract(GoodStrategy) == []
    assert "not a subclass" in c.peer_strategy_contract(object)[0]
    assert any("'name' must be a str" in p for p in c.peer_strategy_contract(NamelessStrategy))
    assert any("abstract" in p for p in c.peer_strategy_contract(AbstractStrategy))
    assert c.peer_strategy_instance_contract(GoodStrategy()) == []
    assert "built object" in c.peer_strategy_instance_contract(object())[0]


class GoodWeights:
    name = "good"

    def apply(self, *a, **k):  # pragma: no cover
        pass

    def describe(self):  # pragma: no cover
        return {}

    @classmethod
    def from_spec(cls, spec):  # pragma: no cover
        return cls()


class WeightsNoFromSpec:
    name = "x"

    def apply(self, *a):  # pragma: no cover
        pass

    def describe(self):  # pragma: no cover
        return {}


def test_weight_update_contract():
    assert c.weight_update_contract(GoodWeights) == []
    assert c.weight_update_contract(UniformAverage) == []
    assert any("missing method from_spec()" in p for p in c.weight_update_contract(WeightsNoFromSpec))
    nameless = type("N", (GoodWeights,), {"name": 3})
    assert any("'name' must be a str" in p for p in c.weight_update_contract(nameless))
    assert c.weight_update_contract(5) == ["5 is not a class"]
    assert c.weight_update_instance_contract(GoodWeights()) == []
    assert c.weight_update_instance_contract(object()) == ["no apply() method"]


def test_aggregator_contract():
    assert c.aggregator_contract(Aggregator) == []
    assert "not a subclass" in c.aggregator_contract(object)[0]
    assert c.aggregator_instance_contract(object()) == ["built object"]


def test_trainer_policy_contract():
    assert c.trainer_policy_contract(Policy) == []
    assert c.trainer_policy_contract(5) == ["a trainer policy must be a class"]
    assert c.trainer_policy_contract(object) == ["missing method train_step()"]
    assert c.trainer_policy_instance_contract(Policy()) == []
    assert c.trainer_policy_instance_contract(object()) == ["no train_step() method"]


def test_an_abstract_train_step_is_not_an_implementation():
    class Base(abc.ABC):
        @abc.abstractmethod
        def train_step(self, node, batch):
            ...

    assert c.trainer_policy_contract(Base) == ["abstract method train_step() is not implemented"]
