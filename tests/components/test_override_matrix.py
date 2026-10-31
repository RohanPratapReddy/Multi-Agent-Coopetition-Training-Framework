"""Reachability matrix: replacing a built-in, kind by kind, through a real build (document 17 section 4.3).

Every *reached* cell asserts one tuple: the slot's component is the project's, flagged
``overrides_builtin``, exactly one ``W001`` named the replacement, and the instance that runs
is the override's. The *inert* cells pin three asymmetries the framework has today (the
override is registered and recorded, yet a default-path build never asks the registry):

* F1 ``trainer_policy:default`` with no ``TRAINER.POLICY`` keeps running ``DefaultTrainPolicy``.
* F2 a plain ``kappa_softmax`` weight strategy, an unauthored peer strategy and
  ``TRAINER.AGGREGATOR: null`` are built directly.
* F3 the two compose: the manifest records a component the run never used.
"""

from __future__ import annotations

import logging

import pytest

from mactf.components.guards.step import GuardedPolicy
from mactf.coopetition.aggregation import Aggregator
from mactf.coopetition.strategy import KappaSoftmaxStrategy
from mactf.coopetition.weights import KappaSoftmaxAverage, UniformAverage
from mactf.training.policy import DefaultTrainPolicy

import custom_kinds as ck
from override_support import built, override_scope, put, settings_for, w001, weight_instance
from p2_support import configure, make_trainer


@pytest.fixture(autouse=True)
def _log(caplog):
    caplog.set_level(logging.WARNING, logger="mactf.components")


def _build(settings, tmp_path, **blocks):
    settings_for(settings, **blocks)
    return make_trainer(settings, tmp_path).build()


# -- trainer_policy --------------------------------------------------------------------
def test_authored_policy_default_runs_the_override_under_a_guard(settings, tmp_path, caplog):
    with override_scope():
        put("trainer_policy", "default", ck.SameDefaultPolicy)
        trainer = _build(settings, tmp_path, TRAINER={"POLICY": "default"})
        component, _ = built(trainer, "TRAINER.POLICY")
        assert component.origin.value == "project" and component.overrides_builtin
        assert isinstance(trainer._policy, GuardedPolicy)
        assert isinstance(trainer._trainer_policy, ck.SameDefaultPolicy)
        assert w001(caplog) == ["trainer_policy:default"]


def test_unauthored_policy_default_is_inert_F1(settings, tmp_path, caplog):
    with override_scope():
        put("trainer_policy", "default", ck.SameDefaultPolicy)
        trainer = _build(settings, tmp_path)
        component, _ = built(trainer, "TRAINER.POLICY")
        assert component.origin.value == "project" and component.overrides_builtin   # the slot says so ...
        assert type(trainer._policy) is DefaultTrainPolicy                           # ... the run did not
        assert trainer._trainer_policy is None
        assert w001(caplog) == ["trainer_policy:default"]


@pytest.mark.parametrize("name, cls", [
    ("anchored_curriculum", ck.AnchoredCurriculumPolicy),
    ("accumulating", ck.AccumulatingPolicy),
    ("void_aware", ck.VoidAwarePolicy),
])
def test_a_new_policy_name_is_reached_and_is_not_an_override(settings, tmp_path, caplog, name, cls):
    with override_scope():
        put("trainer_policy", name, cls)
        trainer = _build(settings, tmp_path, TRAINER={"POLICY": name})
        component, _ = built(trainer, "TRAINER.POLICY")
        assert not component.overrides_builtin and isinstance(trainer._trainer_policy, cls)
        assert w001(caplog) == []


# -- weight_update ---------------------------------------------------------------------
def test_weight_kappa_with_params_reaches_the_override(settings, tmp_path, caplog):
    with override_scope():
        put("weight_update", "kappa_softmax", ck.SameKappa)
        trainer = _build(settings, tmp_path,
                         TRAINER={"WEIGHT_UPDATE": {"type": "kappa_softmax", "params": {"tag": "x"}}})
        component, _ = built(trainer, "TRAINER.WEIGHT_UPDATE")
        instance = weight_instance(trainer)
        assert component.overrides_builtin and isinstance(instance, ck.SameKappa) and instance.tag == "x"
        assert w001(caplog) == ["weight_update:kappa_softmax"]


@pytest.mark.parametrize("trainer_block", [None, {"WEIGHT_UPDATE": "kappa_softmax"}],
                         ids=["unauthored", "plain-name"])
def test_plain_weight_kappa_is_inert_F2(settings, tmp_path, caplog, trainer_block):
    with override_scope():
        put("weight_update", "kappa_softmax", ck.SameKappa)
        blocks = {"TRAINER": trainer_block} if trainer_block else {}
        trainer = _build(settings, tmp_path, **blocks)
        instance = weight_instance(trainer)
        assert type(instance) is KappaSoftmaxAverage
        assert w001(caplog) == ["weight_update:kappa_softmax"]


def test_f3_the_manifest_records_a_component_the_run_never_used(settings, tmp_path):
    with override_scope():
        put("weight_update", "kappa_softmax", ck.SameKappa)
        trainer = _build(settings, tmp_path)
        entry = trainer._component_manifest()["weight_update"]["TRAINER.WEIGHT_UPDATE"]
        assert entry["overrides_builtin"] is True and entry["origin"] == "project"
        assert type(weight_instance(trainer)) is KappaSoftmaxAverage


def test_uniform_override_is_reached_by_name(settings, tmp_path, caplog):
    with override_scope():
        put("weight_update", "uniform", ck.UniformCopy)
        trainer = _build(settings, tmp_path, TRAINER={"WEIGHT_UPDATE": "uniform"})
        assert isinstance(weight_instance(trainer), ck.UniformCopy)
        assert not isinstance(weight_instance(trainer), UniformAverage)
        assert w001(caplog) == ["weight_update:uniform"]


@pytest.mark.parametrize("name, cls", [("entropy_floor", ck.EntropyFloorAverage),
                                       ("rank_weighted", ck.RankWeightedAverage)])
def test_a_new_weight_name_is_reached(settings, tmp_path, caplog, name, cls):
    with override_scope():
        put("weight_update", name, cls)
        trainer = _build(settings, tmp_path, TRAINER={"WEIGHT_UPDATE": name})
        assert isinstance(weight_instance(trainer), cls) and w001(caplog) == []


# -- peer_strategy ---------------------------------------------------------------------
def test_authored_peer_strategy_reaches_the_override(settings, tmp_path, caplog):
    with override_scope():
        put("peer_strategy", "kappa_softmax", ck.SameKappaStrategy)
        trainer = _build(settings, tmp_path, TRAINER={"PEER_STRATEGY": "kappa_softmax"})
        assert isinstance(trainer.strategy_plan.default_instance(), ck.SameKappaStrategy)
        assert w001(caplog) == ["peer_strategy:kappa_softmax"]


def test_unauthored_peer_strategy_is_inert_F2(settings, tmp_path, caplog):
    with override_scope():
        put("peer_strategy", "kappa_softmax", ck.SameKappaStrategy)
        trainer = _build(settings, tmp_path)
        assert type(trainer.strategy_plan.default_instance()) is KappaSoftmaxStrategy
        component, _ = built(trainer, "TRAINER.PEER_STRATEGY")
        assert component.overrides_builtin


def test_a_new_peer_strategy_name_is_reached(settings, tmp_path, caplog):
    with override_scope():
        put("peer_strategy", "staggered_mix", ck.DeterministicStagger)
        trainer = _build(settings, tmp_path, TRAINER={"PEER_STRATEGY": {"type": "staggered_mix",
                                                                         "params": {"period": 2}}})
        assert isinstance(trainer.strategy_plan.default_instance(), ck.DeterministicStagger)
        assert w001(caplog) == []


# -- scorer ----------------------------------------------------------------------------
@pytest.mark.parametrize("stage, cls", [("penalty", ck.SamePenalty), ("reward", ck.SameReward),
                                        ("warning", ck.SameWarning)])
def test_scoring_stage_override_is_reached(settings, tmp_path, caplog, stage, cls):
    with override_scope():
        put("scorer", stage, cls)
        trainer = _build(settings, tmp_path, SCORING={"STAGES": {stage: stage}})
        assert any(type(s) is cls for s in trainer.scorer.chain)
        component, _ = built(trainer, f"scoring.{stage}")
        assert component.origin.value == "project" and component.overrides_builtin
        assert w001(caplog) == [f"scorer:{stage}"]


def test_a_new_stage_name_is_reached(settings, tmp_path, caplog):
    with override_scope():
        put("scorer", "tempered", ck.TemperedRewardScorer)
        trainer = _build(settings, tmp_path, SCORING={"STAGES": {"reward": "tempered"}})
        assert any(type(s) is ck.TemperedRewardScorer for s in trainer.scorer.chain)
        assert w001(caplog) == []


# -- reward_term -----------------------------------------------------------------------
def test_reward_term_override_is_reached_through_composed_reward(settings, tmp_path, caplog):
    with override_scope():
        put("reward_term", "severity_scaled", ck.DiminishingTerm)
        trainer = _build(settings, tmp_path, SCORING={"STAGES": {
            "reward": {"type": "composed_reward", "params": {"terms": [{"type": "severity_scaled"}]}}}})
        stage = next(s for s in trainer.scorer.chain if hasattr(s, "terms"))
        assert [type(term) for _, _, term in stage.terms] == [ck.DiminishingTerm]
        assert w001(caplog) == ["reward_term:severity_scaled"]


# -- rl_function -----------------------------------------------------------------------
def test_advantage_override_is_reached(settings, tmp_path, caplog):
    with override_scope():
        put("rl_function.advantage", "gae", ck.SameGAE)
        trainer = _build(settings, tmp_path, RL={"ADVANTAGE": {"type": "gae"}})
        assert isinstance(trainer.functions.get("advantage", "peer-1"), ck.SameGAE)
        assert w001(caplog) == ["rl_function.advantage:gae"]


def test_loss_override_is_reached(settings, tmp_path, caplog):
    with override_scope():
        put("rl_function.loss", "policy_gradient", ck.SamePolicyGradient)
        trainer = _build(settings, tmp_path, RL={"LOSS": {"type": "policy_gradient"}})
        loss = trainer.loss_for("peer-1")
        # the framework wraps a configured loss with its compliance term; the override is inside
        assert any(isinstance(term, ck.SamePolicyGradient) for term in getattr(loss, "terms", [loss]))
        assert w001(caplog) == ["rl_function.loss:policy_gradient"]


# -- optimizer / scheduler -------------------------------------------------------------
def test_optimizer_adam_override_is_reached(settings, tmp_path, caplog):
    with override_scope():
        put("optimizer", "Adam", ck.SameAdam)
        trainer = _build(settings, tmp_path, OPTIMIZER={"TYPE": "Adam", "LR": 1e-3})
        backend = trainer.optimizers["peer-1"].backend
        assert type(backend.optimizer) is ck.SameAdam
        assert w001(caplog) == ["optimizer:Adam"]


def test_optimizer_torch_class_spelling_is_inert_F4(settings, tmp_path, caplog):
    """F4: ``TYPE: torch, CLASS: Adam`` is read from ``torch.optim`` without asking the registry."""
    import torch

    with override_scope():
        put("optimizer", "Adam", ck.SameAdam)
        trainer = _build(settings, tmp_path, OPTIMIZER={"TYPE": "torch", "CLASS": "Adam", "LR": 1e-3})
        assert type(trainer.optimizers["peer-1"].backend.optimizer) is torch.optim.Adam
        assert w001(caplog) == ["optimizer:Adam"]


def test_scheduler_steplr_override_is_reached(settings, tmp_path, caplog):
    import torch

    seen = []

    def step_lr(optimizer, *, step_size: int = 1, gamma: float = 0.5):
        seen.append(step_size)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    with override_scope():
        put("scheduler", "StepLR", step_lr)
        _build(settings, tmp_path, OPTIMIZER={"TYPE": "torch", "CLASS": "SGD", "LR": 0.1,
                                              "SCHEDULER": "StepLR", "SCHEDULER_KWARGS": {"step_size": 2}})
        assert seen and set(seen) == {2}
        assert w001(caplog) == ["scheduler:StepLR"]


# -- aggregator ------------------------------------------------------------------------
def test_aggregator_override_is_reached_by_name_and_inert_when_unauthored(settings, tmp_path, caplog):
    with override_scope():
        put("aggregator", "default", ck.DeterministicAggregator)
        trainer = _build(settings, tmp_path, TRAINER={"AGGREGATOR": "default"})
        assert type(trainer.aggregator) is ck.DeterministicAggregator
    with override_scope():
        put("aggregator", "default", ck.DeterministicAggregator)
        configure(settings, TRAINER={"AGGREGATOR": None})
        trainer = make_trainer(settings, tmp_path / "again").build()
        assert type(trainer.aggregator) is Aggregator                       # F2: built directly


# -- peer_factory ----------------------------------------------------------------------
def test_peer_factory_clone_override_builds_the_newcomer(settings, tmp_path, caplog):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
    from mutation_support import mutable_trainer, mutation_settings

    calls = []

    def clone(spec, source=None):
        calls.append(spec)
        return ck.byte_clone(spec, source)

    with override_scope():
        put("peer_factory", "clone", clone)
        mutation_settings(settings)
        trainer = mutable_trainer(settings, tmp_path, peers=2)
        trainer.add_peer("peer-9", like="peer-1", factory="clone")
        assert calls and "peer-9" in trainer.peers
        assert w001(caplog) == ["peer_factory:clone"]
