"""An override that changes nothing changes nothing, bit for bit (invariants I1 and I10; document 19.4).

Each case runs two seeded two-epoch trainings on the same settings: once with the built-in,
once with an identity re-implementation registered under the built-in's name and reached by
the configuration. Parameters, aggregation weights, scores and the RNG state must be equal
under ``torch.equal`` (never ``allclose``).
"""

from __future__ import annotations

import pytest

import custom_kinds as ck
from override_support import assert_same_run, identity_run, override_scope, put, run_snapshot, settings_for
from p2_support import configure

CASES = {
    "policy": (("trainer_policy", "default", ck.SameDefaultPolicy), {"TRAINER": {"POLICY": "default"}}),
    "weight_kappa": (("weight_update", "kappa_softmax", ck.SameKappa),
                     {"TRAINER": {"WEIGHT_UPDATE": {"type": "kappa_softmax", "params": {"tag": "t"}}}}),
    "peer_strategy": (("peer_strategy", "kappa_softmax", ck.SameKappaStrategy),
                      {"TRAINER": {"PEER_STRATEGY": "kappa_softmax"}}),
    "penalty": (("scorer", "penalty", ck.SamePenalty), {"SCORING": {"STAGES": {"penalty": "penalty"}}}),
    "reward": (("scorer", "reward", ck.SameReward), {"SCORING": {"STAGES": {"reward": "reward"}}}),
    "warning": (("scorer", "warning", ck.SameWarning), {"SCORING": {"STAGES": {"warning": "warning"}}}),
    "advantage": (("rl_function.advantage", "gae", ck.SameGAE), {"RL": {"ADVANTAGE": {"type": "gae"}}}),
    "loss": (("rl_function.loss", "policy_gradient", ck.SamePolicyGradient),
             {"RL": {"LOSS": {"type": "policy_gradient"}}}),
    "aggregator": (("aggregator", "default", ck.DeterministicAggregator), {"TRAINER": {"AGGREGATOR": "default"}}),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_an_identity_override_is_bitwise_the_builtin(settings, tmp_path, case):
    registration, blocks = CASES[case]
    # the built-in itself takes no ``tag``; only the override is reached through the params spec
    plain = {"TRAINER": {"WEIGHT_UPDATE": {"type": "kappa_softmax"}}} if case == "weight_kappa" else blocks
    settings_for(settings, **plain)
    with override_scope():
        _, baseline = identity_run(settings, tmp_path / "base")
    settings_for(settings, **blocks)
    with override_scope():
        put(*registration)
        _, overridden = identity_run(settings, tmp_path / "over")
    assert_same_run(baseline, overridden)


def test_optimizer_adam_identity(settings, tmp_path):
    base = {"OPTIMIZER": {"TYPE": "torch", "CLASS": "Adam", "LR": 1e-3, "WEIGHT_DECAY": 0.0}}
    settings_for(settings, **base)
    with override_scope():
        _, baseline = identity_run(settings, tmp_path / "base")
    with override_scope():
        put("optimizer", "Adam", ck.SameAdam)
        configure(settings, OPTIMIZER={"TYPE": "Adam"})
        trainer, overridden = identity_run(settings, tmp_path / "over")
        assert type(trainer.optimizers["peer-1"].backend.optimizer) is ck.SameAdam
    # F5: the E056 probe builds a scratch module on the global RNG, so the stream moves; the run does not
    assert_same_run(baseline, overridden, rng=False)


def test_scheduler_steplr_identity(settings, tmp_path):
    import torch

    def step_lr(optimizer, *, step_size: int = 1, gamma: float = 0.5):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    optimizer = {"TYPE": "torch", "CLASS": "SGD", "LR": 0.05, "WEIGHT_DECAY": 0.0,
                 "SCHEDULER": "StepLR", "SCHEDULER_KWARGS": {"step_size": 1, "gamma": 0.5}}
    settings_for(settings, OPTIMIZER=optimizer)
    with override_scope():
        _, baseline = identity_run(settings, tmp_path / "base")
    with override_scope():
        put("scheduler", "StepLR", step_lr)
        _, overridden = identity_run(settings, tmp_path / "over")
    assert_same_run(baseline, overridden, rng=False)    # F5, as for the optimizer


def test_a_reward_term_identity_through_composed_reward(settings, tmp_path):
    from mactf.scoring.reward_terms import SeverityScaledTerm

    class SameTerm(SeverityScaledTerm):
        pass

    stage = {"reward": {"type": "composed_reward", "params": {"terms": [{"type": "severity_scaled"}]}}}
    settings_for(settings, SCORING={"STAGES": stage})
    with override_scope():
        _, baseline = identity_run(settings, tmp_path / "base")
    with override_scope():
        put("reward_term", "severity_scaled", SameTerm)
        _, overridden = identity_run(settings, tmp_path / "over")
    assert_same_run(baseline, overridden)


def test_uniform_weight_identity(settings, tmp_path):
    settings_for(settings, TRAINER={"WEIGHT_UPDATE": "uniform"})
    with override_scope():
        _, baseline = identity_run(settings, tmp_path / "base")
    with override_scope():
        put("weight_update", "uniform", ck.UniformCopy)
        _, overridden = identity_run(settings, tmp_path / "over")
    assert_same_run(baseline, overridden)


def test_the_peer_factory_identity(settings, tmp_path):
    import sys
    from pathlib import Path

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
    from mutation_support import mutable_trainer, mutation_settings

    mutation_settings(settings)
    results = []
    for index, override in enumerate((False, True)):
        with override_scope():
            if override:
                put("peer_factory", "clone", ck.byte_clone)
            trainer = mutable_trainer(settings, tmp_path / str(index), peers=2, seed=5)
            trainer.add_peer("peer-9", like="peer-1", factory="clone")
            results.append({k: v.clone() for k, v in trainer.peers["peer-9"].adapter.module.state_dict().items()})
    assert results[0].keys() == results[1].keys()
    assert all(torch.equal(results[0][key], results[1][key]) for key in results[0])


def test_registered_but_unconfigured_overrides_change_nothing(settings, tmp_path):
    """The default-path build never asks the registry, so a registration alone is invisible (I1)."""
    import torch

    settings_for(settings)
    with override_scope():
        _, baseline = identity_run(settings, tmp_path / "base")
    with override_scope():
        put("trainer_policy", "default", ck.SameDefaultPolicy)
        put("weight_update", "kappa_softmax", ck.SameKappa)
        put("peer_strategy", "kappa_softmax", ck.SameKappaStrategy)
        put("scorer", "penalty", ck.SamePenalty)
        put("scorer", "reward", ck.SameReward)
        put("scorer", "warning", ck.SameWarning)
        put("aggregator", "default", ck.DeterministicAggregator)
        put("optimizer", "Adam", ck.SameAdam)
        _, silent = identity_run(settings, tmp_path / "over")
    assert_same_run(baseline, silent)
    assert torch.is_grad_enabled()


def test_the_run_is_deterministic_without_any_override(settings, tmp_path):
    """The yardstick itself: the same seed twice is the same run, or no identity test above means anything."""
    settings_for(settings)
    with override_scope():
        first = identity_run(settings, tmp_path / "a")[1]
        second = identity_run(settings, tmp_path / "b")[1]
    assert_same_run(first, second)


def test_run_snapshot_sees_scores_and_weights(settings, tmp_path):
    settings_for(settings)
    with override_scope():
        trainer = identity_run(settings, tmp_path)[0]
        snapshot = run_snapshot(trainer, 0)
    assert snapshot["params"] and "weights" in snapshot and "scores" in snapshot


def test_f5_a_reached_optimizer_override_moves_the_global_rng(settings, tmp_path):
    """F5, pinned: the E056 probe draws from the global RNG; parameters stay bitwise equal, the stream does not."""
    import torch

    settings_for(settings, OPTIMIZER={"TYPE": "torch", "CLASS": "Adam", "LR": 1e-3, "WEIGHT_DECAY": 0.0})
    with override_scope():
        _, baseline = identity_run(settings, tmp_path / "base")
    with override_scope():
        put("optimizer", "Adam", ck.SameAdam)
        configure(settings, OPTIMIZER={"TYPE": "Adam"})
        _, overridden = identity_run(settings, tmp_path / "over")
    assert not torch.equal(baseline["rng"], overridden["rng"])
