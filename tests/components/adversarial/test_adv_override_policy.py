"""Every step-guard code, reached through a *built-in's own name* (document 17 sections 7.6 and 19.6).

The offenders of ``test_step_guard`` are registered as ``trainer_policy:default`` and
``trainer_policy:co_distill`` with the override flag. A policy that takes the default's name
gets no softer treatment than one with a new name: it is wrapped, judged, and everything it
touched is put back. A registration the configuration never reaches changes nothing (F1).
"""

from __future__ import annotations

import logging

import pytest
import torch

from mactf.components.guards.step import GuardedPolicy, StepGuardConfig
from mactf.exceptions import StepGuardViolation, UnsafePolicyDenied
from mactf.training.policy import DefaultTrainPolicy

import custom_kinds as ck
from adv_support import breach_trainer, run_epochs
from override_support import moments, override_scope, put, same_tensors, settings_for, step, step_batch, w001
from p2_support import make_trainer, snapshot_parameters
from test_step_guard import (
    BadBreakdown, DetachedButLive, Healthy, HugeGradient, LiesAboutStepping, MutatesWeights, NaNLoss,
    PoisonsWeights, RewardsVoid, SkipsCompliance, StepsOnVoid, VectorNaN,
)

NAMES = ("default", "co_distill")


@pytest.fixture(autouse=True)
def _scope(caplog):
    caplog.set_level(logging.WARNING, logger="mactf.components")
    with override_scope():
        yield


def _trainer(settings, tmp_path, cls, name, mode="quarantine", **guard):
    put("trainer_policy", name, cls)
    settings_for(settings, TRAINER={"POLICY": name, "STEP_GUARD": {"MODE": mode, **guard}})
    return make_trainer(settings, tmp_path).build()


#: (offender, a code it must raise, the batch it needs)
SCENARIOS = [
    pytest.param(NaNLoss, "S001", {}, id="S001-nan"),
    pytest.param(VectorNaN, "S001", {}, id="S001-vector"),
    pytest.param(BadBreakdown, "S005", {}, id="S005-breakdown"),
    pytest.param(PoisonsWeights, "S003", {}, id="S003-poisoned"),
    pytest.param(SkipsCompliance, "S004", {"floor": -0.5}, id="S004-compliance"),
    pytest.param(LiesAboutStepping, "S007", {}, id="S007-lies"),
    pytest.param(DetachedButLive, "S007", {}, id="S007-detached"),
    pytest.param(RewardsVoid, "S008", {"void": True, "signal": 2.0}, id="S008-rewards-void"),
    pytest.param(StepsOnVoid, "S008", {"void": True}, id="S008-steps-on-void"),
]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("cls, code, batch", SCENARIOS)
def test_a_builtin_named_offender_is_quarantined_and_everything_it_touched_is_restored(
        settings, tmp_path, cls, code, batch, name):
    trainer = _trainer(settings, tmp_path, cls, name)
    assert isinstance(trainer._policy, GuardedPolicy), "an override must be wrapped like any other custom policy"
    before, held = snapshot_parameters(trainer), moments(trainer)
    result = step(trainer, step_batch(**batch))
    assert result.metrics.get("quarantined") == 1.0 and result.stepped is False
    after = snapshot_parameters(trainer)
    assert same_tensors(before, after) and all(torch.isfinite(t).all() for t in after.values())
    assert moments(trainer)["state"] == held["state"], "optimizer moments were not restored"
    codes = {v.code for v in trainer._policy_guard.events[-1].violations}
    assert code in codes, codes
    assert trainer.global_step == 0 and trainer.losses == []


@pytest.mark.parametrize("cls, code, batch", SCENARIOS)
def test_enforce_mode_restores_then_raises_for_an_override(settings, tmp_path, cls, code, batch):
    trainer = _trainer(settings, tmp_path, cls, "default", mode="enforce")
    before = snapshot_parameters(trainer)
    with pytest.raises(StepGuardViolation) as raised:
        step(trainer, step_batch(**batch))
    assert raised.value.path == "peer-1"
    assert same_tensors(before, snapshot_parameters(trainer))


@pytest.mark.parametrize("cls, code, batch", [s for s in SCENARIOS if s.id != "S001-vector"])
def test_warn_mode_records_and_lets_it_through(settings, tmp_path, cls, code, batch, caplog):
    trainer = _trainer(settings, tmp_path, cls, "default", mode="warn")
    with caplog.at_level(logging.WARNING):
        result = step(trainer, step_batch(**batch))
    assert result.metrics.get("quarantined") is None
    assert trainer._policy_guard.events[-1].action == "warned"
    assert code in caplog.text


def test_warn_mode_warns_then_the_trainers_own_backward_refuses_a_vector_loss(settings, tmp_path, caplog):
    """``warn`` means warn: the violation is logged, and a non-scalar loss still cannot be back-propagated."""
    trainer = _trainer(settings, tmp_path, VectorNaN, "default", mode="warn")
    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="scalar outputs"):
        step(trainer)
    assert "S001" in caplog.text


# -- S002: the step the trainer takes for a live loss ------------------------------------------------
def test_s002_an_exploding_gradient_through_the_default_name_is_undone(settings, tmp_path):
    trainer = _trainer(settings, tmp_path, HugeGradient, "default", grad_norm_ceiling=10.0)
    trainer._policy_guard.config = StepGuardConfig(mode="warn")
    step(trainer)                                    # builds Adam's moments
    trainer._policy_guard.config = StepGuardConfig(mode="quarantine", grad_norm_ceiling=10.0)
    before, held = snapshot_parameters(trainer), moments(trainer)
    result = step(trainer)
    assert result.metrics.get("quarantined") == 1.0
    assert same_tensors(before, snapshot_parameters(trainer))
    for key, state in held["state"].items():
        for item, value in state.items():
            assert torch.equal(torch.as_tensor(value), torch.as_tensor(moments(trainer)["state"][key][item]))
    assert trainer._policy_guard.events[-1].violations[0].code == "S002"


# -- S009: raised mid-step ------------------------------------------------------------------------------
@pytest.mark.parametrize("name", NAMES)
def test_s009_a_raise_after_writing_is_rolled_back_and_reraised(settings, tmp_path, name):
    trainer = _trainer(settings, tmp_path, MutatesWeights, name)
    before, held = snapshot_parameters(trainer), moments(trainer)
    with pytest.raises(RuntimeError, match="boom half way"):
        step(trainer)
    assert same_tensors(before, snapshot_parameters(trainer))
    assert moments(trainer)["state"] == held["state"]
    event = trainer._policy_guard.events[-1]
    assert event.action == "raised" and event.restored and event.violations[0].code == "S009"


def test_s009_warn_mode_restores_nothing_even_for_an_override(settings, tmp_path):
    trainer = _trainer(settings, tmp_path, MutatesWeights, "default", mode="warn")
    before = snapshot_parameters(trainer)
    with pytest.raises(RuntimeError):
        step(trainer)
    assert not same_tensors(before, snapshot_parameters(trainer))


# -- S006: consecutive quarantines stop the run ----------------------------------------------------------
def test_s006_three_quarantines_in_a_row_stop_the_run(settings, tmp_path):
    trainer = _trainer(settings, tmp_path, NaNLoss, "default", max_consecutive=3)
    assert step(trainer).metrics.get("quarantined") == 1.0
    assert step(trainer).metrics.get("quarantined") == 1.0
    with pytest.raises(StepGuardViolation) as raised:
        step(trainer)
    assert "S006" in {v["code"] for v in raised.value.violations} and "consecutive" in str(raised.value)


# -- the void node ------------------------------------------------------------------------------------------
def test_a_void_node_is_clamped_for_an_override_and_the_signal_is_read_only(settings, tmp_path):
    trainer = _trainer(settings, tmp_path, Healthy, "default")
    policy = trainer._trainer_policy
    step(trainer, step_batch(void=True, signal=5.0))
    seen = policy.seen[-1]
    assert seen["signal"] <= -1.0 and seen["void_advantage_ceiling"] == -1.0
    with pytest.raises(TypeError):
        seen["signal"] = 9.0
    step(trainer, step_batch(void=False, signal=5.0))
    assert policy.seen[-1]["signal"] == 5.0


def test_the_breach_run_hands_the_default_named_override_a_void_clamped_node(settings, tmp_path):
    class Spy(Healthy):
        name = "default"
        seen: list = []

        def train_step(self, node, batch):
            _, peer = self.host.resolve_node(node)
            type(self).seen.append((peer.name, bool(batch.get("void", False)), float(batch.get("signal", 0.0))))
            return super().train_step(node, batch)

    Spy.seen = []
    put("trainer_policy", "default", Spy)
    settings_for(settings, TRAINER={"POLICY": "default", "STEP_GUARD": {"MODE": "enforce"}})
    run_epochs(breach_trainer(settings, tmp_path))
    mine = [row for row in Spy.seen if row[0] == "peer-1"]
    assert mine and all(void and signal <= -1.0 for _, void, signal in mine), mine
    assert not any(void for name, void, _ in Spy.seen if name != "peer-1")


def test_a_bad_step_in_a_breach_run_changes_nothing(settings, tmp_path):
    put("trainer_policy", "default", NaNLoss)
    settings_for(settings, TRAINER={"POLICY": "default", "STEP_GUARD": {"MODE": "enforce"}})
    trainer = breach_trainer(settings, tmp_path)
    before = snapshot_parameters(trainer)
    with pytest.raises(StepGuardViolation):
        trainer.run_epoch(0)
    assert all(torch.isfinite(t).all() for t in snapshot_parameters(trainer).values())
    assert same_tensors(before, snapshot_parameters(trainer))


# -- consent, and what is not reached -------------------------------------------------------------------
def test_an_unsafe_override_of_the_default_name_still_needs_consent(settings, tmp_path):
    put("trainer_policy", "default", ck.UnsafePolicy)
    settings_for(settings, TRAINER={"POLICY": "default"})
    with pytest.raises(UnsafePolicyDenied):
        make_trainer(settings, tmp_path).build()


def test_f1_an_offender_registered_as_default_but_never_configured_does_not_run(settings, tmp_path, caplog):
    put("trainer_policy", "default", NaNLoss)
    settings_for(settings)
    trainer = make_trainer(settings, tmp_path).build()
    assert type(trainer._policy) is DefaultTrainPolicy and trainer._policy_guard is None
    before = snapshot_parameters(trainer)
    trainer.run_epoch(0)
    assert not same_tensors(before, snapshot_parameters(trainer))     # it trained, with the real default
    assert w001(caplog) == ["trainer_policy:default"]                 # and the registration was still announced
