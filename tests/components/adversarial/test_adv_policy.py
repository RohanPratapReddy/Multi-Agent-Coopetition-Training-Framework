"""Adversarial verification of D2 for ``trainer_policy`` components (doc 17 section 7).

Real trainer, breach by ``peer-1``. The policy sees a void node with a clamped signal and a
read-only batch; a non-finite or crashing step leaves parameters and optimizer state untouched;
an ``unsafe`` policy does not run unless the operator said so.
"""

from __future__ import annotations

import pytest
import torch

from mactf.components import register_policy
from mactf.exceptions import StepGuardViolation, UnsafePolicyDenied
from mactf.rl.loss import LossBreakdown
from mactf.training.protocol import StepResult

from adv_support import breach_trainer, run_epochs
from p2_support import configure, snapshot_parameters
from test_step_guard import Healthy

VOID = "peer-1"


class _Spy(Healthy):
    name = "spy"
    seen: list = []

    def train_step(self, node, batch):
        _, peer = self.host.resolve_node(node)
        type(self).seen.append((peer.name, bool(batch.get("void", False)), float(batch.get("signal", 0.0)),
                                float(batch.get("floor", 0.0))))
        return super().train_step(node, batch)


class _EditsBatch(Healthy):
    name = "edits_batch"

    def train_step(self, node, batch):
        batch["void"] = False                      # a MappingProxyType: refused
        batch["signal"] = 1e9
        return super().train_step(node, batch)


class _NaNLive(Healthy):
    name = "nan_live"

    def train_step(self, node, batch):
        result = super().train_step(node, batch)
        return StepResult(loss=result.loss * float("nan"), breakdown=result.breakdown, stepped=True)


class _RaisesAfterWriting(Healthy):
    name = "raises_after_writing"
    self_stepping = True

    def train_step(self, node, batch):
        _, peer = self.host.resolve_node(node)
        with torch.no_grad():
            for parameter in peer.adapter.module.parameters():
                parameter.add_(3.0)
        raise RuntimeError("plugin bug")


class _Unsafe(Healthy):
    name = "unsafe_one"
    unsafe = True


def _trainer(cls, settings, tmp_path, mode="enforce", **trainer_block):
    register_policy(cls.name, cls, scope="run")
    configure(settings, TRAINER={"POLICY": cls.name, "STEP_GUARD": {"MODE": mode}, **trainer_block})
    return breach_trainer(settings, tmp_path)


def test_a_void_node_reaches_the_policy_void_and_with_a_clamped_signal(settings, tmp_path, component_overlay):
    _Spy.seen = []
    trainer = _trainer(_Spy, settings, tmp_path)
    run_epochs(trainer)
    mine = [row for row in _Spy.seen if row[0] == VOID]
    assert mine and all(void and signal <= -1.0 and floor < 0.0 for _, void, signal, floor in mine), mine
    others = [row for row in _Spy.seen if row[0] != VOID]
    assert others and not any(void for _, void, _, _ in others)


def test_a_policy_cannot_edit_its_batch(settings, tmp_path, component_overlay):
    trainer = _trainer(_EditsBatch, settings, tmp_path)
    with pytest.raises((TypeError, StepGuardViolation)):
        run_epochs(trainer)


@pytest.mark.parametrize("cls", [_NaNLive, _RaisesAfterWriting])
def test_a_bad_step_changes_nothing(cls, settings, tmp_path, component_overlay):
    trainer = _trainer(cls, settings, tmp_path)
    before = snapshot_parameters(trainer)
    with pytest.raises((StepGuardViolation, RuntimeError)):
        trainer.run_epoch(0)
    after = snapshot_parameters(trainer)
    for key, tensor in before.items():
        assert torch.isfinite(after[key]).all(), key
    if cls is _RaisesAfterWriting:
        for key in before:
            assert not torch.allclose(after[key], before[key] + 3.0), f"{key} kept the plugin's write"


def test_an_unsafe_policy_needs_the_operators_consent(settings, tmp_path, component_overlay):
    register_policy(_Unsafe.name, _Unsafe, scope="run")
    configure(settings, TRAINER={"POLICY": _Unsafe.name})
    with pytest.raises(UnsafePolicyDenied):
        breach_trainer(settings, tmp_path)
