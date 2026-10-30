"""``StepGuard`` (document 17 section 7.6): every code, every mode, and the adversarial policies.

The policies below are the ones the guard exists for: a policy that skips the compliance charge,
one that zeroes (or rewards) a void node, one that steps on a void node, one that mutates the
weights, one that raises half way through its own step. Each is run against a real trainer, and
the assertions are about the *state afterwards*: parameters and optimizer moments restored,
nothing recorded in the loss history, the exception unchanged.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, ClassVar, Mapping

import pytest
import torch

from mactf.components import register_policy
from mactf.components.guards.step import (
    CODES,
    GuardedPolicy,
    StepGuard,
    StepGuardConfig,
    VoidProbe,
    register_term_check,
    TERM_CHECKS,
)
from mactf.exceptions import StepGuardViolation, UnsafePolicyDenied, ValidationError
from mactf.rl.loss import LossBreakdown
from mactf.training.protocol import StepResult

from p2_support import configure, make_trainer, snapshot_parameters, tune


# -- policies ------------------------------------------------------------------------------
class _Base:
    name: ClassVar[str] = "base"
    unsafe: ClassVar[bool] = False
    self_stepping: ClassVar[bool | None] = False
    requires_terms: ClassVar[frozenset[str]] = frozenset()
    composes_distillation: ClassVar[bool] = False

    def __init__(self) -> None:
        self.host: Any = None
        self.seen: list[Mapping[str, Any]] = []

    def attach(self, host: Any) -> None:
        self.host = host

    def _forward(self, node: Any, batch: Mapping[str, Any]):
        _, peer = self.host.resolve_node(node)
        logits, values = peer.adapter(batch["states"])
        return peer, logits, values

    def train_step(self, node, batch):  # pragma: no cover - overridden
        raise NotImplementedError


class Healthy(_Base):
    """A live loss the trainer steps for: finite, with the compliance term reported."""

    name = "healthy"
    requires_terms = frozenset({"compliance_floor"})

    def train_step(self, node, batch):
        self.seen.append(batch)
        _, logits, _ = self._forward(node, batch)
        floor = float(batch.get("floor", 0.0))
        compliance = max(-floor, 0.0)
        loss = logits.pow(2).mean() + compliance
        return StepResult(
            loss=loss,
            breakdown=LossBreakdown(compliance=compliance, total=float(loss.detach()),
                                    extras={"advantage_mean": -1.0 if batch.get("void") else 0.1}),
            stepped=True,
        )


class NaNLoss(_Base):
    name = "nan_loss"

    def train_step(self, node, batch):
        _, logits, _ = self._forward(node, batch)
        return StepResult(loss=logits.mean() * float("nan"), breakdown=LossBreakdown(), stepped=True)


class HugeGradient(_Base):
    """A finite loss whose gradient the trainer's step reports as enormous (S002)."""

    name = "huge_gradient"

    def train_step(self, node, batch):
        _, logits, _ = self._forward(node, batch)
        loss = logits.pow(2).mean() * 1.0e9
        return StepResult(loss=loss, breakdown=LossBreakdown(total=float(loss.detach())), stepped=True)


class PoisonsWeights(_Base):
    """Steps itself, then writes ``nan`` into a weight: the optimizer's moments move too."""

    name = "poisons_weights"
    self_stepping = True

    def train_step(self, node, batch):
        peer, logits, _ = self._forward(node, batch)
        optimizer = self.host.optimizers[peer.name]
        optimizer.zero_grad()
        logits.pow(2).mean().backward()
        optimizer.step()
        with torch.no_grad():
            next(peer.adapter.module.parameters()).fill_(float("nan"))
        return StepResult(loss=torch.tensor(0.5), breakdown=LossBreakdown(total=0.5), stepped=True)


class MutatesWeights(_Base):
    """Steps itself and then raises: parameters and Adam moments must both roll back."""

    name = "mutates_then_raises"
    self_stepping = True

    def train_step(self, node, batch):
        peer, logits, _ = self._forward(node, batch)
        optimizer = self.host.optimizers[peer.name]
        optimizer.zero_grad()
        logits.pow(2).mean().backward()
        optimizer.step()
        with torch.no_grad():
            for parameter in peer.adapter.module.parameters():
                parameter.add_(1.0)
        raise RuntimeError("boom half way through the step")


class SkipsCompliance(_Base):
    """Declares the compliance term and never charges it."""

    name = "skips_compliance"
    requires_terms = frozenset({"compliance_floor"})

    def train_step(self, node, batch):
        _, logits, _ = self._forward(node, batch)
        loss = logits.pow(2).mean()
        return StepResult(loss=loss, breakdown=LossBreakdown(compliance=0.0, total=float(loss.detach())),
                          stepped=True)


class RewardsVoid(_Base):
    """Reports a positive advantage for a void node."""

    name = "rewards_void"

    def train_step(self, node, batch):
        _, logits, _ = self._forward(node, batch)
        loss = logits.pow(2).mean()
        return StepResult(
            loss=loss,
            breakdown=LossBreakdown(total=float(loss.detach()), extras={"advantage_mean": 0.7}),
            stepped=True,
        )


class StepsOnVoid(_Base):
    """Self-stepping policy that trains a void node on its (reported) positive credit."""

    name = "steps_on_void"
    self_stepping = True

    def train_step(self, node, batch):
        peer, logits, _ = self._forward(node, batch)
        optimizer = self.host.optimizers[peer.name]
        optimizer.zero_grad()
        loss = -logits.mean()
        loss.backward()
        optimizer.step()
        return StepResult(
            loss=loss.detach(),
            breakdown=LossBreakdown(total=float(loss.detach()), extras={"advantage_mean": 0.9}),
            stepped=True,
        )


class LiesAboutStepping(_Base):
    """Declares ``self_stepping=True`` yet hands back a live loss (S007)."""

    name = "lies"
    self_stepping = True

    def train_step(self, node, batch):
        _, logits, _ = self._forward(node, batch)
        return StepResult(loss=logits.pow(2).mean(), breakdown=LossBreakdown(), stepped=True)


class DetachedButLive(_Base):
    """Declares ``self_stepping=False`` and returns a detached loss (S007, the other way)."""

    name = "detached"
    self_stepping = False

    def train_step(self, node, batch):
        _, logits, _ = self._forward(node, batch)
        return StepResult(loss=logits.pow(2).mean().detach(), breakdown=LossBreakdown(), stepped=False)


class VectorNaN(_Base):
    """A vector loss whose mean is finite but whose first entry is ``nan`` is still ``nan``."""

    name = "vector"

    def train_step(self, node, batch):
        _, logits, _ = self._forward(node, batch)
        loss = logits.pow(2).mean(dim=-1)
        loss = torch.cat([loss[:1] * float("inf"), loss[1:]])
        return StepResult(loss=loss, breakdown=LossBreakdown(), stepped=True)


class BadBreakdown(_Base):
    name = "bad_breakdown"

    def train_step(self, node, batch):
        _, logits, _ = self._forward(node, batch)
        loss = logits.pow(2).mean()
        return StepResult(loss=loss, breakdown=LossBreakdown(kl=float("nan"), total=1.0), stepped=True)


# -- harness -----------------------------------------------------------------------------
def _trainer(settings, tmp_path, policy_cls, *, component_overlay, mode="quarantine", **guard):
    tune(settings)
    register_policy(policy_cls.name, policy_cls, scope="run")
    configure(settings, TRAINER={"POLICY": policy_cls.name,
                                 "STEP_GUARD": {"MODE": mode, **guard}})
    return make_trainer(settings, tmp_path).build()


def _batch(*, void: bool = False, signal: float = 0.5, floor: float = 0.0) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(3)
    return {
        "states": torch.randn(8, 4, generator=generator),
        "targets": torch.zeros(8, dtype=torch.long),
        "signal": signal, "void": void, "floor": floor,
    }


def _equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)


def _moments(trainer, path: str) -> dict[str, Any]:
    return copy.deepcopy(trainer.optimizers[path].backend.state_dict())


def _step(trainer, batch=None, path="peer-1"):
    return trainer.train_step(trainer.tree.get(path), batch or _batch())


# -- the healthy path ----------------------------------------------------------------------
def test_a_healthy_policy_is_not_disturbed(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, Healthy, component_overlay=component_overlay)
    before = snapshot_parameters(trainer)
    result = _step(trainer)
    assert result.metrics.get("quarantined") is None
    assert result.stepped
    assert not _equal(before, snapshot_parameters(trainer))   # the trainer took the step
    guard = trainer._policy_guard
    assert guard.checked == 1 and guard.quarantined == 0 and guard.events == []
    assert trainer.global_step == 1 and len(trainer.losses) == 1


def test_the_default_policy_is_never_wrapped(settings, tmp_path):
    tune(settings)
    trainer = make_trainer(settings, tmp_path).build()
    assert not isinstance(trainer._policy, GuardedPolicy)
    assert trainer._policy_guard is None and trainer._trainer_policy is None


# -- S001 / S005: finite loss, finite breakdown ------------------------------------------------
def test_s001_a_nan_loss_never_reaches_backward(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, NaNLoss, component_overlay=component_overlay)
    before, moments = snapshot_parameters(trainer), _moments(trainer, "peer-1")
    result = _step(trainer)
    assert result.metrics == {"quarantined": 1.0} and result.stepped is False
    assert _equal(before, snapshot_parameters(trainer))          # no step was taken
    assert _moments(trainer, "peer-1") == moments or not moments["state"]
    assert trainer.global_step == 0 and trainer.losses == []      # nothing recorded
    event = trainer._policy_guard.events[-1]
    assert event.action == "quarantined" and event.violations[0].code == "S001"
    assert all(p.grad is None or not p.grad.abs().sum() for p in trainer.peers["peer-1"].adapter.module.parameters())


def test_s001_a_vector_loss_is_judged_entry_by_entry(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, VectorNaN, component_overlay=component_overlay)
    result = _step(trainer)
    assert result.metrics.get("quarantined") == 1.0
    assert trainer._policy_guard.events[-1].violations[0].code == "S001"


def test_s005_a_non_finite_breakdown_number_is_refused(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, BadBreakdown, component_overlay=component_overlay)
    result = _step(trainer)
    assert result.metrics.get("quarantined") == 1.0
    codes = {v.code for v in trainer._policy_guard.events[-1].violations}
    assert "S005" in codes


# -- S002 / S003: the step the trainer takes for a live loss ----------------------------------
def test_s002_an_exploding_gradient_is_undone(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, HugeGradient, component_overlay=component_overlay,
                       grad_norm_ceiling=10.0)
    # The first step builds Adam's moments, so the second shows both are restored.
    trainer._policy_guard.config = StepGuardConfig(mode="warn")
    _step(trainer)
    trainer._policy_guard.config = StepGuardConfig(mode="quarantine", grad_norm_ceiling=10.0)
    before, moments = snapshot_parameters(trainer), _moments(trainer, "peer-1")
    result = _step(trainer)
    assert result.metrics.get("quarantined") == 1.0
    assert _equal(before, snapshot_parameters(trainer))
    assert _moments(trainer, "peer-1")["state"].keys() == moments["state"].keys()
    for key in moments["state"]:
        for name, value in moments["state"][key].items():
            assert torch.equal(torch.as_tensor(value), torch.as_tensor(_moments(trainer, "peer-1")["state"][key][name]))
    assert trainer._policy_guard.events[-1].violations[0].code == "S002"


def test_s003_a_policy_that_poisons_the_weights_is_rolled_back(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, PoisonsWeights, component_overlay=component_overlay)
    before, moments = snapshot_parameters(trainer), _moments(trainer, "peer-1")
    result = _step(trainer)
    assert result.metrics.get("quarantined") == 1.0
    after = snapshot_parameters(trainer)
    assert _equal(before, after)
    assert all(torch.isfinite(t).all() for t in after.values())
    restored = _moments(trainer, "peer-1")
    assert restored["state"] == moments["state"]            # Adam's exp_avg is not poisoned
    event = trainer._policy_guard.events[-1]
    assert event.restored and "S003" in {v.code for v in event.violations}


def test_check_params_false_switches_the_s003_check_off(settings, tmp_path, component_overlay):
    """7.6.2: ``CHECK_PARAMS: false`` turns S003 off and nothing else (S001 still fires)."""
    trainer = _trainer(settings, tmp_path, PoisonsWeights, component_overlay=component_overlay,
                       check_params=False)
    _step(trainer)
    codes = {v.code for event in trainer._policy_guard.events for v in event.violations}
    assert "S003" not in codes


def test_s003_is_judged_after_the_trainers_step_too(settings, tmp_path, component_overlay):
    """A finite loss with a NaN gradient: the loss checks pass, the step it causes does not."""

    class NaNGradient(_Base):
        name = "nan_gradient"

        def train_step(self, node, batch):
            _, logits, _ = self._forward(node, batch)
            loss = (logits * 0.0).sqrt().sum()          # 0.0 exactly, d/dx sqrt(0) * 0 = nan
            assert torch.isfinite(loss)
            return StepResult(loss=loss, breakdown=LossBreakdown(), stepped=True)

    trainer = _trainer(settings, tmp_path, NaNGradient, component_overlay=component_overlay)
    before, moments = snapshot_parameters(trainer), _moments(trainer, "peer-1")
    result = _step(trainer)
    assert result.metrics.get("quarantined") == 1.0
    after = snapshot_parameters(trainer)
    assert _equal(before, after) and all(torch.isfinite(t).all() for t in after.values())
    assert _moments(trainer, "peer-1")["state"] == moments["state"]
    codes = {v.code for v in trainer._policy_guard.events[-1].violations}
    assert codes & {"S002", "S003"}
    assert trainer.global_step == 0


# -- S004: the evidence a promised term must leave -------------------------------------------
def test_s004_skipping_the_compliance_charge_on_a_breached_floor(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, SkipsCompliance, component_overlay=component_overlay)
    before = snapshot_parameters(trainer)
    result = _step(trainer, _batch(floor=-0.5))
    assert result.metrics.get("quarantined") == 1.0
    assert _equal(before, snapshot_parameters(trainer))
    violation = trainer._policy_guard.events[-1].violations[0]
    assert violation.code == "S004" and "compliance_floor" in violation.message


def test_s004_the_same_policy_is_fine_when_the_floor_is_not_breached(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, SkipsCompliance, component_overlay=component_overlay)
    result = _step(trainer, _batch(floor=0.0))
    assert result.metrics.get("quarantined") is None


def test_s004_an_unknown_required_term_is_a_violation_not_a_pass(settings, tmp_path, component_overlay):
    class Unknown(Healthy):
        name = "unknown_term"
        requires_terms = frozenset({"no_such_term"})

    trainer = _trainer(settings, tmp_path, Unknown, component_overlay=component_overlay)
    _step(trainer)
    violation = trainer._policy_guard.events[-1].violations[0]
    assert violation.code == "S004" and "no_such_term" in violation.message


def test_s004_a_registered_term_check_extends_the_vocabulary(settings, tmp_path, component_overlay):
    class Custom(Healthy):
        name = "custom_term"
        requires_terms = frozenset({"always_fails"})

    register_term_check("always_fails", lambda result, batch: "never satisfied")
    try:
        with pytest.raises(ValidationError):
            register_term_check("always_fails", lambda r, b: None)
        trainer = _trainer(settings, tmp_path, Custom, component_overlay=component_overlay)
        _step(trainer)
        assert "never satisfied" in trainer._policy_guard.events[-1].violations[0].message
    finally:
        TERM_CHECKS.pop("always_fails", None)


# -- S007: the declaration contradicted ---------------------------------------------------------
def test_s007_self_stepping_true_with_a_live_loss(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, LiesAboutStepping, component_overlay=component_overlay)
    before = snapshot_parameters(trainer)
    result = _step(trainer)
    assert result.metrics.get("quarantined") == 1.0
    assert _equal(before, snapshot_parameters(trainer))
    assert trainer._policy_guard.events[-1].violations[0].code == "S007"


def test_s007_self_stepping_false_with_a_detached_loss(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, DetachedButLive, component_overlay=component_overlay)
    _step(trainer)
    assert trainer._policy_guard.events[-1].violations[0].code == "S007"


# -- S008: the void node -------------------------------------------------------------------------
def test_s008_a_positive_advantage_on_a_void_node(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, RewardsVoid, component_overlay=component_overlay)
    before = snapshot_parameters(trainer)
    result = _step(trainer, _batch(void=True, signal=2.0))
    assert result.metrics.get("quarantined") == 1.0
    assert _equal(before, snapshot_parameters(trainer))
    assert trainer._policy_guard.events[-1].violations[0].code == "S008"
    # the same policy on a healthy node reports the same number and is fine
    assert _step(trainer, _batch(void=False)).metrics.get("quarantined") is None


def test_s008_stepping_on_a_void_node_with_credit_is_undone(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, StepsOnVoid, component_overlay=component_overlay)
    before, moments = snapshot_parameters(trainer), _moments(trainer, "peer-1")
    result = _step(trainer, _batch(void=True))
    assert result.metrics.get("quarantined") == 1.0
    assert _equal(before, snapshot_parameters(trainer))            # the unauthorised step is gone
    assert _moments(trainer, "peer-1")["state"] == moments["state"]
    assert trainer._policy_guard.restored == 1


def test_the_void_probe_hands_the_policy_a_clamped_read_only_signal(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, Healthy, component_overlay=component_overlay)
    policy = trainer._trainer_policy
    _step(trainer, _batch(void=True, signal=5.0))
    seen = policy.seen[-1]
    assert seen["signal"] <= -1.0                                  # a void node cannot see a reward
    assert seen["void_advantage_ceiling"] == -1.0
    with pytest.raises(TypeError):
        seen["signal"] = 9.0                                        # and cannot rewrite what it saw
    _step(trainer, _batch(void=False, signal=5.0))
    assert policy.seen[-1]["signal"] == 5.0                        # a healthy node is untouched


def test_void_clamp_can_be_switched_off(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, Healthy, component_overlay=component_overlay,
                       void_clamp=False)
    _step(trainer, _batch(void=True, signal=5.0))
    assert trainer._trainer_policy.seen[-1]["signal"] == 5.0


def test_require_void_evidence_refuses_a_silent_policy(settings, tmp_path, component_overlay):
    class Silent(Healthy):
        name = "silent"
        requires_terms = frozenset()

        def train_step(self, node, batch):
            result = super().train_step(node, batch)
            result.breakdown.extras.pop("advantage_mean", None)
            return result

    trainer = _trainer(settings, tmp_path, Silent, component_overlay=component_overlay,
                       require_void_evidence=True)
    _step(trainer, _batch(void=True))
    assert trainer._policy_guard.events[-1].violations[0].code == "S008"
    _step(trainer, _batch(void=False))                              # silence is fine off a void node
    assert len(trainer._policy_guard.events) == 1


# -- S009: the policy raised -------------------------------------------------------------------------
def test_s009_a_policy_that_raises_mid_step_is_rolled_back_and_reraised(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, MutatesWeights, component_overlay=component_overlay)
    before, moments = snapshot_parameters(trainer), _moments(trainer, "peer-1")
    with pytest.raises(RuntimeError, match="boom half way"):         # unchanged, not wrapped
        _step(trainer)
    assert _equal(before, snapshot_parameters(trainer))              # the +1.0 and the update are gone
    assert _moments(trainer, "peer-1")["state"] == moments["state"]  # Adam's moments too
    event = trainer._policy_guard.events[-1]
    assert event.action == "raised" and event.restored
    assert event.violations[0].code == "S009"
    assert trainer.global_step == 0


def test_s009_in_warn_mode_nothing_is_restored(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, MutatesWeights, component_overlay=component_overlay,
                       mode="warn")
    before = snapshot_parameters(trainer)
    with pytest.raises(RuntimeError):
        _step(trainer)
    assert not _equal(before, snapshot_parameters(trainer))          # warn means warn
    assert trainer._policy_guard.events[-1].action == "raised"


# -- S006 / modes ------------------------------------------------------------------------------------
def test_s006_consecutive_quarantines_stop_the_run(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, NaNLoss, component_overlay=component_overlay,
                       max_consecutive=3)
    assert _step(trainer).metrics.get("quarantined") == 1.0
    assert _step(trainer).metrics.get("quarantined") == 1.0
    with pytest.raises(StepGuardViolation) as raised:
        _step(trainer)                                               # quarantine mode still raises
    assert raised.value.code == "S001"
    assert "S006" in {v["code"] for v in raised.value.violations}
    assert "consecutive" in str(raised.value)


def test_s006_a_clean_step_resets_the_count(settings, tmp_path, component_overlay):
    class Flaky(_Base):
        name = "flaky"
        calls = 0

        def train_step(self, node, batch):
            type(self).calls += 1
            _, logits, _ = self._forward(node, batch)
            bad = type(self).calls % 3 != 0
            loss = logits.pow(2).mean() * (float("nan") if bad else 1.0)
            return StepResult(loss=loss, breakdown=LossBreakdown(), stepped=True)

    Flaky.calls = 0
    trainer = _trainer(settings, tmp_path, Flaky, component_overlay=component_overlay,
                       max_consecutive=3)
    for _ in range(9):                      # two bad then one good, three times: never three in a row
        _step(trainer)
    assert trainer._policy_guard.quarantined == 6 and trainer.global_step == 3


def test_enforce_mode_restores_then_raises(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, PoisonsWeights, component_overlay=component_overlay,
                       mode="enforce")
    before = snapshot_parameters(trainer)
    with pytest.raises(StepGuardViolation) as raised:
        _step(trainer)
    assert raised.value.path == "peer-1" and raised.value.code == "S003"
    assert _equal(before, snapshot_parameters(trainer))


def test_warn_mode_records_and_returns_the_result(settings, tmp_path, component_overlay, caplog):
    trainer = _trainer(settings, tmp_path, NaNLoss, component_overlay=component_overlay, mode="warn")
    with caplog.at_level(logging.WARNING, logger="mactf.components.step_guard"):
        result = _step(trainer)
    assert result.metrics.get("quarantined") is None
    assert trainer._policy_guard.events[-1].action == "warned"
    assert "S001" in caplog.text


def test_off_mode_is_a_pass_through(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, NaNLoss, component_overlay=component_overlay, mode="off")
    _step(trainer)
    assert trainer._policy_guard.checked == 0 and trainer._policy_guard.events == []


def test_snapshot_never_cannot_undo_a_parameter_write(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, PoisonsWeights, component_overlay=component_overlay,
                       snapshot="never")
    _step(trainer)
    assert trainer._policy_guard.restored == 0
    assert trainer._policy_guard.quarantined == 1


def test_max_grad_norm_only_ever_lowers_the_optimizers_cap(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, Healthy, component_overlay=component_overlay,
                       max_grad_norm=0.25)
    assert all(o.max_grad_norm == 0.25 for o in trainer.optimizers.values())
    guard = StepGuard(StepGuardConfig(max_grad_norm=5.0))
    optimizer = trainer.optimizers["peer-1"]
    guard.rebind("peer-1", optimizer)
    assert optimizer.max_grad_norm == 0.25                           # 5.0 would have raised it


# -- the legacy forms -------------------------------------------------------------------------------------
def test_a_legacy_callable_policy_still_receives_trainer(settings, tmp_path, caplog):
    tune(settings)
    seen = {}

    def policy(node, batch, *, trainer):
        seen["trainer"] = trainer
        _, peer = trainer.resolve_node(node)
        logits, _ = peer.adapter(batch["states"])
        return StepResult(loss=logits.pow(2).mean(), breakdown=LossBreakdown(), stepped=True)

    trainer = make_trainer(settings, tmp_path, trainer=policy).build()
    assert trainer._trainer_policy is not None and isinstance(trainer._policy, GuardedPolicy)
    before = snapshot_parameters(trainer)
    result = _step(trainer)
    assert seen["trainer"] is trainer
    assert result.stepped and not _equal(before, snapshot_parameters(trainer))


def test_a_trainer_instance_in_code_warns_first(settings, tmp_path, component_overlay):
    """R5: an instance that predates the guard is checked in ``warn`` mode, not refused."""
    tune(settings)
    trainer = make_trainer(settings, tmp_path, trainer=NaNLoss()).build()
    assert trainer._policy_guard.config.mode == "warn"
    assert isinstance(trainer._trainer_policy, NaNLoss)              # the caller's object, unwrapped
    result = _step(trainer)
    assert result.metrics.get("quarantined") is None                 # warned, not quarantined
    assert trainer._policy_guard.events[-1].action == "warned"


def test_a_spec_policy_defaults_to_quarantine(settings, tmp_path, component_overlay):
    tune(settings)
    register_policy("nan_loss", NaNLoss, scope="run")
    configure(settings, TRAINER={"POLICY": "nan_loss"})
    trainer = make_trainer(settings, tmp_path).build()
    assert trainer._policy_guard.config.mode == "quarantine"


def test_guarded_policy_forwards_the_flags_and_members(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, Healthy, component_overlay=component_overlay)
    wrapped = trainer._policy
    assert wrapped.self_stepping is False
    assert wrapped.requires_terms == frozenset({"compliance_floor"})
    assert wrapped.inner is trainer._trainer_policy
    assert wrapped.card_metadata("peer-1")["step_guard"]["mode"] == "quarantine"
    assert wrapped.state_dict()["guard"]["checked"] == 0
    with pytest.raises(AttributeError):
        wrapped.no_such_attribute


def test_guard_state_round_trips(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, NaNLoss, component_overlay=component_overlay)
    _step(trainer)
    saved = trainer._policy_guard.state_dict()
    fresh = StepGuard()
    fresh.load_state_dict(saved)
    assert fresh.state_dict() == saved and saved["quarantined"] == 1


# -- unsafe policies (7.7) ------------------------------------------------------------------------------
class Unsafe(Healthy):
    name = "unsafe_one"
    unsafe = True


def test_an_unsafe_policy_is_refused_at_build_not_mid_run(settings, tmp_path, component_overlay):
    tune(settings)
    register_policy("unsafe_one", Unsafe, scope="run")
    configure(settings, TRAINER={"POLICY": "unsafe_one"})
    trainer = make_trainer(settings, tmp_path)
    with pytest.raises(UnsafePolicyDenied) as raised:
        trainer.build()
    assert raised.value.code == "E009" and "ALLOW_UNSAFE_POLICY" in str(raised.value)


def test_an_unsafe_policy_runs_unguarded_with_the_flag_and_is_stamped(settings, tmp_path,
                                                                      component_overlay, caplog):
    tune(settings)
    register_policy("unsafe_one", Unsafe, scope="run")
    configure(settings, TRAINER={"POLICY": "unsafe_one", "ALLOW_UNSAFE_POLICY": True})
    with caplog.at_level(logging.WARNING):
        trainer = make_trainer(settings, tmp_path).build()
    assert "runs unsafe" in caplog.text
    assert isinstance(trainer._policy, Unsafe) and trainer._policy_guard is None
    trainer.run_epoch(0)
    cards = trainer._epoch_cards[-1].peers.values()
    assert all(card.metadata["policy"]["unsafe"] is True for card in cards)


# -- the configuration ------------------------------------------------------------------------------------
def test_the_config_validates_every_key():
    assert StepGuardConfig().mode == "quarantine"
    for bad in ({"MODE": "loud"}, {"SNAPSHOT": "sometimes"}, {"MAX_CONSECUTIVE": 0},
                {"GRAD_NORM_CEILING": 0}, {"MAX_GRAD_NORM": -1}, {"NOPE": 1}):
        with pytest.raises(ValidationError):
            StepGuardConfig.from_settings(bad)
    config = StepGuardConfig.from_settings({"mode": "enforce", "max_consecutive": 5})
    assert config.mode == "enforce" and config.max_consecutive == 5
    assert StepGuardConfig.from_settings(config.as_dict() and {"MODE": "warn"}).mode == "warn"


def test_every_code_is_documented():
    assert set(CODES) == {f"S00{n}" for n in range(1, 10)}


def test_void_probe_alone():
    probe = VoidProbe()
    healthy = probe.apply({"void": False, "signal": 3.0})
    assert healthy["signal"] == 3.0 and healthy["void_advantage_ceiling"] == float("inf")
    voided = probe.apply({"void": True, "signal": 3.0})
    assert voided["signal"] == -1.0
    assert probe.apply({"void": True, "signal": -4.0})["signal"] == -4.0   # never raised, only lowered


# -- a property: whatever the loss is, a quarantined step leaves the node as it found it ---------
from hypothesis import HealthCheck, given, settings as hyp_settings, strategies as st  # noqa: E402


@hyp_settings(max_examples=12, deadline=None, suppress_health_check=list(HealthCheck))
@given(
    scale=st.sampled_from([float("nan"), float("inf"), -float("inf"), 1.0e30, 1.0e12, 3.0, 0.0]),
    void=st.booleans(),
    floor=st.sampled_from([0.0, -0.5, -3.0]),
)
def test_property_a_judged_step_never_leaves_non_finite_parameters(scale, void, floor, tmp_path_factory):
    import sys

    from mactf import conf

    module = type(sys)("mactf_prop_settings")
    root = tmp_path_factory.mktemp("prop")
    module.BASE_DIR = str(root)
    module.CONFIG_FILES = []
    module.DEVICE = "cpu"
    module.PEERS = []
    sys.modules["mactf_prop_settings"] = module
    conf.reset()
    loaded = conf.load_settings("mactf_prop_settings")
    try:
        tune(loaded)

        class Scaled(Healthy):
            name = "scaled"

            def train_step(self, node, batch):
                _, logits, _ = self._forward(node, batch)
                loss = logits.pow(2).mean() * scale
                return StepResult(
                    loss=loss, breakdown=LossBreakdown(compliance=max(-floor, 0.0),
                                                       total=0.0,
                                                       extras={"advantage_mean": -1.0}),
                    stepped=True)

        from mactf.components import overlay

        with overlay():
            register_policy("scaled", Scaled, scope="run")
            configure(loaded, TRAINER={"POLICY": "scaled", "STEP_GUARD": {"MODE": "quarantine",
                                                                         "MAX_CONSECUTIVE": 99}})
            trainer = make_trainer(loaded, root).build()
            before = snapshot_parameters(trainer)
            result = _step(trainer, _batch(void=void, floor=floor))
            after = snapshot_parameters(trainer)
            assert all(torch.isfinite(t).all() for t in after.values())
            if result.metrics.get("quarantined"):
                assert _equal(before, after)
    finally:
        conf.reset()


def test_the_grad_norm_cap_follows_an_optimizer_swap(settings, tmp_path, component_overlay):
    trainer = _trainer(settings, tmp_path, Healthy, component_overlay=component_overlay,
                       max_grad_norm=0.25)
    trainer.swap_optimizer("OPTIMIZER", {"CLASS": "SGD", "LR": 0.05, "WEIGHT_DECAY": 0.0,
                                         "MAX_GRAD_NORM": 3.0}, mode="reset")
    assert all(o.describe()["class"] == "SGD" for o in trainer.optimizers.values())
    assert all(o.max_grad_norm == 0.25 for o in trainer.optimizers.values())   # lowered, never raised
