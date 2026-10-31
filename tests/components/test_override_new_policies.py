"""The four new trainer policies, and the guards they get (document 17 sections 7 and 19.6).

New names are not "custom" by courtesy: a spec-reached policy is wrapped (quarantine), an
in-code one is wrapped (warn), a promised term must leave evidence (S004), a declared
``self_stepping`` must be true (S007), distillation has exactly one owner, unsafe needs consent.
"""

from __future__ import annotations

import logging

import pytest
import torch

import custom_kinds as ck
from mactf.components import HUB
from mactf.components.guards.step import GuardedPolicy
from mactf.components.kinds import KIND_SPECS
from mactf.exceptions import ImproperlyConfigured, UnsafePolicyDenied
from override_support import (
    distillation_block, moments, override_scope, put, same_tensors, settings_for, step, step_batch,
)
from p2_support import configure, make_trainer, snapshot_parameters

POLICIES = [ck.AnchoredCurriculumPolicy, ck.DistillingPolicy, ck.QuietPolicy,
            ck.SelfSteppingLiar, ck.UnsafePolicy, ck.AccumulatingPolicy, ck.VoidAwarePolicy]


def _build(settings, tmp_path, cls, *, mode=None, extra=None, **maker):
    guard = {"STEP_GUARD": {"MODE": mode}} if mode else {}
    settings_for(settings, TRAINER={"POLICY": cls.name, **guard, **(extra or {})})
    put("trainer_policy", cls.name, cls)
    return make_trainer(settings, tmp_path, **maker).build()


@pytest.fixture(autouse=True)
def _scope(caplog):
    caplog.set_level(logging.WARNING, logger="mactf.components")
    with override_scope():
        yield


# -- contract and wrapping ---------------------------------------------------------------------
@pytest.mark.parametrize("cls", POLICIES, ids=lambda c: c.name)
def test_every_new_policy_satisfies_the_kind_contract(cls):
    assert KIND_SPECS["trainer_policy"].contract(cls) == []


def test_a_class_without_train_step_fails_the_contract():
    class Hollow:
        name = "hollow"

    problems = KIND_SPECS["trainer_policy"].contract(Hollow)
    assert problems and any("train_step" in p for p in problems)


@pytest.mark.parametrize("cls", [ck.AnchoredCurriculumPolicy, ck.QuietPolicy, ck.AccumulatingPolicy,
                                 ck.VoidAwarePolicy], ids=lambda c: c.name)
def test_a_spec_reached_new_policy_is_guarded_in_quarantine(settings, tmp_path, cls):
    trainer = _build(settings, tmp_path, cls)
    assert isinstance(trainer._policy, GuardedPolicy) and isinstance(trainer._trainer_policy, cls)
    assert trainer._policy_guard.config.mode == "quarantine"
    assert trainer.slots.get("TRAINER.POLICY").component.origin.value != "builtin"


def test_an_in_code_new_policy_is_guarded_in_warn_mode(settings, tmp_path):
    settings_for(settings)
    trainer = make_trainer(settings, tmp_path, trainer=ck.AnchoredCurriculumPolicy()).build()
    assert isinstance(trainer._policy, GuardedPolicy) and trainer._policy_guard.config.mode == "warn"


def test_the_guard_mode_is_chosen_by_the_author_not_assumed(settings, tmp_path):
    trainer = _build(settings, tmp_path, ck.AnchoredCurriculumPolicy, mode="enforce")
    assert trainer._policy_guard.config.mode == "enforce"


# -- a healthy new policy is not disturbed ---------------------------------------------------------
def test_a_healthy_new_policy_takes_a_real_step(settings, tmp_path):
    trainer = _build(settings, tmp_path, ck.AnchoredCurriculumPolicy)
    before = snapshot_parameters(trainer)
    result = step(trainer)
    assert result.stepped and result.metrics.get("quarantined") is None
    assert not same_tensors(before, snapshot_parameters(trainer))
    assert trainer._policy_guard.checked == 1 and trainer._policy_guard.quarantined == 0


def test_a_new_policy_runs_whole_epochs_and_keeps_its_own_state(settings, tmp_path):
    trainer = _build(settings, tmp_path, ck.AnchoredCurriculumPolicy)
    for epoch in range(2):
        trainer.run_epoch(epoch)
    assert trainer._trainer_policy.epochs_seen == 2
    assert trainer._policy_guard.quarantined == 0
    assert all(torch.isfinite(p).all() for tensor in snapshot_parameters(trainer).values() for p in [tensor])


@pytest.mark.parametrize("cls", [ck.AccumulatingPolicy, ck.VoidAwarePolicy], ids=lambda c: c.name)
def test_the_default_step_subclasses_run_clean(settings, tmp_path, cls):
    trainer = _build(settings, tmp_path, cls)
    trainer.run_epoch(0)
    assert trainer._policy_guard.quarantined == 0


# -- S004: a promised term must leave evidence ------------------------------------------------------
def test_s004_a_policy_that_promised_compliance_and_left_none_is_quarantined(settings, tmp_path):
    trainer = _build(settings, tmp_path, ck.QuietPolicy)
    before, held = snapshot_parameters(trainer), moments(trainer)
    result = step(trainer, step_batch(floor=-0.5))
    assert result.metrics.get("quarantined") == 1.0
    assert same_tensors(before, snapshot_parameters(trainer)) and moments(trainer) == held
    violation = trainer._policy_guard.events[-1].violations[0]
    assert violation.code == "S004" and "compliance_floor" in violation.message


def test_s004_the_same_quiet_policy_is_fine_when_nothing_is_breached(settings, tmp_path):
    trainer = _build(settings, tmp_path, ck.QuietPolicy)
    assert step(trainer, step_batch(floor=0.0)).metrics.get("quarantined") is None


def test_s004_the_curriculum_policy_that_does_evidence_the_term_passes_a_breach(settings, tmp_path):
    trainer = _build(settings, tmp_path, ck.AnchoredCurriculumPolicy)
    assert step(trainer, step_batch(floor=-0.5)).metrics.get("quarantined") is None


# -- S007: the declaration contradicted -------------------------------------------------------------
def test_s007_a_policy_that_says_it_steps_but_returns_a_live_loss(settings, tmp_path):
    trainer = _build(settings, tmp_path, ck.SelfSteppingLiar)
    before = snapshot_parameters(trainer)
    result = step(trainer)
    assert result.metrics.get("quarantined") == 1.0
    assert same_tensors(before, snapshot_parameters(trainer))
    assert trainer._policy_guard.events[-1].violations[0].code == "S007"


# -- distillation has one owner ------------------------------------------------------------------------
def test_distillation_with_a_policy_that_does_not_compose_it_is_refused(settings, tmp_path):
    settings_for(settings, TRAINER={"POLICY": ck.AnchoredCurriculumPolicy.name})
    put("trainer_policy", ck.AnchoredCurriculumPolicy.name, ck.AnchoredCurriculumPolicy)
    trainer = make_trainer(settings, tmp_path, distillation=distillation_block())
    with pytest.raises(ImproperlyConfigured, match="composes_distillation"):
        trainer.build()


def test_distillation_with_a_policy_that_declares_it_is_accepted(settings, tmp_path):
    trainer = _build(settings, tmp_path, ck.DistillingPolicy, distillation=distillation_block())
    assert trainer.distill is not None
    assert trainer._trainer_policy.host is trainer


# -- state ---------------------------------------------------------------------------------------------
def test_the_policy_state_round_trips_through_its_own_methods():
    saved = ck.AnchoredCurriculumPolicy()
    saved.epochs_seen = 4
    fresh = ck.AnchoredCurriculumPolicy()
    fresh.load_state_dict(saved.state_dict())
    assert fresh.epochs_seen == 4 and fresh.describe()["epochs_seen"] == 4


def test_the_policy_state_travels_in_a_checkpoint(settings, tmp_path):
    first = _build(settings, tmp_path / "a", ck.AnchoredCurriculumPolicy)
    first.run_epoch(0)
    first.run_epoch(1)
    payload = first.state_dict()
    fresh = make_trainer(settings, tmp_path / "b")
    fresh.run_id = first.run_id
    fresh.build()
    assert fresh._trainer_policy.epochs_seen == 0
    fresh.load_state_dict(payload)
    assert fresh._trainer_policy.epochs_seen == first._trainer_policy.epochs_seen == 2


# -- unsafe needs consent --------------------------------------------------------------------------------
def test_an_unsafe_policy_is_refused_without_the_flag(settings, tmp_path):
    settings_for(settings, TRAINER={"POLICY": ck.UnsafePolicy.name})
    put("trainer_policy", ck.UnsafePolicy.name, ck.UnsafePolicy)
    with pytest.raises(UnsafePolicyDenied) as raised:
        make_trainer(settings, tmp_path).build()
    assert raised.value.code == "E009" and "ALLOW_UNSAFE_POLICY" in str(raised.value)


def test_an_unsafe_policy_with_the_flag_runs_unguarded_and_stamped(settings, tmp_path, caplog):
    trainer = _build(settings, tmp_path, ck.UnsafePolicy, extra={"ALLOW_UNSAFE_POLICY": True})
    assert isinstance(trainer._policy, ck.UnsafePolicy) and trainer._policy_guard is None
    assert "runs unsafe" in caplog.text
    trainer.run_epoch(0)
    assert all(c.metadata["policy"]["unsafe"] is True for c in trainer._epoch_cards[-1].peers.values())


def test_a_new_name_does_not_warn_w001_and_leaves_the_registry_clean(settings, tmp_path, caplog):
    _build(settings, tmp_path, ck.AnchoredCurriculumPolicy)
    assert not [r for r in caplog.records if r.getMessage().startswith("W001")]
    assert HUB.registry("trainer_policy").find("anchored_curriculum") is not None   # in the overlay
