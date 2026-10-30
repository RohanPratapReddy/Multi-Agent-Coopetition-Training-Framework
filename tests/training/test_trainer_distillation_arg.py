"""The polymorphic ``distillation=`` argument on ``MACTFTrainer`` (doc 14 sections 19.1, 19.6)."""

from __future__ import annotations

import copy
import json

import pytest

from mactf import signals
from mactf.distill import CoDistillStep, DistillationConfig
from mactf.distill.config import DISTILLATION_DEFAULTS
from mactf.exceptions import ImproperlyConfigured, ValidationError
from train_support import make_trainer, tuned  # noqa: F401

VIEW = {"kind": "fixed_random", "out_dim": 8}


def teacher(name: str = "t1") -> dict:
    return {
        "name": name,
        "kind": "module",
        "factory": "torch.nn.Linear",
        "kwargs": {"in_features": 4, "out_features": 3},
        "view": dict(VIEW),
        "trust": {"trust_init": 1.0},
    }


def block(**overrides) -> dict:
    value = {"TEACHERS": [teacher()], "OUT_DIM": 8, "TAU": 0.5, "VIEWS": {"*": dict(VIEW)}}
    value.update(overrides)
    return value


def enable_in_settings(settings, **overrides) -> None:
    merged = copy.deepcopy(DISTILLATION_DEFAULTS)
    merged.update({"ENABLED": True, **block(**overrides)})
    settings["DISTILLATION"] = merged


# -- resolution at construction -----------------------------------------------------------
def test_default_is_off(tuned, tmp_path):
    trainer = make_trainer(tuned, tmp_path)
    assert trainer.distillation.enabled is False
    trainer.build()
    assert trainer._trainer_policy is None
    assert trainer.spec.distillation == {}


def test_a_mapping_enables_distillation_and_attaches_the_step(tuned, tmp_path):
    trainer = make_trainer(tuned, tmp_path, distillation=block()).build()
    assert trainer.distillation.enabled is True
    assert isinstance(trainer._trainer_policy, CoDistillStep)
    assert trainer._trainer_policy.managed is True
    assert set(trainer._trainer_policy._peers) == set(trainer.peers)


def test_true_defers_the_teachers_to_the_settings_block(tuned, tmp_path):
    enable_in_settings(tuned)
    tuned["DISTILLATION"]["ENABLED"] = False
    trainer = make_trainer(tuned, tmp_path, distillation=True).build()
    assert isinstance(trainer._trainer_policy, CoDistillStep)


def test_true_without_any_teacher_fails_at_construction(tuned, tmp_path):
    with pytest.raises(ImproperlyConfigured, match="TEACHERS is empty"):
        make_trainer(tuned, tmp_path, distillation=True)


def test_false_beats_an_enabled_settings_block(tuned, tmp_path):
    enable_in_settings(tuned)
    trainer = make_trainer(tuned, tmp_path, distillation=False).build()
    assert trainer.distillation.enabled is False
    assert not isinstance(trainer._trainer_policy, CoDistillStep)


def test_none_follows_an_enabled_settings_block(tuned, tmp_path):
    enable_in_settings(tuned)
    trainer = make_trainer(tuned, tmp_path).build()
    assert trainer.distillation.enabled is True
    assert isinstance(trainer._trainer_policy, CoDistillStep)


def test_a_config_instance_is_accepted(tuned, tmp_path):
    config = DistillationConfig.from_spec(block(), settings=tuned)
    trainer = make_trainer(tuned, tmp_path, distillation=config)
    assert trainer.distillation is config


def test_unknown_key_fails_at_construction_naming_the_key(tuned, tmp_path):
    with pytest.raises(ValidationError, match="TAUU"):
        make_trainer(tuned, tmp_path, distillation=block(TAUU=0.1))


def test_a_bad_value_fails_at_construction_not_at_epoch_one(tuned, tmp_path):
    with pytest.raises(ImproperlyConfigured, match="SIGMA_MODE"):
        make_trainer(tuned, tmp_path, distillation=block(SIGMA_MODE="bogus"))


# -- exclusivity ------------------------------------------------------------------------
def test_trainer_and_distillation_are_mutually_exclusive(tuned, tmp_path):
    class Policy:
        def setup(self, node): ...

        def train_step(self, node, batch): ...

        def on_epoch_end(self, tree, result): ...

    with pytest.raises(ImproperlyConfigured, match="mutually exclusive"):
        make_trainer(tuned, tmp_path, trainer=Policy(), distillation=block())


def test_trainer_with_distillation_false_is_allowed(tuned, tmp_path):
    enable_in_settings(tuned)

    class Policy:
        def setup(self, node): ...

        def train_step(self, node, batch): ...

        def on_epoch_end(self, tree, result): ...

    trainer = make_trainer(tuned, tmp_path, trainer=Policy(), distillation=False).build()
    assert isinstance(trainer._trainer_policy, Policy)


def test_steps_per_epoch_other_than_one_is_refused_when_on(tuned, tmp_path):
    with pytest.raises(ImproperlyConfigured, match="steps_per_epoch must be 1"):
        make_trainer(tuned, tmp_path, distillation=block(), steps_per_epoch=3)
    make_trainer(tuned, tmp_path, distillation=block(), steps_per_epoch=1)
    make_trainer(tuned, tmp_path, distillation=False, steps_per_epoch=3)


# -- the run itself ----------------------------------------------------------------------
def test_fit_records_distillation_on_the_scorecards_and_the_spec(tuned, tmp_path):
    trainer = make_trainer(tuned, tmp_path, distillation=block())
    trainer.fit(epochs=2)
    record = trainer.spec.distillation
    assert record["ENABLED"] is True
    json.dumps(record)
    for card in trainer.scorecards[-2:]:
        for peer_card in card.peers.values():
            meta = peer_card.metadata["distillation"]
            assert meta["enabled"] is True
            assert meta["teachers"] == ["t1"]
            assert set(meta) >= {"sigma", "per_teacher", "cd_weighted", "peer_margin"}


def test_fit_emits_one_distillation_step_per_epoch(tuned, tmp_path):
    events = []

    def receiver(sender=None, **payload):
        events.append(payload)

    signals.distillation_step.connect(receiver, weak=False)
    try:
        make_trainer(tuned, tmp_path, distillation=block()).fit(epochs=3)
    finally:
        signals.distillation_step.disconnect(receiver)
    assert len(events) == 3


def test_every_peer_takes_exactly_one_train_step_per_epoch(tuned, tmp_path):
    trainer = make_trainer(tuned, tmp_path, distillation=block())
    trainer.fit(epochs=2)
    policy = trainer._trainer_policy
    assert policy.state_dict()["step"] == 2
    assert not policy._buffered


def test_the_pending_rendezvous_loss_is_not_recorded_as_a_peer_loss(tuned, tmp_path):
    trainer = make_trainer(tuned, tmp_path, distillation=block())
    trainer.fit(epochs=3)
    # Two peers call ``train_step`` per epoch, but only the completing call carries the
    # cohort loss. A pending placeholder (loss 0.0) must not enter the history or the mean.
    assert len(trainer.losses) == 3
    assert trainer.global_step == 3
    assert all(value != 0.0 for value in trainer.losses)
    assert all(item.mean_loss != 0.0 for item in trainer.history)
