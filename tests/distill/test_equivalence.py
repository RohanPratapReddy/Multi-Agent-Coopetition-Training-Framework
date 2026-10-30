"""Off-equivalence: with distillation off the trainer is document 08, to 1e-12 (doc 14 section 15)."""

from __future__ import annotations

import copy

import pytest
import torch

from distill_support import Tiny, distillation_block, teacher_spec, tune
from mactf.distill import CoDistillStep
from mactf.training import MACTFTrainer


def fit(settings, tmp_path, name: str, *, seed: int = 1337, epochs: int = 2, **kwargs):
    torch.manual_seed(seed)
    trainer = MACTFTrainer(
        models={"peer-1": Tiny(), "peer-2": Tiny()},
        settings=settings,
        checkpoint_dir=tmp_path / name,
        **kwargs,
    )
    trainer.fit(epochs=epochs)
    return trainer


def parameters(trainer):
    return {name: peer.export_parameters().detach().clone() for name, peer in trainer.peers.items()}


def assert_same_run(left, right):
    for name, value in parameters(left).items():
        other = parameters(right)[name]
        assert torch.allclose(value, other, atol=1e-12, rtol=0.0), name
    assert len(left.reports) == len(right.reports)
    for first, second in zip(left.reports, right.reports, strict=True):
        assert first.kappa == pytest.approx(second.kappa, abs=1e-12)
        assert first.entropy == pytest.approx(second.entropy, abs=1e-12)
        for name, weight in first.weights.items():
            assert second.weights[name] == pytest.approx(weight, abs=1e-12)


def test_default_run_has_no_distillation_policy(settings, tmp_path):
    tune(settings)
    trainer = fit(settings, tmp_path, "plain")
    assert not isinstance(trainer._trainer_policy, CoDistillStep)
    assert trainer.distillation.enabled is False
    assert trainer.capabilities().supports_distillation is False
    assert trainer.spec.distillation == {}
    for card in trainer.scorecards:
        for peer_card in card.peers.values():
            assert "distillation" not in peer_card.metadata


def test_false_matches_the_default_run_to_1e_12(settings, tmp_path):
    tune(settings)
    plain = fit(settings, tmp_path, "plain")
    off = fit(settings, tmp_path, "off", distillation=False)
    assert_same_run(plain, off)


def test_none_matches_the_default_run_to_1e_12(settings, tmp_path):
    tune(settings)
    plain = fit(settings, tmp_path, "plain")
    deferred = fit(settings, tmp_path, "deferred", distillation=None)
    assert_same_run(plain, deferred)


def test_false_beats_a_project_level_enabled_block(settings, tmp_path):
    """``False`` is unconditional: a settings block that turns distillation on loses."""
    tune(settings)
    plain = fit(settings, tmp_path, "plain")
    block = copy.deepcopy(distillation_block())
    settings["DISTILLATION"].update(block)
    assert settings["DISTILLATION"]["ENABLED"] is True
    forced_off = fit(settings, tmp_path, "forced_off", distillation=False)
    assert not isinstance(forced_off._trainer_policy, CoDistillStep)
    assert_same_run(plain, forced_off)


def test_a_disabled_block_with_teachers_changes_nothing(settings, tmp_path):
    tune(settings)
    plain = fit(settings, tmp_path, "plain")
    disabled = fit(
        settings,
        tmp_path,
        "disabled",
        distillation={**distillation_block(), "ENABLED": False},
    )
    assert_same_run(plain, disabled)


def test_enabled_distillation_actually_changes_the_run(settings, tmp_path):
    """The converse gate: equivalence above is meaningful only if "on" differs."""
    tune(settings)
    plain = fit(settings, tmp_path, "plain")
    on = fit(settings, tmp_path, "on", distillation=distillation_block())
    assert isinstance(on._trainer_policy, CoDistillStep)
    differences = [
        float((parameters(plain)[name] - parameters(on)[name]).abs().max()) for name in plain.peers
    ]
    assert max(differences) > 1e-6


def test_enabled_distillation_is_deterministic(settings, tmp_path):
    """I4: the same seed gives bit-identical parameters."""
    tune(settings)
    first = fit(settings, tmp_path, "a", distillation=distillation_block())
    second = fit(settings, tmp_path, "b", distillation=distillation_block())
    for name, value in parameters(first).items():
        assert torch.equal(value, parameters(second)[name]), name


def test_the_teacher_is_untouched_by_a_full_run(settings, tmp_path):
    """I2: the teacher's parameters and gradients are inert after training."""
    tune(settings)
    trainer = fit(
        settings, tmp_path, "teacher", distillation=distillation_block(teacher_spec("t1"))
    )
    teacher = trainer._trainer_policy.teachers["t1"]
    assert all(p.grad is None for p in teacher.module.parameters())
    assert not any(p.requires_grad for p in teacher.module.parameters())
    assert teacher.digest == trainer._trainer_policy.state_dict()["teacher_digests"]["t1"]
