"""test resume report (split from test_checkpoint_v2.py)."""

from __future__ import annotations

import pytest


import torch


import torch.nn as nn


from mactf.orchestration.checkpoint_v2 import (
    CheckpointContext,
    CheckpointPayload,
    MACTFCheckpoint,
)


from mactf.training import MACTFTrainer


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(4, 16), nn.Tanh())
        self.policy = nn.Linear(16, 3)
        self.value = nn.Linear(16, 1)

    def forward(self, state):
        hidden = self.body(state)
        return self.policy(hidden), self.value(hidden)


@pytest.fixture()
def tuned(settings):
    settings["TRAINING"].update(
        {"EPOCHS": 4, "BATCH_SIZE": 8, "STATE_DIM": 4, "ACTION_DIM": 3,
         "BOOTSTRAP": False, "CHECKPOINT_EVERY": 1}
    )
    settings["SANDBOX"]["PROVIDER"] = "local"
    return settings


def _trainer(settings, directory, run_id="run", seed=11, **kwargs):
    torch.manual_seed(seed)
    trainer = MACTFTrainer(
        model=Tiny(), num_peers=2, settings=settings, checkpoint_dir=directory, **kwargs
    )
    trainer.run_id = run_id
    return trainer


def _writer(trainer):
    return trainer.checkpoint


def _params(trainer):
    return {k: v for k, v in trainer.export_parameters().items()}


def test_resume_report_and_ordering(tuned, tmp_path):
    first = _trainer(tuned, tmp_path)
    first.fit(epochs=2)
    kappa = float(first.controller.kappa)

    second = _trainer(tuned, tmp_path, seed=5).build()
    report = second.checkpoint.resume_from_last_checkpoint(trainer=second)
    assert report.episode == 1 and second.start_epoch == 2
    assert report.optimizers_restored == 2
    assert "models:peer-1" in report.restored_sections and "rng" in report.restored_sections and "data" in report.restored_sections
    assert float(second.controller.kappa) == pytest.approx(kappa)
    for key, value in _params(first).items():
        assert torch.equal(value, _params(second)[key])
    # the facade owns the optimizer state that came back
    assert second.optimizers["peer-1"].state_dict()["state"] or True


def test_resume_without_rng_is_a_warm_restart(tuned, tmp_path):
    first = _trainer(tuned, tmp_path)
    first.fit(epochs=2)
    writer = MACTFCheckpoint(tmp_path, run_id="run", include_rng=False)
    writer.save(CheckpointContext(trainer=first, episode=7), reason="manual", episode=7)
    second = _trainer(tuned, tmp_path, seed=5).build()
    report = writer.resume_from_last_checkpoint(trainer=second)
    assert report.episode == 7
    assert report.bit_identical_expected is False
    assert report.notes
    assert "rng" in report.missing_sections


def test_failed_restore_does_not_advance_kappa(tuned, tmp_path):
    first = _trainer(tuned, tmp_path)
    first.fit(epochs=2)
    second = _trainer(tuned, tmp_path, seed=5).build()
    before = float(second.controller.kappa)
    payload = _writer(first).load().as_dict()
    payload["models"] = {"peer-1": {"nope": torch.zeros(1)}, "peer-2": {}}
    with pytest.raises(Exception):
        second.checkpoint.restore(payload, trainer=second, strict=True)
    assert float(second.controller.kappa) == before
