"""test latest valid (split from test_checkpoint_v2.py)."""

from __future__ import annotations

import pytest


import torch


import torch.nn as nn


from mactf.exceptions import ComplianceError, ImproperlyConfigured


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


def test_latest_valid_skips_corrupt_newest_and_records_it(tuned, tmp_path):
    trainer = _trainer(tuned, tmp_path)
    trainer.fit(epochs=3)
    writer = _writer(trainer)
    newest = writer.list()[-1]
    newest.path.write_bytes(b"garbage")
    assert writer.latest(valid_only=True).episode == newest.episode - 1 or \
        writer.latest(valid_only=True).path != newest.path
    fresh = _trainer(tuned, tmp_path).build()
    report = fresh.checkpoint.resume_from_last_checkpoint(trainer=fresh)
    assert report.skipped_checkpoints and newest.path.name == report.skipped_checkpoints[0][0]
    assert report.checkpoint != newest.path


def test_no_valid_checkpoint(tmp_path):
    writer = MACTFCheckpoint(tmp_path, run_id="none")
    with pytest.raises(ComplianceError):
        writer.resume_from_last_checkpoint(strict=True)
    report = writer.resume_from_last_checkpoint(strict=False)
    assert report.notes == ("no valid checkpoint found",)
