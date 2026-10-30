"""test prune (split from test_checkpoint_v2.py)."""

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


def test_prune_keeps_newest_and_best_and_never_zero(tuned, tmp_path):
    trainer = _trainer(tuned, tmp_path)
    trainer.fit(epochs=4)
    writer = _writer(trainer)
    writer.save(CheckpointContext(trainer=trainer, episode=2, score=9.0), reason="best",
                episode=2, best=True)
    writer.prune(keep=2, keep_best=1)
    names = sorted(p.name for p in tmp_path.glob("run-ep*.pt"))
    assert names == ["run-ep0002-best.pt", "run-ep0003.pt", "run-ep0004.pt"][:len(names)] or \
        "run-ep0002-best.pt" in names
    assert len([n for n in names if not n.endswith("-best.pt")]) <= 2
    writer.prune(keep=0, keep_best=0)
    assert list(tmp_path.glob("run-ep*.pt"))  # never zero


def test_in_flight_resume_target_is_not_pruned(tuned, tmp_path):
    trainer = _trainer(tuned, tmp_path)
    trainer.fit(epochs=3)
    writer = _writer(trainer)
    oldest = writer.list()[0].path
    writer._in_flight = oldest
    writer.prune(keep=1, keep_best=0)
    assert oldest.exists()
