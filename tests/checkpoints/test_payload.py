"""test payload (split from test_checkpoint_v2.py)."""

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


def test_payload_roundtrip_hash_and_manifest(tuned, tmp_path):
    trainer = _trainer(tuned, tmp_path)
    trainer.fit(epochs=2)
    writer = _writer(trainer)
    path = writer.latest().path
    payload = writer.load(path)
    assert payload.identity["run_id"] == "run"
    assert set(payload.models) == {"peer-1", "peer-2"}
    assert payload.content_hash() == payload.manifest["content_hash"]
    assert set(payload.manifest["sections"]) == set(CheckpointPayload.SECTIONS)
    again = CheckpointPayload.from_dict(payload.as_dict())
    assert again.content_hash() == payload.content_hash()
    assert payload.manifest["files"][0]["bytes"] == path.stat().st_size
    with pytest.raises(Exception, match="does not accept"):
        CheckpointPayload.from_dict({**payload.as_dict(), "bogus": 1})


def test_constructor_validation(tmp_path):
    with pytest.raises(ImproperlyConfigured):
        MACTFCheckpoint(tmp_path, run_id="")
    with pytest.raises(ImproperlyConfigured):
        MACTFCheckpoint(tmp_path, run_id="r", checksum="md5")
    with pytest.raises(ImproperlyConfigured):
        MACTFCheckpoint(tmp_path, run_id="r", keep=-1)
