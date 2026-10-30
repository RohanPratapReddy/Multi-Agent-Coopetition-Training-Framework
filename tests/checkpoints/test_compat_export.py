"""test compat export (split from test_checkpoint_v2.py)."""

from __future__ import annotations

import os


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


def test_v1_checkpoint_restores_with_missing_sections(tuned, tmp_path):
    from mactf.orchestration.checkpoint import CheckpointManager  # noqa: F401

    first = _trainer(tuned, tmp_path)
    first.fit(epochs=1)
    state = {k: v for k, v in first.peers["peer-1"].module.state_dict().items()}
    raw = {
        "run_id": "run", "episode": 3, "score": 0.5, "created_at": 1.0,
        "peers": {"peer-1": {"adapter": state}, "peer-2": {"adapter": dict(state)}},
        "manifest": {"schema_version": 1, "files": []},
    }
    writer = MACTFCheckpoint(tmp_path / "v1", run_id="run")
    path = writer.path_for(3)
    torch.save(raw, path)
    assert writer.verify(path).ok
    payload = writer.load(path)
    assert payload.identity["schema_version"] == 1
    assert "rng" in payload.missing_sections() and "data" in payload.missing_sections()


def test_export_and_describe(tuned, tmp_path):
    trainer = _trainer(tuned, tmp_path)
    trainer.fit(epochs=1)
    writer = _writer(trainer)
    out = writer.export_parameters(tmp_path / "out" / "w.pt")
    tensors = torch.load(out, map_location="cpu")
    assert any(k.startswith("peer-1.") for k in tensors)
    info = writer.describe()
    assert info["verified"] and info["sections"]["models"]
    with pytest.raises(Exception):
        writer.export_parameters(tmp_path / "x", format="onnx")


def test_memory_format_and_buffers(tmp_path):
    class WithBuffer(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 1)
            self.register_buffer("kept", torch.ones(2))
            self.register_buffer("skipped", torch.ones(2), persistent=False)
            self.conv.weight.requires_grad_(False)

    from mactf.coopetition.peer import Peer  # noqa: F401

    module = WithBuffer()
    state = module.state_dict()
    assert "kept" in state and "skipped" not in state
    assert not module.conv.weight.requires_grad
    assert os.path.isdir(tmp_path)
