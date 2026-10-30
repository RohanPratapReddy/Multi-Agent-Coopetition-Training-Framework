"""test verify (split from test_checkpoint_v2.py)."""

from __future__ import annotations

import pytest


import torch


import torch.nn as nn


from mactf.exceptions import ComplianceError, ImproperlyConfigured


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


def test_verify_detects_truncation_tamper_and_schema(tuned, tmp_path):
    trainer = _trainer(tuned, tmp_path)
    trainer.fit(epochs=2)
    writer = _writer(trainer)
    good = writer.latest().path
    assert writer.verify(good).ok

    truncated = tmp_path / "run-ep0098.pt"
    truncated.write_bytes(good.read_bytes()[: good.stat().st_size // 2])
    assert not writer.verify(truncated).ok

    empty = tmp_path / "run-ep0097.pt"
    empty.write_bytes(b"")
    assert "empty" in writer.verify(empty).problems[0]

    raw = torch.load(good, map_location="cpu", weights_only=False)
    tampered = dict(raw)
    state = {k: v.clone() for k, v in raw["models"]["peer-1"].items()}
    first = next(iter(state))
    state[first] = state[first] + 1.0
    tampered["models"] = {**raw["models"], "peer-1": state}
    tampered_path = tmp_path / "run-ep0096.pt"
    torch.save(tampered, tampered_path)
    result = writer.verify(tampered_path)
    assert not result.ok and any("section 'models'" in p for p in result.problems)

    bad_size = dict(raw)
    manifest = dict(raw["manifest"])
    manifest["files"] = [{**manifest["files"][0], "name": "run-ep0095.pt", "bytes": 1}]
    bad_size["manifest"] = manifest
    torch.save(bad_size, tmp_path / "run-ep0095.pt")
    assert any("bytes" in p for p in writer.verify(tmp_path / "run-ep0095.pt").problems)

    future = dict(raw)
    future["manifest"] = {**raw["manifest"], "schema_version": 99, "files": []}
    torch.save(future, tmp_path / "run-ep0094.pt")
    assert any("unsupported schema_version" in p for p in writer.verify(tmp_path / "run-ep0094.pt").problems)

    assert not writer.verify(tmp_path / "missing.pt").ok
    with pytest.raises(ComplianceError):
        writer.load(truncated)
