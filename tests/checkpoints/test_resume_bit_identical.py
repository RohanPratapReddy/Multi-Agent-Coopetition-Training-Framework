"""test resume bit identical (split from test_checkpoint_v2.py)."""

from __future__ import annotations

import pytest


import torch


import torch.nn as nn


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


def _params(trainer):
    return {k: v for k, v in trainer.export_parameters().items()}


def test_resume_is_bit_identical_when_deterministic(tuned, tmp_path):
    tuned["TRAINING"]["DETERMINISTIC"] = True
    reference = _trainer(tuned, tmp_path / "a")
    reference.fit(epochs=4)
    expected = _params(reference)

    first = _trainer(tuned, tmp_path / "b")
    first.fit(epochs=2)

    second = _trainer(tuned, tmp_path / "b", seed=999)  # different init proves the restore
    result = second.fit(epochs=4, resume=True)
    assert result.epochs == 2
    resumed = _params(second)
    assert expected.keys() == resumed.keys()
    for key in expected:
        assert torch.equal(expected[key], resumed[key]), key
