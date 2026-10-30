"""A three-level trainer fixture shared by the hierarchy training tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mactf.coopetition.tree import HierarchySpec, LevelSpec
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
        {"EPOCHS": 2, "BATCH_SIZE": 8, "STATE_DIM": 4, "ACTION_DIM": 3, "BOOTSTRAP": False}
    )
    settings["SANDBOX"]["PROVIDER"] = "local"
    return settings


def three_level_spec(roots: int = 2, fanout: int = 2, **kwargs) -> HierarchySpec:
    return HierarchySpec(
        levels=(
            LevelSpec("sub", fanout, model=Tiny()),
            LevelSpec("mini", fanout, model=Tiny()),
        ),
        roots=roots,
        root_models={f"peer-{index + 1}": Tiny() for index in range(roots)},
        **kwargs,
    )


def three_level_trainer(settings, tmp_path, *, seed: int = 0, spec=None, **kwargs):
    torch.manual_seed(seed)
    return MACTFTrainer(
        hierarchy=spec or three_level_spec(),
        settings=settings,
        checkpoint_dir=tmp_path / "ckpt",
        **kwargs,
    )
