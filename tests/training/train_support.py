"""Shared model, settings fixture and trainer factory for the training tests."""

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


class Quadratic(nn.Module):
    """A convex least-squares problem with a known minimum."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1, bias=True)


@pytest.fixture()
def tuned(settings, tmp_path):
    settings["TRAINING"].update(
        {"EPOCHS": 2, "BATCH_SIZE": 8, "STATE_DIM": 4, "ACTION_DIM": 3, "BOOTSTRAP": False}
    )
    settings["SANDBOX"]["PROVIDER"] = "local"
    return settings


def make_trainer(settings, tmp_path, *, seed: int = 0, peers: int = 2, **kwargs):
    torch.manual_seed(seed)
    return MACTFTrainer(
        model=Tiny(),
        num_peers=peers,
        settings=settings,
        checkpoint_dir=tmp_path / "ckpt",
        **kwargs,
    )


def least_squares_problem():
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(64, 3, generator=generator, dtype=torch.float64)
    truth = torch.tensor([[1.5], [-2.0], [0.5]], dtype=torch.float64)
    targets = features @ truth + 0.25
    return features, targets, truth
