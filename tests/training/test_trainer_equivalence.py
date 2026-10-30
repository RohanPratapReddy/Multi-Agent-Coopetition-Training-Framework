"""test trainer equivalence (split from test_trainer_fit.py)."""

from __future__ import annotations

import copy


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
def tuned(settings, tmp_path):
    settings["TRAINING"].update(
        {"EPOCHS": 2, "BATCH_SIZE": 8, "STATE_DIM": 4, "ACTION_DIM": 3, "BOOTSTRAP": False}
    )
    settings["SANDBOX"]["PROVIDER"] = "local"
    return settings


def test_matches_orchestrator_kappa_and_entropy(tuned, tmp_path):
    from mactf.orchestration.orchestrator import Orchestrator, RunConfig

    tuned["SANDBOX"]["ALLOW_FALLBACK"] = True
    torch.manual_seed(1337)
    modules = {"peer-1": Tiny(), "peer-2": Tiny()}
    orchestrator = Orchestrator(tuned, modules=copy.deepcopy(modules))
    summary = orchestrator.run(RunConfig(epochs=1))
    expected = summary["history"][0]

    torch.manual_seed(1337)
    trainer = MACTFTrainer(
        models=modules,
        settings=tuned,
        checkpoint_dir=tmp_path / "eq",
    )
    trainer.fit(epochs=1)
    report = trainer.reports[-1]
    assert report.kappa == pytest.approx(expected["kappa"], abs=1e-6)
    assert report.entropy == pytest.approx(expected["entropy"], abs=1e-6)
    assert set(report.weights) == set(expected["weights"])
    for name, weight in report.weights.items():
        assert weight == pytest.approx(expected["weights"][name], abs=1e-6)
    for name, peer in trainer.peers.items():
        assert torch.allclose(
            peer.export_parameters(),
            orchestrator.peers[name].export_parameters(),
            atol=1e-5,
        )
