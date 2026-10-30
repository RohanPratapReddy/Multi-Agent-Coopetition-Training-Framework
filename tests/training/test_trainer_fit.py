"""Loop order, hooks, callbacks, and the Orchestrator equivalence gate."""

from __future__ import annotations

import copy


import math


import pytest


import torch


import torch.nn as nn


from mactf.exceptions import ContractError, ImproperlyConfigured


from mactf.training import MACTFTrainer


from mactf.training.callbacks import Callback


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


def _trainer(settings, tmp_path, **kwargs):
    torch.manual_seed(0)
    return MACTFTrainer(
        model=Tiny(),
        num_peers=2,
        settings=settings,
        checkpoint_dir=tmp_path / "ckpt",
        **kwargs,
    )


def test_fit_runs_the_documented_order(tuned, tmp_path):
    order: list[str] = []
    hooks = {
        "on_epoch_start": lambda trainer, epoch: order.append("start"),
        "on_score": lambda trainer, card: order.append("score"),
        "on_step": lambda trainer, node, result: order.append("step"),
        "on_aggregate": lambda trainer, report: order.append("aggregate"),
        "on_epoch_end": lambda trainer, result: order.append("end"),
    }
    trainer = _trainer(tuned, tmp_path, hooks=hooks)
    result = trainer.fit(epochs=1)
    assert result.status == "completed"
    assert order == ["start", "score", "step", "step", "aggregate", "end"]
    assert math.isfinite(result.mean_loss)


def test_fit_aggregates_and_reports_weights(tuned, tmp_path):
    trainer = _trainer(tuned, tmp_path)
    result = trainer.fit(epochs=2)
    assert result.epochs == 2
    report = trainer.reports[-1]
    assert set(report.weights) == {"peer-1", "peer-2"}
    assert sum(report.weights.values()) == pytest.approx(1.0, abs=1e-5)
    assert report.signal_source == "final_score"
    # aggregation wrote the same blend into both peers
    exported = trainer.export_parameters()
    for key, tensor in exported.items():
        if key.startswith("peer-1/"):
            assert torch.allclose(tensor, exported[key.replace("peer-1/", "peer-2/")])


def test_parameter_identity_survives_build(tuned, tmp_path):
    source = Tiny()
    trainer = MACTFTrainer(model=source, num_peers=2, settings=tuned,
                           checkpoint_dir=tmp_path / "c").build()
    peer = trainer.peers["peer-1"]
    assert peer.adapter.module is peer.module
    before = id(peer.module)
    trainer.fit(epochs=1)
    assert id(peer.module) == before
    assert peer.module is not source


def test_callbacks_add_remove_and_early_stop(tuned, tmp_path):
    class Count(Callback):
        name = "count"

        def __init__(self) -> None:
            self.epochs = 0

        def on_epoch_end(self, trainer, result):
            self.epochs += 1
            trainer.request_stop("enough")

    counter = Count()
    trainer = _trainer(tuned, tmp_path, callbacks=[counter])
    result = trainer.fit(epochs=5)
    assert counter.epochs == 1
    assert result.status == "stopped" and result.reason == "enough"
    assert trainer.remove_callback("count") is True
    assert trainer.remove_callback("count") is False


def test_unknown_hook_is_rejected(tuned, tmp_path):
    with pytest.raises(ImproperlyConfigured, match="unknown hook"):
        _trainer(tuned, tmp_path, hooks={"on_nothing": lambda *a: None})


def test_normalize_episode_is_pure(tuned, tmp_path):
    trainer = _trainer(tuned, tmp_path)
    trainer.fit(epochs=1)
    card = trainer.scorecards[-1]
    before = copy.deepcopy(card.to_dict())
    report = trainer.normalize_episode(card)
    assert report.preserves_order
    assert card.to_dict() == before
    stripped = copy.deepcopy(card)
    stripped.points = None
    for peer_card in stripped.peers.values():
        peer_card.points = None
    with pytest.raises(ContractError, match="points"):
        trainer.normalize_episode(stripped)


def test_infer_and_evaluate(tuned, tmp_path):
    trainer = _trainer(tuned, tmp_path)
    trainer.fit(epochs=1)
    metrics = trainer.evaluate()
    assert "mean_accuracy" in metrics
    answer = trainer.infer(torch.randn(5, 4))
    assert set(answer.root_outputs) == {"peer-1", "peer-2"}


def test_import_export_roundtrip(tuned, tmp_path):
    trainer = _trainer(tuned, tmp_path).build()
    payload = trainer.export_parameters()
    shifted = {key: value + 1.0 for key, value in payload.items()}
    assert trainer.import_parameters(shifted) == len(payload)
    again = trainer.export_parameters()
    assert all(torch.allclose(again[key], shifted[key]) for key in payload)
    with pytest.raises(ContractError):
        trainer.import_parameters({"nope/x": torch.zeros(1)})


def test_state_dict_roundtrip_resumes_position(tuned, tmp_path):
    trainer = _trainer(tuned, tmp_path)
    trainer.fit(epochs=1)
    payload = trainer.state_dict()
    saved = trainer.export_parameters()
    trainer.import_parameters({k: v + 3.0 for k, v in saved.items()})
    report = trainer.load_state_dict(payload, strict=False)
    restored = trainer.export_parameters()
    assert all(torch.allclose(restored[k], saved[k]) for k in saved)
    assert trainer.start_epoch == report.episode + 1
