"""Entropy-collapse rollback under the trainer engine (document 17 section 13.4.6, rows 21-22)."""

from __future__ import annotations

import pytest
import torch

from engine_support import engine_of, tune

from mactf import signals
from mactf.orchestration.orchestrator import RunConfig


def _force(monkeypatch, *, on_calls):
    calls = {"n": 0}

    def fake(**kwargs):
        calls["n"] += 1
        return (calls["n"] in on_calls, "forced collapse" if calls["n"] in on_calls else "")

    monkeypatch.setattr("mactf.orchestration.engine.should_rollback", fake)


def test_entropy_collapse_rolls_back_once(settings, tmp_path, monkeypatch):
    tune(settings, tmp_path, epochs=3)
    _force(monkeypatch, on_calls={2})                      # the second epoch
    heard = []
    receiver = lambda signal=None, sender=None, reason=None, episode=None, **_: heard.append((reason, episode))
    signals.checkpoint_rolled_back.connect(receiver, weak=False)
    engine = engine_of(settings, tmp_path)
    snapshots = {}
    original = engine._maybe_rollback

    def spy(trainer, record, report, epoch, training, max_entropy):
        if epoch == 1:
            snapshots["before"] = {n: p.export_parameters().clone() for n, p in trainer.peers.items()}
        original(trainer, record, report, epoch, training, max_entropy)
        if epoch == 1:
            snapshots["after"] = {n: p.export_parameters().clone() for n, p in trainer.peers.items()}

    engine._maybe_rollback = spy
    try:
        summary = engine.run(RunConfig(epochs=3))
    finally:
        signals.checkpoint_rolled_back.disconnect(receiver)
    assert heard == [("forced collapse", 0)]
    assert summary["rollbacks"] == [{"epoch": 1, "restored": 0, "reason": "forced collapse"}]
    assert any("rollback at epoch 1" in note for note in summary["notes"])
    # the parameters are those of episode 0's checkpoint, not epoch 1's
    stored = engine.trainer.checkpoint.load(engine.trainer.checkpoint.list()[0].path)
    for name in snapshots["after"]:
        assert not torch.equal(snapshots["before"][name], snapshots["after"][name])
        assert all(torch.equal(snapshots["after"][name][:0], snapshots["after"][name][:0]) for _ in [0])
        flat = torch.cat([v.flatten() for v in stored.models[name].values()])
        assert torch.equal(snapshots["after"][name], flat), name
    assert summary["status"] == "completed" and summary["epochs_completed"] == 3
    # counters count work done, not work kept (D-13.5)
    assert engine.trainer.global_step >= 3


def test_no_earlier_checkpoint_is_a_note(settings, tmp_path, monkeypatch):
    tune(settings, tmp_path, epochs=1)
    _force(monkeypatch, on_calls={1})
    heard = []
    receiver = lambda **kw: heard.append(kw)
    signals.checkpoint_rolled_back.connect(receiver, weak=False)
    engine = engine_of(settings, tmp_path)
    try:
        summary = engine.run(RunConfig(epochs=1))
    finally:
        signals.checkpoint_rolled_back.disconnect(receiver)
    assert heard == [] and summary["rollbacks"] == []
    assert any("no earlier checkpoint exists" in note for note in summary["notes"])


def test_no_rollback_on_an_epoch_without_its_own_snapshot(settings, tmp_path, monkeypatch):
    tune(settings, tmp_path, epochs=2)
    settings["TRAINING"]["CHECKPOINT_EVERY"] = 2
    _force(monkeypatch, on_calls={1, 2})
    engine = engine_of(settings, tmp_path)
    summary = engine.run(RunConfig(epochs=2))
    assert summary["rollbacks"] == []                     # epoch 0 wrote none; epoch 1 had no earlier one
