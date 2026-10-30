"""T1, surface parity: ``TrainerEngine`` satisfies the protocol a driver reads (document 17 section 13.3)."""

from __future__ import annotations

import pytest

from engine_support import engine_of, modules, tune

from mactf.orchestration.dispatch import Engine
from mactf.orchestration.orchestrator import Orchestrator, RunConfig

#: Every member of the protocol, with the call site that reads it (13.3).
MEMBERS = (
    "settings", "run_id", "registry", "scorer", "checkpoints", "peers", "reference", "aggregator",
    "runner", "context", "sandbox_choice", "startup_notes", "started_at",
)


def test_members_before_prepare(settings, tmp_path):
    tune(settings, tmp_path)
    engine = engine_of(settings, tmp_path)
    for name in MEMBERS:
        assert hasattr(engine, name), name
    assert engine.peers == {} and engine.context is None and engine.startup_notes == []
    assert isinstance(engine.run_id, str) and len(engine.run_id) == 12
    assert callable(engine.prepare) and callable(engine.run)
    # the engine is not yet built: nothing to close, nothing to describe but its name
    assert engine.describe()["engine"] == "trainer" and engine.describe()["built"] is False
    engine.close()


def test_isinstance_protocol(settings, tmp_path):
    tune(settings, tmp_path)
    assert isinstance(engine_of(settings, tmp_path), Engine)
    assert isinstance(Orchestrator(settings, modules=modules()), Engine)
    assert not isinstance(object(), Engine)


def test_members_after_prepare_are_the_swarm_a_driver_reads(settings, tmp_path):
    tune(settings, tmp_path)
    engine = engine_of(settings, tmp_path)
    context = engine.prepare(RunConfig(epochs=1))
    assert context is engine.context and sorted(engine.peers) == ["peer-1", "peer-2"]
    assert context.runtime.describe()["device"] == "cpu" and context.peers is engine.peers
    assert engine.reference is not None and engine.aggregator is not None and engine.runner is not None
    assert engine.scorer is engine.trainer.scorer and engine.checkpoints is engine.trainer.checkpoint
    assert engine.sandbox_choice is not None and engine.sandbox_choice.describe() in engine.startup_notes
    assert engine.tree is engine.trainer.tree and engine.capabilities() is not None


def test_run_returns_the_driver_keys(settings, tmp_path):
    tune(settings, tmp_path, epochs=2)
    engine = engine_of(settings, tmp_path)
    summary = engine.run(RunConfig(epochs=2))
    for key in ("run_id", "status", "epochs", "epochs_completed", "runtime", "peers", "history", "notes"):
        assert key in summary, key
    assert summary["status"] == "completed" and summary["epochs_completed"] == 2 and summary["epochs"] == 2
    assert summary["peers"] == ["peer-1", "peer-2"] and summary["run_id"] == engine.run_id
    entry = summary["history"][0]
    for key in ("epoch", "duration", "scorecards", "signal_by_peer", "kappa", "entropy", "weights", "adjustments", "score"):
        assert key in entry, key
    # the keys driver.py:293-299 reads
    assert {"status", "history", "epochs_completed"} <= set(summary)


def test_a_dry_run_writes_a_dry_run_row_and_trains_nothing(settings, tmp_path):
    tune(settings, tmp_path)
    engine = engine_of(settings, tmp_path)
    summary = engine.run(RunConfig(epochs=2, dry_run=True))
    assert summary["status"] == "dry_run" and summary["engine"] == "trainer"
    assert engine.registry.get_run(engine.run_id)["status"] == "dry_run"
    assert engine.trainer.global_step == 0 and engine.trainer.history == []


def test_prepare_twice_and_run_reuse_the_same_swarm(settings, tmp_path):
    tune(settings, tmp_path)
    engine = engine_of(settings, tmp_path)
    config = RunConfig(epochs=1)
    first = engine.prepare(config)
    trainer = engine.trainer
    summary = engine.run(config)                         # the driver's pattern: prepare, then run(config)
    assert engine.trainer is trainer and engine.context is first and summary["status"] == "completed"
    assert len([c for c in trainer.callbacks._callbacks if c.name == "engine_phase_bridge"]) == 1
    assert len([c for c in trainer.callbacks._callbacks if c.name == "engine_cumulative_signal"]) == 1


def test_a_peer_count_in_the_run_config_limits_the_swarm(settings, tmp_path):
    tune(settings, tmp_path)
    engine = engine_of(settings, tmp_path, count=3)
    engine.prepare(RunConfig(epochs=1, peers=2))
    assert sorted(engine.peers) == ["peer-1", "peer-2"]


def test_no_peers_is_a_configuration_error(settings, tmp_path):
    from mactf.exceptions import ImproperlyConfigured
    from mactf.orchestration.engine import TrainerEngine

    tune(settings, tmp_path)
    settings["ORCHESTRATOR"]["ENGINE"] = "trainer"
    with pytest.raises(ImproperlyConfigured, match="no peers configured"):
        TrainerEngine(settings, modules={}).prepare(RunConfig(epochs=1))
