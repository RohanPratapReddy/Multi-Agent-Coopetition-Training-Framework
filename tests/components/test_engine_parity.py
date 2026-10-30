"""The helpers duplicated from ``Orchestrator`` on purpose (document 17 section 13.4.3, rows 6-8)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from engine_support import engine_of, fresh, modules, tune

from mactf.orchestration.engine import TrainerEngine
from mactf.orchestration.orchestrator import Orchestrator, RunConfig


def _pair(settings, tmp_path):
    settings["ORCHESTRATOR"]["ENGINE"] = "trainer"
    return Orchestrator(settings, modules=modules()), TrainerEngine(settings, modules=modules())


@pytest.mark.parametrize("task", [None, "synthetic", "synthetic_classification", "jsonl", "no_such_source"])
def test_task_source_matches_orchestrator(settings, tmp_path, task):
    """The same degradation rule: a source that cannot be built falls back to synthetic, with a note."""
    tune(settings, tmp_path)
    orchestrator, engine = _pair(settings, tmp_path)
    config = RunConfig(task_source=task)
    mine, theirs = engine._build_task_source(config), orchestrator._build_task_source(config)
    assert type(mine) is type(theirs)
    assert engine.startup_notes == orchestrator.startup_notes
    import torch

    first_mine, first_theirs = mine.next(0), theirs.next(0)
    assert torch.equal(first_mine.states, first_theirs.states) and torch.equal(first_mine.targets, first_theirs.targets)
    assert engine._task_name(config) == orchestrator._task_name(config)


def test_task_name_precedence_matches_orchestrator(settings, tmp_path):
    tune(settings, tmp_path)
    orchestrator, engine = _pair(settings, tmp_path)
    for orchestrator_task, trainer_task, config in (
        (None, "synthetic", RunConfig()), ("jsonl", "synthetic", RunConfig()), ("jsonl", "synthetic", RunConfig(task_source="x")),
    ):
        settings["ORCHESTRATOR"]["TASK_SOURCE"] = orchestrator_task
        settings["TRAINING"]["TASK"] = trainer_task
        assert engine._task_name(config) == orchestrator._task_name(config)
    assert engine._block("ORCHESTRATOR") == orchestrator._block("ORCHESTRATOR")
    assert engine._block("NO_SUCH_BLOCK") == orchestrator._block("NO_SUCH_BLOCK") == {}


def test_sandbox_choice_matches_orchestrator(settings, tmp_path):
    tune(settings, tmp_path)
    orchestrator, engine = _pair(settings, tmp_path)
    mine, theirs = engine._choose_sandbox(RunConfig()), orchestrator._choose_sandbox(RunConfig())
    assert mine.describe() == theirs.describe() and mine.report.provider == theirs.report.provider
    assert engine.startup_notes == orchestrator.startup_notes
    # and when the requested isolation cannot be had, the same degradation note and the same signal
    settings["SANDBOX"]["ISOLATION"] = "hardware"
    settings["SANDBOX"]["PROVIDER"] = "local"
    from mactf import signals

    heard = {"orchestrator": [], "engine": []}
    which = {"now": "orchestrator"}
    receiver = lambda signal=None, sender=None, reason=None, **_: heard[which["now"]].append(reason)
    signals.integrity_incident.connect(receiver, weak=False)
    try:
        orchestrator, engine = _pair(settings, tmp_path)
        orchestrator._choose_sandbox(RunConfig())
        which["now"] = "engine"
        engine._choose_sandbox(RunConfig())
    finally:
        signals.integrity_incident.disconnect(receiver)
    assert engine.startup_notes == orchestrator.startup_notes
    assert heard["engine"] == heard["orchestrator"]                        # the same incidents, whatever their number


def test_a_missing_provider_is_a_note_not_a_failure(settings, tmp_path, monkeypatch):
    tune(settings, tmp_path)
    orchestrator, engine = _pair(settings, tmp_path)

    def broken(**kwargs):
        raise RuntimeError("no provider")

    monkeypatch.setattr("mactf.orchestration.engine.choose_provider", broken)
    monkeypatch.setattr("mactf.orchestration.orchestrator.choose_provider", broken)
    assert engine._choose_sandbox(RunConfig()) is None and orchestrator._choose_sandbox(RunConfig()) is None
    assert engine.startup_notes == orchestrator.startup_notes == ["sandbox unavailable: no provider"]


def test_checkpoint_directory_matches_orchestrator(settings, tmp_path):
    """``CHECKPOINT.DIR`` beats ``PATHS["CHECKPOINTS"]``, identically; a relative one is resolved against ``ROOT``."""
    tune(settings, tmp_path)
    settings["_AUTHORED"] = {**dict(settings.get("_AUTHORED") or {}), "CHECKPOINT": ["DIR"]}
    orchestrator, engine = _pair(settings, tmp_path)
    assert engine._build_checkpoint_writer().directory == orchestrator._build_checkpoint_manager().directory
    assert engine._build_checkpoint_writer().directory == Path(settings["CHECKPOINT"]["DIR"])
    settings["CHECKPOINT"]["DIR"] = "relative/ck"
    assert engine._build_checkpoint_writer().directory == orchestrator._build_checkpoint_manager().directory
    assert engine._build_checkpoint_writer().directory == settings.path("ROOT") / "relative/ck"
    # nothing authored: the PATHS entry
    settings["CHECKPOINT"]["DIR"] = None
    settings["_AUTHORED"] = {}
    assert engine._build_checkpoint_writer().directory == orchestrator._build_checkpoint_manager().directory
    assert engine._build_checkpoint_writer().directory == settings.ensure_path("CHECKPOINTS")


def test_the_writer_carries_the_registry_and_the_run_id(settings, tmp_path):
    tune(settings, tmp_path)
    engine = engine_of(settings, tmp_path)
    writer = engine._build_checkpoint_writer()
    assert writer.registry is engine.registry and writer.run_id == engine.run_id


def test_a_named_bad_directory_is_not_hidden(settings, tmp_path):
    from mactf.exceptions import ImproperlyConfigured

    tune(settings, tmp_path)
    blocker = tmp_path / "file"
    blocker.write_text("x")
    settings["CHECKPOINT"]["DIR"] = str(blocker / "ck")
    engine = engine_of(settings, tmp_path)
    with pytest.raises((ImproperlyConfigured, OSError)):
        engine._build_checkpoint_writer()
