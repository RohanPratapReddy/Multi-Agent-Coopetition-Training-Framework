"""Resume, branch and the cross-engine guard (document 17 section 13.4.8, rows 18-20; T5)."""

from __future__ import annotations

import copy

import pytest
import torch

from engine_support import engine_of, fresh, same_tensors, tune

from mactf.exceptions import ImproperlyConfigured
from mactf.orchestration.orchestrator import Orchestrator, RunConfig


def _stopping_after(epoch):
    state = {"seen": 0}

    def should_stop():
        state["seen"] += 1
        return state["seen"] > epoch

    return should_stop


def _pause(engine):
    engine.registry.set_run_status(engine.run_id, "paused", finished=False)


def test_resume_continues_at_the_next_epoch(settings, tmp_path):
    """Per-epoch window (``ENGINE_COMPAT: false``): a resumed run is bit-identical to one that never stopped."""
    tune(settings, tmp_path, epochs=4, ENGINE_COMPAT=False)
    uninterrupted = engine_of(settings, tmp_path / "ref")
    settings["CHECKPOINT"]["DIR"] = str(tmp_path / "ref-ck")
    uninterrupted.run(RunConfig(epochs=4, run_id="ref"))

    settings["CHECKPOINT"]["DIR"] = str(tmp_path / "split-ck")
    first = engine_of(settings, tmp_path / "a")
    summary = first.run(RunConfig(epochs=4, run_id="split"), should_stop=_stopping_after(1))
    assert summary["status"] == "cancelled" and summary["epochs_completed"] == 2
    _pause(first)
    first.close()

    second = engine_of(settings, tmp_path / "b")
    resumed = second.run(RunConfig(epochs=4, run_id="split", resume_from=True))
    assert resumed["status"] == "completed" and [e["epoch"] for e in resumed["history"]] == [2, 3]
    assert any("resumed from episode 1" in note for note in resumed["notes"])
    for name, peer in second.peers.items():
        assert torch.equal(peer.export_parameters(), uninterrupted.peers[name].export_parameters()), name
    reports = second.registry.resume_reports("split")
    assert reports[-1]["started_epoch"] == 2 and reports[-1]["checkpoint_episode"] == 1
    assert reports[-1]["report"]["engine"] == "trainer"


def test_resume_restores_warnings(settings, tmp_path):
    from mactf.rules.severity import Severity

    tune(settings, tmp_path, epochs=3, ENGINE_COMPAT=False)
    first = engine_of(settings, tmp_path / "a")
    first.run(RunConfig(epochs=3, run_id="warned"), should_stop=_stopping_after(0))
    first.scorer.registry_for("peer-1").issue("peer-1", "P11", severity=Severity.MINOR, episode=0)
    first.registry.record_warning("warned", "peer-1", first.scorer.registry_for("peer-1").all_active("peer-1")["P11"][0])
    _pause(first)
    second = engine_of(settings, tmp_path / "b")
    second.run(RunConfig(epochs=3, run_id="warned", resume_from=True))
    assert second.scorer.registry_for("peer-1").count("peer-1", "P11") >= 1
    assert any("restored" in note and "warning" in note for note in second.startup_notes)


def test_resume_with_no_checkpoint_starts_at_zero_with_w051(settings, tmp_path):
    tune(settings, tmp_path, epochs=1)
    engine = engine_of(settings, tmp_path)
    summary = engine.run(RunConfig(epochs=1, resume_from=True))
    assert summary["status"] == "completed" and [e["epoch"] for e in summary["history"]] == [0]
    assert any(note.startswith("W051") for note in summary["notes"])


def test_resume_without_a_writer_directory_is_a_configuration_error(settings, tmp_path):
    from mactf.orchestration.engine import TrainerEngine
    from mactf.training import MACTFTrainer

    tune(settings, tmp_path)
    trainer = MACTFTrainer(models=fresh(), settings=settings)
    trainer.build()
    trainer.checkpoint = None
    settings["ORCHESTRATOR"]["ENGINE"] = "trainer"
    engine = TrainerEngine(settings, trainer=trainer)
    with pytest.raises(ImproperlyConfigured, match="writable CHECKPOINT"):
        engine.run(RunConfig(epochs=1, resume_from=True))


def test_resume_across_engines_is_e091(settings, tmp_path):
    """T5: a v1 file under the trainer engine, and a trainer file under the orchestrator, are refused unless allowed."""
    tune(settings, tmp_path, epochs=2)
    # an orchestrator run leaves v1 checkpoints in the directory
    orchestrator = Orchestrator(settings, modules=copy.deepcopy(fresh()))
    orchestrator.run(RunConfig(epochs=1, run_id="mixed", seed=1))
    path = orchestrator.checkpoints.latest().path
    engine = engine_of(settings, tmp_path)
    with pytest.raises(ImproperlyConfigured, match="E091.*orchestrator engine"):
        engine.run(RunConfig(epochs=2, resume_from=path, seed=1))

    # the other direction: the file a trainer engine wrote says so
    trainer_ck = tmp_path / "trainer-ck"
    settings["CHECKPOINT"]["DIR"] = str(trainer_ck)
    writer = engine_of(settings, tmp_path / "w")
    writer.run(RunConfig(epochs=1, run_id="tr", seed=1))
    payload = writer.trainer.checkpoint.load(writer.trainer.checkpoint.latest().path)
    assert payload.identity["engine"] == "trainer"
    other = engine_of(settings, tmp_path / "o", engine="orchestrator")
    settings["ORCHESTRATOR"]["ENGINE"] = "trainer"
    from mactf.orchestration.engine import TrainerEngine

    probe = TrainerEngine(settings, modules=copy.deepcopy(fresh()))
    probe._check_identity(payload, where=trainer_ck)                       # the same engine: fine
    forged = copy.deepcopy(payload.as_dict())
    forged["identity"]["engine"] = "orchestrator"
    with pytest.raises(ImproperlyConfigured, match="E091"):
        probe._check_identity(forged, where=trainer_ck)
    assert other is not None

    # ALLOW_ENGINE_SWITCH: W050, and the parameters are restored
    settings["ORCHESTRATOR"]["ALLOW_ENGINE_SWITCH"] = True
    allowed = engine_of(settings, tmp_path / "x")
    allowed.registry.set_run_status("mixed", "paused", finished=False)
    summary = allowed.run(RunConfig(epochs=2, resume_from=path, seed=1, run_id="mixed"))
    assert any(note.startswith("W050") for note in summary["notes"])
    assert summary["status"] == "completed"


def test_branch_from_a_checkpoint(settings, tmp_path):
    """``resume_from=Path`` on a copied file starts a *new* run at ``episode + 1``."""
    import shutil

    tune(settings, tmp_path, epochs=3, ENGINE_COMPAT=False)
    first = engine_of(settings, tmp_path / "a")
    first.run(RunConfig(epochs=2, run_id="origin", seed=1))
    source = first.trainer.checkpoint.latest().path
    branch_file = tmp_path / "copy" / source.name
    branch_file.parent.mkdir()
    shutil.copy(source, branch_file)

    settings["CHECKPOINT"]["DIR"] = str(tmp_path / "branch-ck")
    branch = engine_of(settings, tmp_path / "b")
    # a different run id owns the file's identity: the branch reads it as its own history (non-strict identity)
    branch.prepare(RunConfig(epochs=3, run_id="branch", seed=1, resume_from=branch_file))
    branch.trainer.strict = False
    summary = branch.run(RunConfig(epochs=3, run_id="branch", seed=1, resume_from=branch_file))
    assert [e["epoch"] for e in summary["history"]] == [2]
    assert any(f"resumed from {branch_file.name}" in note for note in summary["notes"])
    for name, peer in first.peers.items():
        assert peer.export_parameters().shape == branch.peers[name].export_parameters().shape
    with pytest.raises(ImproperlyConfigured, match="does not exist"):
        engine_of(settings, tmp_path / "c").run(RunConfig(epochs=3, resume_from=tmp_path / "nope.pt"))
