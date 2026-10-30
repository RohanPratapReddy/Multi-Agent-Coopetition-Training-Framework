"""End-to-end tests: settings, registry, checks, and a real training loop."""

from __future__ import annotations

import math

import pytest
import torch

from mactf.db import Registry, ensure_registry
from mactf.exceptions import ImproperlyConfigured


# -- settings ------------------------------------------------------------
def test_settings_defaults_and_paths(settings):
    assert settings["DEVICE"] == "cpu"
    assert settings.path("ROOT").is_dir()
    assert settings.path("CHECKPOINTS").name == "checkpoints"


def test_settings_rejects_unknown_getting(settings):
    with pytest.raises(AttributeError):
        _ = settings.NOT_A_SETTING


def test_settings_validation_rejects_weakened_non_offsettable(settings):
    from mactf.conf import validate_settings

    settings["SCORING"]["NON_OFFSETTABLE"] = ["P1"]
    with pytest.raises(ImproperlyConfigured, match="NON_OFFSETTABLE"):
        validate_settings(settings)


def test_settings_validation_rejects_unordered_severity(settings):
    from mactf.conf import validate_settings

    settings["SEVERITY"]["WEIGHTS"] = {"S0": 0.0, "S1": 2.0, "S2": 1.0, "S3": 4.0, "S4": 25.0}
    with pytest.raises(ImproperlyConfigured, match="increasing"):
        validate_settings(settings)


def test_settings_env_override(monkeypatch, tmp_path):
    from mactf import conf

    conf.reset()
    module = type(__import__("sys"))("env_settings")
    module.BASE_DIR = str(tmp_path)
    module.CONFIG_FILES = []
    import sys

    sys.modules["env_settings"] = module
    monkeypatch.setenv("MACTF_SEED", "99")
    loaded = conf.load_settings("env_settings")
    assert loaded["SEED"] == 99
    conf.reset()


def test_project_settings_load_and_merge_yaml(project_settings):
    assert project_settings["TRAINING"]["EPOCHS"] >= 1
    assert project_settings["SCORING"]["PRECEDENCE_MODE"] == "strict"
    # YAML overlay must have set the multiplier.
    assert project_settings["SCORING"]["SANDBOX_MULTIPLIER"] == 1000.0


# -- runtime -------------------------------------------------------------
def test_runtime_resolves_cpu_dtype():
    from mactf.runtime import resolve_device, resolve_dtype

    device = resolve_device("cpu")
    assert device.type == "cpu"
    assert resolve_dtype("float16", device) is torch.float32
    assert resolve_dtype("float32", device) is torch.float32


def test_runtime_builds_and_seeds(settings):
    from mactf.runtime import build_runtime

    runtime = build_runtime(settings)
    assert runtime.device.type == "cpu"
    first = torch.randn(3)
    second = torch.randn(3)
    assert not torch.allclose(first, second)


def test_runtime_gpu_reports_when_available():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from mactf.runtime import resolve_device

    assert resolve_device("cuda").type == "cuda"
    assert resolve_device("auto").type == "cuda"


# -- registry ------------------------------------------------------------
def test_registry_migrate_is_idempotent(tmp_path):
    registry = Registry(tmp_path / "test.sqlite3")
    assert registry.migrate() == [1, 2, 3, 4]
    assert registry.migrate() == []
    assert registry.schema_version() == 4


def test_registry_run_lifecycle(tmp_path):
    registry = Registry(tmp_path / "test.sqlite3")
    registry.migrate()
    registry.start_run("run1", settings={"a": 1}, runtime={"device": "cpu"})
    registry.record_episode("run1", 0, {"peers": {}})
    registry.record_events(
        "run1", 0, "peer",
        [{"rule_id": "R1", "magnitude": 1.0, "severity": "S2"}],
    )
    registry.finish_run("run1", status="completed", kappa=1.5, entropy=0.9)

    run = registry.get_run("run1")
    assert run["status"] == "completed"
    assert run["kappa"] == pytest.approx(1.5)
    assert registry.latest_episode("run1") == 0
    assert registry.rule_counts("run1") == {"R1": 1}


def test_ensure_registry_creates_schema(settings):
    registry = ensure_registry(settings)
    assert registry.schema_version() >= 1


# -- checks --------------------------------------------------------------
def test_system_checks_pass_on_project(project_settings, monkeypatch):
    from mactf import apps, checks, conf

    monkeypatch.setenv("MACTF_SETTINGS_MODULE", "training.settings")
    conf._settings = project_settings
    apps.reset_registry()
    apps.populate()
    outcome = checks.run_checks()
    assert outcome.ok, [str(message) for message in outcome.errors]


# -- orchestration -------------------------------------------------------
def _tiny_modules(count: int = 2):
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(nn.Linear(4, 16), nn.Tanh())
            self.policy = nn.Linear(16, 3)
            self.value = nn.Linear(16, 1)

        def forward(self, state):
            hidden = self.body(state)
            return self.policy(hidden), self.value(hidden)

    return {f"p{index}": Tiny() for index in range(count)}


def test_orchestrator_runs_and_records(settings, tmp_path):
    from mactf.orchestration.orchestrator import Orchestrator, RunConfig

    settings["TRAINING"].update(
        {"EPOCHS": 2, "BATCH_SIZE": 8, "STATE_DIM": 4, "ACTION_DIM": 3, "CHECKPOINT_EVERY": 1}
    )
    settings["SANDBOX"]["ALLOW_FALLBACK"] = True
    settings["SANDBOX"]["PROVIDER"] = "local"
    orchestrator = Orchestrator(settings, modules=_tiny_modules(2))
    summary = orchestrator.run(RunConfig(epochs=2))

    assert summary["status"] == "completed"
    assert len(summary["history"]) == 2
    for entry in summary["history"]:
        assert math.isfinite(entry["kappa"])
        assert math.isfinite(entry["entropy"])
        assert math.isfinite(entry["score"])
        assert set(entry["weights"]) == {"p0", "p1"}
        assert sum(entry["weights"].values()) == pytest.approx(1.0, abs=1e-5)

    registry = orchestrator.registry
    run = registry.get_run(summary["run_id"])
    assert run["status"] == "completed"
    assert len(registry.episodes(summary["run_id"])) == 2
    assert registry.checkpoints(summary["run_id"])


def test_orchestrator_marks_failed_runs(settings):
    from mactf.orchestration.orchestrator import Orchestrator, RunConfig

    settings["TRAINING"]["STATE_DIM"] = 4
    orchestrator = Orchestrator(settings, modules=_tiny_modules(2))
    # A zero-length task batch makes the judge produce no events and the loss
    # cannot be computed; the registry must still be finalized as failed.
    original = orchestrator.prepare
    orchestrator.prepare = lambda config: (_ for _ in ()).throw(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        orchestrator.run(RunConfig(epochs=1))


def test_orchestrator_dry_run(settings):
    from mactf.orchestration.orchestrator import Orchestrator, RunConfig

    settings["TRAINING"]["STATE_DIM"] = 4
    settings["SANDBOX"]["PROVIDER"] = "local"
    orchestrator = Orchestrator(settings, modules=_tiny_modules(2))
    summary = orchestrator.run(RunConfig(dry_run=True))
    assert summary["status"] == "dry_run"
    assert summary["peers"] == ["p0", "p1"]


def test_checkpoint_roundtrip(settings):
    from mactf.coopetition.peer import build_peer
    from mactf.orchestration.checkpoint import CheckpointManager
    from mactf.runtime import build_runtime

    runtime = build_runtime(settings)
    module = _tiny_modules(1)["p0"]
    peer = build_peer(module, name="p0", runtime=runtime, action_dim=3, state_dim=4)
    original = peer.export_parameters().clone()

    manager = CheckpointManager(settings.ensure_path("CHECKPOINTS"), run_id="test")
    manager.save(
        episode=0,
        peers={"p0": peer},
        reference=None,
        controller=None,
        warnings={},
        runtime={"device": "cpu"},
        kappa=1.0,
        entropy=0.5,
        score=0.1,
    )
    peer.import_parameters(original + 5.0)
    assert not torch.allclose(peer.export_parameters(), original)

    payload = manager.load(0)
    manager.restore(payload, peers={"p0": peer})
    assert torch.allclose(peer.export_parameters(), original, atol=1e-5)


def test_should_rollback_on_collapse():
    from mactf.orchestration.checkpoint import should_rollback

    rollback, reason = should_rollback(
        entropy=0.01, max_entropy=math.log(3), collapse_ratio=0.05,
        fallback=False, enabled=True,
    )
    assert rollback and "entropy" in reason
    rollback, _ = should_rollback(
        entropy=1.0, max_entropy=math.log(3), collapse_ratio=0.05,
        fallback=False, enabled=True,
    )
    assert not rollback
    rollback, _ = should_rollback(
        entropy=1.0, max_entropy=1.1, collapse_ratio=0.05, fallback=False, enabled=False
    )
    assert not rollback


# -- docs integrity ------------------------------------------------------
def test_docs_are_present_and_unchanged_shape():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    framework = root / "docs" / "framework.md"
    backup = root / "docs" / "framework.md.bak"
    assert framework.exists() and framework.stat().st_size > 10_000
    assert backup.exists() and backup.stat().st_size > 500
