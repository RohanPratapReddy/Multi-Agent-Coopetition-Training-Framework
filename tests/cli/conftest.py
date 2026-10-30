"""Fixtures for the command-line tests (document 16, section 10.1)."""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from mactf import conf
from mactf.testing.cli import CliResult, invoke

TESTS = Path(__file__).resolve().parent.parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))                # conf_project lives one level up

HAVE_TORCH = importlib.util.find_spec("torch") is not None


def pytest_collection_modifyitems(config, items):
    """A job without torch skips the tests that say they need it."""
    if HAVE_TORCH:
        return
    skip = pytest.mark.skip(reason="torch is not installed in this job")
    for item in items:
        if "needs_torch" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def hermetic(monkeypatch, tmp_path):
    """No environment variable, working directory, or settings cache leaks in or out."""
    for key in [k for k in os.environ if k.startswith("MACTF_")]:
        monkeypatch.delenv(key)
    for key in ("NO_COLOR", "FORCE_COLOR", "COLUMNS", "LINES"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MACTF_NO_DISCOVER", "1")          # tests that exercise discovery unset it
    monkeypatch.setenv("COLUMNS", "100")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    conf.reset()
    yield
    conf.reset()


@dataclass
class Project:
    root: Path                       # the directory that holds the package
    package: Path                    # holds settings.py and conf/
    module: str                      # "<name>.settings"
    env: dict[str, str]

    def invoke(self, command: str, *argv: str, **kw) -> CliResult:
        """Global options follow the command name (4.4), so the settings module goes last."""
        env = {**self.env, **kw.pop("env", {})}
        return invoke([command, *argv, "--settings", self.module], env=env, **kw)


@pytest.fixture()
def project(tmp_path, monkeypatch) -> Project:
    """A generated project (``startproject``) whose settings load, with its package importable."""
    from conf_project import purge_project, scaffold

    purge_project("demo_swarm")
    package = scaffold(tmp_path, "demo_swarm")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield Project(tmp_path, package, "demo_swarm.settings", {})
    purge_project("demo_swarm")


@dataclass
class Seeded:
    project: Project
    registry: object
    settings: object
    checkpoint_dir: Path
    lock: object                     # the RunLock held for run-running-live

    def invoke(self, *argv: str, **kw) -> CliResult:
        return self.project.invoke(*argv, **kw)


def _write_checkpoint(directory: Path, run_id: str, episode: int, *, valid: bool = True) -> Path:
    """A legacy-format checkpoint (``run_id``, ``episode``, ``peers``), or garbage bytes."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}-ep{episode:04d}.pt"
    if valid:
        import torch

        torch.save({"run_id": run_id, "episode": episode, "peers": {"a": {}}}, path)
    else:
        path.write_bytes(b"not a checkpoint")
    return path


def seed_registry(project: Project) -> Seeded:
    """Rows through the registry's own API (never SQL), in the shape of 10.1.1 of the design."""
    from mactf.db import Registry
    from mactf.orchestration.runlock import RunLock, lock_directory

    settings = conf.load_settings(project.module)
    registry = Registry.from_settings(settings)
    registry.migrate()
    directory = settings.path("CHECKPOINTS")

    def run(run_id, status, episodes=0, checkpoints=(), *, valid=True, metrics=True):
        registry.start_run(run_id, settings={"TRAINING": {"EPOCHS": 3}}, runtime={})
        for episode in range(1, episodes + 1):
            registry.record_episode(run_id, episode, {"peers": {"a": {"final_score": 0.5}}})
        for episode in checkpoints:
            path = _write_checkpoint(directory, run_id, episode, valid=valid)
            registry.record_checkpoint(run_id, episode, path, kappa=0.5 if metrics else None,
                                       entropy=0.4, score=0.7 if metrics else None)
        if status != "running":
            registry.set_run_status(run_id, status)

    run("run-completed-a", "completed", 4, (1, 2, 3))
    run("run-completed-b", "completed", 2, (1,), valid=False)
    run("run-paused", "paused", 3, (1, 2))
    run("run-interrupted", "interrupted", 1)
    run("run-running-dead", "running", 1)
    run("run-running-live", "running", 1)
    run("run-failed", "failed", 2)
    run("run-cancelled", "cancelled", 0)
    run("run-null-metrics", "completed", 1, (1,), metrics=False)
    lock = RunLock(lock_directory(settings), "run-running-live", owner="test").acquire()
    return Seeded(project, registry, settings, directory, lock)


@pytest.fixture()
def seeded(project):
    pytest.importorskip("torch")
    state = seed_registry(project)
    yield state
    state.lock.release()
