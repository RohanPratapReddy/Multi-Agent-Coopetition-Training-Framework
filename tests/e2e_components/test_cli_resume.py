"""A project made by ``startproject``, driven through ``manage.py run`` in real processes.

An interrupted-then-resumed run must continue exactly like an uninterrupted one: the weights,
the optimizer state and the scheduler of the last checkpoint compare equal with ``torch.equal``.
"""

from __future__ import annotations

import signal
from pathlib import Path

import pytest

from e2e_support import BASE_FLAGS, assert_same_state, checkpoint_file, manage, spawn, state_of, wait_for

pytestmark = [pytest.mark.subprocess, pytest.mark.posix]

CUSTOM = """\
COMPONENTS:
  MODULES: ["mactf.contrib.components.examples.ema_softmax", "mactf.contrib.components.examples.lion"]
TRAINER:
  WEIGHT_UPDATE: {type: ema_softmax, params: {alpha: 0.5}}
OPTIMIZER: {TYPE: Lion, LR: 0.001, WEIGHT_DECAY: 0.0, SCHEDULER: step_then_flat, SCHEDULER_KWARGS: {drop_at: 3}}
"""
EPOCHS = 6
#: Long enough that the SIGTERM always lands before a fast Orchestrator run completes.
LEGACY_EPOCHS = 60


def _args(project: Path, run_id: str, *, engine: str, custom: bool, extra=()) -> list[str]:
    args = [*BASE_FLAGS, "--seed", "7", "--set", f"ORCHESTRATOR.ENGINE={engine}", "--set", "ORCHESTRATOR.ENGINE_COMPAT=false"]
    if custom:
        (project / "extra.yaml").write_text(CUSTOM)
        args += ["--config", "extra.yaml"]
    return [*args, *extra]


def _interrupt_and_resume(project, run_id, *, engine, custom, sig, after_epoch=2):
    args = _args(project, run_id, engine=engine, custom=custom)
    proc = spawn(project, "run", "--epochs", str(EPOCHS), "--run-id", run_id, *args)
    wait_for(checkpoint_file(project, run_id, after_epoch), proc)
    proc.send_signal(sig)
    proc.wait(timeout=60)
    if sig == signal.SIGKILL:
        reaped = manage(project, "runs", "reap", "--yes")
        assert reaped.returncode == 0, reaped.stderr
    done = manage(project, "run", "--epochs", str(EPOCHS), "--resume", run_id, *args)
    assert done.returncode == 0, done.stderr + done.stdout
    return done


@pytest.mark.parametrize("custom", [False, True], ids=["default", "customised"])
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGKILL], ids=["sigterm", "sigkill"])
def test_an_interrupted_trainer_run_resumes_exactly(project, custom, sig):
    """TrainerEngine, per-epoch signal window: bit-for-bit, with a custom weight update, optimizer, scheduler."""
    args = _args(project, "full", engine="trainer", custom=custom)
    full = manage(project, "run", "--epochs", str(EPOCHS), "--run-id", "full", *args)
    assert full.returncode == 0, full.stderr + full.stdout
    _interrupt_and_resume(project, "cut", engine="trainer", custom=custom, sig=sig)
    last = EPOCHS - 1
    sections = ("models", "reference", "optimizers", "schedulers")
    full_state = state_of(project, "full", last, sections)
    assert_same_state(full_state, state_of(project, "cut", last, sections))


@pytest.mark.parametrize("engine,compat", [("orchestrator", "false"), ("trainer", "true")])
def test_the_legacy_signal_window_resumes_without_error(project, engine, compat):
    """The Orchestrator (and the engine in its compat mode) keep the cumulative signal window in memory only,
    so a resume is *continuous but not bit-identical* (finding in the implementation notes); it must still finish."""
    args = [*_args(project, "x", engine=engine, custom=True)]
    args[args.index("ORCHESTRATOR.ENGINE_COMPAT=false")] = f"ORCHESTRATOR.ENGINE_COMPAT={compat}"
    proc = spawn(project, "run", "--epochs", str(LEGACY_EPOCHS), "--run-id", "x", *args)
    wait_for(checkpoint_file(project, "x", 2), proc)
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=60)
    done = manage(project, "run", "--epochs", str(LEGACY_EPOCHS), "--resume", "x", *args)
    assert done.returncode == 0, done.stderr + done.stdout
    assert "completed" in done.stdout + done.stderr


def test_the_same_seed_gives_the_same_run_in_two_fresh_processes(project):
    """Regression: ``TrainerEngine.prepare`` built the peers' modules before seeding, so the initial weights
    of one ``SEED`` differed from process to process (the Orchestrator seeds first)."""
    args = _args(project, "x", engine="trainer", custom=False)
    for run_id in ("one", "two"):
        done = manage(project, "run", "--epochs", "1", "--run-id", run_id, *args)
        assert done.returncode == 0, done.stderr
    sections = ("models", "reference")
    assert_same_state(state_of(project, "one", 0, sections), state_of(project, "two", 0, sections))
