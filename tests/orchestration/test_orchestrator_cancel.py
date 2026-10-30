"""Run cancellation and per-epoch observation (gaps 7 and B1).

``should_stop`` is polled strictly at epoch boundaries and ``on_epoch`` receives
the *same* dict that lands in ``history``, so the CLI and the SSE stream cannot
diverge on what an epoch event means.
"""

from __future__ import annotations

import pytest

import torch.nn as nn
import torch

from mactf.orchestration.orchestrator import Orchestrator, RunConfig


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(4, 16), nn.Tanh())
        self.policy = nn.Linear(16, 3)
        self.value = nn.Linear(16, 1)

    def forward(self, state):
        hidden = self.body(state)
        return self.policy(hidden), self.value(hidden)


def _modules(count: int = 2):
    return {f"p{index}": Tiny() for index in range(count)}


@pytest.fixture()
def run_settings(settings):
    settings["TRAINING"].update(
        {"EPOCHS": 3, "BATCH_SIZE": 8, "STATE_DIM": 4, "ACTION_DIM": 3, "CHECKPOINT_EVERY": 1}
    )
    settings["SANDBOX"].update({"ALLOW_PROVIDER": "local", "PROVIDER": "local"})
    return settings


def _orchestrator(settings) -> Orchestrator:
    return Orchestrator(settings, modules=_modules(2))


def test_run_without_callbacks_is_unchanged(run_settings):
    summary = _orchestrator(run_settings).run(RunConfig(epochs=2))
    assert summary["status"] == "completed"
    assert summary["epochs_completed"] == 2
    assert len(summary["history"]) == 2


def test_should_stop_early_marks_the_run_cancelled(run_settings):
    orchestrator = _orchestrator(run_settings)
    calls = {"n": 0}

    def should_stop() -> bool:
        calls["n"] += 1
        return calls["n"] >= 1

    summary = orchestrator.run(RunConfig(epochs=3), should_stop=should_stop)

    assert summary["status"] == "cancelled"
    assert summary["epochs_completed"] == 1
    assert len(summary["history"]) == 1
    # The registry row is the audit trail, so it must agree.
    assert orchestrator.registry.get_run(summary["run_id"])["status"] == "cancelled"
    # One check per completed epoch, at the boundary only.
    assert calls["n"] == 1


def test_should_stop_that_never_fires_completes_every_epoch(run_settings):
    summary = _orchestrator(run_settings).run(
        RunConfig(epochs=3), should_stop=lambda: False
    )
    assert summary["status"] == "completed"
    assert summary["epochs_completed"] == 3


def test_cancel_after_all_epochs_is_still_completed(run_settings):
    # A flag set after the last epoch must not flip a finished run to cancelled.
    class Gate:
        def __init__(self, limit: int) -> None:
            self.limit = limit
            self.calls = 0

        def __call__(self) -> bool:
            self.calls += 1
            return self.calls > self.limit

    summary = _orchestrator(run_settings).run(
        RunConfig(epochs=2), should_stop=Gate(2)
    )
    assert summary["status"] == "completed"
    assert summary["epochs_completed"] == 2


def test_on_epoch_receives_the_history_entries(run_settings):
    seen: list[dict] = []
    summary = _orchestrator(run_settings).run(
        RunConfig(epochs=2), on_epoch=seen.append
    )
    assert len(seen) == 2
    # Identity, not equality: the API and the CLI must not build two dicts.
    for entry, recorded in zip(seen, summary["history"]):
        assert entry is recorded
    assert [entry["epoch"] for entry in seen] == [0, 1]


def test_on_epoch_entry_has_the_event_fields(run_settings):
    seen: list[dict] = []
    _orchestrator(run_settings).run(RunConfig(epochs=1), on_epoch=seen.append)
    entry = seen[0]
    for key in ("epoch", "duration", "scorecards", "kappa", "entropy", "weights", "score"):
        assert key in entry


def test_on_epoch_is_called_before_the_stop_check(run_settings):
    order: list[str] = []

    def on_epoch(_entry: dict) -> None:
        order.append("epoch")

    def should_stop() -> bool:
        order.append("stop")
        return True

    _orchestrator(run_settings).run(
        RunConfig(epochs=2), should_stop=should_stop, on_epoch=on_epoch
    )
    # The epoch is observable before it can be the last one.
    assert order == ["epoch", "stop"]


def test_a_raising_on_epoch_fails_the_run_and_records_it(run_settings):
    orchestrator = _orchestrator(run_settings)

    def explode(_entry: dict) -> None:
        raise RuntimeError("subscriber blew up")

    with pytest.raises(RuntimeError, match="subscriber blew up"):
        orchestrator.run(RunConfig(epochs=2), on_epoch=explode)

    run = orchestrator.registry.get_run(orchestrator.run_id)
    assert run["status"] == "failed"
    assert "subscriber blew up" in (run["notes"] or "")


def test_resume_from_true_means_latest_checkpoint(run_settings):
    # The CLI passes ``True``; the annotation used to claim ``Path``.
    config = RunConfig(epochs=2, resume_from=True)
    assert config.resume_from is True


def test_resume_from_accepts_a_path_none_and_true():
    from pathlib import Path

    assert RunConfig(resume_from=None).resume_from is None
    assert RunConfig(resume_from=True).resume_from is True
    assert RunConfig(resume_from=Path("x")).resume_from == Path("x")
    assert RunConfig().resume_from is None


def test_task_source_is_optional_and_defaults_to_none():
    assert RunConfig().task_source is None
    assert RunConfig(task_source="mixed").task_source == "mixed"
