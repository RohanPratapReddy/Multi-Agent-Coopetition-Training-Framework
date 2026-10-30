"""The run manager drives the trainer engine (document 17 section 13.6.2, rows 25-28)."""

from __future__ import annotations

import contextvars
import threading
import time

import pytest
import torch

from mactf.orchestration.live import RunManager, RunState
from mactf.testing.tree import tree_of  # noqa: F401  (the testing package imports cleanly)


class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.Tanh())
        self.policy = torch.nn.Linear(16, 3)
        self.value = torch.nn.Linear(16, 1)

    def forward(self, state):
        hidden = self.body(state)
        return self.policy(hidden), self.value(hidden)


@pytest.fixture()
def live_settings(settings):
    settings["TRAINING"].update(
        {"EPOCHS": 3, "BATCH_SIZE": 8, "STATE_DIM": 4, "ACTION_DIM": 3, "CHECKPOINT_EVERY": 1, "BOOTSTRAP": False}
    )
    settings["SANDBOX"].update({"ALLOW_PROVIDER": "local", "PROVIDER": "local", "ALLOW_FALLBACK": True})
    settings["ORCHESTRATOR"]["ENGINE"] = "trainer"
    return settings


@pytest.fixture()
def manager(live_settings):
    mgr = RunManager(live_settings, modules={"p0": Tiny(), "p1": Tiny(), "p2": Tiny()}, max_concurrent=2, cancel_grace=30.0)
    yield mgr
    mgr.shutdown(timeout=30.0)


def test_manager_drives_the_trainer(manager):
    handle = manager.start({"epochs": 2, "run_id": "live"})
    assert manager.wait("live", timeout=60) == "completed"
    assert type(handle.engine).__name__ == "TrainerEngine" and handle.engine is handle.orchestrator
    names = [event["event"] for event in manager.bus.history("live")]
    for expected in ("run", "phase", "epoch", "checkpoint", "terminal"):
        assert expected in names, expected
    assert names.count("epoch") == 2 and names[-1] == "terminal"
    phases = [e["data"].get("phase") for e in manager.bus.history("live") if e["event"] == "phase"]
    assert {"bootstrap", "debate", "scoring", "local_update", "aggregation"} <= set(phases)
    assert "trainer" in handle.components and handle.components["trainer"] is handle.engine.trainer
    assert manager.registry.get_run("live")["status"] == "completed"


def test_contextvars_reach_the_run_thread(live_settings):
    """E-G8: a value set on the starting thread is visible inside the run thread."""
    marker = contextvars.ContextVar("marker", default="unset")
    seen = []

    class Spy(RunManager):
        def _execute(self, handle):
            seen.append(marker.get())
            super()._execute(handle)

    mgr = Spy(live_settings, modules={"p0": Tiny(), "p1": Tiny()}, max_concurrent=1)
    try:
        marker.set("visible")
        mgr.start({"epochs": 1, "run_id": "ctx"})
        assert mgr.wait("ctx", timeout=60) == "completed"
    finally:
        mgr.shutdown(timeout=30.0)
    assert seen == ["visible"]


def test_tree_is_the_trainers_tree(live_settings):
    live_settings["TRAINER"]["MUTATION"] = {"ENABLED": True}
    mgr = RunManager(live_settings, modules={"p0": Tiny(), "p1": Tiny(), "p2": Tiny()}, max_concurrent=1)
    try:
        mgr.start({"epochs": 1, "run_id": "tr"})
        assert mgr.wait("tr", timeout=60) == "completed"
        handle = mgr.get_handle("tr")
        trainer = handle.engine.trainer
        assert mgr.tree("tr") is trainer.tree is handle.engine.tree
        trainer.retire_peer("p2")
        assert mgr.tree("tr")["p2"].status == "pruned" and mgr.tree("tr") is trainer.tree
    finally:
        mgr.shutdown(timeout=30.0)


def test_a_mutation_is_published_on_the_bus(live_settings):
    live_settings["TRAINER"]["MUTATION"] = {
        "ENABLED": True,
        "SCHEDULE": [{"AT_EPOCH": 1, "OPS": [{"op": "retire", "path": "p2"}]}],
    }
    mgr = RunManager(live_settings, modules={"p0": Tiny(), "p1": Tiny(), "p2": Tiny()}, max_concurrent=1)
    try:
        mgr.start({"epochs": 3, "run_id": "mut"})
        assert mgr.wait("mut", timeout=60) == "completed"
        trees = [e for e in mgr.bus.history("mut") if e["event"] == "tree"]
        assert len(trees) == 1 and trees[0]["data"]["retired"] == ["p2"] and trees[0]["data"]["source"] == "schedule"
    finally:
        mgr.shutdown(timeout=30.0)


def test_cancel_stops_at_the_boundary(live_settings):
    """T8: the epoch in flight is recorded; the summary says ``cancelled``."""
    live_settings["TRAINING"]["EPOCHS"] = 50
    mgr = RunManager(live_settings, modules={"p0": Tiny(), "p1": Tiny()}, max_concurrent=1, cancel_grace=30.0)
    try:
        handle = mgr.start({"epochs": 50, "run_id": "cx"})
        deadline = time.time() + 60
        while handle.epochs_completed < 1 and time.time() < deadline:
            time.sleep(0.01)
        mgr.cancel("cx", wait=True)
        assert mgr.wait("cx", timeout=60) == "cancelled"
        done = handle.epochs_completed
        assert 1 <= done < 50 and handle.summary["status"] == "cancelled"
        assert mgr.registry.count_episodes("cx") == done                   # no half-written epoch row
        assert mgr.registry.get_run("cx")["status"] == "cancelled"
    finally:
        mgr.shutdown(timeout=30.0)
