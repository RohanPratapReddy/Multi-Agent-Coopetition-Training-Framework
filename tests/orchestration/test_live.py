"""RunManager and EventBus (document 01 sections 3-4, criteria for B1)."""

from __future__ import annotations

import threading
import time

import pytest
import torch
import torch.nn as nn

from mactf.db import Registry
from mactf.orchestration.live import (
    CapacityExceeded,
    EventBus,
    RunExists,
    RunManager,
    RunNotFound,
    RunState,
    RunStateError,
    jsonable,
)


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 3)

    def forward(self, x):
        return self.net(x)


@pytest.fixture()
def live_settings(settings):
    settings["TRAINING"].update(
        {"EPOCHS": 3, "BATCH_SIZE": 8, "STATE_DIM": 4, "ACTION_DIM": 3, "CHECKPOINT_EVERY": 1}
    )
    settings["SANDBOX"].update({"ALLOW_PROVIDER": "local", "PROVIDER": "local"})
    return settings


@pytest.fixture()
def manager(live_settings):
    mgr = RunManager(
        live_settings,
        modules={"p0": Tiny(), "p1": Tiny()},
        max_concurrent=1,
        cancel_grace=30.0,
    )
    yield mgr
    mgr.shutdown(timeout=30.0)


# -- EventBus ----------------------------------------------------------------
def test_ids_are_monotonic_per_run():
    bus = EventBus(capacity=8)
    a = [bus.publish("r1", "run", {"n": i})["id"] for i in range(3)]
    b = bus.publish("r2", "run", {})["id"]
    assert a == [1, 2, 3] and b == 1


def test_unknown_event_name_is_rejected():
    with pytest.raises(ValueError):
        EventBus().publish("r", "bogus", {})


def test_replay_after_last_id_and_terminal_closes():
    bus = EventBus()
    for i in range(3):
        bus.publish("r", "epoch", {"epoch": i})
    bus.publish("r", "terminal", {"status": "completed"})
    got = [e["id"] for e in bus.subscribe("r", last_id=2)]
    assert got == [3, 4]
    assert [e["event"] for e in bus.subscribe("r")][-1] == "terminal"


def test_stream_lost_when_id_is_older_than_the_buffer():
    bus = EventBus(capacity=3)
    for i in range(6):
        bus.publish("r", "epoch", {"epoch": i})
    bus.publish("r", "terminal", {"status": "completed"})
    events = list(bus.subscribe("r", last_id=1))
    assert events[0]["event"] == "stream_lost"
    assert events[0]["data"]["oldest"] == bus.oldest_id("r")
    ids = [e["id"] for e in events[1:]]
    assert ids == sorted(ids) and ids[-1] == bus.latest_id("r")


def test_heartbeat_after_silence_and_blocking_wakeup():
    bus = EventBus(heartbeat=0.05)
    sub = bus.subscribe("r")
    assert sub.next()["event"] == "heartbeat"
    threading.Timer(0.05, lambda: bus.publish("r", "run", {"x": 1})).start()
    event = sub.next(timeout=5)
    assert event["event"] == "run" and event["data"]["x"] == 1


def test_cancelled_subscription_ends_a_blocked_reader():
    bus = EventBus(heartbeat=30)
    sub = bus.subscribe("r")
    out = []
    thread = threading.Thread(target=lambda: out.append(sub.next()))
    thread.start()
    time.sleep(0.05)
    sub.cancel()
    thread.join(5)
    assert out == [None]


def test_concurrent_publishers_never_duplicate_ids():
    bus = EventBus(capacity=2000)

    def burst():
        for _ in range(200):
            bus.publish("r", "run", {})

    threads = [threading.Thread(target=burst) for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    ids = [e["id"] for e in bus.history("r")]
    assert ids == list(range(1, 1001))


def test_old_closed_channels_are_evicted():
    bus = EventBus(max_runs=2)
    for name in ("a", "b", "c", "d"):
        bus.publish(name, "terminal", {"status": "completed"})
    assert not bus.has("a")
    assert bus.has("d")


def test_jsonable_handles_nan_paths_and_objects():
    class Thing:
        def to_dict(self):
            return {"x": float("nan"), "t": (1, 2)}

    assert jsonable({"a": Thing()}) == {"a": {"x": None, "t": [1, 2]}}


# -- state machine ----------------------------------------------------------------
def test_state_graph():
    assert RunState.QUEUED.can_become(RunState.RUNNING)
    assert RunState.RUNNING.can_become(RunState.PAUSED)
    assert RunState.PAUSED.can_become(RunState.RUNNING) or RunState.PAUSED.can_become(RunState.RESUMING)
    assert not RunState.COMPLETED.can_become(RunState.RUNNING)
    assert RunState.COMPLETED.terminal and not RunState.PAUSED.terminal


# -- manager ------------------------------------------------------------------------
def test_run_completes_and_streams_events(manager):
    handle = manager.start({"epochs": 2})
    sub = manager.subscribe(handle.run_id)
    names = [e["event"] for e in sub if e["event"] != "heartbeat"]
    assert names[0] == "run" and names[-1] == "terminal"
    assert names.count("epoch") == 2
    assert "phase" in names
    assert manager.wait(handle.run_id, 30) == "completed"
    row = manager.registry.get_run(handle.run_id)
    assert row["status"] == "completed" and row["finished_at"]
    terminal = manager.bus.terminal_event(handle.run_id)
    assert terminal["data"]["status"] == "completed" and terminal["data"]["epochs"] == 2


def test_epoch_event_carries_the_documented_fields(manager):
    handle = manager.start({"epochs": 1})
    handle.wait(60)
    epoch = [e for e in manager.bus.history(handle.run_id) if e["event"] == "epoch"][0]
    assert {"run_id", "epoch", "score", "kappa", "entropy", "weights", "duration"} <= set(epoch["data"])


def test_row_is_visible_as_queued_before_the_thread_runs(live_settings):
    from mactf.orchestration.orchestrator import Orchestrator

    mgr = RunManager(
        live_settings,
        orchestrator_factory=lambda s, r: Orchestrator(
            s, modules={"p0": Tiny(), "p1": Tiny()}, registry=r
        ),
    )
    try:
        handle = mgr.start({"epochs": 1})
        # queue_run happens inside start(), before the thread has necessarily begun
        assert mgr.registry.get_run(handle.run_id) is not None
        handle.wait(60)
        assert handle.status == "completed"
    finally:
        mgr.shutdown(30)


def test_cancel_stops_at_an_epoch_boundary(manager):
    handle = manager.start({"epochs": 50})
    deadline = time.time() + 60
    while handle.epochs_completed < 1 and time.time() < deadline:
        time.sleep(0.02)
    result = manager.cancel(handle.run_id)
    assert result in ("cancelled", "cancelling")
    assert manager.wait(handle.run_id, 60) == "cancelled"
    assert 1 <= handle.epochs_completed < 50
    assert manager.registry.get_run(handle.run_id)["status"] == "cancelled"
    with pytest.raises(RunStateError) as info:
        manager.cancel(handle.run_id)
    assert info.value.current_status == "cancelled"


def test_cancel_unknown_run(manager):
    with pytest.raises(RunNotFound):
        manager.cancel("nope")


def test_capacity_is_bounded_and_released(manager):
    first = manager.start({"epochs": 40})
    with pytest.raises(CapacityExceeded) as info:
        manager.start({"epochs": 1})
    assert info.value.retry_after > 0
    manager.cancel(first.run_id)
    manager.wait(first.run_id, 60)
    second = manager.start({"epochs": 1})
    assert manager.wait(second.run_id, 60) == "completed"


def test_duplicate_run_id_is_rejected(manager):
    handle = manager.start({"epochs": 1, "run_id": "fixedid"})
    handle.wait(60)
    with pytest.raises(RunExists):
        manager.start({"epochs": 1, "run_id": "fixedid"})


def test_unknown_option_is_rejected(manager):
    with pytest.raises(Exception, match="unknown run option"):
        manager.start({"epochz": 1})


def test_prepare_failure_marks_the_run_failed(live_settings):
    mgr = RunManager(live_settings, modules={"p0": Tiny(), "p1": nn.Linear(9, 9)})
    try:
        handle = mgr.start({"epochs": 1})
        assert handle.wait(60)
        assert handle.status == "failed"
        assert handle.error
        assert mgr.registry.get_run(handle.run_id)["status"] == "failed"
        terminal = mgr.bus.terminal_event(handle.run_id)
        assert terminal["data"]["status"] == "failed"
    finally:
        mgr.shutdown(30)


def test_dry_run_is_terminal_without_epochs(manager):
    handle = manager.start({"dry_run": True})
    assert manager.wait(handle.run_id, 60) == "dry_run"
    assert manager.bus.terminal_event(handle.run_id)["data"]["status"] == "dry_run"


def test_pause_then_resume_keeps_the_run_id(manager):
    handle = manager.start({"epochs": 6})
    deadline = time.time() + 60
    while handle.epochs_completed < 1 and time.time() < deadline:
        time.sleep(0.02)
    manager.pause(handle.run_id)
    assert handle.wait(60)
    assert handle.status == "paused"
    assert manager.registry.get_run(handle.run_id)["status"] == "paused"
    assert manager.registry.get_run(handle.run_id)["finished_at"] is None
    resumed = manager.resume(handle.run_id)
    assert resumed.run_id == handle.run_id
    assert manager.wait(handle.run_id, 120) == "completed", handle.error


def test_cancel_a_paused_run_needs_no_thread(manager):
    handle = manager.start({"epochs": 30})
    while handle.epochs_completed < 1:
        time.sleep(0.02)
    manager.pause(handle.run_id)
    handle.wait(60)
    assert manager.cancel(handle.run_id) == "cancelled"
    assert manager.registry.get_run(handle.run_id)["status"] == "cancelled"


def test_resume_of_a_completed_run_is_a_conflict(manager):
    handle = manager.start({"epochs": 1})
    handle.wait(60)
    with pytest.raises(RunStateError):
        manager.resume(handle.run_id)
    with pytest.raises(RunStateError):
        manager.pause(handle.run_id)


def test_reap_orphans_rewrites_running_rows_but_keeps_paused(live_settings, tmp_path):
    registry = Registry(tmp_path / "orph.sqlite3")
    registry.migrate()
    for run_id, status in (("a", "running"), ("b", "queued"), ("c", "paused"), ("d", "completed")):
        registry.queue_run(run_id, settings={})
        registry.set_run_status(run_id, status, finished=status == "completed")
    mgr = RunManager(live_settings, registry=registry, modules={"p0": Tiny(), "p1": Tiny()})
    try:
        assert sorted(mgr.reap_orphans()) == ["a", "b"]
        assert registry.get_run("a")["status"] == "interrupted"
        assert registry.get_run("c")["status"] == "paused"
        assert registry.get_run("d")["status"] == "completed"
    finally:
        mgr.close()


def test_subscribe_to_a_stored_terminal_run_yields_its_outcome(live_settings, tmp_path):
    registry = Registry(tmp_path / "old.sqlite3")
    registry.migrate()
    registry.queue_run("old", settings={})
    registry.set_run_status("old", "completed")
    mgr = RunManager(live_settings, registry=registry, modules={"p0": Tiny(), "p1": Tiny()})
    try:
        events = list(mgr.subscribe("old"))
        assert [e["event"] for e in events] == ["terminal"]
        assert events[0]["data"]["status"] == "completed"
        with pytest.raises(RunNotFound):
            mgr.subscribe("missing")
    finally:
        mgr.close()


def test_signals_are_attributed_to_the_emitting_run_only(manager):
    from mactf import signals

    # a signal from an unrelated thread with no live run is dropped, not misfiled
    signals.integrity_incident.send(sender=None, reason="stray")
    assert manager.bus.history("anything") == []
    handle = manager.start({"epochs": 1})
    handle.wait(60)
    kinds = {e["event"] for e in manager.bus.history(handle.run_id)}
    assert {"run", "phase", "epoch", "terminal"} <= kinds


def test_shutdown_cancels_live_runs(live_settings):
    mgr = RunManager(live_settings, modules={"p0": Tiny(), "p1": Tiny()})
    handle = mgr.start({"epochs": 100})
    while handle.epochs_completed < 1:
        time.sleep(0.02)
    assert mgr.shutdown(timeout=60) == []
    assert handle.state.terminal
    assert torch is not None
