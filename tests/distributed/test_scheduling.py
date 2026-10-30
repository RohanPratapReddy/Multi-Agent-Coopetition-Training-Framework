"""Membership and sharding on the coordinator alone (no sockets)."""

from __future__ import annotations

import pytest

from mactf.distributed.coordinator import Coordinator, UnknownWorker, WorkerConflict
from mactf.distributed.protocol import WorkerHeartbeat, WorkerRegistration
from mactf.exceptions import DistributedError


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def make(settings, **kwargs):
    clock = Clock()
    return Coordinator(settings=settings, clock=clock, **kwargs), clock


def reg(coordinator, worker_id, *, max_peers=1, host=None):
    return coordinator.register(
        WorkerRegistration(worker_id, host or f"host-{worker_id}", {"device": "cpu"}, max_peers=max_peers)
    )


def test_three_peers_two_workers_gives_two_and_one(settings):
    coordinator, _ = make(settings)
    reg(coordinator, "w1", max_peers=2)
    reg(coordinator, "w2", max_peers=2)
    assert coordinator.plan(0, ["p1", "p2", "p3"]) == {"w1": ["p1", "p2"], "w2": ["p3"]}


def test_plan_is_deterministic_and_independent_of_registration_order(settings):
    one, _ = make(settings)
    two, _ = make(settings)
    for worker in ("w2", "w1", "w3"):
        reg(one, worker, max_peers=2)
    for worker in ("w3", "w1", "w2"):
        reg(two, worker, max_peers=2)
    peers = [f"p{i}" for i in range(5)]
    assert one.plan(1, peers) == two.plan(7, peers)


def test_plan_respects_max_peers_by_pushing_the_overflow_on(settings):
    coordinator, _ = make(settings)
    reg(coordinator, "w1", max_peers=1)
    reg(coordinator, "w2", max_peers=3)
    plan = coordinator.plan(0, ["a", "b", "c", "d"])
    assert plan == {"w1": ["a"], "w2": ["b", "c", "d"]}
    assert all(len(v) <= {"w1": 1, "w2": 3}[k] for k, v in plan.items())


def test_unassigned_peers_raise_naming_peers_and_capacity(settings):
    coordinator, _ = make(settings)
    reg(coordinator, "w1", max_peers=1)
    with pytest.raises(DistributedError) as caught:
        coordinator.plan(0, ["a", "b", "c"])
    assert "total capacity 1" in str(caught.value) and "b" in str(caught.value) and "c" in str(caught.value)
    with pytest.raises(DistributedError):
        make(settings)[0].plan(0, ["a"])


def test_same_id_from_another_host_is_rejected_but_two_ids_per_host_are_fine(settings):
    coordinator, _ = make(settings)
    reg(coordinator, "w1", host="alpha")
    with pytest.raises(WorkerConflict):
        reg(coordinator, "w1", host="beta")
    reg(coordinator, "w2", host="alpha")
    reg(coordinator, "w1", host="alpha")  # the same worker coming back is not a conflict
    assert [m["worker_id"] for m in coordinator.members()] == ["w1", "w2"]


def test_max_workers_is_enforced(settings):
    coordinator, _ = make(settings, max_workers=2)
    reg(coordinator, "w1")
    reg(coordinator, "w2")
    with pytest.raises(WorkerConflict):
        reg(coordinator, "w3")


def test_missed_heartbeats_reap_a_worker_and_discard_its_tasks(settings):
    coordinator, clock = make(settings, heartbeat_timeout=30.0)
    reg(coordinator, "w1")
    reg(coordinator, "w2")
    clock.now += 20
    coordinator.heartbeat(WorkerHeartbeat("w1", 0))
    clock.now += 20  # w2 is now 40 s stale, w1 only 20 s
    assert coordinator.reap_stale() == ["w2"]
    assert [m["worker_id"] for m in coordinator.members()] == ["w1"]
    with pytest.raises(UnknownWorker):
        coordinator.heartbeat(WorkerHeartbeat("w2", 0))
    assert coordinator.active()


def test_active_only_with_live_members(settings):
    coordinator, clock = make(settings, heartbeat_timeout=5.0)
    assert not coordinator.active()
    reg(coordinator, "w1")
    assert coordinator.active()
    clock.now += 6
    assert not coordinator.active()


def test_deregister_removes_and_unknown_raises(settings):
    coordinator, _ = make(settings)
    reg(coordinator, "w1")
    coordinator.deregister("w1")
    with pytest.raises(UnknownWorker):
        coordinator.deregister("w1")
    assert coordinator.members() == []
