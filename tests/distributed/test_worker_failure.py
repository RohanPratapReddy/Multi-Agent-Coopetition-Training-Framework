"""A dropped worker contributes zero weight, never a default (document 05 section 7)."""

from __future__ import annotations

import time

import pytest

from mactf.client import HTTPTransport, MACTFClient
from mactf.distributed.coordinator import get_coordinator
from mactf.distributed.protocol import TaskResult
from mactf.distributed.worker import Worker, WorkerClient
from mactf.exceptions import DistributedError


class CrashingWorker(Worker):
    """Takes its assignment, then dies: no result, no deregistration, no more heartbeats."""

    def step(self):
        assignment = self.client.tasks(self.worker_id, wait=self.poll_wait)
        if assignment is None:
            return False
        self._deregister = False
        self._stop.set()
        return True


class WrongPeersWorker(Worker):
    """Returns a result for peers it was not assigned."""

    def run_epoch(self, assignment, peers):
        good = super().run_epoch(assignment, peers)
        return TaskResult(
            task_id=good.task_id, worker_id=good.worker_id, peer_names=("imposter",),
            parameters_b64=good.parameters_b64, checksum=good.checksum, metrics=good.metrics,
            ok=True, epoch=good.epoch, run_id=good.run_id,
        )


class CorruptingWorker(Worker):
    """Returns a blob whose checksum does not match."""

    def run_epoch(self, assignment, peers):
        good = super().run_epoch(assignment, peers)
        return TaskResult(
            task_id=good.task_id, worker_id=good.worker_id, peer_names=good.peer_names,
            parameters_b64=good.parameters_b64, checksum="0" * 64, metrics=good.metrics,
            ok=True, epoch=good.epoch, run_id=good.run_id,
        )


def _incidents(api, run_id):
    return [e for e in api.runs.stream(run_id) if e.event == "incident"]


def _run(failing_server, run_id):
    api = MACTFClient(transport=HTTPTransport(failing_server, token="all-tok", timeout=60))
    run = api.run_to_completion(run_id=run_id, epochs=1, timeout=120)
    assert run.status == "completed", run.notes
    return api, run


def test_a_crashed_worker_gets_zero_weight_and_an_incident(failing_pool, failing_server):
    failing_pool.add("w1")
    failing_pool.add("w2")
    failing_pool.add("w3", cls=CrashingWorker)  # plan: w1 -> p0, w2 -> p1, w3 -> p2
    failing_pool.wait_registered(3)
    api, run = _run(failing_server, "crash")

    assert run.status == "completed"  # the epoch proceeds with the results it has
    report = api.runs.aggregations("crash")[0]
    assert report.weights["p2"] == 0.0
    assert report.weights["p0"] + report.weights["p1"] == pytest.approx(1.0)
    assert "p2" in report.gated_peers
    incidents = _incidents(api, "crash")
    reasons = " ".join(str(e.data.get("reason")) for e in incidents)
    assert "w3" in reasons and "p2" in reasons
    assert any(e.data.get("workers") == ["w3"] and e.data.get("peers") == ["p2"] for e in incidents)
    assert "w3" not in [m["worker_id"] for m in get_coordinator().members()]  # reaped


def test_a_wrong_peer_set_is_rejected_and_the_worker_marked_suspect(failing_pool, failing_server):
    failing_pool.add("w1")
    failing_pool.add("w2")
    failing_pool.add("w3", cls=WrongPeersWorker)
    failing_pool.wait_registered(3)
    api, run = _run(failing_server, "wrong")
    assert run.status == "completed"
    assert api.runs.aggregations("wrong")[0].weights["p2"] == 0.0
    reasons = " ".join(str(e.data.get("reason")) for e in _incidents(api, "wrong"))
    assert "returned peers" in reasons and "imposter" in reasons
    states = {m["worker_id"]: m["status"] for m in get_coordinator().members()}
    assert states.get("w3") == "suspect"


def test_a_corrupt_result_is_rejected_by_checksum(failing_pool, failing_server):
    failing_pool.add("w1")
    failing_pool.add("w2")
    failing_pool.add("w3", cls=CorruptingWorker)
    failing_pool.wait_registered(3)
    api, run = _run(failing_server, "corrupt")
    assert run.status == "completed"
    assert api.runs.aggregations("corrupt")[0].weights["p2"] == 0.0
    assert "checksum" in " ".join(str(e.data.get("reason")) for e in _incidents(api, "corrupt"))


def test_every_worker_lost_falls_back_without_hanging(failing_pool, failing_server):
    failing_pool.add("w1", cls=CrashingWorker, max_peers=3)
    failing_pool.wait_registered(1)
    api, run = _run(failing_server, "alllost")
    assert run.status == "completed"
    report = api.runs.aggregations("alllost")[0]
    assert all(weight == 0.0 for weight in report.weights.values())
    assert report.fallback is True


def test_a_worker_that_returns_after_the_epoch_is_refused(failing_pool, failing_server):
    for worker_id in ("w1", "w2", "w3"):
        failing_pool.add(worker_id)
    failing_pool.wait_registered(3)
    api, run = _run(failing_server, "late")
    assert run.status == "completed"
    client = WorkerClient(failing_server, token="worker-tok")
    late = TaskResult("nope", "w1", ("p0",), "", "", ok=True, epoch=0, run_id="late")
    from mactf.client.errors import Conflict

    with pytest.raises(Conflict):
        client.results(late)


def test_unassigned_peers_fail_the_run_loudly(failing_pool, failing_server):
    # three peers, one worker with room for one: the distributed analogue of require_same_layout failing
    failing_pool.add("w1", max_peers=1)
    failing_pool.wait_registered(1)
    api = MACTFClient(transport=HTTPTransport(failing_server, token="all-tok", timeout=60))
    run = api.run_to_completion(run_id="toomany", epochs=1, timeout=120)
    assert run.status == "failed"
    assert "unassigned" in (api.runs.get("toomany").notes or "").lower() or "capacity" in (run.notes or "")
