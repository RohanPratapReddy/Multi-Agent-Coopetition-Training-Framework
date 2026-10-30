"""Acceptance criterion 12: two workers reproduce the single-host aggregate exactly."""

from __future__ import annotations

import pytest
import torch

from mactf.client import Forbidden, HTTPTransport, MACTFClient, Unauthorized
from mactf.db import ensure_registry
from mactf.distributed.coordinator import get_coordinator
from mactf.orchestration.orchestrator import Orchestrator, RunConfig


def _run(settings, run_id, epochs=1):
    orchestrator = Orchestrator(settings, registry=ensure_registry(settings))
    config = RunConfig(epochs=epochs, run_id=run_id)
    summary = orchestrator.run(config)
    return orchestrator, summary


def _vectors(orchestrator):
    return {name: peer.adapter.export_parameters().detach().clone() for name, peer in orchestrator.peers.items()}


def test_aggregated_vector_equals_the_single_host_vector(pool, dist_settings):
    single, _ = _run(dist_settings, "single")
    assert not get_coordinator().active()  # no workers yet: the untouched single-host path

    pool.add("w1", max_peers=1)
    pool.add("w2", max_peers=1)
    pool.wait_registered(2)
    distributed, _ = _run(dist_settings, "dist")

    expected, actual = _vectors(single), _vectors(distributed)
    assert set(expected) == set(actual) == {"p0", "p1"}
    for name in expected:
        assert torch.allclose(expected[name], actual[name], atol=1e-9, rtol=0.0), name
    # aggregation makes every peer identical: the same vector everywhere
    assert torch.allclose(actual["p0"], actual["p1"], atol=1e-9, rtol=0.0)
    reference = single.reference.adapter.export_parameters()
    assert torch.allclose(reference, distributed.reference.adapter.export_parameters(), atol=1e-9, rtol=0.0)

    # each worker really did one of the two local updates
    assert sorted(w.completed_epochs for w in pool.workers) == [1, 1]
    for report_single, report_dist in zip(single.context.reports, distributed.context.reports):
        assert report_single.weights == pytest.approx(report_dist.weights, abs=1e-9)


def test_workers_load_the_aggregated_vector_after_the_exchange(pool, dist_settings):
    pool.add("w1", max_peers=1)
    pool.add("w2", max_peers=1)
    pool.wait_registered(2)
    distributed, _ = _run(dist_settings, "sync")
    # let the workers finish applying the exchange they were waiting on
    import time

    deadline = time.time() + 10
    while time.time() < deadline and any(w._busy for w in pool.workers):
        time.sleep(0.05)
    aggregated = distributed.peers["p0"].adapter.export_parameters()
    for worker in pool.workers:
        for peer in worker.peers.values():
            assert torch.allclose(peer.adapter.export_parameters(), aggregated, atol=1e-9, rtol=0.0)
        assert worker.reference is not None


def test_one_worker_can_hold_both_peers(pool, dist_settings):
    pool.add("solo", max_peers=2)
    pool.wait_registered(1)
    single_orch, _ = _run(dist_settings, "multi")
    assert sorted(pool.workers[0].peers) == ["p0", "p1"]
    assert single_orch.context.metrics["distributed"]["workers"] == ["solo"]


def test_worker_routes_demand_the_worker_scope(server, dist_settings):
    read = MACTFClient(transport=HTTPTransport(server, token="read-tok", probe=False, prefix="/api/v1"))
    from mactf.distributed.worker import WorkerClient
    from mactf.client.errors import Forbidden as _Forbidden, Unauthorized as _Unauthorized
    from mactf.distributed.protocol import WorkerRegistration

    registration = WorkerRegistration("w9", "h", {}, 1)
    with pytest.raises(_Forbidden):
        WorkerClient(server, token="read-tok").register(registration)
    with pytest.raises(_Unauthorized):
        WorkerClient(server).register(registration)
    # and a worker token cannot read scorecards or start runs
    worker_api = MACTFClient(transport=HTTPTransport(server, token="worker-tok", probe=False, prefix="/api/v1"))
    with pytest.raises(_Forbidden):
        worker_api.runs.create(epochs=1)
    assert read.runs.list().total == 0


def test_spoofed_identity_is_a_409(server, dist_settings):
    from mactf.client.errors import Conflict
    from mactf.distributed.protocol import WorkerRegistration
    from mactf.distributed.worker import WorkerClient

    client = WorkerClient(server, token="worker-tok")
    client.register(WorkerRegistration("w1", "alpha", {}, 1))
    with pytest.raises(Conflict):
        client.register(WorkerRegistration("w1", "beta", {}, 1))
    client.register(WorkerRegistration("w2", "alpha", {}, 1))
    client.deregister("w1")
    client.deregister("w2")


def test_run_through_the_api_uses_the_workers(pool, dist_settings, server):
    pool.add("w1", max_peers=1)
    pool.add("w2", max_peers=1)
    pool.wait_registered(2)
    api = MACTFClient(transport=HTTPTransport(server, token="all-tok", timeout=60))
    run = api.run_to_completion(run_id="viaapi", epochs=1, timeout=120)
    assert run.status == "completed"
    assert sorted(w.completed_epochs for w in pool.workers) == [1, 1]
    assert api.runs.aggregations("viaapi").total == 1
