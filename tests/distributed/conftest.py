"""Fixtures for distributed runs: the API plus worker routes on a real socket, and workers in threads."""

from __future__ import annotations

import pytest

from dist_support import WorkerPool, configure, start_server, teardown


@pytest.fixture()
def dist_settings(settings):
    yield configure(settings)
    teardown()


@pytest.fixture()
def fast_failure_settings(settings):
    yield configure(settings, heartbeat_timeout=0.6, task_timeout=30.0, peers=3)
    teardown()


@pytest.fixture()
def server(dist_settings):
    httpd, thread, url = start_server()
    yield url
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


@pytest.fixture()
def failing_server(fast_failure_settings):
    httpd, thread, url = start_server()
    yield url
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


@pytest.fixture()
def pool(server, dist_settings):
    workers = WorkerPool(server, dist_settings)
    yield workers
    workers.stop()


@pytest.fixture()
def failing_pool(failing_server, fast_failure_settings):
    workers = WorkerPool(failing_server, fast_failure_settings)
    yield workers
    workers.stop()
