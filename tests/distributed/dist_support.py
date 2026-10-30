"""Helpers shared by the distributed tests (a uniquely named module: conftest names collide)."""

from __future__ import annotations

import threading
import time

from mactf import apps as apps_module
from mactf import conf
from mactf.contrib.api import deps
from mactf.distributed.coordinator import get_coordinator, reset_coordinator
from mactf.distributed.worker import Worker
from mactf.http import urls as http_urls
from mactf.http.handler import clear_application_cache, get_wsgi_application

TOKENS = {
    "all-tok": ["read", "run:write", "score:write", "sandbox:exec", "admin"],
    "worker-tok": ["worker"],
    "read-tok": ["read"],
}


def configure(settings, *, heartbeat_timeout=30.0, task_timeout=60.0, distributed=True, peers=2):
    from dist_peers import DistTinyPeer

    apps = [a for a in settings.get("INSTALLED_APPS", []) if a not in ("mactf.contrib.api", "mactf.distributed")]
    apps.append("mactf.contrib.api")
    if distributed:
        apps.append("mactf.distributed")
    settings["INSTALLED_APPS"] = apps
    api = dict(settings.get("API") or {})
    api["TOKENS"] = dict(TOKENS)
    settings["API"] = api
    block = dict(settings.get("DISTRIBUTED") or {})
    block.update(WORKER_HEARTBEAT_TIMEOUT=heartbeat_timeout, WORKER_TASK_TIMEOUT=task_timeout)
    settings["DISTRIBUTED"] = block
    settings["PEERS"] = [{"name": f"p{i}", "class": "DistTinyPeer", "kwargs": {}} for i in range(peers)]
    settings["TRAINING"].update(
        {"EPOCHS": 1, "BATCH_SIZE": 8, "STATE_DIM": 4, "ACTION_DIM": 3, "CHECKPOINT_EVERY": 1}
    )
    settings["DEVICE"] = "cpu"
    settings["SANDBOX"]["ALLOW_FALLBACK"] = True
    settings["SANDBOX"]["PROVIDER"] = "local"
    conf._settings = settings
    apps_module.reset_registry()
    apps_module.populate()
    apps_module.get_peer_registry().register(DistTinyPeer)
    if hasattr(http_urls, "clear_url_caches"):
        http_urls.clear_url_caches()
    clear_application_cache()
    deps.clear_fallback()
    reset_coordinator()
    return settings


def teardown():
    application = get_wsgi_application()
    manager = getattr(application, "state", {}).get("run_manager")
    if manager is not None:
        manager.close()
    reset_coordinator()
    deps.clear_fallback()
    clear_application_cache()
    apps_module.reset_registry()




def start_server(workers=16):
    from mactf.contrib.api.management.commands.serve import build_server

    httpd = build_server("127.0.0.1", 0, workers=workers, quiet=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread, f"http://{httpd.server_address[0]}:{httpd.server_address[1]}"


class WorkerPool:
    def __init__(self, url, settings):
        self.url, self.settings, self.workers = url, settings, []

    def add(self, worker_id, cls=Worker, **kwargs):
        worker = cls(
            coordinator_url=self.url, token="worker-tok", worker_id=worker_id, settings=self.settings,
            device="cpu", heartbeat_interval=0.1, poll_wait=0.5, **kwargs,
        )
        worker.start()
        self.workers.append(worker)
        return worker

    def wait_registered(self, count, timeout=20.0):
        deadline = time.time() + timeout
        coordinator = get_coordinator()
        while time.time() < deadline:
            if len(coordinator.members()) >= count:
                return
            time.sleep(0.05)
        raise AssertionError(f"only {len(get_coordinator().members())} of {count} workers registered")

    def stop(self):
        for worker in self.workers:
            worker.shutdown(timeout=15)
