"""Helpers shared by the API tests (a uniquely named module: conftest names collide)."""

from __future__ import annotations

import json as _json

from mactf import apps as apps_module
from mactf import conf
from mactf.contrib.api import deps
from mactf.http import urls as http_urls
from mactf.http.handler import WSGIHandler, clear_application_cache
from mactf.http.testing import dispatch_wsgi

TOKENS = {
    "all-tok": ["read", "run:write", "score:write", "sandbox:exec", "admin"],
    "read-tok": ["read"],
    "write-tok": ["read", "run:write"],
}


class ApiClient:
    """A tiny client over the WSGI app; ``prefix`` is the mounted base path."""

    def __init__(self, app, prefix="/api/v1"):
        self.app = app
        self.prefix = prefix
        self.token: str | None = "all-tok"

    def request(self, method, path, *, token="__default__", headers=None, **kwargs):
        merged = dict(headers or {})
        token = self.token if token == "__default__" else token
        if token:
            merged["Authorization"] = f"Bearer {token}"
        target = path if path.startswith("/") and path.startswith(self.prefix) else f"{self.prefix}{path}"
        return dispatch_wsgi(self.app, method, target, headers=merged, **kwargs)

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path, json=None, **kwargs):
        if json is not None:
            kwargs["json"] = json
        return self.request("POST", path, **kwargs)


def build_settings(settings, **api):
    settings["INSTALLED_APPS"] = [*settings.get("INSTALLED_APPS", []), "mactf.contrib.api"]
    merged = dict(settings.get("API") or {})
    merged["TOKENS"] = dict(TOKENS)
    merged.update(api)
    settings["API"] = merged
    settings["PEERS"] = [
        {"name": "p0", "class": "ApiTinyPeer", "kwargs": {}},
        {"name": "p1", "class": "ApiTinyPeer", "kwargs": {}},
    ]
    settings["TRAINING"].update(
        {"EPOCHS": 2, "BATCH_SIZE": 8, "STATE_DIM": 4, "ACTION_DIM": 3, "CHECKPOINT_EVERY": 1}
    )
    settings["SANDBOX"]["ALLOW_FALLBACK"] = True
    settings["SANDBOX"]["PROVIDER"] = "local"
    conf._settings = settings
    apps_module.reset_registry()
    apps_module.populate()
    from api_peers import ApiTinyPeer

    apps_module.get_peer_registry().register(ApiTinyPeer)
    http_urls.clear_url_caches()
    clear_application_cache()
    deps.clear_fallback()
    return settings




def body(response):
    return _json.loads(response.content.decode("utf-8"))


def wait_terminal(client, run_id, timeout=60.0):
    import time

    deadline = time.time() + timeout
    payload = {}
    while time.time() < deadline:
        payload = body(client.get(f"/runs/{run_id}"))
        if payload["status"] in ("completed", "failed", "cancelled", "dry_run", "interrupted"):
            return payload
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish: {payload}")
