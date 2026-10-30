"""Fixtures for the client: one API app served in-process and, in a thread, over a real socket."""

from __future__ import annotations

import threading

import pytest

from mactf import apps as apps_module
from mactf import conf
from mactf.client import HTTPTransport, InProcessTransport, MACTFClient
from mactf.contrib.api import deps
from mactf.http import urls as http_urls
from mactf.http.handler import clear_application_cache, get_wsgi_application

TOKENS = {
    "all-tok": ["read", "run:write", "score:write", "sandbox:exec", "admin"],
    "read-tok": ["read"],
}


@pytest.fixture()
def api_settings(settings):
    from client_peers import ClientTinyPeer

    settings["INSTALLED_APPS"] = [*settings.get("INSTALLED_APPS", []), "mactf.contrib.api"]
    api = dict(settings.get("API") or {})
    api["TOKENS"] = dict(TOKENS)
    settings["API"] = api
    settings["PEERS"] = [
        {"name": "p0", "class": "ClientTinyPeer", "kwargs": {}},
        {"name": "p1", "class": "ClientTinyPeer", "kwargs": {}},
    ]
    settings["TRAINING"].update(
        {"EPOCHS": 2, "BATCH_SIZE": 8, "STATE_DIM": 4, "ACTION_DIM": 3, "CHECKPOINT_EVERY": 1}
    )
    settings["SANDBOX"]["ALLOW_FALLBACK"] = True
    settings["SANDBOX"]["PROVIDER"] = "local"
    conf._settings = settings
    apps_module.reset_registry()
    apps_module.populate()
    apps_module.get_peer_registry().register(ClientTinyPeer)
    if hasattr(http_urls, "clear_url_caches"):
        http_urls.clear_url_caches()
    clear_application_cache()
    deps.clear_fallback()
    yield settings
    application = get_wsgi_application()
    manager = getattr(application, "state", {}).get("run_manager")
    if manager is not None:
        manager.close()
    deps.clear_fallback()
    clear_application_cache()
    apps_module.reset_registry()


@pytest.fixture()
def inproc(api_settings):
    return MACTFClient(transport=InProcessTransport(token="all-tok"))


@pytest.fixture()
def server(api_settings):
    from mactf.contrib.api.management.commands.serve import build_server

    httpd = build_server("127.0.0.1", 0, workers=8, quiet=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    yield f"http://{host}:{port}"
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


@pytest.fixture()
def http(server):
    return MACTFClient(transport=HTTPTransport(server, token="all-tok", timeout=15, backoff=0.01))


@pytest.fixture(params=["inproc", "http"])
def both(request):
    """The same client assertions over each transport."""
    return request.getfixturevalue(request.param)
