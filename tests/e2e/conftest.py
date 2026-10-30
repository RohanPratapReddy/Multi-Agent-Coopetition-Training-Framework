"""One project configuration used by the CLI, the in-process API and a real socket."""

from __future__ import annotations

import pytest

from mactf import apps as apps_module
from mactf import conf
from mactf.contrib.api import deps
from mactf.http import urls as http_urls
from mactf.http.handler import clear_application_cache, get_wsgi_application

TOKENS = {"all-tok": ["read", "run:write", "score:write", "sandbox:exec", "admin"]}


@pytest.fixture()
def project(settings):
    from e2e_peers import E2EPeer

    settings["INSTALLED_APPS"] = [*settings.get("INSTALLED_APPS", []), "mactf.contrib.api"]
    api = dict(settings.get("API") or {})
    api["TOKENS"] = dict(TOKENS)
    settings["API"] = api
    settings["PEERS"] = [
        {"name": "p0", "class": "E2EPeer", "kwargs": {}},
        {"name": "p1", "class": "E2EPeer", "kwargs": {}},
    ]
    settings["TRAINING"].update({"EPOCHS": 2, "BATCH_SIZE": 8, "STATE_DIM": 4, "ACTION_DIM": 3})
    settings["SANDBOX"]["ALLOW_FALLBACK"] = True
    settings["SANDBOX"]["PROVIDER"] = "local"
    conf._settings = settings
    apps_module.reset_registry()
    apps_module.populate()
    apps_module.get_peer_registry().register(E2EPeer)
    http_urls.clear_url_caches()
    clear_application_cache()
    deps.clear_fallback()
    yield settings
    manager = getattr(get_wsgi_application(), "state", {}).get("run_manager")
    if manager is not None:
        manager.close()
    deps.clear_fallback()
    clear_application_cache()
    apps_module.reset_registry()
