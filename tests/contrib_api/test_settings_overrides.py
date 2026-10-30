"""../contrib api/test settings overrides (split from test_app_extensions.py)."""

from __future__ import annotations

from mactf import apps as apps_module

import sys


import pytest


from mactf import conf


from mactf.exceptions import CommandError, ImproperlyConfigured


def test_new_settings_are_declared_and_overridable(tmp_path, monkeypatch):
    module = type(sys)("mactf_override_settings")
    module.BASE_DIR = str(tmp_path)
    module.CONFIG_FILES = []
    module.DEVICE = "cpu"
    module.PEERS = []
    module.ROOT_URLCONF = "project.urls"
    module.APPEND_SLASH = False
    module.MIDDLEWARE = ["a.b.C"]
    module.HTTP = {"STREAM_WORKERS": 4}
    module.API = {"BASE_PATH": "svc/v2", "TOKENS": {"t": {"scopes": ["runs:read"]}}}
    sys.modules["mactf_override_settings"] = module
    try:
        loaded = conf.load_settings("mactf_override_settings")
    finally:
        del sys.modules["mactf_override_settings"]
    assert loaded["ROOT_URLCONF"] == "project.urls"
    assert loaded["APPEND_SLASH"] is False
    assert loaded["MIDDLEWARE"] == ["a.b.C"]
    assert loaded["HTTP"]["STREAM_WORKERS"] == 4 and loaded["HTTP"]["MAX_BODY_BYTES"] == 8388608
    assert loaded["API"]["BASE_PATH"] == "svc/v2" and loaded["API"]["EVENT_BUFFER"] == 1024
    assert loaded["DEFAULT_AUTO_FIELD"] == "mactf.db.BigAutoField"


@pytest.mark.parametrize(
    "key,value",
    [
        ("ROOT_URLCONF", ""),
        ("MIDDLEWARE", "not-a-list"),
        ("API", {"BASE_PATH": "/api/"}),
        ("HTTP", {"MAX_BODY_BYTES": 0}),
        ("HTTP", {"STREAM_WORKERS": 999}),
        ("HTTP", {"SSE_HEARTBEAT": 0}),
    ],
)
def test_invalid_http_settings_are_rejected(tmp_path, key, value):
    module = type(sys)("mactf_bad_settings")
    module.BASE_DIR = str(tmp_path)
    module.CONFIG_FILES = []
    module.DEVICE = "cpu"
    module.PEERS = []
    setattr(module, key, value)
    sys.modules["mactf_bad_settings"] = module
    try:
        with pytest.raises(ImproperlyConfigured):
            conf.load_settings("mactf_bad_settings")
    finally:
        del sys.modules["mactf_bad_settings"]


def test_project_overrides_reach_the_running_api(settings):
    from mactf.contrib.api import deps  # noqa: F401
    from mactf.http import urls as http_urls
    from mactf.http.handler import WSGIHandler, clear_application_cache
    from mactf.http.testing import dispatch_wsgi

    from api_support import body, build_settings

    build_settings(settings, BASE_PATH="svc/v9", ALLOW_ANONYMOUS=False)
    http_urls.clear_url_caches()
    clear_application_cache()
    app = WSGIHandler()
    try:
        anonymous = dispatch_wsgi(app, "GET", "/svc/v9/rules")
        assert anonymous.status_code == 401
        authed = dispatch_wsgi(app, "GET", "/svc/v9/rules", headers={"Authorization": "Bearer read-tok"})
        assert authed.status_code == 200
        assert len(body(authed)) == 40
        assert dispatch_wsgi(app, "GET", "/api/v1/rules").status_code == 404
    finally:
        manager = app.state.get("run_manager")
        if manager is not None:
            manager.close()
        apps_module.reset_registry()
        clear_application_cache()


def test_the_concurrency_override_raises_the_run_cap(settings):
    from mactf.http import urls as http_urls
    from mactf.http.handler import WSGIHandler, clear_application_cache

    from api_support import ApiClient, build_settings, wait_terminal

    build_settings(settings, MAX_CONCURRENT_RUNS=2)
    settings["TRAINING"]["EPOCHS"] = 200
    http_urls.clear_url_caches()
    clear_application_cache()
    app = WSGIHandler()
    client = ApiClient(app)
    try:
        assert client.post("/runs", json={"epochs": 200, "run_id": "a"}).status_code == 202
        assert client.post("/runs", json={"epochs": 200, "run_id": "b"}).status_code == 202
        assert client.post("/runs", json={"epochs": 1, "run_id": "c"}).status_code == 429
        for run_id in ("a", "b"):
            client.post(f"/runs/{run_id}/cancel")
            wait_terminal(client, run_id)
    finally:
        manager = app.state.get("run_manager")
        if manager is not None:
            manager.close()
        apps_module.reset_registry()
        clear_application_cache()
