"""Fixtures for the API app: a settings object with the app installed and a WSGI client."""

from __future__ import annotations

import pytest

from mactf import apps as apps_module
from mactf.contrib.api import deps
from mactf.http.handler import WSGIHandler, clear_application_cache

from api_support import ApiClient, build_settings, wait_terminal


@pytest.fixture()
def api_settings(settings):
    yield build_settings(settings)
    deps.clear_fallback()
    clear_application_cache()
    apps_module.reset_registry()


@pytest.fixture()
def app(api_settings):
    handler = WSGIHandler()
    yield handler
    manager = handler.state.get("run_manager")
    if manager is not None:
        manager.close()
    registry = handler.state.get("registry")
    if registry is not None and hasattr(registry, "close"):
        registry.close()


@pytest.fixture()
def client(app):
    return ApiClient(app)


@pytest.fixture()
def points_client(api_settings):
    api_settings["COOPETITION"]["SIGNAL_SOURCE"] = "points"
    handler = WSGIHandler()
    client = ApiClient(handler)
    created = client.post("/runs", json={"epochs": 2, "run_id": "pts"})
    assert created.status_code == 202, created.text
    final = wait_terminal(client, "pts")
    assert final["status"] == "completed", final
    yield client
    manager = handler.state.get("run_manager")
    if manager is not None:
        manager.close()


@pytest.fixture()
def finished(client):
    created = client.post("/runs", json={"epochs": 2, "run_id": "full"})
    assert created.status_code == 202, created.text
    final = wait_terminal(client, "full")
    assert final["status"] == "completed", final
    return "full"
