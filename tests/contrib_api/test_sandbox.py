"""test sandbox (split from test_catalog_and_tools.py)."""

from __future__ import annotations

import pytest

from api_support import ApiClient, body, wait_terminal
from mactf.http.handler import WSGIHandler


def test_sandbox_endpoints(client):
    assert body(client.get("/sandbox/providers"))
    status = client.get("/sandbox/status")
    assert status.status_code == 200
    probe = client.post("/sandbox/probe", json={})
    assert probe.status_code in (200, 202, 422), probe.text
    denied = client.post("/sandbox/exec", json={"command": ["echo", "hi"]}, token="write-tok")
    assert denied.status_code == 403


@pytest.fixture()
def qemu_client(api_settings):
    """A client whose app was built asking for qemu, with fallback allowed."""
    api_settings["SANDBOX"]["PROVIDER"] = "qemu"
    api_settings["SANDBOX"]["ALLOW_FALLBACK"] = True
    handler = WSGIHandler()
    yield ApiClient(handler)
    manager = handler.state.get("run_manager")
    if manager is not None:
        manager.close()


@pytest.fixture()
def strict_qemu_client(api_settings):
    api_settings["SANDBOX"]["PROVIDER"] = "qemu"
    api_settings["SANDBOX"]["ALLOW_FALLBACK"] = False
    handler = WSGIHandler()
    yield ApiClient(handler)
    manager = handler.state.get("run_manager")
    if manager is not None:
        manager.close()


def _skip_if_qemu_works(status):
    if status.get("provider") == "qemu" and status.get("usable"):
        pytest.skip("qemu is genuinely usable on this machine")


def test_requesting_an_unavailable_provider_degrades_honestly(qemu_client):
    status = body(qemu_client.get("/sandbox/status"))
    _skip_if_qemu_works(status)
    assert status["requested"] == "qemu"
    assert status["degraded"] is True
    assert status["provider"] != "qemu"
    assert status["isolation"] != "hardware"
    assert status["reasons"], "a degradation must say why"
    assert any("qemu" in reason.lower() for reason in status["reasons"])


def test_a_degraded_run_records_the_provider_it_really_used(qemu_client):
    qemu_client.post("/runs", json={"epochs": 1, "run_id": "degraded"})
    final = wait_terminal(qemu_client, "degraded")
    assert final["status"] == "completed", final
    _skip_if_qemu_works({"provider": final["sandbox"]["provider"], "usable": True})
    assert final["sandbox"]["provider"] != "qemu"
    assert final["sandbox"]["isolation"] == "process"
    assert final["sandbox"]["reasons"]


def test_without_fallback_an_unavailable_provider_is_a_503_problem(strict_qemu_client):
    response = strict_qemu_client.get("/sandbox/status")
    if response.status_code == 200:
        _skip_if_qemu_works(body(response))
    assert response.status_code == 503, response.text
    assert response.problem["type"] == "urn:mactf:error:sandbox-unavailable"
    assert "qemu" in response.problem["detail"].lower() or "isolation" in response.problem["detail"].lower()


def test_exec_needs_the_sandbox_scope_and_a_live_run(client):
    request = {"peer": "p0", "command": ["echo", "hi"]}
    assert client.post("/sandbox/exec", json=request, token="read-tok").status_code == 403
    refused = client.post("/sandbox/exec", json=request)
    assert refused.status_code == 409
    assert "no active run" in refused.problem["detail"]
