"""The API's run routes under ``ORCHESTRATOR.ENGINE: trainer`` (row 30)."""

from __future__ import annotations

import pytest

from api_support import ApiClient, body, wait_terminal
from mactf.http.handler import WSGIHandler


@pytest.fixture()
def trainer_client(api_settings):
    api_settings["ORCHESTRATOR"]["ENGINE"] = "trainer"
    handler = WSGIHandler()
    yield ApiClient(handler), handler
    manager = handler.state.get("run_manager")
    if manager is not None:
        manager.close()


def test_peers_route_reads_the_trainer(trainer_client):
    client, handler = trainer_client
    assert client.post("/runs", json={"epochs": 2, "run_id": "tapi"}).status_code == 202
    assert wait_terminal(client, "tapi")["status"] == "completed"
    handle = handler.state["run_manager"].get_handle("tapi")
    assert type(handle.orchestrator).__name__ == "TrainerEngine"
    tree = body(client.get("/runs/tapi/peers/tree"))
    assert sorted(node["path"] for node in tree["nodes"]) == ["p0", "p1"]
    node = body(client.get("/runs/tapi/peers/p0"))
    assert node["path"] == "p0"
    # the adapters the view reads are the trainer's own
    assert handle.orchestrator.peers["p0"].adapter is handle.orchestrator.trainer.peers["p0"].adapter


def test_a_trainer_run_has_the_same_resources_as_an_orchestrator_run(trainer_client):
    client, _handler = trainer_client
    client.post("/runs", json={"epochs": 2, "run_id": "res"})
    assert wait_terminal(client, "res")["status"] == "completed"
    assert body(client.get("/runs/res/episodes"))["total"] == 2
    assert client.get("/runs/res/episodes/0/peers/p0").status_code == 200
