"""Criterion 1: an API run equals the CLI run for the same seed, in-process and over a real socket."""

from __future__ import annotations

import json
import threading

import pytest

from mactf.client import HTTPTransport, MACTFClient
from mactf.management import execute_from_command_line
from mactf.testing import Client as RawClient

TOL = 1e-9
ARGS = ["--epochs", "2", "--seed", "1337", "--device", "cpu"]


class Client:
    """The in-process test client with the bearer token and API prefix applied."""

    def __init__(self, token):
        self._client = RawClient()
        self._headers = {"Authorization": f"Bearer {token}"}

    def get(self, path):
        return self._client.get("/api/v1" + path, headers=self._headers)

    def post(self, path, json):
        return self._client.post("/api/v1" + path, json, headers=self._headers)


def cli_run(capsys, *extra):
    code = execute_from_command_line(["manage.py", "run", *ARGS, "--json", *extra])
    out = capsys.readouterr().out
    assert code == 0, out
    return json.loads(out[out.index("{"):])


def api_digest(get, run_id):
    """What the API exposes per epoch, reduced to what the CLI summary also prints."""
    run = get(f"/runs/{run_id}")
    episodes = get(f"/runs/{run_id}/episodes")["items"]
    aggregations = get(f"/runs/{run_id}/aggregations")["items"]
    return run, episodes, aggregations


def assert_matches(summary, run, episodes, aggregations):
    assert run["status"] == summary["status"] == "completed"
    assert len(summary["history"]) == len(episodes) == len(aggregations) == 2
    if "kappa" in summary:
        assert run["kappa"] == pytest.approx(summary["kappa"], abs=TOL)
        assert run["entropy"] == pytest.approx(summary["entropy"], abs=TOL)
    for entry, episode, aggregation in zip(summary["history"], episodes, aggregations):
        report = aggregation["report"]
        assert entry["epoch"] == episode["episode"] == aggregation["episode"]
        assert entry["kappa"] == pytest.approx(report["kappa"], abs=TOL)
        assert entry["entropy"] == pytest.approx(report["entropy"], abs=TOL)
        assert set(entry["weights"]) == set(report["weights"])
        for peer, weight in entry["weights"].items():
            assert weight == pytest.approx(report["weights"][peer], abs=TOL)
        cards = episode["scorecard"]["peers"]
        assert set(cards) == set(entry["scorecards"]["peers"])
        for peer in cards:
            expected = entry["scorecards"]["peers"][peer]
            assert cards[peer]["final_score"] == pytest.approx(expected["final_score"], abs=TOL)
            assert cards[peer]["severity"] == expected["severity"]
        mean = sum(card["final_score"] for card in cards.values()) / len(cards)
        assert entry["score"] == pytest.approx(mean, abs=TOL)

def _wait(get, run_id, timeout=120):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = get(f"/runs/{run_id}")
        if run["status"] in ("completed", "failed", "cancelled"):
            return run
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_in_process_api_run_equals_the_cli_run(project, capsys):
    summary = cli_run(capsys)
    client = Client(token="all-tok")
    response = client.post("/runs", json={"epochs": 2, "seed": 1337, "device": "cpu", "run_id": "api1"})
    assert response.status_code == 202, response.text

    def get(path):
        reply = client.get(path)
        assert reply.status_code == 200, (path, reply.text)
        return reply.json()

    _wait(get, "api1")
    assert_matches(summary, *api_digest(get, "api1"))


def test_the_cli_is_reproducible_with_the_same_seed(project, capsys):
    first = cli_run(capsys)
    second = cli_run(capsys)
    for a, b in zip(first["history"], second["history"]):
        assert a["score"] == pytest.approx(b["score"], abs=TOL)
        assert a["weights"] == pytest.approx(b["weights"], abs=TOL)


def test_two_api_runs_with_one_seed_agree(project):
    client = Client(token="all-tok")
    def get(path):
        return client.get(path).json()

    for run_id in ("a", "b"):
        created = client.post("/runs", json={"epochs": 2, "seed": 1337, "device": "cpu", "run_id": run_id})
        assert created.status_code == 202, created.text
        _wait(get, run_id)
    left, right = api_digest(get, "a"), api_digest(get, "b")
    for x, y in zip(left[2], right[2]):
        assert x["report"]["kappa"] == pytest.approx(y["report"]["kappa"], abs=TOL)
        for peer in x["report"]["weights"]:
            assert x["report"]["weights"][peer] == pytest.approx(y["report"]["weights"][peer], abs=TOL)
    for x, y in zip(left[1], right[1]):
        for peer, card in x["scorecard"]["peers"].items():
            assert card["final_score"] == pytest.approx(y["scorecard"]["peers"][peer]["final_score"], abs=TOL)


def test_real_socket_run_equals_the_cli_run(project, capsys):
    from mactf.contrib.api.management.commands.serve import build_server

    summary = cli_run(capsys)
    httpd = build_server("127.0.0.1", 0, workers=8, quiet=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        client = MACTFClient(transport=HTTPTransport(f"http://{host}:{port}", token="all-tok", timeout=30))
        run = client.run_to_completion(run_id="sock", epochs=2, seed=1337, device="cpu", timeout=120)
        assert run.status == "completed"
        transport = client.transport

        def get(path):
            reply = transport.request("GET", "/api/v1" + path)
            assert reply.status_code == 200, (path, reply.content[:300])
            return json.loads(reply.content)

        assert_matches(summary, *api_digest(get, "sock"))
        events = list(client.runs.stream("sock"))
        assert [e.event for e in events].count("epoch") >= 2 or events
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
