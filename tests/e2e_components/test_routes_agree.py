"""One configuration, three routes: ``manage.py run``, ``manage.py serve`` + the Python client, and a live steer.

The CLI run and the API run use the same settings and seed, so their final checkpoints must hold identical
tensors. A second, long API run is steered while it trains (component PATCH, a tree-mutation dry run) and cancelled.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from e2e_support import BASE_FLAGS, assert_same_state, manage, state_of
from test_cli_resume import CUSTOM

pytestmark = [pytest.mark.subprocess, pytest.mark.posix]

FLAGS = [*BASE_FLAGS, "--set", "ORCHESTRATOR.ENGINE=trainer", "--set", "ORCHESTRATOR.ENGINE_COMPAT=false",
         "--set", "TRAINER.MUTATION.ENABLED=true", "--set", "SEED=7", "--config", "extra.yaml"]


@pytest.fixture
def server(project):
    apps = ["mactf.contrib.scoring", "mactf.contrib.coopetition", "mactf.contrib.sandbox", "mactf.contrib.telemetry",
            "mactf.contrib.components", "mactf.contrib.api", "e2eproj.peers"]
    (project / "extra.yaml").write_text(CUSTOM + f"INSTALLED_APPS: {apps!r}\n".replace("'", '"')
                                     + 'API: {TOKENS: {e2e-token: [read, run:write, score:write, sandbox:exec, admin]}}\n')
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(project), os.environ.get("PYTHONPATH", "")]))}
    port_file = project / "port.txt"
    proc = subprocess.Popen([sys.executable, "manage.py", "serve", "--port", "0", "--port-file", str(port_file), *FLAGS],
                            cwd=project, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 90
    while time.time() < deadline and not (port_file.exists() and port_file.read_text().strip()):
        if proc.poll() is not None:
            raise AssertionError(f"serve exited {proc.returncode}: {proc.stderr.read()[-1500:]}")
        time.sleep(0.1)
    else:
        if not port_file.exists():
            proc.kill()
            raise AssertionError("serve never wrote its port")
    yield f"http://127.0.0.1:{port_file.read_text().strip()}/api/v1/"
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_the_cli_and_the_http_client_produce_the_same_run(project, server):
    from mactf.client import MACTFClient

    with MACTFClient(base_url=server.rstrip("/").removesuffix("/api/v1"), token="e2e-token") as client:
        assert client.health() is not None
        listed = client.components.list(kind="weight_update")
        assert "ema_softmax" in [item["name"] for item in listed.items]
        client.components.validate("weight_update", {"type": "ema_softmax", "params": {"alpha": 0.5}})
        run = client.run_to_completion(epochs=6, seed=7, run_id="viaapi", timeout=180)
        assert run.status == "completed", run
    cli = manage(project, "run", "--epochs", "6", "--run-id", "viacli", "--seed", "7", *FLAGS)
    assert cli.returncode == 0, cli.stderr
    sections = ("models", "reference", "optimizers", "schedulers")
    assert_same_state(state_of(project, "viacli", 5, sections), state_of(project, "viaapi", 5, sections))


def test_a_live_run_is_steered_through_the_client_and_cancelled(project, server):
    from mactf.client import MACTFClient
    from mactf.client.errors import NotFound

    with MACTFClient(base_url=server.rstrip("/").removesuffix("/api/v1"), token="e2e-token") as client:
        handle = client.runs.create(epochs=400, seed=7, run_id="live")
        run = client.wait_for("live", until=lambda r: (r.epochs_completed or 0) >= 1, timeout=120)
        assert not run.terminal
        shown = client.components.show("live")
        assert any(slot["slot"] == "TRAINER.WEIGHT_UPDATE" for slot in shown["slots"])
        client.components.patch("live", "TRAINER.WEIGHT_UPDATE", {"type": "uniform"})
        with pytest.raises(NotFound):
            client.components.patch("live", "TRAINER.WEIGHT_UPDATE", {"type": "no_such_strategy"})
        preview = client.components.mutate("live", {"op": "add", "path": "peer_delta", "like": "peer_alpha"}, dry_run=True)
        assert preview is not None
        client.runs.cancel("live")
        final = client.wait_for("live", timeout=120)
        assert final.status == "cancelled", final.status
