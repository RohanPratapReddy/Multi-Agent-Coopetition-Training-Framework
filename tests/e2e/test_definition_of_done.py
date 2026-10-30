"""Document 06 section 9: the definition-of-done scripts, run as written (no private helpers)."""

from __future__ import annotations

import pytest
import torch

from mactf.client import InProcessTransport, MACTFClient, TransportError
from mactf.coopetition.tree import HierarchySpec
from mactf.scoring import PeerScorecard
from mactf.orchestration.checkpoint_v2 import MACTFCheckpoint
from mactf.training import MACTFTrainer
from mactf.training.logging import MACTFLogger
from mactf.training.watchdog import MACTFWatchdog
from e2e_peers import E2EPeer


@pytest.fixture(params=["in_process", "http"])
def client(request, project):
    if request.param == "in_process":
        yield MACTFClient(transport="in_process", token="all-tok")
        return
    import threading

    from mactf.contrib.api.management.commands.serve import build_server

    httpd = build_server("127.0.0.1", 0, workers=8, quiet=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        yield MACTFClient(transport="http", base_url=f"http://{host}:{port}", token="all-tok", timeout=30)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_the_client_sequence_returns_the_framework_own_objects(client):
    assert client.meta().api_version == "v1"
    assert len(client.rules.list()) == 40
    run = client.runs.create({"epochs": 1, "seed": 1337, "device": "cpu"})
    events = list(client.runs.stream(run.id))
    assert events[-1].type == "terminal"
    assert any(event.epoch == 0 for event in events if event.type == "epoch")
    final = client.runs.get(run.id)
    assert final.status == "completed"
    episode = client.runs.episode(run.id, 0)
    card = episode.peers["p0"]
    assert isinstance(card, PeerScorecard)
    page = client.runs.episodes(run.id)
    assert page.total == 1
    voided = client.score("a", [("P1", "fabricated a citation", 1.0)])
    assert voided.is_void
    assert client.sandbox.status().provider == "local"
    preview = client.peers.preview_weights({"p0": 0.4, "p1": 0.9})
    assert preview is not None


def test_the_two_spellings_of_the_transport_select_the_same_classes(project):
    assert isinstance(MACTFClient(transport="in_process").transport, InProcessTransport)
    with pytest.raises(TransportError):
        MACTFClient(transport="http", base_url="http://127.0.0.1:9", token="t")


def test_a_bad_transport_name_is_rejected(project):
    with pytest.raises(ValueError):
        MACTFClient(transport="carrier-pigeon")
    with pytest.raises(ValueError):
        MACTFClient(transport="http")


# -- the wrapper layer script ------------------------------------------------------
def build_model():
    return E2EPeer()


def test_the_wrapper_layer_script(project, tmp_path):
    project["TRAINING"].update({"EPOCHS": 2, "BATCH_SIZE": 8, "STATE_DIM": 4, "ACTION_DIM": 3, "BOOTSTRAP": False})
    torch.manual_seed(0)
    checkpoints = MACTFCheckpoint(tmp_path / "ckpt", run_id="dod")
    watchdog = MACTFWatchdog(checkpoint=checkpoints, logger=MACTFLogger(run_id="dod", sinks=[], async_write=False))
    trainer = MACTFTrainer(
        model_factory=build_model,
        hierarchy=HierarchySpec.from_spec(
            {
                "roots": 2,
                "levels": [
                    {"label": "subagent", "count": 2},
                    {"label": "miniagent", "count": 2},
                ],
            }
        ),
        peer_strategy="hierarchical",
        weight_update="kappa_softmax",
        settings=project,
        watchdog=watchdog,
    )
    trainer.build()
    result = trainer.fit(epochs=2)
    assert len(trainer.peers) >= 8

    report = trainer.checkpoint.resume_from_last_checkpoint(trainer=trainer)
    assert report.bit_identical_expected

    sample = torch.randn(1, 4)
    out = trainer.infer({"state": sample}, node_path=None)
    assert out is not None
    assert trainer.watchdog is not None
    health = trainer.watchdog.tick(trainer.context)
    assert getattr(health.status, "value", health.status) in {"healthy", "degraded"}

    scale = trainer.scorer.points_table
    assert len(scale.rewards) + len(scale.penalties) == 40
    last = result.episodes[-1]
    card = last.peers[next(iter(last.peers))]
    assert card.metadata["points"]["final_points"] == pytest.approx(card.final_points)
    assert 0.0 <= card.points.final_points <= 100.0
    normalized = trainer.normalize_episode(last)
    assert normalized.preserves_order
    assert all(0.0 <= s.calibrated <= 100.0 for s in normalized.scores)