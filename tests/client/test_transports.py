"""Both transports against the same API: results, errors and streams must be identical."""

from __future__ import annotations

import pytest

from mactf.client import (
    Conflict,
    HTTPTransport,
    InProcessTransport,
    MACTFClient,
    NotFound,
    Unauthorized,
    ValidationError,
    specs,
)


def test_meta_and_health(both):
    assert both.health()["status"] in ("ok", "healthy") or both.health()
    assert both.meta().api_version == "v1"


def test_rules_catalog(both):
    rules = both.rules.list()
    assert len(rules) == 40
    assert both.rules.get("P1").tier_name == "honesty_and_non_deception"
    assert both.rules.validation().ok is True
    assert [s.label for s in both.rules.severities()][0] == "S0"
    assert both.scores.scale().rewards


def test_peers_and_sandbox(both):
    assert {p.name for p in both.peers.list()} >= {"p0", "p1"}
    providers = both.sandbox.providers()
    assert providers and all(hasattr(p, "provider") or hasattr(p, "name") for p in providers)
    assert both.sandbox.status().provider == "local"


def test_evaluate_scores_a_ledger(both):
    card = both.score("a", [{"rule_id": "P1", "evidence": "fabricated a citation", "magnitude": 1.0}])
    assert "P1" in card.violated_non_offsettable
    assert card.final_score < 0


def test_tuple_events(both):
    card = both.score("a", [("R1", "answered correctly", 0.5)])
    assert card.final_score > 0


def test_validation_error_carries_field_errors(both):
    with pytest.raises(ValidationError) as caught:
        both.scoring.evaluate("a", [{"rule_id": "NOPE", "evidence": "x"}])
    assert caught.value.status in (400, 422)


def test_not_found(both):
    with pytest.raises(NotFound):
        both.runs.get("does-not-exist")


def test_unauthorized(api_settings, server):
    anonymous = MACTFClient(transport=HTTPTransport(server, timeout=10))
    with pytest.raises(Unauthorized):
        anonymous.score("a", [("R1", "x")])
    anonymous = MACTFClient(transport=InProcessTransport())
    with pytest.raises(Unauthorized):
        anonymous.score("a", [("R1", "x")])


def test_run_to_completion_and_reads(both):
    run = both.run_to_completion(run_id="c1", epochs=2, timeout=120)
    assert run.status == "completed"
    assert run.epochs_completed == 2
    page = both.runs.episodes("c1")
    assert page.total == 2 and [e.episode for e in page] == [0, 1]
    episode = both.runs.episode("c1", 0)
    assert set(episode.scorecard.peers) == {"p0", "p1"}
    card = both.runs.scorecard("c1", 0, "p0", version=1)
    assert card.peer == "p0"
    aggregations = both.runs.aggregations("c1")
    assert aggregations.total == 2
    assert abs(sum(aggregations[0].weights.values()) - 1.0) < 1e-6
    assert both.runs.rule_events("c1", peer="p0").total > 0
    assert both.runs.tree("c1").root_paths
    assert both.runs.checkpoints("c1").total == 2
    assert both.runs.checkpoint("c1", 0).episode == 0
    assert both.observability.verify_checkpoint("c1", 0).ok is True
    assert both.runs.list().total >= 1
    assert [r.id for r in both.runs.iter()] == ["c1"]
    assert both.runs.summary("c1").run.id == "c1"


def test_v2_scorecard_without_points_is_a_validation_error(both):
    both.run_to_completion(run_id="c2", epochs=1, timeout=120)
    with pytest.raises(ValidationError):
        both.scores.scorecard("c2", 0, "p0", version=2)


def test_stream_ends_with_terminal(both):
    handle = both.runs.create(run_id="c3", epochs=1)
    events = list(both.runs.stream(handle.run_id))
    assert events[-1].event == "terminal"
    assert all(e.id is None or isinstance(e.id, int) for e in events)


def test_stream_resume_from_last_event_id(both):
    both.run_to_completion(run_id="c4", epochs=2, timeout=120)
    full = [e for e in both.runs.stream("c4") if e.event != "heartbeat"]
    assert full[-1].event == "terminal"
    ids = [e.id for e in full if e.id is not None]
    tail = [e for e in both.runs.stream("c4", last_event_id=ids[0]) if e.event != "heartbeat"]
    assert [e.id for e in tail] == ids[1:]


def test_conflict_cancelling_nothing_is_not_an_error_but_resuming_terminal_is(both):
    both.run_to_completion(run_id="c5", epochs=1, timeout=120)
    with pytest.raises(Conflict):
        both.observability.resume("c5")


def test_trainer_preview_roundtrip(both):
    spec = specs.trainer(
        optimizer=specs.adamw(2e-4), num_peers=2,
    )
    resolved = both.trainers.preview(spec)
    assert resolved
    with pytest.raises(ValidationError):
        both.trainers.preview({"optimizer": {"type": "adamw"}})


def test_weights_preview_returns_an_aggregation_report(both):
    report = both.peers.preview_weights({"a": 1.0, "b": 0.0})
    assert abs(sum(report.weights.values()) - 1.0) < 1e-6


def test_transport_precedence(api_settings, server):
    assert isinstance(MACTFClient().transport, InProcessTransport)
    assert isinstance(MACTFClient(base_url=server, token="all-tok").transport, HTTPTransport)
    explicit = InProcessTransport()
    assert MACTFClient(transport=explicit, base_url=server).transport is explicit


def test_peer_detail_and_contract_use_the_name_argument(both):
    assert both.peers.get("p0").name == "p0"
    assert both.peers.contract("p0")
    with pytest.raises(NotFound):
        both.peers.get("ghost")


def test_scoring_workflow_on_a_finished_run(both):
    both.run_to_completion(run_id="c6", epochs=1, timeout=120)
    assert both.scoring.flags("c6")["submitted"] == 0
    assert both.scoring.adjudications("c6").total == 0
    assert both.scores.points("c6").total == 0
    with pytest.raises(Conflict):
        both.scores.normalization("c6", 0)
    both.scoring.adjudicate(
        "c6", peer="p0", rule_id="R1", episode=0, decision="upheld",
        rationale="checked by hand", adjudicator="reviewer",
    )
    assert both.scoring.adjudications("c6").total == 1
    card = both.scoring.append_ledger("c6", 0, "p0", [("P1", "invented a source")])
    assert "P1" in card.violated_non_offsettable
    assert both.observability.logs("c6")["run_id"] == "c6"
    assert both.observability.health("c6")["run_id"] == "c6"
    with pytest.raises(Conflict):  # no live run has provisioned a sandbox for the peer
        both.sandbox.exec("p0", "echo hello")
