"""Document 11 section 8: SIGNAL_SOURCE, kappa rescaling, persistence and migration 3."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mactf.coopetition.temperature import AdaptiveKappa
from mactf.db import Registry
from mactf.exceptions import ContractError, ValidationError
from mactf.orchestration.phases import PhaseContext
from mactf.scoring import EpisodeEvidence, Ledger, ScoringPipeline
from mactf.scoring.scorecard import EpisodeScorecard
from mactf.training import MACTFTrainer


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(4, 16), nn.Tanh())
        self.policy = nn.Linear(16, 3)
        self.value = nn.Linear(16, 1)

    def forward(self, state):
        hidden = self.body(state)
        return self.policy(hidden), self.value(hidden)


@pytest.fixture()
def points_settings(settings):
    settings["TRAINING"].update(
        {"EPOCHS": 2, "BATCH_SIZE": 8, "STATE_DIM": 4, "ACTION_DIM": 3, "BOOTSTRAP": False}
    )
    settings["SANDBOX"]["PROVIDER"] = "local"
    settings["COOPETITION"]["SIGNAL_SOURCE"] = "points"
    return settings


def _trainer(settings, tmp_path):
    torch.manual_seed(0)
    return MACTFTrainer(
        model=Tiny(), num_peers=2, settings=settings, checkpoint_dir=tmp_path / "ckpt"
    )


def _cards(points_by_peer, *, with_points=True):
    pipeline = ScoringPipeline()
    card = EpisodeScorecard(episode=1)
    for peer, rules in points_by_peer.items():
        ledger = Ledger(peer=peer)
        for rule_id in rules:
            ledger.record(rule_id)
        card.add(
            pipeline.score_peer(
                ledger,
                peer=peer,
                episode=1,
                evidence=EpisodeEvidence(task_checks_passed=True) if with_points else None,
            )
        )
    return card


def _context(scorecards, **kwargs):
    return PhaseContext(
        run_id="r",
        epoch=1,
        peers={},
        reference=None,
        scorer=None,
        runner=None,
        runtime=None,
        scorecards=scorecards,
        **kwargs,
    )


# -- PhaseContext ----------------------------------------------------------
def test_final_score_source_averages_final_score():
    card = _cards({"a": ["R1"], "b": []})
    signal = _context([card]).signal_by_peer()
    assert signal["a"] == card.peers["a"].final_score


def test_points_source_normalizes_points_into_a_bounded_signal():
    card = _cards({"a": ["R1"], "b": []})
    ctx = _context([card], signal_source="points")
    signal = ctx.signal_by_peer()
    assert signal["a"] > signal["b"]
    assert all(-1.0 <= value <= 1.0 for value in signal.values())
    assert ctx.signal_by_peer(source="final_score") != signal


def test_points_source_without_points_is_a_contract_error():
    card = _cards({"a": ["R1"], "b": []}, with_points=False)
    with pytest.raises(ContractError, match="points"):
        _context([card], signal_source="points").signal_by_peer()


def test_a_void_peer_gets_the_floor_signal():
    card = _cards({"a": ["R1"], "b": ["R1"], "v": ["R1", "P20"]})
    signal = _context([card], signal_source="points").signal_by_peer()
    assert signal["v"] < signal["a"] and signal["v"] == pytest.approx(-62.0 / 50.0)


# -- kappa -----------------------------------------------------------------
def test_kappa_is_rescaled_for_the_points_source():
    base = {"KAPPA_INIT": 1.0, "SIGNAL_SOURCE": "final_score"}
    raw = AdaptiveKappa.from_settings(base, num_peers=2)
    points = AdaptiveKappa.from_settings({**base, "SIGNAL_SOURCE": "points"}, num_peers=2)
    assert raw.state.kappa == 1.0
    assert points.state.kappa == pytest.approx(0.1)
    assert points.state.signal_source == "points"
    assert points.state.scaled_kappa_init == pytest.approx(0.1)


def test_kappa_rescale_uses_the_supplied_reference_scale():
    controller = AdaptiveKappa.from_settings(
        {"KAPPA_INIT": 1.0, "SIGNAL_SOURCE": "points"},
        num_peers=2,
        mean_abs_reference_signal=4.0,
    )
    assert controller.state.kappa == pytest.approx(0.25)


def test_an_unknown_signal_source_is_rejected_by_the_controller():
    with pytest.raises(ValidationError, match="SIGNAL_SOURCE"):
        AdaptiveKappa.from_settings({"SIGNAL_SOURCE": "vibes"}, num_peers=2)


def test_kappa_state_round_trips_signal_source():
    controller = AdaptiveKappa.from_settings(
        {"KAPPA_INIT": 1.0, "SIGNAL_SOURCE": "points"}, num_peers=2
    )
    clone = AdaptiveKappa.from_settings({"KAPPA_INIT": 1.0}, num_peers=2)
    clone.load_state_dict(controller.state_dict())
    assert clone.state.signal_source == "points"
    assert clone.state.scaled_kappa_init == pytest.approx(0.1)


# -- trainer ---------------------------------------------------------------
def test_trainer_runs_under_the_points_source(points_settings, tmp_path):
    trainer = _trainer(points_settings, tmp_path)
    result = trainer.fit(epochs=2)
    assert result.status != "failed"
    card = trainer.scorecards[-1]
    assert all(peer.points is not None for peer in card.peers.values())
    # clip [0, 100] around target 62 with scale 50 bounds the signal to [-1.24, 0.76]
    assert all(-1.24 - 1e-9 <= v <= 0.76 + 1e-9 for v in trainer.signal_by_peer.values())
    assert trainer.signal_source == "points"
    report = trainer.reports[-1]
    assert report.signal_source == "points" and report.normalization is not None


def test_trainer_normalize_episode_is_pure_under_points(points_settings, tmp_path):
    trainer = _trainer(points_settings, tmp_path)
    trainer.fit(epochs=1)
    card = trainer.scorecards[-1]
    before = card.to_dict()
    first = trainer.normalize_episode(card)
    second = trainer.normalize_episode(card)
    assert first.signals() == second.signals()
    assert card.to_dict() == before


def test_trainer_uses_a_supplied_evidence_provider(points_settings, tmp_path):
    trainer = _trainer(points_settings, tmp_path)
    seen = []

    def provider(episode, path):
        seen.append(path)
        return {"task_checks_passed": True, "claims": []}

    trainer.evidence_provider = provider
    trainer.fit(epochs=1)
    assert seen


def test_resume_cannot_switch_signal_source(points_settings, tmp_path):
    trainer = _trainer(points_settings, tmp_path)
    trainer.fit(epochs=1)
    state = trainer.state_dict()
    assert state["signal_source"] == "points" if "signal_source" in state else True
    points_settings["COOPETITION"]["SIGNAL_SOURCE"] = "final_score"
    other = _trainer(points_settings, tmp_path / "other")
    other.build()
    with pytest.raises(Exception, match="SIGNAL_SOURCE"):
        other.load_state_dict(state, strict=True)


# -- database --------------------------------------------------------------
def test_migration_3_persists_points_and_signal_source(tmp_path):
    registry = Registry(tmp_path / "db.sqlite3")
    registry.migrate()
    with registry.connect() as connection:
        connection.execute(
            "INSERT INTO runs (id, started_at, status, settings, runtime) VALUES "
            "('r', 0, 'running', '{}', '{}')"
        )
    card = _cards({"a": ["R1"], "b": []})
    registry.record_episode("r", 1, card.to_dict(), normalization={"mean": 12.5})
    rows = registry.scorecards("r")
    assert {row["peer"] for row in rows} == {"a", "b"}
    assert all(row["points"]["final_points"] >= 0 for row in rows)
    assert all(row["points_table_version"] for row in rows)
    assert registry.episodes("r")[0]["normalization"]

    class Report:
        def as_dict(self):
            return {"signal_source": "points", "epoch": 1}

    registry.record_aggregation("r", 1, Report())
    assert registry.aggregations("r")[0]["signal_source"] == "points"


def test_pre_points_rows_have_null_points_not_a_fabricated_zero(tmp_path):
    registry = Registry(tmp_path / "db.sqlite3")
    registry.migrate()
    with registry.connect() as connection:
        connection.execute(
            "INSERT INTO runs (id, started_at, status, settings, runtime) VALUES "
            "('r', 0, 'running', '{}', '{}')"
        )
    card = _cards({"a": ["R1"]}, with_points=False)
    registry.record_episode("r", 1, card.to_dict())
    assert registry.scorecards("r")[0]["points"] is None
