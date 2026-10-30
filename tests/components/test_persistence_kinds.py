"""Every kind a run can customise lands in the manifest, once, under its own slot (section 14.9)."""

from __future__ import annotations

import pytest

from persist_support import EVERY_KIND_BLOCKS, register_every_kind

from p2_support import configure, make_trainer, tune

from mactf.components.fingerprint import component_manifest, recorded_slots


@pytest.fixture
def every(settings, tmp_path, component_overlay):
    register_every_kind()
    tune(settings, EPOCHS=2, CHECKPOINT_EVERY=1, DETERMINISTIC=True)
    configure(settings, **EVERY_KIND_BLOCKS)
    trainer = make_trainer(settings, tmp_path, peers=2)
    trainer.run_id = "kinds-run"
    return trainer.build()


def test_every_kind_has_an_entry(every):
    manifest = every._component_manifest()
    assert recorded_slots(manifest) == [
        ("aggregator", "TRAINER.AGGREGATOR"),
        ("optimizer", "OPTIMIZER"),
        ("reward_term", "scoring.reward.terms.sqrt_magnitude"),
        ("rl_function.advantage", "RL.advantage"),
        ("scheduler", "OPTIMIZER:scheduler"),
        ("scorer", "scoring.penalty"),
        ("trainer_policy", "TRAINER.POLICY"),
        ("weight_update", "TRAINER.WEIGHT_UPDATE"),
    ]
    for kind, entries in manifest.items():
        for path, entry in entries.items():
            assert entry["kind"] == kind and entry["slot"] == path and entry["origin"] == "project"
            assert entry["schema"] == 1 and entry["source_digest"] and entry["spec_hash"]
    assert manifest["reward_term"]["scoring.reward.terms.sqrt_magnitude"]["params"] == {"scale": 2.0}
    assert manifest["aggregator"]["TRAINER.AGGREGATOR"]["name"] == "tracked"
    assert manifest["rl_function.advantage"]["RL.advantage"]["params"] == {"clip": 2.0}


def test_the_builtin_terms_and_stages_of_a_customised_pipeline_are_not_recorded(every):
    manifest = every._component_manifest()
    assert "scoring.reward" not in manifest.get("scorer", {})            # composed_reward is the framework's
    assert "scoring.reward.terms.severity_scaled" not in manifest.get("reward_term", {})


def test_a_trainer_with_nothing_customised_in_these_kinds_records_nothing_for_them(settings, tmp_path):
    tune(settings, EPOCHS=1)
    configure(settings, TRAINER={"POLICY": "default"}, SCORING={"STAGES": {"reward": "composed_reward"}})
    trainer = make_trainer(settings, tmp_path, peers=2).build()
    assert trainer._component_manifest() == {}


def test_in_code_components_are_recorded_as_runtime(settings, tmp_path, component_overlay):
    from persist_support import MyPolicy, TrackedAggregator
    from mactf.coopetition.aggregation import Aggregator
    from mactf.coopetition.temperature import AdaptiveKappa

    tune(settings, EPOCHS=1)
    aggregator = TrackedAggregator(controller=AdaptiveKappa(num_peers=2), tag="code")
    trainer = make_trainer(settings, tmp_path, peers=2, aggregator=aggregator, trainer=MyPolicy()).build()
    manifest = trainer._component_manifest()
    assert manifest["aggregator"]["TRAINER.AGGREGATOR"]["origin"] == "runtime"
    assert manifest["trainer_policy"]["TRAINER.POLICY"]["origin"] == "runtime"
    plain = make_trainer(settings, tmp_path / "p", peers=2,
                         aggregator=Aggregator(controller=AdaptiveKappa(num_peers=2))).build()
    assert plain._component_manifest() == {}                           # the framework's own class


def test_in_code_scorer_stages_are_recorded_unless_they_are_the_frameworks(settings, tmp_path, component_overlay):
    from persist_support import MyPenalty

    from mactf.scoring.reward_scorer import RewardScorer
    from mactf.scoring.warning_scorer import WarningScorer

    tune(settings, EPOCHS=1)
    stages = [WarningScorer(), RewardScorer(), MyPenalty()]
    trainer = make_trainer(settings, tmp_path, peers=2, scorers=stages).build()
    assert recorded_slots(trainer._component_manifest()) == [("scorer", "scoring.penalty")]


def test_the_manifest_survives_a_checkpoint_round_trip_unchanged(every):
    every.run_epoch(0)
    record = every.checkpoint.save(
        __import__("mactf.orchestration.checkpoint_v2", fromlist=["CheckpointContext"]).CheckpointContext(
            trainer=every, episode=0), reason="manual")
    stored = every.checkpoint.load(record.path)
    assert stored.identity["components"] == component_manifest(every._slot_view())
