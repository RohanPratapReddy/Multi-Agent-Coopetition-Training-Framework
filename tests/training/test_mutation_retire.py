"""Retire, revive, purge: who is out of play and what outlives it (document 17 section 12.6, rows 38-43)."""

import pytest
import torch

from mactf.exceptions import MutationError
from mactf.training.callbacks import Callback
from mactf.training.mutation import MutationPlan, add_peer, purge_peer, retire_peer
from mutation_support import mutable_trainer, mutation_settings, parameters, snapshot


@pytest.fixture()
def trainer(settings, tmp_path):
    mutation_settings(settings)
    trainer = mutable_trainer(settings, tmp_path, peers=4)
    trainer.run_epoch(0)
    return trainer


def _flat(trainer, path):
    return trainer.peers[path].export_parameters().detach().clone()


def test_a_retired_peer_does_not_roll_out_or_score(trainer):
    seen = {"steps": [], "batches": []}

    class Spy(Callback):
        name = "spy"

        def on_batch(self, trainer, node, batch):
            seen["batches"].append(node.name if hasattr(node, "name") else str(node))

    trainer.retire_peer("peer-3")
    before = {path: _flat(trainer, path) for path in trainer.peers}
    optimizer_before = trainer.optimizers["peer-3"].state_dict()
    trainer.add_callback(Spy())
    result = trainer.run_epoch(1)
    episode, card = trainer.episodes[-1], trainer.scorecards[-1]
    # collect_episode: it never ran
    assert "peer-3" not in episode.participants and "peer-3" not in episode.ledgers
    # score: no scorecard, no signal, no metrics row
    assert "peer-3" not in card.peers and "peer-3" not in trainer.signal_by_peer
    assert "peer-3" not in result.metrics
    # _train_epoch and _write: not one step, not one write
    assert "peer-3" not in seen["batches"] and set(seen["batches"]) == {"peer-1", "peer-2", "peer-4"}
    assert torch.equal(_flat(trainer, "peer-3"), before["peer-3"])
    assert trainer.optimizers["peer-3"].state_dict()["state"] == optimizer_before["state"]
    # update_weights / aggregate: it carries no weight and receives none
    report = trainer._epoch_report
    assert set(report.weights) == {"peer-1", "peer-2", "peer-4"}
    assert sum(report.weights.values()) == pytest.approx(1.0)
    # the others did move
    assert not torch.equal(_flat(trainer, "peer-1"), before["peer-1"])
    # evaluate and infer skip it as well
    assert "peer-3/entropy" not in trainer.evaluate()
    assert "peer-1/entropy" in trainer.evaluate()


def test_a_retired_peer_keeps_its_weights(trainer):
    before = _flat(trainer, "peer-3")
    optimizer = trainer.optimizers["peer-3"]
    trainer.retire_peer("peer-3", reason="resting")
    for epoch in (1, 2):
        trainer.run_epoch(epoch)
    assert trainer.peers["peer-3"] is not None and trainer.tree["peer-3"].status == "pruned"
    trainer.revive_peer("peer-3")
    assert torch.equal(_flat(trainer, "peer-3"), before)                   # parameters unchanged by the round trip
    assert trainer.optimizers["peer-3"] is optimizer and trainer.tree["peer-3"].status == "active"
    trainer.run_epoch(3)
    assert not torch.equal(_flat(trainer, "peer-3"), before)               # a working peer again
    assert trainer.path_state.audit() == []


@pytest.mark.parametrize("mode", ["legacy", "strict"])
def test_a_retired_peer_has_weight_zero(settings, tmp_path, mode):
    mutation_settings(settings)
    settings["TRAINER"]["FREEZE_MODE"] = mode
    trainer = mutable_trainer(settings, tmp_path, peers=4)
    trainer.run_epoch(0)
    trainer.retire_peer("peer-3")
    assert trainer.freeze.excludes("peer-3") and trainer.freeze.skips_training("peer-3")
    assert trainer.freeze.skips_write("peer-3")
    assert trainer.freeze.eligible_for(["peer-1", "peer-3"]) == {"peer-1": True, "peer-3": False}
    assert "mutation:retired" in trainer.freeze.reasons("peer-3")
    trainer.run_epoch(1)
    assert "peer-3" not in trainer._epoch_report.weights                  # zero weight by omission
    assert trainer._last_weights.get("peer-3") in (None, 0.0)


def test_purge_forgets_the_containers(trainer):
    trainer.mutate(MutationPlan.of(add_peer("peer-1/a", like="peer-1"), add_peer("peer-1/b", like="peer-1")))
    trainer.run_epoch(1)
    assert "peer-1/a" in trainer._epoch_node_metrics and "peer-1" in trainer.controllers
    trainer.mutate(MutationPlan.of(retire_peer("peer-1", recursive=True), purge_peer("peer-1", recursive=True)))
    assert trainer.path_state.audit() == []
    for name in ("peers", "optimizers", "signal_by_peer", "_last_weights", "_previous_scores", "_dead_counts",
                 "_epoch_node_metrics"):
        assert not [p for p in getattr(trainer, name) if p.startswith("peer-1")], name
    assert "peer-1" not in trainer.controllers and "peer-1" not in trainer._level_controllers
    assert "peer-1" not in trainer.freeze and trainer.freeze.paths() == []
    assert all(key[2] != "peer-1" for key in trainer.functions._cache)
    assert "peer-1" not in trainer.tree and "peer-1/a" not in trainer.tree
    trainer.run_epoch(2)
    assert trainer.path_state.audit() == []
    assert set(trainer.tree.root_paths) == {"peer-2", "peer-3", "peer-4"}


def test_a_purged_path_cannot_return(trainer):
    trainer.purge_peer("peer-3", force=True)
    with pytest.raises(MutationError) as caught:
        trainer.add_peer("peer-3", like="peer-1")
    assert caught.value.code == "E081" and "cannot be reused" in str(caught.value)
    assert "peer-3" in trainer.topology()["tombstones"]


def test_tombstone_off_allows_the_path_back(settings, tmp_path):
    mutation_settings(settings, TOMBSTONE="off")
    trainer = mutable_trainer(settings, tmp_path, peers=4)
    trainer.purge_peer("peer-3", force=True)
    trainer.add_peer("peer-3", like="peer-1")
    assert "peer-3" in trainer.peers and trainer.topology()["tombstones"] == []


def test_history_outlives_a_purge(trainer):
    from mactf.rules.severity import Severity

    registry = trainer.scorer.registry_for("peer-3")
    registry.issue("peer-3", "P11", severity=Severity.MINOR, episode=0)
    assert registry.count("peer-3", "P11") == 1
    cards_before = len(trainer.scorecards)
    trainer.purge_peer("peer-3", force=True)
    trainer.run_epoch(1)
    # scorecards, episodes and the warning registry keep the path (M3)
    assert "peer-3" in trainer.scorecards[0].peers and "peer-3" in trainer.episodes[0].participants
    assert len(trainer.scorecards) == cards_before + 1
    assert registry.count("peer-3", "P11") == 1
    assert "peer-3" in trainer.scorecards[0].to_dict()["peers"]
    assert any("peer-3" in item.to_dict()["peers"] for item in trainer.scorecards[:1])


def test_retiring_a_peer_does_not_erase_what_it_did(trainer):
    """M3: retire is out of the *next* epochs' scoring; its past scorecard and warnings stay."""
    from mactf.rules.severity import Severity

    registry = trainer.scorer.registry_for("peer-2")
    registry.issue("peer-2", "P11", severity=Severity.MINOR, episode=0)
    past = trainer.scorecards[0].peers["peer-2"].final_score
    trainer.retire_peer("peer-2")
    trainer.run_epoch(1)
    assert trainer.scorecards[0].peers["peer-2"].final_score == past
    assert registry.count("peer-2", "P11") == 1                              # the registry is not touched
    trainer.revive_peer("peer-2")
    assert registry.count("peer-2", "P11") == 1                              # ...and resumes where it stopped


def test_a_new_peer_is_scored_by_the_same_chain_from_its_first_episode(trainer):
    chain_before = [type(stage) for stage in trainer.scorer.chain]
    trainer.add_peer("peer-9", like="peer-1")
    trainer.run_epoch(1)
    card = trainer.scorecards[-1]
    assert "peer-9" in card.peers
    assert [type(stage) for stage in trainer.scorer.chain] == chain_before   # no lighter ruleset was built for it
    old, new = card.peers["peer-1"], card.peers["peer-9"]
    assert set(new.to_dict()) == set(old.to_dict())                          # the same fields: floors, warnings, void
    assert trainer.scorer.registry_for("peer-9") is not None


def test_a_void_peer_stays_void_through_a_retire_of_its_sibling(trainer):
    """Retiring a peer cannot launder another's violation: the void is computed from the epoch's own ledgers."""
    from mactf.rules.severity import Severity

    trainer.retire_peer("peer-4")
    episode = trainer.collect_episode()
    episode.ledger_for("peer-1").record("P1", magnitude=1.0, evidence="forced", source="test")
    card = trainer.score()
    assert card.peers["peer-1"].is_void and not card.peers["peer-2"].is_void
    assert "peer-4" not in card.peers
