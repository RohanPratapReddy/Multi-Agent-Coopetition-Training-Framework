"""Nothing configured means nothing mutates (document 17 section 12.11.8, row 58, invariant I1)."""

import pytest

from mactf.exceptions import MutationError
from mactf.training.mutation import MutationLimits
from mutation_support import mutable_trainer, parameters, snapshot


def _tune(settings):
    settings["TRAINING"].update({"EPOCHS": 2, "BATCH_SIZE": 8, "STATE_DIM": 4, "ACTION_DIM": 3, "BOOTSTRAP": False})
    settings["SANDBOX"]["PROVIDER"] = "local"


def test_no_configuration_means_no_mutation(settings, tmp_path):
    _tune(settings)
    assert MutationLimits().enabled is False
    trainer = mutable_trainer(settings, tmp_path, peers=3)
    assert trainer.mutation_limits.enabled is False
    before = snapshot(trainer)
    for call in (
        lambda: trainer.add_peer("peer-4", like="peer-1"),
        lambda: trainer.retire_peer("peer-3"),
        lambda: trainer.purge_peer("peer-3", force=True),
        lambda: trainer.modify_peer("peer-3", role="x"),
        lambda: trainer.mutate({"ops": [{"op": "retire", "path": "peer-3"}]}),
        lambda: trainer.mutate({"ops": [{"op": "retire", "path": "peer-3"}]}, dry_run=True),
    ):
        with pytest.raises(MutationError) as caught:
            call()
        assert caught.value.code == "E088" and "TRAINER.MUTATION.ENABLED" in str(caught.value)
    assert snapshot(trainer) == before and trainer.mutation_journal == []


def test_an_unmutated_run_has_the_structure_it_was_built_with(settings, tmp_path):
    _tune(settings)
    trainer = mutable_trainer(settings, tmp_path, peers=3)
    assert trainer.active_tree is trainer.tree                         # the view is free when nothing is retired
    assert trainer._initial_nodes == len(trainer.peers) == 3
    trainer.run_epoch(0)
    assert trainer.active_tree is trainer.tree and trainer.freeze.retired() == frozenset()
    assert trainer.topology()["digest"] == trainer._base_digest and trainer.topology()["journal"] == 0
    assert trainer._mutation_state() == {} and trainer.engine.pending() == []
    assert trainer.path_state.audit() == []


def test_the_audit_is_not_run_at_build_unless_mutation_is_enabled(settings, tmp_path, monkeypatch):
    _tune(settings)
    from mactf.training.path_state import PathState

    calls = []
    monkeypatch.setattr(PathState, "audit", lambda self: calls.append(1) or [])
    mutable_trainer(settings, tmp_path / "off", peers=2)
    assert calls == []
    settings["TRAINER"]["MUTATION"] = {"ENABLED": True}
    mutable_trainer(settings, tmp_path / "on", peers=2)
    assert calls == [1]


def test_a_dead_branch_is_pruned_through_the_engine_even_when_mutation_is_off(settings, tmp_path):
    """Behaviour change 1 of 12.11.8: the one thing that happens with nothing configured."""
    _tune(settings)
    trainer = mutable_trainer(settings, tmp_path, peers=3)
    assert trainer._prune_node("peer-3", 4) == 1
    assert "peer-3" not in trainer.peers and trainer.mutation_journal[0]["source"] == "watchdog"
    assert trainer.path_state.audit() == []
    trainer.run_epoch(0)
    assert trainer._prune_node("ghost", 4) == 0
