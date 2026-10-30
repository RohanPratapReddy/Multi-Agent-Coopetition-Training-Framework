"""Groups that appear, change and vanish (document 17 section 12.5.4, rows 44-45)."""

import pytest

from mactf.training.mutation import MutationPlan, add_peer, purge_peer, retire_peer
from mutation_support import mutable_trainer, mutation_settings


@pytest.fixture()
def trainer(settings, tmp_path):
    mutation_settings(settings)
    trainer = mutable_trainer(settings, tmp_path, peers=3)
    trainer.run_epoch(0)
    return trainer


def _grow(trainer):
    return trainer.mutate(MutationPlan.of(add_peer("peer-1/a", like="peer-1"), add_peer("peer-1/b", like="peer-1")))


def test_a_new_group_gets_a_controller_and_an_aggregator(trainer):
    report = _grow(trainer)
    assert report.groups == ("peer-1",)
    assert any(w.startswith("W042") and "'peer-1'" in w for w in report.warnings)
    controller = trainer.controllers["peer-1"]
    assert controller is trainer._level_controllers["peer-1"] and controller.num_peers == 2
    aggregator = trainer._group_aggregator("peer-1")                       # built lazily, with that controller
    assert aggregator.controller is controller and trainer._group_aggregators["peer-1"] is aggregator
    trainer.run_epoch(1)
    assert trainer.path_state.audit() == []
    assert trainer.controllers["peer-1"].kappa > 0


def test_a_group_that_changes_size_is_resized_and_its_caches_dropped(trainer):
    _grow(trainer)
    trainer.run_epoch(1)
    aggregator = trainer._group_aggregator("peer-1")          # a strategy that aggregates children builds these
    trainer.weight_plan.strategy_for("peer-1", aggregator)
    assert "peer-1" in trainer._group_aggregators and "peer-1" in trainer.weight_plan._cache
    controller = trainer.controllers["peer-1"]
    report = trainer.add_peer("peer-1/c", like="peer-1")
    assert trainer.controllers["peer-1"] is controller and controller.num_peers == 3
    assert "peer-1" not in trainer._group_aggregators and "peer-1" not in trainer.weight_plan._cache
    assert report.groups == ("peer-1",) and any("W042" in w for w in report.warnings)
    # the root group was not touched, so its cached strategy and controller are
    assert "root" in trainer.weight_plan._cache
    trainer.run_epoch(2)
    assert trainer.path_state.audit() == []


def test_the_root_group_is_resized_in_place(trainer):
    root = trainer.controller
    assert root.num_peers == 3
    trainer.add_peer("peer-4", like="peer-1")
    assert trainer.controller is root and root.num_peers == 4 and trainer.controllers["root"] is root
    assert trainer.aggregator.controller is root
    trainer.retire_peer("peer-4")
    assert root.num_peers == 3
    trainer.run_epoch(1)
    assert trainer._epoch_report.entropy <= root.state.target_entropy + 10  # a real, finite report
    assert set(trainer._epoch_report.weights) == {"peer-1", "peer-2", "peer-3"}


def test_a_vanished_group_drops_its_controller(trainer):
    _grow(trainer)
    trainer.run_epoch(1)
    root = trainer.controller
    root_state = root.snapshot()
    assert "peer-1" in trainer.controllers
    trainer.mutate(MutationPlan.of(
        retire_peer("peer-1/a"), retire_peer("peer-1/b"), purge_peer("peer-1/a"), purge_peer("peer-1/b"),
    ))
    assert "peer-1" not in trainer.controllers and "peer-1" not in trainer._level_controllers
    assert "peer-1" not in trainer._group_aggregators
    assert trainer.controller is root and root.snapshot() == root_state     # W-G10: the root is not the group
    assert trainer.path_state.audit() == []
    trainer.run_epoch(2)
    assert trainer.tree["peer-1"].children == ()


def test_a_retired_branch_keeps_its_controller_for_a_revive(trainer):
    _grow(trainer)
    trainer.run_epoch(1)
    controller = trainer.controllers["peer-1"]
    trainer.retire_peer("peer-1", recursive=True)
    assert trainer.controllers["peer-1"] is controller                        # the group is out of play, not gone
    trainer.run_epoch(2)
    trainer.revive_peer("peer-1", recursive=True)
    assert trainer.controllers["peer-1"] is controller and controller.num_peers == 2
    trainer.run_epoch(3)
    assert trainer.path_state.audit() == []


def test_kappa_overrides_reach_a_new_group(settings, tmp_path):
    mutation_settings(settings)
    settings["COOPETITION"]["KAPPA_STRICT_GLOBS"] = False        # the glob matches a group that does not exist yet
    settings["COOPETITION"]["KAPPA_OVERRIDES"] = {"peer-1": {"KAPPA_INIT": 2.5, "KAPPA_MAX": 20.0}, "root": {"KAPPA_INIT": 1.0}}
    trainer = mutable_trainer(settings, tmp_path, peers=3)
    _grow(trainer)
    assert trainer.controllers["peer-1"].kappa == pytest.approx(2.5)
