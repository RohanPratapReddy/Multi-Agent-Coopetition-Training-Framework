"""The watchdog as a client of the engine, and the baselines a mutation moves (document 17 section 12.8, rows 46-50)."""

import logging
from types import SimpleNamespace

import pytest

from mactf.training.callbacks import EarlyStopping
from mactf.training.mutation import MutationPlan, add_peer
from mactf.training.watchdog import (
    HealthContext, MACTFWatchdog, TreeGrowthProbe, WatchdogPolicy, WatchdogStatus, default_probes,
)
from mutation_support import mutable_trainer, mutation_settings, snapshot


class Sink:
    def __init__(self):
        self.lines = []

    def error(self, message):
        self.lines.append(message)

    def health(self, *args, **kwargs):
        return None


def _watchdog(trainer, **policy):
    return MACTFWatchdog(
        checkpoint=trainer.checkpoint, logger=Sink(), policy=WatchdogPolicy(dead_branch_patience=2, **policy)
    )


def _context(trainer):
    trainer.run_epoch(0)
    return trainer._health_context(trainer.history[-1] if trainer.history else _result(trainer))


def _result(trainer):
    from mactf.training.protocol import EpochResult

    return EpochResult(epoch=0, steps=1, mean_loss=1.0, metrics={}, completed=True)


@pytest.fixture()
def trainer(settings, tmp_path):
    mutation_settings(settings)
    return mutable_trainer(settings, tmp_path, peers=3)


def test_prune_branch_goes_through_the_engine(trainer):
    watchdog = _watchdog(trainer)
    context = _context(trainer).copies(dead_branches={"peer-3": 5})
    assert context.pruner is not None
    assert watchdog.prune_branch(context) == 1
    (entry,) = trainer.mutation_journal
    assert entry["source"] == "watchdog" and entry["plan"]["ops"][0]["op"] == "purge"
    assert "peer-3" not in trainer.peers and "peer-3" not in trainer.tree
    assert trainer.path_state.audit() == []
    trainer.run_epoch(1)                                                    # the run is whole: this was M-G2
    # below the patience, nothing happens
    assert watchdog.prune_branch(context.copies(dead_branches={"peer-2": 1})) == 0
    assert "peer-2" in trainer.peers


def test_watchdog_prune_off_prunes_nothing(settings, tmp_path):
    mutation_settings(settings, WATCHDOG_PRUNE="off")
    trainer = mutable_trainer(settings, tmp_path, peers=3)
    context = _context(trainer).copies(dead_branches={"peer-3": 9})
    assert _watchdog(trainer).prune_branch(context) == 0
    assert "peer-3" in trainer.peers and trainer.mutation_journal == []
    assert any("WATCHDOG_PRUNE is off" in line for line in trainer.failures)


def test_watchdog_prune_retire_keeps_the_branch(settings, tmp_path):
    mutation_settings(settings, WATCHDOG_PRUNE="retire")
    trainer = mutable_trainer(settings, tmp_path, peers=3)
    context = _context(trainer).copies(dead_branches={"peer-3": 9})
    assert _watchdog(trainer).prune_branch(context) == 1
    assert "peer-3" in trainer.peers and trainer.freeze.is_retired("peer-3")
    trainer.revive_peer("peer-3")


def test_the_watchdog_may_prune_even_when_mutation_is_disabled(settings, tmp_path):
    """12.8.7: the alternative is the corruption of M-G2; every other source needs ENABLED."""
    mutation_settings(settings)
    settings["TRAINER"]["MUTATION"]["ENABLED"] = False
    trainer = mutable_trainer(settings, tmp_path, peers=3)
    context = _context(trainer).copies(dead_branches={"peer-3": 9})
    assert _watchdog(trainer).prune_branch(context) == 1
    assert trainer.mutation_journal[0]["source"] == "watchdog" and trainer.path_state.audit() == []


def test_prune_records_e082_without_ending_the_run(settings, tmp_path):
    mutation_settings(settings, PROTECTED=["peer-3"])
    trainer = mutable_trainer(settings, tmp_path, peers=3)
    context = _context(trainer).copies(dead_branches={"peer-3": 9})
    assert _watchdog(trainer).prune_branch(context) == 0
    assert "peer-3" in trainer.peers and any("E082" in line for line in trainer.failures)
    trainer.run_epoch(1)                                                    # and the run goes on
    # the last-root rule and a distillation blocker are refused the same way
    mutation_settings(settings, MIN_ACTIVE_ROOTS=3)
    trainer = mutable_trainer(settings, tmp_path / "two", peers=3)
    context = _context(trainer).copies(dead_branches={"peer-1": 9})
    assert _watchdog(trainer).prune_branch(context) == 0 and any("E082" in line for line in trainer.failures)


def test_a_hand_built_context_still_uses_remove_node(trainer):
    watchdog = _watchdog(trainer)
    tree = trainer.tree.clone()
    context = HealthContext(dead_branches={"peer-3": 9}, extras={"tree": tree})
    assert watchdog.prune_branch(context) == 1 and "peer-3" not in tree and "peer-3" in trainer.tree


def test_health_context_counts_only_active_nodes(trainer):
    context = _context(trainer)
    assert (context.tree_nodes, context.retired_nodes, context.mutations, context.topology_epoch) == (3, 0, 0, 0)
    assert context.loss_baseline == 0 and context.initial_tree_nodes == 3
    trainer.retire_peer("peer-3")
    trainer.add_peer("peer-4", like="peer-1")
    context = trainer._health_context(trainer.history[-1] if trainer.history else _result(trainer))
    assert (context.tree_nodes, context.retired_nodes, context.mutations, context.topology_epoch) == (3, 1, 2, 2)
    assert "peer-3" not in context.scores and context.extras["tree"] is trainer.tree
    assert context.max_entropy == pytest.approx(__import__("math").log(3))      # three roots in play, not four


def test_loss_probe_judges_the_current_cohort():
    probe = {p.name: p for p in default_probes(WatchdogPolicy())}["loss_trend"]
    spiked = HealthContext(losses=[1.0, 1.0, 1.0, 1.0, 1.0, 50.0])
    assert probe.probe(spiked).status is WatchdogStatus.CRITICAL
    # the same numbers, but the spike is the first loss on a new topology: nothing to compare with
    assert probe.probe(HealthContext(losses=[1.0, 1.0, 1.0, 1.0, 1.0, 50.0], loss_baseline=5)).status is WatchdogStatus.HEALTHY
    # the new cohort is judged against its own trailing median
    assert probe.probe(HealthContext(losses=[1.0] * 5 + [9.0, 9.0, 9.0, 300.0], loss_baseline=5)).status is WatchdogStatus.CRITICAL
    assert probe.probe(HealthContext(losses=[1.0] * 5 + [9.0, 9.0, 9.0, 9.5], loss_baseline=5)).status is WatchdogStatus.HEALTHY
    # loss_baseline=0 is byte-identical to the unsliced probe, and NaN is a NaN whatever the cohort
    for losses in ([1.0, 2.0], [3.0, 1.0, 2.0, 40.0], [1.0]):
        assert probe.probe(HealthContext(losses=losses)) == probe.probe(HealthContext(losses=losses, loss_baseline=0))
    nan = {p.name: p for p in default_probes(WatchdogPolicy())}["nan_loss"]
    assert nan.probe(HealthContext(losses=[float("nan"), 1.0], loss_baseline=5)).status is WatchdogStatus.CRITICAL


def test_a_mutation_moves_the_baselines(trainer):
    trainer.run_epoch(0)
    trainer.history.append(_result(trainer))
    trainer._epoch_seconds[:] = [0.1]
    rss = trainer._initial_rss
    trainer.add_peer("peer-4", like="peer-1")
    assert trainer._loss_baseline == 1 and trainer._rate_baseline == 1
    if rss is not None:
        assert trainer._initial_rss > rss                                   # a new peer is not a leak
    before = trainer._initial_rss
    trainer.retire_peer("peer-4")
    assert trainer._initial_rss == before                                    # memory a retire frees is not credited back


def test_tree_growth_probe():
    probe = TreeGrowthProbe(WatchdogPolicy(max_tree_growth_ratio=2.0))
    assert probe.probe(HealthContext(tree_nodes=7, initial_tree_nodes=3, mutations=4)).status is WatchdogStatus.DEGRADED
    assert probe.probe(HealthContext(tree_nodes=6, initial_tree_nodes=3)).status is WatchdogStatus.HEALTHY    # == ratio
    assert probe.probe(HealthContext(tree_nodes=1, initial_tree_nodes=3)).status is WatchdogStatus.HEALTHY    # never on shrinkage
    assert probe.probe(HealthContext()).status is WatchdogStatus.HEALTHY
    assert probe.probe(HealthContext(tree_nodes=5, initial_tree_nodes=0)).status is WatchdogStatus.HEALTHY
    assert "tree_growth" in [p.name for p in default_probes(WatchdogPolicy())][-1]
    result = probe.probe(HealthContext(tree_nodes=7, initial_tree_nodes=3, retired_nodes=1, mutations=4, topology_epoch=4))
    assert result.metadata == {"nodes": 7, "initial_nodes": 3, "retired": 1, "mutations": 4, "topology_epoch": 4}


def test_early_stopping_starts_over_on_a_mutation(trainer):
    stopper = EarlyStopping(patience=3)
    stopper.best, stopper.bad_epochs = 0.5, 2
    trainer.add_callback(stopper)
    trainer.add_peer("peer-4", like="peer-1")
    assert stopper.best is None and stopper.bad_epochs == 0


def test_the_mutation_is_logged(trainer):
    records = []
    trainer.logger = SimpleNamespace(mutation=lambda report, **kw: records.append(report), epoch=lambda *a, **k: None,
                                     flush=lambda: None)
    trainer.add_peer("peer-4", like="peer-1")
    assert [r.added for r in records] == [("peer-4",)]


def test_the_logger_writes_a_mutation_record():
    from mactf.training.logging import MACTFLogger, MemorySink, MutationRecord

    sink = MemorySink()
    logger = MACTFLogger(None, run_id="r", sinks=[sink], async_write=False)
    logger.mutation({"label": "grow", "source": "api", "epoch": 3, "added": ["p"], "warnings": ["W042: x"]})
    logger.flush()
    (record,) = [r for r in sink.records if isinstance(r, MutationRecord)]
    assert record.kind == "mutation" and record.level == "warning" and record.epoch == 3
    assert record.fields["added"] == ["p"] and "grow" in record.message
