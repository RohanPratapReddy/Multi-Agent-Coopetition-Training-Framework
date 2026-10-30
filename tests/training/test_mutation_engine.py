"""The mutation engine: the gate, dry runs, the journal, the queue (document 17 section 12.5, rows 21-30)."""

import pytest
import torch

from mactf.exceptions import MutationError
from mactf.training.callbacks import Callback
from mactf.training.mutation import MutationPlan, add_peer, retire_peer
from mutation_support import mutable_trainer, mutation_settings, parameters, snapshot


@pytest.fixture()
def trainer(settings, tmp_path):
    mutation_settings(settings)
    return mutable_trainer(settings, tmp_path, peers=3)


def test_dry_run_installs_nothing(trainer):
    before = snapshot(trainer)
    report = trainer.mutate(
        MutationPlan.of(add_peer("peer-4", like="peer-1", factory="perturbed_clone"), retire_peer("peer-3")),
        dry_run=True,
    )
    assert report.dry_run and report.added == ("peer-4",) and report.retired == ("peer-3",)
    assert report.optimizers_rebuilt == ("peer-4",)
    assert snapshot(trainer) == before and trainer.mutation_journal == []
    # a dry run is a *preview* too, and runs every check
    with pytest.raises(MutationError) as caught:
        trainer.preview_mutation(MutationPlan.of(add_peer("peer-4", like="ghost")))
    assert caught.value.code == "E083"


def test_gate_refuses_outside_a_boundary(settings, tmp_path):
    plan = MutationPlan.of(add_peer("peer-4", like="peer-1"))
    # disabled
    settings["TRAINING"].update({"EPOCHS": 2, "BATCH_SIZE": 8, "STATE_DIM": 4, "ACTION_DIM": 3, "BOOTSTRAP": False})
    settings["SANDBOX"]["PROVIDER"] = "local"
    off = mutable_trainer(settings, tmp_path / "off", peers=3)
    with pytest.raises(MutationError) as caught:
        off.mutate(plan)
    assert caught.value.code == "E088" and "ENABLED" in str(caught.value)
    # not built
    mutation_settings(settings)
    unbuilt = mutable_trainer(settings, tmp_path / "unbuilt", peers=3, build=False)
    with pytest.raises(MutationError, match="not built"):
        unbuilt.mutate(plan)

    on = mutable_trainer(settings, tmp_path / "on", peers=3)
    # the epoch phase: a callback that calls mutate directly is refused, and the epoch survives it
    seen = []

    class Eager(Callback):
        name = "eager"

        def on_epoch_start(self, trainer, epoch):
            with pytest.raises(MutationError) as inner:
                trainer.mutate(plan)
            seen.append((trainer._phase, inner.value.code))

    on.add_callback(Eager())
    on.run_epoch(0)
    assert seen == [("epoch", "E088")] and "peer-4" not in on.peers
    # the observe phase is a boundary: a plan from on_epoch_end commits
    class Late(Callback):
        name = "late"

        def on_epoch_end(self, trainer, result):
            trainer.mutate(plan)

    on.remove_callback("eager")
    on.add_callback(Late())
    on.run_epoch(1)
    assert "peer-4" in on.peers and on._phase == "idle"


def test_gate_refuses_a_spent_budget_and_a_blocker(settings, tmp_path):
    mutation_settings(settings, MAX_PLANS_PER_EPOCH=1)
    trainer = mutable_trainer(settings, tmp_path, peers=3)
    trainer.add_peer("peer-4", like="peer-1")
    with pytest.raises(MutationError, match="MAX_PLANS_PER_EPOCH"):
        trainer.add_peer("peer-5", like="peer-1")
    trainer.run_epoch(0)                                   # the budget resets with the epoch
    trainer.add_peer("peer-5", like="peer-1")
    assert {"peer-4", "peer-5"} <= set(trainer.peers)

    class Blocked:
        name = "blocked"

        def train_step(self, node, batch):
            raise AssertionError("not reached")

    mutation_settings(settings)
    blocked = mutable_trainer(settings, tmp_path / "b", peers=3)
    blocked._distillation = type("D", (), {"enabled": True})()
    reasons = blocked.mutation_blockers()
    assert reasons and "DISTILLATION" in reasons[0]
    with pytest.raises(MutationError, match="cannot be mutated"):
        blocked.add_peer("peer-4", like="peer-1")


def test_a_failure_in_commit_rolls_everything_back(trainer, monkeypatch):
    """A factory that raises on the *third* node leaves tree, peers, optimizers, freeze, caches and journal alone."""
    import mactf.training.peer_factories as pf

    calls = {"n": 0}
    original = pf.clone

    def flaky(spec, source=None):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("the third node cannot be built")
        return original(spec, source)

    from mactf.components.registry import HUB

    registry = HUB.registry("peer_factory")
    existing = registry.get("clone")
    monkeypatch.setitem(registry._entries, "clone", type(existing)(**{**existing.__dict__, "factory": flaky}))
    before = snapshot(trainer)
    plan = MutationPlan.of(*(add_peer(f"n{i}", like="peer-1", factory="clone") for i in range(1, 4)))
    with pytest.raises(MutationError) as caught:
        trainer.mutate(plan)
    assert caught.value.code == "E085" and "third node" in str(caught.value)
    assert snapshot(trainer) == before and trainer.path_state.audit() == []


def test_the_post_commit_audit_runs(trainer, monkeypatch):
    """A container deliberately left stale is E089, and the commit is rolled back."""
    before = snapshot(trainer)
    original = type(trainer.path_state).forget

    def leaky(self, **kwargs):
        original(self, **kwargs)
        self.trainer.signal_by_peer["ghost"] = 1.0                 # a path the tree does not have

    monkeypatch.setattr(type(trainer.path_state), "forget", leaky)
    with pytest.raises(MutationError) as caught:
        trainer.retire_peer("peer-3")
    assert caught.value.code == "E089" and caught.value.rolled_back and "ghost" in str(caught.value)
    monkeypatch.undo()
    trainer.signal_by_peer.pop("ghost", None)
    assert snapshot(trainer) == before


def test_journal_entry_names_the_epoch_it_ran_in(trainer):
    class AtTop(Callback):
        name = "at_top"

    trainer.request_mutation(MutationPlan.of(add_peer("peer-4", like="peer-1")), source="callback")
    trainer.run_epoch(5)
    (entry,) = trainer.mutation_journal
    assert entry["epoch"] == 5 and entry["source"] == "callback" and entry["topology_epoch"] == 1
    # called between epochs, a plan names the epoch that just ran
    trainer.add_peer("peer-5", like="peer-1")
    assert trainer.mutation_journal[-1]["epoch"] == 5


def test_topology_epoch_increments_once_per_plan(trainer):
    assert trainer.topology_epoch == 0
    trainer.mutate(MutationPlan.of(add_peer("peer-4", like="peer-1"), add_peer("peer-5", like="peer-1")))
    assert trainer.topology_epoch == 1
    trainer.retire_peer("peer-4")
    assert trainer.topology_epoch == 2 and [e["seq"] for e in trainer.mutation_journal] == [0, 1]
    # a dry run and a refused plan do not count
    trainer.preview_mutation(MutationPlan.of(retire_peer("peer-5")))
    with pytest.raises(MutationError):
        trainer.retire_peer("ghost")
    assert trainer.topology_epoch == 2


def test_requested_plans_run_at_the_top_of_the_next_epoch(trainer):
    seen = {}

    class Probe(Callback):
        name = "probe"

        def on_epoch_start(self, trainer, epoch):
            seen["peers_in_epoch"] = sorted(trainer.active_tree.root_paths)

    trainer.add_callback(Probe())
    parsed = trainer.request_mutation(MutationPlan.of(add_peer("peer-4", like="peer-1")), source="test")
    assert isinstance(parsed, MutationPlan) and "peer-4" not in trainer.peers
    assert trainer.topology()["pending"] and trainer.topology()["pending"][0]["source"] == "test"
    trainer.run_epoch(3)
    assert seen["peers_in_epoch"] == ["peer-1", "peer-2", "peer-3", "peer-4"]
    assert trainer.mutation_journal[-1]["epoch"] == 3 and trainer.topology()["pending"] == []
    # a malformed plan fails when it is requested, not at the next epoch
    with pytest.raises(MutationError) as caught:
        trainer.request_mutation({"ops": [{"op": "explode", "path": "p"}]})
    assert caught.value.code == "E080"


def test_a_refused_queued_plan_is_recorded_not_raised(trainer):
    trainer.request_mutation(MutationPlan.of(retire_peer("ghost")), source="callback")
    before = len(trainer.failures)
    result = trainer.run_epoch(0)
    assert result.completed and len(trainer.failures) == before + 1
    assert "ghost" in trainer.failures[-1] and "callback" in trainer.failures[-1]
    assert trainer.mutation_journal == []


def test_a_scheduled_plan_runs_once_across_a_resume(settings, tmp_path):
    schedule = [{"AT_EPOCH": 1, "LABEL": "grow", "OPS": [{"op": "add", "path": "peer-4", "like": "peer-1"}]}]
    mutation_settings(settings, SCHEDULE=schedule)
    trainer = mutable_trainer(settings, tmp_path, peers=3)
    trainer.run_epoch(0)
    assert trainer.mutation_journal == []
    trainer.run_epoch(1)
    assert [e["source"] for e in trainer.mutation_journal] == ["schedule"]
    assert trainer.engine.apply_pending(1) == []                    # the same epoch again: already journaled
    assert len(trainer.mutation_journal) == 1
    trainer.run_epoch(2)
    assert len(trainer.mutation_journal) == 1                       # not its epoch any more
    # the digest key is (epoch, plan): the same plan scheduled for two epochs runs at both
    other = [{"AT_EPOCH": 0, "OPS": [{"op": "modify", "path": "peer-1", "set": {"metadata": {"k": 1}}}]},
             {"AT_EPOCH": 1, "OPS": [{"op": "modify", "path": "peer-1", "set": {"metadata": {"k": 1}}}]}]
    mutation_settings(settings, SCHEDULE=other)
    twice = mutable_trainer(settings, tmp_path / "twice", peers=3)
    twice.run_epoch(0)
    twice.run_epoch(1)
    assert [e["epoch"] for e in twice.mutation_journal] == [0, 1]


def test_a_failed_rollback_poisons_the_trainer_and_the_next_epoch(trainer, monkeypatch):
    from mactf.training.freeze import FreezeSet

    boom = {"n": 0}
    real = trainer.engine._rebind_groups

    def failing(simulation, undo):
        raise RuntimeError("fault after the freeze step")

    monkeypatch.setattr(trainer.engine, "_rebind_groups", failing)
    monkeypatch.setattr(FreezeSet, "restore", lambda self, snapshot: (_ for _ in ()).throw(OSError("undo failed")))
    with pytest.raises(MutationError) as caught:
        trainer.retire_peer("peer-3")
    assert caught.value.code == "E089" and caught.value.rolled_back is False
    assert "must be discarded" in str(caught.value) and "undo failed" in str(caught.value)
    # the run does not go on half-mutated: the next plan and the next epoch both refuse
    with pytest.raises(MutationError) as again:
        trainer.add_peer("peer-9", like="peer-1")
    assert again.value is caught.value
    with pytest.raises(MutationError):
        trainer.run_epoch(0)
    del boom, real


def test_a_failed_rollback_in_a_queued_plan_ends_the_run(trainer, monkeypatch):
    from mactf.training.freeze import FreezeSet

    monkeypatch.setattr(trainer.engine, "_rebind_groups", lambda simulation, undo: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(FreezeSet, "restore", lambda self, snapshot: (_ for _ in ()).throw(OSError("undo failed")))
    trainer.request_mutation(MutationPlan.of(retire_peer("peer-3")), source="callback")
    with pytest.raises(MutationError) as caught:
        trainer.run_epoch(0)
    assert caught.value.code == "E089" and not caught.value.rolled_back
    assert trainer.failures == [] or all("poisoned" not in line for line in trainer.failures)


def test_the_report_and_the_callbacks_see_a_committed_plan(trainer):
    heard = []

    class Listener(Callback):
        name = "listener"

        def on_tree_mutated(self, trainer, report):
            heard.append((report.added, report.retired, trainer.path_state.audit()))

    from mactf import signals

    signalled = []
    receiver = lambda signal=None, sender=None, report=None, **_: signalled.append(report.plan_digest)
    signals.tree_mutated.connect(receiver, weak=False)
    try:
        trainer.add_callback(Listener())
        report = trainer.mutate(MutationPlan.of(add_peer("peer-4", like="peer-1"), retire_peer("peer-3")))
    finally:
        signals.tree_mutated.disconnect(receiver)
    assert heard == [(("peer-4",), ("peer-3",), [])] and signalled == [report.plan_digest]
    assert report.active_nodes == 3 and report.retired_nodes == 1 and report.changed and not report.dry_run
    assert report.before_digest != report.after_digest and report.warnings


def test_an_observer_that_raises_cannot_undo_a_commit(trainer):
    class Bad(Callback):
        name = "bad"

        def on_tree_mutated(self, trainer, report):
            raise RuntimeError("observer failed")

    trainer.add_callback(Bad())
    trainer.add_peer("peer-4", like="peer-1")
    assert "peer-4" in trainer.peers and any("observer failed" in line for line in trainer.failures)
    assert trainer.path_state.audit() == []


def test_mutation_never_consumes_the_runs_random_stream(trainer):
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    trainer.add_peer("peer-4", like="peer-1", factory="perturbed_clone")
    assert torch.equal(torch.get_rng_state(), before)
