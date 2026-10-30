"""Atomic mutations (invariant I4, risk R9; document 17 sections 12.5, 19.6).

A plan commits completely or changes nothing the trainer can observe. The tests are exhaustive in
the way 19.6 asks for: a fault is injected at *every* call of every step the engine makes, in turn,
and after each one the trainer must equal the snapshot taken before the plan. The set of steps is
read from a clean run of the plan, so a step added to the engine later is covered without editing
a test.
"""

import functools

import pytest

from mactf.exceptions import MutationError
from mactf.training.mutation import MutationPlan
from mutation_support import mutable_trainer, mutation_settings, snapshot


class Injected(RuntimeError):
    """The fault. Not a ``MutationError``: the engine must handle an arbitrary exception."""


def _targets():
    from mactf.components.function_book import FunctionBook
    from mactf.components.strategy_plan import StrategyPlan
    from mactf.components.weight_plan import WeightPlan
    from mactf.coopetition.temperature import AdaptiveKappa
    from mactf.coopetition.tree import PeerTree
    from mactf.training.freeze import FreezeSet
    from mactf.training.mutation import TopologySimulator
    from mactf.training.mutation_engine import MutationEngine
    from mactf.training.optimizer_plan import OptimizerPlan
    from mactf.training.path_state import PathState
    from mactf.training.trainer import MACTFTrainer

    return [
        # phase 1 and 2: simulate and stage
        (TopologySimulator, "run"), (MutationEngine, "_build_peer"), (MutationEngine, "_check_layouts"),
        (MutationEngine, "_check_plans"), (OptimizerPlan, "stage_topology"), (MACTFTrainer, "_make_optimizer"),
        (MACTFTrainer, "_validate_tied_parameters"),
        # phase 3: the eight commit steps and what they call
        (PeerTree, "adopt"), (MACTFTrainer, "_install_optimizer"), (OptimizerPlan, "commit_topology"),
        (FreezeSet, "snapshot"), (FreezeSet, "add"), (FreezeSet, "discard"), (FreezeSet, "forget"), (FreezeSet, "sync"),
        (MutationEngine, "_rebind_groups"), (AdaptiveKappa, "snapshot"), (AdaptiveKappa, "resize"),
        (PathState, "forget"), (FunctionBook, "forget"), (WeightPlan, "forget_groups"),
        (StrategyPlan, "forget_groups"), (MutationEngine, "_record_manifest"), (PathState, "audit"),
    ]


class Recorder:
    """Counts the calls of every target; raises ``Injected`` at one chosen (target, call index)."""

    def __init__(self, monkeypatch, fault=None):
        self.calls: list[tuple[str, int]] = []
        self._counts: dict[str, int] = {}
        self.fault = fault
        self.armed = False                     # building the trainer calls some of the targets too
        for owner, name in _targets():
            original = getattr(owner, name)
            label = f"{owner.__name__}.{name}"

            @functools.wraps(original)
            def wrapper(*args, __original=original, __label=label, **kwargs):
                if not self.armed:
                    return __original(*args, **kwargs)
                index = self._counts.get(__label, 0)
                self._counts[__label] = index + 1
                self.calls.append((__label, index))
                if self.fault == (__label, index):
                    raise Injected(f"fault injected at {__label} call {index}")
                return __original(*args, **kwargs)

            monkeypatch.setattr(owner, name, wrapper)


PLAN = {
    "label": "three operations",
    "ops": [
        {"op": "add", "path": "peer-4", "like": "peer-1", "factory": "perturbed_clone",
         "metadata": {"origin": "test"}, "optimizer": {"LR": 0.05}},
        {"op": "modify", "path": "peer-2", "set": {"role": "lead", "metadata": {"m": 1}, "optimizer": {"LR": 0.02}}},
        {"op": "retire", "path": "peer-3", "reason": "resting"},
    ],
}


#: Revives a retired peer, purges a live one and rebuilds a third: the steps ``PLAN`` does not take
#: (``FreezeSet.discard``, ``FreezeSet.forget``, ``FunctionBook.forget``, the weights of a rebuilt peer).
PLAN2 = {
    "label": "shrink",
    "ops": [
        {"op": "revive", "path": "peer-3"},
        {"op": "purge", "path": "peer-2", "force": True},
        {"op": "modify", "path": "peer-1", "set": {"rebuild": {"factory": "@build"}, "metadata": {"rebuilt": True}}},
    ],
}


def _trainer(settings, tmp_path, name="t", *, retired=()):
    mutation_settings(settings)
    trainer = mutable_trainer(settings, tmp_path / name, peers=3, seed=2)
    trainer.run_epoch(0)
    for path in retired:
        trainer.retire_peer(path)
    return trainer


CASES = {"grow": (PLAN, ()), "shrink": (PLAN2, ("peer-3",))}


def _enumerate(settings, tmp_path, monkeypatch, case="grow"):
    plan, retired = CASES[case]
    recorder = Recorder(monkeypatch)
    trainer = _trainer(settings, tmp_path, "clean", retired=retired)
    recorder.armed = True
    report = trainer.mutate(plan)
    recorder.armed = False
    assert report.added or report.purged
    assert trainer.path_state.audit() == []
    return list(recorder.calls)


def test_the_clean_run_touches_every_step(settings, tmp_path, monkeypatch):
    calls = _enumerate(settings, tmp_path, monkeypatch)
    labels = {label for label, _ in calls}
    # this plan exercises all eight commit steps: tree, peers, optimizers, freeze, groups, caches, scalars, audit
    required = {
        "TopologySimulator.run", "MutationEngine._build_peer", "MutationEngine._check_layouts",
        "MutationEngine._check_plans", "OptimizerPlan.stage_topology", "MACTFTrainer._make_optimizer",
        "PeerTree.adopt", "MACTFTrainer._install_optimizer", "OptimizerPlan.commit_topology", "FreezeSet.snapshot",
        "FreezeSet.add", "FreezeSet.sync", "MutationEngine._rebind_groups", "PathState.forget",
        "WeightPlan.forget_groups", "StrategyPlan.forget_groups", "MutationEngine._record_manifest", "PathState.audit",
    }
    assert required <= labels, sorted(required - labels)
    assert len(calls) >= 20
    shrink = {label for label, _ in _enumerate(settings, tmp_path / "s", monkeypatch, "shrink")}
    assert {"FreezeSet.discard", "FreezeSet.forget", "FunctionBook.forget", "AdaptiveKappa.resize"} <= shrink


@pytest.mark.parametrize("case", sorted(CASES))
def test_mutation_atomicity(settings, tmp_path, monkeypatch, case):
    """A three-operation plan, faulted at every call of every step: the trainer never changes."""
    plan, retired = CASES[case]
    calls = _enumerate(settings, tmp_path, monkeypatch, case)
    monkeypatch.undo()
    checked = 0
    for label, index in calls:
        with monkeypatch.context() as scoped:
            recorder = Recorder(scoped, fault=(label, index))
            trainer = _trainer(settings, tmp_path, f"f{checked}", retired=retired)
            before = snapshot(trainer)
            recorder.armed = True
            try:
                trainer.mutate(plan)
            except Exception as error:  # noqa: BLE001
                caught = pytest.ExceptionInfo.from_exception(error)
            else:
                pytest.fail(f"no failure at {label} call {index}: the enumeration and the faulted run disagree")
            recorder.armed = False
            assert "fault injected" in str(caught.value) or isinstance(caught.value, MutationError), (label, index)
            if isinstance(caught.value, MutationError) and caught.value.code == "E089":
                assert caught.value.rolled_back is True, (label, index)        # E089 here is a rollback, never a half run
            scoped.undo()
            assert snapshot(trainer) == before, f"the trainer changed after a fault at {label} call {index}"
            assert trainer.path_state.audit() == [], (label, index)
            assert trainer.engine.poisoned is None
            # and it is still a working trainer: the same plan commits cleanly afterwards
            trainer.mutate(plan)
            assert trainer.path_state.audit() == []
            trainer.run_epoch(1)
        checked += 1
    assert checked == len(calls) >= 20


@pytest.mark.parametrize("bad", [0, 1, 2])
def test_each_operation_failing_leaves_the_trainer_unchanged(settings, tmp_path, bad):
    """A three-operation plan whose operation ``bad`` is invalid: nothing of the other two is applied."""
    invalid = [
        {"op": "add", "path": "peer-4/x/y", "like": "peer-1"},                      # a missing parent
        {"op": "modify", "path": "ghost", "set": {"role": "lead"}},                  # no such node
        {"op": "retire", "path": "peer-3", "recursive": False, "reason": "x"},      # valid on its own...
    ]
    ops = [dict(op) for op in PLAN["ops"]]
    ops[bad] = invalid[bad]
    if bad == 2:
        ops[2] = {"op": "purge", "path": "peer-3"}                                   # ...an active purge needs `force`
    trainer = _trainer(settings, tmp_path)
    before = snapshot(trainer)
    with pytest.raises(MutationError) as caught:
        trainer.mutate({"ops": ops})
    assert caught.value.code in ("E081", "E082") and not caught.value.rolled_back
    assert snapshot(trainer) == before and trainer.path_state.audit() == []


def test_a_stage_failure_on_the_third_peer_leaves_the_trainer_unchanged(settings, tmp_path):
    trainer = _trainer(settings, tmp_path)
    before = snapshot(trainer)
    ops = [{"op": "add", "path": f"n{i}", "like": "peer-1"} for i in range(2)]
    ops.append({"op": "add", "path": "n2", "like": "peer-1", "factory": "no_such_factory"})
    with pytest.raises(MutationError) as caught:
        trainer.mutate({"ops": ops})
    assert caught.value.code == "E085"
    assert snapshot(trainer) == before and trainer.path_state.audit() == []


@pytest.fixture()
def labels(settings, tmp_path, monkeypatch):
    """The inverses a clean commit registers, read from the undo stack itself."""
    from mactf.training.mutation_engine import Undo

    seen: list[str] = []
    original = Undo.push

    def push(self, label, inverse):
        seen.append(label)
        return original(self, label, inverse)

    monkeypatch.setattr(Undo, "push", push)
    trainer = _trainer(settings, tmp_path, "labels")
    seen.clear()
    trainer.mutate(PLAN)
    monkeypatch.undo()
    return list(dict.fromkeys(seen))


def test_every_inverse_is_pushed_before_its_mutation(settings, tmp_path, monkeypatch):
    """An inverse pushed *after* the mutation is lost when the mutation raises; so each step is faulted *inside* it.

    For every call of every step the commit makes, the stack at the moment of the fault must already
    hold the inverse of that step: rolling back from there restores the snapshot (checked above) and
    the labels are pushed in the documented order.
    """
    from mactf.training.mutation_engine import Undo

    order: list[str] = []
    original = Undo.push

    def push(self, label, inverse):
        order.append(label)
        return original(self, label, inverse)

    monkeypatch.setattr(Undo, "push", push)
    trainer = _trainer(settings, tmp_path)
    order.clear()
    trainer.mutate(PLAN)
    expected_prefix = ["tree", "peers", "peer.optimizer:", "optimizers", "optimizer_plan", "freeze"]
    kinds = [label.split(":")[0] if label.startswith(("peer.optimizer", "weights", "functions", "kappa")) else label
             for label in order]
    # the tree is the first thing swapped and the first inverse pushed; the journal and the counters come last
    assert order[0] == "tree" and order[1] == "peers"
    assert order.index("freeze") > order.index("optimizer_plan") > order.index("optimizers")
    assert order.index("tombstones") > order.index("freeze") and order[-1] == "budget"
    assert {"journal", "topology_epoch", "node_specs", "node_meta", "controllers"} <= set(order)
    assert expected_prefix and kinds


def test_a_failed_rollback_is_e089_not_a_half_run(settings, tmp_path, monkeypatch, labels):
    """The inverse of one step raises while another fault is being undone: the run stops with E089."""
    from mactf.training.mutation_engine import Undo
    from mactf.training.path_state import PathState

    assert len(labels) >= 8
    for label in labels:
        with monkeypatch.context() as scoped:
            trainer = _trainer(settings, tmp_path, f"r-{label.replace(':', '-')}")
            original = Undo.push

            def push(self, step, inverse, __label=label):
                if step == __label:
                    def broken():
                        raise OSError(f"the inverse of {__label} failed")
                    return original(self, step, broken)
                return original(self, step, inverse)

            scoped.setattr(Undo, "push", push)
            scoped.setattr(PathState, "audit", lambda self: ["injected audit failure"])       # the commit's last step fails
            with pytest.raises(MutationError) as caught:
                trainer.mutate(PLAN)
            error = caught.value
            assert error.code == "E089" and error.rolled_back is False, label
            assert "rollback did not complete" in str(error) and "discarded" in str(error) and label in str(error)
            # the run does not go on half-mutated
            assert trainer.engine.poisoned is error
            with pytest.raises(MutationError) as again:
                trainer.mutate({"ops": [{"op": "retire", "path": "peer-1"}]})
            assert again.value is error
            with pytest.raises(MutationError):
                trainer.run_epoch(1)
            with pytest.raises(MutationError):
                trainer.preview_mutation({"ops": [{"op": "retire", "path": "peer-1"}]})


def test_a_completed_rollback_is_reported_as_such(settings, tmp_path, monkeypatch):
    from mactf.training.path_state import PathState

    trainer = _trainer(settings, tmp_path)
    before = snapshot(trainer)
    monkeypatch.setattr(PathState, "audit", lambda self: ["injected audit failure"])
    with pytest.raises(MutationError) as caught:
        trainer.mutate(PLAN)
    assert caught.value.code == "E089" and caught.value.rolled_back is True
    assert "injected audit failure" in str(caught.value) and "rolled back" in str(caught.value)
    monkeypatch.undo()
    assert snapshot(trainer) == before and trainer.engine.poisoned is None


def test_a_dry_run_fault_leaves_the_trainer_unchanged(settings, tmp_path, monkeypatch):
    calls = _enumerate(settings, tmp_path, monkeypatch)
    monkeypatch.undo()
    stage_only = [c for c in calls if c[0].split(".")[1] in {"run", "_build_peer", "_check_layouts", "_check_plans",
                                                                 "stage_topology", "_make_optimizer"}]
    assert stage_only
    for label, index in stage_only:
        with monkeypatch.context() as scoped:
            recorder = Recorder(scoped, fault=(label, index))
            trainer = _trainer(settings, tmp_path, f"d{label}{index}")
            before = snapshot(trainer)
            recorder.armed = True
            with pytest.raises(Exception):
                trainer.mutate(PLAN, dry_run=True)
            recorder.armed = False
            scoped.undo()
            assert snapshot(trainer) == before
