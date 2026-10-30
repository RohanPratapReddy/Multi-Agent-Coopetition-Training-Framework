"""Optimizer entries, overrides, swaps and resume (document 17 sections 9.4 to 9.8; invariants O1-O10)."""

from __future__ import annotations

import copy
import logging

import pytest
import torch

from mactf.components import HUB, register_optimizer, register_scheduler
from mactf.components.kinds import Boundary
from mactf.exceptions import (
    ComponentLocked,
    ComponentMigrationError,
    ImproperlyConfigured,
    ValidationError,
)
from mactf.training.optimizer import MACTFOptimizer, NullOptimizer, RouterOptimizer
from mactf.training.optimizer_plan import (
    OptimizerPlan,
    compile_entry,
    merge_blocks,
    verify_entry,
)

from p2_support import Lion, configure, make_trainer, snapshot_parameters, step_then_flat, tune


@pytest.fixture
def parts(component_overlay):
    register_optimizer("Lion", Lion, version="1.0", scope="run")
    register_scheduler("step_then_flat", step_then_flat, version="1", scope="run")
    return component_overlay


def _trainer(settings, tmp_path, block=None, *, peers=2, epochs=2, **kwargs):
    tune(settings, EPOCHS=epochs)
    if block is not None:
        configure(settings, OPTIMIZER=block)
    return make_trainer(settings, tmp_path, peers=peers, **kwargs).build()


def _classes(trainer):
    return {path: optimizer.describe()["class"] for path, optimizer in trainer.optimizers.items()}


def _batch(**kwargs):
    generator = torch.Generator().manual_seed(4)
    base = {"states": torch.randn(8, 4, generator=generator), "targets": torch.zeros(8, dtype=torch.long),
            "signal": 0.5, "void": False, "floor": 0.0}
    base.update(kwargs)
    return base


def _base_lr(optimizer):
    """The rate the entry asked for, before a scheduler's factor at the current step."""
    group = optimizer.backend.param_groups[0]
    return group.get("initial_lr", group["lr"])


def _train(trainer, path="peer-1", times=1):
    for _ in range(times):
        trainer.train_step(trainer.tree.get(path), _batch())


# -- O1: nothing configured, nothing changes -------------------------------------------------------------------
def test_default_run_unchanged_the_inline_adam_and_a_plain_state(settings, tmp_path):
    trainer = _trainer(settings, tmp_path)
    plan = trainer.optimizer_plan
    assert plan.default.kind == "inline" and not plan.customized() and plan.history == ()
    assert set(_classes(trainer).values()) == {"Adam"}
    state = trainer.optimizers["peer-1"].state_dict()
    assert "spec_hash" not in state and "slot" not in state and "scheduler_name" not in state
    assert "optimizer_plan" not in trainer._trainer_state()
    assert trainer._optimizer_warnings == []


def test_authoring_only_the_plan_keys_does_not_author_the_default_optimizer(settings, tmp_path, parts):
    trainer = _trainer(settings, tmp_path, {"OVERRIDES": {"peer-2": {"TYPE": "torch", "CLASS": "SGD", "LR": 0.05}}})
    assert _classes(trainer) == {"peer-1": "Adam", "peer-2": "SGD"}                   # peer-1 keeps the legacy Adam
    assert trainer.optimizers["peer-1"].learning_rate == pytest.approx(
        float(trainer._training_settings()["LEARNING_RATE"]))
    assert "spec_hash" not in trainer.optimizers["peer-1"].state_dict()
    assert trainer.optimizers["peer-2"].state_dict()["spec_hash"] == trainer.optimizer_plan.select("peer-2").spec_hash


def test_the_authored_block_without_overrides_is_the_legacy_path(settings, tmp_path):
    trainer = _trainer(settings, tmp_path, {"CLASS": "SGD", "LR": 0.02, "WEIGHT_DECAY": 0.0})
    assert not trainer.optimizer_plan.customized() and set(_classes(trainer).values()) == {"SGD"}
    assert "spec_hash" not in trainer.optimizers["peer-1"].state_dict()


# -- O2 / O3: choosing and inheriting --------------------------------------------------------------------------------------
def test_override_specificity_exact_beats_glob_beats_default(settings, tmp_path, parts):
    trainer = _trainer(settings, tmp_path, {
        "CLASS": "AdamW", "LR": 1e-3,
        "OVERRIDES": {
            "peer-*": {"TYPE": "Lion", "LR": 3e-4},
            "peer-2": {"LR": 1e-4},                    # only the rate: the glob's Lion is NOT inherited by name
        }}, peers=3)
    classes = _classes(trainer)
    assert classes["peer-1"] == "Lion"
    assert classes["peer-2"] == "AdamW" and trainer.optimizers["peer-2"].learning_rate == pytest.approx(1e-4)
    assert classes["peer-3"] == "Lion"


def test_override_inherits_facade_keys_one_by_one(settings, tmp_path, parts):
    trainer = _trainer(settings, tmp_path, {
        "CLASS": "AdamW", "LR": 1e-3, "WEIGHT_DECAY": 0.05, "MAX_GRAD_NORM": 0.7, "GRAD_ACCUMULATION": 2,
        "SCHEDULER": "warmup_cosine", "SCHEDULER_KWARGS": {"warmup_steps": 1},
        "OVERRIDES": {"peer-2": {"LR": 5e-4}}})
    first, second = trainer.optimizers["peer-1"], trainer.optimizers["peer-2"]
    assert _base_lr(second) == pytest.approx(5e-4)               # the key it set
    assert second.describe()["weight_decay"] == 0.05                  # inherited
    assert second.max_grad_norm == 0.7 and second.grad_accumulation == 2
    assert second.scheduler_name == "warmup_cosine"
    assert _base_lr(first) != _base_lr(second)


def test_override_backend_replaces_backend_keys(settings, tmp_path, parts):
    """The default's ``PARAMS`` (Lion's betas) must not leak into an AdamW override (9.4.1)."""
    trainer = _trainer(settings, tmp_path, {
        "TYPE": "Lion", "LR": 3e-4, "WEIGHT_DECAY": 0.01, "PARAMS": {"betas": [0.9, 0.95]},
        "OVERRIDES": {"peer-2": {"TYPE": "torch", "CLASS": "AdamW", "LR": 1e-4}}})
    assert _classes(trainer) == {"peer-1": "Lion", "peer-2": "AdamW"}
    lion = trainer.optimizers["peer-1"].backend.optimizer.param_groups[0]
    assert tuple(lion["betas"]) == (0.9, 0.95) and lion["weight_decay"] == 0.01
    adamw = trainer.optimizers["peer-2"].backend.optimizer.param_groups[0]
    assert adamw["lr"] == 1e-4 and adamw["weight_decay"] == 0.01        # facade keys are inherited
    merged = merge_blocks({"TYPE": "Lion", "PARAMS": {"betas": [1, 2]}, "LR": 1.0},
                          {"PARAMS": {"extra": 1}})
    assert merged["PARAMS"] == {"betas": [1, 2], "extra": 1}              # no backend selector: params merge


def test_a_scheduler_null_in_an_override_clears_it():
    merged = merge_blocks({"SCHEDULER": "warmup_cosine", "SCHEDULER_KWARGS": {"warmup_steps": 1}, "LR": 1},
                          {"SCHEDULER": None})
    assert "SCHEDULER" not in merged and merged["SCHEDULER_KWARGS"] == {"warmup_steps": 1}


def test_e051_two_overrides_of_equal_specificity(settings, tmp_path, parts):
    with pytest.raises(ImproperlyConfigured, match="E051"):
        _trainer(settings, tmp_path, {"CLASS": "AdamW", "OVERRIDES": {
            "peer-?": {"LR": 1e-4}, "*eer-1": {"LR": 2e-4}}})


def test_e050_a_glob_that_matches_no_node(settings, tmp_path, parts):
    with pytest.raises(ImproperlyConfigured, match=r"E050.*ghost"):
        _trainer(settings, tmp_path, {"CLASS": "AdamW", "OVERRIDES": {"ghost-*": {"LR": 1e-4}}})
    trainer = _trainer(settings, tmp_path / "ok", {"CLASS": "AdamW", "STRICT_GLOBS": False,
                                                   "OVERRIDES": {"ghost-*": {"LR": 1e-4}}})
    assert any(w.startswith("W011") for w in trainer._optimizer_warnings)


# -- O4: an invalid entry never reaches a run ------------------------------------------------------------------------------
@pytest.mark.parametrize("block,code", [
    ({"TYPE": "Nonexistent"}, "unknown optimizer"),
    ({"CLASS": "AdamW", "PARAMS": {"nope": 1}}, "E057"),
    ({"TYPE": "Lion", "PARAMS": {"nope": 1}}, "E057"),
    ({"CLASS": "AdamW", "LR": 1e-3, "MAX_GRAD_NORM": 1.0, "SCHEDULER": "NoSuchScheduler"}, "unknown scheduler"),
    ({"CLASS": "AdamW", "SCHEDULER": "ReduceLROnPlateau"}, "E052"),
    ({"CLASS": "AdamW", "SCHEDULER": "warmup_cosine", "SCHEDULER_KWARGS": {"warmup_steps": 5, "total_steps": 3}}, "E053"),
    ({"CLASS": "AdamW", "SCHEDULER": "StepLR", "SCHEDULER_KWARGS": {"bogus": 1}}, "E053"),
    ({"TYPE": "scipy", "GRAD_ACCUMULATION": 3}, "E057"),
    ({"TYPE": "scipy", "SCHEDULER": "warmup_cosine"}, "E052"),
    ({"CLASS": "AdamW", "OVERRIDES": {"peer-1": {"MIGRATE": "sideways"}}}, "E057"),
    ({"CLASS": "AdamW", "LR": 1e-3, "PARAMS": {"lr": 1e-2}}, "E057"),
])
def test_bad_entry_fails_at_build(settings, tmp_path, parts, block, code):
    with pytest.raises((ImproperlyConfigured, ValidationError), match=code):
        _trainer(settings, tmp_path, block)


def test_bad_entry_in_an_override_names_its_slot(settings, tmp_path, parts):
    with pytest.raises(ImproperlyConfigured, match=r"OPTIMIZER.peer-2.*E057|E057.*OPTIMIZER.peer-2"):
        _trainer(settings, tmp_path, {"CLASS": "AdamW", "OVERRIDES": {"peer-2": {"PARAMS": {"nope": 1}}}})


def test_the_probe_catches_an_optimizer_that_breaks_at_step_one(settings, tmp_path, parts):
    class Exploding(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-3):
            super().__init__(params, dict(lr=lr))

        @torch.no_grad()
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group["params"]:
                    p.fill_(float("nan"))

    register_optimizer("Exploding", Exploding, scope="run")
    with pytest.raises(ImproperlyConfigured, match=r"E056.*O003"):
        _trainer(settings, tmp_path, {"TYPE": "Exploding", "LR": 1e-3})

    class Raising(Exploding):
        def step(self, closure=None):
            raise RuntimeError("no step for you")

    register_optimizer("Raising", Raising, scope="run")
    with pytest.raises(ImproperlyConfigured, match=r"E056.*O002.*no step for you"):
        _trainer(settings, tmp_path / "r", {"TYPE": "Raising", "LR": 1e-3})
    trainer = _trainer(settings, tmp_path / "off", {"TYPE": "Raising", "LR": 1e-3, "VERIFY": "off"})
    assert _classes(trainer)["peer-1"] == "Raising"                                     # the operator's choice


def test_verify_entry_skips_what_cannot_be_stepped_blind():
    assert verify_entry(compile_entry("OPTIMIZER", {"TYPE": "null"}, glob=None, horizon=4, written=None)) is None


# -- null, router, scheduler registration ------------------------------------------------------------------------------------------
def test_a_null_entry_freezes_its_node_and_never_moves_it(settings, tmp_path, parts):
    trainer = _trainer(settings, tmp_path, {"CLASS": "AdamW", "OVERRIDES": {"peer-2": {"TYPE": "null"}}})
    assert isinstance(trainer.optimizers["peer-2"].backend, NullOptimizer)
    assert trainer.freeze.reasons("peer-2") == ("optimizer:null",) and "peer-1" not in trainer.freeze
    assert trainer.tree.get("peer-2").status == "frozen"
    before = snapshot_parameters(trainer)
    _train(trainer, "peer-2", times=3)
    after = snapshot_parameters(trainer)
    assert all(torch.equal(before[k], after[k]) for k in before if k.startswith("peer-2"))
    _train(trainer, "peer-1")
    assert any(not torch.equal(before[k], snapshot_parameters(trainer)[k]) for k in before if k.startswith("peer-1"))


def test_type_none_is_a_synonym_for_null(settings, tmp_path):
    trainer = _trainer(settings, tmp_path, {"CLASS": "AdamW", "OVERRIDES": {"peer-2": {"TYPE": "none"}}})
    assert isinstance(trainer.optimizers["peer-2"].backend, NullOptimizer)


def test_all_null_is_refused(settings, tmp_path):
    with pytest.raises(ImproperlyConfigured, match="E054"):
        _trainer(settings, tmp_path, {"TYPE": "null"})
    with pytest.raises(ImproperlyConfigured, match="E054"):
        _trainer(settings, tmp_path / "b", {"CLASS": "AdamW", "OVERRIDES": {"peer-*": {"TYPE": "null"}}})


def test_a_router_is_described_by_data_and_survives_a_device_move(settings, tmp_path):
    block = {"TYPE": "router", "DEFAULT": {"TYPE": "AdamW", "LR": 1e-3},
             "ROUTES": [{"PATTERN": r".*policy\..*", "TYPE": "SGD", "LR": 0.1, "PRIORITY": 10}]}
    trainer = _trainer(settings, tmp_path, block)
    optimizer = trainer.optimizers["peer-1"]
    assert isinstance(optimizer.backend, RouterOptimizer)
    assigned = optimizer.backend.assignment
    assert any(".policy." in name and routed == "SGD" for name, routed in assigned.items())
    assert any(".body." in name and routed == "AdamW" for name, routed in assigned.items())
    _train(trainer, times=2)
    before = dict(optimizer.backend.assignment)
    trainer.to(trainer.runtime)                              # Peer.to / move_to path
    assert dict(optimizer.backend.assignment) == before      # O-G7: names survived the rebind
    _train(trainer)
    assert optimizer.backend.state_dict()["routes"]


def test_a_router_validates_its_routes(settings, tmp_path):
    with pytest.raises(ValidationError, match="appears twice"):
        RouterOptimizer(default={"TYPE": "AdamW"}, routes=[{"pattern": "a", "type": "SGD", "lr": 0.1},
                                                           {"pattern": "a", "type": "SGD", "lr": 0.2}])
    with pytest.raises(ValidationError, match="needs a non-empty 'pattern'"):
        RouterOptimizer(default={"TYPE": "AdamW"}, routes=[{"type": "SGD"}])
    with pytest.raises(ValidationError, match="cannot contain another router"):
        RouterOptimizer(default={"TYPE": "router"})
    with pytest.raises(TypeError):
        RouterOptimizer(routes=[])                            # `default` is mandatory


def test_a_registered_scheduler_builds_and_total_steps_is_filled_in(settings, tmp_path):
    trainer = _trainer(settings, tmp_path, {"CLASS": "AdamW", "LR": 1e-3, "SCHEDULER": "warmup_cosine",
                                            "SCHEDULER_KWARGS": {"warmup_steps": 2}}, epochs=6)
    entry = trainer.optimizer_plan.default
    assert entry.scheduler[1]["total_steps"] == trainer.optimizer_plan.horizon == 6
    rates = []
    for _ in range(4):
        _train(trainer)
        rates.append(trainer.optimizers["peer-1"].learning_rate)
    assert rates[0] != rates[-1]                                                # the scheduler moves the rate


def test_horizon_accounts_for_epochs_steps_and_accumulation():
    from mactf.training.schedulers import horizon

    assert horizon(3, 4, 1) == 12 and horizon(3, 4, 4) == 3 and horizon(0, 0, 0) == 1


def test_the_worked_lion_example_trains_with_a_step_down(settings, tmp_path, parts):
    """18.5: Lion with a 10x drop, the worker subtree keeps AdamW with the same scheduler."""
    trainer = _trainer(settings, tmp_path, {
        "TYPE": "Lion", "LR": 3e-4, "WEIGHT_DECAY": 0.01, "PARAMS": {"betas": [0.9, 0.99]},
        "SCHEDULER": "step_then_flat", "SCHEDULER_KWARGS": {"drop_at": 2, "factor": 0.1},
        "MAX_GRAD_NORM": 1.0, "MIGRATE": "carry",
        "OVERRIDES": {"peer-2": {"TYPE": "torch", "CLASS": "AdamW", "LR": 1e-4}}})
    lion, adamw = trainer.optimizers["peer-1"], trainer.optimizers["peer-2"]
    assert (lion.describe()["class"], adamw.describe()["class"]) == ("Lion", "AdamW")
    rates = []
    for _ in range(4):
        _train(trainer, "peer-1")
        rates.append(lion.learning_rate)
    assert rates[0] == pytest.approx(3e-4) and rates[1] == pytest.approx(3e-5)      # drops after 2 steps
    assert lion.scheduler_name == adamw.scheduler_name == "step_then_flat"
    _train(trainer, "peer-2", times=3)
    assert adamw.learning_rate == pytest.approx(1e-5)
    trainer.fit(epochs=2)                                                              # and it still runs


def test_unregistered_names_name_what_is_accepted(settings, tmp_path):
    with pytest.raises(ValidationError, match="unknown optimizer backend 'Lion'.*Adam"):
        MACTFOptimizer.from_spec({"type": "Lion"})
    with pytest.raises(ValidationError, match="unknown torch optimizer 'Lion'"):
        MACTFOptimizer.from_spec({"type": "torch", "class": "Lion"})


def test_builtin_optimizer_and_scheduler_registries_are_complete():
    optimizers, schedulers = HUB.registry("optimizer"), HUB.registry("scheduler")
    assert {"Adam", "AdamW", "SGD", "scipy", "callable", "null", "router"} <= set(optimizers.names())
    assert "LBFGS" not in optimizers.names() and "SparseAdam" not in optimizers.names()
    assert {"warmup_cosine", "warmup_linear", "constant_with_warmup", "inverse_sqrt", "StepLR",
            "OneCycleLR"} <= set(schedulers.names())
    assert all(optimizers.get(name).origin.value == "builtin" for name in optimizers.names()
               if name in {"Adam", "null", "router"})


# -- O6: swaps are atomic ---------------------------------------------------------------------------------------------------------
def _live(settings, tmp_path, parts, *, peers=2, **block):
    base = {"CLASS": "AdamW", "LR": 1e-3, "SCHEDULER": "warmup_cosine",
            "SCHEDULER_KWARGS": {"warmup_steps": 1}, **block}
    trainer = _trainer(settings, tmp_path, base, peers=peers)
    for path in trainer.peers:
        _train(trainer, path, times=2)
    return trainer


def test_optimizer_swap_replaces_every_selected_node_and_records_history(settings, tmp_path, parts):
    trainer = _live(settings, tmp_path, parts)
    records = trainer.swap_optimizer("OPTIMIZER", {"CLASS": "SGD", "LR": 0.05, "WEIGHT_DECAY": 0.0}, mode="reset")
    assert {r.slot for r in records} == {"OPTIMIZER"} and len(records) == 2
    assert set(_classes(trainer).values()) == {"SGD"}
    assert trainer.optimizer_plan.default.backend == "SGD"
    assert trainer.optimizer_plan.history == tuple(records)
    assert trainer.peers["peer-1"].optimizer is trainer.optimizers["peer-1"]            # the facade is the peer's
    assert all(r.boundary == "episode" and r.from_name == "AdamW" and r.to_name == "SGD" for r in records)


def test_optimizer_swap_is_atomic(settings, tmp_path, parts):
    """The exit criterion: after any failure the old facades, the plan and the freeze set are as before."""
    trainer = _live(settings, tmp_path, parts)
    old = dict(trainer.optimizers)
    entries = dict(trainer.optimizer_plan.entries)
    frozen = list(trainer.freeze)
    history = trainer.optimizer_plan.history
    state = {path: copy.deepcopy(o.state_dict()) for path, o in old.items()}

    def same(left, right):
        if torch.is_tensor(left) or torch.is_tensor(right):
            return torch.equal(torch.as_tensor(left), torch.as_tensor(right))
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
        if isinstance(left, (list, tuple)):
            return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
        return left == right

    def intact():
        assert trainer.optimizers == old and all(trainer.peers[p].optimizer is o for p, o in old.items())
        assert dict(trainer.optimizer_plan.entries) == entries
        assert list(trainer.freeze) == frozen and trainer.optimizer_plan.history == history
        assert all(same(o.state_dict()["state"], state[p]["state"]) for p, o in old.items())

    # 1. a block the compiler refuses
    with pytest.raises((ImproperlyConfigured, ValidationError)):
        trainer.swap_optimizer("OPTIMIZER", {"CLASS": "AdamW", "PARAMS": {"nope": 1}})
    intact()
    # 2. a migration that refuses: the old optimizers hold state, `fail` will not drop it
    with pytest.raises(ComponentMigrationError, match="migrate: fail"):
        trainer.swap_optimizer("OPTIMIZER", {"CLASS": "SGD", "LR": 0.1}, mode="fail")
    intact()
    # 3. an incompatible carry: AdamW moments cannot become SGD's
    with pytest.raises(ComponentMigrationError, match="cannot carry optimizer state"):
        trainer.swap_optimizer("OPTIMIZER", {"CLASS": "SGD", "LR": 0.1}, mode="carry")
    intact()
    # 4. a probe failure on the replacement entry (raised after staging, before the commit)
    class Broken(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-3):
            super().__init__(params, dict(lr=lr))

        def step(self, closure=None):
            raise RuntimeError("broken")

    register_optimizer("Broken", Broken, scope="run")
    with pytest.raises(ImproperlyConfigured, match="E056"):
        trainer.swap_optimizer("OPTIMIZER", {"TYPE": "Broken", "LR": 1e-3}, mode="reset")
    intact()
    # 5. a locked slot
    from mactf.components import HubPolicy

    with HUB.scoped("lock", policy=HubPolicy(locked=("slot:OPTIMIZER*",))):
        with pytest.raises(ComponentLocked, match="locked"):
            trainer.swap_optimizer("OPTIMIZER", {"CLASS": "SGD", "LR": 0.1}, mode="reset")
    intact()
    # 6. a boundary that is too fine
    with pytest.raises(ComponentLocked, match="run_start or an episode boundary"):
        trainer.swap_optimizer("OPTIMIZER", {"CLASS": "SGD", "LR": 0.1}, boundary=Boundary.STEP, mode="reset")
    intact()
    # 7. a change that would leave nothing able to learn
    with pytest.raises(ImproperlyConfigured, match="E054"):
        trainer.swap_optimizer("OPTIMIZER", {"TYPE": "null"}, mode="reset")
    intact()
    # and the old optimizers still train
    _train(trainer)


def test_a_failing_bind_part_way_through_leaves_the_first_nodes_untouched(settings, tmp_path, parts):
    """Staging builds node by node: a failure at node 2 must not have installed node 1."""
    trainer = _live(settings, tmp_path, parts)
    old = dict(trainer.optimizers)
    original = trainer._make_optimizer
    calls = []

    def flaky(path, entry, runtime):
        calls.append(path)
        if len(calls) == 2:
            raise RuntimeError("bind failed on the second node")
        return original(path, entry, runtime)

    trainer._make_optimizer = flaky
    with pytest.raises(RuntimeError, match="second node"):
        trainer.swap_optimizer("OPTIMIZER", {"CLASS": "SGD", "LR": 0.1}, mode="reset")
    trainer._make_optimizer = original
    assert trainer.optimizers == old and trainer.optimizer_plan.default.backend == "AdamW"
    assert trainer.optimizer_plan.history == ()


def test_swapping_only_one_overrides_nodes_leaves_the_others_alone(settings, tmp_path, parts):
    trainer = _live(settings, tmp_path, parts, OVERRIDES={"peer-2": {"LR": 5e-4}})
    before = dict(trainer.optimizers)
    records = trainer.swap_optimizer("OPTIMIZER.peer-2", {"LR": 2e-4}, mode="carry")
    assert len(records) == 1 and trainer.optimizers["peer-1"] is before["peer-1"]
    assert trainer.optimizers["peer-2"] is not before["peer-2"]
    assert _base_lr(trainer.optimizers["peer-2"]) == pytest.approx(2e-4)


def test_removing_an_override_falls_back_to_the_default(settings, tmp_path, parts):
    trainer = _live(settings, tmp_path, parts, OVERRIDES={"peer-2": {"TYPE": "torch", "CLASS": "SGD", "LR": 0.1,
                                                                   "WEIGHT_DECAY": 0.0}})
    assert _classes(trainer)["peer-2"] == "SGD"
    trainer.swap_optimizer("OPTIMIZER.peer-2", None, mode="reset")
    assert _classes(trainer)["peer-2"] == "AdamW" and "OPTIMIZER.peer-2" not in trainer.optimizer_plan.entries
    with pytest.raises(ImproperlyConfigured, match="E054.*cannot be removed"):
        trainer.swap_optimizer("OPTIMIZER", None)
    with pytest.raises(ComponentLocked, match="no such entry"):
        trainer.swap_optimizer("OPTIMIZER.ghost", None)


def test_null_entry_removal_thaws_only_its_nodes(settings, tmp_path, parts):
    trainer = _live(settings, tmp_path, parts, peers=3,
                    OVERRIDES={"peer-2": {"TYPE": "null"}, "peer-3": {"TYPE": "null"}})
    trainer.freeze.add("peer-1", "strategy:frozen_peer")                      # frozen by someone else
    assert sorted(trainer.freeze.paths("optimizer:null")) == ["peer-2", "peer-3"]
    trainer.swap_optimizer("OPTIMIZER.peer-2", None, mode="reset")
    assert "peer-2" not in trainer.freeze and "peer-3" in trainer.freeze
    assert trainer.tree.get("peer-2").status == "active" and trainer.tree.get("peer-3").status == "frozen"
    assert trainer.freeze.reasons("peer-1") == ("strategy:frozen_peer",)       # not touched
    assert not isinstance(trainer.optimizers["peer-2"].backend, NullOptimizer)
    # and replacing a node's optimizer with null freezes exactly that node
    trainer.swap_optimizer("OPTIMIZER.peer-2", {"TYPE": "null"}, mode="reset")
    assert "peer-2" in trainer.freeze


def test_a_queued_swap_is_applied_at_the_episode_boundary(settings, tmp_path, parts):
    trainer = _live(settings, tmp_path, parts)
    trainer.optimizer_plan.request("OPTIMIZER", {"CLASS": "SGD", "LR": 0.05, "WEIGHT_DECAY": 0.0}, mode="reset")
    assert trainer.optimizer_plan.pending()["OPTIMIZER"]["migrate"] == "reset"
    assert set(_classes(trainer).values()) == {"AdamW"}
    trainer.run_epoch(0)
    assert set(_classes(trainer).values()) == {"SGD"} and trainer.optimizer_plan.pending() == {}
    assert trainer.optimizer_plan.history[0].episode == 0


def test_a_queued_swap_that_fails_is_reported_and_dropped(settings, tmp_path, parts):
    trainer = _live(settings, tmp_path, parts)
    trainer.optimizer_plan.request("OPTIMIZER", {"CLASS": "SGD", "LR": 0.05}, mode="fail")
    trainer.run_epoch(0)                                                     # the AdamW state makes `fail` refuse
    assert any("queued optimizer swap" in message for message in trainer.failures)
    assert set(_classes(trainer).values()) == {"AdamW"} and trainer.optimizer_plan.pending() == {}


def test_request_validates_eagerly(settings, tmp_path, parts):
    trainer = _live(settings, tmp_path, parts)
    plan = trainer.optimizer_plan
    with pytest.raises((ImproperlyConfigured, ValidationError)):
        plan.request("OPTIMIZER", {"CLASS": "Nope"})
    with pytest.raises((ComponentMigrationError, ImproperlyConfigured), match="sideways"):
        plan.request("OPTIMIZER", {"CLASS": "SGD"}, mode="sideways")
    with pytest.raises(ComponentLocked, match="an optimizer slot is"):
        plan.request("WRONG", {"CLASS": "SGD"})
    assert plan.pending() == {} and plan.drop_pending("OPTIMIZER") is False


# -- reading the plan ------------------------------------------------------------------------------------------------------------------------
def test_entries_describe_and_hash_their_spec(settings, tmp_path, parts):
    trainer = _trainer(settings, tmp_path, {"TYPE": "Lion", "LR": 3e-4, "OVERRIDES": {"peer-2": {"LR": 1e-4}}})
    description = {slot: entry.describe() for slot, entry in trainer.optimizer_plan.entries.items()}
    assert description["OPTIMIZER"]["backend"] == "Lion" and description["OPTIMIZER"]["origin"] == "project"
    assert description["OPTIMIZER"]["spec_hash"] != description["OPTIMIZER.peer-2"]["spec_hash"]
    again = _trainer(settings, tmp_path / "again", {"TYPE": "Lion", "LR": 3e-4, "OVERRIDES": {"peer-2": {"LR": 1e-4}}})
    assert {s: e.spec_hash for s, e in again.optimizer_plan.entries.items()} == \
        {s: e.spec_hash for s, e in trainer.optimizer_plan.entries.items()}                # deterministic
