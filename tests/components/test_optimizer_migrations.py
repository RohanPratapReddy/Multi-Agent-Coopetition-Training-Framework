"""Optimizer swap migrations and the restore check (document 17 sections 9.2.2 and 9.6)."""

from __future__ import annotations

import copy
import logging

import pytest
import torch

from mactf.components import register_optimizer
from mactf.exceptions import ComponentMigrationError, ImproperlyConfigured, ValidationError
from mactf.orchestration.checkpoint_v2 import ContractMismatch
from mactf.training import optimizer_migrations as migrations
from mactf.training import schedulers
from mactf.training.optimizer import (
    BackendBase,
    CallableOptimizer,
    MACTFOptimizer,
    NullOptimizer,
    ParamGroupRouter,
    Route,
    TorchOptimizer,
    _Backend,
)

from p2_support import Lion, configure, make_trainer, snapshot_parameters, tune


def _module():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))


def _facade(module, spec, *, scheduler=None, accumulation=1):
    optimizer = MACTFOptimizer.from_spec({**spec, "grad_accumulation": accumulation, "node_path": "n"})
    optimizer.plan_scheduler(scheduler)
    optimizer.bind(list(module.parameters()), named=list(module.named_parameters()))
    return optimizer


def _fit(module, optimizer, steps=3):
    batch = torch.randn(8, 3, generator=torch.Generator().manual_seed(1))
    for _ in range(steps):
        optimizer.zero_grad()
        module(batch).pow(2).mean().backward()
        optimizer.step()


ADAM = {"type": "torch", "class": "Adam", "lr": 1e-2}
ADAMW = {"type": "torch", "class": "AdamW", "lr": 3e-3, "weight_decay": 0.1}
SGD = {"type": "torch", "class": "SGD", "lr": 0.1}


# -- has_state / compatible_state ----------------------------------------------------------------------
def test_has_state_is_false_until_something_ran():
    module = _module()
    optimizer = _facade(module, ADAM)
    assert not migrations.has_state(optimizer)
    _fit(module, optimizer, 1)
    assert migrations.has_state(optimizer)
    assert not migrations.has_state(_facade(_module(), {"type": "null"}))


def test_has_state_counts_a_scheduler_position_and_an_open_window():
    module = _module()
    scheduled = _facade(module, ADAM, scheduler=("StepLR", {"step_size": 1}))
    assert not migrations.has_state(scheduled)
    scheduled.scheduler.step()
    assert migrations.has_state(scheduled)
    window = _facade(_module(), SGD, accumulation=4)
    window._accumulated = 2
    assert migrations.has_state(window)


def test_compatible_state_knows_which_optimizers_can_share_moments():
    module = _module()
    adam = _facade(module, ADAM)
    _fit(module, adam)
    saved = adam.backend.state_dict()
    adamw = _facade(_module(), ADAMW)
    sgd = _facade(_module(), SGD)
    assert migrations.compatible_state(saved, adamw.backend, "Adam") is None        # same keys
    assert "hyper-parameter sets differ" in migrations.compatible_state(saved, sgd.backend, "Adam")
    assert "cannot restore another optimizer" in migrations.compatible_state(
        saved, CallableOptimizer(lambda params, grads: params), "Adam")
    assert migrations.compatible_state({}, sgd.backend) is None                   # nothing saved, nothing to refuse
    other_groups = _facade(_module(), ADAM)
    other_groups.backend.add_param_group({"params": [torch.nn.Parameter(torch.zeros(1))]})
    assert "parameter-group count" in migrations.compatible_state(saved, other_groups.backend)


# -- reset / carry / fail -------------------------------------------------------------------------------
def test_optimizer_migration_reset_starts_clean_and_logs(caplog):
    module = _module()
    old = _facade(module, ADAM)
    _fit(module, old)
    new = _facade(module, ADAMW)
    with caplog.at_level(logging.WARNING, logger="mactf.training.optimizer_migrations"):
        migrations.reset(old, new)
    assert "discards the state" in caplog.text
    assert not migrations.has_state(new)


def test_optimizer_migration_carry_moves_moments_keeps_the_new_hyper_parameters():
    module = _module()
    old = _facade(module, ADAM)
    _fit(module, old)
    new = _facade(module, ADAMW)
    migrations.carry(old, new)
    old_state, new_state = old.backend.state_dict()["state"], new.backend.state_dict()["state"]
    assert new_state.keys() == old_state.keys()
    for key in old_state:
        assert torch.equal(old_state[key]["exp_avg"], new_state[key]["exp_avg"])
        assert new_state[key]["step"] == old_state[key]["step"]
    group = new.backend.param_groups[0]
    assert group["lr"] == pytest.approx(3e-3) and group["weight_decay"] == pytest.approx(0.1)   # never the old rate


def test_optimizer_migration_carry_replays_the_scheduler_position():
    module = _module()
    spec = ("warmup_linear", {"warmup_steps": 2, "total_steps": 20})
    old = _facade(module, ADAM, scheduler=spec)
    _fit(module, old, 5)
    new = _facade(module, ADAMW, scheduler=spec)
    assert schedulers.position_of(new.scheduler) == 0
    migrations.carry(old, new)
    assert schedulers.position_of(new.scheduler) == schedulers.position_of(old.scheduler) == 5
    assert new.learning_rate == pytest.approx(old.learning_rate * 3e-3 / 1e-2)        # the new base, the old step


def test_optimizer_migration_carry_refuses_incompatible_state_and_says_how_to_proceed():
    module = _module()
    old = _facade(module, ADAM)
    _fit(module, old)
    new = _facade(module, SGD)
    with pytest.raises(ComponentMigrationError, match="choose migrate: reset"):
        migrations.carry(old, new)
    assert not migrations.has_state(new)                                              # nothing was half-loaded


def test_optimizer_migration_carry_from_a_stateless_optimizer_is_a_no_op():
    module = _module()
    migrations.carry(_facade(module, ADAM), _facade(module, SGD))


def test_optimizer_migration_carry_warns_when_the_position_has_nowhere_to_go(caplog):
    module = _module()
    old = _facade(module, ADAM, scheduler=("StepLR", {"step_size": 1}))
    _fit(module, old, 2)
    new = _facade(module, ADAMW)
    with caplog.at_level(logging.WARNING, logger="mactf.training.optimizer_migrations"):
        migrations.carry(old, new)
    assert "W014" in caplog.text and "no scheduler" in caplog.text


def test_optimizer_migration_carry_keeps_the_open_accumulation_window_only_when_unchanged(caplog):
    module = _module()
    old = _facade(module, ADAM, accumulation=4)
    batch = torch.randn(8, 3)
    for _ in range(2):
        module(batch).pow(2).mean().backward()
        old.step()
    assert old._accumulated == 2
    same = _facade(module, ADAMW, accumulation=4)
    migrations.carry(old, same)
    assert same._accumulated == 2
    other = _facade(module, ADAMW, accumulation=2)
    with caplog.at_level(logging.WARNING, logger="mactf.training.optimizer_migrations"):
        migrations.carry(old, other)
    assert other._accumulated == 0 and "W014" in caplog.text


def test_optimizer_migration_fail_refuses_state_and_allows_a_fresh_one():
    module = _module()
    old = _facade(module, ADAM)
    migrations.fail(old, _facade(module, SGD))                         # nothing to lose
    _fit(module, old)
    with pytest.raises(ComponentMigrationError, match="migrate: fail"):
        migrations.fail(old, _facade(module, SGD))


def test_optimizer_migration_freezing_is_never_implicit():
    """AdamW to null always loses state, so only ``reset`` gets through (9.6)."""
    module = _module()
    old = _facade(module, ADAM)
    _fit(module, old)
    null = _facade(module, {"type": "null"})
    for mode in (migrations.carry, migrations.fail):
        with pytest.raises(ComponentMigrationError):
            mode(old, null)
    migrations.reset(old, null)
    migrations.carry(null, _facade(module, ADAMW))                     # and thawing has nothing to carry


def test_optimizer_migration_selection():
    assert migrations.migration_for(None) is migrations.reset and migrations.migration_for("carry") is migrations.carry
    assert migrations.migration_for("FAIL") is migrations.fail
    with pytest.raises(ComponentMigrationError, match="unknown migration"):
        migrations.migration_for("sideways")


# -- the restore check (9.2.2) ------------------------------------------------------------------------------------------------
def _hashed(module, spec, hash_, *, strict=False, scheduler=None):
    optimizer = _facade(module, spec, scheduler=scheduler)
    optimizer.spec_hash, optimizer.slot, optimizer.strict_restore = hash_, "OPTIMIZER", strict
    return optimizer


def test_resume_with_an_unchanged_spec_is_a_full_load():
    module = _module()
    old = _hashed(module, ADAM, "aaaa", scheduler=("StepLR", {"step_size": 1}))
    _fit(module, old, 3)
    payload = copy.deepcopy(old.state_dict())
    assert payload["spec_hash"] == "aaaa" and payload["scheduler_name"] == "StepLR"
    new = _hashed(_module(), ADAM, "aaaa", scheduler=("StepLR", {"step_size": 1}))
    new.load_state_dict(payload)
    assert new.backend.state_dict()["state"].keys() == old.backend.state_dict()["state"].keys()
    assert new.learning_rate == pytest.approx(old.learning_rate)
    assert schedulers.position_of(new.scheduler) == 3


def test_resume_with_changed_optimizer_carries_compatible_state_under_the_new_hyper_parameters(caplog):
    module = _module()
    old = _hashed(module, ADAM, "aaaa", scheduler=("warmup_linear", {"warmup_steps": 1, "total_steps": 30}))
    _fit(module, old, 4)
    payload = copy.deepcopy(old.state_dict())
    new = _hashed(_module(), ADAMW, "bbbb", scheduler=("warmup_linear", {"warmup_steps": 1, "total_steps": 30}))
    with caplog.at_level(logging.WARNING, logger="mactf.training.optimizer"):
        new.load_state_dict(payload)
    assert "W015" in caplog.text and "carried" in caplog.text
    group = new.backend.param_groups[0]
    assert group["weight_decay"] == pytest.approx(0.1)                       # torch copied the saved groups; we put ours back
    assert group["initial_lr"] == pytest.approx(3e-3)
    assert new.backend.state_dict()["state"]
    assert schedulers.position_of(new.scheduler) == 4                          # replayed, not loaded


def test_resume_with_changed_optimizer_drops_incompatible_state_and_says_so(caplog):
    module = _module()
    old = _hashed(module, ADAM, "aaaa")
    _fit(module, old, 3)
    payload = copy.deepcopy(old.state_dict())
    new = _hashed(_module(), SGD, "bbbb")
    before = new.learning_rate
    with caplog.at_level(logging.WARNING, logger="mactf.training.optimizer"):
        new.load_state_dict(payload)
    assert "W015" in caplog.text and "dropped" in caplog.text
    assert not new.backend.state_dict()["state"]                                  # started fresh
    assert new.learning_rate == pytest.approx(before)
    _fit(module, new, 1)                                                          # and the first step works


def test_resume_with_changed_optimizer_under_strict_restore_is_e055():
    module = _module()
    old = _hashed(module, ADAM, "aaaa")
    _fit(module, old)
    new = _hashed(_module(), ADAMW, "bbbb", strict=True)
    with pytest.raises(ContractMismatch, match="E055"):
        new.load_state_dict(copy.deepcopy(old.state_dict()))
    assert not new.backend.state_dict()["state"]                                  # nothing was loaded


def test_a_legacy_checkpoint_or_a_legacy_facade_loads_exactly_as_before():
    module = _module()
    legacy = _facade(module, ADAM)
    _fit(module, legacy)
    payload = copy.deepcopy(legacy.state_dict())
    assert "spec_hash" not in payload
    hashed = _hashed(_module(), ADAMW, "bbbb")
    hashed.load_state_dict(payload)                                              # no saved hash: trust the position
    assert hashed.backend.state_dict()["state"]
    plain = _facade(_module(), ADAM)
    plain.load_state_dict(copy.deepcopy(_hashed(module, ADAM, "aaaa").state_dict()))   # no current hash: ditto


def test_resume_accepts_a_zero_learning_rate(caplog):
    """O-G12: a callable, scipy or null backend reports 0.0, which ``set_lr`` rejects."""
    module = _module()
    null = _facade(module, {"type": "null"})
    payload = null.state_dict()
    assert payload["lr"] == 0.0
    _facade(_module(), {"type": "null"}).load_state_dict(payload)
    callable_ = _facade(module, {"type": "callable", "fn": lambda params, grads: params})
    _facade(_module(), {"type": "callable", "fn": lambda params, grads: params}).load_state_dict(callable_.state_dict())


def test_null_refuses_state_written_by_a_real_optimizer():
    with pytest.raises(Exception, match="no state to restore"):
        NullOptimizer().load_state_dict({"state": {1: 1}})


# -- backend plumbing (O10) ------------------------------------------------------------------------------------------------------------
def test_backend_base_keeps_its_old_name():
    assert _Backend is BackendBase and issubclass(TorchOptimizer, _Backend)


def test_custom_backend_is_reset():
    """O-G10: ``RESET_OPTIMIZERS`` clears momentum for a backend with no ``_instance``."""

    class Heavy(BackendBase):
        name = "heavy"

        def __init__(self):
            super().__init__()
            self.momentum = 5

        def step(self, closure=None):
            return None

        def reset_state(self):
            self.momentum = 0

    optimizer = MACTFOptimizer(Heavy())
    optimizer.bind(list(_module().parameters()))
    optimizer.reset_state()
    assert optimizer.backend.momentum == 0
    torch_backed = _facade(_module(), ADAM)
    module = _module()
    adam = _facade(module, ADAM)
    _fit(module, adam)
    assert adam.backend.state_dict()["state"]
    adam.reset_state()
    assert not adam.backend.state_dict()["state"]


def test_the_trainer_clears_a_custom_backend_after_aggregation(settings, tmp_path, component_overlay):
    cleared = []

    class Heavy(BackendBase):
        name = "heavy"
        requires_parameters = False

        def __init__(self):
            super().__init__()

        def step(self, closure=None):
            return None

        def reset_state(self):
            cleared.append(True)

    register_optimizer("Heavy", Heavy, scope="run")
    tune(settings)
    configure(settings, OPTIMIZER={"TYPE": "Heavy"})
    trainer = make_trainer(settings, tmp_path).build()
    trainer.run_epoch(0)
    assert cleared                                                                  # the coopetition write-back reset it


def test_router_survives_device_move_and_keeps_its_state():
    module = _module()
    import functools

    router = ParamGroupRouter([Route(r".*0\.weight", functools.partial(torch.optim.SGD, lr=0.1), priority=1)],
                              default=functools_partial_adam())
    optimizer = MACTFOptimizer(None, router=router)
    optimizer.bind(list(module.parameters()), named=list(module.named_parameters()))
    _fit(module, optimizer)
    before = dict(router.assignment)
    optimizer.move_to(None)
    assert dict(router.assignment) == before and "0.weight" in " ".join(before)


def functools_partial_adam():
    import functools

    return functools.partial(torch.optim.Adam, lr=1e-3)


def test_move_to_leaves_a_non_rebinding_backend_alone():
    module = _module()
    optimizer = _facade(module, {"type": "callable", "fn": lambda params, grads: params})
    optimizer.move_to(None)                                                          # must not raise


# -- schedulers (9.3) --------------------------------------------------------------------------------------------------------------------------
def _lrs(name, steps, **kwargs):
    parameter = torch.nn.Parameter(torch.zeros(1))
    native = torch.optim.SGD([parameter], lr=1.0)
    scheduler = schedulers.build_scheduler(name, native, kwargs)
    rates = []
    for _ in range(steps):
        rates.append(native.param_groups[0]["lr"])
        native.step()
        scheduler.step()
    return rates


def test_warmup_cosine_warms_up_then_decays_to_the_floor():
    rates = _lrs("warmup_cosine", 12, warmup_steps=3, total_steps=12, min_lr_ratio=0.1)
    assert rates[0] == pytest.approx(0.25) and rates[2] == pytest.approx(0.75)         # never 0: step 0 trains
    assert rates[3] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(rates[3:], rates[4:]))
    assert rates[-1] == pytest.approx(0.1, abs=0.05)


def test_warmup_linear_and_constant_and_inverse_sqrt():
    linear = _lrs("warmup_linear", 8, warmup_steps=2, total_steps=8)
    assert linear[0] < linear[1] < linear[2] == pytest.approx(1.0) and linear[-1] < linear[3]
    flat = _lrs("constant_with_warmup", 6, warmup_steps=2)
    assert flat[2:] == [pytest.approx(1.0)] * 4 and flat[0] < flat[1] < 1.0
    inverse = _lrs("inverse_sqrt", 8, warmup_steps=2)
    assert inverse[2] == pytest.approx(1.0) and inverse[3] < inverse[2] and inverse[7] < inverse[3]


def test_scheduler_parameters_are_checked():
    for name, kwargs in (("warmup_cosine", {"warmup_steps": 5, "total_steps": 5}),
                         ("warmup_cosine", {"warmup_steps": 0, "total_steps": 0}),
                         ("warmup_linear", {"total_steps": 4, "min_lr_ratio": 2.0}),
                         ("constant_with_warmup", {"warmup_steps": -1}),
                         ("inverse_sqrt", {"warmup_steps": -1})):
        with pytest.raises(Exception, match="must be"):
            _lrs(name, 1, **kwargs)


def test_reduce_lr_on_plateau_is_refused_at_attach_not_at_step_one():
    """O-G4."""
    module = _module()
    optimizer = MACTFOptimizer.from_spec(ADAM)
    optimizer.bind(list(module.parameters()))
    with pytest.raises(ValidationError, match="E052.*needs a metric"):
        optimizer.attach_scheduler("ReduceLROnPlateau")


def test_a_scheduler_on_a_backend_without_a_torch_optimizer_is_e052():
    optimizer = MACTFOptimizer.from_spec({"type": "callable", "fn": lambda params, grads: params})
    optimizer.bind(list(_module().parameters()))
    with pytest.raises(ValidationError, match="E052"):
        optimizer.attach_scheduler("StepLR", step_size=1)


def test_a_torch_scheduler_that_is_not_registered_still_works():
    module = _module()
    optimizer = MACTFOptimizer.from_spec(ADAM)
    optimizer.bind(list(module.parameters()))
    optimizer.attach_scheduler("LambdaLR", lr_lambda=lambda step: 0.5)
    assert optimizer.scheduler_name == "LambdaLR"
    with pytest.raises(Exception, match="NoSuchScheduler"):
        optimizer.attach_scheduler("NoSuchScheduler")


def test_fast_forward_replays_and_refuses_the_absurd():
    parameter = torch.nn.Parameter(torch.zeros(1))
    native = torch.optim.SGD([parameter], lr=1.0)
    scheduler = schedulers.build_scheduler("warmup_linear", native, {"warmup_steps": 1, "total_steps": 10})
    schedulers.fast_forward(scheduler, 4)
    assert schedulers.position_of(scheduler) == 4
    with pytest.raises(ValueError, match="cannot fast-forward"):
        schedulers.fast_forward(scheduler, -1)
    with pytest.raises(ValueError, match="refusing to replay"):
        schedulers.fast_forward(scheduler, schedulers.MAX_FAST_FORWARD + 1)
    assert schedulers.position_of(object()) is None


def test_accepts_total_steps():
    assert schedulers.accepts_total_steps("warmup_cosine") and schedulers.accepts_total_steps("OneCycleLR")
    assert not schedulers.accepts_total_steps("StepLR") and not schedulers.accepts_total_steps("nope")


# -- the trainer resumes a swapped plan (9.8) --------------------------------------------------------------------------------------------------------
def _run(settings, tmp_path, block, epochs=2, **kwargs):
    tune(settings)
    configure(settings, OPTIMIZER=block)
    trainer = make_trainer(settings, tmp_path, **kwargs)
    trainer.run_id = "optimizer-run"               # a restore is strict about the run it belongs to
    trainer.build()
    for epoch in range(epochs):
        trainer.run_epoch(epoch)
    return trainer


def test_a_swapped_plan_is_checkpointed_and_wins_over_the_file_on_resume(settings, tmp_path, component_overlay):
    block = {"CLASS": "AdamW", "LR": 1e-3, "WEIGHT_DECAY": 0.0}
    first = _run(settings, tmp_path / "a", block)
    first.swap_optimizer("OPTIMIZER", {"CLASS": "SGD", "LR": 0.05, "WEIGHT_DECAY": 0.0}, mode="reset")
    first.run_epoch(2)
    payload = first.state_dict()
    saved = payload["identity"]["trainer"]["optimizer_plan"]
    assert saved["entries"]["OPTIMIZER"]["block"]["CLASS"] == "SGD" and saved["history"]
    second = _run(settings, tmp_path / "b", block, epochs=0)
    assert {o.describe()["class"] for o in second.optimizers.values()} == {"AdamW"}
    second.load_state_dict(payload, strict=False)
    assert {o.describe()["class"] for o in second.optimizers.values()} == {"SGD"}   # continues under what it had
    assert any(note.startswith("W015") for note in second._optimizer_warnings)
    assert second.optimizer_plan.history[0].to_name == "SGD"


def test_a_swapped_plan_under_strict_restore_is_e055(settings, tmp_path, component_overlay):
    block = {"CLASS": "AdamW", "LR": 1e-3, "WEIGHT_DECAY": 0.0}
    first = _run(settings, tmp_path / "a", block)
    first.swap_optimizer("OPTIMIZER", {"CLASS": "SGD", "LR": 0.05, "WEIGHT_DECAY": 0.0}, mode="reset")
    payload = first.state_dict()
    configure(settings, COMPONENTS={"STRICT_RESTORE": True})
    second = _run(settings, tmp_path / "b", block, epochs=0)
    with pytest.raises(ImproperlyConfigured, match="E055"):
        second.load_state_dict(payload, strict=False)


def test_resume_after_the_optimizer_block_was_edited_warns_and_continues(settings, tmp_path, component_overlay):
    first = _run(settings, tmp_path / "a", {"CLASS": "AdamW", "LR": 1e-3, "WEIGHT_DECAY": 0.0,
                                            "OVERRIDES": {"peer-2": {"CLASS": "Adam", "LR": 2e-3}}})
    payload = first.state_dict()
    second = _run(settings, tmp_path / "b", {"CLASS": "AdamW", "LR": 1e-3, "WEIGHT_DECAY": 0.0,
                                             "OVERRIDES": {"peer-2": {"CLASS": "SGD", "LR": 0.1}}}, epochs=0)
    second.load_state_dict(payload, strict=False)
    _ = second.run_epoch(2)                                                         # no crash at the first step
    assert second.optimizers["peer-2"].describe()["class"] == "SGD"                  # the edited file wins: nothing was swapped
    assert any(not torch.isnan(t).any() for t in snapshot_parameters(second).values())


def test_ensure_optimizer_builds_the_optimizer_a_restore_needs(settings, tmp_path):
    """O-G8: ``checkpoint_v2`` calls ``trainer._ensure_optimizer(path)``; it used to be missing."""
    trainer = _run(settings, tmp_path, {"CLASS": "AdamW", "LR": 1e-3, "WEIGHT_DECAY": 0.0}, epochs=1)
    del trainer.optimizers["peer-2"]
    rebuilt = trainer._ensure_optimizer("peer-2")
    assert rebuilt is trainer.optimizers["peer-2"] and rebuilt.bound
    assert trainer._ensure_optimizer("ghost") is None


def test_a_checkpoint_round_trip_with_overrides_restores_every_nodes_moments(settings, tmp_path, component_overlay):
    block = {"CLASS": "AdamW", "LR": 1e-3, "WEIGHT_DECAY": 0.0, "OVERRIDES": {"peer-2": {"CLASS": "SGD", "LR": 0.1,
                                                                                      "WEIGHT_DECAY": 0.0}}}
    first = _run(settings, tmp_path / "a", block)
    payload = first.state_dict()
    second = _run(settings, tmp_path / "b", block, epochs=0)
    report = second.load_state_dict(payload, strict=True)
    assert report is not None
    for path in ("peer-1", "peer-2"):
        assert second.optimizers[path].describe()["class"] == first.optimizers[path].describe()["class"]
        assert second.optimizers[path].state_dict().get("spec_hash") == first.optimizers[path].state_dict().get("spec_hash")
