"""Optimizers and schedulers under a built-in's name meet the same probe and the same migrations (document 17 section 9).

A broken class registered as ``optimizer:Adam`` (reached by ``TYPE: Adam``) is refused by the
probe (E056: O001-O006) at build and at swap, with the old optimizers, plan and moments exactly
as they were; ``OPTIMIZER.VERIFY: off`` is the only way past, and says so. A scheduler override
meets E052 and E053. Carrying state out of an override raises W014/W015 like any other.
"""

from __future__ import annotations

import copy
import logging

import pytest
import torch

import custom_kinds as ck
from mactf.exceptions import ImproperlyConfigured, ValidationError
from override_support import moments, moments_equal, override_scope, put, same_tensors, w001
from p2_support import configure, make_trainer, snapshot_parameters, tune
from test_optimizer_probe_findings import _BadBind, _NanRate, _NoReload, _Sgd, _Unpicklable

REFUSED = (ImproperlyConfigured, ValidationError)


@pytest.fixture(autouse=True)
def _scope(caplog):
    caplog.set_level(logging.WARNING)
    with override_scope():
        yield


def _trainer(settings, tmp_path, block, *, peers=2, epochs=2):
    tune(settings, EPOCHS=epochs)
    configure(settings, COMPONENTS={"PLUGINS": "off"}, OPTIMIZER=block)
    return make_trainer(settings, tmp_path, peers=peers).build()


def _batch():
    generator = torch.Generator().manual_seed(4)
    return {"states": torch.randn(8, 4, generator=generator), "targets": torch.zeros(8, dtype=torch.long),
            "signal": 0.5, "void": False, "floor": 0.0}


def _train(trainer, path="peer-1", times=1):
    for _ in range(times):
        trainer.train_step(trainer.tree.get(path), _batch())


class _Exploding(_Sgd):
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                p.fill_(float("nan"))


class _Raising(_Sgd):
    def step(self, closure=None):
        raise RuntimeError("no step for you")


PROBE = [(_BadBind, "O001"), (_Exploding, "O003"), (_Raising, "O002"), (_Unpicklable, "O004"),
         (_NoReload, "O005"), (_NanRate, "O006")]


# -- the probe -------------------------------------------------------------------------------------
@pytest.mark.parametrize("target", ["Adam", "AdamW", "SGD"])
@pytest.mark.parametrize("cls, code", PROBE, ids=[c.__name__ for c, _ in PROBE])
def test_a_broken_optimizer_under_a_builtin_name_is_refused_at_build(cls, code, target, settings, tmp_path):
    put("optimizer", target, cls)
    with pytest.raises(ImproperlyConfigured, match=rf"E056.*{code}"):
        _trainer(settings, tmp_path, {"TYPE": target, "LR": 1e-2})


@pytest.mark.parametrize("cls, code", PROBE[:3], ids=[c.__name__ for c, _ in PROBE[:3]])
def test_verify_off_is_the_only_way_a_broken_override_is_built(cls, code, settings, tmp_path):
    put("optimizer", "Adam", cls)
    with pytest.raises(ImproperlyConfigured, match="E056"):                             # control
        _trainer(settings, tmp_path / "on", {"TYPE": "Adam", "LR": 1e-2})
    if cls is _BadBind:
        return                                           # cannot even be constructed: no operator choice helps
    trainer = _trainer(settings, tmp_path / "off", {"TYPE": "Adam", "LR": 1e-2, "VERIFY": "off"})
    assert trainer.optimizers["peer-1"].describe()["class"] == cls.__name__


def test_a_well_behaved_override_passes_in_every_mode(settings, tmp_path, caplog):
    put("optimizer", "Adam", ck.SameAdam)
    for verify in ("custom", "all"):
        trainer = _trainer(settings, tmp_path / verify, {"TYPE": "Adam", "LR": 1e-2, "VERIFY": verify})
        assert trainer.optimizers["peer-1"].describe()["class"] == "SameAdam"
    assert set(w001(caplog)) == {"optimizer:Adam"}


# -- refusal at swap leaves everything as it was -----------------------------------------------------------
def test_a_refused_swap_to_a_broken_override_leaves_state_plan_and_moments_intact(settings, tmp_path):
    put("optimizer", "Adam", ck.SameAdam)
    trainer = _trainer(settings, tmp_path, {"TYPE": "Adam", "LR": 1e-2})
    _train(trainer, times=2)
    put("optimizer", "SGD", _Exploding)
    old = dict(trainer.optimizers)
    plan_backend = trainer.optimizer_plan.default.backend
    before, held = snapshot_parameters(trainer), moments(trainer)
    with pytest.raises(ImproperlyConfigured, match="E056.*O003"):
        trainer.swap_optimizer("OPTIMIZER", {"TYPE": "SGD", "LR": 1e-2}, mode="reset")
    assert trainer.optimizers == old and trainer.optimizer_plan.default.backend == plan_backend
    assert trainer.optimizer_plan.history == ()
    assert same_tensors(before, snapshot_parameters(trainer)) and moments_equal(moments(trainer), held)
    _train(trainer)                                                  # and the old optimizer still trains
    assert not same_tensors(before, snapshot_parameters(trainer))


def test_a_queued_swap_to_a_broken_override_fails_softly_and_the_run_goes_on(settings, tmp_path):
    put("optimizer", "Adam", ck.SameAdam)
    put("optimizer", "SGD", _Exploding)
    trainer = _trainer(settings, tmp_path, {"TYPE": "Adam", "LR": 1e-2})
    trainer.optimizer_plan.request("OPTIMIZER", {"TYPE": "SGD", "LR": 1e-2}, mode="reset")
    trainer.run_epoch(0)
    assert any("queued optimizer swap" in failure and "E056" in failure for failure in trainer.failures)
    assert trainer.optimizers["peer-1"].describe()["class"] == "SameAdam"
    assert all(torch.isfinite(t).all() for t in snapshot_parameters(trainer).values())


# -- schedulers: E052 and E053 --------------------------------------------------------------------------------
def _step_lr(optimizer, *, step_size: int = 1, gamma: float = 0.5):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def test_e053_bad_scheduler_kwargs_are_refused_for_a_scheduler_override(settings, tmp_path):
    put("scheduler", "StepLR", _step_lr)
    block = {"TYPE": "torch", "CLASS": "AdamW", "LR": 1e-3, "SCHEDULER": "StepLR",
             "SCHEDULER_KWARGS": {"bogus": 1}}
    with pytest.raises(REFUSED, match="E053"):
        _trainer(settings, tmp_path, block)


def test_e053_an_override_that_rejects_its_own_horizon_is_refused_at_build(settings, tmp_path):
    def warmup_cosine(optimizer, *, warmup_steps: int, total_steps: int):
        if warmup_steps >= total_steps:
            raise ValueError("warmup_steps must be below total_steps")
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    put("scheduler", "warmup_cosine", warmup_cosine)
    block = {"TYPE": "torch", "CLASS": "AdamW", "LR": 1e-3, "SCHEDULER": "warmup_cosine",
             "SCHEDULER_KWARGS": {"warmup_steps": 5, "total_steps": 3}}
    with pytest.raises(REFUSED, match="E053"):
        _trainer(settings, tmp_path, block)


def test_an_override_that_does_not_validate_its_horizon_is_not_rescued_by_the_framework(settings, tmp_path):
    """Pinned: the horizon check is the built-in factory's own; a replacement owns that duty."""
    def warmup_cosine(optimizer, *, warmup_steps: int, total_steps: int):
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    put("scheduler", "warmup_cosine", warmup_cosine)
    block = {"TYPE": "torch", "CLASS": "AdamW", "LR": 1e-3, "SCHEDULER": "warmup_cosine",
             "SCHEDULER_KWARGS": {"warmup_steps": 5, "total_steps": 3}}
    assert _trainer(settings, tmp_path, block) is not None


def test_e052_a_scheduler_that_needs_a_metric_is_refused_under_its_own_name(settings, tmp_path):
    def plateau(optimizer, *, patience: int = 1):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience)

    put("scheduler", "ReduceLROnPlateau", plateau)
    block = {"TYPE": "torch", "CLASS": "AdamW", "LR": 1e-3, "SCHEDULER": "ReduceLROnPlateau"}
    with pytest.raises(REFUSED, match="E052"):
        _trainer(settings, tmp_path, block)


def test_a_metric_scheduler_hidden_behind_a_builtin_name_is_caught_by_the_probe(settings, tmp_path):
    def plateau(optimizer, *, step_size: int = 1):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    put("scheduler", "StepLR", plateau)
    block = {"TYPE": "torch", "CLASS": "AdamW", "LR": 1e-3, "SCHEDULER": "StepLR"}
    with pytest.raises(REFUSED, match="E05[26]"):
        _trainer(settings, tmp_path, block)

def test_a_scheduler_override_that_is_honest_trains_with_the_same_rate_schedule(settings, tmp_path):
    block = {"TYPE": "torch", "CLASS": "SGD", "LR": 0.05, "WEIGHT_DECAY": 0.0, "SCHEDULER": "StepLR",
             "SCHEDULER_KWARGS": {"step_size": 1, "gamma": 0.5}}
    reference = _trainer(settings, tmp_path / "ref", block)
    reference.run_epoch(0)
    put("scheduler", "StepLR", _step_lr)
    trainer = _trainer(settings, tmp_path / "over", block)
    trainer.run_epoch(0)
    assert trainer.optimizers["peer-1"].learning_rate == reference.optimizers["peer-1"].learning_rate


# -- W014 / W015 ----------------------------------------------------------------------------------------------------
def test_w014_carrying_out_of_an_override_with_a_scheduler_into_one_without_warns(settings, tmp_path, caplog):
    put("optimizer", "Adam", ck.SameAdam)
    put("scheduler", "StepLR", _step_lr)
    trainer = _trainer(settings, tmp_path, {"TYPE": "Adam", "LR": 1e-2, "SCHEDULER": "StepLR",
                                            "SCHEDULER_KWARGS": {"step_size": 1}})
    trainer.run_epoch(0)
    with caplog.at_level(logging.WARNING):
        trainer.swap_optimizer("OPTIMIZER", {"TYPE": "torch", "CLASS": "AdamW", "LR": 1e-2}, mode="carry")
    assert "W014" in caplog.text and "no scheduler" in caplog.text


def test_w015_loading_override_state_into_another_optimizer_says_what_became_of_it(settings, tmp_path, caplog):
    put("optimizer", "Adam", ck.SameAdam)
    first = _trainer(settings, tmp_path / "a", {"TYPE": "Adam", "LR": 1e-2})
    _train(first, times=2)
    saved = copy.deepcopy(first.optimizers["peer-1"].state_dict())
    fresh = _trainer(settings, tmp_path / "b", {"TYPE": "AdamW", "LR": 1e-2})
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        fresh.optimizers["peer-1"].load_state_dict(saved)
    assert "W015" in caplog.text and "carried" in caplog.text


def test_w015_incompatible_override_state_is_dropped_and_says_so(settings, tmp_path, caplog):
    put("optimizer", "Adam", ck.SameAdam)
    first = _trainer(settings, tmp_path / "a", {"TYPE": "Adam", "LR": 1e-2})
    _train(first, times=2)
    saved = copy.deepcopy(first.optimizers["peer-1"].state_dict())
    fresh = _trainer(settings, tmp_path / "b", {"TYPE": "SGD", "LR": 1e-2})
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        fresh.optimizers["peer-1"].load_state_dict(saved)
    assert "W015" in caplog.text and "dropped" in caplog.text
    _train(fresh)                                                    # and the first step works

def test_the_new_scheduler_and_optimizer_corpus_runs_clean(settings, tmp_path):
    put("scheduler", "cosine_with_floor", ck.cosine_with_floor)
    trainer = _trainer(settings, tmp_path, {"TYPE": "torch", "CLASS": "AdamW", "LR": 1e-3,
                                            "SCHEDULER": "cosine_with_floor"})
    trainer.run_epoch(0)
    assert all(torch.isfinite(t).all() for t in snapshot_parameters(trainer).values())
