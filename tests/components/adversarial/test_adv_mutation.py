"""Adversarial verification of D2 for peer factories and mutation plans (doc 17 sections 12-13).

A peer factory builds a new participant; a mutation plan changes the tree. Neither may admit a
non-finite or aliased participant, and a refused plan leaves the trainer exactly as it was.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))

from mactf.components import register_peer_factory  # noqa: E402
from mactf.exceptions import MutationError  # noqa: E402
from mactf.training.mutation import MutationPlan, add_peer, retire_peer  # noqa: E402
from mutation_support import Tiny, mutable_trainer, mutation_settings, snapshot  # noqa: E402


@pytest.fixture
def trainer(settings, tmp_path, component_overlay):
    mutation_settings(settings)
    return mutable_trainer(settings, tmp_path, peers=3)


def _register(name, build):
    register_peer_factory(name, build, scope="run", capabilities={"finetuned": True})


def _filled(value):
    def build(spec):
        module = Tiny()
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.fill_(value)
        return module

    return build


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_a_factory_cannot_admit_non_finite_parameters(trainer, value):
    _register("adv_nonfinite", _filled(value))
    before = snapshot(trainer)
    with pytest.raises(MutationError) as caught:
        trainer.add_peer("peer-9", factory="adv_nonfinite")
    assert caught.value.code == "E085" and "non-finite" in str(caught.value)
    assert snapshot(trainer) == before and "peer-9" not in trainer.peers


def test_a_factory_cannot_hand_back_an_existing_peers_module(trainer):
    _register("adv_alias", lambda spec: trainer.peers["peer-1"].adapter.module)
    before = snapshot(trainer)
    with pytest.raises(MutationError, match="E085"):
        trainer.add_peer("peer-9", factory="adv_alias")
    assert snapshot(trainer) == before


def test_a_factory_with_a_non_finite_buffer_is_refused(trainer):
    def build(spec):
        module = Tiny()
        module.register_buffer("running", torch.full((3,), float("nan")))
        return module

    _register("adv_buffer", build)
    with pytest.raises(MutationError, match="non-finite"):
        trainer.add_peer("peer-9", factory="adv_buffer")


@pytest.mark.parametrize("std", [float("inf"), float("nan"), -1.0, True, "big"])
def test_a_bad_noise_level_is_refused(trainer, std):
    before = snapshot(trainer)
    with pytest.raises(MutationError):
        trainer.add_peer("peer-9", like="peer-1", factory={"type": "perturbed_clone", "params": {"std": std}})
    assert snapshot(trainer) == before


def test_a_factory_that_raises_or_returns_junk_changes_nothing(trainer):
    def boom(spec):
        raise RuntimeError("plugin bug")

    _register("adv_boom", boom)
    _register("adv_junk", lambda spec: object())
    before = snapshot(trainer)
    for name in ("adv_boom", "adv_junk"):
        with pytest.raises(MutationError) as caught:
            trainer.add_peer("peer-9", factory=name)
        assert caught.value.code == "E085"
    assert snapshot(trainer) == before


def test_a_plan_that_fails_half_way_is_rolled_back_as_a_whole(trainer):
    _register("adv_nonfinite", _filled(float("nan")))
    before = snapshot(trainer)
    plan = MutationPlan.of(add_peer("peer-8", like="peer-1", factory="clone"),
                           add_peer("peer-9", factory="adv_nonfinite"))
    with pytest.raises(MutationError):
        trainer.apply_mutation(plan) if hasattr(trainer, "apply_mutation") else plan.apply(trainer)
    assert snapshot(trainer) == before and "peer-8" not in trainer.peers


def test_a_plan_cannot_retire_everyone(trainer):
    before = snapshot(trainer)
    plan = MutationPlan.of(*(retire_peer(path, reason="test") for path in sorted(trainer.peers)))
    with pytest.raises(MutationError):
        plan.apply(trainer)
    assert snapshot(trainer) == before
