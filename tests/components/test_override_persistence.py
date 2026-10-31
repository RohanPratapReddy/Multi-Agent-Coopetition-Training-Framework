"""An override survives a checkpoint, and a resume without it is never silent (document 17 section 14).

The manifest records every slot that holds an override (``overrides_builtin``); resuming
without the override is W052, or E094 under ``STRICT_RESTORE``; a recorded name this build
cannot provide is E093, found before one parameter moves; the state of a stateful override
travels with the weights.
"""

from __future__ import annotations

import logging

import pytest
import torch

import custom_kinds as ck
from mactf.components import HUB, overlay
from mactf.components.registry import HubPolicy
from mactf.components.restore import extract_manifest
from mactf.exceptions import ImproperlyConfigured
from override_support import override_scope, put, settings_for, weight_instance
from p2_support import make_trainer, snapshot_parameters

OVERRIDES = {
    "policy": ("trainer_policy", "default", ck.SameDefaultPolicy, {"TRAINER": {"POLICY": "default"}}),
    "weight": ("weight_update", "kappa_softmax", ck.SameKappa,
               {"TRAINER": {"WEIGHT_UPDATE": {"type": "kappa_softmax", "params": {"tag": "t"}}}}),
    "advantage": ("rl_function.advantage", "gae", ck.SameGAE, {"RL": {"ADVANTAGE": {"type": "gae"}}}),
    "penalty": ("scorer", "penalty", ck.SamePenalty, {"SCORING": {"STAGES": {"penalty": "penalty"}}}),
}


@pytest.fixture(autouse=True)
def _warnings(caplog):
    caplog.set_level(logging.WARNING, logger="mactf.components")
    caplog.set_level(logging.WARNING, logger="mactf.checkpoint")


def _trained(settings, tmp_path, *, epochs=1):
    trainer = make_trainer(settings, tmp_path, seed=7, peers=2).build()
    for epoch in range(epochs):
        trainer.run_epoch(epoch)
    return trainer


def _overridden_entries(payload) -> list[tuple[str, str]]:
    manifest = extract_manifest(payload) or {}
    return sorted((kind, entry["name"]) for kind, slots in manifest.items()
                  for entry in slots.values() if entry.get("overrides_builtin"))


def _resumed(settings, tmp_path, run_id, payload, *, build_in=None):
    fresh = make_trainer(settings, tmp_path, seed=7, peers=2)
    fresh.run_id = run_id
    fresh.build()
    return fresh, fresh.load_state_dict(payload)


# -- the manifest ----------------------------------------------------------------------------
@pytest.mark.parametrize("label", sorted(OVERRIDES))
def test_the_manifest_records_a_reached_override(settings, tmp_path, label):
    kind, name, factory, blocks = OVERRIDES[label]
    settings_for(settings, **blocks)
    with override_scope():
        put(kind, name, factory)
        payload = _trained(settings, tmp_path).state_dict()
    assert (kind, name) in _overridden_entries(payload)


def test_f3_an_unreached_override_is_still_recorded_in_the_manifest(settings, tmp_path):
    """F3, pinned: the slot holds the registered component's record though the run built the default."""
    settings_for(settings)
    with override_scope():
        put("trainer_policy", "default", ck.SameDefaultPolicy)
        put("weight_update", "kappa_softmax", ck.SameKappa)
        payload = _trained(settings, tmp_path).state_dict()
    assert _overridden_entries(payload) == [("trainer_policy", "default"), ("weight_update", "kappa_softmax")]


def test_a_plain_run_records_no_override(settings, tmp_path):
    settings_for(settings)
    with override_scope():
        payload = _trained(settings, tmp_path).state_dict()
    assert _overridden_entries(payload) == []


# -- resume with and without the override ------------------------------------------------------
def test_resuming_with_the_same_override_is_all_ok(settings, tmp_path, caplog):
    settings_for(settings, TRAINER={"POLICY": "default"})
    with override_scope():
        put("trainer_policy", "default", ck.SameDefaultPolicy)
        first = _trained(settings, tmp_path / "a")
        payload = first.state_dict()
        fresh, _ = _resumed(settings, tmp_path / "b", first.run_id, payload)
    assert not [r for r in caplog.records if "W052" in r.getMessage() or "E094" in r.getMessage()]
    assert snapshot_parameters(fresh) is not None


def test_resuming_without_the_override_is_w052_and_the_configured_component_runs(settings, tmp_path, caplog):
    settings_for(settings, TRAINER={"POLICY": "default"})
    with override_scope():
        put("trainer_policy", "default", ck.SameDefaultPolicy)
        first = _trained(settings, tmp_path / "a")
        payload = first.state_dict()
    caplog.clear()
    with override_scope():                          # the override is no longer registered
        fresh, _ = _resumed(settings, tmp_path / "b", first.run_id, payload)
        component = fresh.slots.get("TRAINER.POLICY").component
        assert not component.overrides_builtin
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "W052" in text and "overrides_builtin" in text


def test_strict_restore_makes_the_missing_override_e094_before_anything_moves(settings, tmp_path):
    settings_for(settings, TRAINER={"POLICY": "default"})
    with override_scope():
        put("trainer_policy", "default", ck.SameDefaultPolicy)
        first = _trained(settings, tmp_path / "a")
        payload = first.state_dict()
    with overlay(policy=HubPolicy(allow_builtin_override=True, strict_restore=True)):
        fresh = make_trainer(settings, tmp_path / "b", seed=7, peers=2)
        fresh.run_id = first.run_id
        fresh.build()
        before = snapshot_parameters(fresh)
        with pytest.raises(ImproperlyConfigured, match="E094"):
            fresh.load_state_dict(payload)
        after = snapshot_parameters(fresh)
    assert all(torch.equal(before[key], after[key]) for key in before), "a refused restore moved parameters"


def test_a_recorded_override_of_a_new_name_this_build_lacks_is_e093(settings, tmp_path):
    settings_for(settings, TRAINER={"WEIGHT_UPDATE": {"type": "entropy_floor", "params": {"nudge": 0.3}}})
    with override_scope():
        put("weight_update", "entropy_floor", ck.EntropyFloorAverage)
        first = _trained(settings, tmp_path / "a")
        payload = first.state_dict()
    settings_for(settings, TRAINER={"WEIGHT_UPDATE": {"type": "kappa_softmax"}})   # a build that never heard of it
    with override_scope():
        fresh = make_trainer(settings, tmp_path / "b", seed=7, peers=2)
        fresh.run_id = first.run_id
        fresh.build()
        before = snapshot_parameters(fresh)
        with pytest.raises(ImproperlyConfigured, match="E093.*entropy_floor"):
            fresh.load_state_dict(payload)
        after = snapshot_parameters(fresh)
    assert all(torch.equal(before[key], after[key]) for key in before)


# -- state travels --------------------------------------------------------------------------
@pytest.mark.parametrize("registered_as", ["entropy_floor", "kappa_softmax"])
def test_a_stateful_override_carries_its_state_across_a_checkpoint(settings, tmp_path, registered_as):
    """``kappa_softmax`` here is the built-in's own name, so this is a stateful *override*."""
    spec = {"type": registered_as, "params": {"nudge": 0.3}}
    settings_for(settings, TRAINER={"WEIGHT_UPDATE": spec})
    with override_scope():
        put("weight_update", registered_as, ck.EntropyFloorAverage)
        first = _trained(settings, tmp_path / "a", epochs=2)
        calls = weight_instance(first).calls
        assert calls > 0
        payload = first.state_dict()
        fresh, _ = _resumed(settings, tmp_path / "b", first.run_id, payload)
        restored = weight_instance(fresh)
        assert isinstance(restored, ck.EntropyFloorAverage) and restored.calls == calls
        fresh.run_epoch(2)
        assert weight_instance(fresh).calls > calls


def test_a_changed_class_under_the_same_name_does_not_inherit_foreign_state(settings, tmp_path):
    settings_for(settings, TRAINER={"WEIGHT_UPDATE": {"type": "kappa_softmax"}})
    with override_scope():
        put("weight_update", "kappa_softmax", ck.EntropyFloorAverage)
        first = _trained(settings, tmp_path / "a", epochs=1)
        payload = first.state_dict()
    with override_scope():
        put("weight_update", "kappa_softmax", ck.SameKappa)
        fresh = make_trainer(settings, tmp_path / "b", seed=7, peers=2)
        fresh.run_id = first.run_id
        fresh.build()
        try:
            fresh.load_state_dict(payload)
        except ImproperlyConfigured:
            return                                   # refusing is an acceptable answer; inheriting is not
        assert not hasattr(weight_instance(fresh), "calls")
