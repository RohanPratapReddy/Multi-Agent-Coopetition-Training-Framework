"""The gate in front of a built-in override: the flag, the locks, the freeze, the cap (document 17 section 15).

A built-in is replaceable only on purpose. Without ``COMPONENTS.ALLOW_BUILTIN_OVERRIDE`` the
attempt is E002 and the framework's own class keeps running; with it, the replacement is
announced once (W001) and recorded; a lock or a freeze still beats the flag.
"""

from __future__ import annotations

import logging
import sys

import pytest

import custom_kinds as ck
from component_helpers import OtherPolicy, Policy
from mactf.components import HUB, Boundary, ComponentResolver, HubPolicy, LegacyTable, Origin, SlotTable, overlay, plugins
from mactf.components import load_components
from mactf.exceptions import ComponentCollision, ComponentContractError, ComponentLocked, ComponentMissing
from mactf.training.policy import DefaultTrainPolicy
from override_support import identity_run, override_scope, put, settings_for, w001

POLICIES = "trainer_policy"


@pytest.fixture(autouse=True)
def _warnings(caplog):
    caplog.set_level(logging.WARNING, logger="mactf.components")


# -- the flag ---------------------------------------------------------------------------------
def test_without_the_flag_a_builtin_name_is_a_collision_and_the_builtin_still_runs(settings, tmp_path, caplog):
    settings_for(settings, TRAINER={"POLICY": "default"})
    before = HUB.snapshot()
    with override_scope(allowed=False):
        with pytest.raises(ComponentCollision) as caught:
            put(POLICIES, "default", ck.SameDefaultPolicy)
        assert caught.value.code == "E002" and "ALLOW_BUILTIN_OVERRIDE" in str(caught.value)
        trainer, _ = identity_run(settings, tmp_path)
        component = trainer.slots.get("TRAINER.POLICY").component
        assert component.factory is DefaultTrainPolicy and not component.overrides_builtin
    assert HUB.snapshot() == before and w001(caplog) == []


@pytest.mark.parametrize("kind, name, factory", [
    ("weight_update", "kappa_softmax", ck.SameKappa),
    ("peer_strategy", "kappa_softmax", ck.SameKappaStrategy),
    ("scorer", "penalty", ck.SamePenalty),
    ("aggregator", "default", ck.DeterministicAggregator),
    ("rl_function.advantage", "gae", ck.SameGAE),
])
def test_every_kind_refuses_a_builtin_name_without_the_flag(kind, name, factory):
    with override_scope(allowed=False):
        with pytest.raises(ComponentCollision, match="ALLOW_BUILTIN_OVERRIDE"):
            put(kind, name, factory)


def test_the_flag_without_replace_is_still_a_collision():
    from mactf.components import register

    with override_scope():
        with pytest.raises(ComponentCollision, match="replace=True"):
            register(POLICIES, "default", ck.SameDefaultPolicy, scope="run")


def test_with_the_flag_the_override_warns_once_and_is_recorded(settings, tmp_path, caplog):
    settings_for(settings, TRAINER={"POLICY": "default"})
    with override_scope():
        component = put(POLICIES, "default", ck.SameDefaultPolicy)
        assert component.overrides_builtin and component.origin is not Origin.BUILTIN
        assert w001(caplog) == ["trainer_policy:default"]
        identity_run(settings, tmp_path)
        assert w001(caplog) == ["trainer_policy:default"], "a run must not repeat the warning"


def test_replacing_a_nonbuiltin_twice_warns_only_for_the_builtin(caplog):
    with override_scope():
        put(POLICIES, "default", ck.SameDefaultPolicy)
        put(POLICIES, "default", ck.SameDefaultPolicy)   # now over an override, not a built-in
        put(POLICIES, "fresh_name", ck.AccumulatingPolicy)
    assert w001(caplog) == ["trainer_policy:default"]


def test_the_settings_flag_opens_the_gate_for_a_global_registration(restore_hub, caplog):
    registry = HUB.registry(POLICIES)
    load_components({"ALLOW_BUILTIN_OVERRIDE": True, "PLUGINS": "off"})
    component = registry.register("default", ck.SameDefaultPolicy, replace=True)
    assert component.overrides_builtin and registry.get("default").factory is ck.SameDefaultPolicy
    assert w001(caplog) == ["trainer_policy:default"]
    load_components({"PLUGINS": "off"})
    with pytest.raises(ComponentCollision):      # the flag went away: another built-in is closed again
        HUB.registry("scorer").register("penalty", ck.SamePenalty, replace=True)


# -- locks ------------------------------------------------------------------------------------
@pytest.mark.parametrize("pattern", ["trainer_policy:default", "trainer_policy:*", "*:default", "trainer_*:def*"])
def test_a_kind_lock_beats_the_flag(pattern):
    before = HUB.snapshot()
    with overlay(policy=HubPolicy(allow_builtin_override=True, locked=frozenset({pattern}))):
        with pytest.raises(ComponentLocked) as caught:
            put(POLICIES, "default", ck.SameDefaultPolicy)
        assert caught.value.code == "E003"
    assert HUB.snapshot() == before


def test_a_lock_on_another_name_does_not_stop_the_override(caplog):
    with overlay(policy=HubPolicy(allow_builtin_override=True, locked=frozenset({"trainer_policy:co_distill"}))):
        assert put(POLICIES, "default", ck.SameDefaultPolicy).overrides_builtin


def test_a_scorer_wide_lock_refuses_every_scorer_override():
    with overlay(policy=HubPolicy(allow_builtin_override=True, locked=frozenset({"scorer:*"}))):
        for name, factory in (("penalty", ck.SamePenalty), ("reward", ck.SameReward), ("warning", ck.SameWarning)):
            with pytest.raises(ComponentLocked):
                put("scorer", name, factory)
        assert put(POLICIES, "default", ck.SameDefaultPolicy).overrides_builtin   # other kinds are free


def test_a_slot_lock_lets_an_override_install_but_not_move():
    slots = SlotTable(ComponentResolver())
    with overlay(policy=HubPolicy(allow_builtin_override=True, locked=frozenset({"slot:TRAINER.POLICY*"}))):
        put(POLICIES, "default", ck.SameDefaultPolicy)
        slots.install("TRAINER.POLICY", POLICIES, "default")
        assert slots.get("TRAINER.POLICY").component.overrides_builtin
        with pytest.raises(ComponentLocked, match="slot is locked"):
            slots.swap("TRAINER.POLICY", "co_distill", boundary=Boundary.RUN_START)
        with pytest.raises(ComponentLocked, match="slot is locked"):
            slots.request("TRAINER.POLICY", "co_distill")
        assert slots.get("TRAINER.POLICY").component.name == "default" and slots.pending() == {}


def test_locks_come_after_modules_so_a_module_may_still_register(tmp_path, monkeypatch, restore_hub):
    """Phase 1 loads without locks; the lock applies to what comes after (document 17 section 15.5)."""
    (tmp_path / "zz_ovr_locked.py").write_text(
        "from mactf.components import register\nfrom custom_kinds import SameDefaultPolicy\n"
        "register('trainer_policy', 'default', SameDefaultPolicy, replace=True)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    plugins.reset_for_tests()
    try:
        load_components({"MODULES": ["zz_ovr_locked"], "ALLOW_BUILTIN_OVERRIDE": True,
                         "LOCKED": ["trainer_policy:default"], "PLUGINS": "off"})
        assert HUB.registry(POLICIES).get("default").overrides_builtin
        with pytest.raises(ComponentLocked):
            HUB.registry(POLICIES).register("default", ck.AccumulatingPolicy, replace=True)
    finally:
        sys.modules.pop("zz_ovr_locked", None)
        plugins.reset_for_tests()


# -- freeze -----------------------------------------------------------------------------------
def test_freeze_after_load_refuses_a_global_override_but_a_run_may_still_carry_one(restore_hub):
    plugins.reset_for_tests()
    try:
        load_components({"ALLOW_BUILTIN_OVERRIDE": True, "FREEZE_AFTER_LOAD": True, "PLUGINS": "off"})
        registry = HUB.registry(POLICIES)
        assert registry.frozen
        with pytest.raises(ComponentLocked, match="frozen"):
            registry.register("default", ck.SameDefaultPolicy, replace=True)
        assert registry.get("default").factory is DefaultTrainPolicy
        with HUB.scoped():
            component = registry.register("default", ck.SameDefaultPolicy, replace=True, scope="run")
            assert component.overrides_builtin
        assert registry.get("default").factory is DefaultTrainPolicy
    finally:
        plugins.reset_for_tests()


def test_a_frozen_registry_refuses_to_unregister_a_builtin_too(restore_hub):
    plugins.reset_for_tests()
    try:
        load_components({"ALLOW_BUILTIN_OVERRIDE": True, "FREEZE_AFTER_LOAD": True, "PLUGINS": "off"})
        with pytest.raises(ComponentLocked, match="frozen"):
            HUB.registry(POLICIES).unregister("default")
    finally:
        plugins.reset_for_tests()


# -- COMPONENTS.MODULES ------------------------------------------------------------------------
def _module(tmp_path, monkeypatch, name: str, body: str) -> None:
    (tmp_path / f"{name}.py").write_text(body)
    monkeypatch.syspath_prepend(str(tmp_path))


def test_a_project_module_may_override_a_builtin_when_the_flag_is_set(tmp_path, monkeypatch, restore_hub, caplog):
    _module(tmp_path, monkeypatch, "zz_ovr_mod",
            "from mactf.components import register\nfrom custom_kinds import SameDefaultPolicy, SamePenalty\n"
            "register('trainer_policy', 'default', SameDefaultPolicy, replace=True)\n"
            "register('scorer', 'penalty', SamePenalty, replace=True)\n")
    plugins.reset_for_tests()
    try:
        report = load_components({"MODULES": ["zz_ovr_mod"], "ALLOW_BUILTIN_OVERRIDE": True, "PLUGINS": "off"})
        assert report.modules == ["zz_ovr_mod"]
        for kind, name, factory in ((POLICIES, "default", ck.SameDefaultPolicy), ("scorer", "penalty", ck.SamePenalty)):
            component = HUB.registry(kind).get(name)
            assert component.factory is factory and component.overrides_builtin
            assert component.origin is Origin.PROJECT
        assert sorted(w001(caplog)) == ["scorer:penalty", "trainer_policy:default"]
    finally:
        sys.modules.pop("zz_ovr_mod", None)
        plugins.reset_for_tests()


def test_a_project_module_without_the_flag_fails_the_load_and_changes_nothing(tmp_path, monkeypatch, restore_hub):
    _module(tmp_path, monkeypatch, "zz_ovr_nf",
            "from mactf.components import register\nfrom custom_kinds import SameDefaultPolicy\n"
            "register('trainer_policy', 'default', SameDefaultPolicy, replace=True)\n")
    plugins.reset_for_tests()
    try:
        with pytest.raises(ComponentCollision, match="ALLOW_BUILTIN_OVERRIDE"):
            load_components({"MODULES": ["zz_ovr_nf"], "PLUGINS": "off"})
        assert HUB.registry(POLICIES).get("default").factory is DefaultTrainPolicy
    finally:
        sys.modules.pop("zz_ovr_nf", None)
        plugins.reset_for_tests()


# -- restore_builtin and the legacy door -------------------------------------------------------
def test_restore_builtin_puts_the_framework_class_back(restore_hub):
    registry = HUB.registry(POLICIES)
    load_components({"ALLOW_BUILTIN_OVERRIDE": True, "PLUGINS": "off"})
    registry.register("default", ck.SameDefaultPolicy, replace=True)
    assert registry.get("default").overrides_builtin
    restored = registry.restore_builtin("default")
    assert restored.origin is Origin.BUILTIN and restored.factory is DefaultTrainPolicy
    assert not restored.overrides_builtin and registry.get("default") is restored


def test_restore_builtin_of_an_unknown_name_is_missing(restore_hub):
    with pytest.raises(ComponentMissing):
        HUB.registry(POLICIES).restore_builtin("zz_never_shipped")


def test_a_legacy_write_over_a_builtin_logs_w005_and_w001_without_the_flag(restore_hub, caplog):
    load_components({"PLUGINS": "off"})
    table = LegacyTable(POLICIES, hub=HUB)
    table["default"] = ck.SameDefaultPolicy
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("W005 trainer_policy:default") for m in messages)
    assert w001(caplog) == ["trainer_policy:default"]
    assert HUB.registry(POLICIES).get("default").overrides_builtin


def test_strict_legacy_closes_the_legacy_door(restore_hub):
    load_components({"STRICT_LEGACY": True, "PLUGINS": "off"})
    table = LegacyTable(POLICIES, hub=HUB)
    with pytest.raises(ComponentLocked, match="STRICT_LEGACY"):
        table["default"] = ck.SameDefaultPolicy
    assert HUB.registry(POLICIES).get("default").factory is DefaultTrainPolicy


def test_writing_the_original_back_through_the_legacy_table_restores_the_builtin(restore_hub):
    load_components({"PLUGINS": "off"})
    table = LegacyTable(POLICIES, hub=HUB)
    table["default"] = ck.SameDefaultPolicy
    table["default"] = DefaultTrainPolicy
    component = HUB.registry(POLICIES).get("default")
    assert component.origin is Origin.BUILTIN and not component.overrides_builtin


# -- the cap -----------------------------------------------------------------------------------
def test_max_per_kind_stops_new_names_but_not_a_replacement(hub, policy_for):
    registry = hub.registry(POLICIES)
    registry.register_builtin("target", Policy)
    hub.configure(policy_for(allow_builtin_override=True, max_components_per_kind=1))
    component = registry.register("target", OtherPolicy, replace=True)
    assert component.overrides_builtin
    with pytest.raises(ComponentContractError, match="registry is full"):
        registry.register("another", Policy)


def test_the_cap_does_not_stop_the_frameworks_own_builtins(hub, policy_for):
    hub.configure(policy_for(max_components_per_kind=1))
    registry = hub.registry(POLICIES)
    registry.register_builtin("one", Policy)
    registry.register_builtin("two", OtherPolicy)
    assert registry.has("one") and registry.has("two")
