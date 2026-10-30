"""Section 4.3, the replacement and removal table, tested cell by cell.

Rows are the actions, columns the origin of the component being acted on, plus the two
modifiers (locked, frozen). The expected outcome of every cell is written out here from the
table in the document, not derived from the code under test.
"""

from __future__ import annotations

import logging

import pytest

from component_helpers import OtherPolicy, Policy
from mactf.components import LegacyTable, Origin
from mactf.exceptions import ComponentCollision, ComponentLocked

OK = "ok"
FLAG = "needs flag"  # ComponentCollision on register, ComponentLocked on unregister, unless the flag is on
COLLISION = ComponentCollision
LOCKED = ComponentLocked

TARGETS = [Origin.BUILTIN, Origin.PLUGIN, Origin.PROJECT, Origin.RUNTIME]

#: (action, new origin) -> {target origin: outcome}; rows 2-5 of the table.
REPLACE_TABLE = {
    ("register", None): {Origin.BUILTIN: COLLISION, Origin.PLUGIN: COLLISION,
                         Origin.PROJECT: COLLISION, Origin.RUNTIME: COLLISION},   # replace=False, taken name
    ("replace", Origin.PLUGIN): {Origin.BUILTIN: FLAG, Origin.PLUGIN: OK,
                                 Origin.PROJECT: COLLISION, Origin.RUNTIME: COLLISION},
    ("replace", Origin.PROJECT): {Origin.BUILTIN: FLAG, Origin.PLUGIN: OK,
                                  Origin.PROJECT: OK, Origin.RUNTIME: COLLISION},
    ("replace", Origin.RUNTIME): {Origin.BUILTIN: FLAG, Origin.PLUGIN: OK,
                                  Origin.PROJECT: OK, Origin.RUNTIME: OK},
}


def _seed(hub, origin: Origin):
    registry = hub.registry("trainer_policy")
    if origin is Origin.BUILTIN:
        registry.register_builtin("target", Policy)
    else:
        registry.register("target", Policy, origin=origin)
    return registry


def _cells():
    for (action, new_origin), row in REPLACE_TABLE.items():
        for target in TARGETS:
            yield pytest.param(action, new_origin, target, row[target],
                               id=f"{action}-{new_origin.value if new_origin else 'plain'}-over-{target.value}")


@pytest.mark.parametrize("action, new_origin, target, expected", list(_cells()))
@pytest.mark.parametrize("flag", [False, True], ids=["no-flag", "flag"])
def test_register_on_a_taken_name(hub, policy_for, action, new_origin, target, expected, flag):
    hub.configure(policy_for(allow_builtin_override=flag))
    registry = _seed(hub, target)
    kwargs = {"replace": action == "replace"}
    if new_origin is not None:
        kwargs["origin"] = new_origin
    if expected == FLAG:
        expected = OK if flag else COLLISION
    if expected is OK:
        component = registry.register("target", OtherPolicy, **kwargs)
        assert registry.get("target").factory is OtherPolicy
        assert component.overrides_builtin is (target is Origin.BUILTIN)
        assert component.origin is new_origin
    else:
        with pytest.raises(expected):
            registry.register("target", OtherPolicy, **kwargs)
        assert registry.get("target").factory is Policy  # a refusal changes nothing
        assert registry.get("target").origin is target


@pytest.mark.parametrize("new_origin", [None, Origin.PLUGIN, Origin.PROJECT, Origin.RUNTIME])
def test_register_on_a_free_name_is_ok_for_every_origin(hub, new_origin):
    registry = hub.registry("trainer_policy")
    kwargs = {} if new_origin is None else {"origin": new_origin}
    assert registry.register("free", Policy, **kwargs).name == "free"


@pytest.mark.parametrize("target", TARGETS, ids=lambda o: o.value)
@pytest.mark.parametrize("flag", [False, True], ids=["no-flag", "flag"])
def test_unregister(hub, policy_for, target, flag):
    hub.configure(policy_for(allow_builtin_override=flag))
    registry = _seed(hub, target)
    if target is Origin.BUILTIN and not flag:
        with pytest.raises(ComponentLocked, match="removing it needs COMPONENTS.ALLOW_BUILTIN_OVERRIDE"):
            registry.unregister("target")
        assert registry.has("target")
    else:
        registry.unregister("target")
        assert not registry.has("target")


@pytest.mark.parametrize("target", TARGETS, ids=lambda o: o.value)
@pytest.mark.parametrize("action", ["register", "replace", "unregister"])
def test_a_locked_name_is_refused_for_every_action_and_every_target(hub, policy_for, target, action):
    registry = _seed(hub, target)
    hub.configure(policy_for(locked=frozenset({"trainer_policy:target"}), allow_builtin_override=True))
    with pytest.raises(ComponentLocked, match="COMPONENTS.LOCKED"):
        if action == "unregister":
            registry.unregister("target")
        else:
            registry.register("target", OtherPolicy, replace=action == "replace")
    assert registry.get("target").factory is Policy


def test_a_locked_name_refuses_even_a_free_registration(hub, policy_for):
    hub.configure(policy_for(locked=frozenset({"trainer_policy:reserved"})))
    with pytest.raises(ComponentLocked):
        hub.registry("trainer_policy").register("reserved", Policy)


@pytest.mark.parametrize("target", TARGETS, ids=lambda o: o.value)
def test_a_frozen_registry_refuses_global_writes_but_not_run_scope(hub, policy_for, target):
    hub.configure(policy_for(allow_builtin_override=True))
    registry = _seed(hub, target)
    registry.freeze()
    for call in (
        lambda: registry.register("free", Policy),
        lambda: registry.register("target", OtherPolicy, replace=True),
        lambda: registry.unregister("target"),
    ):
        with pytest.raises(ComponentLocked, match="frozen"):
            call()
    with hub.scoped():
        registry.register("free", Policy, scope="run")
        registry.register("target", OtherPolicy, scope="run", replace=True, origin=Origin.RUNTIME)
        assert registry.get("target").factory is OtherPolicy
    assert registry.get("target").factory is Policy


# -- the last row: a legacy write -------------------------------------------------
def _legacy_cells():
    # (target, strict_legacy, flag) -> outcome of ``table["target"] = Other``
    for target in TARGETS:
        for strict in (False, True):
            for flag in (False, True):
                if target is Origin.BUILTIN:
                    outcome = OK if (not strict or flag) else LOCKED
                else:
                    outcome = OK  # runtime-origin write over plugin/project/runtime
                yield pytest.param(target, strict, flag, outcome,
                                   id=f"{target.value}-strict{int(strict)}-flag{int(flag)}")


@pytest.mark.parametrize("target, strict, flag, outcome", list(_legacy_cells()))
def test_a_legacy_dict_write(hub, policy_for, caplog, target, strict, flag, outcome):
    hub.configure(policy_for(strict_legacy=strict, allow_builtin_override=flag))
    registry = _seed(hub, target)
    table = LegacyTable("trainer_policy", hub=hub)
    assert table["target"] is Policy
    with caplog.at_level(logging.WARNING, logger="mactf.components"):
        if outcome is OK:
            table["target"] = OtherPolicy
            assert registry.get("target").factory is OtherPolicy and table["target"] is OtherPolicy
            assert registry.get("target").origin is Origin.RUNTIME
        else:
            with pytest.raises(outcome, match="STRICT_LEGACY"):
                table["target"] = OtherPolicy
            assert table["target"] is Policy and registry.get("target").factory is Policy
    messages = [r.getMessage() for r in caplog.records]
    warned_w005 = any(m.startswith("W005 trainer_policy:target replaced") for m in messages)
    warned_w001 = any(m.startswith("W001") for m in messages)
    if target is Origin.BUILTIN and outcome is OK:
        # a built-in was overridden through the legacy door, however permission was obtained
        assert warned_w001 and warned_w005
    else:
        assert not warned_w005 and not warned_w001


def test_a_legacy_write_that_replaces_a_builtin_logs_w005_and_w001(hub, caplog):
    registry = _seed(hub, Origin.BUILTIN)
    table = LegacyTable("trainer_policy", hub=hub)
    with caplog.at_level(logging.WARNING, logger="mactf.components"):
        table["target"] = OtherPolicy
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("W005 trainer_policy:target") for m in messages)
    assert any(m.startswith("W001 trainer_policy:target") for m in messages)
    assert registry.get("target").overrides_builtin


def test_unregistering_a_component_a_run_is_using_does_not_touch_the_run():
    """Section 4.3: a slot holds the built instance and its record, not a registry reference."""
    from mactf.components import HUB, ComponentResolver, SlotTable

    slots = SlotTable(ComponentResolver())
    with HUB.scoped():
        HUB.registry("trainer_policy").register("zz_in_use", Policy, scope="run")
        slot = slots.install("TRAINER.POLICY", "trainer_policy", "zz_in_use")
        instance = slot.instance
        HUB.registry("trainer_policy").unregister("zz_in_use", scope="run")
        assert not HUB.registry("trainer_policy").has("zz_in_use")
    assert slots.get("TRAINER.POLICY").instance is instance
    assert slots.get("TRAINER.POLICY").component.name == "zz_in_use"
