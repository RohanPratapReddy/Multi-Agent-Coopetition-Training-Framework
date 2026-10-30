"""Locking (document 17 section 4.5): components, slots, the grammar of ``COMPONENTS.LOCKED``."""

from __future__ import annotations

import pytest

from component_helpers import OtherPolicy, Policy
from mactf.components import (
    HUB,
    Boundary,
    ComponentResolver,
    HubPolicy,
    SlotTable,
    load_components,
    plugins,
)
from mactf.components.known import LOCKABLE_KINDS, lock_pattern_problem
from mactf.exceptions import ComponentLocked, ImproperlyConfigured


@pytest.mark.parametrize(
    "locked, kind, name, expected",
    [
        (["scorer:penalty"], "scorer", "penalty", True),
        (["scorer:penalty"], "scorer", "reward", False),
        (["scorer:penalty"], "peer_strategy", "penalty", False),
        (["weight_update:*"], "weight_update", "anything", True),
        (["weight_update:*"], "scorer", "anything", False),
        (["rl_function.loss:ppo"], "rl_function.loss", "ppo", True),
        (["rl_function.loss:ppo"], "rl_function.advantage", "ppo", False),
        (["rl_function.*:kl*"], "rl_function.loss", "kl_reference", True),
        (["scorer:pen*", "scorer:war*"], "scorer", "warning", True),
        (["slot:SCORING.CHAIN.penalty"], "scorer", "penalty", False),  # a slot lock locks no component
        ([], "scorer", "penalty", False),
    ],
)
def test_is_locked_is_fnmatch_over_kind_colon_name(locked, kind, name, expected):
    assert HubPolicy(locked=frozenset(locked)).is_locked(kind, name) is expected


def test_a_locked_component_is_not_shadowed_by_an_overlay(hub, policy_for):
    registry = hub.registry("trainer_policy")
    registry.register("pinned", Policy)
    hub.configure(policy_for(locked=frozenset({"trainer_policy:pinned"})))
    with hub.scoped():
        with pytest.raises(ComponentLocked):
            registry.register("pinned", OtherPolicy, scope="run", replace=True)
        with pytest.raises(ComponentLocked):
            registry.unregister("pinned", scope="run")
        assert registry.get("pinned").factory is Policy
    with pytest.raises(ComponentLocked):
        registry.register("pinned", OtherPolicy, replace=True)


def test_the_lock_comes_from_the_scoped_policy_too(hub, policy_for):
    registry = hub.registry("trainer_policy")
    with hub.scoped(policy=policy_for(locked=frozenset({"trainer_policy:*"}))):
        with pytest.raises(ComponentLocked):
            registry.register("x", Policy, scope="run")
    registry.register("x", Policy)


# -- slot locks --------------------------------------------------------------------
def test_a_slot_lock_blocks_swap_and_request_but_not_install():
    from mactf.components import ComponentSpec  # noqa: F401  (documenting the public type)

    slots = SlotTable(ComponentResolver())
    with HUB.scoped(policy=HubPolicy(locked=frozenset({"slot:TRAINER.POLICY*"}))):
        HUB.registry("trainer_policy").register("zz_a", Policy, scope="run")
        HUB.registry("trainer_policy").register("zz_b", OtherPolicy, scope="run")
        slots.install("TRAINER.POLICY", "trainer_policy", "zz_a")  # build time is not a change
        with pytest.raises(ComponentLocked, match="slot is locked"):
            slots.swap("TRAINER.POLICY", "zz_b", boundary=Boundary.RUN_START)
        with pytest.raises(ComponentLocked, match="slot is locked"):
            slots.request("TRAINER.POLICY", "zz_b")
        assert slots.get("TRAINER.POLICY").component.name == "zz_a"
        assert slots.pending() == {}


def test_a_slot_lock_does_not_lock_other_slots():
    slots = SlotTable(ComponentResolver())
    with HUB.scoped(policy=HubPolicy(locked=frozenset({"slot:SCORING.CHAIN.penalty"}))):
        HUB.registry("trainer_policy").register("zz_a", Policy, scope="run")
        HUB.registry("trainer_policy").register("zz_b", OtherPolicy, scope="run")
        slots.install("OTHER", "trainer_policy", "zz_a")
        assert slots.swap("OTHER", "zz_b", boundary=Boundary.RUN_START).to_name == "zz_b"


def test_a_kind_lock_is_not_a_slot_lock():
    slots = SlotTable(ComponentResolver())
    with HUB.scoped(policy=HubPolicy(locked=frozenset({"trainer_policy:zz_a"}))):
        HUB.registry("trainer_policy").register("zz_b", Policy, scope="run")
        HUB.registry("trainer_policy").register("zz_c", OtherPolicy, scope="run")
        slots.install("S", "trainer_policy", "zz_b")
        slots.swap("S", "zz_c", boundary=Boundary.RUN_START)


# -- the grammar -------------------------------------------------------------------
@pytest.mark.parametrize(
    "pattern",
    [
        "scorer:penalty", "weight_update:*", "peer_strategy:kappa_softmax",
        "rl_function.loss:ppo", "optimizer:AdamW", "aggregator:default",
        "trainer_policy:co_distill", "slot:SCORING.CHAIN.penalty", "slot:TRAINER.WEIGHT_UPDATE*",
        "slot:*", "*:penalty", "rl_function.*:kl*", "scorer:with spaces ok?",
    ],
)
def test_a_lock_pattern_needs_a_kind_or_slot(pattern):
    assert lock_pattern_problem(pattern, kinds=LOCKABLE_KINDS | {"slot"}) is None


@pytest.mark.parametrize(
    "pattern, fragment",
    [
        ("penalty", "must be 'kind:name' or 'slot:<path>'"),
        ("scorer:", "must be 'kind:name' or 'slot:<path>'"),
        (":penalty", "must be 'kind:name' or 'slot:<path>'"),
        ("nonsense:penalty", "names unknown kind 'nonsense'"),
        ("rl_function.nonsense:x", "names unknown kind 'rl_function.nonsense'"),
        # a bare family-less kind matches no registry key, so it would lock nothing
        ("rl_function:ppo", "names unknown kind 'rl_function'"),
        ("", "is not a usable pattern"),
        (" scorer:penalty", "is not a usable pattern"),
        ("scorer:penalty ", "is not a usable pattern"),
        (7, "is not a usable pattern"),
        (None, "is not a usable pattern"),
    ],
)
def test_a_bad_lock_pattern_is_explained(pattern, fragment):
    problem = lock_pattern_problem(pattern, kinds=LOCKABLE_KINDS | {"slot"})
    assert problem is not None and fragment in problem


# -- locks apply after loading ------------------------------------------------------
def test_locks_are_applied_in_phase_two_so_plugin_modules_can_register(restore_hub, monkeypatch):
    import sys
    import types

    module = types.ModuleType("zz_locked_module")

    def register_on_import():
        HUB.registry("trainer_policy").register("zz_pinned", Policy)

    module.register_on_import = register_on_import
    monkeypatch.setitem(sys.modules, "zz_locked_module", module)
    # importing the module registers during phase 1 (no locks yet) even though the name is locked
    original = plugins.importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "zz_locked_module":
            module.register_on_import()
            return module
        return original(name, *args, **kwargs)

    monkeypatch.setattr(plugins.importlib, "import_module", fake_import)
    plugins.reset_for_tests()
    report = load_components(
        {"MODULES": ["zz_locked_module"], "LOCKED": ["trainer_policy:zz_pinned"], "PLUGINS": "off"}
    )
    assert report.modules == ["zz_locked_module"]
    assert HUB.registry("trainer_policy").has("zz_pinned")
    assert HUB.policy.is_locked("trainer_policy", "zz_pinned")  # phase 2 locked it afterwards
    with pytest.raises(ComponentLocked):
        HUB.registry("trainer_policy").register("zz_pinned", OtherPolicy, replace=True)
    with pytest.raises(ComponentLocked):
        HUB.registry("trainer_policy").unregister("zz_pinned")


def test_builtin_registration_is_never_blocked_by_a_lock_or_a_freeze(hub, policy_for):
    """A deployment that locks ``scorer:*`` must still be able to import ``mactf.scoring``."""
    hub.configure(policy_for(locked=frozenset({"trainer_policy:*"}), max_components_per_kind=1))
    registry = hub.registry("trainer_policy")
    registry.freeze()
    registry.register_builtin("a", Policy)
    registry.register_builtin("b", Policy)
    assert registry.names() == ["a", "b"]


# -- the settings spelling of a slot (doc 17 4.4 / 4.5 lock examples) ---------------
@pytest.mark.parametrize(
    "pattern, path, locked",
    [
        ("slot:SCORING.CHAIN.penalty", "scoring.penalty", True),
        ("slot:SCORING.CHAIN.penalty", "scoring.reward", False),
        ("slot:SCORING.CHAIN.*", "scoring.warning", True),
        ("slot:scoring.penalty", "scoring.penalty", True),
        ("slot:SCORING.REWARD_TERMS.*", "scoring.reward.terms.tone", True),
        ("slot:TRAINER.OPTIMIZER.default", "OPTIMIZER", True),
        ("slot:TRAINER.OPTIMIZER.*", "OPTIMIZER.root/a", True),
        ("slot:TRAINER.OPTIMIZER.default", "OPTIMIZER.root/a", False),
    ],
)
def test_a_lock_in_the_settings_spelling_locks_the_engine_slot(pattern, path, locked):
    from mactf.components.slots import slot_locked_by

    assert slot_locked_by(path, [pattern]) is locked
