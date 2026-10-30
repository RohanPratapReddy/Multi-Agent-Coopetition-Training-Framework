"""Adversarial verification of D2 for the registries, the legacy dict views and the import
allow-list (doc 17 sections 3, 4, 5): a customisation cannot take a built-in's name, cannot give
itself the ``builtin`` origin, cannot reach code outside the allow-list, and a dict write that
skips the registry does not make a component trusted."""

from __future__ import annotations

import sys
import threading
import types

import pytest

from mactf.components import HUB, ComponentResolver, ComponentSpec, HubPolicy, Origin, overlay, register_policy
from mactf.components.decorators import register_scorer
from mactf.coopetition.strategy import BUILTIN_STRATEGY_TYPES, PEER_STRATEGIES, PeerStrategy
from mactf.coopetition.weights import BUILTIN_WEIGHT_TYPES, WEIGHT_STRATEGIES
from mactf.exceptions import (
    ComplianceGuardViolation,
    ComponentCollision,
    ComponentContractError,
    ComponentImportDenied,
    ComponentLocked,
    ComponentSpecError,
)
from mactf.scoring.penalty import PenaltyScorer
from mactf.scoring.registry import SCORER_TYPES

from adv_support import breach_trainer, run_epochs
from p2_support import configure
from scoring_helpers import penalty_variant, with_output
from dataclasses import replace


class _Policy:
    name = "evil_policy"
    unsafe = True
    self_stepping = False

    def train_step(self, node, batch):  # pragma: no cover
        raise NotImplementedError


def _evil_penalty():
    return penalty_variant(lambda out, ctx: replace(out, amount=0.0, void_rules=(),
                                                    breakdown={**out.breakdown, "amounts": {}}))


# -- the builtin origin is the framework's -----------------------------------------------------
def test_a_component_cannot_register_itself_as_builtin(component_overlay):
    with pytest.raises(ComponentContractError, match="reserved for the framework"):
        HUB.registry("trainer_policy").register("evil_policy", _Policy, origin=Origin.BUILTIN, scope="run")
    with pytest.raises(ComponentContractError, match="reserved for the framework"):
        register_policy("evil_policy", _Policy, origin=Origin.BUILTIN, scope="run")
    assert not HUB.registry("trainer_policy").has("evil_policy")


def test_the_frameworks_own_modules_can_still_register_builtins():
    assert HUB.registry("peer_factory").get("clone").origin is Origin.BUILTIN
    assert HUB.registry("trainer_policy").get("default").origin is Origin.BUILTIN


# -- replacing a built-in -------------------------------------------------------------------------
@pytest.fixture
def restore_tables():
    """Whatever a test does to the legacy tables, the framework's own classes come back."""
    yield
    for kind, table in (("scorer", SCORER_TYPES), ("peer_strategy", PEER_STRATEGIES),
                        ("weight_update", WEIGHT_STRATEGIES)):
        registry = HUB.registry(kind)
        for name in list(registry._builtin_factories):
            registry.restore_builtin(name)
        for name in [n for n in table if n not in registry._builtin_factories]:
            registry.unregister(name, via="legacy")


@pytest.mark.parametrize("table, name", [(SCORER_TYPES, "penalty"), (PEER_STRATEGIES, "kappa_softmax"),
                                         (WEIGHT_STRATEGIES, "kappa_softmax")])
def test_strict_legacy_refuses_every_write_over_a_builtin(table, name, restore_tables):
    original = dict(table)
    from mactf.coopetition.weights import WeightStrategy

    if table is SCORER_TYPES:
        evil = _evil_penalty()
    elif table is PEER_STRATEGIES:
        evil = type("Evil", (PeerStrategy,), {"name": "evil", "coordinate": lambda self, ctx: None})
    else:
        evil = type("Evil", (WeightStrategy,), {"name": "evil", "apply": lambda self, *a, **k: None})
    with HUB.scoped(policy=HubPolicy(strict_legacy=True)):
        for attempt in (
            lambda: table.__setitem__(name, evil),
            lambda: table.update({name: evil}),
            lambda: table.__delitem__(name),
            lambda: table.pop(name),
            lambda: table.clear(),
            lambda: table.__ior__({name: evil}),
        ):
            with pytest.raises((ComponentLocked, ComponentCollision)):
                attempt()
            assert dict(table) == original, "a built-in table was changed under STRICT_LEGACY"


def test_a_legacy_replacement_is_logged_and_the_replacement_is_a_custom_class(restore_tables, caplog):
    """Default compatibility mode lets the old dict-write idiom through (W001) - but the class that
    now answers to the built-in's name gets no built-in trust."""
    import logging

    from mactf.components.guards.compliance import ComplianceGuard

    evil = _evil_penalty()
    with caplog.at_level(logging.WARNING):
        SCORER_TYPES["penalty"] = evil
    assert "W001" in caplog.text
    assert not ComplianceGuard()._trusted(evil(config={}))
    SCORER_TYPES["penalty"] = PenaltyScorer                   # putting the original back restores the origin
    assert HUB.registry("scorer").get("penalty").origin is Origin.BUILTIN


def test_the_decorator_cannot_take_a_builtin_name(component_overlay):
    with pytest.raises((ComponentCollision, ComponentLocked)):
        register_scorer("penalty", _evil_penalty(), scope="run")
    assert SCORER_TYPES["penalty"] is PenaltyScorer


def test_the_flag_allows_replacement_but_the_replacement_is_still_a_custom_stage(settings, tmp_path, component_overlay):
    """ALLOW_BUILTIN_OVERRIDE lets an operator swap the name; it never makes the class trusted."""
    with HUB.scoped(policy=HubPolicy(allow_builtin_override=True)):
        register_scorer("penalty", _evil_penalty(), replace=True, scope="run")
        configure(settings, SCORING={"STAGES": {"penalty": "penalty"}})
        trainer = breach_trainer(settings, tmp_path)
        with pytest.raises(ComplianceGuardViolation) as caught:
            run_epochs(trainer)
    assert caught.value.code == "C002"


def test_a_raw_dict_write_does_not_make_a_class_trusted():
    """``dict.__setitem__`` skips the registry; the trust tables are frozen snapshots."""
    saved = dict(PEER_STRATEGIES)
    evil = type("Evil", (PeerStrategy,), {"name": "evil"})
    try:
        dict.__setitem__(PEER_STRATEGIES, "kappa_softmax", evil)
        assert evil not in BUILTIN_STRATEGY_TYPES and evil not in BUILTIN_WEIGHT_TYPES
    finally:
        dict.clear(PEER_STRATEGIES)
        dict.update(PEER_STRATEGIES, saved)
    with pytest.raises(AttributeError):
        BUILTIN_STRATEGY_TYPES.add(evil)                     # a frozenset


def test_a_class_with_a_lying_metaclass_is_not_in_the_trust_tables():
    from abc import ABCMeta

    class Liar(ABCMeta):
        def __eq__(cls, other):
            return True

        def __hash__(cls):
            return hash(next(iter(BUILTIN_STRATEGY_TYPES)))

    from mactf.components.guards.strategy import StrategyGuard

    evil = Liar("Evil", (PeerStrategy,), {})
    assert StrategyGuard(scope="custom").covers(evil())
    from mactf.coopetition.weight_guard import WeightGuard

    from mactf.coopetition.weights import UniformAverage

    assert WeightGuard(scope="custom").covers(Liar("W", (UniformAverage,), {})(), None)


def test_a_locked_slot_refuses_registration(component_overlay):
    with HUB.scoped(policy=HubPolicy(locked=("trainer_policy:*",))):
        with pytest.raises(ComponentLocked):
            register_policy("late", _Policy, scope="run")


def test_concurrent_registration_loses_nothing_and_collides_loudly():
    errors: list[BaseException] = []
    names = [f"adv_thread_{i}" for i in range(12)]

    def work(name):
        try:
            register_policy(name, type(name, (_Policy,), {"name": name, "unsafe": False}))
            register_policy(name, type(name + "x", (_Policy,), {"name": name, "unsafe": False}))
        except ComponentCollision:
            pass
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=work, args=(n,)) for n in names]
    try:
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert not errors
        assert all(HUB.registry("trainer_policy").has(n) for n in names)
    finally:
        registry = HUB.registry("trainer_policy")
        with HUB.scoped(policy=HubPolicy(allow_builtin_override=True)):
            for name in names:
                try:
                    registry.unregister(name)
                except Exception:  # noqa: BLE001
                    pass


# -- the import allow-list -----------------------------------------------------------------------
@pytest.fixture
def acme(monkeypatch):
    import os

    module = types.ModuleType("adv_acme_policies")

    class Good(_Policy):
        name = "good"
        unsafe = False

    module.Good = Good
    module.os = os                                           # a module the allow-list never named
    module.sys = sys
    module.sneaky = types.SimpleNamespace(Good=Good)
    monkeypatch.setitem(sys.modules, "adv_acme_policies", module)
    return module


def _resolve(path, kind="trainer_policy", **policy):
    spec = ComponentSpec.parse(kind, {"import": path})
    with HUB.scoped(policy=HubPolicy(allow_import=True, import_allowlist=("adv_acme_policies",), **policy)):
        return ComponentResolver().resolve(spec, trusted=False, check_instance=False)


@pytest.mark.parametrize("path", ["adv_acme_policies:os.system", "adv_acme_policies:sys.exit",
                                  "adv_acme_policies:os.path.join"])
def test_an_attribute_path_cannot_walk_out_of_an_allowed_module(path, acme, component_overlay):
    with pytest.raises(ComponentImportDenied, match="reaches module"):
        _resolve(path)


def test_an_allowed_module_still_resolves(acme, component_overlay):
    assert _resolve("adv_acme_policies:Good").component.factory is acme.Good


@pytest.mark.parametrize("path", ["os:system", "subprocess:Popen", "adv_acme_policies_evil:Good",
                                  "adv_acme:Good", "ADV_ACME_POLICIES:Good"])
def test_a_module_outside_the_list_is_never_imported(path, acme, component_overlay):
    with pytest.raises(ComponentImportDenied):
        _resolve(path)


@pytest.mark.parametrize("raw", [".mod:X", "pkg.mod:", "pkg.mod", "pkg/mod:X", "pkg.mod:X;os",
                                 "pkg.mod:__import__('os')", "pkg mod:X", "../x:X"])
def test_a_malformed_import_string_is_a_spec_error(raw):
    with pytest.raises(ComponentSpecError):
        ComponentSpec.parse("trainer_policy", {"import": raw})


def test_a_spec_cannot_set_a_parameter_the_framework_injects():
    from mactf.components import register
    resolver = ComponentResolver()
    with overlay():
        class Needs(_Policy):
            unsafe = False

            def __init__(self, *, trainer=None):
                self.trainer = trainer

        register_policy("needs_trainer", Needs, scope="run")
        with pytest.raises(ComponentSpecError, match="E011"):
            resolver.validate(ComponentSpec.parse("trainer_policy", {"type": "needs_trainer",
                                                                      "params": {"trainer": "x"}}),
                              trusted=True, inject={"trainer": object()})
