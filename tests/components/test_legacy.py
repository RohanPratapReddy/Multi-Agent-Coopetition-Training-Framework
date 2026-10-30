"""``LegacyTable`` and the four real legacy views (document 17 section 3.6)."""

from __future__ import annotations

import copy
import json
import logging
import pickle

import pytest

from component_helpers import OtherPolicy, Policy
from mactf.components import HUB, LegacyTable, Origin
from mactf.exceptions import ComponentCollision, ComponentContractError, ComponentLocked


@pytest.fixture
def table(hub):
    registry = hub.registry("trainer_policy")
    registry.register_builtin("shipped", Policy)
    return LegacyTable("trainer_policy", hub=hub), registry


# -- it is a dict --------------------------------------------------------------------
def test_a_table_is_a_real_dict(table):
    legacy, _ = table
    assert isinstance(legacy, dict) and legacy == {"shipped": Policy}
    assert dict(legacy) == {"shipped": Policy} and sorted(legacy) == ["shipped"]
    assert list(legacy.items()) == [("shipped", Policy)] and len(legacy) == 1
    assert legacy.get("nope") is None and "shipped" in legacy
    assert json.dumps({k: v.__name__ for k, v in legacy.items()}) == '{"shipped": "Policy"}'
    assert legacy.copy() == {"shipped": Policy} and type(legacy.copy()) is dict
    assert legacy.kind_id == "trainer_policy"


def test_a_table_pickles_and_copies_as_a_plain_snapshot(table):
    legacy, _ = table
    assert type(pickle.loads(pickle.dumps(legacy))) is dict
    assert type(copy.deepcopy(legacy)) is dict and copy.deepcopy(legacy) == dict(legacy)


def test_a_table_registers_its_builtins_at_construction(hub):
    legacy = LegacyTable("trainer_policy", {"a": Policy, "b": OtherPolicy}, hub=hub)
    registry = hub.registry("trainer_policy")
    assert dict(legacy) == {"a": Policy, "b": OtherPolicy}
    assert [(c.name, c.origin) for c in registry.components()] == [
        ("a", Origin.BUILTIN), ("b", Origin.BUILTIN)
    ]


def test_a_builtin_that_fails_its_contract_stops_the_import(hub):
    with pytest.raises(ComponentContractError, match="missing method train_step"):
        LegacyTable("trainer_policy", {"bad": object}, hub=hub)


# -- writes go through the registry ----------------------------------------------------
def test_setitem_registers_a_runtime_component(table):
    legacy, registry = table
    legacy["fresh"] = OtherPolicy
    assert legacy["fresh"] is OtherPolicy
    component = registry.get("fresh")
    assert component.factory is OtherPolicy and component.origin is Origin.RUNTIME
    assert not component.overrides_builtin


def test_setitem_over_a_builtin_is_tolerated_and_warns(table, caplog):
    legacy, registry = table
    with caplog.at_level(logging.WARNING, logger="mactf.components"):
        legacy["shipped"] = OtherPolicy
    assert registry.get("shipped").overrides_builtin and legacy["shipped"] is OtherPolicy
    assert {r.getMessage().split()[0] for r in caplog.records} >= {"W001", "W005"}


def test_setitem_with_an_invalid_factory_fails_at_write_time_and_changes_nothing(table):
    legacy, registry = table
    before = (dict(legacy), registry.names())
    with pytest.raises(ComponentContractError):
        legacy["bad"] = object
    with pytest.raises(ComponentContractError, match="invalid trainer_policy type name"):
        legacy["not a name"] = Policy
    assert (dict(legacy), registry.names()) == before


def test_restoring_the_original_builtin_makes_it_a_builtin_again(table):
    """The pattern every existing test uses: replace, then put the original back."""
    legacy, registry = table
    original = legacy["shipped"]
    legacy["shipped"] = OtherPolicy
    assert registry.get("shipped").origin is Origin.RUNTIME
    legacy["shipped"] = original
    restored = registry.get("shipped")
    assert restored.origin is Origin.BUILTIN and not restored.overrides_builtin
    assert restored.factory is Policy and legacy["shipped"] is Policy


def test_restoring_works_after_a_delete_too(table):
    legacy, registry = table
    original = legacy.pop("shipped")
    assert "shipped" not in legacy and not registry.has("shipped")
    legacy["shipped"] = original
    assert registry.get("shipped").origin is Origin.RUNTIME  # re-adding after removal is a new write


def test_delitem_pop_popitem_clear(table):
    legacy, registry = table
    legacy["a"] = Policy
    legacy["b"] = OtherPolicy
    del legacy["a"]
    assert "a" not in legacy and not registry.has("a")
    with pytest.raises(KeyError):
        del legacy["a"]
    assert legacy.pop("b") is OtherPolicy and not registry.has("b")
    assert legacy.pop("b", "default") == "default"
    with pytest.raises(KeyError):
        legacy.pop("b")
    legacy["c"] = Policy
    name, factory = legacy.popitem()
    assert (name, factory) == ("c", Policy)
    legacy.clear()
    assert dict(legacy) == {} and registry.names() == []
    with pytest.raises(KeyError):
        legacy.popitem()


def test_setdefault_update_and_in_place_or(table):
    legacy, registry = table
    assert legacy.setdefault("shipped", OtherPolicy) is Policy  # present: untouched
    assert legacy.setdefault("new", OtherPolicy) is OtherPolicy and registry.has("new")
    legacy.update({"u1": Policy}, u2=OtherPolicy)
    legacy |= {"u3": Policy}
    assert {"u1", "u2", "u3"} <= set(legacy) and all(registry.has(n) for n in ("u1", "u2", "u3"))
    assert registry.get("u3").origin is Origin.RUNTIME


def test_a_frozen_registry_refuses_legacy_writes(table):
    legacy, registry = table
    registry.freeze()
    with pytest.raises(ComponentLocked, match="frozen"):
        legacy["x"] = Policy
    with pytest.raises(ComponentLocked, match="frozen"):
        legacy.pop("shipped")
    assert dict(legacy) == {"shipped": Policy}


def test_a_locked_name_refuses_legacy_writes(table, hub, policy_for):
    legacy, _ = table
    hub.configure(policy_for(locked=frozenset({"trainer_policy:shipped"})))
    with pytest.raises(ComponentLocked):
        legacy["shipped"] = OtherPolicy
    with pytest.raises(ComponentLocked):
        del legacy["shipped"]
    assert legacy["shipped"] is Policy


def test_the_table_equals_the_base_layer_and_ignores_overlays(table, hub):
    legacy, registry = table
    with hub.scoped():
        registry.register("private", Policy, scope="run")
        assert registry.has("private") and "private" not in legacy
    assert dict(legacy) == {c.name: c.factory for c in registry.components()}


# -- the real tables ------------------------------------------------------------------
def test_the_real_tables_are_views_of_the_hub():
    from mactf.coopetition.strategy import PEER_STRATEGIES
    from mactf.coopetition.weights import WEIGHT_STRATEGIES
    from mactf.rl.functions.registry import CAPABILITY_ORDER, FUNCTION_TYPES
    from mactf.scoring.registry import SCORER_TYPES

    views = {
        "scorer": SCORER_TYPES,
        "peer_strategy": PEER_STRATEGIES,
        "weight_update": WEIGHT_STRATEGIES,
        **{f"rl_function.{family}": FUNCTION_TYPES[family] for family in CAPABILITY_ORDER},
    }
    assert len(views) == 15
    for kind_id, view in views.items():
        assert isinstance(view, LegacyTable) and isinstance(view, dict) and view.kind_id == kind_id
        registry = HUB.registry(kind_id)
        assert dict(view) == {c.name: c.factory for c in registry.components()}, kind_id
        assert all(c.origin is Origin.BUILTIN for c in registry.components()), kind_id
        assert set(view) == set(registry.names())
    assert type(FUNCTION_TYPES) is dict  # the outer dict is unchanged; its values are views


def test_the_documented_builtin_names_are_unchanged():
    from mactf.coopetition.strategy import PEER_STRATEGIES
    from mactf.coopetition.weights import WEIGHT_STRATEGIES
    from mactf.rl.functions.registry import FUNCTION_TYPES
    from mactf.scoring.registry import SCORER_TYPES

    # document 17 sections 6.3 and 6.8 add three built-in stage names
    assert set(SCORER_TYPES) == {
        "reward", "penalty", "warning", "hierarchy", "composed_reward", "zero_reward", "no_warnings",
    }
    assert set(PEER_STRATEGIES) == {
        "independent", "cooperative", "kappa_softmax", "frozen_peer", "hierarchical"}
    assert set(WEIGHT_STRATEGIES) == {
        "kappa_softmax", "uniform", "score_weighted", "reference_only", "none", "explicit"}
    # document 17 section 8.2 adds ``kappa_weighted``, the alias ``build_rollup`` always accepted
    assert set(FUNCTION_TYPES["rollup"]) == {
        "kappa", "kappa_weighted", "mean", "sum", "minimum", "worst_case", "explicit"}
    assert len(FUNCTION_TYPES) == 12


def test_the_family_aliases_are_the_same_objects_as_the_tables():
    from mactf.rl.functions import registry
    from mactf.rl.functions.advantage import ADVANTAGE_FUNCTION_TYPES
    from mactf.rl.functions.loss import LOSS_FUNCTION_TYPES
    from mactf.rl.functions.value import VALUE_FUNCTION_TYPES

    assert registry.VALUE_TYPES is VALUE_FUNCTION_TYPES is registry.FUNCTION_TYPES["value"]
    assert registry.ADVANTAGE_TYPES is ADVANTAGE_FUNCTION_TYPES
    assert registry.LOSS_TYPES is LOSS_FUNCTION_TYPES
    for alias, family in (
        ("ROLLUP_TYPES", "rollup"), ("RETURN_TYPES", "return"), ("REWARD_TYPES", "reward"),
        ("REGULARIZATION_TYPES", "regularization"), ("OBJECTIVE_TYPES", "objective"),
        ("GRADIENT_TYPES", "gradient"), ("METRIC_TYPES", "metric"), ("ACTION_TYPES", "action"),
        ("SYNTHESIS_TYPES", "synthesis"),
    ):
        assert getattr(registry, alias) is registry.FUNCTION_TYPES[family]


def test_legacy_tables_are_views(restore_hub):
    """A write to each legacy dict reaches the hub, and a read through the hub sees it."""
    from mactf.coopetition.strategy import PEER_STRATEGIES, IndependentStrategy
    from mactf.coopetition.weights import WEIGHT_STRATEGIES, UniformAverage
    from mactf.rl.functions.registry import FUNCTION_TYPES
    from mactf.scoring.registry import SCORER_TYPES
    from mactf.scoring.reward_scorer import RewardScorer

    class Strat(IndependentStrategy):
        name = "zz_strat"

    class Avg(UniformAverage):
        name = "zz_avg"

    class Stage(RewardScorer):
        pass

    class Marker:
        pass

    probes = [
        (SCORER_TYPES, "scorer", "zz_stage", Stage),
        (PEER_STRATEGIES, "peer_strategy", "zz_strat", Strat),
        (WEIGHT_STRATEGIES, "weight_update", "zz_avg", Avg),
        (FUNCTION_TYPES["rollup"], "rl_function.rollup", "zz_marker", Marker),
        (FUNCTION_TYPES["metric"], "rl_function.metric", "zz_marker", Marker),
    ]
    for table, kind_id, name, cls in probes:
        table[name] = cls
        assert HUB.registry(kind_id).get(name).factory is cls
        assert HUB.registry(kind_id).get(name).origin is Origin.RUNTIME
        del table[name]
        assert not HUB.registry(kind_id).has(name)
        assert name not in table


def test_replacing_a_real_builtin_through_the_table_logs_w005_and_the_original_restores(
    restore_hub, caplog
):
    from mactf.coopetition.weights import WEIGHT_STRATEGIES, UniformAverage

    class Mine(UniformAverage):
        name = "uniform"

    original = WEIGHT_STRATEGIES["uniform"]
    with caplog.at_level(logging.WARNING, logger="mactf.components"):
        WEIGHT_STRATEGIES["uniform"] = Mine
    assert any(r.getMessage().startswith("W005 weight_update:uniform") for r in caplog.records)
    assert HUB.registry("weight_update").get("uniform").overrides_builtin
    WEIGHT_STRATEGIES["uniform"] = original
    assert HUB.registry("weight_update").get("uniform").origin is Origin.BUILTIN


def test_register_scorer_and_register_function_keep_their_legacy_return_values(restore_hub):
    from mactf.rl.functions.registry import FUNCTION_TYPES, register_function
    from mactf.scoring.registry import SCORER_TYPES, register_scorer
    from mactf.scoring.reward_scorer import RewardScorer

    class Stage(RewardScorer):
        pass

    class Marker:
        pass

    assert register_scorer("zz_legacy_stage", Stage) is None
    assert register_function("rollup", "zz_legacy_marker", Marker) is None
    assert SCORER_TYPES["zz_legacy_stage"] is Stage
    assert FUNCTION_TYPES["rollup"]["zz_legacy_marker"] is Marker
    with pytest.raises(ComponentCollision):
        register_scorer("zz_legacy_stage", Stage)
