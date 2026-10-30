"""Property tests: the registry against a reference model of section 4.3, and layered specs."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from component_helpers import Policy
from mactf.components import ComponentSpec, Hub, HubPolicy, Origin, merge_layers
from mactf.components.registry import Component
from mactf.exceptions import ComponentCollision, ComponentLocked, ComponentMissing

NAMES = ["a", "b", "c"]
ORIGINS = [Origin.PLUGIN, Origin.PROJECT, Origin.RUNTIME]


def make_factory(index):
    return type(f"F{index}", (Policy,), {})


FACTORIES = [make_factory(i) for i in range(4)]

operation = st.one_of(
    st.tuples(st.just("builtin"), st.sampled_from(NAMES), st.integers(0, 3)),
    st.tuples(
        st.just("register"), st.sampled_from(NAMES), st.integers(0, 3), st.sampled_from(ORIGINS),
        st.booleans(), st.booleans(),  # replace, scope=run
    ),
    st.tuples(st.just("unregister"), st.sampled_from(NAMES), st.booleans()),
    st.tuples(st.just("freeze"), st.booleans()),
    st.tuples(st.just("policy"), st.booleans(), st.booleans()),  # allow_builtin_override, lock "c"
)


class Model:
    """What section 4.3 says, written independently of the registry's code."""

    def __init__(self):
        self.base = {}  # name -> (origin, factory index)
        self.overlay = {}  # name -> (origin, factory index) | None (hidden)
        self.flag = False
        self.lock_c = False
        self.frozen = False

    def visible(self, name):
        if name in self.overlay:
            return self.overlay[name]
        return self.base.get(name)

    def register(self, name, factory, origin, replace, run):
        if self.lock_c and name == "c":
            return "locked"
        if not run and self.frozen:
            return "locked"
        existing = self.visible(name) if run else self.base.get(name)
        if existing is not None:
            if not replace:
                return "collision"
            if existing[0] is Origin.BUILTIN:
                if not self.flag:
                    return "collision"
            elif origin.rank < existing[0].rank:
                return "collision"
        (self.overlay if run else self.base)[name] = (origin, factory)
        return "ok"

    def unregister(self, name, run):
        if self.lock_c and name == "c":
            return "locked"
        if not run and self.frozen:
            return "locked"
        existing = self.visible(name) if run else self.base.get(name)
        if existing is None:
            return "missing"
        if existing[0] is Origin.BUILTIN and not self.flag:
            return "locked"
        if run:
            self.overlay[name] = None
        else:
            del self.base[name]
        return "ok"


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(operation, max_size=25))
def test_the_registry_agrees_with_the_reference_model(ops):
    hub = Hub()
    registry = hub.registry("trainer_policy")
    mirror: dict = {}
    registry.attach_mirror(mirror)
    model = Model()
    with hub.scoped():
        for op in ops:
            kind = op[0]
            if kind == "builtin":
                _, name, index = op
                if name in model.base and model.base[name][0] is not Origin.BUILTIN:
                    continue
                registry.register_builtin(name, FACTORIES[index])
                model.base[name] = (Origin.BUILTIN, index)
                continue
            if kind == "freeze":
                (registry.freeze if op[1] else registry.unfreeze)()
                model.frozen = op[1]
                continue
            if kind == "policy":
                _, model.flag, model.lock_c = op
                hub.configure(HubPolicy(
                    allow_builtin_override=model.flag,
                    locked=frozenset({"trainer_policy:c"}) if model.lock_c else frozenset()))
                continue
            before_base = dict(registry._entries)
            before_mirror = dict(mirror)
            if kind == "register":
                _, name, index, origin, replace, run = op
                expected = model.register(name, index, origin, replace, run)
                try:
                    registry.register(name, FACTORIES[index], origin=origin, replace=replace,
                                      scope="run" if run else "global")
                    actual = "ok"
                except ComponentCollision:
                    actual = "collision"
                except ComponentLocked:
                    actual = "locked"
            else:
                _, name, run = op
                expected = model.unregister(name, run)
                try:
                    registry.unregister(name, scope="run" if run else "global")
                    actual = "ok"
                except ComponentLocked:
                    actual = "locked"
                except ComponentMissing:
                    actual = "missing"
            assert actual == expected, (op, actual, expected)
            if actual != "ok":  # a refusal is atomic: base, mirror and overlay are untouched
                assert dict(registry._entries) == before_base and dict(mirror) == before_mirror
            # the visible set, the factories and the origins agree with the model
            for name in NAMES:
                found = model.visible(name)
                got = registry.find(name)
                if found is None:
                    assert got is None, (name, op)
                else:
                    assert got is not None and got.origin is found[0], (name, op)
                    assert got.factory is FACTORIES[found[1]]
            # the mirror always equals the base layer
            assert mirror == {n: c.factory for n, c in registry._entries.items()}
            assert registry.names() == sorted(registry.names())
            assert isinstance(next(iter(registry._entries.values()), None), (Component, type(None)))
    # leaving the scope drops every overlay write
    assert {n: (c.origin, c.factory) for n, c in registry._entries.items()} == {
        n: (o, FACTORIES[i]) for n, (o, i) in model.base.items()}
    assert hub.active_overlays() == ()


layer_names = ["default", "settings", "conf", "config_files", "env", "config", "set", "api"]
param_dict = st.dictionaries(st.sampled_from("xyz"), st.integers(0, 5), max_size=3)
layer_spec = st.one_of(
    st.none(),
    st.sampled_from(["a", "b"]).map(lambda t: {"type": t}),
    st.tuples(st.sampled_from(["a", "b"]), param_dict).map(lambda p: {"type": p[0], "params": p[1]}),
    param_dict.map(lambda p: {"params": p}),
)


@settings(max_examples=200, deadline=None)
@given(st.fixed_dictionaries({name: layer_spec for name in layer_names}))
def test_layered_resolution_agrees_with_a_fold_over_the_layers(layers):
    """Highest layer wins per parameter; a different component drops everything below it."""
    chosen = {name: value for name, value in layers.items() if value is not None}
    ordered = [(name, chosen[name]) for name in layer_names if name in chosen]
    first_named = next((i for i, (_n, v) in enumerate(ordered) if "type" in v), None)
    if first_named is None:
        if ordered:
            with pytest.raises(Exception):
                merge_layers("weight_update", layers)
        else:
            assert merge_layers("weight_update", layers).spec is None
        return
    # patches below the first named layer have nothing to patch; they are an error only if first
    if first_named != 0:
        with pytest.raises(Exception):
            merge_layers("weight_update", layers)
        return
    result = merge_layers("weight_update", layers)
    expected_type, expected_params, expected_layers = None, {}, {}
    for name, value in ordered:
        if "type" in value and value["type"] != expected_type:
            expected_type, expected_params, expected_layers = value["type"], {}, {}
        for key, number in value.get("params", {}).items():
            expected_params[key] = number
            expected_layers[key] = name
    assert result.spec.type == expected_type
    assert dict(result.spec.params) == expected_params
    assert dict(result.param_layers) == expected_layers
    assert result.spec == ComponentSpec("weight_update", type=expected_type, params=expected_params)
