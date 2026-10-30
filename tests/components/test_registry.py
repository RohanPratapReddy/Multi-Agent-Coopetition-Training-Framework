"""``ComponentRegistry`` and ``Component`` (document 17 sections 3.3-3.4, 3.9).

Unit tests on private hubs: no torch, nothing registered on the process registry.
"""

from __future__ import annotations

import logging

import pytest

from component_helpers import OtherPolicy, Policy, Term, make_scheduler
from mactf.components import Component, Hub, HubPolicy, Mutability, Origin, ParamSpec
from mactf.exceptions import (
    ComponentBuildError,
    ComponentCollision,
    ComponentContractError,
    ComponentLocked,
    ComponentMissing,
    ComponentParamError,
    ComponentSpecError,
    ValidationError,
)


@pytest.fixture
def policies(hub):
    return hub.registry("trainer_policy")


def test_register_get_names_are_sorted(policies):
    for name in ("zeta", "alpha", "mid"):
        policies.register(name, Policy)
    assert policies.names() == ["alpha", "mid", "zeta"]
    assert policies.get("mid").factory is Policy
    assert [c.name for c in policies.components()] == ["alpha", "mid", "zeta"]
    assert policies.has("alpha") and not policies.has("nope")


def test_an_unknown_name_is_e001_with_a_suggestion(policies):
    policies.register("trimmed_mean", Policy)
    with pytest.raises(ComponentMissing) as caught:
        policies.get("trimed_mean")
    error = caught.value
    assert error.code == "E001" and error.kind == "trainer_policy"
    assert error.known == ("trimmed_mean",)
    assert "did you mean 'trimmed_mean'?" in str(error)
    assert isinstance(error, ValidationError)


def test_the_component_record_is_complete(policies):
    component = policies.register(
        "p", Policy, version="2", description="d", capabilities={"hot": True},
        requires={"scorer:penalty"},
    )
    assert component.kind == "trainer_policy" and component.name == "p"
    assert component.origin is Origin.PROJECT  # the default origin
    assert component.mutability is Mutability.STATIC
    assert component.version == "2" and component.description == "d"
    assert component.module == Policy.__module__ and component.qualname == "Policy"
    assert component.requires == frozenset({"scorer:penalty"})
    assert component.seq == 1 and policies.register("q", Policy).seq == 2
    assert isinstance(component, Component)
    with pytest.raises(Exception):
        component.name = "other"  # frozen


def test_description_defaults_to_the_first_docstring_line(policies):
    assert policies.register("p", Policy).description == "A trainer policy: the lightest registrable kind."


@pytest.mark.parametrize("name", ["", "a b", "a-b", "a.b", "naïve!", "x/y"])
def test_invalid_names_are_contract_errors(policies, name):
    with pytest.raises(ComponentContractError, match="invalid trainer_policy type name"):
        policies.register(name, Policy)
    assert policies.names() == []


def test_valid_names(policies):
    for name in ("a", "A1", "snake_case", "_x", "9lives"):
        policies.register(name, Policy)
    assert len(policies.names()) == 5


def test_function_families_use_the_historical_name_message(hub):
    with pytest.raises(ComponentContractError, match="invalid function type name 'not a name'"):
        hub.registry("rl_function.rollup").register("not a name", type("X", (), {}))


@pytest.mark.parametrize("kind_id", ["peer_strategy", "weight_update"])
def test_strategy_and_weight_names_must_be_lower_case(hub, kind_id):
    registry = hub.registry(kind_id)
    with pytest.raises(ComponentContractError, match="lower-case"):
        registry.register("Mixed", object)


def test_a_contract_failure_leaves_registry_and_mirror_untouched(hub, policies):
    mirror: dict = {}
    policies.attach_mirror(mirror)
    policies.register("ok", Policy)
    before = (dict(policies._entries), dict(mirror), hub.generation)
    with pytest.raises(ComponentContractError) as caught:
        policies.register("bad", object)  # no train_step
    assert caught.value.code == "E007" and "missing method train_step()" in str(caught.value)
    assert (dict(policies._entries), dict(mirror), hub.generation) == before


def test_a_collision_leaves_registry_and_mirror_untouched(hub, policies):
    mirror: dict = {}
    policies.attach_mirror(mirror)
    policies.register("p", Policy)
    before = (dict(policies._entries), dict(mirror), hub.generation)
    with pytest.raises(ComponentCollision) as caught:
        policies.register("p", OtherPolicy)
    assert caught.value.code == "E002" and "already registered" in str(caught.value)
    assert (dict(policies._entries), dict(mirror), hub.generation) == before
    assert mirror == {"p": Policy}


def test_replace_swaps_the_factory_and_the_mirror(policies):
    mirror: dict = {}
    policies.attach_mirror(mirror)
    policies.register("p", Policy)
    policies.register("p", OtherPolicy, replace=True)
    assert policies.get("p").factory is OtherPolicy and mirror["p"] is OtherPolicy


def test_a_looser_mutability_than_the_kind_default_is_refused(hub):
    terms = hub.registry("reward_term")  # default: episode
    with pytest.raises(ComponentContractError, match="looser than the kind default"):
        terms.register("t", Term, mutability=Mutability.STEP)
    stricter = terms.register("t", Term, mutability=Mutability.STATIC)
    assert stricter.mutability is Mutability.STATIC
    assert terms.register("u", Term).mutability is Mutability.EPISODE
    with pytest.raises(ComponentContractError):  # the static kind cannot be loosened either
        hub.registry("trainer_policy").register("p", Policy, mutability=Mutability.EPISODE)


def test_the_per_kind_cap(hub, policy_for):
    hub.configure(policy_for(max_components_per_kind=2))
    policies = hub.registry("trainer_policy")
    policies.register("a", Policy)
    policies.register("b", Policy)
    with pytest.raises(ComponentContractError, match=r"registry is full \(2 components\)"):
        policies.register("c", Policy)
    policies.register("a", OtherPolicy, replace=True)  # replacing is not growing
    assert policies.names() == ["a", "b"]


def test_a_frozen_registry_rejects_global_writes_and_keeps_its_content(policies):
    policies.register("p", Policy)
    policies.freeze()
    assert policies.frozen
    with pytest.raises(ComponentLocked, match="frozen"):
        policies.register("q", Policy)
    with pytest.raises(ComponentLocked, match="frozen"):
        policies.unregister("p")
    assert policies.names() == ["p"]
    policies.unfreeze()
    policies.register("q", Policy)
    assert policies.names() == ["p", "q"]


def test_unregister_removes_from_registry_and_mirror(policies):
    mirror: dict = {}
    policies.attach_mirror(mirror)
    policies.register("p", Policy)
    removed = policies.unregister("p")
    assert removed.name == "p" and mirror == {} and not policies.has("p")
    with pytest.raises(ComponentMissing):
        policies.unregister("p")


def test_the_mirror_equals_name_to_factory_at_all_times(policies):
    mirror: dict = {}
    policies.attach_mirror(mirror)
    steps = [
        lambda: policies.register("a", Policy),
        lambda: policies.register("b", OtherPolicy),
        lambda: policies.register("a", OtherPolicy, replace=True),
        lambda: policies.unregister("b"),
        lambda: policies.register("c", Policy),
    ]
    for step in steps:
        step()
        assert mirror == {c.name: c.factory for c in policies.components()}


def test_attaching_a_mirror_syncs_existing_entries(policies):
    policies.register("pre", Policy)
    mirror = {"stale": 1}
    policies.attach_mirror(mirror)
    assert mirror == {"pre": Policy}


def test_the_generation_counter_moves_on_every_global_change(hub, policies):
    start = hub.generation
    policies.register("p", Policy)
    policies.register("p", OtherPolicy, replace=True)
    policies.unregister("p")
    assert hub.generation == start + 3
    with hub.scoped() as overlay:
        policies.register("q", Policy, scope="run")
    assert hub.generation == start + 3  # an overlay write is not a global change


# -- build ---------------------------------------------------------------------
def test_build_calls_the_factory_with_the_params(policies):
    policies.register("p", Policy)
    assert policies.build("p", {"rate": 0.25}).rate == 0.25
    assert policies.build("p").rate == 1.0


def test_an_unknown_param_is_e004_listing_the_accepted_keys(policies):
    policies.register("p", Policy)
    with pytest.raises(ComponentParamError) as caught:
        policies.build("p", {"rat": 1})
    message = str(caught.value)
    assert caught.value.code == "E004"
    assert "does not accept 'rat'" in message and "['rate']" in message


def test_a_missing_required_param_is_named(hub):
    class Needs:
        def __init__(self, *, must, opt=1):
            pass

        def train_step(self, node, batch):
            return None

    registry = hub.registry("trainer_policy")
    registry.register("needs", Needs)
    with pytest.raises(ComponentParamError, match="missing required parameter 'must'"):
        registry.build("needs", {})
    assert registry.build("needs", {"must": 1}) is not None


def test_a_var_keyword_factory_accepts_any_key_unless_it_has_a_schema(hub):
    class Open:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def train_step(self, node, batch):
            return None

    registry = hub.registry("trainer_policy")
    registry.register("open", Open)
    assert registry.build("open", {"anything": 1}).kwargs == {"anything": 1}
    registry.register(
        "typed", Open, params_schema={"alpha": ParamSpec(type=float, minimum=0.0, required=True)}
    )
    with pytest.raises(ComponentParamError) as caught:
        registry.build("typed", {"anything": 1})
    assert "does not accept 'anything'" in str(caught.value)
    assert "missing required parameter 'alpha'" in str(caught.value)


def test_a_params_schema_validates_before_the_constructor_runs(hub):
    ran = []

    class Typed:
        def __init__(self, *, alpha=1.0):
            ran.append(alpha)

        def train_step(self, node, batch):
            return None

    registry = hub.registry("trainer_policy")
    registry.register("t", Typed, params_schema={"alpha": ParamSpec(type=float, minimum=0.0)})
    with pytest.raises(ComponentParamError, match="alpha: -1.0 is below the minimum 0.0"):
        registry.build("t", {"alpha": -1.0})
    with pytest.raises(ComponentParamError, match="alpha: expected float, got str"):
        registry.build("t", {"alpha": "x"})
    assert ran == []
    registry.build("t", {"alpha": 2})
    assert ran == [2]
    assert registry.get("t").accepted_params() == {"alpha"}


def test_injection_is_an_offer_not_a_demand(hub):
    class Wants:
        def __init__(self, *, config=None, scale=1):
            self.config, self.scale = config, scale

        def train_step(self, node, batch):
            return None

    class Declines:
        def __init__(self, *, scale=1):
            self.scale = scale

        def train_step(self, node, batch):
            return None

    registry = hub.registry("trainer_policy")
    registry.register("wants", Wants)
    registry.register("declines", Declines)
    offer = {"config": {"k": 1}, "unrelated": 5}
    assert registry.build("wants", {"scale": 2}, inject=offer).config == {"k": 1}
    assert registry.build("declines", {"scale": 2}, inject=offer).scale == 2  # no TypeError


def test_a_spec_may_not_set_an_injected_key_e011(hub):
    class Wants:
        def __init__(self, *, config=None):
            pass

        def train_step(self, node, batch):
            return None

    registry = hub.registry("trainer_policy")
    registry.register("wants", Wants)
    with pytest.raises(ComponentSpecError) as caught:
        registry.build("wants", {"config": {}}, inject={"config": {"k": 1}})
    assert caught.value.code == "E011" and "supplied by the framework" in str(caught.value)


def test_a_required_injected_key_is_not_a_missing_param(hub):
    class NeedsConfig:
        def __init__(self, *, config):
            self.config = config

        def train_step(self, node, batch):
            return None

    registry = hub.registry("trainer_policy")
    registry.register("n", NeedsConfig)
    assert registry.build("n", {}, inject={"config": 1}).config == 1
    with pytest.raises(ComponentParamError, match="missing required parameter 'config'"):
        registry.build("n", {})


def test_type_and_value_errors_from_a_factory_are_e008_with_the_cause_chained(hub):
    class Raises:
        def __init__(self, *, n=1):
            raise ValueError("n must be even")

        def train_step(self, node, batch):
            return None

    registry = hub.registry("trainer_policy")
    registry.register("r", Raises)
    with pytest.raises(ComponentBuildError) as caught:
        registry.build("r", {"n": 3})
    assert caught.value.code == "E008" and isinstance(caught.value.cause, ValueError)
    assert caught.value.__cause__ is caught.value.cause
    assert "n must be even" in str(caught.value)


def test_a_validation_error_from_the_component_itself_propagates_unchanged(hub):
    class Strict:
        def __init__(self, *, n=1):
            raise ValidationError("my own reason")

        def train_step(self, node, batch):
            return None

    registry = hub.registry("trainer_policy")
    registry.register("s", Strict)
    with pytest.raises(ValidationError, match="^my own reason$") as caught:
        registry.build("s")
    assert type(caught.value) is ValidationError


def test_the_instance_contract_is_checked_after_construction(hub):
    class Liar:
        def __init__(self):
            pass

        def train_step(self, node, batch):
            return None

        def __getattribute__(self, name):
            if name == "train_step":
                raise AttributeError(name)
            return object.__getattribute__(self, name)

    registry = hub.registry("trainer_policy")
    registry.register("liar", Liar)
    with pytest.raises(ComponentContractError, match="no train_step"):
        registry.build("liar")


def test_a_custom_builder_replaces_direct_construction(hub):
    registry = hub.registry("trainer_policy")
    seen = {}

    def builder(component, values, injected):
        seen.update(component=component.name, values=dict(values), injected=dict(injected))
        return Policy(**values)

    object.__setattr__(registry.spec, "builder", builder)
    try:
        registry.register("p", Policy)
        assert registry.build("p", {"rate": 3.0}).rate == 3.0
        assert seen == {"component": "p", "values": {"rate": 3.0}, "injected": {}}
    finally:
        object.__setattr__(registry.spec, "builder", None)


def test_schedulers_register_as_functions_and_build_with_the_optimizer_injected(hub):
    registry = hub.registry("scheduler")
    registry.register("halve", make_scheduler)
    built = registry.build("halve", {"gamma": 0.1}, inject={"optimizer": object()})
    assert callable(built.step)
    with pytest.raises(ComponentContractError, match="'optimizer' argument"):
        registry.register("bad", lambda *, gamma=1: None)


# -- fingerprints and descriptions --------------------------------------------------
def test_a_builtin_has_no_source_digest_and_other_origins_do(hub):
    registry = hub.registry("trainer_policy")
    builtin = registry.register_builtin("shipped", Policy)
    custom = registry.register("custom", Policy)
    assert builtin.fingerprint() == {
        "kind": "trainer_policy", "name": "shipped", "origin": "builtin", "version": "0",
    }
    fingerprint = custom.fingerprint(spec_hash="abc")
    assert fingerprint["origin"] == "project" and fingerprint["spec_hash"] == "abc"
    assert len(fingerprint["source_digest"]) == 16
    assert "source_digest" not in custom.fingerprint(with_source=False)
    assert "spec_hash" not in custom.fingerprint()


def test_describe_is_json_ready(policies):
    import json

    component = policies.register(
        "p", Policy, params_schema={"rate": ParamSpec(type=float, default=1.0, doc="r")},
        capabilities={"hot": True},
    )
    described = component.describe()
    json.dumps(described)
    assert described["kwargs"] == ["rate"] and described["params"]["rate"]["type"] == "float"
    assert described["capabilities"] == {"hot": True} and described["mutability"] == "static"
    assert policies.capabilities() == [described]


def test_w001_is_logged_when_a_builtin_is_replaced(hub, policy_for, caplog):
    hub.configure(policy_for(allow_builtin_override=True))
    registry = hub.registry("trainer_policy")
    registry.register_builtin("shipped", Policy)
    with caplog.at_level(logging.WARNING, logger="mactf.components"):
        replaced = registry.register("shipped", OtherPolicy, replace=True)
    assert replaced.overrides_builtin and replaced.fingerprint()["overrides_builtin"] is True
    assert any(r.getMessage().startswith("W001 trainer_policy:shipped") for r in caplog.records)
