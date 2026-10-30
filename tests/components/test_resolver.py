"""``ComponentSpec`` and ``ComponentResolver`` (document 17 section 5.1, 5.2, 5.4)."""

from __future__ import annotations

import sys
import types

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from component_helpers import OtherPolicy, Policy
from mactf.components import (
    HUB,
    ComponentResolver,
    ComponentSpec,
    HubPolicy,
    Mutability,
    Origin,
    spec_hash,
)
from mactf.exceptions import (
    ComponentBuildError,
    ComponentCollision,
    ComponentContractError,
    ComponentImportDenied,
    ComponentMissing,
    ComponentParamError,
    ComponentSpecError,
)

KIND = "trainer_policy"


# -- parsing ------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("kappa_softmax", ComponentSpec(KIND, type="kappa_softmax")),
        ({"type": "x"}, ComponentSpec(KIND, type="x")),
        ({"type": "x", "params": {"a": 1}}, ComponentSpec(KIND, type="x", params={"a": 1})),
        ({"type": "x", "a": 1, "b": 2}, ComponentSpec(KIND, type="x", params={"a": 1, "b": 2})),  # flat RL form
        ({"type": "x", "params": {}}, ComponentSpec(KIND, type="x")),
        ({"import": "pkg.mod:Cls"}, ComponentSpec(KIND, import_path="pkg.mod:Cls")),
        ({"import": "pkg.mod:Cls", "type": "name", "params": {"k": 1}},
         ComponentSpec(KIND, type="name", import_path="pkg.mod:Cls", params={"k": 1})),
        ({"import": "pkg.mod:Outer.Inner"}, ComponentSpec(KIND, import_path="pkg.mod:Outer.Inner")),
        ({"type": 7}, ComponentSpec(KIND, type="7")),
    ],
)
def test_parse_accepts(raw, expected):
    assert ComponentSpec.parse(KIND, raw) == expected


@pytest.mark.parametrize(
    "raw, code, fragment",
    [
        ("", "E005", "an empty component name"),
        (3, "E005", "a name or a mapping, got int"),
        (["x"], "E005", "a name or a mapping, got list"),
        (None, "E005", "a name or a mapping, got NoneType"),
        ({}, "E005", "needs 'type' or 'import'"),
        ({"params": {"a": 1}}, "E005", "needs 'type' or 'import'"),
        ({"type": "x", "params": {"a": 1}, "b": 2}, "E012", "not both; flat keys: ['b']"),
        ({"type": "x", "params": [1]}, "E005", "'params' must be a mapping"),
        ({"import": "not a path"}, "E005", "must look like 'pkg.mod:Name'"),
        ({"import": "pkg.mod"}, "E005", "must look like 'pkg.mod:Name'"),
        ({"import": "pkg.mod:"}, "E005", "must look like 'pkg.mod:Name'"),
        ({"import": "1pkg:Cls"}, "E005", "must look like 'pkg.mod:Name'"),
        ({"import": "pkg:Cls; rm -rf /"}, "E005", "must look like 'pkg.mod:Name'"),
    ],
)
def test_parse_rejects(raw, code, fragment):
    with pytest.raises(ComponentSpecError) as caught:
        ComponentSpec.parse(KIND, raw)
    assert caught.value.code == code and fragment in str(caught.value)


def test_parse_passes_a_spec_through_and_checks_its_kind():
    spec = ComponentSpec(KIND, type="x")
    assert ComponentSpec.parse(KIND, spec) is spec
    with pytest.raises(ComponentSpecError, match="a scorer spec was given for a trainer_policy slot"):
        ComponentSpec.parse(KIND, ComponentSpec("scorer", type="x"))


def test_parse_allows_a_patch_only_when_asked():
    assert ComponentSpec.parse(KIND, {"params": {"a": 1}}, allow_patch=True).is_patch
    assert ComponentSpec.parse(KIND, {"a": 1}, allow_patch=True).params == {"a": 1}
    assert not ComponentSpec.parse(KIND, "x", allow_patch=True).is_patch


def test_parse_copies_the_params():
    original = {"a": 1}
    spec = ComponentSpec.parse(KIND, {"type": "x", "params": original})
    original["a"] = 2
    assert spec.params == {"a": 1}


def test_canonical_form_and_hash():
    spec = ComponentSpec.parse(KIND, {"type": "trimmed_mean", "params": {"trim": 0.2}})
    assert spec.canonical() == {"type": "trimmed_mean", "params": {"trim": 0.2}}
    # the worked example of section 5.5
    assert spec.hash() == spec_hash({"type": "trimmed_mean", "params": {"trim": 0.2}})
    assert ComponentSpec.parse(KIND, {"params": {"b": 1, "a": 2}, "type": "x"}).hash() == \
        ComponentSpec.parse(KIND, {"type": "x", "a": 2, "b": 1}).hash()
    assert ComponentSpec.parse(KIND, "x").canonical() == {"type": "x", "params": {}}
    assert ComponentSpec.parse(KIND, {"import": "a.b:C"}).canonical() == {"import": "a.b:C", "params": {}}
    assert ComponentSpec.parse(KIND, "x").hash() != ComponentSpec.parse(KIND, "y").hash()
    assert ComponentSpec.parse(KIND, {"type": "x", "p": 1}).hash() != ComponentSpec.parse(KIND, "x").hash()


# -- merged_over --------------------------------------------------------------------------
BASE = ComponentSpec(KIND, type="trimmed_mean", params={"trim": 0.1, "temperature": 1.0})


@pytest.mark.parametrize(
    "upper, expected_type, expected_params",
    [
        # same type: params merge key by key, the higher layer wins on a clash
        (ComponentSpec(KIND, type="trimmed_mean", params={"trim": 0.2}), "trimmed_mean", {"trim": 0.2, "temperature": 1.0}),
        # a different type replaces wholesale (the lower params belong to another component)
        (ComponentSpec(KIND, type="uniform", params={"x": 1}), "uniform", {"x": 1}),
        (ComponentSpec(KIND, type="uniform"), "uniform", {}),
        # a patch applies to the lower layer's component
        (ComponentSpec(KIND, params={"trim": 0.3}), "trimmed_mean", {"trim": 0.3, "temperature": 1.0}),
        (ComponentSpec(KIND), "trimmed_mean", {"trim": 0.1, "temperature": 1.0}),
        # the same name via a different route is a different component
        (ComponentSpec(KIND, type="trimmed_mean", import_path="a:B", params={"z": 1}), "trimmed_mean", {"z": 1}),
    ],
)
def test_merged_over(upper, expected_type, expected_params):
    merged = upper.merged_over(BASE)
    assert merged.type == expected_type and dict(merged.params) == expected_params


def test_merged_over_nothing_is_identity_and_never_mutates_either_side():
    spec = ComponentSpec(KIND, type="x", params={"a": 1})
    assert spec.merged_over(None) is spec
    base_params = dict(BASE.params)
    ComponentSpec(KIND, params={"trim": 9}).merged_over(BASE)
    assert dict(BASE.params) == base_params


def test_a_patch_over_an_import_keeps_the_import():
    base = ComponentSpec(KIND, import_path="a.b:C", params={"k": 1})
    merged = ComponentSpec(KIND, params={"k": 2, "j": 3}).merged_over(base)
    assert merged.import_path == "a.b:C" and dict(merged.params) == {"k": 2, "j": 3}


params_strategy = st.dictionaries(st.sampled_from("abcde"), st.integers(), max_size=4)


@settings(max_examples=100, deadline=None)
@given(params_strategy, params_strategy, params_strategy)
def test_merging_same_type_layers_is_associative(a, b, c):
    one, two, three = (ComponentSpec(KIND, type="t", params=p) for p in (a, b, c))
    assert three.merged_over(two.merged_over(one)) == three.merged_over(two).merged_over(one)
    assert dict(three.merged_over(two.merged_over(one)).params) == {**a, **b, **c}


# -- resolve / validate / adopt -------------------------------------------------------------
@pytest.fixture
def resolver(component_overlay):
    HUB.registry(KIND).register("rv_p", Policy, scope="run")
    return ComponentResolver()


def test_resolve_builds_through_the_registry(resolver):
    resolved = resolver.resolve(ComponentSpec.parse(KIND, {"type": "rv_p", "rate": 0.5}), trusted=True)
    assert isinstance(resolved.instance, Policy) and resolved.instance.rate == 0.5
    assert resolved.component.name == "rv_p" and resolved.component.origin is Origin.PROJECT
    assert resolved.spec_hash == resolved.spec.hash()
    assert resolved.fingerprint()["spec_hash"] == resolved.spec_hash
    assert resolved.fingerprint()["origin"] == "project"


def test_validate_checks_the_name_and_params_without_building(resolver):
    assert resolver.validate(ComponentSpec.parse(KIND, "rv_p"), trusted=True).name == "rv_p"
    with pytest.raises(ComponentMissing):
        resolver.validate(ComponentSpec.parse(KIND, "rv_nope"), trusted=True)
    with pytest.raises(ComponentParamError, match="does not accept 'bad'"):
        resolver.validate(ComponentSpec.parse(KIND, {"type": "rv_p", "bad": 1}), trusted=True)


def test_validate_treats_injected_keys_as_offered_not_missing(component_overlay):
    class NeedsConfig:
        def __init__(self, *, config, rate=1.0):
            pass

        def train_step(self, node, batch):  # pragma: no cover
            pass

    HUB.registry(KIND).register("rv_cfg", NeedsConfig, scope="run")
    resolver = ComponentResolver()
    spec = ComponentSpec.parse(KIND, {"type": "rv_cfg", "rate": 2.0})
    with pytest.raises(ComponentParamError, match="missing required parameter 'config'"):
        resolver.validate(spec, trusted=True)
    resolver.validate(spec, trusted=True, inject={"config": {}})
    with pytest.raises(ComponentSpecError) as caught:
        resolver.validate(ComponentSpec.parse(KIND, {"type": "rv_cfg", "config": 1}), trusted=True,
                          inject={"config": {}})
    assert caught.value.code == "E011"


def test_a_patch_cannot_be_resolved(resolver):
    with pytest.raises(ComponentSpecError, match="needs 'type' or 'import'"):
        resolver.resolve(ComponentSpec(KIND, params={"a": 1}), trusted=True)


def test_adopt_records_an_in_code_instance_as_runtime(resolver):
    instance = Policy(rate=4.0)
    adopted = resolver.adopt(KIND, instance)
    assert adopted.instance is instance
    assert adopted.component.origin is Origin.RUNTIME
    assert adopted.component.mutability is Mutability.STATIC  # the kind default is static
    assert adopted.component.name == "instance" and adopted.component.qualname == "Policy"
    assert adopted.spec.type == "instance" and adopted.spec.params == {}
    assert adopted.spec_hash == spec_hash({"adopted": f"{Policy.__module__}.Policy"})
    assert adopted.fingerprint()["source_digest"]  # a runtime component carries a digest
    named = resolver.adopt(KIND, type("Named", (Policy,), {"name": "mine"})())
    assert named.component.name == "mine"
    assert resolver.adopt("reward_term", type("T", (), {"contribution": lambda s: 0})()).component.mutability \
        is Mutability.EPISODE
    with pytest.raises(ComponentContractError, match="no train_step"):
        resolver.adopt(KIND, object())


# -- the import allow-list -------------------------------------------------------------------
@pytest.fixture
def acme(monkeypatch):
    module = types.ModuleType("acme_scoring")

    class TrimmedMean(Policy):
        pass

    class Outer:
        class Inner(Policy):
            pass

    module.TrimmedMean = TrimmedMean
    module.Outer = Outer
    module.not_a_policy = object()
    monkeypatch.setitem(sys.modules, "acme_scoring", module)
    return module


def _spec(path, **extra):
    return ComponentSpec.parse(KIND, {"import": path, **extra})


def test_a_trusted_surface_may_import_any_module(acme, component_overlay):
    resolved = ComponentResolver().resolve(_spec("acme_scoring:TrimmedMean"), trusted=True)
    assert isinstance(resolved.instance, acme.TrimmedMean)
    # the name defaults to the snake-cased attribute and the component is a runtime one
    assert resolved.component.name == "trimmed_mean" and resolved.component.origin is Origin.RUNTIME
    # imports register into the active run overlay, not the process registry
    assert HUB.registry(KIND).has("trimmed_mean")
    assert "trimmed_mean" not in HUB.registry(KIND)._entries


def test_an_untrusted_surface_is_refused_by_default(acme, component_overlay):
    with pytest.raises(ComponentImportDenied) as caught:
        ComponentResolver().resolve(_spec("acme_scoring:TrimmedMean"), trusted=False)
    assert caught.value.code == "E006" and "ALLOW_IMPORT is false" in str(caught.value)
    assert caught.value.target == "acme_scoring:TrimmedMean"
    assert not HUB.registry(KIND).has("trimmed_mean")


@pytest.mark.parametrize(
    "allowlist, module, ok",
    [
        (["acme_scoring"], "acme_scoring", True),
        (["acme_*"], "acme_scoring", True),
        (["acme_scoring*"], "acme_scoring", True),
        (["other"], "acme_scoring", False),
        (["acme"], "acme_scoring", False),  # no prefix semantics: fnmatch on the whole name
        ([], "acme_scoring", False),
        (["acme_scoring.sub"], "acme_scoring", False),
        (["x", "acme_scoring"], "acme_scoring", True),
    ],
)
def test_an_import_outside_the_allow_list_is_refused(acme, component_overlay, allowlist, module, ok):
    policy = HubPolicy(allow_import=True, import_allowlist=tuple(allowlist))
    with HUB.scoped(policy=policy):
        if ok:
            ComponentResolver().resolve(_spec(f"{module}:TrimmedMean"), trusted=False)
        else:
            with pytest.raises(ComponentImportDenied, match="not in COMPONENTS.IMPORT_ALLOWLIST"):
                ComponentResolver().resolve(_spec(f"{module}:TrimmedMean"), trusted=False)


def test_the_allow_list_without_the_switch_still_denies(acme, component_overlay):
    with HUB.scoped(policy=HubPolicy(allow_import=False, import_allowlist=("acme_scoring",))):
        with pytest.raises(ComponentImportDenied, match="ALLOW_IMPORT is false"):
            ComponentResolver().resolve(_spec("acme_scoring:TrimmedMean"), trusted=False)


def test_a_denied_import_never_imports_the_module(monkeypatch, component_overlay):
    imported = []
    import importlib

    real = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", lambda n, *a, **k: imported.append(n) or real(n, *a, **k))
    with pytest.raises(ComponentImportDenied):
        ComponentResolver().resolve(_spec("never_imported_module:X"), trusted=False)
    assert "never_imported_module" not in imported


def test_a_registered_name_needs_no_import_permission(resolver):
    # "An API caller can always use a registered name": the trust decision is the registration
    assert ComponentResolver().resolve(ComponentSpec.parse(KIND, "rv_p"), trusted=False).instance


def test_import_name_comes_from_the_spec_when_given(acme, component_overlay):
    resolved = ComponentResolver().resolve(_spec("acme_scoring:TrimmedMean", type="my_policy"), trusted=True)
    assert resolved.component.name == "my_policy"
    assert HUB.registry(KIND).has("my_policy")


def test_a_dotted_attribute_path_is_followed(acme, component_overlay):
    resolved = ComponentResolver().resolve(_spec("acme_scoring:Outer.Inner"), trusted=True)
    assert isinstance(resolved.instance, acme.Outer.Inner) and resolved.component.name == "inner"


def test_importing_the_same_factory_twice_is_idempotent(acme, component_overlay):
    resolver = ComponentResolver()
    first = resolver.resolve(_spec("acme_scoring:TrimmedMean"), trusted=True).component
    second = resolver.resolve(_spec("acme_scoring:TrimmedMean"), trusted=True).component
    assert first is second


def test_importing_a_different_factory_under_a_taken_name_is_a_collision(acme, component_overlay):
    HUB.registry(KIND).register("trimmed_mean", OtherPolicy, scope="run")
    with pytest.raises(ComponentCollision, match="cannot take the same name"):
        ComponentResolver().resolve(_spec("acme_scoring:TrimmedMean"), trusted=True)


def test_a_missing_attribute_is_a_spec_error(acme, component_overlay):
    with pytest.raises(ComponentSpecError, match="has no attribute 'Nope'"):
        ComponentResolver().resolve(_spec("acme_scoring:Nope"), trusted=True)


def test_an_attribute_that_breaks_the_contract_is_a_contract_error(acme, component_overlay):
    with pytest.raises(ComponentContractError):
        ComponentResolver().resolve(_spec("acme_scoring:not_a_policy"), trusted=True)
    assert not HUB.registry(KIND).has("not_a_policy")


def test_a_module_that_raises_while_importing_is_e008_with_the_cause(monkeypatch, component_overlay):
    import importlib

    def boom(name, *args, **kwargs):
        raise RuntimeError("module exploded")

    monkeypatch.setattr(importlib, "import_module", boom)
    with pytest.raises(ComponentBuildError) as caught:
        ComponentResolver().resolve(_spec("exploding_module:X"), trusted=True)
    assert caught.value.code == "E008" and isinstance(caught.value.cause, RuntimeError)


def test_a_missing_module_is_an_error_not_a_registration(component_overlay):
    with pytest.raises(ComponentBuildError):
        ComponentResolver().resolve(_spec("definitely_not_a_module_zz:X"), trusted=True)


def test_an_import_outside_any_overlay_registers_globally(acme, restore_hub):
    resolved = ComponentResolver().resolve(_spec("acme_scoring:TrimmedMean"), trusted=True)
    assert HUB.registry(KIND)._entries["trimmed_mean"] is resolved.component
