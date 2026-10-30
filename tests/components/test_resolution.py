"""The resolution order (document 17 section 5.3): nine layers, lowest to highest."""

from __future__ import annotations

import pytest

from component_helpers import Policy
from mactf.components import LAYERS, ComponentSpec, merge_layers, spec_hash
from mactf.exceptions import ComponentSpecError

KIND = "weight_update"


def test_the_nine_layers_in_order():
    assert LAYERS == ("default", "settings", "conf", "config_files", "env", "config", "set", "api", "code")
    assert len(LAYERS) == 9


def test_resolution_order():
    """The same key set at every layer resolves to the latest one, and describe() says which."""
    spec_layers = [name for name in LAYERS if name != "code"]
    for upto in range(1, len(spec_layers) + 1):
        layers = {name: {"type": "trimmed_mean", "params": {"trim": index / 10}}
                  for index, name in enumerate(spec_layers[:upto], start=1)}
        result = merge_layers(KIND, layers)
        winner = spec_layers[upto - 1]
        assert result.spec.params["trim"] == upto / 10
        assert result.decided_by == winner
        assert result.param_layers == {"trim": winner}
        assert result.describe()["decided_by"] == winner
        assert result.describe()["params"] == {"trim": winner}
        assert [entry["layer"] for entry in result.describe()["layers"]] == spec_layers[:upto]


def test_the_five_layers_of_the_section_5_3_example():
    """yaml (conf), environment, --set, the API body and code, each setting the same key."""
    result = merge_layers(
        KIND,
        {
            "conf": {"type": "trimmed_mean", "params": {"trim": 0.1}},
            "env": {"type": "trimmed_mean", "params": {"trim": 0.15}},
            "set": {"type": "trimmed_mean", "params": {"trim": 0.2}},
            "api": {"params": {"trim": 0.3}},  # a patch: names no component
        },
    )
    assert result.spec.params["trim"] == 0.3 and result.param_layers["trim"] == "api"
    assert result.decided_by == "set"  # the API patch changed a parameter, not the component
    # code beats everything and is adopted without a build
    marker = object()
    coded = merge_layers(KIND, {"set": "uniform", "api": {"params": {"x": 1}}, "code": marker})
    assert coded.adopted and coded.instance is marker and coded.decided_by == "code"
    assert coded.describe()["adopted"] is True


def test_the_worked_example_of_section_5_5():
    result = merge_layers(
        KIND,
        {
            "conf": {"type": "trimmed_mean", "params": {"trim": 0.1}},
            "set": {"params": {"trim": 0.2}},
        },
    )
    assert result.spec == ComponentSpec(KIND, type="trimmed_mean", params={"trim": 0.2})
    assert result.spec.hash() == spec_hash({"type": "trimmed_mean", "params": {"trim": 0.2}})
    assert result.decided_by == "conf" and result.param_layers == {"trim": "set"}


def test_the_override_chain_of_section_15_9_3():
    result = merge_layers(
        KIND,
        {
            "default": "kappa_softmax",
            "conf": {"type": "trimmed_mean", "params": {"trim": 0.1}},
            "set": {"params": {"trim": 0.3}},
            "api": {"params": {"trim": 0.3, "temperature": 2.0}},
        },
    )
    assert result.spec.canonical() == {"type": "trimmed_mean", "params": {"trim": 0.3, "temperature": 2.0}}
    assert result.spec.hash() == spec_hash({"type": "trimmed_mean", "params": {"trim": 0.3, "temperature": 2.0}})
    assert result.param_layers == {"trim": "api", "temperature": "api"}


@pytest.mark.parametrize(
    "layers, expected_type, expected_params, decided_by",
    [
        # a higher layer naming another component replaces the lower layers wholesale
        ({"conf": {"type": "a", "params": {"x": 1}}, "set": "b"}, "b", {}, "set"),
        ({"conf": {"type": "a", "params": {"x": 1}}, "set": {"type": "b", "y": 2}}, "b", {"y": 2}, "set"),
        # the same name merges
        ({"conf": {"type": "a", "params": {"x": 1}}, "set": "a"}, "a", {"x": 1}, "set"),
        ({"conf": {"type": "a", "params": {"x": 1}}, "env": {"params": {"y": 2}}, "set": {"params": {"x": 9}}},
         "a", {"x": 9, "y": 2}, "conf"),
        # an import is a different component from a name
        ({"conf": {"type": "a"}, "set": {"import": "m:C"}}, None, {}, "set"),
        # a lower layer's params do not leak into a replacement, but later patches do apply to it
        ({"conf": {"type": "a", "params": {"x": 1}}, "set": "b", "api": {"params": {"z": 1}}}, "b", {"z": 1}, "set"),
        # empty layers say nothing
        ({"default": None, "conf": "a", "set": None}, "a", {}, "conf"),
    ],
)
def test_layer_merging_table(layers, expected_type, expected_params, decided_by):
    result = merge_layers(KIND, layers)
    assert result.spec.type == expected_type
    assert dict(result.spec.params) == expected_params
    assert result.decided_by == decided_by


def test_a_replacement_forgets_which_layer_set_the_old_params():
    result = merge_layers(KIND, {"conf": {"type": "a", "params": {"x": 1}}, "set": {"type": "b", "params": {"y": 1}}})
    assert result.param_layers == {"y": "set"}


def test_a_sequence_is_taken_in_the_order_given():
    result = merge_layers(KIND, [("late", "b"), ("early", "a")])
    assert result.spec.type == "a" and result.decided_by == "early"  # the last entry wins


def test_the_resolver_sees_a_mapping_in_layer_order_not_insertion_order():
    one = merge_layers(KIND, {"set": "b", "conf": "a"})
    two = merge_layers(KIND, {"conf": "a", "set": "b"})
    assert one.spec == two.spec == ComponentSpec(KIND, type="b")


def test_an_unknown_layer_name_is_refused():
    with pytest.raises(ComponentSpecError, match="unknown resolution layer"):
        merge_layers(KIND, {"cnof": "a"})


def test_an_instance_is_only_allowed_from_the_code_layer():
    with pytest.raises(ComponentSpecError, match="only the 'code' layer may supply an instance"):
        merge_layers(KIND, {"set": Policy()})
    assert merge_layers(KIND, {"code": Policy()}).adopted


@pytest.mark.parametrize(
    "layers",
    [
        {},
        {"default": None},
        {"set": {"params": {"a": 1}}},
    ],
)
def test_nothing_to_resolve_or_only_a_patch_is_an_error_or_empty(layers):
    if not any(v is not None for v in layers.values()):
        result = merge_layers(KIND, layers)
        assert result.spec is None and result.decided_by == "" and not result.adopted
    else:
        with pytest.raises(ComponentSpecError):
            merge_layers(KIND, layers)


def test_a_bad_layer_value_names_the_problem():
    with pytest.raises(ComponentSpecError, match="E012"):
        merge_layers(KIND, {"conf": {"type": "a", "params": {"x": 1}, "y": 2}})
    with pytest.raises(ComponentSpecError, match="'import' must look like"):
        merge_layers(KIND, {"conf": {"import": "nope"}})


def test_merge_layers_does_not_mutate_its_input():
    conf = {"type": "a", "params": {"x": 1}}
    api = {"params": {"x": 2}}
    merge_layers(KIND, {"conf": conf, "api": api})
    assert conf == {"type": "a", "params": {"x": 1}} and api == {"params": {"x": 2}}
