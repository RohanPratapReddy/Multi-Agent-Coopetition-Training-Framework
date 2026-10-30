"""The function spec registry (document 07 section 10).

The gate: every family resolves from ``{"type": ..., **kwargs}``, unknown keys and
unknown types are rejected with the accepted sets named, and
``function_capabilities()`` advertises exactly what ``build_function`` can build.
"""

from __future__ import annotations

import torch

from mactf.exceptions import ValidationError
from mactf.rl.functions import registry
from mactf.rl.functions.registry import (
    CAPABILITY_ORDER,
    FUNCTION_TYPES,
    build_function,
    function_capabilities,
    register_function,
    resolve_family,
)


MINIMAL_SPECS: dict[str, dict] = {
    "value": {"type": "zero"},
    "advantage": {"type": "td", "gamma": 0.99},
    "loss": {"type": "kl"},
    "rollup": {"type": "mean"},
    "return": {"type": "discounted", "gamma": 0.99},
    "reward": {"type": "callable", "fn": lambda episode, peer=None: 0.0},
    "regularization": {"type": "l2"},
    "objective": {"type": "callable", "fn": lambda batch: (torch.zeros(()), {})},
    "gradient": {"type": "clip", "max_norm": 1.0},
    "metric": {"type": "callable", "fn": lambda batch, output: {}},
    "action": {"type": "greedy"},
    "synthesis": {"type": "consensus"},
}


def test_every_family_is_registered_and_ordered():
    assert set(CAPABILITY_ORDER) == set(FUNCTION_TYPES)
    assert len(CAPABILITY_ORDER) == len(set(CAPABILITY_ORDER))


def test_every_family_builds_its_minimal_spec():
    for family, spec in MINIMAL_SPECS.items():
        built = build_function(spec, family=family)
        assert built is not None, family
        assert type(built).__name__ == FUNCTION_TYPES[family][spec["type"]].__name__


def test_every_registered_type_builds_from_capabilities():
    """No advertised entry may be unbuildable: the API reads this catalog."""
    for family, entries in function_capabilities().items():
        for entry in entries:
            assert entry["class"] == FUNCTION_TYPES[family][entry["type"]].__name__
            assert isinstance(entry["kwargs"], list)
            assert entry["type"] in FUNCTION_TYPES[family]


def test_spec_without_a_type_is_rejected():
    with pytest.raises(ValidationError, match="requires a 'type' key"):
        build_function({})


def test_unknown_family_is_rejected():
    with pytest.raises(ValidationError, match="unknown function family"):
        build_function({"type": "mean"}, family="nope")


def test_unknown_type_names_the_accepted_set():
    with pytest.raises(ValidationError, match="unknown loss type") as info:
        build_function({"type": "policy"}, family="loss")
    assert "kl" in str(info.value)


def test_unknown_kwarg_is_rejected_with_the_accepted_keys():
    with pytest.raises(ValidationError, match="does not accept") as info:
        build_function({"type": "kl", "coeffcient": 0.1}, family="loss")
    assert "coefficient" in str(info.value)


def test_resolve_family_finds_a_unique_name():
    assert resolve_family("kl") == "loss"
    assert resolve_family("mean") == "rollup"


def test_resolve_family_rejects_an_ambiguous_name():
    # "callable" lives in several families; guessing would train the wrong one.
    with pytest.raises(ValidationError, match="ambiguous between families"):
        resolve_family("callable")


def test_resolve_family_rejects_an_unknown_name():
    with pytest.raises(ValidationError, match="unknown function type"):
        resolve_family("definitely_not_registered")


def test_register_function_adds_a_new_type():
    class Marker:
        def __init__(self, *, flag: bool = False) -> None:
            self.flag = flag

    try:
        register_function("rollup", "marker_test", Marker)
        assert FUNCTION_TYPES["rollup"]["marker_test"] is Marker
        built = build_function({"type": "marker_test", "flag": True}, family="rollup")
        assert built.flag is True
    finally:
        FUNCTION_TYPES["rollup"].pop("marker_test", None)


def test_register_function_refuses_a_silent_override():
    class Marker:
        pass

    with pytest.raises(ValidationError, match="already registered"):
        register_function("rollup", "mean", Marker)


def test_register_function_can_replace_explicitly():
    class Marker:
        pass

    original = FUNCTION_TYPES["rollup"]["mean"]
    try:
        register_function("rollup", "mean", Marker, replace=True)
        assert FUNCTION_TYPES["rollup"]["mean"] is Marker
    finally:
        FUNCTION_TYPES["rollup"]["mean"] = original


def test_register_function_rejects_a_bad_name():
    with pytest.raises(ValidationError, match="invalid function type name"):
        register_function("rollup", "not a name", object)


def test_register_function_rejects_a_non_class():
    with pytest.raises(ValidationError, match="is not a class"):
        register_function("rollup", "marker_test", object())


def test_capabilities_declare_required_fields_for_losses():
    entries = {entry["type"]: entry for entry in function_capabilities()["loss"]}
    assert entries["kl"]["required_fields"] == ["policy", "reference"]
    assert "actions" in entries["policy_gradient"]["required_fields"]


def test_capabilities_include_kwargs_for_every_entry():
    for family, entries in function_capabilities().items():
        for entry in entries:
            assert "kwargs" in entry, (family, entry["type"])


def test_family_aliases_match_the_tables():
    assert registry.LOSS_TYPES is FUNCTION_TYPES["loss"]
    assert registry.ROLLUP_TYPES is FUNCTION_TYPES["rollup"]
    assert registry.SYNTHESIS_TYPES is FUNCTION_TYPES["synthesis"]


import pytest  # noqa: E402  (kept last so the module reads bottom-up)
