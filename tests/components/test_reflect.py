"""Constructor reflection, parameter specs, canonical JSON and spec hashing (section 3.3)."""

from __future__ import annotations

import enum
import functools
import hashlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mactf.components import ParamSpec, canonical_json, constructor_keys, reflect, source_digest, spec_hash
from mactf.components.params import MISSING


# -- reflection -------------------------------------------------------------------
class Plain:
    def __init__(self, a, b=1, *, c=2):
        pass


class Forwarding:
    def __init__(self, x=0, **kwargs):
        super().__init__()


class Base:
    def __init__(self, *, base_key=1, hidden=2):
        pass


class Child(Base):
    def __init__(self, *, child_key=1, **kwargs):
        super().__init__(**kwargs)


class Opaque:
    """Forwards **kwargs and nobody takes them."""

    def __init__(self, **kwargs):
        pass


class Inherited(Plain):
    """No __init__ of its own: the base's signature is the class's."""


def test_reflect_reads_a_plain_class():
    found = reflect(Plain)
    assert found.keys == {"a", "b", "c"} and found.required == {"a"} and not found.var_keyword


def test_reflect_follows_the_mro_only_while_kwargs_forward():
    found = reflect(Child)
    assert found.keys == {"child_key", "base_key", "hidden"}
    assert not found.var_keyword  # the chain ends at Base, which names its keys
    assert reflect(Forwarding).keys == {"x"}
    assert reflect(Forwarding).var_keyword


def test_reflect_a_class_that_forwards_to_object_accepts_anything():
    assert reflect(Opaque).var_keyword and reflect(Opaque).keys == frozenset()


def test_reflect_inherits_a_missing_init():
    assert reflect(Inherited).keys == {"a", "b", "c"}


def test_reflect_a_function_and_its_var_keyword():
    def fn(optimizer, *, gamma=0.5, **extra):
        pass

    found = reflect(fn)
    assert found.keys == {"optimizer", "gamma"} and found.required == {"optimizer"}
    assert found.var_keyword


def test_reflect_a_partial_removes_the_bound_keywords():
    found = reflect(functools.partial(Plain, c=9))
    assert found.keys == {"a", "b"} and found.required == {"a"}


def test_reflect_positional_only_and_star_args_are_not_keywords():
    def fn(p, /, q, *args, k=1):
        pass

    assert reflect(fn).keys == {"q", "k"}


def test_reflect_a_builtin_without_a_signature_is_opaque():
    assert reflect(print).var_keyword or reflect(print).keys is not None  # never raises
    assert constructor_keys(Plain) == frozenset({"a", "b", "c"})


# -- ParamSpec --------------------------------------------------------------------
@pytest.mark.parametrize(
    "spec, value, fragment",
    [
        (ParamSpec(type=float), 0.5, None),
        (ParamSpec(type=float), 1, None),  # an int is a valid float (YAML writes lr: 1)
        (ParamSpec(type=int), 1.5, "expected int, got float"),
        (ParamSpec(type=int), True, "expected a number, got bool"),
        (ParamSpec(type=float), True, "expected a number, got bool"),
        (ParamSpec(type=str), 3, "expected str, got int"),
        (ParamSpec(type=(int, float)), "x", "expected int/float, got str"),
        (ParamSpec(type=bool), True, None),
        (ParamSpec(choices=("a", "b")), "c", "is not one of ['a', 'b']"),
        (ParamSpec(choices=("a", "b")), "a", None),
        (ParamSpec(type=float, minimum=0.0), -1.0, "below the minimum 0.0"),
        (ParamSpec(type=float, maximum=1.0), 2.0, "above the maximum 1.0"),
        (ParamSpec(type=float, minimum=0.0, maximum=1.0), 1.0, None),
        (ParamSpec(), object(), None),
    ],
)
def test_param_spec_problems(spec, value, fragment):
    problems = spec.problems("p", value)
    if fragment is None:
        assert problems == []
    else:
        assert len(problems) == 1 and fragment in problems[0] and problems[0].startswith("p:")


def test_param_spec_describe_lists_only_what_is_declared():
    assert ParamSpec(type=float, default=0.1, minimum=0.0, doc="rate").describe() == {
        "required": False, "doc": "rate", "type": "float", "default": 0.1, "minimum": 0.0,
    }
    assert ParamSpec(required=True, choices=(1, 2)).describe()["choices"] == [1, 2]
    assert "default" not in ParamSpec().describe()
    assert ParamSpec().default is MISSING


# -- canonical JSON and hashes ---------------------------------------------------------
class Color(enum.Enum):
    RED = "red"


def test_canonical_json_sorts_keys_and_has_no_whitespace():
    assert canonical_json({"b": 1, "a": [1, 2], "c": {"z": 1, "y": 2}}) == (
        '{"a":[1,2],"b":1,"c":{"y":2,"z":1}}'
    )


def test_canonical_json_handles_sets_enums_and_unknown_objects_deterministically():
    assert canonical_json({"s": {3, 1, 2}}) == '{"s":[1,2,3]}'
    assert canonical_json({"s": frozenset({"b", "a"})}) == '{"s":["a","b"]}'
    assert canonical_json({"e": Color.RED}) == '{"e":"red"}'
    assert canonical_json({"o": Plain}) == '{"o":"' + repr(Plain) + '"}'


def test_spec_hash_is_sixteen_hex_and_pinned():
    digest = spec_hash({"type": "trimmed_mean", "params": {"trim": 0.2}})
    assert digest == hashlib.sha256(
        b'{"params":{"trim":0.2},"type":"trimmed_mean"}'
    ).hexdigest()[:16]
    assert len(digest) == 16 and int(digest, 16) >= 0


def test_float_repr_is_part_of_the_hash():
    assert spec_hash({"params": {"x": 0.1 + 0.2}}) != spec_hash({"params": {"x": 0.3}})
    assert canonical_json({"x": 0.1 + 0.2}) == '{"x":0.30000000000000004}'


def test_source_digest_is_stable_and_none_without_source():
    assert source_digest(Plain) == source_digest(Plain)
    assert len(source_digest(Plain)) == 16
    assert source_digest(Plain) != source_digest(Forwarding)
    assert source_digest(len) is None
    assert source_digest(functools.partial(Plain, c=1)) == source_digest(Plain)


json_values = st.recursive(
    st.none() | st.booleans() | st.integers(-10**6, 10**6) | st.text(max_size=8)
    | st.floats(allow_nan=False, allow_infinity=False),
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(st.text(min_size=1, max_size=5), children, max_size=4),
    max_leaves=12,
)


@settings(max_examples=100, deadline=None)
@given(st.dictionaries(st.text(min_size=1, max_size=5), json_values, max_size=5))
def test_spec_hash_ignores_key_order_and_is_deterministic(mapping):
    shuffled = dict(reversed(list(mapping.items())))
    assert spec_hash(mapping) == spec_hash(shuffled) == spec_hash(dict(mapping))
    assert canonical_json(mapping) == canonical_json(shuffled)
