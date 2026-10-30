"""Fuzz of spec parsing, layer merging, canonical JSON and the resolver with garbage input.

The contract under test: whatever a user writes in a conf file, ``--set`` or an API body, the answer
is a spec or a *documented* error (``ComponentError`` / ``ImproperlyConfigured`` family), never a raw
``TypeError``/``KeyError`` and never a hang.
"""

from __future__ import annotations

import json
import math

import pytest
from hypothesis import given, settings as hyp_settings, strategies as st

from mactf.components import HUB, ComponentSpec
from mactf.components.reflect import canonical_json, spec_hash
from mactf.components.resolver import LAYERS, ComponentResolver, merge_layers
from mactf.exceptions import ImproperlyConfigured, MACTFError

KIND = "weight_update"

scalar = st.one_of(st.none(), st.booleans(), st.integers(-5, 5), st.floats(allow_nan=False, allow_infinity=False),
                   st.text(max_size=8))
json_value = st.recursive(scalar, lambda inner: st.one_of(
    st.lists(inner, max_size=3), st.dictionaries(st.text(max_size=5), inner, max_size=3)), max_leaves=8)
# garbage includes things YAML can produce: non-string keys, sets are not YAML but tuples/bytes can arrive from code
garbage = st.one_of(
    json_value,
    st.dictionaries(st.one_of(st.integers(), st.text(max_size=4), st.none(), st.booleans()), scalar, max_size=4),
    st.binary(max_size=4), st.tuples(st.integers()), st.just(object()),
)
names = st.sampled_from(["kappa_softmax", "uniform_average", "nope", "", "Kappa-Softmax"])
spec_like = st.one_of(
    names,
    st.builds(lambda n, p: {"type": n, "params": p}, names, st.dictionaries(st.text(max_size=5), scalar, max_size=3)),
    st.builds(lambda n, p: {"type": n, **p}, names, st.dictionaries(st.text(max_size=5).filter(lambda k: k not in {"type", "import", "params"}), scalar, max_size=3)),
    st.builds(lambda p: {"params": p}, st.dictionaries(st.text(max_size=5), scalar, max_size=3)),
    garbage,
)


@hyp_settings(database=None, max_examples=300, deadline=2000)
@given(raw=garbage)
def test_parse_of_garbage_is_a_spec_or_a_documented_error(raw):
    try:
        spec = ComponentSpec.parse(KIND, raw)
    except MACTFError:
        return
    assert spec.type or spec.import_path


@hyp_settings(database=None, max_examples=300, deadline=2000)
@given(layers=st.lists(st.tuples(st.sampled_from(list(LAYERS)), st.one_of(st.none(), spec_like)), max_size=6))
def test_merge_layers_never_crashes_and_matches_a_reference_model(layers):
    try:
        result = merge_layers(KIND, layers)
    except MACTFError:
        return
    # reference model: the last full spec decides the type; params merge key by key while the type is unchanged
    type_name, params = None, {}
    for name, raw in layers:
        if raw is None or (name == "code" and not isinstance(raw, (str, dict))):
            continue                                              # an in-code instance wins outright; not a spec
        spec = ComponentSpec.parse(KIND, raw, allow_patch=type_name is not None)
        if spec.is_patch:
            params = {**params, **spec.params}
        elif spec.type == type_name and spec.import_path is None:
            params = {**params, **spec.params}
        else:
            type_name, params = spec.type, dict(spec.params)
    if result.spec is not None:
        assert result.spec.type == type_name
        assert dict(result.spec.params) == params


@hyp_settings(database=None, max_examples=200, deadline=2000)
@given(data=json_value)
def test_canonical_json_is_stable_under_reordering_and_a_round_trip(data):
    first = canonical_json(data)
    again = canonical_json(json.loads(json.dumps(data)))
    assert first == again
    if isinstance(data, dict):
        reordered = dict(reversed(list(data.items())))
        assert canonical_json(reordered) == first
        assert spec_hash(reordered) == spec_hash(data)
    assert not math.isnan(len(first))


@pytest.mark.parametrize("raw", [
    {"type": "kappa_softmax", "params": {1: 0.5, "a": 1}},
    {"type": "kappa_softmax", 1: 2, "a": 3},
    {"type": "kappa_softmax", "params": {None: 1}},
])
def test_non_string_parameter_names_are_a_spec_error_not_a_typeerror(raw):
    """Regression: YAML reads ``1:`` as an int; ``check_params`` sorted the keys and raised a raw TypeError."""
    from mactf.exceptions import ComponentSpecError

    with pytest.raises(ComponentSpecError, match="parameter names must be strings"):
        ComponentSpec.parse(KIND, raw)


@hyp_settings(database=None, max_examples=150, deadline=5000)
@given(raw=spec_like)
def test_the_resolver_answers_every_spec_with_a_component_or_a_documented_error(raw):
    resolver = ComponentResolver()
    try:
        spec = ComponentSpec.parse(KIND, raw)
        resolver.validate(spec, trusted=False)
    except ImproperlyConfigured:
        return
    assert HUB.registry(KIND).has(spec.type)
