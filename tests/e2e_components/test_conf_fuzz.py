"""``--set`` / env parsing and settings validation with garbage: a documented error, never a raw traceback."""

from __future__ import annotations

import copy
import os

import pytest
from hypothesis import HealthCheck, given, settings as hyp_settings, strategies as st

from mactf import conf
from mactf.conf import Settings, validate_settings
from mactf.conf_overrides import apply_overrides, parse_override
from mactf.exceptions import MACTFError


def _paths(tree, prefix=()):
    for key, value in tree.items():
        if str(key).startswith("_"):
            continue
        yield ".".join((*prefix, str(key)))
        if isinstance(value, dict) and len(prefix) < 3:
            yield from _paths(value, (*prefix, str(key)))


KEYS = sorted(set(_paths(conf.DEFAULTS)))
SUFFIX = st.sampled_from(["", ".type", ".params", ".params.k", ".nope", ".0", ".PARAMS.x"])
VALUES = st.one_of(
    st.sampled_from(["", "null", "~", "true", "-1", "0", "1e999", "nan", "inf", "[1,2", "{a: 1", "{a: [}", "'x", "[]", "{}",
                     "{type: kappa_softmax}", "{type: nope, params: {a: 1}}", "[{op: add}]", "!!python/object:os.system x",
                     "a: b: c", "\u0000", "0x10", "1_000", "9" * 400, "{1: 2}", "[null]", "{type: 5}"]),
    st.text(max_size=12), st.integers(-3, 3).map(str), st.floats(allow_nan=True, allow_infinity=True).map(repr),
)
OVERRIDE = st.builds(lambda k, s, v: f"{k}{s}={v}", st.sampled_from(KEYS), SUFFIX, VALUES)
RAW = st.one_of(OVERRIDE, st.text(max_size=20), st.builds(lambda a, b: f"{a}={b}", st.text(max_size=8), st.text(max_size=8)))


@hyp_settings(database=None, max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(texts=st.lists(RAW, min_size=1, max_size=4))
def test_set_overrides_end_in_a_value_or_a_documented_error(texts):
    values = copy.deepcopy(conf.DEFAULTS)
    try:
        apply_overrides(values, [parse_override(t) for t in texts], defaults=conf.DEFAULTS)
        validate_settings(Settings(values))
    except MACTFError:
        return
    # accepted: the result must still be a coherent settings tree
    assert isinstance(values["TRAINER"], dict)


@hyp_settings(database=None, max_examples=150, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(name=st.sampled_from(["TRAINER", "TRAINER__MUTATION", "COMPONENTS", "OPTIMIZER", "TRAINING", "SCORING", "COMPLIANCE", "X", ""]),
       leaf=st.sampled_from(["", "__ENABLED", "__MODULES", "__WEIGHT_UPDATE", "__WEIGHT_UPDATE__TYPE", "__LOCKED", "__LR", "__", "____"]),
       value=VALUES.filter(lambda v: "\x00" not in v))
def test_environment_overrides_end_in_a_value_or_a_documented_error(name, leaf, value, monkeypatch, tmp_path):
    for key in [k for k in os.environ if k.startswith("MACTF_") and k not in {"MACTF_SETTINGS_MODULE"}]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(f"MACTF_{name}{leaf}", value)
    values = copy.deepcopy(conf.DEFAULTS)
    try:
        conf._apply_env(values)
        validate_settings(Settings(values))
    except MACTFError:
        return
