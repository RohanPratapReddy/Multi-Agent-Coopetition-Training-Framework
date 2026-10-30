"""``MACTF_A__B__C`` environment overrides (document 17 section 15.6, 15.10 rows 20-28)."""

from __future__ import annotations

import copy
import logging

import pytest

from mactf import conf
from mactf.exceptions import ImproperlyConfigured


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    import os

    for key in [k for k in os.environ if k.startswith("MACTF_")]:
        monkeypatch.delenv(key)


def apply(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    values = copy.deepcopy(conf.DEFAULTS)
    conf._apply_env(values)
    return values


def test_a_nested_override_reaches_the_block(monkeypatch):
    values = apply(monkeypatch, MACTF_ORCHESTRATOR__ENGINE="trainer")
    assert values["ORCHESTRATOR"]["ENGINE"] == "trainer"


def test_a_third_segment_reaches_a_sub_key(monkeypatch):
    values = apply(monkeypatch, MACTF_TRAINER__MUTATION__ENABLED="true")
    assert values["TRAINER"]["MUTATION"]["ENABLED"] is True


def test_the_value_is_coerced_by_the_existing_type(monkeypatch):
    values = apply(
        monkeypatch,
        MACTF_TRAINER__MUTATION__ENABLED="true",
        MACTF_TRAINER__MUTATION__MAX_OPS_PER_PLAN="7",
        MACTF_TRAINER__MUTATION__PROTECTED='["peer-1"]',
    )
    mutation = values["TRAINER"]["MUTATION"]
    assert mutation["ENABLED"] is True and mutation["MAX_OPS_PER_PLAN"] == 7
    assert mutation["PROTECTED"] == ["peer-1"]


def test_a_whole_mapping_assign_parses_as_a_mapping(monkeypatch):
    values = apply(monkeypatch, MACTF_TRAINER__PEER_STRATEGY_OVERRIDES='{"team-*": independent}')
    assert values["TRAINER"]["PEER_STRATEGY_OVERRIDES"] == {"team-*": "independent"}


def test_a_flat_scalar_still_works(monkeypatch):
    assert apply(monkeypatch, MACTF_SEED="7")["SEED"] == 7


def test_a_flat_variable_naming_no_setting_is_still_ignored(monkeypatch):
    """MACTF_UPDATE_GOLDEN, MACTF_AGENT_PORT and friends are not settings and never were."""
    assert apply(monkeypatch, MACTF_UPDATE_GOLDEN="1") == apply(monkeypatch)


def test_a_mapping_flat_variable_is_e018(monkeypatch):
    with pytest.raises(ImproperlyConfigured, match="E018: MACTF_TRAINER sets TRAINER, which is a mapping"):
        apply(monkeypatch, MACTF_TRAINER="x")


def test_an_unknown_block_is_e018_with_a_suggestion(monkeypatch):
    with pytest.raises(ImproperlyConfigured, match=r"E018: .*unknown setting 'TRAINR'; did you mean TRAINER"):
        apply(monkeypatch, MACTF_TRAINR__EPOCHS="3")


def test_an_unknown_key_in_a_closed_block_is_e018(monkeypatch):
    with pytest.raises(ImproperlyConfigured, match=r"E018: .*has no key 'EPOCHZ'"):
        apply(monkeypatch, MACTF_TRAINER__EPOCHZ="3")


def test_a_scalar_container_is_e018(monkeypatch):
    with pytest.raises(ImproperlyConfigured, match=r"E018: .*TRAINER.EPOCHS is a int, not a mapping"):
        apply(monkeypatch, MACTF_TRAINER__EPOCHS__X="1")


def test_a_fifth_segment_is_e018(monkeypatch):
    with pytest.raises(ImproperlyConfigured, match="E018: .*5 segments; at most 4"):
        apply(monkeypatch, MACTF_TRAINER__MUTATION__A__B__C="1")


def test_an_empty_segment_is_e018(monkeypatch):
    with pytest.raises(ImproperlyConfigured, match="E018: .*empty segment"):
        apply(monkeypatch, MACTF_TRAINER____EPOCHS="1")


def test_creating_an_undeclared_key_is_w007(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="mactf.conf"):
        values = apply(monkeypatch, MACTF_API__TOKENS__ZZ="3")
    assert values["API"]["TOKENS"]["ZZ"] == 3
    assert any(r.getMessage().startswith("W007 ") for r in caplog.records)


def test_a_spec_leaf_keeps_the_lower_layers_params(monkeypatch):
    """15.9.2: ``..._WEIGHT_UPDATE__TYPE`` addresses ``type`` only when the layer below is a mapping."""
    monkeypatch.setenv("MACTF_TRAINER__WEIGHT_UPDATE", "uniform")
    values = copy.deepcopy(conf.DEFAULTS)
    conf._apply_env(values)
    assert values["TRAINER"]["WEIGHT_UPDATE"] == "uniform"
    values = copy.deepcopy(conf.DEFAULTS)
    values["TRAINER"]["WEIGHT_UPDATE"] = {"type": "trimmed_mean", "params": {"trim": 0.1}}
    monkeypatch.delenv("MACTF_TRAINER__WEIGHT_UPDATE")
    monkeypatch.setenv("MACTF_TRAINER__WEIGHT_UPDATE__TYPE", "uniform")
    conf._apply_env(values)
    assert values["TRAINER"]["WEIGHT_UPDATE"] == {"type": "uniform", "params": {"trim": 0.1}}
