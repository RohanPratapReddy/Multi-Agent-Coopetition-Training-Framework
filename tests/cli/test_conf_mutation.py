"""``TRAINER.MUTATION`` through the configuration surface (document 17 section 12.10, rows 59-60)."""

import copy

import pytest

from mactf import conf
from mactf.exceptions import ImproperlyConfigured


def _settings(block=None, **trainer):
    values = copy.deepcopy(conf.DEFAULTS)
    if block is not None:
        values["TRAINER"]["MUTATION"] = block
    values["TRAINER"].update(trainer)
    return conf.Settings(values, module="tests")


def test_the_defaults_are_inert_and_valid():
    settings = _settings()
    conf.validate_settings(settings)
    assert settings["TRAINER"]["MUTATION"]["ENABLED"] is False
    assert settings["ORCHESTRATOR"]["ENGINE"] == "orchestrator" and settings["ORCHESTRATOR"]["ENGINE_COMPAT"] is True


@pytest.mark.parametrize(
    "block, text",
    [
        ({"ENABLE": True}, "unknown TRAINER.MUTATION keys"),
        ({"ENABLED": "yes"}, "ENABLED must be true or false"),
        ({"MAX_OPS_PER_PLAN": 0}, "MAX_OPS_PER_PLAN"),
        ({"TOMBSTONE": "never"}, "TOMBSTONE must be one of"),
        ({"WATCHDOG_PRUNE": "delete"}, "WATCHDOG_PRUNE must be one of"),
        ({"PROTECTED": "peer-1"}, "PROTECTED must be a list"),
        ({"SCHEDULE": [{"AT_EPOCH": 2, "OPS": [{"op": "explode", "path": "p"}]}], "ENABLED": True}, "op"),
        ({"SCHEDULE": [{"OPS": []}], "ENABLED": True}, "AT_EPOCH"),
        ({"SCHEDULE": [{"AT_EPOCH": 1, "OPS": [{"op": "retire", "path": "p"}]}]}, "ENABLED is false"),
        ("on", "must be a mapping"),
    ],
)
def test_the_yaml_block_is_validated_at_load(block, text):
    with pytest.raises(ImproperlyConfigured, match=text):
        conf.validate_settings(_settings(block))


def test_a_good_block_validates():
    # the same block with one wrong value is refused, so "validates" is not "validates anything"
    with pytest.raises(ImproperlyConfigured):
        conf.validate_settings(_settings({"ENABLED": True, "MIN_ACTIVE_ROOTS": 0}))
    conf.validate_settings(_settings({
        "ENABLED": True, "MIN_ACTIVE_ROOTS": 2, "PROTECTED": ["peer-1", "team-a/*"], "TOMBSTONE": "off",
        "SCHEDULE": [{"AT_EPOCH": 6, "OPS": [{"op": "add", "path": "peer-4", "like": "peer-1"}]}],
    }))


def test_the_set_form_parses_lists_and_mappings():
    """``--set`` addresses every key of the block, and a value is YAML (document 16 section 5.2).

    The environment form (``MACTF_TRAINER__MUTATION__PROTECTED='["peer-1"]'``) was added by P6 and is
    pinned in ``tests/test_conf_env.py`` (row 60 of 12.11.5 is the pair of tests).
    """
    from mactf.conf_overrides import apply_overrides, parse_override

    values = copy.deepcopy(conf.DEFAULTS)
    texts = [
        "TRAINER.MUTATION.ENABLED=true",
        'TRAINER.MUTATION.PROTECTED=["peer-1"]',
        "TRAINER.MUTATION.MAX_PLANS_PER_EPOCH=1",
        "TRAINER.MUTATION.MIN_ACTIVE_ROOTS=2",
        "TRAINER.MUTATION.SCHEDULE=[{AT_EPOCH: 2, OPS: [{op: retire, path: peer-2}]}]",
    ]
    parsed = [parse_override(text) for text in texts]
    written = apply_overrides(values, parsed, defaults=conf.DEFAULTS)
    block = values["TRAINER"]["MUTATION"]
    assert block["ENABLED"] is True and block["PROTECTED"] == ["peer-1"] and block["MAX_PLANS_PER_EPOCH"] == 1
    assert block["MIN_ACTIVE_ROOTS"] == 2
    assert block["SCHEDULE"] == [{"AT_EPOCH": 2, "OPS": [{"op": "retire", "path": "peer-2"}]}]
    assert "MUTATION" in written.trainer_keys
    conf.validate_settings(conf.Settings(values, module="tests"))
    # a typo cannot invent a key
    from mactf.exceptions import CommandError

    with pytest.raises(CommandError, match="did you mean ENABLED"):
        apply_overrides(copy.deepcopy(conf.DEFAULTS), [parse_override("TRAINER.MUTATION.ENABLE=true")], defaults=conf.DEFAULTS)


def test_the_engine_keys_are_validated():
    with pytest.raises(ImproperlyConfigured, match="E090"):
        conf.validate_settings(_settings_orchestrator(ENGINE="trianer"))
    with pytest.raises(ImproperlyConfigured, match="ENGINE_COMPAT must be true or false"):
        conf.validate_settings(_settings_orchestrator(ENGINE_COMPAT="yes"))
    with pytest.raises(ImproperlyConfigured, match="ALLOW_ENGINE_SWITCH"):
        conf.validate_settings(_settings_orchestrator(ALLOW_ENGINE_SWITCH=1))
    conf.validate_settings(_settings_orchestrator(ENGINE="Trainer"))               # case-insensitive, like engine_name


def _settings_orchestrator(**block):
    values = copy.deepcopy(conf.DEFAULTS)
    values["ORCHESTRATOR"].update(block)
    return conf.Settings(values, module="tests")
