"""The ``COMPONENTS`` block and its validators (document 17 section 15.1, 15.4, 15.10 rows 1-12, 42-43)."""

from __future__ import annotations

import copy
import logging
import sys
import types

import pytest

from mactf import conf
from mactf.components import HUB, HubPolicy, plugins, policy_from_settings
from mactf.conf import COMPONENTS_DEFAULTS, DEFAULTS, Settings
from mactf.exceptions import ImproperlyConfigured

DOCUMENTED_KEYS = {
    "ALLOW_BUILTIN_OVERRIDE", "LOCKED", "STRICT_LEGACY", "ALLOW_IMPORT", "IMPORT_ALLOWLIST",
    "MAX_PER_KIND", "PLUGINS", "MODULES", "FREEZE_AFTER_LOAD", "STRICT_RESTORE",
}


def validate(**block):
    values = copy.deepcopy(DEFAULTS)
    values["COMPONENTS"] = {**copy.deepcopy(COMPONENTS_DEFAULTS), **block}
    conf.validate_settings(Settings(values, module="t"))


def validate_raw(block):
    values = copy.deepcopy(DEFAULTS)
    values["COMPONENTS"] = block
    conf.validate_settings(Settings(values, module="t"))


def test_defaults_block_has_every_documented_key():
    assert set(COMPONENTS_DEFAULTS) == DOCUMENTED_KEYS


def test_defaults_are_inert(caplog):
    """P1: a project writing nothing gets HubPolicy() and no W006/W007."""
    with caplog.at_level(logging.WARNING):
        validate()
    assert policy_from_settings(COMPONENTS_DEFAULTS) == HubPolicy()
    assert not [r for r in caplog.records if r.getMessage().startswith(("W006", "W007"))]
    assert COMPONENTS_DEFAULTS == {
        "ALLOW_BUILTIN_OVERRIDE": False, "LOCKED": [], "STRICT_LEGACY": False, "ALLOW_IMPORT": False,
        "IMPORT_ALLOWLIST": [], "MAX_PER_KIND": 512, "PLUGINS": "auto", "MODULES": [],
        "FREEZE_AFTER_LOAD": False, "STRICT_RESTORE": False,
    }


def test_the_block_is_a_copy_of_the_constant():
    assert DEFAULTS["COMPONENTS"] == COMPONENTS_DEFAULTS
    assert DEFAULTS["COMPONENTS"] is not COMPONENTS_DEFAULTS
    assert DEFAULTS["COMPONENTS"]["LOCKED"] is not COMPONENTS_DEFAULTS["LOCKED"]
    DEFAULTS["COMPONENTS"]["MODULES"].append("x")
    try:
        assert COMPONENTS_DEFAULTS["MODULES"] == []
    finally:
        DEFAULTS["COMPONENTS"]["MODULES"].remove("x")


@pytest.mark.parametrize("flag", ["ALLOW_BUILTIN_OVERRIDE", "STRICT_LEGACY", "ALLOW_IMPORT",
                                  "FREEZE_AFTER_LOAD", "STRICT_RESTORE"])
@pytest.mark.parametrize("value", ["yes", 1, 0, None, "true", [True]])
def test_a_non_bool_flag_is_e013(flag, value):
    with pytest.raises(ImproperlyConfigured, match=rf"E013: COMPONENTS.{flag} must be true or false"):
        validate(**{flag: value})


@pytest.mark.parametrize("flag", ["ALLOW_BUILTIN_OVERRIDE", "STRICT_LEGACY", "ALLOW_IMPORT",
                                  "FREEZE_AFTER_LOAD", "STRICT_RESTORE"])
def test_a_bool_flag_is_accepted(flag):
    validate(**{flag: True})
    validate(**{flag: False})


def test_an_unknown_key_is_e013_and_lists_the_accepted_set():
    with pytest.raises(ImproperlyConfigured) as caught:
        validate(ALLOW_BUILTIN_OVERIDE=True)
    message = str(caught.value)
    assert message.startswith("E013: COMPONENTS has unknown key(s) ['ALLOW_BUILTIN_OVERIDE']")
    assert all(key in message for key in DOCUMENTED_KEYS)


def test_keys_are_case_insensitive_like_the_alias_fold():
    validate_raw({"max_per_kind": 5})
    with pytest.raises(ImproperlyConfigured, match="E013"):
        validate_raw({"max_per_kind": 0})


@pytest.mark.parametrize("value", [3, "x", [1], None])
def test_a_block_that_is_not_a_mapping_is_e013(value):
    values = copy.deepcopy(DEFAULTS)
    values["COMPONENTS"] = value
    with pytest.raises(ImproperlyConfigured, match="E013: COMPONENTS must be a mapping"):
        conf.validate_settings(Settings(values, module="t"))


@pytest.mark.parametrize("value", [0, -1, 1.5, "512", True, None])
def test_max_per_kind_must_be_a_positive_integer(value):
    with pytest.raises(ImproperlyConfigured, match="MAX_PER_KIND must be an integer >= 1"):
        validate(MAX_PER_KIND=value)
    validate(MAX_PER_KIND=1)


@pytest.mark.parametrize("entry", ["./components.py", "acme/scoring", "acme.", "acme..x", ".acme",
                                   "1acme", "acme-x", "", 5, None, {"a": 1}])
def test_a_module_path_is_e014(entry):
    with pytest.raises(ImproperlyConfigured, match="E014: COMPONENTS.MODULES entry"):
        validate(MODULES=[entry])


def test_modules_must_be_a_list_and_dotted_names_pass():
    with pytest.raises(ImproperlyConfigured, match="MODULES must be a list of dotted module names"):
        validate(MODULES="acme.components")
    validate(MODULES=["acme", "acme.components", "_x.y_2"])
    validate(MODULES=("a.b",))


@pytest.mark.parametrize("value", ["auto", "AUTO", "off", "Off", [], ["mactf-extras"], ("a", "b")])
def test_plugins_forms(value):
    validate(PLUGINS=value)


@pytest.mark.parametrize("value", ["sometimes", "", 1, True, None, ["ok", ""], [" "], [1]])
def test_a_bad_plugins_value_is_e015(value):
    with pytest.raises(ImproperlyConfigured, match="E015"):
        validate(PLUGINS=value)


@pytest.mark.parametrize("pattern", ["penalty", "scorer:", ":penalty", "nonsense:penalty", "", " scorer:x", 3,
                                     "rl_function.nonsense:x"])
def test_a_lock_pattern_needs_a_kind_or_slot(pattern):
    with pytest.raises(ImproperlyConfigured, match="E016: COMPONENTS.LOCKED"):
        validate(LOCKED=[pattern])


@pytest.mark.parametrize("pattern", ["scorer:penalty", "weight_update:*", "optimizer:AdamW", "aggregator:default",
                                     "rl_function.loss:ppo", "slot:SCORING.CHAIN.penalty", "slot:TRAINER.*",
                                     "trainer_policy:co_distill", "reward_term:capped"])
def test_every_documented_lock_pattern_is_accepted_for_every_kind(pattern):
    validate(LOCKED=[pattern])


def test_locked_must_be_a_list():
    with pytest.raises(ImproperlyConfigured, match="LOCKED must be a list of patterns"):
        validate(LOCKED="scorer:penalty")


@pytest.mark.parametrize("pattern", ["*", "**"])
def test_a_blanket_allowlist_is_refused(pattern):
    with pytest.raises(ImproperlyConfigured, match="E017: COMPONENTS.IMPORT_ALLOWLIST cannot be"):
        validate(IMPORT_ALLOWLIST=[pattern])


@pytest.mark.parametrize("entry", ["", "  ", 3, None])
def test_a_bad_allowlist_entry_is_e017(entry):
    with pytest.raises(ImproperlyConfigured, match="E017"):
        validate(IMPORT_ALLOWLIST=[entry])
    validate(IMPORT_ALLOWLIST=["acme.*", "myproject.components"])
    with pytest.raises(ImproperlyConfigured, match="IMPORT_ALLOWLIST must be a list"):
        validate(IMPORT_ALLOWLIST="acme.*")


# -- the alias ------------------------------------------------------------------------
def test_the_alias_is_w006(caplog):
    with caplog.at_level(logging.WARNING, logger="mactf.conf"):
        validate_raw({"API_ALLOW_IMPORT": True})
    assert not [r for r in caplog.records if r.getMessage().startswith("W006")]  # the alias alone is silent
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="mactf.conf"):
        validate_raw({"API_ALLOW_IMPORT": True, "ALLOW_IMPORT": True})
    warned = [r.getMessage() for r in caplog.records if r.getMessage().startswith("W006")]
    assert len(warned) == 1 and "COMPONENTS.ALLOW_IMPORT" in warned[0]
    assert policy_from_settings({"API_ALLOW_IMPORT": True}).allow_import is True


def test_the_alias_conflict_is_e013():
    with pytest.raises(ImproperlyConfigured, match="E013: COMPONENTS.* both set ALLOW_IMPORT"):
        validate_raw({"API_ALLOW_IMPORT": True, "ALLOW_IMPORT": False})
    with pytest.raises(ImproperlyConfigured, match="E013"):
        validate_raw({"ALLOW_IMPORT": False, "API_ALLOW_IMPORT": True})


def test_the_alias_folds_onto_the_canonical_key_and_uppercases():
    assert conf._canonical_components({"api_allow_import": True}) == {"ALLOW_IMPORT": True}
    assert conf._canonical_components({"Max_Per_Kind": 3, "x": 1}) == {"MAX_PER_KIND": 3, "X": 1}
    assert conf._canonical_components({}) == {} and conf._canonical_components(None) == {}


# -- the TRAINER names now read the static table -----------------------------------------------
def validate_trainer(trainer, components=None):
    values = copy.deepcopy(DEFAULTS)
    values["TRAINER"].update(trainer)
    if components is not None:
        values["COMPONENTS"] = components
    conf.validate_settings(Settings(values, module="t"))


@pytest.fixture
def no_plugins(monkeypatch):
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group=None, **k: [])


def test_the_trainer_names_keep_their_message_and_case_folding(no_plugins):
    with pytest.raises(ImproperlyConfigured, match="TRAINER.PEER_STRATEGY must be one of cooperative, "):
        validate_trainer({"PEER_STRATEGY": "nonsense"})
    with pytest.raises(ImproperlyConfigured, match="TRAINER.WEIGHT_UPDATE must be one of "):
        validate_trainer({"WEIGHT_UPDATE": "trimmed_mean"})
    validate_trainer({"PEER_STRATEGY": "COOPERATIVE", "WEIGHT_UPDATE": "Score_Weighted"})


def test_a_plugin_name_is_accepted_when_modules_could_provide_it(no_plugins):
    validate_trainer({"WEIGHT_UPDATE": "trimmed_mean"}, {"MODULES": ["acme.strategies"]})
    validate_trainer({"PEER_STRATEGY": "mix_every_n"}, {"MODULES": ["acme.strategies"]})


def test_a_mapping_spec_is_accepted_by_shape(no_plugins):
    validate_trainer({"WEIGHT_UPDATE": {"type": "uniform"}})
    validate_trainer({"WEIGHT_UPDATE": {"import": "a.b:C", "params": {"k": 1}}})
    with pytest.raises(ImproperlyConfigured, match="needs 'type'"):
        validate_trainer({"WEIGHT_UPDATE": {"params": {}}})


# -- loading with the settings --------------------------------------------------------------------
@pytest.fixture
def component_settings(tmp_path, monkeypatch, restore_hub):
    """A settings module whose COMPONENTS block names a module that registers a policy."""
    imports: list[str] = []
    module = types.ModuleType("zz_conf_components_mod")

    def register():
        HUB.registry("trainer_policy").register("zz_from_conf", type("P", (), {"train_step": lambda s, n, b: None}))

    real_import = plugins.importlib.import_module

    def fake_import(name, *a, **k):
        if name == "zz_conf_components_mod":
            imports.append(name)
            register()
            return module
        return real_import(name, *a, **k)

    monkeypatch.setattr(plugins.importlib, "import_module", fake_import)
    plugins.reset_for_tests()

    def make(components):
        settings_module = types.ModuleType("mactf_components_settings")
        settings_module.BASE_DIR = str(tmp_path)
        settings_module.CONFIG_FILES = []
        settings_module.DEVICE = "cpu"
        settings_module.PEERS = []
        settings_module.COMPONENTS = components
        monkeypatch.setitem(sys.modules, "mactf_components_settings", settings_module)
        return conf.configure("mactf_components_settings", force=True)

    yield make, imports
    conf.reset()
    plugins.reset_for_tests()


def test_configure_loads_the_modules_the_block_names(component_settings):
    make, imports = component_settings
    loaded = make({"MODULES": ["zz_conf_components_mod"], "PLUGINS": "off", "MAX_PER_KIND": 40})
    assert loaded["COMPONENTS"]["MODULES"] == ["zz_conf_components_mod"]
    assert imports == ["zz_conf_components_mod"]
    assert HUB.registry("trainer_policy").has("zz_from_conf")
    assert HUB.policy.max_components_per_kind == 40  # the policy came from the settings


def test_ensure_components_loaded_is_idempotent_across_configure(component_settings):
    make, imports = component_settings
    block = {"MODULES": ["zz_conf_components_mod"], "PLUGINS": "off"}
    make(block)
    make(block)  # a second configure with the same block imports nothing again
    assert imports == ["zz_conf_components_mod"]


def test_a_changed_block_reloads(component_settings):
    make, imports = component_settings
    make({"MODULES": ["zz_conf_components_mod"], "PLUGINS": "off"})
    make({"MODULES": ["zz_conf_components_mod"], "PLUGINS": "off", "STRICT_RESTORE": True})
    assert HUB.policy.strict_restore is True
    assert imports == ["zz_conf_components_mod"]  # the module itself is loaded once per process


def test_a_default_configure_leaves_the_default_policy(component_settings):
    make, _imports = component_settings
    make({})
    assert HUB.policy == HubPolicy()


def test_a_bad_block_never_reaches_the_hub(component_settings):
    make, imports = component_settings
    with pytest.raises(ImproperlyConfigured, match="E013"):
        make({"NOT_A_KEY": 1})
    assert imports == [] and HUB.policy == HubPolicy()


def test_strict_restore_reaches_the_hub_policy():
    """15.10 row 12: the block key, the policy field and the restorer's ``getattr`` agree."""
    assert policy_from_settings({}).strict_restore is False
    assert policy_from_settings(dict(COMPONENTS_DEFAULTS)).strict_restore is False
    assert policy_from_settings({"STRICT_RESTORE": True}).strict_restore is True
    assert getattr(policy_from_settings({"STRICT_RESTORE": True}), "strict_restore", False) is True
