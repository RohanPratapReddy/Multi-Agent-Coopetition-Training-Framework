"""The TRAINER keys of document 17 sections 10 and 11, and the torch-free mirrors that validate them."""

from __future__ import annotations

import copy

import pytest

from mactf import conf
from mactf.components.strategy_plan import MIGRATE_MODES as STRATEGY_MIGRATE_MODES
from mactf.components.weight_plan import MIGRATE_MODES as WEIGHT_MIGRATE_MODES
from mactf.components.guards.strategy import GUARD_MODES as STRATEGY_GUARD_MODES
from mactf.components.guards.strategy import GUARD_SCOPES as STRATEGY_GUARD_SCOPES
from mactf.coopetition.weight_guard import GUARD_MODES as WEIGHT_GUARD_MODES
from mactf.coopetition.weight_guard import GUARD_SCOPES as WEIGHT_GUARD_SCOPES
from mactf.exceptions import ImproperlyConfigured
from mactf.training.freeze import FREEZE_MODES


def test_the_torch_free_mirrors_equal_the_live_enumerations() -> None:
    assert conf._WEIGHT_GUARD_MODES == WEIGHT_GUARD_MODES and conf._WEIGHT_GUARD_SCOPES == WEIGHT_GUARD_SCOPES
    assert conf._STRATEGY_GUARD_MODES == STRATEGY_GUARD_MODES and conf._STRATEGY_GUARD_SCOPES == STRATEGY_GUARD_SCOPES
    assert conf._MIGRATE_MODES == STRATEGY_MIGRATE_MODES == WEIGHT_MIGRATE_MODES
    assert conf._FREEZE_MODES == FREEZE_MODES


def test_the_defaults_change_nothing() -> None:
    defaults = conf.TRAINER_DEFAULTS
    assert defaults["WEIGHT_UPDATE"] == "kappa_softmax" and defaults["PEER_STRATEGY"] == "kappa_softmax"
    assert defaults["WEIGHT_UPDATE_OVERRIDES"] == {} and defaults["PEER_STRATEGY_OVERRIDES"] == {}
    assert defaults["WEIGHT_GUARD"] == {"MODE": "repair", "SCOPE": "custom", "TOLERANCE": 1e-6, "PROBE": True}
    assert defaults["STRATEGY_GUARD"] == {"MODE": "skip", "SCOPE": "custom", "PROBE": True, "CHECK_FROZEN": True}
    assert defaults["WEIGHT_OPTIONS"] == defaults["PEER_STRATEGY_OPTIONS"] == {"STRICT_GLOBS": True, "MIGRATE": "reset"}
    assert defaults["AGGREGATOR"] is None and defaults["FREEZE_MODE"] == "legacy"
    # the guards' own constructors agree with the documented defaults
    from mactf.coopetition.weight_guard import WeightGuard
    from mactf.components.guards.strategy import StrategyGuard

    assert WeightGuard.from_settings(defaults["WEIGHT_GUARD"]).describe() == {
        "mode": "repair", "scope": "custom", "tolerance": 1e-6, "probe": True}
    assert StrategyGuard.from_settings(defaults["STRATEGY_GUARD"]).describe() == {
        "mode": "skip", "scope": "custom", "probe": True, "check_frozen": True}


@pytest.mark.parametrize("block,match", [
    ({"WEIGHT_UPDATE_OVERRIDES": "uniform"}, "mapping of glob"),
    ({"WEIGHT_UPDATE_OVERRIDES": {"a": 3}}, "WEIGHT_UPDATE_OVERRIDES"),
    ({"WEIGHT_UPDATE_OVERRIDES": {"": "uniform"}}, "usable glob"),
    ({"WEIGHT_UPDATE_OVERRIDES": {"a": {"params": {}}}}, "needs 'type'"),
    ({"WEIGHT_GUARD": ["repair"]}, "must be a mapping"),
    ({"WEIGHT_GUARD": {"MODE": "skip"}}, "E060.*MODE"),
    ({"WEIGHT_GUARD": {"SCOPE": "none"}}, "E060.*SCOPE"),
    ({"WEIGHT_GUARD": {"TOLERANCE": 0}}, "E060.*TOLERANCE"),
    ({"WEIGHT_GUARD": {"TOLERANCE": 0.5}}, "E060.*TOLERANCE"),
    ({"WEIGHT_GUARD": {"TOLERANCE": True}}, "E060.*TOLERANCE"),
    ({"WEIGHT_GUARD": {"EXTRA": 1}}, "E060.*unknown keys"),
    ({"WEIGHT_OPTIONS": {"MIGRATE": "merge"}}, "E060.*MIGRATE"),
    ({"WEIGHT_OPTIONS": {"OTHER": 1}}, "E060.*unknown keys"),
    ({"AGGREGATOR": 5}, "TRAINER.AGGREGATOR"),
    ({"AGGREGATOR": "no_such_aggregator"}, "must be one of default"),
    ({"PEER_STRATEGY_OVERRIDES": {"a": "no_such_strategy"}}, "PEER_STRATEGY_OVERRIDES"),
    ({"FREEZE_MODE": "frozen"}, "E070.*FREEZE_MODE"),
])
def test_bad_blocks_are_refused_at_load(settings, block, match) -> None:
    settings["TRAINER"] = {**copy.deepcopy(dict(settings["TRAINER"])), **block}
    with pytest.raises(ImproperlyConfigured, match=match):
        conf._validate_blocks(settings)


@pytest.mark.parametrize("block", [
    {"WEIGHT_UPDATE": {"type": "ema_softmax", "params": {"alpha": 0.3}}},        # a spec: shape only at load
    {"WEIGHT_UPDATE": {"import": "pkg.mod:Strategy"}},
    {"WEIGHT_UPDATE_OVERRIDES": {"team-*": "uniform", "root": {"type": "explicit", "params": {"weights": {}}}}},
    {"WEIGHT_GUARD": {"mode": "enforce", "scope": "all", "tolerance": 1e-3, "probe": False}},
    {"AGGREGATOR": {"type": "softclip_aggregator", "params": {"clip": 4.0}}},
    {"AGGREGATOR": "default"},
    {"FREEZE_MODE": "Strict"},
    {"PEER_STRATEGY_OPTIONS": {"migrate": "CARRY", "strict_globs": False}},
])
def test_good_blocks_pass_at_load(settings, block) -> None:
    from mactf.components.known import may_be_extended

    settings["TRAINER"] = {**copy.deepcopy(dict(settings["TRAINER"])), **block}
    settings["COMPONENTS"] = {**dict(settings["COMPONENTS"]), "MODULES": ["acme.components"]}   # names may be extended
    assert may_be_extended(settings["COMPONENTS"])
    conf._validate_blocks(settings)


def test_an_unknown_name_is_still_refused_when_nothing_could_provide_it(settings) -> None:
    settings["TRAINER"] = {**copy.deepcopy(dict(settings["TRAINER"])), "WEIGHT_UPDATE": "ema_softmax"}
    from mactf.components.known import may_be_extended

    if not may_be_extended(settings["COMPONENTS"]):
        with pytest.raises(ImproperlyConfigured, match="TRAINER.WEIGHT_UPDATE must be one of"):
            conf._validate_blocks(settings)


def test_kappa_overrides_shape_is_checked_at_load(settings) -> None:
    settings["COOPETITION"] = {**dict(settings["COOPETITION"]), "KAPPA_OVERRIDES": ["root"]}
    with pytest.raises(ImproperlyConfigured, match="E062.*KAPPA_OVERRIDES"):
        conf._validate_blocks(settings)
    settings["COOPETITION"] = {**dict(settings["COOPETITION"]), "KAPPA_OVERRIDES": {"root": {"ETA_KAPPA": 0.1}}}
    conf._validate_blocks(settings)
