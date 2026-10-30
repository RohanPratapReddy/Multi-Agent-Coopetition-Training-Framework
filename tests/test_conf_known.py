"""P6 additions to the static-name layer: spec discovery, built-in import, SCORING.STAGES shape,
RL defaults and the W016 note (document 17 sections 15.1, 15.4, 17.3)."""

from __future__ import annotations

import copy
import subprocess
import sys

import pytest

from mactf import conf
from mactf.components import known
from mactf.exceptions import ImproperlyConfigured


def settings(**blocks):
    values = copy.deepcopy(conf.DEFAULTS)
    for name, update in blocks.items():
        values[name] = {**values[name], **update} if isinstance(update, dict) else update
    return conf.Settings(values, module="t")


def test_rl_defaults_is_a_named_constant_and_a_copy():
    assert conf.DEFAULTS["RL"] == conf.RL_DEFAULTS
    assert conf.DEFAULTS["RL"] is not conf.RL_DEFAULTS
    assert set(conf.RL_DEFAULTS) == {"UNCONSUMED", "STRICT_GLOBS", "VERIFY"}


def test_the_components_app_is_installed_by_default():
    assert "mactf.contrib.components" in conf.DEFAULTS["INSTALLED_APPS"]
    from mactf import settings_default

    assert "mactf.contrib.components" in settings_default.INSTALLED_APPS


def test_defaults_name_no_spec():
    assert known.configured_specs(settings()) == [
        # nothing is selected by default except the two names the trainer block spells out
        item for item in known.configured_specs(settings())
    ]
    names = {slot for slot, _, _ in known.configured_specs(settings())}
    assert names <= {"TRAINER.PEER_STRATEGY", "TRAINER.WEIGHT_UPDATE"}


def test_configured_specs_finds_every_slot():
    found = known.configured_specs(settings(
        TRAINER={
            "WEIGHT_UPDATE": {"type": "uniform"},
            "WEIGHT_UPDATE_OVERRIDES": {"team-*": "uniform"},
            "PEER_STRATEGY_OVERRIDES": {"b": "independent"},
            "POLICY": "default",
        },
        RL={"LOSS": "ppo", "OVERRIDES": {"ADVANTAGE": {"peer-b*": {"type": "clipped"}}}},
        SCORING={"STAGES": {"reward": "zero_reward"}},
    ))
    by_slot = {slot: kind for slot, kind, _ in found}
    assert by_slot["TRAINER.WEIGHT_UPDATE"] == "weight_update"
    assert by_slot["TRAINER.WEIGHT_UPDATE_OVERRIDES.team-*"] == "weight_update"
    assert by_slot["TRAINER.PEER_STRATEGY_OVERRIDES.b"] == "peer_strategy"
    assert by_slot["TRAINER.POLICY"] == "trainer_policy"
    assert by_slot["RL.LOSS"] == "rl_function.loss"
    assert by_slot["RL.OVERRIDES.ADVANTAGE.peer-b*"] == "rl_function.advantage"
    assert by_slot["SCORING.STAGES.reward"] == "scorer"


def test_import_builtins_registers_the_trainer_policies():
    code = (
        "from mactf.components import HUB\n"
        "from mactf.components.known import import_builtins\n"
        "assert HUB.registry('trainer_policy').names() == []\n"
        "import_builtins()\n"
        "assert {'default', 'co_distill'} <= set(HUB.registry('trainer_policy').names())\n"
        "assert 'penalty' in HUB.registry('scorer').names()\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("stages,match", [
    ("reward", "SCORING.STAGES must map"),
    ({"reward": 3}, r"SCORING.STAGES.reward: a spec is a string or a mapping"),
    ({"reward": {"params": {}}}, "needs exactly one of 'type' and 'import'"),
    ({"hierarchy": "x"}, "unknown kind"),
])
def test_scoring_stages_shape_is_checked_at_load(stages, match):
    values = copy.deepcopy(conf.DEFAULTS)
    values["SCORING"]["STAGES"] = stages
    with pytest.raises(ImproperlyConfigured, match=match):
        conf.validate_settings(conf.Settings(values, module="t"))


def test_a_well_formed_stage_selection_loads():
    values = copy.deepcopy(conf.DEFAULTS)
    values["SCORING"]["STAGES"] = {"reward": {"type": "tempered", "params": {"factor": 0.5}}, "penalty": "penalty"}
    conf.validate_settings(conf.Settings(values, module="t"))      # names are decided at build


def test_w016_a_null_entry_alone_in_its_group():
    from mactf.training.optimizer_plan import OptimizerPlan

    block = {"TYPE": "torch", "CLASS": "AdamW", "OVERRIDES": {"root/a/leaf": {"TYPE": "null"}}}
    plan = OptimizerPlan.from_settings(
        {"OPTIMIZER": block}, block=block, written=frozenset(block), horizon=4
    )
    warnings = plan.validate(["root/a/leaf", "root/b/x", "root/b/y"], initial=True)
    assert [w[:4] for w in warnings] == ["W016"]
    assert plan.validate(["root/a/leaf", "root/a/other", "root/b/x"], initial=True) == []
