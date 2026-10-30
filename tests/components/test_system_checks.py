"""The component system checks (document 17 section 17.3)."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import pytest

from mactf import checks, conf
import mactf.contrib.components.systems  # noqa: F401  (registers the checks, as ready() does)
from mactf.components import HUB, register_optimizer, register_policy, register_weight_update
from mactf.components.kinds import Mutability
from mactf.contrib.components import checks as component_checks
from mactf.contrib.components.checks import (
    check_builtin_replacements,
    check_configured_specs,
    check_dead_trainer_settings,
    check_engine_support,
    check_guard_modes,
    check_registry_contracts,
    check_unsafe_policy,
)

ALL_CHECKS = (
    check_registry_contracts, check_configured_specs, check_builtin_replacements,
    check_dead_trainer_settings, check_guard_modes, check_unsafe_policy, check_engine_support,
)


@pytest.fixture
def settings(restore_hub, monkeypatch):
    """Install a settings object built from the defaults plus per-block updates."""
    from mactf.components.known import import_builtins

    import_builtins()           # the checks load every kind's built-ins; so the snapshots below include them

    def install(**blocks):
        values = copy.deepcopy(conf.DEFAULTS)
        for name, update in blocks.items():
            if isinstance(update, dict) and isinstance(values.get(name), dict):
                values[name].update(update)
            else:
                values[name] = update
        loaded = conf.Settings(values, module="t")
        monkeypatch.setattr(conf, "_settings", loaded)
        return loaded

    install()
    yield install
    monkeypatch.setattr(conf, "_settings", None)


def ids(messages):
    return sorted(message.id for message in messages)


# -- every check has its code and its hint -------------------------------------------
def test_each_check_has_its_code_and_hint(settings, component_overlay):
    # (fixture order matters: ``settings`` loads the built-ins before the overlay snapshots)
    """Provoke each check once; every message carries an id, a hint and the app namespace."""
    from mactf.components.codes import CODE_TABLE

    settings(
        COMPLIANCE={"GUARD_MODE": "fallback"},
        RL={"VERIFY": "off"},
        TRAINER={"WEIGHT_UPDATE": "nonexistent", "PEER_STRATEGY": "independent", "POLICY": {"type": "nope"}},
        ORCHESTRATOR={"ENGINE": "orchestrator"},
    )
    messages = []
    for check in ALL_CHECKS:
        messages.extend(check())
    assert messages
    for message in messages:
        assert message.id and message.id.startswith("mactf.components."), message
        assert message.id.rsplit(".", 1)[1] in CODE_TABLE, message
        assert message.hint, message
        assert message.level in (checks.ERROR, checks.WARNING)
    assert {"mactf.components.W002", "mactf.components.W003", "mactf.components.W013",
            "mactf.components.E001"} <= set(ids(messages))


def test_the_checks_are_registered_under_the_app_and_tagged():
    listed = {info.name.rsplit(".", 1)[1]: info for info in checks.list_checks() if info.app == "mactf_components"}
    assert set(listed) == {check.__name__ for check in ALL_CHECKS}
    assert all(info.tags == ("components",) and info.requires == "mactf.contrib.components"
               for info in listed.values())


# -- defaults are silent --------------------------------------------------------------
def test_the_defaults_produce_no_message(settings):
    for check in ALL_CHECKS:
        assert check() == [], check.__name__


# -- one test per check that provokes it ------------------------------------------------
def test_registry_contracts_reports_e007(settings, component_overlay):
    class NotAStrategy:                       # passes registration through the legacy back door
        pass

    registry = HUB.registry("weight_update")
    component = registry.get("uniform")
    broken = type(component)(
        kind=component.kind, name="broken", factory=NotAStrategy, origin=component.origin,
        mutability=component.mutability,
    )
    registry._entries["broken"] = broken
    try:
        found = check_registry_contracts()
    finally:
        registry._entries.pop("broken", None)
    assert any(m.id == "mactf.components.E007" and "weight_update:broken" in m.message for m in found)


def test_registry_contracts_reports_e002_for_names_differing_by_case(settings, component_overlay):
    import torch

    @register_optimizer("LionX", scope="run")
    class A(torch.optim.Optimizer):
        def __init__(self, params, lr: float = 1e-3):
            super().__init__(params, {"lr": lr})

    registry = HUB.registry("optimizer")
    clone = registry.get("LionX")
    registry._entries["lionx"] = type(clone)(
        kind=clone.kind, name="lionx", factory=clone.factory, origin=clone.origin, mutability=clone.mutability
    )
    try:
        found = check_registry_contracts()
    finally:
        registry._entries.pop("lionx", None)
    assert any(m.id == "mactf.components.E002" for m in found)


def test_configured_specs_reports_every_class_of_error(settings):
    settings(
        TRAINER={
            "WEIGHT_UPDATE": {"type": "uniform", "params": {"trimm": 0.2}},
            "PEER_STRATEGY": "nonexistent_strategy",
        },
        SCORING={**conf.DEFAULTS["SCORING"], "STAGES": {"reward": {"type": "zero_reward", "params": {"nope": 1}}}},
    )
    found = check_configured_specs()
    by_slot = {m.message.split(":", 1)[0]: m for m in found}
    assert by_slot["TRAINER.WEIGHT_UPDATE"].id == "mactf.components.E004"
    assert by_slot["TRAINER.PEER_STRATEGY"].id == "mactf.components.E001"
    assert "Accepted" in by_slot["TRAINER.PEER_STRATEGY"].hint
    assert by_slot["SCORING.STAGES.reward"].id == "mactf.components.E004"


def test_configured_specs_reports_an_unimportable_module_as_e008(settings):
    settings(TRAINER={"WEIGHT_UPDATE": {"import": "no_such_module_zz:Thing"}})
    found = check_configured_specs()
    assert [m.id for m in found] == ["mactf.components.E008"]


def test_configured_specs_accepts_a_registered_plugin_name(settings, component_overlay):
    from p3_examples import EmaSoftmax

    register_weight_update("ema_softmax", EmaSoftmax, scope="run")
    settings(TRAINER={"WEIGHT_UPDATE": {"type": "ema_softmax", "params": {"alpha": 0.5}}})
    assert check_configured_specs() == []


def test_builtin_replacements_reports_w001_only_when_the_flag_is_on(settings, component_overlay):
    from mactf.components import HubPolicy
    from mactf.coopetition.weights import UniformAverage

    settings(COMPONENTS={"ALLOW_BUILTIN_OVERRIDE": True})
    HUB.configure(HubPolicy(allow_builtin_override=True))

    class Replacement(UniformAverage):
        pass

    HUB.registry("weight_update").register("uniform", Replacement, replace=True, scope="run")
    found = check_builtin_replacements()
    assert [m.id for m in found] == ["mactf.components.W001"]
    settings()
    assert check_builtin_replacements() == []


def test_dead_trainer_settings_w002_follows_the_engine(settings):
    settings(TRAINER={"WEIGHT_UPDATE": "uniform"})
    assert ids(check_dead_trainer_settings()) == ["mactf.components.W002"]
    settings(TRAINER={"WEIGHT_UPDATE": "uniform"}, ORCHESTRATOR={"ENGINE": "trainer"})
    assert check_dead_trainer_settings() == []


def test_guard_modes_w003_and_w013(settings):
    settings(COMPLIANCE={"GUARD_MODE": "fallback"})
    assert ids(check_guard_modes()) == ["mactf.components.W003"]
    settings(RL={"VERIFY": "off"})
    assert ids(check_guard_modes()) == ["mactf.components.W013"]
    settings(COMPLIANCE={"GUARD_MODE": "fallback"}, RL={"VERIFY": "off"})
    assert ids(check_guard_modes()) == ["mactf.components.W003", "mactf.components.W013"]


def _unsafe_policy(component_overlay):
    class Reckless:
        unsafe = True

        def train_step(self, node, batch): ...

    register_policy("reckless_check", Reckless, scope="run")

    return Reckless


def test_check_unsafe_policy_is_e009_when_enforced(settings, component_overlay):
    _unsafe_policy(component_overlay)
    settings(TRAINER={"POLICY": {"type": "reckless_check"}})
    found = check_unsafe_policy()
    assert [m.id for m in found] == ["mactf.components.W004", "mactf.components.E009"]
    assert [m.level for m in found] == [checks.WARNING, checks.ERROR]
    settings(TRAINER={"POLICY": {"type": "reckless_check"}, "ALLOW_UNSAFE_POLICY": True})
    found = check_unsafe_policy()
    assert [m.id for m in found] == ["mactf.components.W004"]


def test_a_guarded_policy_is_not_flagged(settings):
    settings(TRAINER={"POLICY": {"type": "default"}})
    assert check_unsafe_policy() == []


def test_engine_support_e090_and_e092(settings):
    settings(ORCHESTRATOR={"ENGINE": "quantum"})
    assert ids(check_engine_support()) == ["mactf.components.E090"]
    settings(ORCHESTRATOR={"ENGINE": "trainer"}, INSTALLED_APPS=["mactf.distributed"])
    assert ids(check_engine_support()) == ["mactf.components.E092"]
    settings(ORCHESTRATOR={"ENGINE": "orchestrator"}, INSTALLED_APPS=["mactf.distributed"])
    assert check_engine_support() == []


def test_dead_trainer_settings_does_not_double_report_a_bad_engine(settings):
    settings(ORCHESTRATOR={"ENGINE": "quantum"}, TRAINER={"WEIGHT_UPDATE": "uniform"})
    assert check_dead_trainer_settings() == []


# -- the clean project ------------------------------------------------------------------
def test_a_clean_project_has_no_messages(tmp_path):
    """``mactf check`` on a fresh ``mactf startproject``: the components app is silent.

    (Messages from other apps are environmental: the sandbox app warns on a host with no
    hardware virtualization, which is every CI container.)
    """
    env = {**os.environ, "NO_COLOR": "1", "MACTF_NO_DISCOVER": ""}
    env.pop("MACTF_NO_DISCOVER")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(tmp_path), env.get("PYTHONPATH", "")]))
    made = subprocess.run(
        [sys.executable, "-m", "mactf", "startproject", "demo_clean", "--directory", str(tmp_path)],
        capture_output=True, text=True, cwd=tmp_path, env=env, timeout=120,
    )
    if made.returncode != 0:                       # older flag spelling: run in the directory
        made = subprocess.run(
            [sys.executable, "-m", "mactf", "startproject", "demo_clean"],
            capture_output=True, text=True, cwd=tmp_path, env=env, timeout=120,
        )
    assert made.returncode == 0, made.stderr
    project = next(p for p in tmp_path.rglob("manage.py")).parent.parent
    ran = subprocess.run(
        [sys.executable, "-m", "mactf", "check", "--format", "json", "--settings", "demo_clean.settings"],
        capture_output=True, text=True, cwd=project, env={**env, "PYTHONPATH": str(project)}, timeout=180,
    )
    assert ran.returncode == 0, ran.stderr
    payload = json.loads(ran.stdout)
    ours = [m for m in payload["messages"] if (m["id"] or "").startswith("mactf.components.")]
    assert ours == []
    assert payload["counts"]["ERROR"] == 0
    others = [m for m in payload["messages"] if not (m["id"] or "").startswith("mactf.sandbox.")]
    assert others == []
    narrowed = subprocess.run(
        [sys.executable, "-m", "mactf", "check", "--app", "mactf_components", "--format", "json",
         "--settings", "demo_clean.settings"],
        capture_output=True, text=True, cwd=project, env={**env, "PYTHONPATH": str(project)}, timeout=180,
    )
    body = json.loads(narrowed.stdout)
    assert body["checked"] == len(ALL_CHECKS) and body["messages"] == []


def test_the_module_exports_exactly_the_seven_checks():
    assert [name for name in dir(component_checks) if name.startswith("check_")] == sorted(
        check.__name__ for check in ALL_CHECKS
    )
