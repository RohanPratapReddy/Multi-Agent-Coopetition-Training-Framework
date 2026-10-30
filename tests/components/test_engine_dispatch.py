"""T2, selection is configuration (document 17 section 13.3, 13.5)."""

from __future__ import annotations

import copy
import subprocess
import sys

import pytest

from mactf import conf
from mactf.exceptions import ImproperlyConfigured
from mactf.orchestration.dispatch import (
    DEFAULT_ENGINE, ENGINES, engine_factory, engine_for, engine_name, engine_of,
)


def test_engine_name_defaults_to_orchestrator():
    assert engine_name({}) == engine_name({"ORCHESTRATOR": {}}) == "orchestrator" == DEFAULT_ENGINE
    assert engine_name({"ORCHESTRATOR": {"ENGINE": None}}) == "orchestrator"
    assert engine_name({"ORCHESTRATOR": {"ENGINE": " Trainer "}}) == "trainer"
    assert engine_name({"ORCHESTRATOR": "not a mapping"}) == "orchestrator"
    assert ENGINES == ("orchestrator", "trainer")


@pytest.mark.parametrize("value", ["trianer", "", "orchestrator,trainer", 3, ["trainer"], "TRAINER2"])
def test_unknown_engine_is_e090(value):
    if value == "":
        assert engine_name({"ORCHESTRATOR": {"ENGINE": value}}) == "orchestrator"      # empty is "unset"
        return
    with pytest.raises(ImproperlyConfigured, match="E090"):
        engine_name({"ORCHESTRATOR": {"ENGINE": value}})
    values = copy.deepcopy(conf.DEFAULTS)
    values["ORCHESTRATOR"]["ENGINE"] = value
    with pytest.raises(ImproperlyConfigured, match="E090"):
        conf.validate_settings(conf.Settings(values, module="tests"))     # and again at settings validation


def test_engine_for_returns_the_class(settings):
    from mactf.orchestration.engine import TrainerEngine
    from mactf.orchestration.orchestrator import Orchestrator

    assert engine_for(settings) is Orchestrator
    settings["ORCHESTRATOR"]["ENGINE"] = "trainer"
    assert engine_for(settings) is TrainerEngine


def test_engine_of_reads_the_class_name_not_the_import():
    class TrainerEngine:
        pass

    class Derived(TrainerEngine):
        pass

    class Orchestrator:
        pass

    assert engine_of(TrainerEngine()) == engine_of(Derived()) == "trainer"
    assert engine_of(Orchestrator()) == engine_of(object()) == "orchestrator"


def test_the_factory_builds_the_configured_engine(settings):
    from mactf.orchestration.engine import TrainerEngine
    from mactf.orchestration.orchestrator import Orchestrator

    assert type(engine_factory(settings)) is Orchestrator
    settings["ORCHESTRATOR"]["ENGINE"] = "trainer"
    assert type(engine_factory(settings)) is TrainerEngine


def test_dispatch_is_torch_free():
    code = (
        "import sys\n"
        "from mactf.orchestration.dispatch import engine_name, ENGINES, Engine\n"
        "from mactf.orchestration import engine_factory, ENGINE_TRAINER\n"
        "assert engine_name({'ORCHESTRATOR': {'ENGINE': 'trainer'}}) == 'trainer'\n"
        "from mactf import conf; conf.validate_settings(conf.load_settings())\n"
        "assert 'torch' not in sys.modules, [m for m in sys.modules if m.startswith('torch')][:3]\n"
    )
    run = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]


def test_the_orchestrator_exports_resolve_lazily():
    import mactf.orchestration as orchestration

    assert orchestration.engine_name is engine_name and orchestration.ENGINE_TRAINER == "trainer"
    assert orchestration.TrainerEngine.__name__ == "TrainerEngine"
