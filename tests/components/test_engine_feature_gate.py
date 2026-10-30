"""T9, feature honesty (document 17 section 13.4.2, rows 31-32)."""

from __future__ import annotations

import pytest

from engine_support import engine_of, fresh, modules, tune

from mactf.exceptions import ImproperlyConfigured
from mactf.orchestration.engine import TrainerEngine
from mactf.orchestration.orchestrator import Orchestrator, RunConfig


def test_distributed_is_e092(settings, tmp_path):
    tune(settings, tmp_path)
    settings["INSTALLED_APPS"] = [*settings["INSTALLED_APPS"], "mactf.distributed"]
    settings["ORCHESTRATOR"]["ENGINE"] = "trainer"
    with pytest.raises(ImproperlyConfigured, match="E092.*distributed"):
        TrainerEngine(settings, modules=modules()).prepare(RunConfig(epochs=1))
    # the refusal comes before any row or model: nothing was built
    engine = TrainerEngine(settings, modules=modules())
    with pytest.raises(ImproperlyConfigured):
        engine.prepare(RunConfig(epochs=1))
    assert engine.trainer is None and engine.registry.get_run(engine.run_id) is None


def test_the_orchestrator_with_the_app_installed_is_unaffected(settings, tmp_path):
    tune(settings, tmp_path)
    settings["INSTALLED_APPS"] = [*settings["INSTALLED_APPS"], "mactf.distributed"]
    assert Orchestrator(settings, modules=modules()).prepare(RunConfig(epochs=1)).distributor is not None


def test_prebuilt_trainer_with_a_peer_count(settings, tmp_path):
    from mactf.training import MACTFTrainer

    tune(settings, tmp_path)
    settings["ORCHESTRATOR"]["ENGINE"] = "trainer"
    trainer = MACTFTrainer(models=fresh(2), settings=settings, checkpoint_dir=tmp_path / "ck")
    engine = TrainerEngine(settings, trainer=trainer)
    with pytest.raises(ImproperlyConfigured, match="E092.*RunConfig.peers=3"):
        engine.prepare(RunConfig(epochs=1, peers=3))
    summary = TrainerEngine(settings, trainer=MACTFTrainer(models=fresh(2), settings=settings, checkpoint_dir=tmp_path / "c2")).run(
        RunConfig(epochs=1, peers=2)
    )
    assert summary["status"] == "completed"
