"""T4, one run row per run (document 17 section 13.4.7, rows 16-17)."""

from __future__ import annotations

import copy
import json

import pytest

from engine_support import POINTS_KEYS, engine_of, fresh, tune

from mactf.orchestration.orchestrator import Orchestrator, RunConfig


def _strip(value):
    if isinstance(value, dict):
        return {key: _strip(item) for key, item in value.items() if key not in POINTS_KEYS}
    if isinstance(value, list):
        return [_strip(item) for item in value]
    return value


def _rows(registry, run_id):
    run = registry.get_run(run_id)
    events, _total = registry.rule_events(run_id)
    return {
        "run": {key: run[key] for key in ("status", "kappa", "entropy")},
        "settings": json.loads(run["settings"]),
        "episodes": [_strip(json.loads(json.dumps(item["scorecard"], default=str))) for item in registry.episodes(run_id)],
        "checkpoint_episodes": [item["episode"] for item in registry.checkpoints(run_id)],
        "aggregations": [item["report"] for item in registry.aggregations(run_id)],
        "events": [
            (e["episode"], e["peer"], e["rule_id"], e["magnitude"], e["severity"], e["evidence"], e["source"])
            for e in events
        ],
        "warnings": sorted(
            (w["peer"], w["rule_id"], w["episode"]) for w in registry.all_warnings(run_id)
        ),
    }


def test_run_rows_match_the_orchestrator(settings, tmp_path):
    """``runs``, ``episodes``, ``checkpoints``, ``aggregations``, ``rule_events``, ``warnings``: the same shapes and values."""
    tune(settings, tmp_path, epochs=2)
    orchestrator = Orchestrator(settings, modules=copy.deepcopy(fresh()))
    orchestrator.run(RunConfig(epochs=2, seed=1))
    settings["CHECKPOINT"]["DIR"] = str(tmp_path / "trainer-ck")
    engine = engine_of(settings, tmp_path)
    engine.run(RunConfig(epochs=2, seed=1))
    left, right = _rows(orchestrator.registry, orchestrator.run_id), _rows(engine.registry, engine.run_id)
    assert right["run"] == left["run"] and right["run"]["status"] == "completed"
    assert right["checkpoint_episodes"] == left["checkpoint_episodes"] == [0, 1]
    assert right["episodes"] == left["episodes"] and len(left["episodes"]) == 2
    assert right["aggregations"] == left["aggregations"] and len(left["aggregations"]) == 2
    assert right["events"] == left["events"] and left["events"]
    assert right["warnings"] == left["warnings"]
    assert {k: v for k, v in right["settings"].items() if k != "engine"} == left["settings"]


def test_the_run_row_names_its_engine(settings, tmp_path):
    """D-13.4: ``settings.engine`` is the only way to answer "which engine ran this?" from the registry."""
    tune(settings, tmp_path, epochs=1)
    engine = engine_of(settings, tmp_path)
    engine.run(RunConfig(epochs=1))
    assert json.loads(engine.registry.get_run(engine.run_id)["settings"])["engine"] == "trainer"
    orchestrator = Orchestrator(settings, modules=copy.deepcopy(fresh()))
    orchestrator.run(RunConfig(epochs=1))
    assert "engine" not in json.loads(orchestrator.registry.get_run(orchestrator.run_id)["settings"])


def test_finish_run_on_failure(settings, tmp_path):
    tune(settings, tmp_path)
    engine = engine_of(settings, tmp_path)
    config = RunConfig(epochs=2)
    engine.prepare(config)

    def explode(epoch):
        raise RuntimeError("the task source broke")

    engine.trainer.task_source = type("Broken", (), {"next": staticmethod(explode)})()
    with pytest.raises(RuntimeError, match="task source broke"):
        engine.run(config)
    row = engine.registry.get_run(engine.run_id)
    assert row["status"] == "failed" and "RuntimeError" in (row["notes"] or "")


def test_a_default_run_records_no_components_and_an_extension_does(settings, tmp_path, component_overlay):
    """14.6.4: the manifest the run starts with, from the same function the checkpoint uses; ``NULL`` for a default run."""
    from p3_examples import EmaSoftmax

    from mactf.components import register_weight_update

    tune(settings, tmp_path, epochs=1)
    plain = engine_of(settings, tmp_path / "plain")
    plain.run(RunConfig(epochs=1))
    assert plain.registry.components(plain.run_id) is None

    register_weight_update("ema_softmax", EmaSoftmax, version="1", scope="run")
    settings["TRAINER"]["WEIGHT_UPDATE"] = {"type": "ema_softmax", "params": {"alpha": 0.5}}
    settings["CHECKPOINT"]["DIR"] = str(tmp_path / "custom-ck")
    custom = engine_of(settings, tmp_path / "custom")
    custom.run(RunConfig(epochs=1))
    manifest = custom.registry.components(custom.run_id)
    assert manifest["weight_update"]["TRAINER.WEIGHT_UPDATE"]["name"] == "ema_softmax"
    assert manifest == custom.trainer._component_manifest()
