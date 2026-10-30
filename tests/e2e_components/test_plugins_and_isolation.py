"""Entry-point plugins found by a real interpreter, repeated registration, and runs that must not see each other."""

from __future__ import annotations

import json
import threading

import pytest

from component_helpers import Policy
from e2e_support import BASE_FLAGS, manage
from mactf.components import HUB, Hub, overlay, register_weight_update
from mactf.exceptions import ComponentCollision
from p2_support import configure, make_trainer, tune
from p3_examples import EmaSoftmax

PLUGIN = '''
from mactf.components import register_weight_update
from mactf.coopetition.weights import WeightStrategy, WeightTraits, make_report


@register_weight_update("plug_uniform", version="1")
class PlugUniform(WeightStrategy):
    name = "plug_uniform"
    traits = WeightTraits()

    def apply(self, vectors, scores, *, kappa, level=0, eligible=None, confidence=None):
        names = sorted(vectors)
        flags = {n: bool(eligible.get(n, True)) if eligible else True for n in names}
        live = [n for n in names if flags[n]]
        return make_report({n: 1.0 / len(live) for n in live}, names, kappa=kappa, eligible=flags)
'''


@pytest.mark.subprocess
def test_a_plugin_installed_as_an_entry_point_is_found_listed_and_used(project, tmp_path, monkeypatch):
    site = tmp_path / "site"
    (site / "e2eplug-1.0.dist-info").mkdir(parents=True)
    (site / "e2eplug.py").write_text(PLUGIN)
    (site / "e2eplug-1.0.dist-info" / "METADATA").write_text("Metadata-Version: 2.1\nName: e2eplug\nVersion: 1.0\n")
    (site / "e2eplug-1.0.dist-info" / "entry_points.txt").write_text("[mactf.components]\ne2eplug = e2eplug\n")
    monkeypatch.setenv("PYTHONPATH", str(site))
    listed = manage(project, "components", "list", "--kind", "weight_update", "--json")
    assert listed.returncode == 0, listed.stderr
    rows = json.loads(listed.stdout)
    rows = rows.get("components", rows.get("items", rows)) if isinstance(rows, dict) else rows
    assert any(row["name"] == "plug_uniform" and row["origin"] == "plugin" for row in rows), rows
    done = manage(project, "run", "--epochs", "2", "--run-id", "plug", *BASE_FLAGS,
                  "--set", "ORCHESTRATOR.ENGINE=trainer", "--set", "TRAINER.WEIGHT_UPDATE=plug_uniform")
    assert done.returncode == 0, done.stderr
    off = manage(project, "run", "--epochs", "1", "--run-id", "plug2", *BASE_FLAGS,
                 "--set", "COMPONENTS.PLUGINS=off", "--set", "TRAINER.WEIGHT_UPDATE=plug_uniform")
    assert off.returncode != 0 and "Traceback" not in off.stderr and "TRAINER.WEIGHT_UPDATE" in off.stderr


def test_registering_the_same_name_twice_is_a_collision_not_a_silent_replace():
    """Hot reload is a non-goal (Q4): re-importing a module that registers by decorator is the documented ``E002``."""
    registry = Hub().registry("trainer_policy")
    registry.register("dup", Policy)
    for _ in range(3):
        with pytest.raises(ComponentCollision, match="E002"):
            registry.register("dup", type("Policy", (Policy,), {}))
    assert registry.get("dup").factory is Policy
    registry.register("dup", type("Policy", (Policy,), {}), replace=True)


def _counting(tag: str):
    class Counting(EmaSoftmax):
        calls = 0
        owner = tag

        def apply(self, *args, **kwargs):
            type(self).calls += 1
            return super().apply(*args, **kwargs)

    return Counting


def _run(settings, tmp_path, tag, epochs, results):
    from mactf import conf

    own = conf.load_settings("mactf_test_settings")
    with overlay():
        cls = _counting(tag)
        register_weight_update("probe", cls, version=tag, scope="run")
        tune(own, EPOCHS=epochs, CHECKPOINT_EVERY=1, DETERMINISTIC=True)
        configure(own, TRAINER={"WEIGHT_UPDATE": {"type": "probe", "params": {"alpha": 0.5}}})
        trainer = make_trainer(own, tmp_path / tag, seed=3, peers=2)
        trainer.run_id = f"iso-{tag}"
        trainer.fit(epochs=epochs)
        live = trainer.weight_plan.strategy_for("root", trainer.aggregator)
        assert type(live) is cls and live.owner == tag
        manifest = trainer._component_manifest()["weight_update"]
        assert [entry["version"] for entry in manifest.values()] == [tag]
        results[tag] = (cls.calls, epochs)
        trainer.close()


def test_two_runs_with_different_overlays_do_not_leak(settings, tmp_path):
    before = HUB.snapshot()
    results: dict[str, tuple] = {}
    _run(settings, tmp_path, "a", 2, results)
    _run(settings, tmp_path, "b", 3, results)
    assert results["a"][0] > 0 and results["b"][0] > results["a"][0]   # each run called its own class, more epochs, more calls
    assert HUB.snapshot() == before
    assert not HUB.registry("weight_update").has("probe")


def test_runs_in_two_threads_each_see_only_their_own_overlay(settings, tmp_path):
    before = HUB.snapshot()
    results: dict[str, tuple] = {}
    errors: list[BaseException] = []

    def go(tag, epochs):
        try:
            _run(settings, tmp_path, tag, epochs, results)
        except BaseException as error:                          # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=go, args=("t1", 2)), threading.Thread(target=go, args=("t2", 3))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=240)
    assert not errors, errors[0]
    assert results["t1"][0] > 0 and results["t2"][0] > results["t1"][0]
    assert HUB.snapshot() == before
