"""Document 17 section 13.8 and 19.5: ``TrainerEngine`` reproduces ``Orchestrator``, epoch for epoch."""

from __future__ import annotations

import copy
import json

import pytest
import torch

from engine_support import NO_CLIP, SCENARIOS, Tiny, card_json, fresh, history_equal, same_tensors, tune

from mactf.exceptions import ImproperlyConfigured
from mactf.orchestration.dispatch import engine_factory, engine_name
from mactf.orchestration.engine import TrainerEngine
from mactf.orchestration.orchestrator import Orchestrator, RunConfig


@pytest.fixture()
def tuned(settings, tmp_path):
    return tune(settings, tmp_path)


def _apply(settings, blocks):
    for name, value in blocks.items():
        if isinstance(settings.get(name), dict) and isinstance(value, dict):
            for key, item in value.items():
                if isinstance(settings[name].get(key), dict) and isinstance(item, dict):
                    settings[name][key].update(item)
                else:
                    settings[name][key] = copy.deepcopy(item)
        else:
            settings[name] = copy.deepcopy(value)


def _run_both(settings, tmp_path, *, epochs=3, blocks=None, modules=None):
    modules = modules if modules is not None else fresh()
    _apply(settings, blocks or {})
    torch.manual_seed(1337)
    orchestrator = Orchestrator(settings, modules=copy.deepcopy(modules))
    oracle = orchestrator.run(RunConfig(epochs=epochs, seed=1337))
    settings["ORCHESTRATOR"]["ENGINE"] = "trainer"
    settings["CHECKPOINT"]["DIR"] = str(tmp_path / "trainer-ck")
    torch.manual_seed(1337)
    engine = TrainerEngine(settings, modules=copy.deepcopy(modules))
    summary = engine.run(RunConfig(epochs=epochs, seed=1337))
    return orchestrator, oracle, engine, summary


def test_engine_selection_is_configuration(tuned):
    assert engine_name(tuned) == "orchestrator"
    tuned["ORCHESTRATOR"]["ENGINE"] = "trainer"
    assert engine_name(tuned) == "trainer"
    assert isinstance(engine_factory(tuned), TrainerEngine)


def test_unknown_engine_is_refused(tuned):
    tuned["ORCHESTRATOR"]["ENGINE"] = "trianer"
    with pytest.raises(ImproperlyConfigured, match="E090"):
        engine_name(tuned)


def test_engine_rng_parity(tuned, tmp_path):
    """Both engines leave the same RNG state behind after ``prepare`` (13.8.1, last row)."""
    modules = fresh()
    torch.manual_seed(0)
    orchestrator = Orchestrator(tuned, modules=copy.deepcopy(modules))
    orchestrator.prepare(RunConfig(epochs=1))
    after_orchestrator = torch.get_rng_state().clone()

    torch.manual_seed(0)
    tuned["ORCHESTRATOR"]["ENGINE"] = "trainer"
    engine = TrainerEngine(tuned, modules=copy.deepcopy(modules))
    engine.prepare(RunConfig(epochs=1))
    assert torch.equal(torch.get_rng_state(), after_orchestrator)


def test_epoch_by_epoch_equivalence(tuned, tmp_path):
    orchestrator, oracle, engine, summary = _run_both(tuned, tmp_path)
    assert summary["status"] == oracle["status"] == "completed"
    assert len(summary["history"]) == len(oracle["history"]) == 3
    history_equal(oracle["history"], summary["history"])
    for name, peer in engine.peers.items():
        same_tensors({"p": peer.export_parameters()}, {"p": orchestrator.peers[name].export_parameters()})
    assert all(entry["engine"] == "trainer" for entry in summary["history"]) and summary["engine"] == "trainer"


def test_checkpoint_sections_agree(tuned, tmp_path):
    """The v1 and v2 writers hash the same tensors to the same digest (13.8.1)."""
    from mactf.orchestration.checkpoint_v2 import _HashWriter

    modules = fresh()
    torch.manual_seed(1337)
    orchestrator = Orchestrator(tuned, modules=copy.deepcopy(modules))
    orchestrator.run(RunConfig(epochs=1, seed=1337))
    v1 = orchestrator.checkpoints.load(orchestrator.checkpoints.latest().episode)

    torch.manual_seed(1337)
    tuned["ORCHESTRATOR"]["ENGINE"] = "trainer"
    tuned["CHECKPOINT"]["DIR"] = str(tmp_path / "trainer-ck")
    engine = TrainerEngine(tuned, modules=copy.deepcopy(modules))
    engine.run(RunConfig(epochs=1, seed=1337))
    writer = engine.trainer.checkpoint
    v2 = writer.load(writer.latest().path)

    assert set(v1["peers"]) == set(v2.models)
    for path, payload in v1["peers"].items():
        # v1 stores the adapter's state (keys prefixed ``module.``), v2 the module's own
        models = {key.removeprefix("module."): value for key, value in payload["adapter"].items()}
        stored = dict(v2.models[path])
        assert set(stored) == set(models) and stored, path
        assert _HashWriter.digest(stored) == _HashWriter.digest(models), (
            f"{path}: the two writers hash the same tensors differently"
        )
    assert v2.identity["engine"] == "trainer" and "identity" not in v1


def test_per_epoch_window_when_compat_is_off(tuned, tmp_path):
    """``ENGINE_COMPAT: false`` restores the trainer's documented per-epoch window (T6, both directions)."""
    modules = fresh()
    torch.manual_seed(1337)
    orchestrator = Orchestrator(tuned, modules=copy.deepcopy(modules))
    oracle = orchestrator.run(RunConfig(epochs=3, seed=1337))

    torch.manual_seed(1337)
    tuned["ORCHESTRATOR"].update({"ENGINE": "trainer", "ENGINE_COMPAT": False})
    tuned["CHECKPOINT"]["DIR"] = str(tmp_path / "trainer-ck")
    engine = TrainerEngine(tuned, modules=copy.deepcopy(modules))
    summary = engine.run(RunConfig(epochs=3, seed=1337))

    assert summary["history"][0]["weights"] == oracle["history"][0]["weights"]
    assert summary["history"][-1]["weights"] != oracle["history"][-1]["weights"]
    assert summary["history"][-1]["signal_by_peer"] != oracle["history"][-1]["signal_by_peer"]


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_orchestrator_trainer_equivalence(name, settings, tmp_path):
    """Section 19.5: the four scenarios, parameters by ``torch.equal`` and everything else by ``==``."""
    tune(settings, tmp_path)
    if name == "hierarchy":
        return _hierarchy_equivalence(settings, tmp_path)
    orchestrator, oracle, engine, summary = _run_both(settings, tmp_path, blocks=SCENARIOS[name])
    assert summary["status"] == oracle["status"] == "completed"
    history_equal(oracle["history"], summary["history"])
    for peer, other in zip(engine.peers.values(), orchestrator.peers.values()):
        assert torch.equal(peer.export_parameters(), other.export_parameters())
    # the RNG state after the whole run
    assert summary["rollbacks"] == []


def _hierarchy_equivalence(settings, tmp_path):
    """``Orchestrator`` is flat and nothing in settings builds a hierarchy, so the oracle for a tree is the
    trainer itself: ``TrainerEngine(trainer=...)`` only adds bookkeeping, and must not change a number."""
    from mactf.coopetition.tree import HierarchySpec, LevelSpec
    from mactf.training import MACTFTrainer

    def build(directory):
        torch.manual_seed(1337)
        spec = HierarchySpec(
            levels=(LevelSpec("sub", 2, model=Tiny()),), roots=2,
            root_models={"peer-1": Tiny(), "peer-2": Tiny()},
        )
        return MACTFTrainer(hierarchy=spec, settings=settings, checkpoint_dir=directory)

    direct = build(tmp_path / "direct")
    direct.fit(epochs=3)
    settings["ORCHESTRATOR"].update({"ENGINE": "trainer", "ENGINE_COMPAT": False})
    wrapped_trainer = build(tmp_path / "wrapped")
    engine = TrainerEngine(settings, trainer=wrapped_trainer)
    summary = engine.run(RunConfig(epochs=3, seed=1337))
    assert summary["status"] == "completed" and len(engine.peers) == 6
    for path, peer in wrapped_trainer.peers.items():
        assert torch.equal(peer.export_parameters(), direct.peers[path].export_parameters()), path
    last = summary["history"][-1]["weights"]
    assert last and abs(sum(last.values()) - 1.0) < 1e-6 and set(last) <= set(wrapped_trainer.peers)
    assert wrapped_trainer.tree is engine.tree and sorted(engine.tree.root_paths) == ["peer-1", "peer-2"]


def test_the_default_gradient_clip_is_the_only_divergence(settings, tmp_path):
    """Finding: ``clip_gradients`` exhausts its generator, so ``Orchestrator`` never clips (D-13.8).

    With the framework's default ``TRAINING.MAX_GRAD_NORM`` the engines agree to float tolerance and no
    better; without a clip they are bit-identical (the tests above). Both facts are pinned here, so the
    day the Orchestrator's clip is fixed this test says so.
    """
    tune(settings, tmp_path, clip=1.0)
    orchestrator, oracle, engine, summary = _run_both(settings, tmp_path)
    exact = all(
        torch.equal(peer.export_parameters(), orchestrator.peers[name].export_parameters())
        for name, peer in engine.peers.items()
    )
    assert not exact
    for name, peer in engine.peers.items():
        assert torch.allclose(peer.export_parameters(), orchestrator.peers[name].export_parameters(), atol=1e-5)
    assert [e["epoch"] for e in summary["history"]] == [e["epoch"] for e in oracle["history"]]


def test_the_trainer_derives_points_the_orchestrator_does_not(settings, tmp_path):
    """D-13.9: the only difference left between two scorecards is the points block beside ``final_score``."""
    tune(settings, tmp_path)
    _o, oracle, _e, summary = _run_both(settings, tmp_path, epochs=1)
    plain = json.dumps(oracle["history"][0]["scorecards"], sort_keys=True)
    derived = json.dumps(summary["history"][0]["scorecards"], sort_keys=True)
    assert plain != derived and card_json(oracle["history"][0]["scorecards"]) == card_json(summary["history"][0]["scorecards"])
    assert '"points"' in derived and '"points": null' in plain.replace('"points":null', '"points": null') or '"points"' not in plain
