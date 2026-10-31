"""Helpers for the built-in override suites (document 17 sections 4.3, 19.4 and 19.6).

Everything here registers inside a scoped overlay carrying ``allow_builtin_override``; the
process registry is asserted identical on exit, so an override can never leak into another test.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import torch

from mactf.components import HUB, overlay
from mactf.components.registry import HubPolicy

from p2_support import configure, make_trainer, snapshot_parameters, tune


@contextlib.contextmanager
def override_scope(*, allowed: bool = True):
    """An overlay in which replacing a built-in is (or is not) allowed; nothing leaks out."""
    before = HUB.snapshot()
    with overlay(policy=HubPolicy(allow_builtin_override=allowed)) as scope:
        yield scope
    assert HUB.snapshot() == before, "an override leaked into the process registry"


def settings_for(settings, *, epochs: int = 2, **blocks: Any):
    """Tuned settings; ``COMPONENTS.PLUGINS: off`` so loading imports nothing."""
    tune(settings, EPOCHS=epochs, CHECKPOINT_EVERY=0)
    configure(settings, COMPONENTS={"PLUGINS": "off"}, **blocks)
    return settings


def put(kind: str, name: str, factory: Any, **options: Any):
    """Replace-register ``factory`` as ``kind:name`` in the active overlay (no legacy W005)."""
    from mactf.components import register

    return register(kind, name, factory, replace=True, scope="run", **options)


def w001(caplog) -> list[str]:
    """The W001 lines captured so far, as ``kind:name`` strings."""
    found = []
    for record in caplog.records:
        message = record.getMessage()
        if message.startswith("W001 "):
            found.append(message.split()[1])
    return found


def built(trainer, path: str):
    """``(component, running instance)`` of a slot."""
    slot = trainer.slots.get(path)
    return slot.component, slot.instance


def weight_instance(trainer, label: str = "root"):
    return trainer.weight_plan.strategy_for(label, trainer.aggregator)


def run_snapshot(trainer, epochs: int = 2) -> dict[str, Any]:
    for epoch in range(epochs):
        trainer.run_epoch(epoch)
    return {
        "params": snapshot_parameters(trainer),
        "weights": [dict(report.weights) for report in trainer.reports],
        "scores": [
            {name: card.final_score for name, card in scorecard.peers.items()}
            for scorecard in trainer.scorecards
        ],
        "rng": torch.get_rng_state().clone(),
    }


def identity_run(settings, tmp_path, *, epochs: int = 2, seed: int = 7, peers: int = 2):
    trainer = make_trainer(settings, tmp_path, seed=seed, peers=peers).build()
    return trainer, run_snapshot(trainer, epochs)


def assert_same_run(left: dict[str, Any], right: dict[str, Any], *, rng: bool = True) -> None:
    assert left["params"].keys() == right["params"].keys()
    for key in left["params"]:
        assert torch.equal(left["params"][key], right["params"][key]), f"parameter {key} differs"
    assert left["weights"] == right["weights"]
    assert left["scores"] == right["scores"]
    if rng:
        assert torch.equal(left["rng"], right["rng"])


def quiet(caplog, level: int = logging.WARNING):
    return caplog.at_level(level, logger="mactf.components")


# -- step-level helpers (the guards' unit of work) ------------------------------------------------
def step_batch(*, void: bool = False, signal: float = 0.5, floor: float = 0.0) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(3)
    return {
        "states": torch.randn(8, 4, generator=generator),
        "targets": torch.zeros(8, dtype=torch.long),
        "signal": signal, "void": void, "floor": floor,
    }


def step(trainer, batch=None, path: str = "peer-1"):
    return trainer.train_step(trainer.tree.get(path), batch or step_batch())


def same_tensors(left, right) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)


def moments(trainer, path: str = "peer-1") -> dict[str, Any]:
    import copy

    return copy.deepcopy(trainer.optimizers[path].backend.state_dict())


def distillation_block(**overrides: Any) -> dict[str, Any]:
    view = {"kind": "fixed_random", "out_dim": 8}
    teacher = {"name": "t1", "kind": "module", "factory": "torch.nn.Linear",
               "kwargs": {"in_features": 4, "out_features": 3}, "view": dict(view),
               "trust": {"trust_init": 1.0}}
    value = {"TEACHERS": [teacher], "OUT_DIM": 8, "TAU": 0.5, "VIEWS": {"*": dict(view)}}
    value.update(overrides)
    return value


def moments_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Optimizer ``state_dict`` equality that compares tensors by value, not by truth value."""
    if left.keys() != right.keys():
        return False
    for key in left["state"]:
        if key not in right["state"] or left["state"][key].keys() != right["state"][key].keys():
            return False
        for name, value in left["state"][key].items():
            if not torch.equal(torch.as_tensor(value), torch.as_tensor(right["state"][key][name])):
                return False
    return left["state"].keys() == right["state"].keys()
