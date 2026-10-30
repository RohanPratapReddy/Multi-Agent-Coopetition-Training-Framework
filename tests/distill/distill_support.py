"""Shared harness for the distillation step tests: a real trainer, real peers, real teachers."""

from __future__ import annotations

import torch
import torch.nn as nn

from mactf.distill import CoDistillStep
from mactf.distill.teacher import TeacherSpec
from mactf.training import MACTFTrainer

VIEW = {"kind": "fixed_random", "out_dim": 8}


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(4, 16), nn.Tanh())
        self.policy = nn.Linear(16, 3)
        self.value = nn.Linear(16, 1)

    def forward(self, state):
        hidden = self.body(state)
        return self.policy(hidden), self.value(hidden)


def tune(settings) -> None:
    settings["TRAINING"].update(
        {"EPOCHS": 2, "BATCH_SIZE": 8, "STATE_DIM": 4, "ACTION_DIM": 3, "BOOTSTRAP": False}
    )
    settings["SANDBOX"]["PROVIDER"] = "local"


def teacher_spec(name: str = "t1", **overrides) -> dict:
    spec = {
        "name": name,
        "kind": "module",
        "factory": "torch.nn.Linear",
        "kwargs": {"in_features": 4, "out_features": 3},
        "view": dict(VIEW),
        "trust": {"trust_init": 1.0},
    }
    spec.update(overrides)
    return spec


def distillation_block(*teachers: dict, **overrides) -> dict:
    block = {
        "ENABLED": True,
        "TEACHERS": list(teachers) or [teacher_spec()],
        "OUT_DIM": 8,
        "TAU": 0.5,
        "VIEWS": {"*": dict(VIEW)},
    }
    block.update(overrides)
    return block


def build_trainer(settings, tmp_path, *, peers: int = 2, seed: int = 0, **kwargs) -> MACTFTrainer:
    tune(settings)
    torch.manual_seed(seed)
    trainer = MACTFTrainer(
        model=Tiny(),
        num_peers=peers,
        settings=settings,
        checkpoint_dir=tmp_path / "ckpt",
        **kwargs,
    )
    return trainer


def build_step(
    settings,
    tmp_path,
    *,
    peers: int = 2,
    seed: int = 0,
    teachers=None,
    managed: bool = True,
    register: bool = True,
    **kwargs,
) -> tuple[CoDistillStep, MACTFTrainer]:
    """A built (distillation-off) trainer plus a hand-attached ``CoDistillStep``."""
    trainer = build_trainer(settings, tmp_path, peers=peers, seed=seed).build()
    torch.manual_seed(seed + 100)
    kwargs.setdefault("views", {"*": dict(VIEW)})
    kwargs.setdefault("out_dim", 8)
    kwargs.setdefault("tau", 0.5)
    step = CoDistillStep(
        teachers=teachers if teachers is not None else [teacher_spec()],
        runtime=trainer._runtime,
        settings=settings,
        **kwargs,
    )
    if managed:
        step.attach(trainer)
    else:
        step._external_optimizers = trainer.optimizers
        step._shared_reference = trainer.reference
    # The trainer calls setup() for every peer before the first train_step; mirror it,
    # because the rendezvous only completes once every peer is known.
    for peer_path in sorted(trainer.peers) if register else ():
        step.setup(trainer.peers[peer_path])
    return step, trainer


def shared_batch(rows: int = 6, seed: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, 4, generator=generator)


def signals(trainer: MACTFTrainer, value: float = 0.0) -> dict[str, float]:
    return {path: float(value) for path in trainer.peers}


def parameters_of(module: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()]).clone()


def teachers_spec_object(name: str = "t1", **overrides) -> TeacherSpec:
    return TeacherSpec.from_spec(teacher_spec(name, **overrides))
