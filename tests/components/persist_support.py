"""Shared pieces of the section 14 persistence tests: a customised trainer and its registrations."""

from __future__ import annotations

from typing import Any

import torch

from mactf.components import register_optimizer, register_scheduler, register_weight_update

from p2_support import Lion, configure, make_trainer, step_then_flat, tune
from p3_examples import EmaSoftmax

__all__ = ["register_extensions", "custom_trainer", "default_trainer", "params", "FRAMEWORK_IDENTITY_KEYS"]

#: The literal list of ``checkpoint_v2._collect`` before document 17. A future key must be a
#: decision, not a side effect (section 14.8.1).
FRAMEWORK_IDENTITY_KEYS = (
    "run_id", "episode", "global_step", "schema_version", "created_at", "torch_version",
    "python_version", "mactf_version", "hostname", "command_line", "device", "dtype", "amp",
    "mode", "score",
)

CUSTOM_TRAINER = {"WEIGHT_UPDATE": {"type": "ema_softmax", "params": {"alpha": 0.5}}}
CUSTOM_OPTIMIZER = {
    "TYPE": "Lion", "LR": 1e-3, "WEIGHT_DECAY": 0.0, "SCHEDULER": "step_then_flat",
    "SCHEDULER_KWARGS": {"drop_at": 3},
}


def register_extensions(*, version: str = "1") -> None:
    """Register the worked extensions in the current overlay (a fresh process's registry state)."""
    register_weight_update("ema_softmax", EmaSoftmax, version=version, scope="run")
    register_optimizer("Lion", Lion, version="1.0", scope="run")
    register_scheduler("step_then_flat", step_then_flat, version="1", scope="run")


def custom_trainer(settings, tmp_path, *, epochs: int = 4, run_id: str = "persist-run", seed: int = 3,
                   trainer_block: dict[str, Any] | None = None, optimizer: dict[str, Any] | None = None,
                   **kwargs):
    tune(settings, EPOCHS=epochs, CHECKPOINT_EVERY=1, DETERMINISTIC=True)
    configure(
        settings,
        TRAINER=CUSTOM_TRAINER if trainer_block is None else trainer_block,
        OPTIMIZER=CUSTOM_OPTIMIZER if optimizer is None else optimizer,
    )
    trainer = make_trainer(settings, tmp_path, seed=seed, peers=2, **kwargs)
    trainer.run_id = run_id
    return trainer


def default_trainer(settings, tmp_path, *, epochs: int = 2, run_id: str = "persist-run", seed: int = 3, **kwargs):
    tune(settings, EPOCHS=epochs, CHECKPOINT_EVERY=1, DETERMINISTIC=True)
    trainer = make_trainer(settings, tmp_path, seed=seed, peers=2, **kwargs)
    trainer.run_id = run_id
    return trainer


def params(trainer) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in trainer.export_parameters().items()}


# -- one extension of every kind ------------------------------------------------------------------
import math  # noqa: E402

from mactf.components import (  # noqa: E402
    register_aggregator, register_function, register_policy, register_reward_term, register_scorer,
)
from mactf.coopetition.aggregation import Aggregator  # noqa: E402
from mactf.scoring.penalty import PenaltyScorer  # noqa: E402
from mactf.scoring.reward_terms import RewardTerm  # noqa: E402
from mactf.training.policy import DefaultTrainPolicy  # noqa: E402

from p2_support import ClippedAdvantage  # noqa: E402


class SqrtTerm(RewardTerm):
    name = "sqrt_magnitude"

    def __init__(self, *, scale: float = 1.0):
        self.scale = scale

    def contribution(self, event, ctx):
        return self.scale * math.sqrt(event.magnitude)


class MyPenalty(PenaltyScorer):
    """A penalty stage that is the framework's, under another name."""


class MyPolicy(DefaultTrainPolicy):
    name = "my_policy"


class TrackedAggregator(Aggregator):
    def __init__(self, *, tag: str = "t", **kwargs) -> None:
        super().__init__(**kwargs)
        self.tag = tag

    def options(self):
        return {**super().options(), "tag": self.tag}


def register_every_kind() -> None:
    register_reward_term("sqrt_magnitude", SqrtTerm, version="1", scope="run")
    register_scorer("my_penalty", MyPenalty, version="1", scope="run")
    register_policy("my_policy", MyPolicy, version="1", scope="run")
    register_aggregator("tracked", TrackedAggregator, version="1", scope="run")
    register_function("advantage", "clipped", ClippedAdvantage, version="1", scope="run")
    register_extensions()


EVERY_KIND_BLOCKS: dict[str, Any] = {
    "SCORING": {"STAGES": {
        "reward": {"type": "composed_reward", "terms": [
            {"type": "severity_scaled", "weight": 1.0},
            {"type": "sqrt_magnitude", "weight": 0.5, "params": {"scale": 2.0}},
        ]},
        "penalty": "my_penalty",
    }},
    "TRAINER": {
        **CUSTOM_TRAINER,
        "POLICY": {"type": "my_policy"},
        "AGGREGATOR": {"type": "tracked", "params": {"tag": "k"}},
    },
    "RL": {"ADVANTAGE": {"type": "clipped", "params": {"clip": 2.0}}},
    "OPTIMIZER": CUSTOM_OPTIMIZER,
}
