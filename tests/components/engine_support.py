"""Shared pieces of the section 13 tests: tuned settings, small modules, and a run comparison."""

from __future__ import annotations

import copy
import json
from typing import Any

import torch
import torch.nn as nn

__all__ = ["engine_of", "Tiny", "tune", "modules", "fresh", "same_tensors", "history_equal", "card_json", "SCENARIOS", "NO_CLIP"]


class Tiny(nn.Module):
    """The shape ``tests/training/test_trainer_equivalence.py`` uses."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(4, 16), nn.Tanh())
        self.policy = nn.Linear(16, 3)
        self.value = nn.Linear(16, 1)

    def forward(self, state):
        hidden = self.body(state)
        return self.policy(hidden), self.value(hidden)


#: ``rl.loss.clip_gradients`` consumes the ``parameters`` generator it is given before it clips, so
#: ``Orchestrator``'s clip is a no-op, while the trainer's facade clips for real. A run whose gradients
#: are never clipped is therefore the one on which the two engines can be bit-identical (P5 notes, D-13.8).
NO_CLIP = 1.0e12


def tune(settings, tmp_path, *, epochs: int = 3, clip: float | None = NO_CLIP, **orchestrator: Any):
    settings["TRAINING"].update(
        {"EPOCHS": epochs, "BATCH_SIZE": 8, "STATE_DIM": 4, "ACTION_DIM": 3, "BOOTSTRAP": False,
         "CHECKPOINT_EVERY": 1}
    )
    if clip is not None:
        settings["TRAINING"]["MAX_GRAD_NORM"] = clip
    settings["DEVICE"] = "cpu"
    settings["DETERMINISTIC"] = True
    settings["SANDBOX"].update({"PROVIDER": "local", "ALLOW_FALLBACK": True})
    settings["CHECKPOINT"]["DIR"] = str(tmp_path / "ck")
    settings["COOPETITION"]["DISAGREEMENT_MAX"] = 1.0
    settings["ORCHESTRATOR"]["ENGINE_COMPAT"] = True
    settings["ORCHESTRATOR"].update(orchestrator)
    return settings


def modules(count: int = 2) -> dict[str, nn.Module]:
    return {f"peer-{index + 1}": Tiny() for index in range(count)}


def fresh(count: int = 2, seed: int = 1337) -> dict[str, nn.Module]:
    """The same weights for every engine: one seeded build, deep-copied by the caller."""
    torch.manual_seed(seed)
    return modules(count)


def same_tensors(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert set(left) == set(right)
    for name, tensor in left.items():
        assert torch.equal(tensor, right[name]), f"{name} diverged"


#: The trainer derives points evidence beside ``final_score`` (``_episode_evidence``); ``Orchestrator``
#: passes none. Nothing in ``final_score`` depends on it (P5 notes, D-13.9), so a card comparison drops it.
POINTS_KEYS = ("points", "evidence_digest", "points_table_version")


def card_json(card: dict[str, Any]) -> str:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items() if key not in POINTS_KEYS}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return json.dumps(clean(card), sort_keys=True)


def history_equal(oracle: list[dict[str, Any]], actual: list[dict[str, Any]], *, keys=None) -> None:
    keys = keys or ("epoch", "kappa", "entropy", "weights", "score", "signal_by_peer")
    assert len(oracle) == len(actual)
    for expected, got in zip(oracle, actual):
        for key in keys:
            assert got[key] == expected[key], f"epoch {expected['epoch']}: {key}: {got[key]!r} != {expected[key]!r}"
        assert card_json(got["scorecards"]) == card_json(expected["scorecards"]), f"epoch {expected['epoch']}: scorecards"


#: Section 19.5. ``hierarchy`` cannot be configured from settings in this tree (nothing reads
#: ``TRAINER.HIERARCHY``), so ``Orchestrator``, which is flat, has no hierarchy to be compared with;
#: that scenario is compared between ``TrainerEngine`` and ``MACTFTrainer.fit`` (see the test).
SCENARIOS: dict[str, dict[str, Any]] = {
    "plain": {},
    "hierarchy": {},
    "mutating": {"TRAINER": {"MUTATION": {"ENABLED": True}}},
    "custom": {"RL": {"LOSS": {"type": "composite"}}},        # the default loss, spelled out
}


def engine_of(settings, tmp_path, *, engine: str = "trainer", count: int = 2, seed: int = 1337, **blocks):
    """A fresh engine over ``settings`` (already tuned), seeded so two engines start equal."""
    import copy as _copy

    from mactf.orchestration.dispatch import engine_factory

    settings["ORCHESTRATOR"]["ENGINE"] = engine
    for name, value in blocks.items():
        settings[name].update(_copy.deepcopy(value))
    torch.manual_seed(seed)
    return engine_factory(settings, modules=_copy.deepcopy(fresh(count, seed)))
