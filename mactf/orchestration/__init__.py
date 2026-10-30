"""Orchestration: episodes, phases, checkpoints, and the run driver.

Public names resolve lazily (PEP 562) so ``import mactf.orchestration.recovery`` and its
siblings do not import ``torch``; ``from mactf.orchestration import Orchestrator`` still works.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._lazy import lazy_exports

_EXPORTS: dict[str, str] = {
    "CheckpointManager": "checkpoint", "CheckpointRecord": "checkpoint", "should_rollback": "checkpoint",
    "Episode": "episode", "EpisodeRunner": "episode", "PeerEpisode": "episode", "Task": "episode",
    "TaskSource": "episode", "consensus": "episode", "random_task": "episode",
    "Orchestrator": "orchestrator", "RunConfig": "orchestrator",
    "PhaseContext": "phases", "aggregate": "phases", "bootstrap": "phases", "debate": "phases",
    "local_update": "phases", "score_episode": "phases", "stress_test": "phases",
    # Document 17 section 13: which class runs the epochs.
    "Engine": "dispatch", "ENGINES": "dispatch", "ENGINE_ORCHESTRATOR": "dispatch",
    "ENGINE_TRAINER": "dispatch", "engine_factory": "dispatch", "engine_for": "dispatch",
    "engine_name": "dispatch", "engine_of": "dispatch",
    "TrainerEngine": "engine",
}

if TYPE_CHECKING:  # pragma: no cover - static analysers see the real imports
    from .checkpoint import CheckpointManager, CheckpointRecord, should_rollback
    from .episode import Episode, EpisodeRunner, PeerEpisode, Task, TaskSource, consensus, random_task
    from .dispatch import (
        ENGINES,
        ENGINE_ORCHESTRATOR,
        ENGINE_TRAINER,
        Engine,
        engine_factory,
        engine_for,
        engine_name,
        engine_of,
    )
    from .engine import TrainerEngine
    from .orchestrator import Orchestrator, RunConfig
    from .phases import PhaseContext, aggregate, bootstrap, debate, local_update, score_episode, stress_test

__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS, globals())
