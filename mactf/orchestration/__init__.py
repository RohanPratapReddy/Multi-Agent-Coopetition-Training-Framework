"""Orchestration: episodes, phases, checkpoints, and the run driver."""

from __future__ import annotations

from .checkpoint import CheckpointManager, CheckpointRecord, should_rollback
from .episode import (
    Episode,
    EpisodeRunner,
    PeerEpisode,
    Task,
    TaskSource,
    consensus,
    random_task,
)
from .orchestrator import Orchestrator, RunConfig
from .phases import (
    PhaseContext,
    aggregate,
    bootstrap,
    debate,
    local_update,
    score_episode,
    stress_test,
)

__all__ = [
    "CheckpointManager",
    "CheckpointRecord",
    "Episode",
    "EpisodeRunner",
    "Orchestrator",
    "PeerEpisode",
    "PhaseContext",
    "RunConfig",
    "Task",
    "TaskSource",
    "aggregate",
    "bootstrap",
    "consensus",
    "debate",
    "local_update",
    "random_task",
    "score_episode",
    "should_rollback",
    "stress_test",
]
