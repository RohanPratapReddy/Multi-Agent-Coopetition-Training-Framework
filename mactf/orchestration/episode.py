"""Episode construction: building a task batch and the peer ledgers for it."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .. import signals
from ..scoring.ledger import Ledger
from ..scoring.scorecard import EpisodeScorecard


@dataclass
class Task:
    """One unit of work for the swarm."""

    id: str
    states: torch.Tensor
    targets: torch.Tensor
    prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.states.shape[0])

    def describe(self) -> str:
        return f"task[{self.id}] batch={self.batch_size} target={tuple(self.targets.shape)}"


class TaskSource:
    """Generates tasks for an epoch.

    The default source is a deterministic synthetic classification problem so the
    framework is runnable without external data. Operators replace it by passing
    their own callable, which is the extension point Django users expect.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        batch_size: int,
        seed: int = 0,
        generator: Callable[[int, int, int, int], Task] | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.seed = seed
        self._generator = generator
        self._counter = 0

    def next(self, epoch: int) -> Task:
        self._counter += 1
        if self._generator is not None:
            return self._generator(epoch, self._counter, self.state_dim, self.action_dim)
        # Deterministic per (seed, epoch, counter) so reruns are reproducible.
        local = torch.Generator().manual_seed(
            hash((self.seed, epoch, self._counter)) % (2**31 - 1)
        )
        states = torch.randn(self.batch_size, self.state_dim, generator=local)
        # A learnable rule: the correct class is the argmax of a fixed projection.
        projection = torch.randn(self.state_dim, self.action_dim, generator=local)
        targets = (states @ projection).argmax(dim=-1)
        return Task(
            id=f"epoch{epoch}-task{self._counter}",
            states=states,
            targets=targets,
            prompt="synthetic classification",
            metadata={"source": "synthetic", "epoch": epoch},
        )


@dataclass
class PeerEpisode:
    """One peer's participation in an episode."""

    peer: str
    ledger: Ledger
    logits: torch.Tensor | None = None
    values: torch.Tensor | None = None
    actions: torch.Tensor | None = None
    proposal: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """An episode: one task, every peer's proposal, and the resulting ledgers."""

    index: int
    task: Task
    participants: dict[str, PeerEpisode] = field(default_factory=dict)
    scorecard: EpisodeScorecard | None = None
    started_at: float = 0.0

    def ledger_for(self, peer: str) -> Ledger:
        if peer not in self.participants:
            self.participants[peer] = PeerEpisode(peer=peer, ledger=Ledger(peer=peer))
        return self.participants[peer].ledger

    @property
    def ledgers(self) -> dict[str, Ledger]:
        return {
            peer: episode.ledger for peer, episode in self.participants.items()
        }

    def describe(self) -> str:
        names = ", ".join(self.participants) or "none"
        return f"episode[{self.index}] {self.task.describe()} participants=[{names}]"


class EpisodeRunner:
    """Runs the forward pass for every peer and collects proposals."""

    def __init__(self, *, runtime: Any, runner: Callable[..., Any] | None = None) -> None:
        self.runtime = runtime
        self._runner = runner

    def run(self, episode: Episode, peers: dict[str, Any], *, sample: bool = True) -> Episode:
        from ..rl.rollout import collect

        episode.started_at = episode.started_at or 0.0
        signals.episode_started.send(sender=self, episode=episode)

        if self._runner is not None:
            payload = self._runner(episode, peers, runtime=self.runtime)
            for peer, data in payload.items():
                participant = episode.participants.setdefault(
                    peer, PeerEpisode(peer=peer, ledger=Ledger(peer=peer))
                )
                participant.logits = data.get("logits")
                participant.values = data.get("values")
                participant.actions = data.get("actions")
                participant.proposal = data.get("proposal", {})
            return episode

        states = self.runtime.to(episode.task.states)
        rollout = collect(peers, states, runtime=self.runtime, sample=sample)
        for transition in rollout.transitions:
            peer = transition.metadata["peer"]
            participant = episode.participants.setdefault(
                peer, PeerEpisode(peer=peer, ledger=Ledger(peer=peer))
            )
            participant.logits = transition.logits
            participant.values = transition.values
            participant.actions = transition.actions
        return episode

    def proposals(self, episode: Episode) -> dict[str, torch.Tensor]:
        return {
            peer: participant.logits
            for peer, participant in episode.participants.items()
            if participant.logits is not None
        }


def consensus(proposals: dict[str, torch.Tensor]) -> torch.Tensor | None:
    """Average peer logits into a single group proposal.

    Used by the synthesis phase; it is not a scoring mechanism, because
    framework.md 9.6.4 forbids voting a claim into acceptance.
    """
    if not proposals:
        return None
    stacked = torch.stack([tensor.float() for tensor in proposals.values()], dim=0)
    return stacked.mean(dim=0)


def random_task(
    *, state_dim: int, action_dim: int, batch_size: int, seed: int | None = None
) -> Task:
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    states = torch.randn(batch_size, state_dim, generator=generator)
    targets = torch.randint(0, action_dim, (batch_size,), generator=generator)
    return Task(id=f"task-{seed}", states=states, targets=targets)
