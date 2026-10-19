"""The five training phases, matching both source documents.

1. **Bootstrap** - dataset/rule priming (docx Phase 2, SFT-style).
2. **Debate** - peers propose, then critique, then revise (docx Phase 3 step 1).
3. **Scoring** - the meta-judge scores ledgers (docx Phase 3 step 2).
4. **Local update** - per-peer gradient descent (docx Phase 3 step 3).
5. **Aggregation** - reward-modulated global sync (docx Phase 3 step 4).
6. **Stress test** - optional red-team pass (docx Phase 4).
7. **Checkpoint** - persist or roll back (docx Phase 5).

Escalation of warnings and the non-offsettable floor are enforced inside phase 3,
so no phase can observe an unconstrained score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .. import signals
from ..coopetition.aggregation import AggregationReport, Aggregator
from ..coopetition.peer import Peer
from ..rl.advantage import penalise_void_advantage
from ..rl.loss import LossConfig, clip_gradients, compute_loss
from ..rl.policy import make_policy
from ..runtime import Runtime, autocast
from ..scoring.pipeline import Scorer
from ..scoring.scorecard import EpisodeScorecard
from .episode import Episode, EpisodeRunner, Task

PhaseHook = Callable[..., Any]


@dataclass
class PhaseContext:
    """Mutable state passed between phases."""

    run_id: str
    epoch: int
    peers: dict[str, Peer]
    reference: Peer
    scorer: Scorer
    runner: EpisodeRunner
    runtime: Runtime
    tasks: list[Task] = field(default_factory=list)
    episodes: list[Episode] = field(default_factory=list)
    scorecards: list[EpisodeScorecard] = field(default_factory=list)
    reports: list[AggregationReport] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def signal_by_peer(self) -> dict[str, float]:
        """Average each peer's final score across the epoch's episodes."""
        totals: dict[str, list[float]] = {}
        for card in self.scorecards:
            for peer, peer_card in card.peers.items():
                totals.setdefault(peer, []).append(peer_card.final_score)
        return {
            peer: sum(values) / len(values) for peer, values in totals.items() if values
        }


# ---------------------------------------------------------------------------
# Phase 1: bootstrap
# ---------------------------------------------------------------------------
def bootstrap(context: PhaseContext, *, enabled: bool = True) -> dict[str, Any]:
    signals.phase_started.send(sender=context, phase="bootstrap")
    summary: dict[str, Any] = {"enabled": enabled}
    if not enabled:
        signals.phase_finished.send(sender=context, phase="bootstrap", summary=summary)
        return summary

    # Priming is architectural here: peers start from the same weights and the
    # reference is a frozen snapshot, which is what rule memorization needs.
    signals.phase_finished.send(sender=context, phase="bootstrap", summary=summary)
    return summary


# ---------------------------------------------------------------------------
# Phase 2: debate
# ---------------------------------------------------------------------------
def debate(
    context: PhaseContext,
    task: Task,
    *,
    rounds: int = 1,
) -> Episode:
    """Collect proposals, run critique rounds, and return the populated episode."""
    from ..scoring.judge import TaskJudge

    signals.phase_started.send(sender=context, phase="debate")
    episode = Episode(index=len(context.episodes), task=task)
    context.runner.run(episode, context.peers)

    # The task judge records verifiable rules (R1, R2, R14). It never penalizes an
    # honest mistake, because framework.md penalizes dishonesty rather than error.
    judge = TaskJudge()
    targets = task.targets
    for peer_name, participant in episode.participants.items():
        judge.judge(
            participant.ledger,
            logits=participant.logits,
            targets=targets,
            peer=peer_name,
        )

    for round_index in range(max(0, rounds)):
        proposals = context.runner.proposals(episode)
        if len(proposals) < 2:
            break
        # Each peer sees the mean of the others as a critique signal. Reward credit
        # for considering it is recorded by the judge, not assumed here.
        for peer_name, participant in episode.participants.items():
            others = [tensor for name, tensor in proposals.items() if name != peer_name]
            if not others:
                continue
            critique = torch.stack([tensor.float() for tensor in others], dim=0).mean(0)
            participant.proposal["critique_round"] = round_index + 1
            participant.proposal["critique_mean_logit"] = float(critique.mean().item())

    signals.phase_finished.send(sender=context, phase="debate", episode=episode)
    return episode


# ---------------------------------------------------------------------------
# Phase 3: scoring
# ---------------------------------------------------------------------------
def score_episode(context: PhaseContext, episode: Episode) -> EpisodeScorecard:
    signals.phase_started.send(sender=context, phase="scoring")
    card = context.scorer.score_episode(episode.ledgers, episode=episode.index)
    episode.scorecard = card
    context.scorecards.append(card)
    context.episodes.append(episode)
    signals.phase_finished.send(sender=context, phase="scoring", scorecard=card)
    return card


# ---------------------------------------------------------------------------
# Phase 4: local update
# ---------------------------------------------------------------------------
def local_update(
    context: PhaseContext,
    *,
    loss_config: LossConfig | None = None,
    max_grad_norm: float | None = None,
) -> dict[str, Any]:
    """One gradient step per peer against its own episode card."""
    from ..training_utils import state_targets_for

    signals.phase_started.send(sender=context, phase="local_update")
    loss_config = loss_config or LossConfig()
    signals_map = context.signal_by_peer()
    statistics: dict[str, dict[str, float]] = {}

    for peer_name, peer in context.peers.items():
        peer_cards = [
            card.peers[peer_name] for card in context.scorecards if peer_name in card.peers
        ]
        if not peer_cards or peer_name not in signals_map:
            continue

        states, targets = state_targets_for(context)
        if states is None:
            continue

        peer.adapter.train()
        peer.zero_grad()
        with autocast(context.runtime):
            logits, values = peer.adapter(states)

        policy = make_policy(logits)
        with torch.no_grad():
            ref_logits, _ = context.reference.adapter(states)
        reference_policy = make_policy(ref_logits)
        actions = policy.probs.argmax(dim=-1)

        signal_tensor = torch.full(
            (states.shape[0],),
            float(signals_map[peer_name]),
            device=logits.device,
            dtype=logits.dtype,
        )
        void = torch.tensor(
            [bool(card.is_void) for card in peer_cards][:1] * int(states.shape[0]),
            device=logits.device,
        )
        advantages = penalise_void_advantage(signal_tensor, void, floor=-1.0)

        total_loss, breakdown = compute_loss(
            policy,
            reference_policy,
            actions,
            advantages,
            values,
            signal_tensor.unsqueeze(-1),
            config=loss_config,
            compliance_floor=torch.tensor(
                sum(card.floor for card in peer_cards) / max(1, len(peer_cards)),
                device=logits.device,
            ),
            void_mask=void,
        )

        if context.runtime.amp and context.runtime.grad_scaler is not None:
            context.runtime.grad_scaler.scale(total_loss).backward()
            context.runtime.grad_scaler.unscale_(peer.optimizer)  # type: ignore[arg-type]
            if max_grad_norm:
                clip_gradients(peer.adapter.parameters(), max_norm=max_grad_norm)
            context.runtime.grad_scaler.step(peer.optimizer)  # type: ignore[arg-type]
            context.runtime.grad_scaler.update()
        else:
            total_loss.backward()
            if max_grad_norm:
                clip_gradients(peer.adapter.parameters(), max_norm=max_grad_norm)
            peer.step()

        statistics[peer_name] = breakdown.as_dict()

    context.metrics["local_update"] = statistics
    signals.phase_finished.send(sender=context, phase="local_update", statistics=statistics)
    return statistics


# ---------------------------------------------------------------------------
# Phase 5: aggregation
# ---------------------------------------------------------------------------
def aggregate(
    context: PhaseContext,
    aggregator: Aggregator,
    *,
    reset_optimizers: bool = True,
    reference_ema: float = 0.05,
) -> AggregationReport:
    signals.phase_started.send(sender=context, phase="aggregation")
    signals_map = context.signal_by_peer()
    names = list(context.peers)
    if not names:
        raise RuntimeError("no peers to aggregate")

    scores = [signals_map.get(name, 0.0) for name in names]
    eligible = [not _peer_void(context, name) for name in names]
    confidence = [_peer_confidence(context, name) for name in names]

    report = aggregator.compute_weights(
        names, scores, eligible=eligible, confidence=confidence
    )
    context.reports.append(report)

    adapters = {name: context.peers[name].adapter for name in names}
    result = aggregator.aggregate(adapters, report, reference=context.reference.adapter)
    aggregator.apply(
        adapters,
        result,
        reference=context.reference.adapter,
        reference_ema=reference_ema,
        reset_optimizers=reset_optimizers,
        optimizers={name: peer.optimizer for name, peer in context.peers.items()},
    )
    context.metrics["aggregation"] = report.as_dict()
    signals.phase_finished.send(sender=context, phase="aggregation", report=report)
    return report


def _peer_void(context: PhaseContext, peer: str) -> bool:
    """Participation gate: exclude peers whose work was voided this epoch."""
    cards = [card.peers[peer] for card in context.scorecards if peer in card.peers]
    return bool(cards) and all(card.is_void for card in cards)


def _peer_confidence(context: PhaseContext, peer: str) -> float:
    """Confidence proxy: proportion of the epoch's episodes the peer completed."""
    total = len(context.scorecards) or 1
    present = sum(1 for card in context.scorecards if peer in card.peers)
    return present / total


# ---------------------------------------------------------------------------
# Phase 6: stress test
# ---------------------------------------------------------------------------
def stress_test(
    context: PhaseContext,
    *,
    injectors: list[Callable[[PhaseContext], None]] | None = None,
) -> dict[str, Any]:
    """Red-team pass. Injectors receive the context and may record rule events."""
    signals.phase_started.send(sender=context, phase="stress_test")
    results: dict[str, Any] = {"injectors": 0, "findings": []}
    for injector in injectors or []:
        name = getattr(injector, "__name__", repr(injector))
        try:
            injector(context)
            results["injectors"] += 1
            results["findings"].append({"injector": name, "status": "ok"})
        except Exception as exc:  # noqa: BLE001 - a failing injector is a finding
            results["findings"].append(
                {"injector": name, "status": "failed", "error": repr(exc)}
            )
    context.metrics["stress_test"] = results
    signals.phase_finished.send(sender=context, phase="stress_test", results=results)
    return results
