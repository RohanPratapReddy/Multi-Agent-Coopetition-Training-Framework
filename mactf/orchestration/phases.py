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
from ..scoring.pipeline import ScoringPipeline as Scorer
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
    #: The run's task source, so multi-epoch phases (bootstrap) can iterate past
    #: the current epoch. ``None`` means "build a synthetic one from settings".
    source: Any = None
    #: The resolved ``TRAINING`` block, so phases need not re-read settings.
    training_settings: dict[str, Any] = field(default_factory=dict)

    #: A ``mactf.distributed`` coordinator, when that app is installed. The local-update and
    #: aggregation phases hand over to it only while it has live workers.
    distributor: Any = None

    #: ``COOPETITION.SIGNAL_SOURCE``: ``"final_score"`` or ``"points"`` (doc 11).
    signal_source: str = "final_score"
    #: The resolved ``COOPETITION.NORMALIZATION`` block (points source only).
    normalization: dict[str, Any] = field(default_factory=dict)

    def signal_by_peer(self, source: str | None = None) -> dict[str, float]:
        """Average each peer's signal across the epoch's episodes.

        Under ``final_score`` this is the mean ``final_score``. Under ``points`` it
        is the cohort-normalized mean ``final_points``, which requires every peer
        card to carry a points breakdown.
        """
        source = (source or self.signal_source or "final_score").lower()
        if source == "points":
            return self._points_signal()
        totals: dict[str, list[float]] = {}
        for card in self.scorecards:
            for peer, peer_card in card.peers.items():
                totals.setdefault(peer, []).append(peer_card.final_score)
        return {
            peer: sum(values) / len(values) for peer, values in totals.items() if values
        }

    def _points_signal(self) -> dict[str, float]:
        report = self.normalization_report()
        if report is None:
            return {}
        return {score.peer: score.signal for score in report.scores}

    def normalization_report(self) -> Any | None:
        """The cohort :class:`NormalizationReport` for this epoch, or ``None`` when
        no scorecard has been recorded. Requires every card to carry points."""
        from ..exceptions import ContractError
        from ..scoring.normalize import CohortNormalizer, NormalizationPolicy

        totals: dict[str, list[float]] = {}
        voids: dict[str, bool] = {}
        for card in self.scorecards:
            for peer, peer_card in card.peers.items():
                points = peer_card.points
                if not points or "final_points" not in points:
                    raise ContractError(
                        "COOPETITION.SIGNAL_SOURCE='points' needs every scorecard to carry "
                        f"points; peer {peer!r} has none (pass evidence to the scorer)"
                    )
                totals.setdefault(peer, []).append(float(points["final_points"]))
                voids[peer] = voids.get(peer, False) or bool(points.get("void", False))
        means = {peer: sum(values) / len(values) for peer, values in totals.items() if values}
        if not means:
            return None
        return CohortNormalizer(NormalizationPolicy.from_settings(self.normalization)).normalize(
            means, episode=self.epoch, void=voids
        )


# ---------------------------------------------------------------------------
# Phase 1: bootstrap
# ---------------------------------------------------------------------------
def bootstrap(context: PhaseContext, *, enabled: bool = True) -> dict[str, Any]:
    signals.phase_started.send(sender=context, phase="bootstrap")
    if not enabled:
        summary: dict[str, Any] = {"enabled": False, "epochs_run": 0}
        context.metrics["bootstrap"] = summary
        signals.phase_finished.send(sender=context, phase="bootstrap", summary=summary)
        return summary

    from .bootstrap import BootstrapConfig, BootstrapTrainer, build_source_from_settings

    training = dict(context.training_settings or {})
    raw_config = dict(training.get("BOOTSTRAP_CONFIG") or {})
    config = BootstrapConfig(
        epochs=int(raw_config.get("EPOCHS", 1)),
        batch_size=int(raw_config.get("BATCH_SIZE", training.get("BATCH_SIZE", 16))),
        learning_rate=float(raw_config.get("LEARNING_RATE", training.get("LEARNING_RATE", 1e-3))),
        weight_decay=float(raw_config.get("WEIGHT_DECAY", 0.0)),
        max_grad_norm=float(raw_config.get("MAX_GRAD_NORM", training.get("MAX_GRAD_NORM", 1.0))),
        update_reference=bool(raw_config.get("UPDATE_REFERENCE", True)),
        patience=int(raw_config.get("PATIENCE", 0)),
        batches_per_epoch=int(raw_config.get("BATCHES_PER_EPOCH", 0)),
    )
    source = context.source or build_source_from_settings(training)
    trainer = BootstrapTrainer(source=source, runtime=context.runtime, config=config)
    report = trainer.train(context.peers, context.reference)
    summary = report.as_dict() | {"enabled": True}
    context.metrics["bootstrap"] = summary
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
    evaluator: Any = None,
) -> Episode:
    """Collect proposals, run critique rounds, and return the populated episode.

    When ``evaluator`` is supplied, each participant's structured evidence is
    scored into its ledger after the task judge runs. Evidence families the
    episode cannot produce are passed empty, so the evaluators emit nothing for
    them rather than guessing.
    """
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

    if evaluator is not None:
        from ..scoring.evaluator import EvaluationSubject

        proposals = {
            name: dict(participant.proposal) for name, participant in episode.participants.items()
        }
        for peer_name, participant in episode.participants.items():
            subject = EvaluationSubject(
                peer=peer_name,
                episode=episode.index,
                task=task,
                proposal=dict(participant.proposal),
                peer_proposals={
                    name: proposal for name, proposal in proposals.items() if name != peer_name
                },
                claims=tuple(participant.proposal.get("claims", ()) or ()),
                suggestions=tuple(participant.proposal.get("suggestions", ()) or ()),
                actions=tuple(participant.proposal.get("actions", ()) or ()),
                flags=tuple(participant.proposal.get("flags", ()) or ()),
                corrections=tuple(participant.proposal.get("corrections", ()) or ()),
                trace=tuple(participant.proposal.get("trace", ()) or ()),
                metadata=dict(task.metadata),
            )
            evaluator.apply(participant.ledger, subject)

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
    evidence = None
    if getattr(context, "signal_source", "final_score") == "points":
        evidence = _default_evidence(episode)
    card = context.scorer.score_episode(episode.ledgers, episode=episode.index, evidence=evidence)
    episode.scorecard = card
    context.scorecards.append(card)
    context.episodes.append(episode)
    signals.phase_finished.send(sender=context, phase="scoring", scorecard=card)
    return card


def _default_evidence(episode: Episode) -> dict[str, Any]:
    """The only evidence the orchestrator itself observes: the task judge's verdict.

    ``task_checks_passed`` is derived from whether R1 was recorded for a participant.
    Everything else stays absent, so each guard reports the missing evidence instead
    of guessing; a caller with richer evidence scores through the evaluator instead.
    """
    from ..scoring.evidence import EpisodeEvidence

    participants = set(getattr(episode, "participants", ()) or ())
    return {
        peer: EpisodeEvidence(
            task_checks_passed=("R1" in ledger.triggered_ids()) if peer in participants else None
        )
        for peer, ledger in episode.ledgers.items()
    }


# ---------------------------------------------------------------------------
# Phase 4: local update
# ---------------------------------------------------------------------------
def run_local_step(
    peer: Peer,
    reference: Peer,
    states: torch.Tensor,
    signal: float,
    *,
    runtime: Runtime,
    void: bool = False,
    floor: float = 0.0,
    loss_config: LossConfig | None = None,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    """One gradient step for one peer; shared by the in-process loop and the distributed worker.

    ``signal`` is the peer's scalar score for the epoch (``final_score`` or points, per
    ``SIGNAL_SOURCE``); ``void`` penalises its advantage and ``floor`` is its mean compliance
    floor. Returns the loss breakdown as a plain dict.
    """
    loss_config = loss_config or LossConfig()
    peer.adapter.train()
    peer.zero_grad()
    with autocast(runtime):
        logits, values = peer.adapter(states)

    policy = make_policy(logits)
    with torch.no_grad():
        ref_logits, _ = reference.adapter(states)
    reference_policy = make_policy(ref_logits)
    actions = policy.probs.argmax(dim=-1)

    signal_tensor = torch.full(
        (states.shape[0],), float(signal), device=logits.device, dtype=logits.dtype
    )
    void_mask = torch.tensor([bool(void)] * int(states.shape[0]), device=logits.device)
    advantages = penalise_void_advantage(signal_tensor, void_mask, floor=-1.0)

    total_loss, breakdown = compute_loss(
        policy,
        reference_policy,
        actions,
        advantages,
        values,
        signal_tensor.unsqueeze(-1),
        config=loss_config,
        compliance_floor=torch.tensor(float(floor), device=logits.device),
        void_mask=void_mask,
    )

    if runtime.amp and runtime.grad_scaler is not None:
        runtime.grad_scaler.scale(total_loss).backward()
        runtime.grad_scaler.unscale_(peer.optimizer)  # type: ignore[arg-type]
        if max_grad_norm:
            clip_gradients(peer.adapter.parameters(), max_norm=max_grad_norm)
        runtime.grad_scaler.step(peer.optimizer)  # type: ignore[arg-type]
        runtime.grad_scaler.update()
    else:
        total_loss.backward()
        if max_grad_norm:
            clip_gradients(peer.adapter.parameters(), max_norm=max_grad_norm)
        peer.step()
    return breakdown.as_dict()


def local_update(
    context: PhaseContext,
    *,
    loss_config: LossConfig | None = None,
    max_grad_norm: float | None = None,
) -> dict[str, Any]:
    """One gradient step per peer against its own episode card."""
    from ..training_utils import state_targets_for

    loss_config = loss_config or LossConfig()
    distributor = context.distributor
    if distributor is not None and distributor.active():
        return distributor.local_update(context, loss_config=loss_config, max_grad_norm=max_grad_norm)
    signals.phase_started.send(sender=context, phase="local_update")
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

        statistics[peer_name] = run_local_step(
            peer,
            context.reference,
            states,
            float(signals_map[peer_name]),
            runtime=context.runtime,
            void=bool(peer_cards[0].is_void),
            floor=sum(card.floor for card in peer_cards) / max(1, len(peer_cards)),
            loss_config=loss_config,
            max_grad_norm=max_grad_norm,
        )

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
    distributor = context.distributor
    if distributor is not None and distributor.has_epoch(context):
        return distributor.aggregate_epoch(
            context, aggregator, reset_optimizers=reset_optimizers, reference_ema=reference_ema
        )
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
