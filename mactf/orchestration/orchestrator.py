"""The orchestrator: builds the swarm, runs epochs, and records the run.

This is the equivalent of Django's ``runserver`` entry point: one object that
owns settings, peers, scorer, aggregator, sandbox choice, checkpoints, and the run
registry, so the CLI stays thin.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from .. import signals
from ..coopetition.aggregation import Aggregator, feasible_bounds
from ..coopetition.peer import Peer, build_peer, build_reference, require_same_layout
from ..coopetition.temperature import entropy_collapsed
from ..db import Registry, ensure_registry
from ..exceptions import ComplianceError, ImproperlyConfigured
from ..rl.loss import LossConfig
from ..runtime import Runtime, build_runtime
from ..sandbox.detect import build_spec, choose_provider
from ..scoring.pipeline import Scorer, build_scorer
from .checkpoint import CheckpointManager, should_rollback
from .episode import EpisodeRunner, Task, TaskSource
from .phases import (
    PhaseContext,
    aggregate,
    bootstrap,
    debate,
    local_update,
    score_episode,
    stress_test,
)

logger = logging.getLogger("mactf.orchestrator")


@dataclass
class RunConfig:
    """Per-run overrides applied on top of settings."""

    epochs: int | None = None
    device: str | None = None
    seed: int | None = None
    peers: int | None = None
    dry_run: bool = False
    redteam: bool | None = None
    resume_from: Path | None = None
    run_id: str | None = None


class Orchestrator:
    """Drives one training run to completion."""

    def __init__(
        self,
        settings: Any,
        *,
        modules: dict[str, Any] | None = None,
        task_source: Any | None = None,
        registry: Registry | None = None,
    ) -> None:
        self.settings = settings
        self.run_id = uuid.uuid4().hex[:12]
        self.registry = registry or ensure_registry(settings)
        self.scorer: Scorer = build_scorer(settings)
        self.checkpoints: CheckpointManager | None = None
        self.peers: dict[str, Peer] = {}
        self.reference: Peer | None = None
        self.aggregator: Aggregator | None = None
        self.runner: EpisodeRunner | None = None
        self.context: PhaseContext | None = None
        self.sandbox_choice: Any = None
        self.startup_notes: list[str] = []
        self._modules = modules or {}
        self._task_source = task_source
        self.started_at = 0.0

    # -- construction ----------------------------------------------------
    def prepare(self, config: RunConfig | None = None) -> PhaseContext:
        """Resolve runtime, build peers, and choose a sandbox provider."""
        config = config or RunConfig()
        settings = self.settings

        if config.device:
            settings["DEVICE"] = config.device
        if config.seed is not None:
            settings["SEED"] = config.seed
        runtime = build_runtime(settings)
        if config.run_id:
            self.run_id = config.run_id

        training = settings["TRAINING"]
        modules = self._modules or _default_modules(settings)
        names = list(modules)
        if config.peers is not None:
            names = names[: config.peers]
        if not names:
            raise ImproperlyConfigured(
                "no peers configured; set PEERS in settings or pass modules to the orchestrator"
            )

        for name in names:
            module = modules[name]
            # Deep-copy per peer so every swarm member starts independent.
            import copy

            instance = copy.deepcopy(module)
            self.peers[name] = build_peer(
                instance,
                name=name,
                runtime=runtime,
                learning_rate=float(training["LEARNING_RATE"]),
                action_dim=int(training["ACTION_DIM"]),
                state_dim=int(training["STATE_DIM"]),
            )
        require_same_layout(self.peers)

        first_peer = next(iter(self.peers.values()))
        self.reference = build_reference(first_peer, runtime=runtime)
        self.aggregator = Aggregator.from_settings(
            settings["COOPETITION"], num_peers=len(self.peers)
        )
        feasible_bounds(
            len(self.peers),
            floor=float(settings["COOPETITION"]["WEIGHT_FLOOR"]),
            cap=float(settings["COOPETITION"]["WEIGHT_CAP"]),
        )

        source = self._task_source or TaskSource(
            state_dim=int(training["STATE_DIM"]),
            action_dim=int(training["ACTION_DIM"]),
            batch_size=int(training["BATCH_SIZE"]),
            seed=int(settings["SEED"]),
        )
        self.runner = EpisodeRunner(runtime=runtime)

        self.checkpoints = CheckpointManager(
            settings.ensure_path("CHECKPOINTS"), run_id=self.run_id
        )
        self.sandbox_choice = self._choose_sandbox(config)
        if self.sandbox_choice is not None:
            self.startup_notes.append(self.sandbox_choice.describe())

        self.context = PhaseContext(
            run_id=self.run_id,
            epoch=0,
            peers=self.peers,
            reference=self.reference,
            scorer=self.scorer,
            runner=self.runner,
            runtime=runtime,
            metrics={"task_source": type(source).__name__},
        )
        self._source = source
        return self.context

    def _choose_sandbox(self, config: RunConfig) -> Any:
        sandbox_settings = self.settings["SANDBOX"]
        try:
            choice = choose_provider(
                requested=str(sandbox_settings.get("PROVIDER", "auto")),
                isolation=sandbox_settings.get("ISOLATION", "hardware"),
                settings=sandbox_settings,
            )
        except Exception as exc:  # noqa: BLE001 - sandbox is optional for training
            self.startup_notes.append(f"sandbox unavailable: {exc}")
            return None
        if choice.degraded:
            message = (
                f"hardware isolation requested but running at {choice.report.isolation.label}: "
                + "; ".join(choice.reasons)
            )
            self.startup_notes.append(message)
            signals.integrity_incident.send(sender=self, reason=message)
            logger.warning(message)
        spec = build_spec(
            "swarm",
            settings=sandbox_settings,
            root=self.settings.path("ROOT"),
        )
        return choice

    # -- execution -------------------------------------------------------
    def run(self, config: RunConfig | None = None) -> dict[str, Any]:
        config = config or RunConfig()
        context = self.prepare(config)
        assert self.aggregator is not None and self.reference is not None
        training = self.settings["TRAINING"]
        epochs = int(config.epochs or training["EPOCHS"])

        self.started_at = time.time()
        runtime_info = context.runtime.describe()
        self.registry.start_run(
            self.run_id,
            settings={"device": self.settings["DEVICE"], "epochs": epochs},
            runtime=runtime_info,
            sandbox=self.sandbox_choice.report.as_dict() if self.sandbox_choice else None,
            notes="; ".join(self.startup_notes),
        )
        try:
            return self._run_epochs(config, context, epochs, runtime_info)
        except BaseException as exc:
            # A run must never be left in the "running" state: the registry is the
            # audit trail that framework.md section 11.8 depends on.
            try:
                self.registry.finish_run(
                    self.run_id, status="failed", notes=f"{type(exc).__name__}: {exc}"[:500]
                )
            except Exception:  # noqa: BLE001 - never mask the original failure
                logger.exception("could not mark run %s as failed", self.run_id)
            raise

    def _run_epochs(
        self,
        config: RunConfig,
        context: PhaseContext,
        epochs: int,
        runtime_info: dict[str, Any],
    ) -> dict[str, Any]:
        assert self.aggregator is not None and self.reference is not None
        training = self.settings["TRAINING"]

        if config.dry_run:
            summary = {
                "run_id": self.run_id,
                "status": "dry_run",
                "runtime": runtime_info,
                "peers": list(self.peers),
                "sandbox": self.sandbox_choice.describe() if self.sandbox_choice else "none",
                "notes": list(self.startup_notes),
            }
            self.registry.finish_run(self.run_id, status="dry_run")
            return summary

        start_epoch = 0
        if config.resume_from is not None:
            payload = self.checkpoints.load()
            start_epoch = self.checkpoints.restore(
                payload,
                peers=self.peers,
                reference=self.reference,
                controller=self.aggregator.controller,
            ) + 1
            self.startup_notes.append(f"resumed from episode {start_epoch - 1}")

        bootstrap(context, enabled=bool(training["BOOTSTRAP"]))
        history: list[dict[str, Any]] = []

        for epoch in range(start_epoch, epochs):
            context.epoch = epoch
            start = time.time()
            task = self._source.next(epoch)
            episode = debate(context, task, rounds=int(training["CRITIQUE_ROUNDS"]))
            card = score_episode(context, episode)

            local_update(
                context,
                loss_config=LossConfig(
                    value_coef=float(training["VALUE_COEF"]),
                    kl_coef=float(training["KL_COEF"]),
                    entropy_coef=float(training["ENTROPY_COEF"]),
                    clip_epsilon=float(training["CLIP_EPSILON"]),
                ),
                max_grad_norm=float(training["MAX_GRAD_NORM"]),
            )

            report = aggregate(
                context,
                self.aggregator,
                reset_optimizers=bool(self.settings["COOPETITION"]["RESET_OPTIMIZERS"]),
                reference_ema=float(self.settings["COOPETITION"]["REFERENCE_EMA"]),
            )

            if config.redteam if config.redteam is not None else bool(training["REDTEAM"]):
                stress_test(context)

            score = sum(card.scores.values()) / max(1, len(card.scores))
            record = self._checkpoint(epoch, report, score)
            max_entropy = float(torch.log(torch.tensor(float(len(self.peers)))).item())
            rollback, reason = should_rollback(
                entropy=report.entropy,
                max_entropy=max_entropy,
                collapse_ratio=float(training["ENTROPY_COLLAPSE_RATIO"]),
                fallback=report.fallback,
                enabled=bool(training["ROLLBACK_ON_ENTROPY_COLLAPSE"]),
            )
            if rollback and record is not None:
                previous = self.checkpoints.latest()
                if previous is not None and previous.episode < epoch:
                    payload = self.checkpoints.load(previous.episode)
                    self.checkpoints.restore(
                        payload,
                        peers=self.peers,
                        reference=self.reference,
                        controller=self.aggregator.controller,
                    )
                    signals.checkpoint_rolled_back.send(
                        sender=self, reason=reason, episode=previous.episode
                    )
                    self.startup_notes.append(f"rollback at epoch {epoch}: {reason}")

            self._record_episode(epoch, card)
            entry = {
                "epoch": epoch,
                "duration": time.time() - start,
                "scorecards": card.to_dict(),
                "signal_by_peer": context.signal_by_peer(),
                "kappa": report.kappa,
                "entropy": report.entropy,
                "weights": report.weights,
                "adjustments": report.adjustments,
                "score": score,
            }
            history.append(entry)
            logger.info(
                "epoch %d kappa=%.4f entropy=%.4f score=%.4f",
                epoch,
                report.kappa,
                report.entropy,
                score,
            )

        self.registry.finish_run(
            self.run_id,
            status="completed",
            kappa=self.aggregator.controller.kappa,
            entropy=self.aggregator.controller.entropy,
        )
        return {
            "run_id": self.run_id,
            "status": "completed",
            "epochs": epochs,
            "runtime": runtime_info,
            "peers": list(self.peers),
            "history": history,
            "notes": list(self.startup_notes),
        }

    # -- persistence helpers ---------------------------------------------
    def _checkpoint(self, epoch: int, report: Any, score: float) -> Any:
        if self.checkpoints is None or self.aggregator is None:
            return None
        every = int(self.settings["TRAINING"]["CHECKPOINT_EVERY"])
        if every > 0 and (epoch + 1) % every != 0:
            return None
        assert self.context is not None
        record = self.checkpoints.save(
            episode=epoch,
            peers=self.peers,
            reference=self.reference,
            controller=self.aggregator.controller,
            warnings={
                peer: self.scorer.registry_for(peer).snapshot(peer) for peer in self.peers
            },
            runtime={"device": self.context.runtime.device_name},
            kappa=report.kappa,
            entropy=report.entropy,
            score=score,
        )
        self.registry.record_checkpoint(
            self.run_id,
            epoch,
            record.path,
            kappa=report.kappa,
            entropy=report.entropy,
            score=score,
        )
        return record

    def _record_episode(self, epoch: int, card: Any) -> None:
        assert self.context is not None
        self.registry.record_episode(self.run_id, epoch, card.to_dict())
        for peer, ledger in card.peers.items():
            events = [
                {
                    "rule_id": event.rule_id,
                    "magnitude": event.magnitude,
                    "severity": event.resolved_severity().label,
                    "evidence": event.evidence,
                    "source": event.source,
                }
                for event in self.context.episodes[-1].participants[peer].ledger
            ]
            self.registry.record_events(self.run_id, epoch, peer, events)


def _default_modules(settings: Any) -> dict[str, Any]:
    """Instantiate peers declared in ``PEERS`` settings."""
    from ..apps import get_peer_registry

    declared = settings.get("PEERS") or []
    if not declared:
        return {}
    registry = get_peer_registry()
    modules: dict[str, Any] = {}
    for entry in declared:
        if isinstance(entry, str):
            peer_class = registry.get(entry)
            modules[entry] = peer_class(
                state_dim=int(settings["TRAINING"]["STATE_DIM"]),
                action_dim=int(settings["TRAINING"]["ACTION_DIM"]),
            )
        elif isinstance(entry, dict):
            name = entry["name"]
            peer_class = registry.get(entry.get("class", name))
            kwargs = entry.get("kwargs", {})
            modules[name] = peer_class(**kwargs)
        else:
            raise ImproperlyConfigured(f"invalid PEERS entry: {entry!r}")
    return modules
