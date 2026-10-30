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
from ..conf_files import authored_keys, is_authored
from ..coopetition.aggregation import Aggregator, feasible_bounds
from ..coopetition.peer import Peer, build_peer, build_reference, require_same_layout
from ..coopetition.temperature import entropy_collapsed
from ..db import Registry, ensure_registry
from ..exceptions import ComplianceError, ImproperlyConfigured
from ..rl.loss import LossConfig
from ..runtime import Runtime, build_runtime, load_checkpoint
from ..sandbox.detect import build_spec, choose_provider
from ..scoring.pipeline import ScoringPipeline as Scorer
from ..scoring.pipeline import build_scorer
from .checkpoint import CheckpointManager, should_rollback
from .episode import EpisodeRunner, Task
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
    resume_from: Path | bool | None = None
    run_id: str | None = None
    #: Named task source, overriding ``TRAINING["TASK"]`` for this run.
    task_source: str | None = None


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
        self._prepared_config: RunConfig | None = None

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
                learning_rate=self._learning_rate(training),
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

        source = self._task_source or self._build_task_source(config)

        self.runner = EpisodeRunner(runtime=runtime)

        self.checkpoints = self._build_checkpoint_manager()
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
            source=source,
            signal_source=str(
                settings["COOPETITION"].get("SIGNAL_SOURCE", "final_score")
            ).lower(),
            normalization=dict(settings["COOPETITION"].get("NORMALIZATION") or {}),
            training_settings=dict(training)
            | {
                "SEED": int(settings["SEED"]),
                "TASK": self._task_name(config),
            },
        )
        if any(str(app).startswith("mactf.distributed") for app in (settings.get("INSTALLED_APPS") or [])):
            from ..distributed.coordinator import get_coordinator

            self.context.distributor = get_coordinator(settings)
        self._source = source
        self._prepared_config = config
        return self.context

    # -- document 15 bindings ---------------------------------------------
    def _block(self, name: str) -> dict[str, Any]:
        """A settings block, or ``{}`` when absent or not a mapping."""
        block = self.settings.get(name, {}) if hasattr(self.settings, "get") else {}
        return dict(block) if isinstance(block, dict) else {}

    def _learning_rate(self, training: dict[str, Any]) -> float:
        """``OPTIMIZER.LR`` when the project wrote it, else ``TRAINING.LEARNING_RATE``."""
        written = authored_keys(self.settings, "OPTIMIZER")
        if written and "LR" in written:
            configured = self._block("OPTIMIZER").get("LR")
            if configured is not None:
                return float(configured)
        return float(training["LEARNING_RATE"])

    def _loss_config(self, training: dict[str, Any]) -> LossConfig:
        """``LOSS`` wins as a whole block when written; otherwise ``TRAINING``."""
        loss = self._block("LOSS") if is_authored(self.settings, "LOSS") else {}
        source = loss or training
        return LossConfig(
            value_coef=float(source["VALUE_COEF"]),
            kl_coef=float(source["KL_COEF"]),
            entropy_coef=float(source["ENTROPY_COEF"]),
            clip_epsilon=float(source["CLIP_EPSILON"]),
        )

    def _task_name(self, config: RunConfig) -> str | None:
        """``RunConfig.task_source``, then ``ORCHESTRATOR.TASK_SOURCE``, then ``TRAINING.TASK``."""
        configured = self._block("ORCHESTRATOR").get("TASK_SOURCE")
        return (
            config.task_source
            or (str(configured) if configured else None)
            or self.settings["TRAINING"].get("TASK")
        )

    def _epochs(self, config: RunConfig) -> int:
        """``--epochs``, then ``ORCHESTRATOR.MAX_EPOCHS``, then ``TRAINING.EPOCHS``."""
        return planned_epochs(self.settings, config)

    def _build_checkpoint_manager(self) -> CheckpointManager:
        """Retention and directory from ``CHECKPOINT`` when the project wrote it."""
        block = self._block("CHECKPOINT") if is_authored(self.settings, "CHECKPOINT") else {}
        directory = block.get("DIR")
        if directory:
            target = Path(str(directory)).expanduser()
            if not target.is_absolute():
                target = self.settings.path("ROOT") / target
        else:
            target = self.settings.ensure_path("CHECKPOINTS")
        keep = int(block.get("KEEP", 5)) if block else 5
        return CheckpointManager(target, run_id=self.run_id, keep=keep)

    def _build_task_source(self, config: RunConfig) -> Any:
        """Resolve the configured or per-run task source.

        ``TRAINING["TASK"]`` names a registered source; ``RunConfig.task_source``
        overrides it for one run. A source that cannot be built from real data
        (for example ``jsonl`` with no dataset directory) falls back to the
        synthetic generator with a note, rather than failing a run that was
        launched for a different reason.
        """
        from ..data.registry import build_source

        name = self._task_name(config) or "synthetic"
        try:
            return build_source(str(name), settings=self.settings)
        except Exception as exc:  # noqa: BLE001 - degraded, recorded, not silent
            message = f"task source {name!r} unavailable ({exc}); using synthetic"
            self.startup_notes.append(message)
            logger.warning(message)
            return build_source("synthetic", settings=self.settings)

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
    def run(
        self,
        config: RunConfig | None = None,
        *,
        should_stop: Callable[[], bool] | None = None,
        on_epoch: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run the epoch loop to completion, or until ``should_stop`` says stop.

        ``should_stop`` is polled at each epoch boundary - after the epoch's work
        is recorded and before the next one starts - so a cancellation never
        interrupts a torch operation and never leaves optimizer state
        inconsistent. ``on_epoch`` receives the same ``entry`` dict appended to
        ``history``, which is what the SSE stream emits, so the CLI and the API
        share one event definition.
        """
        config = config or RunConfig()
        # A caller that already called ``prepare(config)`` (the run manager does,
        # so it can report ``queued -> running`` once preparation succeeds) must
        # not pay for building the swarm twice.
        if self.context is not None and self._prepared_config is config:
            context = self.context
        else:
            context = self.prepare(config)
        assert self.aggregator is not None and self.reference is not None
        training = self.settings["TRAINING"]
        epochs = self._epochs(config)

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
            return self._run_epochs(
                config,
                context,
                epochs,
                runtime_info,
                should_stop=should_stop,
                on_epoch=on_epoch,
            )
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
        *,
        should_stop: Callable[[], bool] | None = None,
        on_epoch: Callable[[dict[str, Any]], None] | None = None,
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
            start_epoch = self._resume(config)

        bootstrap(context, enabled=bool(training["BOOTSTRAP"]))
        history: list[dict[str, Any]] = []
        cancelled = False
        completed_epochs = 0

        for epoch in range(start_epoch, epochs):
            context.epoch = epoch
            start = time.time()
            task = self._source.next(epoch)
            episode = debate(context, task, rounds=int(training["CRITIQUE_ROUNDS"]))
            card = score_episode(context, episode)

            local_update(
                context,
                loss_config=self._loss_config(training),
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

            self._record_episode(epoch, card, report)
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
            completed_epochs += 1
            logger.info(
                "epoch %d kappa=%.4f entropy=%.4f score=%.4f",
                epoch,
                report.kappa,
                report.entropy,
                score,
            )
            if on_epoch is not None:
                on_epoch(entry)
            # The boundary is the only safe cancellation point: every tensor is
            # out of scope and the epoch's record is already durable. Checking
            # earlier would interrupt a torch op and desynchronize the optimizer.
            if should_stop is not None and should_stop():
                cancelled = True
                break

        status = "cancelled" if cancelled else "completed"
        self.registry.finish_run(
            self.run_id,
            status=status,
            kappa=self.aggregator.controller.kappa,
            entropy=self.aggregator.controller.entropy,
        )
        return {
            "run_id": self.run_id,
            "status": status,
            "epochs": epochs,
            "epochs_completed": completed_epochs,
            "runtime": runtime_info,
            "peers": list(self.peers),
            "history": history,
            "notes": list(self.startup_notes),
        }

    # -- resume ----------------------------------------------------------
    def _resume(self, config: RunConfig) -> int:
        """Restore from a checkpoint when one exists, always restoring warnings.

        ``resume_from`` accepts a path, ``True`` (auto-detect the latest
        checkpoint), or ``None`` (do not resume). Warning state is restored from
        the registry in every case, which is what makes a resume *without* a
        checkpoint keep its warnings - the behaviour doc 04.4 requires.
        """
        restored_epoch = None
        source = config.resume_from
        if source is True:
            # ORCHESTRATOR.RESUME_FROM: where a bare ``--resume`` looks.
            configured = self._block("ORCHESTRATOR").get("RESUME_FROM")
            if configured:
                target = Path(str(configured)).expanduser()
                if not target.is_absolute():
                    target = self.settings.path("ROOT") / target
                if target.is_dir():
                    self.checkpoints = CheckpointManager(target, run_id=self.run_id)
                    self.startup_notes.append(f"resume directory from ORCHESTRATOR.RESUME_FROM: {target}")
                else:
                    source = target
        if source is True or source is None:
            record = self.checkpoints.latest() if self.checkpoints else None
            explicit = None
            if record is not None:
                restored_epoch = record.episode
        else:
            explicit = Path(source)
            if explicit.is_dir():
                self.checkpoints = CheckpointManager(explicit, run_id=self.run_id)
                record = self.checkpoints.latest()
                restored_epoch = record.episode if record is not None else None
            elif explicit.exists():
                payload = load_checkpoint(explicit)
                restored_epoch = int(payload.get("episode", 0))
                self.checkpoints.restore(
                    payload,
                    peers=self.peers,
                    reference=self.reference,
                    controller=self.aggregator.controller if self.aggregator else None,
                )
                self.startup_notes.append(f"resumed from {explicit.name}")
            else:
                raise ImproperlyConfigured(f"resume_from path does not exist: {explicit}")

        if source is True and restored_epoch is not None and self.checkpoints is not None:
            payload = self.checkpoints.load(restored_epoch)
            self.checkpoints.restore(
                payload,
                peers=self.peers,
                reference=self.reference,
                controller=self.aggregator.controller if self.aggregator else None,
            )
            self.startup_notes.append(f"resumed from episode {restored_epoch}")

        restored_warnings = 0
        for peer in self.peers:
            restored_warnings += self.registry.restore_warnings(
                self.run_id, self.scorer.registry_for(peer), peer=peer
            )

        start_epoch = 0 if restored_epoch is None else restored_epoch + 1
        self.registry.record_resume_report(
            self.run_id,
            checkpoint_episode=restored_epoch,
            started_epoch=start_epoch,
            warning_count=restored_warnings,
            report={
                "source": str(source),
                "warnings_restored": restored_warnings,
                "notes": list(self.startup_notes),
            },
        )
        if restored_warnings:
            self.startup_notes.append(f"restored {restored_warnings} warning record(s)")
        return start_epoch

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

    def _record_episode(self, epoch: int, card: Any, report: Any = None) -> None:
        assert self.context is not None
        normalization = None
        if self.context.signal_source == "points":
            report = self.context.normalization_report()
            normalization = report.as_dict() if report is not None else None
        self.registry.record_episode(
            self.run_id, epoch, card.to_dict(), normalization=normalization
        )
        if report is not None:
            self.registry.record_aggregation(self.run_id, epoch, report)
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
            self._persist_warnings(epoch, peer)

    def _persist_warnings(self, epoch: int, peer: str) -> None:
        """Write this peer's warning state to the registry database.

        Both active and expired records are written: an expired warning is still
        part of the audit trail (framework.md W5), and its expiry is what a
        resumed run needs in order to derive the clean streak correctly.
        """
        registry = self.scorer.registry_for(peer)
        for rule_id, records in registry.all_active(peer).items():
            for record in records:
                self.registry.record_warning(self.run_id, peer, record)
        for rule_id, records in registry._warnings.get(peer, {}).items():
            for record in records:
                if record.active:
                    continue
                if record.expiry_episode > epoch:
                    continue
                self.registry.record_warning(self.run_id, peer, record)
                self.registry.expire_warning(
                    self.run_id, peer, rule_id, record.episode, expired_at=time.time()
                )


def planned_epochs(settings: Any, config: RunConfig | None = None) -> int:
    """The total epoch target: ``config.epochs``, then ``ORCHESTRATOR.MAX_EPOCHS``, then ``TRAINING.EPOCHS``.

    A module function so a command can compute the target without building an orchestrator.
    """
    block = settings.get("ORCHESTRATOR", {}) if hasattr(settings, "get") else {}
    configured = block.get("MAX_EPOCHS") if isinstance(block, dict) else None
    requested = config.epochs if config is not None else None
    return int(requested or configured or settings["TRAINING"]["EPOCHS"])


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
