"""The ``TrainerEngine`` adapter: ``MACTFTrainer`` behind the ``Orchestrator`` surface.

Section 13 of document 17. Three rules govern every line below:

1. The trainer trains; the engine records. No gradient, loss, weight or temperature
   arithmetic happens here.
2. Everything that can fail fails in ``prepare``, before ``registry.start_run`` writes a
   ``running`` row, exactly as ``Orchestrator.prepare`` does (``orchestrator.py:91-177``).
3. Where the orchestrator and the trainer disagree about *bookkeeping*, the engine
   reproduces the orchestrator unless a documented switch says otherwise (13.8).
"""

from __future__ import annotations

import dataclasses
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from .. import signals
from ..conf_files import is_authored
from ..coopetition.aggregation import feasible_bounds
from ..db import Registry, ensure_registry
from ..exceptions import ContractError, ImproperlyConfigured
from ..runtime import Runtime, load_checkpoint, seed_everything
from ..sandbox.detect import build_spec, choose_provider
from ..scoring.normalize import CohortNormalizer
from ..training.callbacks import Callback
from ..training.trainer import MACTFTrainer
from .checkpoint import CheckpointRecord, should_rollback
from .checkpoint_v2 import CheckpointContext, CheckpointPayload, MACTFCheckpoint
from .dispatch import ENGINE_TRAINER
from .episode import EpisodeRunner
from .orchestrator import RunConfig, _default_modules, planned_epochs
from .phases import PhaseContext, bootstrap, stress_test

log = logging.getLogger("mactf.orchestration.engine")

__all__ = ["CumulativeSignal", "EnginePhaseBridge", "TrainerEngine"]


class TrainerEngine:
    """Drive one ``MACTFTrainer`` and record the run like ``Orchestrator`` does.

    ``trainer=`` exists for tests and for the API's preview path: a caller that already
    built a trainer (with its own hooks, callbacks and optimizer plan) hands it over and
    the engine only adds bookkeeping.

    ``engine_name`` is a class attribute so ``dispatch.engine_of`` and the checkpoint's
    ``identity["engine"]`` agree without importing this module.
    """

    engine_name = ENGINE_TRAINER

    def __init__(
        self,
        settings: Any,
        *,
        modules: Mapping[str, Any] | None = None,
        registry: Registry | None = None,
        trainer: MACTFTrainer | None = None,
        task_source: Any | None = None,
        logger: Any | None = None,
    ) -> None:
        self.settings = settings
        self.run_id = uuid.uuid4().hex[:12]
        self.registry: Registry = registry or ensure_registry(settings)
        self.trainer: MACTFTrainer | None = trainer
        # -- the surface, filled by prepare() ------------------------------
        self.scorer: Any = getattr(trainer, "scorer", None)
        self.checkpoints: MACTFCheckpoint | None = getattr(trainer, "checkpoint", None)
        self.peers: dict[str, Any] = dict(getattr(trainer, "peers", {}) or {})
        self.reference: Any = getattr(trainer, "reference", None)
        self.aggregator: Any = getattr(trainer, "aggregator", None)
        self.runner: EpisodeRunner | None = getattr(trainer, "runner", None)
        self.context: PhaseContext | None = None
        self.sandbox_choice: Any = None
        self.startup_notes: list[str] = []
        self.started_at = 0.0
        # -- inputs ---------------------------------------------------------
        self._modules: dict[str, Any] = dict(modules) if modules else {}
        self._task_source = task_source
        self._logger = logger
        # -- run state ------------------------------------------------------
        self._prepared_config: RunConfig | None = None
        self._source: Any = None
        self._bridge: EnginePhaseBridge | None = None
        self._signal: CumulativeSignal | None = None
        self._rollbacks: list[dict[str, Any]] = []
        self._bootstrap: dict[str, Any] = {}
        self._regions: list[str] = []

    # -- convenience ------------------------------------------------------
    @property
    def tree(self) -> Any:
        """The trainer's ``PeerTree``, or ``None`` before ``prepare``."""
        return getattr(self.trainer, "tree", None)

    def attach_logger(self, logger: Any) -> None:
        """Give the trainer the run logger ``RunManager`` created (``live.py:751-757``).

        Assigned after ``build()`` on purpose: the trainer reads ``self.logger`` in
        ``_observe`` (``trainer.py:1924``), so a logger attached at any point after
        construction is used from the next epoch on and the log file is closed by
        ``trainer.close()``.
        """
        self._logger = logger
        if self.trainer is not None:
            self.trainer.logger = logger

    def capabilities(self) -> Any:
        """The trainer's capability record, for a caller that wants it from the engine."""
        trainer = self._require_trainer()
        return trainer.capabilities()

    def close(self) -> None:
        """Flush and release the trainer (idempotent, like ``RunManager.shutdown``)."""
        if self.trainer is not None:
            self.trainer.close()

    def describe(self) -> dict[str, Any]:
        """What the engine is, for a log line and for the API's run detail."""
        trainer = self.trainer
        return {
            "engine": self.engine_name,
            "run_id": self.run_id,
            "built": bool(getattr(trainer, "built", False)),
            "nodes": len(self.peers),
            "checkpoint_schema": 2 if isinstance(self.checkpoints, MACTFCheckpoint) else None,
            "compat": self._compat,
            "rollbacks": len(self._rollbacks),
        }

    # -- construction ------------------------------------------------------
    def prepare(self, config: RunConfig | None = None) -> PhaseContext:
        """Resolve the runtime, build the trainer, and surface what a driver reads."""
        config = config or RunConfig()
        settings = self.settings

        if config.device:
            settings["DEVICE"] = config.device
        if config.seed is not None:
            settings["SEED"] = config.seed
        if config.run_id:
            self.run_id = config.run_id

        training = settings["TRAINING"]
        self._refuse_unsupported(config)

        trainer = self.trainer
        if trainer is None:
            if not self._modules:
                # Orchestrator.prepare seeds (build_runtime) before it builds the peers' modules; without
                # this the initial weights of the same SEED differed from process to process.
                seed_everything(int(settings["SEED"]), deterministic=bool(settings["DETERMINISTIC"]))
            modules = self._modules or _default_modules(settings)
            names = list(modules)
            if config.peers is not None:
                names = names[: config.peers]
            if not names:
                raise ImproperlyConfigured(
                    "no peers configured; set PEERS in settings, pass modules to the engine, "
                    "or hand it a built trainer"
                )
            models = {name: modules[name] for name in names}
            source = self._build_task_source(config)
            self._source = source
            trainer = MACTFTrainer(
                models=models,
                settings=settings,
                task_source=source,
                checkpoint=self._build_checkpoint_writer(),
                logger=self._logger,
                strict=True,
            )
            trainer.run_id = self.run_id
            trainer.build()
            self.trainer = trainer
        else:
            if not trainer.built:
                trainer.run_id = self.run_id
            else:
                self.run_id = trainer.run_id         # a built trainer's id is the run's id
            trainer.build()
            self._source = trainer.task_source
            self._refuse_unsupported(config)             # now the node count is known

        # The swarm the driver, the API and the phases read.
        self.scorer = trainer.scorer
        self.peers = trainer.peers
        self.reference = trainer.reference
        self.aggregator = trainer.aggregator
        self.runner = trainer.runner
        self.checkpoints = trainer.checkpoint
        feasible_bounds(
            len(trainer.active_tree.root_paths),
            floor=float(settings["COOPETITION"]["WEIGHT_FLOOR"]),
            cap=float(settings["COOPETITION"]["WEIGHT_CAP"]),
        )

        if self.sandbox_choice is None:                  # a second ``prepare`` keeps the first choice and its note
            self.sandbox_choice = self._choose_sandbox(config)
            if self.sandbox_choice is not None:
                self.startup_notes.append(self.sandbox_choice.describe())
        for note in trainer.failures:
            if note not in self.startup_notes:
                self.startup_notes.append(note)

        self.context = self._phase_context(trainer, config)
        self._install_hooks(trainer, self.context)
        self._regions = ["trainer:" + path for path in sorted(self.peers)]
        self._prepared_config = config
        return self.context

    def _refuse_unsupported(self, config: RunConfig) -> None:
        """Refuse a configuration the trainer cannot honour (``E092``), at prepare."""
        apps = list(self.settings.get("INSTALLED_APPS") or [])
        if any(str(app).startswith("mactf.distributed") for app in apps):
            raise ImproperlyConfigured(
                "E092: ORCHESTRATOR.ENGINE='trainer' cannot run with mactf.distributed "
                "installed: MACTFTrainer has no distributor bridge, so a worker cohort "
                "would be reported by the API and never used. Use ENGINE='orchestrator' "
                "for a distributed run."
            )
        if self.trainer is not None and self.trainer.built and config.peers is not None:
            if int(config.peers) != len(self.trainer.peers):
                raise ImproperlyConfigured(
                    f"E092: a pre-built trainer has {len(self.trainer.peers)} nodes but "
                    f"RunConfig.peers={config.peers}; build the trainer with the peers you mean"
                )

    @property
    def _compat(self) -> bool:
        """``ORCHESTRATOR.ENGINE_COMPAT``: reproduce the orchestrator's signal window."""
        value = self._block("ORCHESTRATOR").get("ENGINE_COMPAT", True)
        return bool(value)

    # -- mirrors of orchestrator.py:179-274, kept equal by test_engine_parity ------
    def _block(self, name: str) -> dict[str, Any]:
        """A settings block, or ``{}`` when absent or not a mapping (``orchestrator.py:179``)."""
        block = self.settings.get(name, {}) if hasattr(self.settings, "get") else {}
        return dict(block) if isinstance(block, dict) else {}

    def _task_name(self, config: RunConfig) -> str | None:
        """``RunConfig.task_source``, then ``ORCHESTRATOR.TASK_SOURCE``, then ``TRAINING.TASK``."""
        configured = self._block("ORCHESTRATOR").get("TASK_SOURCE")
        return (
            config.task_source
            or (str(configured) if configured else None)
            or self.settings["TRAINING"].get("TASK")
        )

    def _build_task_source(self, config: RunConfig) -> Any:
        """Resolve the task source with the same degradation rule the orchestrator uses.

        ``orchestrator.py:230-249``: a source that cannot be built from real data falls
        back to the synthetic generator with a note, because a run launched for another
        reason must not fail on a missing dataset directory.
        """
        from ..data.registry import build_source

        name = self._task_name(config) or "synthetic"
        try:
            return build_source(str(name), settings=self.settings)
        except Exception as exc:  # noqa: BLE001 - degraded, recorded, not silent
            message = f"task source {name!r} unavailable ({exc}); using synthetic"
            self.startup_notes.append(message)
            log.warning(message)
            return build_source("synthetic", settings=self.settings)

    def _choose_sandbox(self, config: RunConfig) -> Any:
        """``orchestrator.py:250-274``: choose a provider, and never fail a run over it."""
        del config  # the choice does not depend on the run config; kept for signature parity
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
            log.warning(message)
        build_spec(
            "swarm",
            settings=sandbox_settings,
            root=self.settings.path("ROOT"),
        )
        return choice

    def _build_checkpoint_writer(self) -> MACTFCheckpoint | None:
        """The v2 writer this run uses, or ``None`` when the project has no directory.

        ``CHECKPOINT.DIR`` (when authored) is resolved against ``ROOT``; otherwise
        ``PATHS["CHECKPOINTS"]``. The writer is given the registry, so every
        ``save`` also writes the ``checkpoints`` row the API lists
        (``checkpoint_v2.py:725-746``) -- the engine never calls ``record_checkpoint``
        itself, which is what keeps the two in step.
        """
        block = self._block("CHECKPOINT") if is_authored(self.settings, "CHECKPOINT") else {}
        directory = block.get("DIR")
        target: Path | None = None
        if directory:
            candidate = Path(str(directory)).expanduser()
            if not candidate.is_absolute():
                candidate = self.settings.path("ROOT") / candidate
            target = candidate
        try:
            return MACTFCheckpoint.from_settings(
                self.settings,
                run_id=self.run_id,
                directory=target,
                registry=self.registry,
            )
        except ImproperlyConfigured:
            if target is not None:
                raise  # the project named a directory; do not hide a bad path
            return None

    def _phase_context(self, trainer: MACTFTrainer, config: RunConfig) -> PhaseContext:
        """A live view over the trainer, in the shape the phase functions expect."""
        runtime = trainer.runtime
        if runtime is None:  # pragma: no cover - build() always resolves one
            raise ContractError("the trainer has no runtime after build()")
        training = dict(self.settings["TRAINING"])
        coopetition = dict(self.settings["COOPETITION"])
        return PhaseContext(
            run_id=self.run_id,
            epoch=0,
            peers=trainer.peers,
            reference=trainer.reference,
            scorer=trainer.scorer,
            runner=trainer.runner,
            runtime=runtime,
            metrics={
                "task_source": type(trainer.task_source).__name__,
                "engine": ENGINE_TRAINER,
            },
            source=trainer.task_source,
            signal_source=str(coopetition.get("SIGNAL_SOURCE", "final_score")).lower(),
            normalization=dict(coopetition.get("NORMALIZATION") or {}),
            training_settings=training
            | {"SEED": int(self.settings["SEED"]), "TASK": self._task_name(config)},
        )

    def _sync_context(self, trainer: MACTFTrainer) -> PhaseContext:
        """Re-point the view at the trainer's live containers before every epoch.

        A tree mutation (section 12) rebinds ``trainer.peers``, ``reference`` and
        ``runner``; the context the driver holds must follow, or ``mactf redteam``
        would score against the pre-mutation swarm.
        """
        context = self.context
        if context is None:  # pragma: no cover - prepare() sets it
            raise ContractError("_sync_context needs prepare() to have run")
        context.peers = trainer.peers
        context.reference = trainer.reference
        context.scorer = trainer.scorer
        context.runner = trainer.runner
        context.runtime = trainer.runtime or context.runtime
        context.source = trainer.task_source
        self.peers = trainer.peers
        self.reference = trainer.reference
        self.scorer = trainer.scorer
        self.runner = trainer.runner
        self.aggregator = trainer.aggregator
        return context

    def _require_trainer(self) -> MACTFTrainer:
        if self.trainer is None:
            raise ContractError("the engine has no trainer; call prepare() first")
        return self.trainer

    # -- execution ---------------------------------------------------------
    def run(
        self,
        config: RunConfig | None = None,
        *,
        should_stop: Callable[[], bool] | None = None,
        on_epoch: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Drive ``trainer.run_epoch`` to completion or to ``should_stop``.

        ``should_stop`` is polled at the epoch boundary only (T8), after the epoch's
        rows are durable, so a cancel never interrupts a torch operation and never
        leaves an optimizer half-stepped.
        """
        config = config or RunConfig()
        if self.context is not None and self._prepared_config is config:
            context = self.context
        else:
            context = self.prepare(config)
        trainer = self._require_trainer()
        epochs = planned_epochs(self.settings, config)

        self.started_at = time.time()
        runtime_info = context.runtime.describe()
        row_settings: dict[str, Any] = {
            "device": self.settings["DEVICE"],
            "epochs": epochs,
            # Additive and diagnostic: a run row written by ``Orchestrator`` has no
            # ``engine`` key, which is exactly how ``_check_identity`` reads it.
            "engine": self.engine_name,
        }
        self.registry.start_run(
            self.run_id,
            settings=row_settings,
            runtime=runtime_info,
            sandbox=self.sandbox_choice.report.as_dict() if self.sandbox_choice else None,
            notes="; ".join(self.startup_notes),
        )
        # Section 14.6: the manifest the run starts with, from the same function the checkpoint uses,
        # so the row and the file cannot disagree about the initial set. ``None`` for a default run.
        manifest = trainer._component_manifest()
        if manifest:
            try:
                self.registry.record_components(self.run_id, manifest, source="engine")
            except ImproperlyConfigured as error:       # an unmigrated registry: the file stays authoritative
                log.warning("could not record the component manifest of %s: %s", self.run_id, error)
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
            try:
                self.registry.finish_run(
                    self.run_id, status="failed", notes=f"{type(exc).__name__}: {exc}"[:500]
                )
            except Exception:  # noqa: BLE001 - never mask the original failure
                log.exception("could not mark run %s as failed", self.run_id)
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
        trainer = self._require_trainer()
        training = self.settings["TRAINING"]

        if config.dry_run:
            summary = {
                "run_id": self.run_id,
                "status": "dry_run",
                "runtime": runtime_info,
                "peers": list(self.peers),
                "sandbox": self.sandbox_choice.describe() if self.sandbox_choice else "none",
                "notes": list(self.startup_notes),
                "engine": self.engine_name,
            }
            self.registry.finish_run(self.run_id, status="dry_run")
            return summary

        start_epoch = 0
        if config.resume_from not in (None, False):
            start_epoch = self._resume(config)

        # The orchestrator's first phase, run for the same reason and by the same
        # function: the two engines must prime the swarm identically (T6, 13.8.2).
        self._bootstrap = dict(bootstrap(context, enabled=bool(training["BOOTSTRAP"])) or {})

        history: list[dict[str, Any]] = []
        cancelled = False
        completed_epochs = 0
        max_entropy = float(torch.log(torch.tensor(float(max(1, len(self.peers))))).item())

        for epoch in range(start_epoch, epochs):
            context.epoch = epoch
            self._sync_context(trainer)
            self._apply_boundary(trainer)

            start = time.time()
            reports_before = len(trainer.reports)
            try:
                trainer.run_epoch(epoch)
            except BaseException:
                # ``run_epoch`` dispatches ``on_epoch_end`` only on success, so the bridge
                # closes whatever the epoch left open here (T7).
                if self._bridge is not None:
                    self._bridge._close_all(epoch=epoch, failed=True)
                raise

            card = trainer.scorecards[-1]
            fresh = trainer.reports[reports_before:]
            report = fresh[-1] if fresh else None

            if config.redteam if config.redteam is not None else bool(training["REDTEAM"]):
                stress_test(context)

            score = sum(card.scores.values()) / max(1, len(card.scores))
            record = self._epoch_checkpoint(trainer, epoch)
            self._maybe_rollback(trainer, record, report, epoch, training, max_entropy)
            self._record_episode(epoch, card, report)

            report = self._epoch_report(trainer, report)
            entry = {
                "epoch": epoch,
                "duration": time.time() - start,
                "scorecards": card.to_dict(),
                "signal_by_peer": dict(trainer.signal_by_peer),
                "kappa": float(report.kappa) if report is not None else self._kappa(trainer),
                "entropy": float(report.entropy) if report is not None else self._entropy(trainer),
                "weights": dict(report.weights) if report is not None else {},
                "adjustments": list(report.adjustments) if report is not None else [],
                "score": score,
                "engine": self.engine_name,
            }
            history.append(entry)
            completed_epochs += 1
            log.info(
                "epoch %d kappa=%.4f entropy=%.4f score=%.4f",
                epoch,
                entry["kappa"],
                entry["entropy"],
                score,
            )
            if on_epoch is not None:
                on_epoch(entry)
            if should_stop is not None and should_stop():
                cancelled = True
                break

        status = "cancelled" if cancelled else "completed"
        self.registry.finish_run(
            self.run_id,
            status=status,
            kappa=self._kappa(trainer),
            entropy=self._entropy(trainer),
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
            "engine": self.engine_name,
            "bootstrap": dict(self._bootstrap),
            "rollbacks": list(self._rollbacks),
        }

    # -- small readers ------------------------------------------------------
    @staticmethod
    def _kappa(trainer: MACTFTrainer) -> float:
        controller = trainer.controller
        return float(controller.kappa) if controller is not None else 0.0

    @staticmethod
    def _entropy(trainer: MACTFTrainer) -> float:
        controller = trainer.controller
        return float(controller.entropy) if controller is not None else 0.0

    @staticmethod
    def _epoch_report(trainer: MACTFTrainer, report: Any) -> Any:
        """The report the epoch produced, or the trainer's cached one.

        ``_coordinate`` may legitimately return ``None`` (a strategy that does not
        aggregate this epoch, ``trainer.py:1843-1859``). The trainer still stores the
        last report in ``_epoch_report``; the engine prefers the epoch's own report and
        falls back to that, so the SSE ``epoch`` event keeps its ``kappa`` field.
        """
        if report is not None:
            return report
        return getattr(trainer, "_epoch_report", None)

    def _apply_boundary(self, trainer: MACTFTrainer) -> None:
        """Apply queued component swaps at the episode boundary (section 4.4)."""
        slots = getattr(trainer, "slots", None)
        apply_pending = getattr(slots, "apply_pending", None)
        if slots is None or not callable(apply_pending):
            return
        from ..components.kinds import Boundary

        # A scoring stage built for a swap needs the pipeline's configuration (it is offered as
        # ``inject``), and the pipeline must then adopt it: a slot that says one thing while the
        # chain runs another would put a false name in the checkpoint's manifest.
        scorer = getattr(trainer, "scorer", None)
        pipeline = getattr(scorer, "pipeline", scorer)
        offer = getattr(pipeline, "stage_inject", None)
        inject = offer(trusted=bool(getattr(trainer, "_components_trusted", True))) if callable(offer) else None
        for record in apply_pending(Boundary.EPISODE, inject=inject):
            log.info(
                "slot %s: %s -> %s at episode %d",
                record.slot, record.from_name, record.to_name, int(record.episode),
            )
        sync = getattr(pipeline, "sync_from_slots", None)
        if callable(sync) and any(path.startswith("scoring.") for path in slots.paths()):
            sync(slots)

    # -- checkpoint ---------------------------------------------------------
    def _epoch_checkpoint(self, trainer: MACTFTrainer, epoch: int) -> CheckpointRecord | None:
        """The record this epoch wrote, or ``None`` when the epoch wrote none.

        A run with ``TRAINING.CHECKPOINT_EVERY = 2`` writes on odd epochs only; the
        rollback check must not fire on an epoch that has no snapshot of its own.
        """
        writer = trainer.checkpoint
        listing = getattr(writer, "list", None)
        if writer is None or not callable(listing):
            return None
        records = listing(valid_only=False)
        record = records[-1] if records else None
        if record is None or int(record.episode) != int(epoch):
            return None
        return record

    def _previous_record(self, writer: MACTFCheckpoint, epoch: int) -> CheckpointRecord | None:
        """The newest valid record strictly older than ``epoch``."""
        older = [record for record in writer.list() if int(record.episode) < int(epoch)]
        return older[-1] if older else None

    def _maybe_rollback(
        self,
        trainer: MACTFTrainer,
        record: CheckpointRecord | None,
        report: Any,
        epoch: int,
        training: Mapping[str, Any],
        max_entropy: float,
    ) -> None:
        """``orchestrator.py:390-413``, with the trainer's full v2 restore.

        ``should_rollback`` is the same function and the same arguments as the
        orchestrator's, so the two engines roll back at the same epoch for the same
        reason. What they restore differs (D-13.2): the orchestrator's v1 restore has
        peers, the reference and the controller in it, while the trainer's v2 restore
        also has the optimizers, the RNG state and the data position -- which is what a
        run that continues after a rollback actually needs.
        """
        if record is None or report is None:
            return
        rollback, reason = should_rollback(
            entropy=report.entropy,
            max_entropy=max_entropy,
            collapse_ratio=float(training["ENTROPY_COLLAPSE_RATIO"]),
            fallback=report.fallback,
            enabled=bool(training["ROLLBACK_ON_ENTROPY_COLLAPSE"]),
        )
        if not rollback:
            return
        writer = trainer.checkpoint
        if not isinstance(writer, MACTFCheckpoint):
            return
        previous = self._previous_record(writer, epoch)
        if previous is None:
            self.startup_notes.append(
                f"rollback wanted at epoch {epoch} ({reason}) but no earlier checkpoint exists"
            )
            return
        payload = writer.load(previous.path)
        writer.restore(payload, trainer=trainer, strict=trainer.strict)
        signals.checkpoint_rolled_back.send(
            sender=self, reason=reason, episode=previous.episode
        )
        self.startup_notes.append(f"rollback at epoch {epoch}: {reason}")
        self._rollbacks.append(
            {"epoch": epoch, "restored": int(previous.episode), "reason": str(reason)}
        )
        log.warning("epoch %d rolled back to episode %d: %s", epoch, previous.episode, reason)

    # -- persistence --------------------------------------------------------
    def _record_episode(self, epoch: int, card: Any, report: Any) -> None:
        """``orchestrator.py:574-597``: the episode, the report, the events, the warnings."""
        context = self.context
        if context is None:  # pragma: no cover - prepare() sets it
            raise ContractError("_record_episode needs prepare() to have run")
        context.scorecards.append(card)
        trainer = self._require_trainer()
        if trainer.episodes:
            context.episodes.append(trainer.episodes[-1])
        if report is not None:
            context.reports.append(report)

        normalization = None
        recorded_report = report
        if context.signal_source == "points":
            # Kept bug-for-bug: the orchestrator replaces ``report`` with the
            # normalization report here and then writes it into the aggregation slot
            # (``orchestrator.py:577-582``). See Q-13.2.
            recorded_report = context.normalization_report()
            normalization = recorded_report.as_dict() if recorded_report is not None else None
        self.registry.record_episode(
            self.run_id, epoch, card.to_dict(), normalization=normalization
        )
        if recorded_report is not None:
            self.registry.record_aggregation(self.run_id, epoch, recorded_report)

        episode = trainer.episodes[-1] if trainer.episodes else None
        for peer in card.peers:
            ledger = (
                episode.participants[peer].ledger
                if episode is not None and peer in episode.participants
                else None
            )
            events = [
                {
                    "rule_id": event.rule_id,
                    "magnitude": event.magnitude,
                    "severity": event.resolved_severity().label,
                    "evidence": event.evidence,
                    "source": event.source,
                }
                for event in (ledger or ())
            ]
            self.registry.record_events(self.run_id, epoch, peer, events)
            self._persist_warnings(epoch, peer)

    def _persist_warnings(self, epoch: int, peer: str) -> None:
        """``orchestrator.py:599-619``: active and expired warning records."""
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

    # -- resume -------------------------------------------------------------
    def _resume(self, config: RunConfig) -> int:
        """``orchestrator.py:465-542`` for a v2 checkpoint and a live trainer."""
        trainer = self._require_trainer()
        writer = trainer.checkpoint
        if not isinstance(writer, MACTFCheckpoint):
            raise ImproperlyConfigured(
                "resume_from needs a writable CHECKPOINT directory; set CHECKPOINT.DIR or "
                "PATHS['CHECKPOINTS'], or pass a trainer with a checkpoint writer"
            )
        restored_episode: int | None = None
        restored_path: Path | None = None
        source = config.resume_from
        explicit: Path | None = None

        if source is True:
            configured = self._block("ORCHESTRATOR").get("RESUME_FROM")
            if configured:
                target = Path(str(configured)).expanduser()
                if not target.is_absolute():
                    target = self.settings.path("ROOT") / target
                if target.is_dir():
                    writer = MACTFCheckpoint.from_settings(
                        self.settings, run_id=self.run_id, directory=target,
                        registry=self.registry,
                    )
                    trainer.checkpoint = writer
                    self.checkpoints = writer
                    self.startup_notes.append(
                        f"resume directory from ORCHESTRATOR.RESUME_FROM: {target}"
                    )
                else:
                    source = target

        if source is True or source is None:
            record = writer.latest()
            if record is not None:
                restored_episode = int(record.episode)
                restored_path = Path(record.path)
            elif source is True:
                self.startup_notes.append(
                    f"W051: no valid checkpoint under {writer.directory}; the run starts at epoch 0"
                )
        else:
            explicit = Path(source)
            if explicit.is_dir():
                writer = MACTFCheckpoint.from_settings(
                    self.settings, run_id=self.run_id, directory=explicit,
                    registry=self.registry,
                )
                trainer.checkpoint = writer
                self.checkpoints = writer
                record = writer.latest()
                restored_episode = int(record.episode) if record is not None else None
                restored_path = Path(record.path) if record is not None else None
            elif explicit.exists():
                payload = self._read(writer, explicit)
                restored_episode = int(payload.identity.get("episode", 0))
                recorded_run = str(payload.identity.get("run_id", ""))
                if recorded_run and recorded_run != self.run_id:
                    # A branch: a new run continues another run's file. The identity check exists to
                    # stop a *resume* of the wrong run, so the file is adopted under this run's id.
                    payload = dataclasses.replace(payload, identity={**dict(payload.identity), "run_id": self.run_id})
                    self.startup_notes.append(f"branched from run {recorded_run} at episode {restored_episode}")
                self._restore(trainer, payload, where=explicit)
                self.startup_notes.append(f"resumed from {explicit.name}")
            else:
                raise ImproperlyConfigured(f"resume_from path does not exist: {explicit}")

        if (source is True or (explicit is not None and explicit.is_dir())) and restored_path is not None:
            payload = writer.load(restored_path)
            self._restore(trainer, payload, where=restored_path)
            self.startup_notes.append(f"resumed from episode {restored_episode}")

        restored_warnings = 0
        for peer in self.peers:
            restored_warnings += self.registry.restore_warnings(
                self.run_id, self.scorer.registry_for(peer), peer=peer
            )
        start_epoch = 0 if restored_episode is None else restored_episode + 1
        self.registry.record_resume_report(
            self.run_id,
            checkpoint_episode=restored_episode,
            started_epoch=start_epoch,
            warning_count=restored_warnings,
            report={
                "source": str(source),
                "warnings_restored": restored_warnings,
                "engine": self.engine_name,
                "notes": list(self.startup_notes),
            },
        )
        if restored_warnings:
            self.startup_notes.append(f"restored {restored_warnings} warning record(s)")
        return start_epoch

    @staticmethod
    def _read(writer: MACTFCheckpoint, path: Path) -> CheckpointPayload:
        """A verified v2 file, or a v1 (``Orchestrator``) file read as a v2 payload (no manifest to verify)."""
        raw = load_checkpoint(path)
        if isinstance(raw, Mapping) and not raw.get("manifest"):
            return CheckpointPayload.from_dict(MACTFCheckpoint._migrate_v1(raw))
        return writer.load(path)

    def _restore(self, trainer: MACTFTrainer, payload: Any, *, where: Path) -> None:
        """Check the engine identity (``E091``) and restore through the trainer.

        ``MACTFTrainer.load_state_dict`` restores in the v2 order and then applies the
        trainer-owned sections (strategy, callbacks, distillation) -- which is exactly
        what a resume of a customized run needs (section 14).
        """
        self._check_identity(payload, where=where)
        trainer.load_state_dict(payload, strict=trainer.strict)

    def _check_identity(self, payload: Any, *, where: Path) -> None:
        """Refuse to continue another engine's run unless the project opted in (**T5**).

        ``identity["engine"]`` is written by section 14. A checkpoint without the key was
        written by ``Orchestrator`` (whose ``CheckpointManager`` payloads have no
        identity at all), which is the documented default.
        """
        identity = (
            dict(payload.get("identity") or {})
            if isinstance(payload, Mapping)
            else dict(getattr(payload, "identity", {}) or {})
        )
        recorded = str(identity.get("engine") or "orchestrator").strip().lower()
        if recorded == self.engine_name:
            return
        allowed = bool(self._block("ORCHESTRATOR").get("ALLOW_ENGINE_SWITCH", False))
        if not allowed:
            raise ImproperlyConfigured(
                f"E091: {where} was written by the {recorded} engine, but this run uses "
                f"{self.engine_name!r}. Set ORCHESTRATOR.ALLOW_ENGINE_SWITCH: true to "
                "continue it under the other engine (the parameters, optimizers and "
                "warnings are compatible; the epoch bookkeeping is not resumed)."
            )
        self.startup_notes.append(
            f"W050: continued a {recorded} checkpoint under the {self.engine_name} engine"
        )

    def _install_hooks(self, trainer: MACTFTrainer, context: PhaseContext) -> None:
        """Attach the phase bridge once, and the compat signal window when enabled."""
        if self._bridge is None:
            self._bridge = EnginePhaseBridge(self)
            trainer.add_callback(self._bridge)
        if self._compat and self._signal is None:
            self._signal = CumulativeSignal(trainer, context.signal_source)
            trainer.add_callback(self._signal)


class CumulativeSignal(Callback):
    """The orchestrator's signal window, recomputed from every card the run produced.

    ``final_score`` mode reproduces ``PhaseContext.signal_by_peer``
    (``phases.py:77-84``) exactly; ``points`` mode reproduces ``_points_signal``
    (``phases.py:86-89``), which pools the raw ``final_points`` of every card recorded
    so far and normalizes the pooled means **once**, not once per epoch.
    """

    name = "engine_cumulative_signal"

    def __init__(self, trainer: MACTFTrainer, mode: str) -> None:
        self.mode = str(mode).lower()
        self.cards: list[Any] = []
        self.policy = trainer.normalizer.policy

    def values(self) -> dict[str, float]:
        """The cumulative signal map, in the mode the run uses."""
        if self.mode == "points":
            return self._points()
        totals: dict[str, list[float]] = {}
        for card in self.cards:
            for peer, peer_card in card.peers.items():
                totals.setdefault(peer, []).append(float(peer_card.final_score))
        return {peer: sum(values) / len(values) for peer, values in totals.items() if values}

    def _points(self) -> dict[str, float]:
        totals: dict[str, list[float]] = {}
        voids: dict[str, bool] = {}
        for card in self.cards:
            for peer, peer_card in card.peers.items():
                points = peer_card.points
                value = (
                    points.get("final_points")
                    if isinstance(points, Mapping)
                    else getattr(points, "final_points", None)
                )
                if value is None:
                    raise ContractError(
                        "COOPETITION.SIGNAL_SOURCE='points' needs every scorecard to carry "
                        f"points; peer {peer!r} has none (pass evidence to the scorer)"
                    )
                totals.setdefault(peer, []).append(float(value))
                voids[peer] = voids.get(peer, False) or bool(peer_card.is_void)
        means = {peer: sum(values) / len(values) for peer, values in totals.items() if values}
        if not means:
            return {}
        report = CohortNormalizer(self.policy).normalize(
            means, episode=len(self.cards) - 1, void=voids
        )
        return report.signals()

    def on_score(self, trainer: MACTFTrainer, card: Any) -> None:
        self.cards.append(card)
        trainer.signal_by_peer = self.values()
        return None


class EnginePhaseBridge(Callback):
    """Emit the phase signals from the trainer's hook points.

    A trainer epoch dispatches, in this order: ``on_epoch_start``, ``on_score`` (after
    the card exists), ``on_batch`` and ``on_step`` per node (inside the update),
    ``on_aggregate`` (after the group report exists), ``on_epoch_end``. The bridge maps
    the orchestrator's six phases onto those points, so the SSE ``phase`` event has the
    same names in the same order, and ``on_epoch_end`` closes anything still open --
    which is what makes **T7** true even when an epoch raises (``run_epoch`` dispatches
    ``on_epoch_end`` only on success, so the engine also closes in a ``finally``; see
    ``_close_phases``).
    """

    name = "engine_phase_bridge"

    def __init__(self, sender: Any) -> None:
        self.sender = sender
        self.open: list[str] = []

    # -- emission ---------------------------------------------------------
    def _start(self, phase: str, **payload: Any) -> None:
        signals.phase_started.send(sender=self.sender, phase=phase, **payload)
        self.open.append(phase)

    def _finish(self, phase: str, **payload: Any) -> None:
        if phase in self.open:
            self.open.remove(phase)
        signals.phase_finished.send(sender=self.sender, phase=phase, **payload)

    def _close_all(self, **payload: Any) -> None:
        for phase in list(reversed(self.open)):
            self._finish(phase, **payload)

    # -- hooks ------------------------------------------------------------
    def on_epoch_start(self, trainer: MACTFTrainer, epoch: int) -> None:
        self._start("debate", epoch=int(epoch))

    def on_score(self, trainer: MACTFTrainer, card: Any) -> None:
        # ``debate`` produced the ledgers; ``scoring`` produced this card; the update
        # that follows is ``local_update`` until the group report exists.
        self._finish("debate", episode=getattr(card, "episode", None))
        self._start("scoring")
        self._finish("scoring", scorecard=card)
        self._start("local_update")

    def on_aggregate(self, trainer: MACTFTrainer, report: Any) -> None:
        self._finish("local_update")
        self._start("aggregation")
        self._finish("aggregation", report=report)

    def on_epoch_end(self, trainer: MACTFTrainer, result: Any) -> None:
        # A strategy that aggregates nothing (``independent``) never dispatches
        # ``on_aggregate``; close the update here so no phase is left open.
        self._close_all(result=result)
