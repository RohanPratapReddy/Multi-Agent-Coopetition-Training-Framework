"""The run driver: one start/bound/stop sequence for ``run``, ``redteam``, and ``runs resume``.

Document 16, sections 8.2.1 and 8.2.2. The driver owns the cross-process lock, the
interrupt flag, the per-run log, the live progress lines, and the translation of "how did
the loop end" into a registry status and an exit code. A command supplies a ``RunPlan``
(what to run) and gets a ``DriveResult`` (how it ended); it never touches the lock or the
signal handlers itself.

Importing this module imports no torch: the engine (``Orchestrator`` or ``TrainerEngine``, chosen by
``ORCHESTRATOR.ENGINE``) is imported inside ``execute``.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from ..exceptions import CommandError, ResourceNotFound, StateConflict
from ..orchestration.recovery import RESUMABLE_STATUSES, branch_checkpoint
from ..orchestration.runview import resolve_run_id
from .base import BaseCommand
from .durations import format_duration
from .exitcodes import ExitCode

RESUME_AUTO = "@auto"
_LEGACY_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Why a stop reached the loop, as ``StopFlag`` and the lock report it.
_SIGNAL_STOPS = ("timeout", "SIGINT", "SIGTERM", "SIGBREAK", "signal")


@dataclass(frozen=True)
class RunPlan:
    """What a run command will do, resolved before anything is locked or built."""

    run_id: str
    config: Any                   # RunConfig
    mode: str                     # "new", "resume", "branch"
    source_run: str | None        # the run resumed, or the run a branch was cut from
    source_episode: int | None    # the checkpoint episode training continues after
    epochs: int                   # the total epoch target


def _new_run_id() -> str:
    return uuid.uuid4().hex[:12]          # the same shape Orchestrator generates


def _registry(command: BaseCommand, *, must_exist: bool):
    if must_exist and not command.need_settings().database_path().exists():
        raise ResourceNotFound(
            f"no run registry at {command.settings.database_path()}; there is nothing to continue"
        )
    return command.registry_for_write()       # migrates an old schema, as run always has


def _require_unused(registry, run_id: str) -> None:
    row = registry.get_run(run_id)
    if row is not None:
        hint = (f"continue it with --resume {run_id} or choose another --run-id"
                if row["status"] in RESUMABLE_STATUSES else "choose another --run-id")
        raise StateConflict(f"run {run_id} already exists ({row['status']}); {hint}", run_id=run_id)


def plan_run(command: BaseCommand, options: Mapping[str, Any]) -> RunPlan:
    """Resolve the run options into a ``RunPlan``. Pure in the options and the registry."""
    from ..orchestration.orchestrator import RunConfig, planned_epochs

    settings = command.need_settings()
    redteam = True if options.get("redteam") else (False if options.get("no_redteam") else None)
    base = dict(
        epochs=options.get("epochs"), peers=options.get("peers"), device=options.get("device"),
        seed=options.get("seed"), dry_run=bool(options.get("dry_run")), redteam=redteam,
        task_source=options.get("task"),
    )
    planned = planned_epochs(settings, RunConfig(epochs=options.get("epochs")))
    run_id_option = options.get("run_id")
    resume = options.get("resume")
    branch = options.get("branch_from")

    if resume is not None:
        registry = _registry(command, must_exist=True)
        if resume == RESUME_AUTO or str(resume).lower() in _LEGACY_TRUTHY:
            wanted = run_id_option                       # None: the newest resumable run
        else:
            if run_id_option and run_id_option != resume:
                raise CommandError.usage(f"--resume {resume} and --run-id {run_id_option} name different runs")
            wanted = resume
        try:
            run_id = resolve_run_id(registry, wanted, statuses=RESUMABLE_STATUSES)
        except StateConflict as exc:
            raise StateConflict(
                f"{exc}; to continue from its checkpoint as a new run use --branch-from {wanted}",
                **exc.details,
            ) from None
        except ResourceNotFound:
            if wanted is None:
                raise ResourceNotFound("no paused or interrupted run to resume") from None
            raise
        latest = registry.latest_checkpoint(run_id)
        done = int(latest["episode"]) + 1 if latest is not None else 0
        if done >= planned:
            raise StateConflict(
                f"run {run_id} already reached epoch {done} of {planned}; pass --epochs larger than {done}",
                run_id=run_id,
            )
        config = RunConfig(**base, resume_from=True, run_id=run_id)
        return RunPlan(run_id, config, "resume", run_id,
                       int(latest["episode"]) if latest is not None else None, planned)

    if branch is not None:
        registry = _registry(command, must_exist=True)
        source_text, _, episode_text = str(branch).partition(":")
        source = resolve_run_id(registry, source_text or None)
        path, episode = branch_checkpoint(registry, source, episode_text or "latest")
        run_id = run_id_option or _new_run_id()
        _require_unused(registry, run_id)
        if episode + 1 >= planned:
            raise StateConflict(
                f"checkpoint {source}@{episode} is at epoch {episode + 1} of {planned}; "
                f"pass --epochs larger than {episode + 1}", run_id=source,
            )
        config = RunConfig(**base, resume_from=Path(path), run_id=run_id)
        return RunPlan(run_id, config, "branch", source, episode, planned)

    registry = command.registry_for_write()
    run_id = run_id_option or _new_run_id()
    _require_unused(registry, run_id)
    return RunPlan(run_id, RunConfig(**base, run_id=run_id), "new", None, None, planned)


@dataclass
class DriveResult:
    run_id: str
    status: str                       # completed, dry_run, paused, cancelled, failed
    exit_code: int
    summary: dict[str, Any]           # what Orchestrator.run returned (empty on failure)
    stopped_by: str | None = None     # SIGINT, SIGTERM, SIGBREAK, timeout, lock_lost
    log_file: Path | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)   # whatever ``after`` returned
    #: Notes already printed at start-up, so the closing lines do not repeat them.
    shown_notes: list[str] = field(default_factory=list)

    def payload(self, plan: RunPlan) -> dict[str, Any]:
        body = dict(self.summary)
        body.update({
            "run_id": self.run_id,
            "status": self.status,
            "mode": plan.mode,
            "source_run": plan.source_run,
            "source_episode": plan.source_episode,
            "stopped_by": self.stopped_by,
            "log_file": str(self.log_file) if self.log_file is not None else None,
            "error": self.error,
            "exit_code": self.exit_code,
        })
        return body


class RunDriver:
    """Drives one ``RunPlan`` to an end the operator can act on."""

    def __init__(self, command: BaseCommand) -> None:
        self.command = command
        self.settings = command.need_settings()
        self.out = command.out
        self._shown_notes: list[str] = []

    def execute(self, plan: RunPlan, *, timeout: float | None = None, on_interrupt: str = "pause",
                log_path: Path | str | None = None, log_enabled: bool = True,
                after: Callable[[Any], Mapping[str, Any] | None] | None = None,
                label: str = "run") -> DriveResult:
        from ..orchestration.dispatch import engine_factory
        from ..orchestration.runlock import RunLock, lock_directory
        from .interrupt import StopFlag

        lock = RunLock(lock_directory(self.settings), plan.run_id, owner="cli")
        with lock, StopFlag(timeout=timeout, announce=self.out.warn) as stop:
            engine = engine_factory(self.settings)      # ORCHESTRATOR.ENGINE decides (document 17 section 13)
            self._recheck(plan, engine.registry)
            engine.prepare(plan.config)           # raises before any row is written
            logger, log_file = self._open_log(plan, log_path, log_enabled)
            try:
                return self._drive(plan, engine, lock, stop, logger, log_file,
                                   on_interrupt, after, label)
            finally:
                if logger is not None:
                    logger.close()

    # -- the window between planning and locking --------------------------------------
    def _recheck(self, plan: RunPlan, registry: Any) -> None:
        """The plan was made before the lock; confirm nothing changed in between."""
        row = registry.get_run(plan.run_id)
        if plan.mode == "resume":
            if row is None or row["status"] not in RESUMABLE_STATUSES:
                raise StateConflict(
                    f"run {plan.run_id} is no longer resumable "
                    f"({row['status'] if row else 'missing'})", run_id=plan.run_id)
        elif row is not None:
            raise StateConflict(f"run {plan.run_id} already exists ({row['status']})", run_id=plan.run_id)

    # -- the per-run log ---------------------------------------------------------------
    def _open_log(self, plan: RunPlan, log_path: Path | str | None, enabled: bool):
        if not enabled:
            return None, None
        from ..training.logging import open_run_logger, run_log_path

        target = Path(log_path).expanduser() if log_path else run_log_path(self.settings, plan.run_id)
        return open_run_logger(self.settings, plan.run_id, path=target), target

    def _log(self, logger: Any, kind: str, message: str, *, run_id: str, epoch: int = 0,
             duration: float = 0.0, level: str = "info", **fields: Any) -> None:
        if logger is None:
            return
        from ..training.logging import LogRecord

        logger.log(LogRecord(kind=kind, run_id=run_id, epoch=epoch, wall_time=time.time(),
                             monotonic=time.monotonic(), duration=duration, level=level,
                             message=message, fields=fields))

    # -- the loop ----------------------------------------------------------------------
    def _drive(self, plan, engine, lock, stop, logger, log_file, on_interrupt, after, label):
        out = self.out
        run_id = plan.run_id
        registry = engine.registry
        runtime = engine.context.runtime.describe()
        peers = list(engine.peers)
        sandbox = engine.sandbox_choice.describe() if engine.sandbox_choice else "none"
        epochs = plan.epochs
        notes = list(engine.startup_notes)
        self._shown_notes = notes

        out.event({"event": "start", "run_id": run_id, "mode": plan.mode, "epochs": epochs,
                   "peers": peers, "device": runtime.get("device"), "dtype": runtime.get("dtype"),
                   "sandbox": sandbox, "source_run": plan.source_run,
                   "source_episode": plan.source_episode,
                   "log_file": str(log_file) if log_file else None},
                  text=f"{label} {run_id} [running] ({plan.mode})")
        for line in (
            f"  device   : {runtime.get('device')} dtype={runtime.get('dtype')}",
            f"  peers    : {', '.join(peers)}",
            f"  sandbox  : {sandbox}",
            f"  epochs   : {epochs}" + (f" (continuing after episode {plan.source_episode})"
                                        if plan.source_episode is not None else ""),
            *(f"  note     : {note}" for note in notes),
        ):
            out.info(line)
        out.debug(f"  lock     : {lock.path}")
        out.debug(f"  log      : {log_file or 'disabled'}")
        self._log(logger, "run", "started", run_id=run_id, mode=plan.mode, epochs=epochs,
                  peers=peers, device=runtime.get("device"), seed=self.settings.get("SEED"),
                  source_run=plan.source_run)

        began = time.monotonic()

        def on_epoch(entry: Mapping[str, Any]) -> None:
            weights = ", ".join(f"{peer}={weight:.3f}" for peer, weight in entry.get("weights", {}).items())
            out.event(
                {"event": "epoch", "run_id": run_id, "epoch": entry["epoch"], "epochs": epochs,
                 "score": entry["score"], "kappa": entry["kappa"], "entropy": entry["entropy"],
                 "duration": entry["duration"], "weights": entry.get("weights", {})},
                text=(f"  epoch {entry['epoch']}: score={entry['score']:.4f} kappa={entry['kappa']:.3f} "
                      f"entropy={entry['entropy']:.3f} [{weights}] {format_duration(entry['duration'])}"),
            )
            self._log(logger, "epoch", f"epoch {entry['epoch']}", run_id=run_id, epoch=int(entry["epoch"]),
                      duration=float(entry["duration"]), score=entry["score"], kappa=entry["kappa"],
                      entropy=entry["entropy"], weights=entry.get("weights", {}))

        def should_stop() -> bool:
            return stop.is_set() or lock.lost

        try:
            summary = engine.run(plan.config, should_stop=should_stop, on_epoch=on_epoch)
        except KeyboardInterrupt:
            # The second Ctrl-C. Orchestrator.run already wrote the row "failed"; the run is not
            # failed, its last epoch is on disk, so say so.
            registry.set_run_status(run_id, "interrupted", notes="aborted by a second interrupt")
            self._log(logger, "run", "aborted", run_id=run_id, level="warning", reason="second interrupt")
            raise
        except Exception as exc:  # noqa: BLE001 - the row is already "failed"; report it
            if self.command.traceback_wanted:
                raise
            error = f"{type(exc).__name__}: {exc}"
            self._log(logger, "run", "failed", run_id=run_id, level="error", error=error)
            out.error(f"training failed: {error}")
            return DriveResult(run_id, "failed", int(ExitCode.FAILURE), {}, log_file=log_file, error=error)

        return self._settle(plan, engine, summary, lock, stop, logger, log_file,
                            on_interrupt, after, began)

    def _settle(self, plan, engine, summary, lock, stop, logger, log_file, on_interrupt, after, began):
        registry = engine.registry
        run_id = plan.run_id
        status = str(summary["status"])
        history = summary.get("history") or []
        last = int(history[-1]["epoch"]) if history else None
        stopped_by: str | None = None
        exit_code = int(ExitCode.OK)

        if status == "cancelled":                 # Orchestrator's word for "should_stop said stop"
            if lock.lost:
                stopped_by = "lock_lost"
            elif stop.timed_out:
                stopped_by = "timeout"
            elif stop.is_set():
                stopped_by = stop.signal_name or "signal"

            if stopped_by in _SIGNAL_STOPS and last is not None and last >= plan.epochs - 1:
                registry.finish_run(run_id, status="completed")      # stop arrived after the last epoch
                status, stopped_by = "completed", None
            elif stopped_by == "lock_lost":
                exit_code = int(ExitCode.CONFLICT)                  # another process owns the run now
            elif stopped_by == "timeout":
                registry.set_run_status(run_id, "paused", notes=f"timeout after epoch {last}", finished=False)
                status, exit_code = "paused", int(ExitCode.TIMEOUT)
            elif stopped_by is not None:
                exit_code = int(ExitCode.INTERRUPTED)
                if on_interrupt == "pause":
                    registry.set_run_status(run_id, "paused", notes=f"{stopped_by} after epoch {last}",
                                            finished=False)
                    status = "paused"
        summary["status"] = status

        self._log(logger, "run", "finished", run_id=run_id, status=status, stopped_by=stopped_by,
                  epochs_completed=summary.get("epochs_completed", 0),
                  duration=time.monotonic() - began)
        result = DriveResult(run_id, status, exit_code, summary, stopped_by=stopped_by, log_file=log_file,
                             shown_notes=list(self._shown_notes))
        if after is not None and status in ("completed", "dry_run"):
            result.extra = dict(after(engine) or {})
        return result


def present_run_end(command: BaseCommand, plan: RunPlan, result: DriveResult) -> None:
    """Print how a run ended. A module function: ``runs resume`` ends the same way as ``run``."""
    out = command.out
    payload = result.payload(plan)
    if out.fmt == "jsonl":
        out.jsonl({"event": "end", **{key: value for key, value in payload.items() if key != "history"}})
        return
    resume_hint = f"{command.prog} run --resume {result.run_id}"
    if result.status == "failed":
        line = f"run {result.run_id} [failed]"
    elif result.status in ("completed", "dry_run"):
        line = f"run {result.run_id} [{result.status}] {payload.get('epochs_completed', 0)} epoch(s)"
    elif result.status == "paused":
        reason = "timeout reached" if result.stopped_by == "timeout" else f"stopped by {result.stopped_by}"
        line = f"run {result.run_id} [paused] {reason}; resume with: {resume_hint}"
    else:
        line = f"run {result.run_id} [{result.status}]" + (f" ({result.stopped_by})" if result.stopped_by else "")

    def text() -> None:
        for note in result.summary.get("notes", []):
            if note not in result.shown_notes:
                out.info(f"  note     : {note}")
        if result.log_file is not None:
            out.info(f"  log      : {result.log_file}")
        out.result(line)

    out.emit(payload, text=text)
