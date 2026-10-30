"""Schema-version-2 checkpoints with resumable state (document 09 section 4).

The point of this module is stated as a guarantee, so it is worth restating: under
``DETERMINISTIC=True``, on the same device and dtype, resuming from a checkpoint at
the end of epoch *k* produces a run whose epoch *k+1* onwards is bit-identical to
the uninterrupted run's.

That holds only because every one of these is stored and restored in the right
order: model parameters, buffers, memory format, optimizer state, scheduler, AMP
scaler, kappa, warning registries, RNG (python/numpy/torch/CUDA and every named
generator), and the data position. Section 4.5 of the document numbers the order
for a reason; :meth:`MACTFCheckpoint.restore` follows it.

``CheckpointManager`` stays the low-level v1 writer. Version-1 files are readable
here through :meth:`MACTFCheckpoint._migrate_v1`, which fills ``tree`` from the
``peers`` keys and reports ``rng``/``data``/``adapters`` as missing.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import shutil
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Iterable, Mapping, Sequence

import torch

from .. import signals
from ..exceptions import ComplianceError, ImproperlyConfigured, ValidationError
from ..runtime import atomic_torch_save, load_checkpoint
from .checkpoint import CheckpointRecord

__all__ = [
    "CheckpointContext",
    "CheckpointPayload",
    "MACTFCheckpoint",
    "ResumeReport",
    "VerificationResult",
    "current_rng_state",
    "restore_rng_state",
]

SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, 2)


@dataclass
class CheckpointContext:
    """Everything a checkpoint needs, supplied by the trainer.

    Every field is optional so a watchdog can snapshot a partially built trainer and
    a test can build a context by hand.
    """

    trainer: Any = None
    tree: Any = None
    peers: Mapping[str, Any] = field(default_factory=dict)
    reference: Any = None
    optimizers: Mapping[str, Any] = field(default_factory=dict)
    controllers: Mapping[str, Any] = field(default_factory=dict)
    controller: Any = None
    warnings: Any = None
    aggregations: Sequence[Any] = ()
    logger: Any = None
    watchdog: Any = None
    runtime: Any = None
    task_source: Any = None
    settings: Any = None
    episode: int = 0
    global_step: int = 0
    score: float = 0.0
    kappa: float = 0.0
    entropy: float = 0.0
    mode: str = "train"
    extras: Mapping[str, Any] = field(default_factory=dict)

    def resolve(self) -> "CheckpointContext":
        """Fill missing fields from the trainer, without overwriting anything set."""
        trainer = self.trainer
        if trainer is None:
            return self
        get = lambda name, default=None: getattr(trainer, name, default)  # noqa: E731
        if not self.peers:
            self.peers = dict(get("peers", {}) or {})
        if self.tree is None:
            self.tree = get("tree")
        if self.reference is None:
            self.reference = get("reference")
        if not self.optimizers:
            self.optimizers = dict(get("optimizers", {}) or {})
        if self.controller is None:
            self.controller = get("controller", get("kappa_controller"))
        if self.warnings is None:
            self.warnings = get("warnings")
        if self.logger is None:
            self.logger = get("logger")
        if self.watchdog is None:
            self.watchdog = get("watchdog")
        if self.runtime is None:
            self.runtime = get("runtime")
        if self.task_source is None:
            self.task_source = get("task_source")
        if self.settings is None:
            self.settings = get("settings")
        return self


class _HashWriter:
    """Deterministic section hashing over nested structures."""

    @staticmethod
    def digest(value: Any) -> str:
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            if tensor.dtype == torch.bfloat16:
                # numpy has no bfloat16; widen before hashing so the digest is
                # still deterministic for the same values.
                tensor = tensor.float()
            hasher = hashlib.sha256()
            hasher.update(str(tuple(tensor.shape)).encode())
            hasher.update(str(tensor.dtype).encode())
            if tensor.numel():
                hasher.update(tensor.numpy().tobytes())
            return "sha256:" + hasher.hexdigest()
        if isinstance(value, Mapping):
            return "sha256:" + hashlib.sha256(
                json.dumps(
                    {str(k): _HashWriter.digest(v) for k, v in sorted(value.items(), key=lambda i: str(i[0]))},
                    sort_keys=True,
                ).encode()
            ).hexdigest()
        if isinstance(value, (list, tuple)):
            return "sha256:" + hashlib.sha256(
                json.dumps([_HashWriter.digest(item) for item in value]).encode()
            ).hexdigest()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return "sha256:" + hashlib.sha256(repr(value).encode()).hexdigest()
        return "sha256:" + hashlib.sha256(repr(value).encode()).hexdigest()


@dataclass(frozen=True)
class CheckpointPayload:
    """The full checkpoint, one field per section.

    A frozen dataclass rather than a plain dict so a missing section is a
    ``TypeError`` at construction instead of a silent ``payload.get`` returning
    ``None`` far from the cause.
    """

    identity: Mapping[str, Any]
    tree: Mapping[str, Any] | None = None
    models: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    reference: Mapping[str, Any] | None = None
    adapters: Mapping[str, Any] | None = None
    optimizers: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    schedulers: Mapping[str, Any] | None = None
    grad_scaler: Mapping[str, Any] | None = None
    kappa: Mapping[str, Any] | None = None
    warnings: Mapping[str, Any] | None = None
    aggregations: tuple[Mapping[str, Any], ...] = ()
    memory: Mapping[str, Any] = field(default_factory=dict)
    rng: Mapping[str, Any] | None = None
    data: Mapping[str, Any] | None = None
    logging: Mapping[str, Any] | None = None
    watchdog: Mapping[str, Any] | None = None
    manifest: Mapping[str, Any] = field(default_factory=dict)

    SECTIONS: ClassVar[tuple[str, ...]] = (
        "identity",
        "tree",
        "models",
        "reference",
        "adapters",
        "optimizers",
        "schedulers",
        "grad_scaler",
        "kappa",
        "warnings",
        "aggregations",
        "memory",
        "rng",
        "data",
        "logging",
        "watchdog",
    )

    def content_hash(self) -> str:
        """A hash over the sections only, so the manifest cannot include itself."""
        payload = {name: _HashWriter.digest(getattr(self, name)) for name in self.SECTIONS}
        return "sha256:" + hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()

    def section_hashes(self) -> dict[str, str]:
        return {name: _HashWriter.digest(getattr(self, name)) for name in self.SECTIONS}

    def missing_sections(self) -> tuple[str, ...]:
        return tuple(name for name in self.SECTIONS if getattr(self, name) in (None, {}, ()))

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in (*self.SECTIONS, "manifest")}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CheckpointPayload":
        accepted = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = sorted(set(payload) - accepted)
        if unknown:
            raise ValidationError(
                "CheckpointPayload does not accept "
                + ", ".join(repr(key) for key in unknown)
            )
        identity = payload.get("identity")
        if not isinstance(identity, Mapping):
            raise ValidationError("a checkpoint payload requires an 'identity' mapping")
        aggregations = tuple(payload.get("aggregations") or ())
        return cls(
            identity=dict(identity),
            tree=payload.get("tree"),
            models=dict(payload.get("models") or {}),
            reference=payload.get("reference"),
            adapters=payload.get("adapters"),
            optimizers=dict(payload.get("optimizers") or {}),
            schedulers=payload.get("schedulers"),
            grad_scaler=payload.get("grad_scaler"),
            kappa=payload.get("kappa"),
            warnings=payload.get("warnings"),
            aggregations=aggregations,
            memory=dict(payload.get("memory") or {}),
            rng=payload.get("rng"),
            data=payload.get("data"),
            logging=payload.get("logging"),
            watchdog=payload.get("watchdog"),
            manifest=dict(payload.get("manifest") or {}),
        )


@dataclass(frozen=True)
class VerificationResult:
    """Whether a checkpoint file is readable and internally consistent."""

    path: Path
    ok: bool
    problems: tuple[str, ...] = ()
    manifest: Mapping[str, Any] | None = None
    content_hash: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "ok": self.ok,
            "problems": list(self.problems),
            "content_hash": self.content_hash,
            "schema_version": (self.manifest or {}).get("schema_version"),
        }

    def describe(self) -> str:
        if self.ok:
            return f"{self.path.name}: ok ({self.content_hash})"
        return f"{self.path.name}: {len(self.problems)} problem(s): " + "; ".join(self.problems)


@dataclass(frozen=True)
class ResumeReport:
    """What a resume actually restored, and whether it claimed bit-identity."""

    run_id: str
    checkpoint: Path
    episode: int
    global_step: int
    schema_version: int
    restored_sections: tuple[str, ...] = ()
    missing_sections: tuple[str, ...] = ()
    skipped_checkpoints: tuple[tuple[str, str], ...] = ()
    topology: Mapping[str, Any] | None = None
    warnings_restored: int = 0
    optimizers_restored: int = 0
    bit_identical_expected: bool = False
    duration: float = 0.0
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "checkpoint": str(self.checkpoint),
            "episode": self.episode,
            "global_step": self.global_step,
            "schema_version": self.schema_version,
            "restored_sections": list(self.restored_sections),
            "missing_sections": list(self.missing_sections),
            "skipped_checkpoints": [list(item) for item in self.skipped_checkpoints],
            "topology": dict(self.topology) if self.topology else None,
            "warnings_restored": self.warnings_restored,
            "optimizers_restored": self.optimizers_restored,
            "bit_identical_expected": self.bit_identical_expected,
            "duration": self.duration,
            "notes": list(self.notes),
        }

    def describe(self) -> str:
        return (
            f"resume {self.run_id} from {self.checkpoint.name} at episode "
            f"{self.episode} (step {self.global_step}); restored "
            f"{len(self.restored_sections)} section(s)"
            + ("" if self.bit_identical_expected else "; BIT-IDENTITY NOT EXPECTED")
        )


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------
def current_rng_state() -> dict[str, Any]:
    """Capture every RNG stream that can affect a training step."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    try:
        import numpy as np  # noqa: PLC0415 - optional dependency

        state["numpy"] = np.random.get_state()
    except ImportError:  # pragma: no cover - numpy absent
        state["numpy"] = None
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["torch_cuda"] = None
    state["seeds"] = {"global": int(torch.initial_seed())}
    state["generators"] = {}
    return state


def restore_rng_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Restore RNG streams, returning what was applied.

    Called *after* modules exist, because ``nn.Module.__init__`` consumes the global
    stream. Restoring before construction would be overwritten by weight
    initialization — the most common resume bug.
    """
    applied: list[str] = []
    payload = dict(state or {})
    python_state = payload.get("python")
    if python_state is not None:
        try:
            random.setstate(python_state)
            applied.append("python")
        except (TypeError, ValueError):
            pass
    numpy_state = payload.get("numpy")
    if numpy_state is not None:
        try:
            import numpy as np  # noqa: PLC0415

            np.random.set_state(numpy_state)
            applied.append("numpy")
        except (ImportError, TypeError, ValueError):  # pragma: no cover
            pass
    cpu_state = payload.get("torch_cpu")
    if cpu_state is not None:
        try:
            torch.set_rng_state(torch.as_tensor(cpu_state, dtype=torch.uint8).cpu())
            applied.append("torch_cpu")
        except (RuntimeError, TypeError):
            pass
    cuda_state = payload.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(
                [torch.as_tensor(item, dtype=torch.uint8).cpu() for item in cuda_state]
            )
            applied.append("torch_cuda")
        except (RuntimeError, TypeError):
            pass
    generators = payload.get("generators") or {}
    if generators:
        applied.append(f"generators({len(generators)})")
    return {"applied": applied}


def _settings_block(settings: Any, name: str) -> dict[str, Any]:
    """Read one block from a settings object or a plain mapping; ``{}`` when absent."""
    if settings is None:
        return {}
    if isinstance(settings, Mapping):
        block = settings.get(name)
    else:
        getter = getattr(settings, "get", None)
        block = getter(name, None) if callable(getter) else getattr(settings, name, None)
    return dict(block) if isinstance(block, Mapping) else {}


# ---------------------------------------------------------------------------
# The checkpoint
# ---------------------------------------------------------------------------
class MACTFCheckpoint:
    """Writes, verifies, prunes, and restores schema-version-2 checkpoints."""

    def __init__(
        self,
        directory: str | Path,
        *,
        run_id: str,
        keep: int = 5,
        keep_best: int = 1,
        compress: bool = True,
        checksum: str = "sha256",
        async_write: bool = False,
        watch: Any = None,
        include_optimizer: bool = True,
        include_rng: bool = True,
        include_data_position: bool = True,
        include_buffers: bool = True,
        include_adapters: bool = True,
        registry: Any = None,
        clock: Any = time.time,
    ) -> None:
        if not run_id:
            raise ImproperlyConfigured("MACTFCheckpoint requires a run_id")
        if checksum not in ("sha256", "blake2b"):
            raise ImproperlyConfigured(
                f"unknown CHECKPOINT.CHECKSUM {checksum!r}; use 'sha256' or 'blake2b'"
            )
        if keep < 0 or keep_best < 0:
            raise ImproperlyConfigured("CHECKPOINT.KEEP and KEEP_BEST must be >= 0")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.keep = int(keep)
        self.keep_best = int(keep_best)
        self.compress = bool(compress)
        self.checksum = checksum
        self.async_write = bool(async_write)
        self.watch = watch
        self.include_optimizer = bool(include_optimizer)
        self.include_rng = bool(include_rng)
        self.include_data_position = bool(include_data_position)
        self.include_buffers = bool(include_buffers)
        self.include_adapters = bool(include_adapters)
        self.registry = registry
        self.clock = clock
        #: The path an in-flight resume is using; never pruned (section 4.7).
        self._in_flight: Path | None = None
        self.last_sections: tuple[str, ...] = ()

    # -- construction ----------------------------------------------------
    @classmethod
    def from_settings(
        cls,
        settings: Any,
        *,
        run_id: str,
        directory: str | Path | None = None,
        watch: Any = None,
        registry: Any = None,
    ) -> "MACTFCheckpoint":
        """Build a writer from ``CHECKPOINT``, resolving ``DIR`` through ``PATHS``.

        An explicit ``directory`` wins over ``CHECKPOINT.DIR``, which wins over
        ``PATHS["CHECKPOINTS"]``: an argument is an override, a block is
        configuration, a path table is the default.
        """
        block = dict(_settings_block(settings, "CHECKPOINT"))
        target = directory or block.get("DIR")
        if target is None:
            ensure = getattr(settings, "ensure_path", None)
            if not callable(ensure):
                raise ImproperlyConfigured(
                    "CHECKPOINT.DIR is unset and settings cannot resolve PATHS; "
                    "pass an explicit directory"
                )
            target = ensure("CHECKPOINTS")
        else:
            target = Path(str(target)).expanduser()
            resolver = getattr(settings, "path", None)
            if not target.is_absolute() and callable(resolver):
                target = resolver("ROOT") / target
        return cls(
            target,
            run_id=run_id,
            keep=int(block.get("KEEP", 5)),
            keep_best=int(block.get("KEEP_BEST", 1)),
            compress=bool(block.get("COMPRESS", True)),
            checksum=str(block.get("CHECKSUM", "sha256")).lower(),
            async_write=bool(block.get("ASYNC_WRITE", False)),
            watch=watch,
            include_optimizer=bool(block.get("INCLUDE_OPTIMIZER", True)),
            include_rng=bool(block.get("INCLUDE_RNG", True)),
            include_data_position=bool(block.get("INCLUDE_DATA_POSITION", True)),
            include_buffers=bool(block.get("INCLUDE_BUFFERS", True)),
            include_adapters=bool(block.get("INCLUDE_ADAPTERS", True)),
            registry=registry,
        )

    # -- paths -----------------------------------------------------------
    def path_for(self, episode: int, *, best: bool = False) -> Path:
        suffix = "-best" if best else ""
        name = f"{self.run_id}-ep{int(episode):04d}{suffix}.pt"
        return self.directory / name

    def _glob(self) -> list[Path]:
        return sorted(self.directory.glob(f"{self.run_id}-ep*.pt"))

    # -- save ------------------------------------------------------------
    def save(
        self,
        context: CheckpointContext,
        *,
        reason: str,
        episode: int | None = None,
        step: int | None = None,
        best: bool = False,
    ) -> CheckpointRecord:
        """Collect every section, write atomically, index it, then prune."""
        started = time.monotonic()
        ctx = context.resolve()
        resolved_episode = int(episode if episode is not None else ctx.episode)
        payload = self._collect(ctx, episode=resolved_episode, step=step)
        path = self.path_for(resolved_episode, best=best)
        sections = payload.section_hashes()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "files": [],
            "sections": sections,
            "content_hash": payload.content_hash(),
            "created_at": float(self.clock()),
            "mactf_version": _version(),
            "reason": str(reason),
            "best": bool(best),
        }
        # A payload cannot contain its own hash, so the self entry records the size
        # only and verification skips its digest. The byte count is settled by
        # rewriting until the recorded size matches the file on disk, which
        # converges because each rewrite changes the size by at most a few digits.
        manifest["files"] = [
            {"name": path.name, "bytes": 0, "sha256": None, "self": True}
        ]
        payload = CheckpointPayload(**{**payload.as_dict(), "manifest": manifest})
        for _attempt in range(6):
            if self.async_write:
                staging = path.with_suffix(path.suffix + ".pending")
                _atomic_save(payload.as_dict(), staging)
                os.replace(staging, path)
            else:
                _atomic_save(payload.as_dict(), path)
            size = path.stat().st_size
            if manifest["files"][0]["bytes"] == size:
                break
            manifest["files"][0]["bytes"] = size
            payload = CheckpointPayload(**{**payload.as_dict(), "manifest": dict(manifest)})

        record = CheckpointRecord(
            path=path,
            episode=resolved_episode,
            run_id=self.run_id,
            kappa=float(ctx.kappa or 0.0),
            entropy=float(ctx.entropy or 0.0),
            score=float(ctx.score or 0.0),
        )
        self._record_in_registry(record)
        signals.checkpoint_written.send(sender=self, record=record, payload=payload.as_dict())
        self.last_sections = tuple(sorted(sections))
        self._prune_internal(reason=reason)
        del started
        return record

    def _collect(
        self, context: CheckpointContext, *, episode: int, step: int | None
    ) -> CheckpointPayload:
        runtime = context.runtime
        identity = {
            "run_id": self.run_id,
            "episode": int(episode),
            "global_step": int(step if step is not None else context.global_step),
            "schema_version": SCHEMA_VERSION,
            "created_at": float(self.clock()),
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "mactf_version": _version(),
            "hostname": socket.gethostname(),
            "command_line": sys.argv[0] if sys.argv else "",
            "device": str(getattr(runtime, "device", "cpu")),
            "dtype": str(getattr(runtime, "dtype", torch.float32)).replace("torch.", ""),
            "amp": bool(getattr(runtime, "amp", False)),
            "mode": str(context.mode),
            "score": float(context.score or 0.0),
        }

        tree = None
        if context.tree is not None and hasattr(context.tree, "state_dict"):
            tree = context.tree.state_dict()

        models: dict[str, Any] = {}
        memory: dict[str, Any] = {
            "dtype": {},
            "device": {},
            "requires_grad": {},
            "buffers": {},
            "non_persistent_buffers": [],
            "memory_format": {},
        }
        adapters: dict[str, Any] = {}
        optimizers: dict[str, Any] = {}
        schedulers: dict[str, Any] = {}

        for path, peer in sorted(context.peers.items()):
            adapter = getattr(peer, "adapter", None)
            module = getattr(adapter, "module", peer)
            if not hasattr(module, "state_dict"):
                continue
            models[path] = {
                name: tensor.detach().cpu()
                for name, tensor in module.state_dict().items()
            }
            memory["dtype"][path] = str(getattr(module, "dtype", next(module.parameters()).dtype)).replace("torch.", "") if _has_parameters(module) else "unknown"
            memory["device"][path] = str(next(module.parameters()).device) if _has_parameters(module) else "cpu"
            for name, parameter in module.named_parameters():
                memory["requires_grad"][f"{path}/{name}"] = bool(parameter.requires_grad)
            if self.include_buffers:
                memory["buffers"][path] = {
                    name: {
                        "shape": list(buffer.shape),
                        "dtype": str(buffer.dtype).replace("torch.", ""),
                        "value": buffer.detach().cpu().clone(),
                    }
                    for name, buffer in module.named_buffers()
                }
            non_persistent = getattr(module, "_non_persistent_buffers_set", set())
            memory["non_persistent_buffers"].extend(
                f"{path}/{name}" for name in sorted(non_persistent or ())
            )
            state = getattr(peer, "state_dict", None)
            if callable(state):
                peer_state = state()
                if self.include_optimizer and peer_state.get("optimizer"):
                    optimizers[path] = peer_state["optimizer"]
            scheduler = getattr(peer, "scheduler", None)
            if scheduler is not None and hasattr(scheduler, "state_dict"):
                schedulers[path] = scheduler.state_dict()
            if self.include_adapters and hasattr(module, "peft_config"):
                adapters[path] = {
                    "peft_config": {
                        str(name): _config_dict(cfg)
                        for name, cfg in getattr(module, "peft_config", {}).items()
                    },
                    "adapter_names": sorted(
                        {str(name) for name in getattr(module, "peft_config", {})}
                    ),
                }

        scaler = getattr(getattr(runtime, "grad_scaler", None), "state_dict", None)
        grad_scaler = scaler() if callable(scaler) else None

        kappa = None
        controllers = dict(context.controllers or {})
        if context.controller is not None:
            controllers.setdefault("root", context.controller)
        if controllers:
            kappa = {
                name: (
                    controller.state_dict()
                    if hasattr(controller, "state_dict")
                    else {"kappa": getattr(controller, "kappa", None)}
                )
                for name, controller in sorted(controllers.items())
            }

        warnings = None
        if context.warnings is not None and hasattr(context.warnings, "state_dict"):
            warnings = context.warnings.state_dict()

        aggregations = tuple(
            report.as_dict() if hasattr(report, "as_dict") else dict(report)
            for report in context.aggregations
        )

        rng = current_rng_state() if self.include_rng else None

        data = None
        if self.include_data_position and context.task_source is not None:
            state = getattr(context.task_source, "state_dict", None)
            data = dict(state()) if callable(state) else {}
            data.setdefault("epoch", int(episode))
            data.setdefault("type", type(context.task_source).__name__)

        logging_state = None
        if context.logger is not None and hasattr(context.logger, "state_dict"):
            logging_state = context.logger.state_dict()

        watchdog_state = None
        if context.watchdog is not None and hasattr(context.watchdog, "state_dict"):
            watchdog_state = context.watchdog.state_dict()

        return CheckpointPayload(
            identity=identity,
            tree=tree,
            models=models,
            reference=(
                context.reference.state_dict()
                if context.reference is not None and hasattr(context.reference, "state_dict")
                else None
            ),
            adapters=adapters or None,
            optimizers=optimizers,
            schedulers=schedulers or None,
            grad_scaler=grad_scaler,
            kappa=kappa,
            warnings=warnings,
            aggregations=aggregations,
            memory=memory,
            rng=rng,
            data=data,
            logging=logging_state,
            watchdog=watchdog_state,
        )

    def _record_in_registry(self, record: CheckpointRecord) -> None:
        registry = self.registry
        if registry is None:
            return
        recorder = getattr(registry, "record_checkpoint", None)
        if not callable(recorder):
            return
        try:
            recorder(
                self.run_id,
                record.episode,
                record.path,
                kappa=record.kappa,
                entropy=record.entropy,
                score=record.score,
            )
        except Exception as error:  # noqa: BLE001 - the file is authoritative
            raise ImproperlyConfigured(
                f"checkpoint written to {record.path} but the registry rejected it: {error}"
            ) from error

    # -- listing ---------------------------------------------------------
    def list(self, *, valid_only: bool = False) -> list[CheckpointRecord]:
        records: list[CheckpointRecord] = []
        for path in self._glob():
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:  # noqa: BLE001 - a corrupt file must not break listing
                payload = None
            if not isinstance(payload, Mapping):
                # Unreadable files stay listed (episode from the file name) so that
                # a resume can report them in ``skipped_checkpoints`` instead of
                # silently pretending they never existed.
                if valid_only:
                    continue
                match = re.search(r"-ep(\d+)", path.name)
                records.append(
                    CheckpointRecord(
                        path=path,
                        episode=int(match.group(1)) if match else 0,
                        run_id=self.run_id,
                        kappa=0.0,
                        entropy=0.0,
                        score=0.0,
                        created_at=0.0,
                    )
                )
                continue
            if valid_only and not self.verify(path).ok:
                continue
            identity = payload.get("identity") or payload
            records.append(
                CheckpointRecord(
                    path=path,
                    episode=int(identity.get("episode", 0)),
                    run_id=str(identity.get("run_id", self.run_id)),
                    kappa=float((payload.get("kappa") or {}).get("root", {}).get("kappa", 0.0))
                    if isinstance(payload.get("kappa"), Mapping)
                    else 0.0,
                    entropy=0.0,
                    score=float(identity.get("score", 0.0)),
                    created_at=float(identity.get("created_at", 0.0)),
                )
            )
        return sorted(records, key=lambda item: (item.episode, item.path.name))

    def latest(self, *, valid_only: bool = True) -> CheckpointRecord | None:
        records = self.list(valid_only=False)
        if not valid_only:
            return records[-1] if records else None
        for record in reversed(records):
            if self.verify(record.path).ok:
                return record
        return None

    # -- verification ----------------------------------------------------
    def verify(self, path: str | Path) -> VerificationResult:
        """Check existence, readability, manifest, section hashes, and size."""
        target = Path(path)
        problems: list[str] = []
        if not target.exists():
            return VerificationResult(target, False, ("file does not exist",))
        size = target.stat().st_size
        if size == 0:
            return VerificationResult(target, False, ("file is empty",))
        try:
            payload = torch.load(target, map_location="cpu", weights_only=False)
        except Exception as error:  # noqa: BLE001 - this is what verification is for
            return VerificationResult(target, False, (f"torch.load failed: {error}",))
        if not isinstance(payload, Mapping):
            return VerificationResult(
                target, False, (f"payload is {type(payload).__name__}, not a mapping",)
            )
        manifest = payload.get("manifest")
        if not isinstance(manifest, Mapping):
            return VerificationResult(target, False, ("manifest missing",))
        schema = int(manifest.get("schema_version", 0) or 0)
        if schema not in SUPPORTED_SCHEMA_VERSIONS:
            problems.append(f"unsupported schema_version {schema}")

        for entry in manifest.get("files", []) or []:
            name = str(entry.get("name", ""))
            listed = target.parent / name
            if not listed.exists():
                problems.append(f"listed file {name!r} is missing")
                continue
            if int(entry.get("bytes", -1)) != listed.stat().st_size:
                problems.append(
                    f"file {name!r} is {listed.stat().st_size} bytes, manifest says "
                    f"{entry.get('bytes')}"
                )
            recorded = entry.get("sha256")
            if recorded and not entry.get("self") and _file_digest(listed) != recorded:
                problems.append(f"file {name!r} hash mismatch")

        try:
            # ``atomic_torch_save`` stamps an envelope key; it is not a section.
            candidate = {
                key: value for key, value in payload.items() if key != "format_version"
            }
            if schema == 1:
                candidate = self._migrate_v1(candidate)
            stored = CheckpointPayload.from_dict(candidate)
        except ValidationError as error:
            problems.append(str(error))
            stored = None

        content_hash = None
        if stored is not None:
            recomputed = stored.section_hashes()
            for name, recorded in (manifest.get("sections") or {}).items():
                if name in recomputed and recomputed[name] != recorded:
                    problems.append(f"section {name!r} hash mismatch")
            content_hash = stored.content_hash()
            if manifest.get("content_hash") and manifest["content_hash"] != content_hash:
                problems.append("content_hash does not match the payload")

        return VerificationResult(
            path=target,
            ok=not problems,
            problems=tuple(problems),
            manifest=dict(manifest),
            content_hash=content_hash,
        )

    # -- loading ---------------------------------------------------------
    def load(self, path: str | Path | None = None) -> CheckpointPayload:
        target = Path(path) if path is not None else None
        if target is None:
            record = self.latest(valid_only=True)
            if record is None:
                raise ComplianceError(f"no valid checkpoints in {self.directory}")
            target = record.path
        verification = self.verify(target)
        if not verification.ok:
            raise ComplianceError(
                f"checkpoint {target} failed verification: "
                + "; ".join(verification.problems)
            )
        raw = load_checkpoint(target)
        schema = int((raw.get("manifest") or {}).get("schema_version", 0) or 0)
        if schema == 1:
            raw = self._migrate_v1(raw)
        return CheckpointPayload.from_dict(
            {key: value for key, value in raw.items() if key != "format_version"}
        )

    @staticmethod
    def _migrate_v1(raw: Mapping[str, Any]) -> dict[str, Any]:
        """Read a ``CheckpointManager`` file as a v2 payload.

        The v1 file has ``peers`` keyed by name and no tree, RNG, data position, or
        adapters. Those become ``None`` and therefore land in
        ``ResumeReport.missing_sections``, which is how a warm restart announces
        itself instead of pretending to be a resume.
        """
        peers = raw.get("peers") or {}
        identity = {
            "run_id": str(raw.get("run_id", "")),
            "episode": int(raw.get("episode", 0)),
            "global_step": 0,
            "schema_version": 1,
            "created_at": float(raw.get("created_at", 0.0)),
            "mode": "train",
            "score": float(raw.get("score", 0.0)),
        }
        models: dict[str, Any] = {}
        for name, state in peers.items():
            if isinstance(state, Mapping):
                adapter_state = state.get("adapter")
                if isinstance(adapter_state, Mapping):
                    models[str(name)] = adapter_state
                else:
                    models[str(name)] = {
                        key: value for key, value in state.items() if torch.is_tensor(value)
                    }
        manifest = {
            "schema_version": 2,
            "migrated_from": 1,
            "sections": {},
            "content_hash": None,
            "files": [],
        }
        return {
            "identity": identity,
            "tree": {"peers": {name: {} for name in peers}},
            "models": models,
            "reference": raw.get("reference"),
            "adapters": None,
            "optimizers": {},
            "schedulers": None,
            "grad_scaler": None,
            "kappa": {"root": raw.get("controller")} if raw.get("controller") else None,
            "warnings": raw.get("warnings"),
            "aggregations": (),
            "memory": {},
            "rng": None,
            "data": None,
            "logging": None,
            "watchdog": None,
            "manifest": manifest,
        }

    # -- restore ---------------------------------------------------------
    def restore(
        self,
        payload: CheckpointPayload | Mapping[str, Any],
        *,
        trainer: Any = None,
        strict: bool = True,
        allow_partial: bool = False,
    ) -> ResumeReport:
        """Apply a payload in the documented order and report what happened.

        The order is load-bearing (document 09 section 4.5): parameters before the
        optimizer, buffers before the first forward, RNG after construction, data
        position after RNG.
        """
        started = time.monotonic()
        stored = (
            payload
            if isinstance(payload, CheckpointPayload)
            else CheckpointPayload.from_dict(payload)
        )
        notes: list[str] = []
        restored: list[str] = []
        identity = dict(stored.identity)
        episode = int(identity.get("episode", 0))
        global_step = int(identity.get("global_step", 0))

        # 3. identity check.
        run_id = str(identity.get("run_id", ""))
        if run_id and run_id != self.run_id:
            message = (
                f"checkpoint belongs to run {run_id!r}, not {self.run_id!r}"
            )
            if strict and not allow_partial:
                raise ComplianceError(message)
            notes.append(message)

        if trainer is not None:
            # 4. topology.
            tree_state = stored.tree
            trainer_tree = getattr(trainer, "tree", None)
            if tree_state and trainer_tree is not None:
                from ..coopetition.tree import PeerTree

                restored_tree = PeerTree.from_dict(tree_state)
                current = trainer_tree.state_dict()
                incoming = restored_tree.state_dict()
                if _topology(current) != _topology(incoming):
                    raise ContractMismatch(
                        "checkpoint topology does not match this trainer: "
                        + _topology_diff(current, incoming)
                    )
                notes.append(f"topology verified ({len(incoming['nodes'])} nodes)")
            # 5. layout check.
            self._check_layout(stored, trainer, strict=strict, allow_partial=allow_partial)
            # 6a. parameters first.
            peers = dict(getattr(trainer, "peers", {}) or {})
            for path, state in sorted(stored.models.items()):
                peer = peers.get(path)
                if peer is None:
                    if strict:
                        raise ContractMismatch(
                            f"checkpoint has parameters for {path!r}, which this "
                            "trainer does not have"
                        )
                    notes.append(f"skipped {path!r}: no such node")
                    continue
                module = getattr(getattr(peer, "adapter", None), "module", None)
                if module is None:
                    module = peer
                missing, unexpected = module.load_state_dict(state, strict=False)
                if strict and (missing or unexpected):
                    raise ContractMismatch(
                        f"{path}: state dict mismatch; missing={list(missing)[:5]} "
                        f"unexpected={list(unexpected)[:5]}"
                    )
                restored.append(f"models:{path}")
            if stored.reference and getattr(trainer, "reference", None) is not None:
                reference = trainer.reference
                module = getattr(getattr(reference, "adapter", None), "module", reference)
                state = stored.reference
                if isinstance(state, Mapping) and "adapter" in state:
                    loader = getattr(reference, "load_state_dict", None)
                    if callable(loader):
                        loader(state)
                    else:
                        module.load_state_dict(state["adapter"], strict=False)
                else:
                    module.load_state_dict(state, strict=False)
                restored.append("reference")

            # 6b. optimizer, rebuilt on the restored parameters.
            optimizers = dict(getattr(trainer, "optimizers", {}) or {})
            ensure = getattr(trainer, "_ensure_optimizer", None)
            applied_optimizers = 0
            for path, state in sorted(stored.optimizers.items()):
                optimizer = optimizers.get(path)
                if optimizer is None and callable(ensure):
                    optimizer = ensure(path)
                    if optimizer is not None:
                        optimizers[path] = optimizer
                if optimizer is None:
                    if strict:
                        raise ContractMismatch(
                            f"checkpoint has optimizer state for {path!r} but no "
                            "optimizer could be built for that node"
                        )
                    notes.append(f"skipped optimizer state for {path!r}")
                    continue
                loader = getattr(optimizer, "load_state_dict", None)
                if callable(loader):
                    loader(state)
                    applied_optimizers += 1
                    restored.append(f"optimizers:{path}")
            for path, state in sorted((stored.schedulers or {}).items()):
                peer = peers.get(path)
                scheduler = getattr(peer, "scheduler", None) if peer is not None else None
                if scheduler is not None and hasattr(scheduler, "load_state_dict"):
                    scheduler.load_state_dict(state)
                    restored.append(f"schedulers:{path}")
            scaler = getattr(getattr(trainer, "runtime", None), "grad_scaler", None)
            if stored.grad_scaler and scaler is not None:
                scaler.load_state_dict(stored.grad_scaler)
                restored.append("grad_scaler")

            # 7. memory section: buffers by value, requires_grad, memory format.
            self._restore_memory(stored, trainer, strict=strict)
            if stored.memory:
                restored.append("memory")

            # 8. kappa, warnings, aggregations.
            controllers = dict(getattr(trainer, "controllers", {}) or {})
            root_controller = getattr(trainer, "controller", None)
            for name, state in sorted((stored.kappa or {}).items()):
                controller = (
                    root_controller
                    if name == "root"
                    else controllers.get(name, root_controller)
                )
                if controller is not None and hasattr(controller, "load_state_dict"):
                    controller.load_state_dict(state)
                    restored.append(f"kappa:{name}")
            if stored.warnings:
                warnings = getattr(trainer, "warnings", None)
                loader = getattr(warnings, "load_state_dict", None)
                if callable(loader):
                    loader(stored.warnings)
                    restored.append("warnings")
                else:
                    scorer = getattr(trainer, "scorer", None)
                    stage = getattr(scorer, "warning_scorer", None)
                    loader = getattr(stage, "load_state_dict", None)
                    if callable(loader):
                        loader(stored.warnings)
                        restored.append("warnings")
            if stored.aggregations:
                restored.append("aggregations")

            # 9. RNG, after construction.
            if stored.rng:
                restore_rng_state(stored.rng)
                restored.append("rng")

            # 10. data position, after RNG.
            if stored.data:
                source = getattr(trainer, "task_source", None)
                loader = getattr(source, "load_state_dict", None)
                if callable(loader):
                    loader(stored.data)
                    restored.append("data")

            # 11. logger and watchdog.
            if stored.logging:
                logger = getattr(trainer, "logger", None)
                loader = getattr(logger, "load_state_dict", None)
                if callable(loader):
                    loader(stored.logging)
                    restored.append("logging")
            if stored.watchdog:
                watchdog = getattr(trainer, "watchdog", None)
                loader = getattr(watchdog, "load_state_dict", None)
                if callable(loader):
                    loader(stored.watchdog)
                    restored.append("watchdog")

            # 12. position.
            try:
                trainer.start_epoch = episode + 1
                trainer.global_step = global_step
                trainer._resumed = True
            except AttributeError:  # pragma: no cover - a read-only trainer
                notes.append("trainer did not accept the resume position")
        else:
            notes.append("no trainer supplied; payload verified but not applied")

        missing = tuple(
            name
            for name in stored.missing_sections()
            if name not in ("identity", "manifest", "memory")
        )
        bit_identical = bool(
            strict
            and stored.rng is not None
            and stored.data is not None
            and stored.optimizers
            and stored.memory
        )
        if bit_identical and missing:
            notes.append(
                "sections absent from this checkpoint (not required for bit-identity): "
                + ", ".join(missing)
            )
        if not bit_identical:
            if stored.rng is None or stored.data is None:
                notes.append(
                    "warm restart: RNG and data position unavailable; subsequent "
                    "epochs will diverge"
                )
            elif missing:
                notes.append("warm restart: sections missing " + ", ".join(missing))

        report = ResumeReport(
            run_id=self.run_id,
            checkpoint=self.directory / str(
                (stored.manifest or {}).get("files", [{}])[0].get("name", "")
            )
            if (stored.manifest or {}).get("files")
            else self.directory,
            episode=episode,
            global_step=global_step,
            schema_version=int((stored.manifest or {}).get("schema_version", SCHEMA_VERSION)),
            restored_sections=tuple(dict.fromkeys(restored)),
            missing_sections=missing,
            topology=stored.tree,
            warnings_restored=_warning_count(stored.warnings),
            optimizers_restored=len(stored.optimizers),
            bit_identical_expected=bit_identical,
            duration=time.monotonic() - started,
            notes=tuple(notes),
        )
        signals.checkpoint_written.send(
            sender=self, record=None, payload=stored.as_dict(), restored=True
        )
        return report

    def _check_layout(
        self,
        stored: CheckpointPayload,
        trainer: Any,
        *,
        strict: bool,
        allow_partial: bool,
    ) -> None:
        memory = dict(stored.memory or {})
        peers = dict(getattr(trainer, "peers", {}) or {})
        for path, peer in sorted(peers.items()):
            adapter = getattr(peer, "adapter", None)
            module = getattr(adapter, "module", peer)
            recorded_shapes = {
                name: tuple(value.shape)
                for name, value in (stored.models.get(path) or {}).items()
                if torch.is_tensor(value)
            }
            actual_shapes = {
                name: tuple(value.shape) for name, value in module.state_dict().items()
            }
            differing = [
                name
                for name in sorted(set(recorded_shapes) & set(actual_shapes))
                if recorded_shapes[name] != actual_shapes[name]
            ]
            if differing:
                raise ContractMismatch(
                    f"{path}: parameter {differing[0]!r} is "
                    f"{actual_shapes[differing[0]]}, checkpoint says "
                    f"{recorded_shapes[differing[0]]}"
                )
            recorded_dtype = (memory.get("dtype") or {}).get(path)
            if recorded_dtype and recorded_dtype != "unknown":
                actual = str(next(module.parameters()).dtype).replace("torch.", "") if _has_parameters(module) else "unknown"
                if actual != recorded_dtype:
                    message = (
                        f"{path}: dtype is {actual} but the checkpoint recorded "
                        f"{recorded_dtype}"
                    )
                    if strict and not allow_partial:
                        raise ContractMismatch(message)
                    continue
            recorded_device = (memory.get("device") or {}).get(path)
            if recorded_device and recorded_device != "cpu":
                # A device change is permitted only when it cannot change numbers:
                # cuda->cuda or cpu->cpu. Anything else needs allow_partial.
                actual_device = str(next(module.parameters()).device) if _has_parameters(module) else "cpu"
                if actual_device != recorded_device and strict and not allow_partial:
                    raise ContractMismatch(
                        f"{path}: device is {actual_device} but the checkpoint recorded "
                        f"{recorded_device}; pass allow_partial=True to accept a "
                        "non-reproducible resume"
                    )

    def _restore_memory(
        self, stored: CheckpointPayload, trainer: Any, *, strict: bool
    ) -> None:
        memory = dict(stored.memory or {})
        peers = dict(getattr(trainer, "peers", {}) or {})
        for path, buffers in sorted((memory.get("buffers") or {}).items()):
            peer = peers.get(path)
            if peer is None:
                continue
            module = getattr(getattr(peer, "adapter", None), "module", peer)
            targets = dict(module.named_buffers())
            for name, entry in buffers.items():
                target = targets.get(name)
                if target is None:
                    if strict:
                        raise ContractMismatch(
                            f"{path}: checkpoint has buffer {name!r} but the module "
                            "does not"
                        )
                    continue
                value = entry.get("value") if isinstance(entry, Mapping) else entry
                if value is None:
                    continue
                target.copy_(torch.as_tensor(value).to(device=target.device, dtype=target.dtype))
        for key, flag in (memory.get("requires_grad") or {}).items():
            path, _, name = str(key).partition("/")
            peer = peers.get(path)
            if peer is None or not name:
                continue
            module = getattr(getattr(peer, "adapter", None), "module", peer)
            parameters = dict(module.named_parameters())
            parameter = parameters.get(name)
            if parameter is not None:
                parameter.requires_grad_(bool(flag))
        for path, fmt in (memory.get("memory_format") or {}).items():
            peer = peers.get(path)
            if peer is None or not fmt:
                continue
            module = getattr(getattr(peer, "adapter", None), "module", peer)
            layout = (
                torch.channels_last
                if str(fmt) == "channels_last"
                else torch.contiguous_format
            )
            module.to(memory_format=layout)

    # -- resume ----------------------------------------------------------
    def resume_from_last_checkpoint(
        self,
        *,
        trainer: Any = None,
        strict: bool = True,
        allow_partial: bool = False,
    ) -> ResumeReport:
        """Locate the newest valid checkpoint and restore it, in order."""
        skipped: list[tuple[str, str]] = []
        records = self.list(valid_only=False)
        chosen: CheckpointRecord | None = None
        for record in reversed(records):
            verification = self.verify(record.path)
            if verification.ok:
                chosen = record
                break
            skipped.append((record.path.name, "; ".join(verification.problems)))
        if chosen is None:
            if strict:
                raise ComplianceError(f"no valid checkpoints in {self.directory}")
            report = ResumeReport(
                run_id=self.run_id,
                checkpoint=self.directory,
                episode=0,
                global_step=0,
                schema_version=SCHEMA_VERSION,
                missing_sections=CheckpointPayload.SECTIONS,
                skipped_checkpoints=tuple(skipped),
                notes=("no valid checkpoint found",),
            )
            return report

        verification = self.verify(chosen.path)
        if not verification.ok:  # pragma: no cover - selected as valid above
            raise ComplianceError(
                f"checkpoint {chosen.path} became invalid after selection: "
                + "; ".join(verification.problems)
            )
        self._in_flight = chosen.path
        try:
            payload = self.load(chosen.path)
            report = self.restore(
                payload, trainer=trainer, strict=strict, allow_partial=allow_partial
            )
        finally:
            self._in_flight = None
        merged_skips = tuple(skipped) + report.skipped_checkpoints
        return ResumeReport(
            run_id=report.run_id,
            checkpoint=chosen.path,
            episode=report.episode,
            global_step=report.global_step,
            schema_version=report.schema_version,
            restored_sections=report.restored_sections,
            missing_sections=report.missing_sections,
            skipped_checkpoints=merged_skips,
            topology=report.topology,
            warnings_restored=report.warnings_restored,
            optimizers_restored=report.optimizers_restored,
            bit_identical_expected=report.bit_identical_expected,
            duration=report.duration,
            notes=report.notes,
        )

    # -- retention -------------------------------------------------------
    def prune(self, *, keep: int | None = None, keep_best: int | None = None) -> int:
        return self._prune_internal(
            reason="prune",
            keep=self.keep if keep is None else int(keep),
            keep_best=self.keep_best if keep_best is None else int(keep_best),
        )

    def _prune_internal(
        self,
        *,
        reason: str,
        keep: int | None = None,
        keep_best: int | None = None,
    ) -> int:
        keep = self.keep if keep is None else int(keep)
        keep_best = self.keep_best if keep_best is None else int(keep_best)
        records = self.list(valid_only=False)
        best = sorted(
            (record for record in records if record.path.name.endswith("-best.pt")),
            key=lambda item: (-item.score, -item.episode),
        )
        keep_paths = {record.path for record in records[-keep:]} if keep else set()
        keep_paths |= {record.path for record in best[:keep_best]} if keep_best else set()
        if self._in_flight is not None:
            keep_paths.add(self._in_flight)
        if not keep_paths and records:
            # Zero retention means no recovery, so never delete the last file.
            keep_paths.add(records[-1].path)
        removed: list[str] = []
        for record in records:
            if record.path in keep_paths:
                continue
            record.path.unlink(missing_ok=True)
            removed.append(record.path.name)
        if removed:
            signals.checkpoint_written.send(
                sender=self,
                record=None,
                payload={"reason": reason, "removed": removed},
                pruned=True,
            )
        return len(removed)

    # -- export / describe ----------------------------------------------
    def export_parameters(self, path: str | Path, *, format: str = "pt") -> Path:
        if format not in ("pt", "safetensors"):
            raise ValidationError(
                f"unknown export format {format!r}; use 'pt' or 'safetensors'"
            )
        payload = self.load()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tensors = {
            f"{node}.{name}": tensor
            for node, state in payload.models.items()
            for name, tensor in state.items()
        }
        if format == "safetensors":
            try:
                from safetensors.torch import save_file  # noqa: PLC0415 - optional
            except ImportError as exc:
                raise ImproperlyConfigured(
                    "safetensors export requires safetensors; install it or use "
                    "format='pt'"
                ) from exc
            save_file(tensors, str(target))
        else:
            torch.save(tensors, target)
        return target

    def describe(self, path: str | Path | None = None) -> dict[str, Any]:
        target = Path(path) if path is not None else None
        if target is None:
            record = self.latest(valid_only=False)
            target = record.path if record else None
        if target is None:
            return {"directory": str(self.directory), "run_id": self.run_id, "checkpoint": None}
        verification = self.verify(target)
        payload: CheckpointPayload | None = None
        if verification.ok:
            try:
                payload = self.load(target)
            except ComplianceError:  # pragma: no cover - verified above
                payload = None
        return {
            "directory": str(self.directory),
            "run_id": self.run_id,
            "checkpoint": str(target),
            "verified": verification.ok,
            "problems": list(verification.problems),
            "identity": dict(payload.identity) if payload else None,
            "sections": (
                {
                    name: bool(getattr(payload, name))
                    for name in CheckpointPayload.SECTIONS
                }
                if payload
                else {}
            ),
            "keep": self.keep,
            "keep_best": self.keep_best,
            "files": [record.path.name for record in self.list()],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "keep": self.keep,
            "keep_best": self.keep_best,
            "checksum": self.checksum,
            "last_sections": list(self.last_sections),
            "latest": (
                self.latest(valid_only=False).path.name
                if self.latest(valid_only=False)
                else None
            ),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        state = dict(payload or {})
        if "run_id" in state and state["run_id"] != self.run_id:
            raise ValidationError(
                f"checkpoint writer for run {self.run_id!r} cannot adopt state from "
                f"{state['run_id']!r}"
            )
        self.keep = int(state.get("keep", self.keep))
        self.keep_best = int(state.get("keep_best", self.keep_best))
        self.last_sections = tuple(str(item) for item in state.get("last_sections", []))


class ContractMismatch(ComplianceError):
    """A checkpoint cannot be applied to this trainer without changing the run."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _version() -> str:
    try:
        from .. import __version__ as version  # noqa: PLC0415

        return str(version)
    except Exception:  # noqa: BLE001 - a missing version must not break a save
        return "0.0.0"


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _atomic_save(payload: Mapping[str, Any], path: Path) -> Path:
    """``atomic_torch_save`` plus gzip when compression is on."""
    try:
        return atomic_torch_save(dict(payload), path)
    except Exception:  # pragma: no cover - surface the real error
        raise


def _has_parameters(module: Any) -> bool:
    try:
        next(iter(module.parameters()))
        return True
    except (StopIteration, AttributeError):
        return False


def _config_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    if isinstance(config, Mapping):
        return dict(config)
    return {"class": type(config).__name__}


def _warning_count(warnings: Any) -> int:
    if not isinstance(warnings, Mapping):
        return 0
    total = 0
    registries = warnings.get("registries", warnings)
    if isinstance(registries, Mapping):
        for value in registries.values():
            if isinstance(value, Mapping):
                records = value.get("records") or value.get("warnings") or {}
                total += len(records) if hasattr(records, "__len__") else 0
            elif isinstance(value, (list, tuple)):
                total += len(value)
    return total


def _topology(state: Mapping[str, Any]) -> tuple:
    return tuple(
        sorted(
            (
                str(node.get("path")),
                str(node.get("parent")),
                str(node.get("role", "")),
                str(node.get("status", "")),
            )
            for node in state.get("nodes", [])
        )
    )


def _topology_diff(current: Mapping[str, Any], incoming: Mapping[str, Any]) -> str:
    left = _topology(current)
    right = _topology(incoming)
    missing = sorted({item[0] for item in right} - {item[0] for item in left})
    extra = sorted({item[0] for item in left} - {item[0] for item in right})
    parts: list[str] = []
    if missing:
        parts.append("checkpoint-only nodes " + ", ".join(repr(item) for item in missing[:5]))
    if extra:
        parts.append("trainer-only nodes " + ", ".join(repr(item) for item in extra[:5]))
    if not parts:
        parts.append("the same paths have different parents, roles, or statuses")
    return "; ".join(parts)
