"""Per-node optimizer selection, compilation and live swaps (document 17 section 9.4-9.6)."""

from __future__ import annotations

import fnmatch
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from ..components.contracts import BIND_KEYS
from ..components.function_book import specificity
from ..components.kinds import Boundary, Origin
from ..components.reflect import spec_hash
from ..components.registry import HUB, Component
from ..components.resolver import ComponentResolver
from ..components.slots import SwapRecord
from ..exceptions import (
    ComponentLocked,
    ComponentMigrationError,
    ImproperlyConfigured,
    ValidationError,
)
from . import schedulers
from .optimizer import MACTFOptimizer, optimizer_spec
from .optimizer_migrations import migration_for

log = logging.getLogger("mactf.training.optimizer_plan")

#: Keys of the block that configure the plan, not an optimizer. ``optimizer_spec`` drops them.
PLAN_KEYS: frozenset[str] = frozenset({"OVERRIDES", "MIGRATE", "VERIFY", "STRICT_GLOBS", "PARAMS"})
#: Keys an override inherits one by one.
FACADE_KEYS: frozenset[str] = frozenset(
    {"LR", "WEIGHT_DECAY", "GRAD_ACCUMULATION", "MAX_GRAD_NORM", "SCHEDULER", "SCHEDULER_KWARGS"}
)
_BACKEND_SELECTORS: tuple[str, ...] = ("TYPE", "BACKEND", "CLASS", "CLASS_NAME", "NAME")
LEGACY_KINDS: frozenset[str] = frozenset({"torch", "scipy", "callable"})
VERIFY_MODES: tuple[str, ...] = ("custom", "all", "off")
MIGRATE_MODES: tuple[str, ...] = ("reset", "carry", "fail")
DEFAULT_SLOT = "OPTIMIZER"

#: The shipped values of the keys the sieve may drop (``conf.OPTIMIZER_DEFAULTS``).
_SHIPPED: Mapping[str, Any] = {"betas": (0.9, 0.999), "eps": 1e-8}


def _same(left: Any, right: Any) -> bool:
    """Equality that reads ``[0.9, 0.999]`` and ``(0.9, 0.999)`` as the same betas."""
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return tuple(left) == tuple(right)
    return left == right


def _upper(block: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key).upper(): value for key, value in dict(block or {}).items()}


def merge_blocks(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """The effective block of an override: the default's, changed by the override's keys."""
    over = {key: value for key, value in _upper(override).items() if key not in PLAN_KEYS - {"PARAMS"}}
    swaps_backend = any(key in over and over[key] is not None for key in _BACKEND_SELECTORS)
    merged: dict[str, Any] = {}
    for key, value in _upper(base).items():
        if key in PLAN_KEYS - {"PARAMS"}:
            continue
        if swaps_backend and key not in FACADE_KEYS:
            continue
        merged[key] = value
    for key, value in over.items():
        if key == "PARAMS":
            continue                     # merged below: they travel with the backend keys
        if value is None and key in FACADE_KEYS:
            merged.pop(key, None)        # ``SCHEDULER: null`` clears; ``LR: null`` unsets
        elif value is not None:
            merged[key] = value
    if over.get("PARAMS"):
        inherited = {} if swaps_backend else dict(merged.get("PARAMS") or {})
        merged["PARAMS"] = {**inherited, **dict(over["PARAMS"])}
    return merged


@dataclass(frozen=True)
class OptimizerEntry:
    """One compiled, validated ``OPTIMIZER`` block."""

    slot: str                                   # "OPTIMIZER" or "OPTIMIZER.<glob>"
    glob: str | None
    kind: str                                   # "block" | "code" | "inline"
    block: Mapping[str, Any]                    # effective block, UPPER keys
    payload: Mapping[str, Any]                  # ``from_spec`` payload, lower keys, sieved
    scheduler: tuple[str, dict[str, Any]] | None
    backend: str                                # registry name or legacy kind
    migrate: str | None
    spec_hash: str
    customized: bool                            # false => byte-identical to the pre-17 path
    explicit: frozenset[str]                    # facade keys the override itself set
    origin: str                                 # builtin | project | plugin | runtime

    @property
    def frozen(self) -> bool:
        return self.backend == "null"

    def describe(self) -> dict[str, Any]:
        return {
            "slot": self.slot, "kind": self.kind, "backend": self.backend,
            "scheduler": self.scheduler[0] if self.scheduler else None,
            "migrate": self.migrate, "spec_hash": self.spec_hash,
            "customized": self.customized, "origin": self.origin, "frozen": self.frozen,
        }


def _backend_component(payload: Mapping[str, Any]) -> tuple[str, Component | None]:
    """``(name, component)`` for the payload's backend; ``component`` is ``None`` only for the
    three legacy kinds that have no registry entry of their own beyond the built-ins."""
    registry = HUB.registry("optimizer")
    kind = str(payload.get("type", "torch"))
    if kind.lower() == "torch":
        name = str(payload.get("class", "Adam"))
        return name, registry.find(name)
    return kind, registry.find(kind) or registry.find(kind.lower())


def _facade_error(code: str, slot: str, message: str) -> ImproperlyConfigured:
    return ImproperlyConfigured(f"{code}: {slot}: {message}")


def compile_entry(
    slot: str,
    block: Mapping[str, Any],
    *,
    glob: str | None,
    horizon: int,
    written: frozenset[str] | None,
    explicit: frozenset[str] = frozenset(),
    migrate: str | None = None,
    kind: str = "block",
) -> OptimizerEntry:
    """Validate and freeze one block. Everything that can fail fails here, before any run.

    ``written`` is the set of keys the project itself wrote in the *default* block (the
    shipped ``AdamW``/``BETAS``/``EPS`` defaults are always present beside them). A shipped
    default the chosen backend does not accept is dropped; a written one is an error.
    ``None`` means "every key was written" (an override).
    """
    upper = _upper(block)
    extra = dict(upper.get("PARAMS") or {})
    payload = optimizer_spec({k: v for k, v in upper.items() if k != "PARAMS"})
    # A shipped default (BETAS, EPS) beside the same key under PARAMS is not a clash: the
    # project cannot have meant the default it never wrote.
    for key, shipped in _SHIPPED.items():
        if key in extra and key in payload and _same(payload[key], shipped):
            payload.pop(key)
    clash = sorted(set(extra) & set(payload))
    if clash:
        raise _facade_error("E057", slot, f"{clash} set both flat and under PARAMS")
    payload.update(extra)

    name, component = _backend_component(payload)
    plain_torch = str(payload.get("type", "torch")).lower() == "torch" and isinstance(
        getattr(torch.optim, name, None), type
    )
    if component is None and not plain_torch:
        raise ValidationError(
            f"{slot}: unknown optimizer {name!r}; accepted: torch, scipy, callable, "
            + ", ".join(HUB.registry("optimizer").names())
        )
    if component is not None:
        accepted = component.accepted_params()
        facade = {"type", "class", "lr", "learning_rate", "weight_decay", "grad_accumulation",
                  "max_grad_norm", "node_path"}
        for key, shipped in _SHIPPED.items():
            if key not in payload or key in extra or not _same(payload[key], shipped):
                continue
            # A registered backend (``Lion``, ``router``) has its own idea of ``betas``: the
            # shipped Adam default is never passed to it. A plain torch class keeps the legacy
            # sieve (``optimizer_spec`` already dropped what it does not accept).
            if not plain_torch or (
                key not in accepted and written is not None and key.upper() not in written
            ):
                payload.pop(key)
        values = {k: v for k, v in payload.items() if k not in facade}
        problems = component.check_params(values, ignore=BIND_KEYS)
        if problems:
            raise _facade_error("E057", slot, f"optimizer {name!r}: " + "; ".join(problems))

    # A frozen node has no update to schedule: an inherited SCHEDULER is simply not built for it.
    scheduler = None if name == "null" else _compile_scheduler(slot, upper, horizon)
    backend_is_torch = name in {c for c, _ in _torch_names()} or (
        component is not None
        and isinstance(component.factory, type)
        and issubclass(component.factory, torch.optim.Optimizer)
    )
    if scheduler is not None and not backend_is_torch:
        raise _facade_error(
            "E052", slot, f"scheduler {scheduler[0]!r} needs a torch backend; {name!r} has none"
        )
    accumulation = int(payload.get("grad_accumulation", 1))
    if name == "scipy" and accumulation > 1:
        raise _facade_error("E057", slot, f"scipy is full-batch and cannot use GRAD_ACCUMULATION={accumulation}")
    if migrate is not None and migrate not in MIGRATE_MODES:
        raise _facade_error("E057", slot, f"MIGRATE must be one of {list(MIGRATE_MODES)}, got {migrate!r}")

    sched_component = HUB.registry("scheduler").find(scheduler[0]) if scheduler else None
    customized = (
        slot != DEFAULT_SLOT
        or kind != "block"
        or (component is not None and component.origin is not Origin.BUILTIN)
        or str(payload.get("type", "torch")).lower() not in LEGACY_KINDS
        or (
            scheduler is not None
            and getattr(torch.optim.lr_scheduler, scheduler[0], None) is None
        )
        or (sched_component is not None and sched_component.origin is not Origin.BUILTIN)
    )
    identity = {
        "payload": {k: v for k, v in payload.items() if k != "node_path"},
        "scheduler": [scheduler[0], scheduler[1]] if scheduler else None,
    }
    return OptimizerEntry(
        slot=slot, glob=glob, kind=kind, block=dict(upper), payload=dict(payload),
        scheduler=scheduler, backend=name, migrate=migrate,
        spec_hash=spec_hash(identity), customized=customized, explicit=explicit,
        origin=(component.origin.value if component is not None else "builtin"),
    )


def _torch_names() -> Iterable[tuple[str, type]]:
    from .optimizer_builtins import _torch_classes

    return _torch_classes()


def _compile_scheduler(slot: str, upper: Mapping[str, Any], horizon: int) -> tuple[str, dict[str, Any]] | None:
    name = upper.get("SCHEDULER")
    if not name:
        return None
    kwargs = dict(upper.get("SCHEDULER_KWARGS") or {})
    name = str(name)
    registry = HUB.registry("scheduler")
    component = registry.find(name)
    if component is None and not isinstance(getattr(torch.optim.lr_scheduler, name, None), type):
        raise ValidationError(
            f"{slot}: unknown scheduler {name!r}; registered: {registry.names()}, or any class "
            "of torch.optim.lr_scheduler"
        )
    if name in schedulers.UNSUPPORTED:
        raise _facade_error("E052", slot, f"scheduler {name!r} needs a metric; it cannot be used here")
    if "total_steps" not in kwargs and schedulers.accepts_total_steps(name):
        kwargs["total_steps"] = int(horizon)
    if component is not None:
        problems = component.check_params(kwargs, ignore=frozenset({"optimizer"}))
        if problems:
            raise _facade_error("E053", slot, f"scheduler {name!r}: " + "; ".join(problems))
    # Build it once over a scratch optimizer: a value the factory rejects (``warmup_steps >=
    # total_steps``, ``drop_at < 1``) fails here, at build, and not at step 1 of a run.
    scratch = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1, momentum=0.9)
    try:
        schedulers.build_scheduler(name, scratch, kwargs)
    except Exception as error:  # noqa: BLE001 - reported as the entry's scheduler problem
        raise _facade_error("E053", slot, f"scheduler {name!r}: {type(error).__name__}: {error}") from error
    return name, kwargs



@dataclass
class StagedSwap:
    """A swap that has been built and migrated but not yet installed."""

    slot: str
    entry: OptimizerEntry | None             # None: the entry is being removed
    previous: OptimizerEntry | None
    optimizers: dict[str, MACTFOptimizer] = field(default_factory=dict)
    frozen: set[str] = field(default_factory=set)
    thawed: set[str] = field(default_factory=set)
    records: list[SwapRecord] = field(default_factory=list)


@dataclass
class StagedTopology:
    entries: dict[str, OptimizerEntry]
    previous: dict[str, OptimizerEntry]
    optimizers: dict[str, MACTFOptimizer] = field(default_factory=dict)
    frozen: set[str] = field(default_factory=set)
    thawed: set[str] = field(default_factory=set)
    dropped: list[str] = field(default_factory=list)
    records: list[SwapRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class OptimizerPlan:
    """Compiled ``OPTIMIZER`` entries plus the rules that choose and change them."""

    def __init__(
        self,
        *,
        horizon: int,
        strict_globs: bool = True,
        verify: str = "custom",
        migrate: str = "reset",
        strict_restore: bool = False,
        resolver: ComponentResolver | None = None,
        trusted: bool = True,
    ) -> None:
        if verify not in VERIFY_MODES:
            raise ImproperlyConfigured(f"OPTIMIZER.VERIFY must be one of {list(VERIFY_MODES)}, got {verify!r}")
        if migrate not in MIGRATE_MODES:
            raise ImproperlyConfigured(f"OPTIMIZER.MIGRATE must be one of {list(MIGRATE_MODES)}, got {migrate!r}")
        self.horizon = int(horizon)
        self.strict_globs = bool(strict_globs)
        self.verify = verify
        self.migrate = migrate
        self.strict_restore = bool(strict_restore)
        self._resolver = resolver or ComponentResolver()
        self._trusted = trusted
        self._entries: dict[str, OptimizerEntry] = {}
        self._pending: dict[str, tuple[Mapping[str, Any] | None, str | None]] = {}
        self._history: list[SwapRecord] = []
        #: Slots a live swap replaced, and slots it removed: the only entries a checkpoint
        #: carries, because everything else is re-derived from the configuration on resume.
        self._swapped: set[str] = set()
        self._removed: set[str] = set()
        self._lock = threading.RLock()
        self.episode = 0

    # -- construction ----------------------------------------------------
    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any],
        *,
        block: Mapping[str, Any],
        written: frozenset[str] | None,
        horizon: int,
        code_spec: Any = None,
        inline_lr: float = 1e-3,
        resolver: ComponentResolver | None = None,
        trusted: bool = True,
    ) -> "OptimizerPlan":
        """Compile the default entry and every override.

        ``block`` is ``trainer._optimizer_block()``: ``{}`` unless the project wrote the
        ``OPTIMIZER`` block. ``code_spec`` is the ``optimizer=`` constructor argument. The
        default entry is, in the order the trainer has always used: the in-code spec, else the
        block, else Adam at ``inline_lr``. Overrides apply on top of whichever it is.
        """
        raw = _upper(settings.get("OPTIMIZER") or {})
        components = dict(settings.get("COMPONENTS") or {})
        plan = cls(
            horizon=horizon,
            strict_globs=bool(raw.get("STRICT_GLOBS", True)),
            verify=str(raw.get("VERIFY", "custom")).lower(),
            migrate=str(raw.get("MIGRATE", "reset")).lower(),
            strict_restore=bool(components.get("STRICT_RESTORE", False)),
            resolver=resolver,
            trusted=trusted,
        )
        if code_spec is not None:
            plan._entries[DEFAULT_SLOT] = plan._code_entry(code_spec)
            base: Mapping[str, Any] = {}
        elif block:
            base = _upper(block)
            plan._entries[DEFAULT_SLOT] = compile_entry(
                DEFAULT_SLOT, base, glob=None, horizon=horizon, written=written,
                migrate=plan.migrate,
            )
        else:
            plan._entries[DEFAULT_SLOT] = plan._inline_entry(inline_lr)
            base = {}
        overrides = raw.get("OVERRIDES") or {}
        if not isinstance(overrides, Mapping):
            raise ImproperlyConfigured("OPTIMIZER.OVERRIDES must map a node-path glob to an OPTIMIZER block")
        # An in-code spec is opaque (``block == {}``): an override then starts from the shipped
        # defaults. An inline default exposes only its learning rate, which overrides inherit.
        default_block = base or dict(plan._entries[DEFAULT_SLOT].block)
        for glob, override in sorted(overrides.items(), key=lambda item: str(item[0])):
            plan._entries[f"{DEFAULT_SLOT}.{glob}"] = plan._override_entry(
                str(glob), override, default_block
            )
        return plan

    def _code_entry(self, spec: Any) -> OptimizerEntry:
        """The ``optimizer=`` argument. It is built by ``MACTFOptimizer.from_spec`` exactly as
        before, so it is not compiled: its hash is that of its ``describe()``."""
        probe = MACTFOptimizer.from_spec(spec) if isinstance(spec, Mapping) else None
        described = probe.describe() if probe is not None else {"class": type(spec).__name__}
        return OptimizerEntry(
            slot=DEFAULT_SLOT, glob=None, kind="code", block={}, payload={"spec": spec},
            scheduler=None, backend=str(described.get("class", "code")), migrate=self.migrate,
            spec_hash=spec_hash({"code": str(described.get("class")), "lr": described.get("lr")}),
            customized=False, explicit=frozenset(), origin="runtime",
        )

    def _inline_entry(self, lr: float) -> OptimizerEntry:
        payload = {"type": "torch", "class": "Adam", "lr": float(lr)}
        return OptimizerEntry(
            slot=DEFAULT_SLOT, glob=None, kind="inline", block={"LR": float(lr)}, payload=payload,
            scheduler=None, backend="Adam", migrate=self.migrate,
            spec_hash=spec_hash({"payload": payload}), customized=False,
            explicit=frozenset(), origin="builtin",
        )

    def _override_entry(self, glob: str, override: Any, default_block: Mapping[str, Any]) -> OptimizerEntry:
        slot = f"{DEFAULT_SLOT}.{glob}"
        if not isinstance(override, Mapping):
            raise ImproperlyConfigured(f"OPTIMIZER.OVERRIDES[{glob!r}] must be a mapping, got {type(override).__name__}")
        migrate = _upper(override).get("MIGRATE")
        explicit = frozenset(k for k in _upper(override) if k in FACADE_KEYS)
        return compile_entry(
            slot, merge_blocks(default_block, override), glob=glob, horizon=self.horizon,
            written=None, explicit=explicit,
            migrate=None if migrate is None else str(migrate).lower(),
        )

    # -- reading ---------------------------------------------------------
    @property
    def entries(self) -> Mapping[str, OptimizerEntry]:
        return dict(self._entries)

    @property
    def default(self) -> OptimizerEntry:
        return self._entries[DEFAULT_SLOT]

    @property
    def history(self) -> tuple[SwapRecord, ...]:
        return tuple(self._history)

    def customized(self) -> bool:
        return any(entry.customized for entry in self._entries.values())

    def swapped(self) -> bool:
        """True once a live swap or removal has changed the configured entries."""
        return bool(self._swapped or self._removed or self._history)

    def _candidates(self) -> list[tuple[str, OptimizerEntry]]:
        prefix = DEFAULT_SLOT + "."
        return [
            (slot[len(prefix):], entry)
            for slot, entry in sorted(self._entries.items())
            if slot.startswith(prefix)
        ]

    def select(self, path: str, entries: Mapping[str, OptimizerEntry] | None = None) -> OptimizerEntry:
        """The entry for a node: exact glob, else strictly most specific, else the default.

        Two *different* matching globs of equal specificity are ``E051``. The same rule as
        ``FunctionBook._select`` (8.4), so a reader learns it once.
        """
        pool = dict(self._entries if entries is None else entries)
        best: tuple[int, str, OptimizerEntry] | None = None
        tied: list[str] = []
        prefix = DEFAULT_SLOT + "."
        for slot, entry in sorted(pool.items()):
            if not slot.startswith(prefix):
                continue
            glob = slot[len(prefix):]
            if glob == path:
                return entry
            if not fnmatch.fnmatchcase(path, glob):
                continue
            score = specificity(glob)
            if best is None or score > best[0]:
                best, tied = (score, glob, entry), []
            elif score == best[0]:
                tied.append(glob)
        if best is not None:
            if tied:
                raise ImproperlyConfigured(
                    f"E051: node {path!r} is matched by equally specific optimizer overrides "
                    f"{sorted([best[1], *tied])}"
                )
            return best[2]
        return pool[DEFAULT_SLOT]

    def instantiate(self, path: str, entry: OptimizerEntry | None = None) -> MACTFOptimizer:
        """A fresh, unbound facade for ``path``. The trainer finishes and binds it (9.7)."""
        entry = entry or self.select(path)
        if entry.kind == "code":
            optimizer = MACTFOptimizer.from_spec(entry.payload["spec"])
        elif entry.kind == "inline":
            optimizer = MACTFOptimizer.from_spec(dict(entry.payload))
        else:
            payload = dict(entry.payload)
            payload["node_path"] = path
            optimizer = MACTFOptimizer.from_spec(payload)
            optimizer.plan_scheduler(entry.scheduler)
        if entry.customized:
            optimizer.spec_hash = entry.spec_hash
            optimizer.slot = entry.slot
        optimizer.strict_restore = self.strict_restore
        return optimizer

    # -- validation ------------------------------------------------------
    def validate(self, node_paths: Sequence[str], *, initial: bool = True) -> list[str]:
        """Return warnings; raise ``ImproperlyConfigured`` for errors. Run at build and after a mutation.

        * ``E050`` an override glob matches no node (``W011`` after a mutation, or with
          ``STRICT_GLOBS: false``);
        * ``E051`` two overrides of equal specificity match one node;
        * ``E054`` every node would be frozen, so nothing could learn;
        * ``W016`` a ``null`` entry selects the only node of a group (it would never be updated).
        """
        warnings_: list[str] = []
        paths = list(node_paths)
        for glob, entry in self._candidates():
            if not any(fnmatch.fnmatchcase(path, glob) or glob == path for path in paths):
                message = f"optimizer override {glob!r} matches no node of {paths[:6]}..."
                if initial and self.strict_globs:
                    raise ImproperlyConfigured(f"E050: {message}")
                warnings_.append(f"W011: {message}")
        chosen = {path: self.select(path) for path in paths}      # raises E051
        if paths and all(entry.frozen for entry in chosen.values()):
            raise ImproperlyConfigured(
                "E054: every node resolves to the null optimizer, so nothing in this run can learn"
            )
        groups: dict[str, list[str]] = {}
        for path in paths:
            groups.setdefault(path.rpartition("/")[0], []).append(path)
        for parent, members in sorted(groups.items()):
            if len(members) == 1 and chosen[members[0]].frozen and len(groups) > 1:
                warnings_.append(
                    f"W016: {members[0]} is the only node of its group "
                    f"({parent or 'root'}) and its optimizer is null, so it is never updated"
                )
        return warnings_

    def frozen_paths(self, node_paths: Iterable[str]) -> list[str]:
        return [path for path in node_paths if self.select(path).frozen]

    # -- changing a plan live --------------------------------------------
    def _boundary_allowed(self, slot: str, boundary: Boundary) -> None:
        if boundary.rank > Boundary.EPISODE.rank:
            raise ComponentLocked(
                "optimizer", slot,
                f"an optimizer may change at run_start or an episode boundary, not {boundary.value}",
            )
        from ..components.slots import slot_locked_by

        if slot_locked_by(slot, HUB.effective_policy().locked):
            raise ComponentLocked("optimizer", slot, "the slot is locked by COMPONENTS.LOCKED")

    def request(self, slot: str, block: Mapping[str, Any] | None, *, mode: str | None = None) -> None:
        """Queue a swap (or, with ``block=None``, a removal) for the next episode boundary.

        Validated eagerly: an invalid block fails the request, not the next episode.
        """
        with self._lock:
            self._boundary_allowed(slot, Boundary.EPISODE)
            if block is not None:
                self._compile_for(slot, block, mode)
            elif slot == DEFAULT_SLOT:
                raise ImproperlyConfigured("E054: the default optimizer entry cannot be removed; replace it")
            elif slot not in self._entries:
                raise ComponentLocked("optimizer", slot, f"no such entry; entries: {sorted(self._entries)}")
            if mode is not None and mode not in MIGRATE_MODES:
                raise ComponentMigrationError(f"unknown migration {mode!r}; accepted: {list(MIGRATE_MODES)}")
            self._pending[slot] = (None if block is None else dict(block), mode)

    def pending(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                slot: {"block": None if block is None else dict(block), "migrate": mode}
                for slot, (block, mode) in sorted(self._pending.items())
            }

    def drop_pending(self, slot: str) -> bool:
        with self._lock:
            return self._pending.pop(slot, None) is not None

    def pending_slots(self) -> list[str]:
        with self._lock:
            return sorted(self._pending)

    def take_pending(self, slot: str) -> tuple[Mapping[str, Any] | None, str | None]:
        with self._lock:
            return self._pending.pop(slot)

    def _compile_for(self, slot: str, block: Mapping[str, Any], mode: str | None) -> OptimizerEntry:
        if slot == DEFAULT_SLOT:
            return compile_entry(
                slot, merge_blocks({}, block), glob=None, horizon=self.horizon, written=None,
                migrate=mode or self.default.migrate,
            )
        glob = slot[len(DEFAULT_SLOT) + 1:]
        if not slot.startswith(DEFAULT_SLOT + ".") or not glob:
            raise ComponentLocked("optimizer", slot, "an optimizer slot is OPTIMIZER or OPTIMIZER.<glob>")
        return self._override_entry(glob, {**dict(block), **({"MIGRATE": mode} if mode else {})}, self.default.block)

    def stage(
        self,
        slot: str,
        block: Mapping[str, Any] | None,
        *,
        boundary: Boundary,
        node_paths: Sequence[str],
        live: Mapping[str, MACTFOptimizer],
        make: Callable[[str, OptimizerEntry], MACTFOptimizer],
        mode: str | None = None,
    ) -> StagedSwap:
        """Build and migrate every affected node's facade. Mutates nothing that is live.

        ``make(path, entry)`` is the trainer's ``_make_optimizer``: it builds, finishes and
        binds a facade over the node's parameters. ``live`` is ``trainer.optimizers``.
        A failure anywhere (a lock, a bad block, a refusing migration, a bind error) raises
        before any facade is installed, so the old optimizers keep running (invariant O6).
        """
        with self._lock:
            self._boundary_allowed(slot, boundary)
            previous = self._entries.get(slot)
            if block is None:
                if previous is None:
                    raise ComponentLocked("optimizer", slot, f"no such entry; entries: {sorted(self._entries)}")
                if slot == DEFAULT_SLOT:
                    raise ImproperlyConfigured("E054: the default optimizer entry cannot be removed; replace it")
                entry = None
                after = {k: v for k, v in self._entries.items() if k != slot}
            else:
                entry = self._compile_for(slot, block, mode)
                after = {**self._entries, slot: entry}
            staged = StagedSwap(slot=slot, entry=entry, previous=previous)
            affected: list[str] = []
            for path in node_paths:
                old_entry = self.select(path)
                new_entry = self.select(path, after)               # may raise E051
                if old_entry.spec_hash == new_entry.spec_hash and old_entry.slot == new_entry.slot:
                    continue
                affected.append(path)
                if new_entry.frozen and not old_entry.frozen:
                    staged.frozen.add(path)
                if old_entry.frozen and not new_entry.frozen:
                    staged.thawed.add(path)
            if node_paths and all(self.select(path, after).frozen for path in node_paths):
                raise ImproperlyConfigured("E054: the change would leave no node able to learn")
            chosen = migration_for(mode or (entry.migrate if entry is not None else None) or self.migrate)
            for path in affected:
                new_entry = self.select(path, after)
                old = live.get(path)
                fresh = make(path, new_entry)
                if old is not None:
                    chosen(old, fresh)
                staged.optimizers[path] = fresh
                staged.records.append(
                    SwapRecord(
                        slot=slot, kind_id="optimizer",
                        from_name=old.describe().get("class", "none") if old is not None else "none",
                        to_name=fresh.describe().get("class", "none"),
                        from_spec_hash=getattr(old, "spec_hash", "") if old is not None else "",
                        to_spec_hash=new_entry.spec_hash, boundary=boundary.value,
                        episode=self.episode, at=time.time(),
                    )
                )
            return staged

    def commit(self, staged: StagedSwap) -> None:
        """Make a staged swap the plan's truth. Cannot fail: it only assigns."""
        with self._lock:
            if staged.entry is None:
                self._entries.pop(staged.slot, None)
                self._swapped.discard(staged.slot)
                self._removed.add(staged.slot)
            else:
                self._entries[staged.slot] = staged.entry
                self._swapped.add(staged.slot)
                self._removed.discard(staged.slot)
            self._history.extend(staged.records)
            self._pending.pop(staged.slot, None)

    def stage_topology(
        self,
        *,
        set_blocks: Mapping[str, Mapping[str, Any] | None],
        build: Sequence[str],
        drop: Sequence[str],
        active: Sequence[str],
        live: Mapping[str, MACTFOptimizer],
        make: Callable[[str, OptimizerEntry], MACTFOptimizer],
    ) -> StagedTopology:
        """Stage the optimizer side of a mutation. Mutates nothing that is live.

        ``set_blocks`` maps a node path to its new per-node block (the slot ``OPTIMIZER.<path>``),
        or to ``None`` to remove that override. ``build`` lists nodes that need a fresh facade
        regardless (new or rebuilt peers: their old state, if any, is meaningless). ``drop``
        lists purged nodes, whose per-node slot is removed. ``active`` is every node that will
        be able to learn afterwards. ``make(path, entry)`` is ``_make_optimizer`` bound to the
        staged peer.
        """
        with self._lock:
            previous = dict(self._entries)
            entries = dict(previous)
            for path in drop:
                entries.pop(f"{DEFAULT_SLOT}.{path}", None)
            for path, block in set_blocks.items():
                slot = f"{DEFAULT_SLOT}.{path}"
                if block is None:
                    entries.pop(slot, None)
                else:
                    entries[slot] = self._compile_for(slot, block, None)
            staged = StagedTopology(entries=entries, previous=previous, dropped=list(drop))
            fresh_paths = [path for path in dict.fromkeys([*build, *set_blocks]) if path in set(active) | set(build)]
            for path in fresh_paths:
                old_entry = self.select(path) if path not in build else None
                new_entry = self.select(path, entries)                     # may raise E051
                is_new = path in build
                if not is_new and old_entry is not None and (
                    old_entry.spec_hash == new_entry.spec_hash and old_entry.slot == new_entry.slot
                ):
                    continue
                old = live.get(path)
                fresh = make(path, new_entry)
                if old is not None and not is_new:
                    migration_for(new_entry.migrate)(old, fresh)
                staged.optimizers[path] = fresh
                if new_entry.frozen and (is_new or old_entry is None or not old_entry.frozen):
                    staged.frozen.add(path)
                if not new_entry.frozen and old_entry is not None and old_entry.frozen:
                    staged.thawed.add(path)
                staged.records.append(
                    SwapRecord(
                        slot=new_entry.slot, kind_id="optimizer",
                        from_name=old.describe().get("class", "none") if old is not None else "none",
                        to_name=fresh.describe().get("class", "none"),
                        from_spec_hash=getattr(old, "spec_hash", "") if old is not None else "",
                        to_spec_hash=new_entry.spec_hash, boundary=Boundary.EPISODE.value,
                        episode=self.episode, at=time.time(),
                    )
                )
            learners = [
                path for path in active
                if not self.select(path, entries).frozen
            ]
            if active and not learners:
                raise ImproperlyConfigured("E054: the topology change would leave no node able to learn")
            prefix = DEFAULT_SLOT + "."
            for slot in sorted(entries):
                if not slot.startswith(prefix):
                    continue
                glob = slot[len(prefix):]
                if not any(path == glob or fnmatch.fnmatchcase(path, glob) for path in active):
                    staged.warnings.append(
                        f"W011: optimizer override {glob!r} matches no node after the mutation"
                    )
            return staged

    def commit_topology(self, staged: StagedTopology) -> Callable[[], None]:
        """Make a staged topology change the plan's truth. Returns the function that undoes it."""
        prefix = DEFAULT_SLOT + "."
        with self._lock:
            history_length = len(self._history)
            previous = dict(self._entries)
            swapped, removed = set(self._swapped), set(self._removed)
            self._entries = dict(staged.entries)
            self._history.extend(staged.records)
            # A per-node block is "what the run did": it is what the checkpoint carries (12.9.6).
            for slot, entry in staged.entries.items():
                if slot.startswith(prefix) and (slot not in previous or previous[slot].spec_hash != entry.spec_hash):
                    self._swapped.add(slot)
                    self._removed.discard(slot)
            for slot in previous:
                if slot.startswith(prefix) and slot not in staged.entries:
                    self._swapped.discard(slot)
                    self._removed.add(slot)

        def undo() -> None:
            with self._lock:
                self._entries = previous
                self._swapped, self._removed = swapped, removed
                del self._history[history_length:]

        return undo

    # -- persistence -----------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """The entries a live swap changed, and the ones it removed, with the swap history.

        Only what the run *did* is stored. An entry that was never swapped is re-derived from
        the configuration on resume, so editing ``conf/optimizer.yaml`` between save and
        resume takes effect (per node, :meth:`MACTFOptimizer._reconcile` decides what state
        survives), while a swap made at run time is not silently undone by the file.
        """
        return {
            "version": 1,
            "entries": {
                slot: {"block": dict(entry.block), "migrate": entry.migrate, "spec_hash": entry.spec_hash}
                for slot, entry in sorted(self._entries.items())
                if slot in self._swapped and entry.kind == "block"
            },
            "removed": sorted(self._removed),
            "history": [record.as_dict() for record in self._history],
            "episode": self.episode,
        }

    def load_state_dict(self, payload: Mapping[str, Any], *, strict: bool = False) -> list[str]:
        """Re-apply the swaps of the checkpointed run so a resume continues under what it had.

        A swapped entry whose stored hash equals the configured one is left alone; one that
        differs replaces the configured entry (the checkpoint wins over the file for a swap,
        as for functions, 8.8.5) and ``W015`` says so; with ``strict`` it is ``E055``. A slot
        the run removed stays removed. Returns the warnings.
        """
        notes: list[str] = []
        data = dict(payload or {})
        stored = dict(data.get("entries") or {})
        for slot, entry in sorted(stored.items()):
            current = self._entries.get(slot)
            if current is not None and current.spec_hash == entry.get("spec_hash"):
                self._swapped.add(slot)
                continue
            message = (
                f"optimizer entry {slot!r} was swapped during the checkpointed run "
                f"(saved spec {entry.get('spec_hash')}, configured "
                f"{getattr(current, 'spec_hash', 'none')})"
            )
            if strict:
                raise ImproperlyConfigured(f"E055: {message}; COMPONENTS.STRICT_RESTORE is on")
            notes.append(f"W015: {message}; the checkpoint's entry is used")
            self._entries[slot] = self._compile_for(slot, entry["block"], entry.get("migrate"))
            self._swapped.add(slot)
            self._removed.discard(slot)
        for slot in data.get("removed") or ():
            if slot == DEFAULT_SLOT or slot not in self._entries:
                continue
            message = f"optimizer entry {slot!r} was removed during the checkpointed run"
            if strict:
                raise ImproperlyConfigured(f"E055: {message}; COMPONENTS.STRICT_RESTORE is on")
            notes.append(f"W015: {message}; it stays removed")
            del self._entries[slot]
            self._removed.add(slot)
        if not self._history:
            for record in data.get("history") or ():
                self._history.append(SwapRecord(**dict(record)))
        self.episode = int(data.get("episode", self.episode))
        return notes


_PROBE_FINDINGS = {
    "O001": "bind raised",
    "O002": "step raised",
    "O003": "a parameter became non-finite after a step",
    "O004": "state_dict is not a mapping or cannot be serialised with torch.save",
    "O005": "load_state_dict did not accept the optimizer's own state_dict",
    "O006": "learning_rate is not a finite float",
}


def verify_entry(entry: OptimizerEntry) -> None:
    """Probe ``entry`` on a scratch module; raise ``E056`` listing every finding."""
    import io

    if entry.frozen or entry.backend in {"scipy", "callable"}:
        return                      # closure-driven or user-function backends cannot be stepped blind
    findings: list[str] = []
    module = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Tanh(), torch.nn.Linear(2, 1))
    optimizer = MACTFOptimizer.from_spec({**dict(entry.payload), "node_path": "probe"})
    optimizer.plan_scheduler(entry.scheduler)
    try:
        optimizer.bind(list(module.parameters()), named=list(module.named_parameters()))
    except Exception as error:  # noqa: BLE001 - reported as a finding
        raise ImproperlyConfigured(f"E056: {entry.slot}: O001 {_PROBE_FINDINGS['O001']}: {error}") from error
    generator = torch.Generator().manual_seed(0)
    batch = torch.randn(8, 3, generator=generator)
    for _ in range(2):
        optimizer.zero_grad()
        module(batch).pow(2).mean().backward()
        try:
            optimizer.step()
        except Exception as error:  # noqa: BLE001
            findings.append(f"O002 {_PROBE_FINDINGS['O002']}: {error}")
            break
        if any(not torch.isfinite(p).all() for p in module.parameters()):
            findings.append(f"O003 {_PROBE_FINDINGS['O003']}")
            break
    state = optimizer.backend.state_dict()
    if not isinstance(state, Mapping):
        findings.append(f"O004 {_PROBE_FINDINGS['O004']}")
    else:
        try:
            torch.save(state, io.BytesIO())
        except Exception as error:  # noqa: BLE001
            findings.append(f"O004 {_PROBE_FINDINGS['O004']}: {error}")
        try:
            optimizer.backend.load_state_dict(state)
        except Exception as error:  # noqa: BLE001
            findings.append(f"O005 {_PROBE_FINDINGS['O005']}: {error}")
    rate = optimizer.backend.learning_rate
    if not isinstance(rate, (int, float)) or isinstance(rate, bool) or rate != rate or rate in (float("inf"), float("-inf")):
        findings.append(f"O006 {_PROBE_FINDINGS['O006']}: {rate!r}")
    if findings:
        raise ImproperlyConfigured(f"E056: {entry.slot} ({entry.backend}): " + "; ".join(findings))
