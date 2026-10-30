"""StrategyPlan: the ``peer_strategy`` slots of one run (document 17 section 11.5)."""

from __future__ import annotations

import copy
import fnmatch
import logging
import threading
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..coopetition.strategy import (
    BUILTIN_STRATEGY_TYPES,
    GroupRef,
    HierarchicalStrategy,
    KappaSoftmaxStrategy,
    TreeView,
    coordinates_via_context,
    defines_nothing,
    legacy_dispatch,
    traits_of,
)
from ..exceptions import (
    ComponentLocked,
    ComponentMissing,
    ImproperlyConfigured,
)
from .kinds import Boundary
from .resolver import ComponentResolver, ComponentSpec, Resolved
from .slots import Slot, SlotTable, SwapRecord
from .weight_plan import ROOT_LABEL, group_label, pick_glob

log = logging.getLogger("mactf.strategy")

KIND = "peer_strategy"
DEFAULT_PATH = "TRAINER.PEER_STRATEGY"
OVERRIDE_PREFIX = "TRAINER.PEER_STRATEGY_OVERRIDES."
MIGRATE_MODES = ("reset", "carry", "fail")
FRAMEWORK_DEFAULT = "kappa_softmax"                     # TRAINER_DEFAULTS["PEER_STRATEGY"]
_REMOVE = object()


def _lowered(raw: Any) -> Any:
    """The raw spec with the strategy name lower-cased, as ``build_peer_strategy`` always did."""
    if isinstance(raw, str):
        return raw.strip().lower()
    if isinstance(raw, Mapping):
        payload = dict(raw)
        written = payload.pop("name", None)
        if "type" not in payload and "import" not in payload and written is not None:
            payload["type"] = written
        if "type" in payload:
            payload["type"] = str(payload["type"]).strip().lower()
        return payload
    return raw


def default_authored(settings: Mapping[str, Any]) -> bool:
    """Whether the project wrote ``TRAINER.PEER_STRATEGY`` (see 11.5, "What the default is")."""
    from ..conf_files import authored_keys

    written = authored_keys(settings, "TRAINER")
    if written is not None:
        return "PEER_STRATEGY" in written
    value = dict(settings.get("TRAINER") or {}).get("PEER_STRATEGY")
    if isinstance(value, str):
        return value.strip().lower() != FRAMEWORK_DEFAULT
    return value is not None


class StrategyPlan:
    """The ``peer_strategy`` slots of one run and the rules that choose and change them."""

    def __init__(
        self,
        slots: SlotTable,
        *,
        resolver: ComponentResolver | None = None,
        trusted: bool = True,
        guard: Any = None,
        strict_globs: bool = True,
        migrate: str = "reset",
        strict_restore: bool = False,
        hierarchy: bool = False,
    ) -> None:
        from ..components.guards.strategy import StrategyGuard

        if migrate not in MIGRATE_MODES:
            raise ImproperlyConfigured(
                f"E070: TRAINER.PEER_STRATEGY_OPTIONS.MIGRATE must be one of {list(MIGRATE_MODES)}, got {migrate!r}"
            )
        self._slots = slots
        self._resolver = resolver or ComponentResolver()
        self._trusted = trusted
        self.guard = guard or StrategyGuard()
        self.strict_globs = bool(strict_globs)
        self.migrate = migrate
        self.strict_restore = bool(strict_restore)
        self.hierarchy = bool(hierarchy)
        self._adopted: dict[str, Any] = {}
        self._pending: dict[str, tuple[Any, str | None]] = {}
        self._group_cache: dict[tuple[str, str], tuple[Resolved, Any]] = {}
        self._restore: dict[str, Mapping[str, Any]] = {}
        self._lock = threading.RLock()
        self.last_audit: list[Any] = []
        #: Errors from ``attach`` after a committed change; the swap is done, so they are
        #: reported (the trainer drains them into ``failures``), never raised.
        self.errors: list[str] = []

    # -- construction ----------------------------------------------------
    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any],
        slots: SlotTable,
        *,
        code_spec: Any = None,
        hierarchy: bool = False,
        resolver: ComponentResolver | None = None,
        trusted: bool = True,
    ) -> "StrategyPlan":
        from ..components.guards.strategy import StrategyGuard

        trainer = dict(settings.get("TRAINER") or {})
        options = {str(k).upper(): v for k, v in dict(trainer.get("PEER_STRATEGY_OPTIONS") or {}).items()}
        unknown = sorted(set(options) - {"STRICT_GLOBS", "MIGRATE"})
        if unknown:
            raise ImproperlyConfigured(
                f"E070: TRAINER.PEER_STRATEGY_OPTIONS has unknown keys {unknown}; accepted: MIGRATE, STRICT_GLOBS"
            )
        components = dict(settings.get("COMPONENTS") or {})
        plan = cls(
            slots,
            resolver=resolver,
            trusted=trusted,
            guard=StrategyGuard.from_settings(trainer.get("STRATEGY_GUARD")),
            strict_globs=bool(options.get("STRICT_GLOBS", True)),
            migrate=str(options.get("MIGRATE", "reset")).lower(),
            strict_restore=bool(components.get("STRICT_RESTORE", False)),
            hierarchy=hierarchy,
        )
        if code_spec is not None and not isinstance(code_spec, (str, Mapping, ComponentSpec)):
            plan.adopt_default(code_spec)
        elif code_spec is not None:
            plan.install_default(code_spec)
        elif default_authored(settings):
            plan.install_default(trainer.get("PEER_STRATEGY"))
        else:
            plan.adopt_framework_default()
        overrides = trainer.get("PEER_STRATEGY_OVERRIDES") or {}
        if not isinstance(overrides, Mapping):
            raise ImproperlyConfigured("E070: TRAINER.PEER_STRATEGY_OVERRIDES must be a mapping of glob to spec")
        for glob, raw in overrides.items():
            plan.install_override(str(glob), raw)
        return plan

    def install_default(self, raw: Any) -> Slot:
        return self._slots.install(DEFAULT_PATH, KIND, _lowered(raw))

    def adopt_framework_default(self) -> Slot:
        """The default nothing was written for: ``hierarchical`` on a hierarchy, else ``kappa_softmax``.

        Built with a plain call, not through the registry, so that a default run makes no
        component-registry call at all (19.4, condition 4); the slot still records the
        registered built-in, so its origin and fingerprint are the registry's.
        """
        name = "hierarchical" if self.hierarchy else FRAMEWORK_DEFAULT
        instance = HierarchicalStrategy() if self.hierarchy else KappaSoftmaxStrategy()
        return self._slots.adopt_registered(DEFAULT_PATH, KIND, instance, name=name)

    def adopt_default(self, instance: Any) -> Slot:
        slot = self._slots.adopt(DEFAULT_PATH, KIND, instance, name=type(instance).__name__)
        self._adopted[DEFAULT_PATH] = instance
        return slot

    def install_override(self, glob: str, raw: Any) -> Slot:
        if not glob or glob != glob.strip():
            raise ImproperlyConfigured(f"E070: invalid override glob {glob!r}")
        path = OVERRIDE_PREFIX + glob
        if isinstance(raw, (str, Mapping, ComponentSpec)):
            return self._slots.install(path, KIND, _lowered(raw))
        slot = self._slots.adopt(path, KIND, raw, name=type(raw).__name__)
        self._adopted[path] = raw
        return slot

    # -- reading ---------------------------------------------------------
    def override_globs(self) -> list[str]:
        return sorted(p[len(OVERRIDE_PREFIX):] for p in self._slots.paths()
                      if p.startswith(OVERRIDE_PREFIX))

    def select(self, label: str) -> str:
        """The slot path serving ``label``. ``E071`` on an ambiguous tie."""
        glob = pick_glob(label, self.override_globs(), what="peer_strategy overrides", code="E071")
        return DEFAULT_PATH if glob is None else OVERRIDE_PREFIX + glob

    def default_instance(self) -> Any:
        return self._slots.get(DEFAULT_PATH).instance

    def customized(self) -> bool:
        """Whether anything differs from what a legacy run would have built."""
        if any(p.startswith(OVERRIDE_PREFIX) for p in self._slots.paths()) or self.history():
            return True
        if DEFAULT_PATH in self._adopted:
            return not any(type(self._adopted[DEFAULT_PATH]) is built for built in BUILTIN_STRATEGY_TYPES)
        resolved = self._slots.get(DEFAULT_PATH).resolved
        return resolved.component.origin.value != "builtin"

    def history(self) -> list[SwapRecord]:
        return [r for r in self._slots.history if r.slot.startswith(DEFAULT_PATH)]

    def instance_for(self, label: str) -> Any:
        """The instance serving ``label``. One per slot, or one per group for ``per_group``."""
        with self._lock:
            path = self.select(label)
            slot = self._slots.get(path)
            if not traits_of(slot.instance).per_group:
                return slot.instance
            key = (path, label)
            cached = self._group_cache.get(key)
            if cached is not None and cached[0] is slot.resolved:
                return cached[1]
            if path in self._adopted:
                clone = getattr(self._adopted[path], "clone", None)
                instance = clone() if callable(clone) else self._adopted[path]
            else:
                instance = self._resolver.resolve(slot.resolved.spec, trusted=self._trusted).instance
            self._seed_state(label, path, instance)
            self._group_cache[key] = (slot.resolved, instance)
            return instance

    def live_instances(self) -> list[Any]:
        """Every distinct live instance, default first, then overrides by glob."""
        seen: set[int] = set()
        out: list[Any] = []
        paths = [DEFAULT_PATH] + [OVERRIDE_PREFIX + g for g in self.override_globs()]
        with self._lock:
            for path in paths:
                instances = [self._slots.get(path).instance]
                instances += [inst for (p, _l), (_r, inst) in sorted(self._group_cache.items()) if p == path]
                for instance in instances:
                    if id(instance) not in seen:
                        seen.add(id(instance))
                        out.append(instance)
        return out

    def reaches(self, path: str, instance: Any, group: GroupRef) -> bool:
        """The reach rule of 11.5."""
        if path != DEFAULT_PATH:
            return True
        return group.is_root or traits_of(instance).walk == "tree"

    def _seed_state(self, label: str, path: str, instance: Any) -> None:
        entry = self._restore.pop(f"{path}|{label}", None)
        if entry is None:
            return
        load = getattr(instance, "load_state_dict", None)
        if not callable(load) or entry.get("class") != type(instance).__name__:
            if entry.get("state"):
                log.warning("W031: state saved for %s at %s does not fit %s; dropped",
                            label, path, type(instance).__name__)
            return
        try:
            load(copy.deepcopy(entry["state"]))
        except Exception as error:  # noqa: BLE001 - a resume must not end on one group's stale state
            log.warning("W031: state saved for %s at %s could not be loaded (%s: %s); dropped",
                        label, path, type(error).__name__, error)

    # -- the epoch -------------------------------------------------------
    def forget_groups(self, labels: Iterable[str]) -> Callable[[], None]:
        """Drop the cached per-group strategy instances of groups that changed. Returns the inverse."""
        wanted = set(labels)
        with self._lock:
            dropped = {key: self._group_cache.pop(key) for key in list(self._group_cache) if key[1] in wanted}

        def undo() -> None:
            with self._lock:
                self._group_cache.update(dropped)

        return undo

    def epoch_start(self, trainer: Any, epoch: int, view: TreeView | None = None) -> None:
        if view is not None:                       # per-group instances exist before they are called
            for group in view.groups:
                self.instance_for(group.label)
        for instance in self.live_instances():
            instance.on_epoch_start(trainer, epoch)

    def on_peer_step(self, trainer: Any, node: Any, result: Any) -> None:
        label = group_label(node.parent)
        self.instance_for(label).on_peer_step(trainer, node, result)

    def run(self, trainer: Any, epoch: int, view: TreeView) -> Any:
        """Step 6 of ``run_epoch``: coordinate every group. Returns the epoch's root report."""
        from ..training.coordination import CoordinationContext, CoordinationState

        trainer._epoch_report = None
        state = CoordinationState(epoch)
        served: list[tuple[GroupRef, str, Any]] = []
        for group in reversed(view.groups):
            path = self.select(group.label)
            instance = self.instance_for(group.label)
            if self.reaches(path, instance, group):
                served.append((group, path, instance))

        done: set[int] = set()
        for _group, _path, instance in served:
            if not coordinates_via_context(instance) and id(instance) not in done:
                done.add(id(instance))
                legacy_dispatch(trainer, epoch, instance)

        root = next((entry for entry in served if entry[0].is_root), None)
        if root is not None and coordinates_via_context(root[2]) and traits_of(root[2]).precompute_root:
            CoordinationContext(trainer, state, (root[0],), owner=self._owner(root[1], root[2]),
                                checks=False).update_root()

        for group, path, instance in served:
            if not coordinates_via_context(instance):
                continue
            owner = self._owner(path, instance)
            covered = self.guard.covers(instance)
            ctx = CoordinationContext(trainer, state, (group,), owner=owner, checks=covered)
            if covered:
                self.guard.run(owner, instance, ctx, state, backend=trainer)
            else:
                instance.coordinate(ctx)
        self.last_audit = list(state.events)
        return trainer._epoch_report

    @staticmethod
    def _owner(path: str, instance: Any) -> str:
        return str(getattr(instance, "name", None) or type(instance).__name__)

    # -- validation ------------------------------------------------------
    def validate(self, view: TreeView, *, initial: bool) -> list[str]:
        """Check the plan against the groups that exist. Returns warnings; raises on errors.

        ``E070`` an override matches no group at build, or two groups share a label;
        ``E071`` ambiguous overrides; ``E072`` a legacy strategy used where only a context-driven
        one can work; ``E074`` a strategy's ``validate`` rejected the tree, or it defines no entry point.
        """
        labels = list(view.labels())
        if len(set(labels)) != len(labels):
            duplicates = sorted({label for label in labels if labels.count(label) > 1})
            raise ImproperlyConfigured(
                f"E070: aggregation group labels collide: {duplicates}; the label {ROOT_LABEL!r} is "
                "reserved for the top group and cannot also be a node path with children"
            )
        notes: list[str] = []
        for glob in self.override_globs():
            if any(label == glob or fnmatch.fnmatchcase(label, glob) for label in labels):
                continue
            message = f"override {glob!r} matches no aggregation group; groups are {labels}"
            if initial and self.strict_globs:
                raise ImproperlyConfigured(f"E070: TRAINER.PEER_STRATEGY_OVERRIDES: {message}")
            notes.append(f"W033: {message}")
        for group in view.groups:
            self.select(group.label)                                       # E071
        self._check_legacy_placement()
        for path in [DEFAULT_PATH] + [OVERRIDE_PREFIX + g for g in self.override_globs()]:
            self._check_instance(path, self._slots.get(path).instance, view)
            if initial:
                self.probe(path, view=view)
        for note in notes:
            log.warning(note)
        return notes

    def _check_legacy_placement(self, candidate: Any = None, path: str | None = None) -> None:
        overrides = bool(self.override_globs()) or (path is not None and path != DEFAULT_PATH)
        for slot_path in [DEFAULT_PATH] + [OVERRIDE_PREFIX + g for g in self.override_globs()]:
            instance = candidate if (candidate is not None and slot_path == path) else self._slots.get(slot_path).instance
            if coordinates_via_context(instance):
                continue
            if slot_path != DEFAULT_PATH:
                raise ImproperlyConfigured(
                    f"E072: {slot_path} holds {type(instance).__name__}, which does not define coordinate(ctx); "
                    "a strategy that serves one group needs the context API (legacy strategies drive the whole trainer)"
                )
            if overrides:
                raise ImproperlyConfigured(
                    f"E072: {DEFAULT_PATH} holds {type(instance).__name__}, which does not define coordinate(ctx), "
                    "and overrides exist; a legacy strategy cannot share a tree with per-group strategies"
                )

    def _check_instance(self, path: str, instance: Any, view: TreeView) -> None:
        if defines_nothing(instance):
            raise ImproperlyConfigured(
                f"E074: {path}: {type(instance).__name__} defines neither coordinate(ctx) nor on_epoch_end(); "
                "it would do nothing"
            )
        check = getattr(instance, "validate", None)
        if not callable(check):
            return
        problems = [str(item) for item in (check(view) or ())]
        if problems:
            raise ImproperlyConfigured(f"E074: {path}: " + "; ".join(problems))

    def probe(self, path: str | None = None, *, view: TreeView, instance: Any = None,
              spec: ComponentSpec | None = None) -> None:
        """Run the section 11.8 probe on a slot's strategy (or a candidate), over the real tree's shape."""
        from ..coopetition.strategy_probe import verify_peer_strategy

        if not self.guard.probe:
            return
        target = path or DEFAULT_PATH
        live = instance if instance is not None else self._slots.get(target).instance
        if not self.guard.covers(live):
            return
        verify_peer_strategy(lambda: self._scratch(target, instance, spec), slot=target, view=view)

    def _scratch(self, path: str, instance: Any = None, spec: ComponentSpec | None = None) -> Any:
        if instance is not None and spec is None:
            clone = getattr(instance, "clone", None)
            if callable(clone):
                return clone()
            try:
                return copy.deepcopy(instance)
            except Exception:  # noqa: BLE001 - a user object that cannot be copied is probed as is
                return instance
        if spec is not None:
            return self._resolver.resolve(spec, trusted=self._trusted).instance
        if path in self._adopted:
            return self._scratch(path, self._adopted[path])
        return self._resolver.resolve(self._slots.get(path).resolved.spec, trusted=self._trusted).instance

    # -- live changes ----------------------------------------------------
    def _mode(self, mode: str | None) -> str:
        chosen = (mode or self.migrate).lower()
        if chosen not in MIGRATE_MODES:
            raise ImproperlyConfigured(f"E074: migrate must be one of {list(MIGRATE_MODES)}, got {mode!r}")
        return chosen

    def set(
        self,
        glob: str | None,
        raw: Any,
        *,
        view: TreeView,
        host: Any = None,
        boundary: Boundary = Boundary.EPISODE,
        mode: str | None = None,
    ) -> SwapRecord | None:
        """Replace the default (``glob=None``), replace one override, or add an override. Now.

        Order is the atomicity argument (T10): everything that can fail (parse, build, contract,
        ``validate``, placement, probe, ambiguity, lock, boundary, migration) happens before
        the slot write; the slot write is the single mutation; ``detach`` and ``attach`` after
        it cannot fail the swap (a raising ``detach`` is logged).
        """
        chosen = self._mode(mode)
        path = DEFAULT_PATH if glob is None else OVERRIDE_PREFIX + glob
        with self._lock:
            adding = path not in self._slots.paths()
            if glob is not None and (not glob or glob != glob.strip()):
                raise ImproperlyConfigured(f"E070: invalid override glob {glob!r}")
            if boundary.rank > Boundary.EPISODE.rank:
                raise ComponentLocked(KIND, path, "peer_strategy may only change at episode boundaries")
            if self._slots._slot_locked(path):
                raise ComponentLocked(KIND, path, "the slot is locked by COMPONENTS.LOCKED")
            adopted = not isinstance(raw, (str, Mapping, ComponentSpec))
            spec = None if adopted else ComponentSpec.parse(KIND, _lowered(raw))
            candidate = raw if adopted else self._resolver.resolve(spec, trusted=self._trusted).instance
            if not adopted and not coordinates_via_context(candidate) and glob is not None:
                raise ImproperlyConfigured(f"E072: {path}: {type(candidate).__name__} does not define coordinate(ctx)")
            self._check_legacy_placement(candidate, path)
            self._check_instance(path, candidate, view)
            self.probe(path, view=view, instance=candidate, spec=spec)
            old = None if adding else self._slots.get(path).instance
            carried = self._collect_state(old, candidate, chosen, path)
            if carried is not None:
                # Prove the state loads before anything is written: a candidate that cannot take it
                # fails here, and the swap below loads it into the live instance as its ``migrate``
                # step, which ``SlotTable.swap`` runs before it rewrites the slot.
                self._load_carried(candidate, carried, path)
            # -- the single mutation -----------------------------------------
            if adding:
                if adopted:
                    self._slots.adopt(path, KIND, raw, name=type(raw).__name__)
                else:
                    self._slots.install(path, KIND, spec)
                record = None
            elif adopted:
                self._slots.adopt(path, KIND, raw, name=type(raw).__name__)
                record = None
            else:
                migrate = None
                if carried is not None:
                    def migrate(_old: Any, new: Any, _state: Any = carried) -> None:
                        new.load_state_dict(copy.deepcopy(_state))
                record = self._slots.swap(path, spec, boundary=boundary, migrate=migrate)
            if adopted:
                self._adopted[path] = raw
            else:
                self._adopted.pop(path, None)
            for key in [k for k in self._group_cache if k[0] == path]:
                del self._group_cache[key]
            live = self._slots.get(path).instance
        self._after_change(old, live, host)
        return record

    @staticmethod
    def _load_carried(candidate: Any, state: Mapping[str, Any], path: str) -> None:
        try:
            candidate.load_state_dict(copy.deepcopy(state))
        except Exception as error:  # noqa: BLE001 - reported before anything is written
            raise ImproperlyConfigured(
                f"E074: {path}: {type(candidate).__name__} cannot take the carried state "
                f"({type(error).__name__}: {error}); nothing was changed"
            ) from error

    def _collect_state(self, old: Any, new: Any, mode: str, path: str) -> dict[str, Any] | None:
        read = getattr(old, "state_dict", None) if old is not None else None
        state = dict(read()) if callable(read) else {}
        meaningful = {k: v for k, v in state.items() if k not in ("class", "name")}
        if not meaningful:
            return None
        if mode == "fail":
            raise ImproperlyConfigured(
                f"E074: replacing {type(old).__name__} at {path} would discard its state "
                f"({sorted(meaningful)}); TRAINER.PEER_STRATEGY_OPTIONS.MIGRATE is 'fail'"
            )
        if mode == "reset":
            return None
        if type(old) is not type(new) or not callable(getattr(new, "load_state_dict", None)):
            log.warning("W031: carry from %s to %s is impossible (different classes); state dropped",
                        type(old).__name__, type(new).__name__)
            return None
        return state

    def _after_change(self, old: Any, new: Any, host: Any) -> None:
        if host is None:
            return
        if old is not None and old is not new:
            try:
                old.detach(host)
            except Exception as error:  # noqa: BLE001 - the swap is done; report, do not undo
                log.error("detach of %s raised: %s", type(old).__name__, error)
        try:
            new.attach(host)
        except Exception as error:  # noqa: BLE001 - the swap is done; report, do not undo
            log.error("attach of %s raised: %s", type(new).__name__, error)
            self.errors.append(f"attach of {type(new).__name__} raised {type(error).__name__}: {error}")

    def drain_errors(self) -> list[str]:
        with self._lock:
            out, self.errors = self.errors, []
        return out

    def remove(
        self, glob: str, *, view: TreeView, host: Any = None, boundary: Boundary = Boundary.EPISODE
    ) -> SwapRecord:
        """Delete one override; its groups fall back to the next most specific, or the default."""
        path = OVERRIDE_PREFIX + glob
        with self._lock:
            if path not in self._slots.paths():
                raise ComponentMissing("slot", path, self._slots.paths())
            if self._slots._slot_locked(path):
                raise ComponentLocked(KIND, path, "the slot is locked by COMPONENTS.LOCKED")
            if boundary.rank > Boundary.EPISODE.rank:
                raise ComponentLocked(KIND, path, "peer_strategy may only change at episode boundaries")
            before = self.override_globs()
            after = [g for g in before if g != glob]
            for group in view.groups:                               # E071 if a tie appears
                pick_glob(group.label, after, what="peer_strategy overrides", code="E071")
            old = self._slots.get(path).instance
            record = self._slots.remove(path, boundary=boundary)
            self._adopted.pop(path, None)
            for key in [k for k in self._group_cache if k[0] == path]:
                del self._group_cache[key]
        self._after_change(old, self.default_instance(), host)
        return record

    def request(self, glob: str | None, raw: Any, *, mode: str | None = None) -> None:
        """Queue a replacement for the next episode boundary. Validated eagerly."""
        self._mode(mode)
        path = DEFAULT_PATH if glob is None else OVERRIDE_PREFIX + glob
        if isinstance(raw, (str, Mapping, ComponentSpec)):
            self._resolver.validate(ComponentSpec.parse(KIND, _lowered(raw)), trusted=self._trusted)
        with self._lock:
            self._pending[path] = (raw, mode)

    def request_remove(self, glob: str) -> None:
        path = OVERRIDE_PREFIX + glob
        if path not in self._slots.paths():
            raise ComponentMissing("slot", path, self._slots.paths())
        with self._lock:
            self._pending[path] = (_REMOVE, None)

    def pending(self) -> dict[str, str]:
        with self._lock:
            return {path: ("remove" if value is _REMOVE else "set")
                    for path, (value, _mode) in self._pending.items()}

    def take_pending(self) -> list[tuple[str | None, Any, str | None]]:
        with self._lock:
            items = sorted(self._pending.items(), key=lambda kv: (kv[0] != DEFAULT_PATH, kv[0]))
            self._pending.clear()
        out: list[tuple[str | None, Any, str | None]] = []
        for path, (value, mode) in items:
            glob = None if path == DEFAULT_PATH else path[len(OVERRIDE_PREFIX):]
            out.append((glob, value, mode))
        return out

    @staticmethod
    def is_removal(value: Any) -> bool:
        return value is _REMOVE

    # -- persistence -----------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        slots: dict[str, Any] = {}
        state: dict[str, Any] = {}
        for path in [DEFAULT_PATH] + [OVERRIDE_PREFIX + g for g in self.override_globs()]:
            resolved = self._slots.get(path).resolved
            if path in self._adopted:
                comp = resolved.component
                slots[path] = {"adopted": f"{comp.module}.{comp.qualname}"}
            else:
                slots[path] = {"spec": resolved.spec.canonical(), "hash": resolved.spec_hash}
            instance = self._slots.get(path).instance
            read = getattr(instance, "state_dict", None)
            body = dict(read()) if callable(read) else {}
            if {k for k in body} - {"class", "name"}:
                state[f"{path}|*"] = {"class": type(instance).__name__, "state": body}
        for (path, label), (_resolved, instance) in sorted(self._group_cache.items()):
            read = getattr(instance, "state_dict", None)
            body = dict(read()) if callable(read) else {}
            if {k for k in body} - {"class", "name"}:
                state[f"{path}|{label}"] = {"class": type(instance).__name__, "state": body}
        return {
            "version": 1,
            "slots": slots,
            "state": state,
            "history": [r.as_dict() for r in self.history()][-64:],
        }

    def load_state_dict(self, payload: Mapping[str, Any], *, strict: bool = False) -> list[str]:
        """Reconcile the live plan with a saved one. The saved plan wins (``W031``); under
        ``strict`` (``COMPONENTS.STRICT_RESTORE``) any difference is ``E076``."""
        notes: list[str] = []
        saved = dict(payload.get("slots") or {})
        with self._lock:
            for path, entry in sorted(saved.items()):
                if "spec" not in entry:
                    notes.append(
                        f"W031: {path} held an in-code instance ({entry.get('adopted')}) when saved; "
                        "the configured strategy is kept"
                    )
                    continue
                live = self._slots.get(path) if path in self._slots.paths() else None
                if live is not None and path not in self._adopted and live.resolved.spec_hash == entry.get("hash"):
                    continue
                message = f"{path}: the saved run used {entry['spec']}, this run is configured differently"
                if strict:
                    raise ImproperlyConfigured(f"E076: {message}; COMPONENTS.STRICT_RESTORE is on")
                self._slots.install(path, KIND, entry["spec"])
                self._adopted.pop(path, None)
                notes.append(f"W031: {message}; the saved plan wins")
            for path in [p for p in self._slots.paths()
                         if p.startswith(OVERRIDE_PREFIX) and p not in saved]:
                message = f"{path} exists now but not in the saved run (it had been removed)"
                if strict:
                    raise ImproperlyConfigured(f"E076: {message}; COMPONENTS.STRICT_RESTORE is on")
                self._slots.remove(path, boundary=Boundary.RUN_START)
                self._adopted.pop(path, None)
                notes.append(f"W031: {message}; the saved plan wins")
            self._group_cache.clear()
            self._restore = {}
            for key, entry in dict(payload.get("state") or {}).items():
                path, _, label = key.partition("|")
                if label == "*":
                    instance = self._slots.get(path).instance if path in self._slots.paths() else None
                    load = getattr(instance, "load_state_dict", None)
                    if instance is not None and callable(load) and entry.get("class") == type(instance).__name__:
                        load(copy.deepcopy(entry["state"]))
                    else:
                        notes.append(f"W031: state saved for {path} does not fit the live strategy; dropped")
                else:
                    self._restore[key] = dict(entry)
        for note in notes:
            log.warning(note)
        return notes

    # -- reporting -------------------------------------------------------
    def describe(self, labels: Sequence[str] = ()) -> dict[str, Any]:
        slots = {
            path: self._slots.get(path).resolved.fingerprint()
            for path in self._slots.paths()
            if path == DEFAULT_PATH or path.startswith(OVERRIDE_PREFIX)
        }
        return {
            "slots": slots,
            "serving": {label: self.select(label) for label in labels},
            "guard": self.guard.describe(),
            "pending": self.pending(),
        }
