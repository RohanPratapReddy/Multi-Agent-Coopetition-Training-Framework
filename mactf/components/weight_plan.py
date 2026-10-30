"""WeightPlan: the ``weight_update`` slots of one run (document 17 section 10.5)."""

from __future__ import annotations

import copy
import fnmatch
import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..coopetition.aggregation import Aggregator
from ..coopetition.weight_guard import WeightGuard
from ..coopetition.weight_probe import verify_weight_update
from ..coopetition.weights import GroupView, KappaSoftmaxAverage, _split_spec, traits_of
from ..exceptions import (
    ComponentLocked,
    ComponentMissing,
    ImproperlyConfigured,
)
from .function_book import specificity
from .kinds import Boundary
from .resolver import ComponentResolver, ComponentSpec, Resolved
from .slots import Slot, SlotTable, SwapRecord

log = logging.getLogger("mactf.weights")

KIND = "weight_update"
DEFAULT_PATH = "TRAINER.WEIGHT_UPDATE"
OVERRIDE_PREFIX = "TRAINER.WEIGHT_UPDATE_OVERRIDES."
ROOT_LABEL = "root"
MIGRATE_MODES = ("reset", "carry", "fail")
_REMOVE = object()


def _prepared(raw: Any) -> Any:
    """``raw`` with the strategy name lower-cased and a legacy ``aggregator`` key dropped.

    ``build_weight_update`` always did this; the plan injects the group's aggregator itself.
    """
    if isinstance(raw, (str, Mapping)):
        return _split_spec(raw, None)[0]
    return raw


def _plain_kappa(spec: ComponentSpec) -> bool:
    """The framework default, unparameterised: built directly, not through the registry."""
    return spec.type == "kappa_softmax" and not spec.params and spec.import_path is None


def group_label(parent: str | None) -> str:
    return ROOT_LABEL if parent is None else parent


def pick_glob(label: str, globs: Sequence[str], *, what: str, code: str) -> str | None:
    """Exact label, else the strictly most specific matching glob, else ``None``.

    Shared by ``WeightPlan`` (``E061``) and ``KappaPlan`` (``E062``); it is the rule of
    8.4 and 9.5 with the error code as a parameter.
    """
    if label in globs:
        return label
    best_score, best, tied = -1, None, []
    for glob in sorted(globs):
        if not fnmatch.fnmatchcase(label, glob):
            continue
        score = specificity(glob)
        if score > best_score:
            best_score, best, tied = score, glob, []
        elif score == best_score:
            tied.append(glob)
    if tied:
        raise ImproperlyConfigured(
            f"{code}: group {label!r} is matched by equally specific {what} "
            f"{sorted([best, *tied])}"
        )
    return best


@dataclass
class _Built:
    slot_path: str
    resolved: Resolved
    aggregator: Aggregator
    instance: Any


class WeightPlan:
    """The ``weight_update`` slots of one run and the rules that choose and change them."""

    def __init__(
        self,
        slots: SlotTable,
        *,
        resolver: ComponentResolver | None = None,
        trusted: bool = True,
        guard: WeightGuard | None = None,
        strict_globs: bool = True,
        migrate: str = "reset",
        strict_restore: bool = False,
    ) -> None:
        if migrate not in MIGRATE_MODES:
            raise ImproperlyConfigured(
                f"E060: TRAINER.WEIGHT_OPTIONS.MIGRATE must be one of {list(MIGRATE_MODES)}, got {migrate!r}"
            )
        self._slots = slots
        self._resolver = resolver or ComponentResolver()
        self._trusted = trusted
        self.guard = guard or WeightGuard()
        self.strict_globs = bool(strict_globs)
        self.migrate = migrate
        self.strict_restore = bool(strict_restore)
        self._aggregator: Aggregator | None = None
        self._adopted: dict[str, Any] = {}
        self._cache: dict[str, _Built] = {}
        self._pending: dict[str, tuple[Any, str | None]] = {}
        self._carry: dict[str, dict[str, Any]] = {}
        self._restore: dict[str, Mapping[str, Any]] = {}
        self._lock = threading.RLock()

    def forget_groups(self, labels: Iterable[str]) -> Callable[[], None]:
        """Drop the per-group strategy instances (and carried state) of groups that changed.

        They are rebuilt on the next ``strategy_for``, seeded from ``_restore`` when a
        checkpoint supplied state, otherwise fresh. Returns the inverse (document 17 section 12.6.2).
        """
        wanted = set(labels)
        with self._lock:
            dropped_cache = {label: self._cache.pop(label) for label in list(self._cache) if label in wanted}
            dropped_carry = {label: self._carry.pop(label) for label in list(self._carry) if label in wanted}

        def undo() -> None:
            with self._lock:
                self._cache.update(dropped_cache)
                self._carry.update(dropped_carry)

        return undo

    # -- construction ----------------------------------------------------
    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any],
        slots: SlotTable,
        *,
        aggregator: Aggregator,
        code_spec: Any = None,
        resolver: ComponentResolver | None = None,
        trusted: bool = True,
    ) -> "WeightPlan":
        """Install the default and every override.

        Precedence for the default, highest first: the ``weight_update=`` constructor
        argument (an instance is *adopted*, a name or mapping is a spec), then
        ``TRAINER.WEIGHT_UPDATE``, then ``kappa_softmax``.
        """
        trainer = dict(settings.get("TRAINER") or {})
        options = {str(k).upper(): v for k, v in dict(trainer.get("WEIGHT_OPTIONS") or {}).items()}
        unknown = sorted(set(options) - {"STRICT_GLOBS", "MIGRATE"})
        if unknown:
            raise ImproperlyConfigured(
                f"E060: TRAINER.WEIGHT_OPTIONS has unknown keys {unknown}; accepted: MIGRATE, STRICT_GLOBS"
            )
        components = dict(settings.get("COMPONENTS") or {})
        plan = cls(
            slots,
            resolver=resolver,
            trusted=trusted,
            guard=WeightGuard.from_settings(trainer.get("WEIGHT_GUARD")),
            strict_globs=bool(options.get("STRICT_GLOBS", True)),
            migrate=str(options.get("MIGRATE", "reset")).lower(),
            strict_restore=bool(components.get("STRICT_RESTORE", False)),
        )
        plan.bind(aggregator)
        if code_spec is not None and not isinstance(code_spec, (str, Mapping, ComponentSpec)):
            plan.adopt_default(code_spec)
        else:
            plan.install_default(code_spec if code_spec is not None
                                 else trainer.get("WEIGHT_UPDATE", "kappa_softmax"))
        overrides = trainer.get("WEIGHT_UPDATE_OVERRIDES") or {}
        if not isinstance(overrides, Mapping):
            raise ImproperlyConfigured("E060: TRAINER.WEIGHT_UPDATE_OVERRIDES must be a mapping of glob to spec")
        for glob, raw in overrides.items():
            plan.install_override(str(glob), raw)
        return plan

    def bind(self, aggregator: Aggregator) -> None:
        """The aggregator prototypes are built with (the root's)."""
        self._aggregator = aggregator

    def _inject(self) -> dict[str, Any]:
        return {"aggregator": self._aggregator} if self._aggregator is not None else {}

    def install_default(self, raw: Any) -> Slot:
        spec = ComponentSpec.parse(KIND, _prepared(raw))
        if _plain_kappa(spec) and self._aggregator is not None:
            # The default run: a plain call, so no component-registry call is made (19.4,
            # condition 4). The slot still records the registered built-in.
            return self._slots.adopt_registered(
                DEFAULT_PATH, KIND, KappaSoftmaxAverage(self._aggregator), name="kappa_softmax"
            )
        return self._slots.install(DEFAULT_PATH, KIND, spec, inject=self._inject())

    def adopt_default(self, instance: Any) -> Slot:
        slot = self._slots.adopt(DEFAULT_PATH, KIND, instance, name=type(instance).__name__)
        self._adopted[DEFAULT_PATH] = instance
        return slot

    def install_override(self, glob: str, raw: Any) -> Slot:
        if not glob or glob != glob.strip():
            raise ImproperlyConfigured(f"E060: invalid override glob {glob!r}")
        path = OVERRIDE_PREFIX + glob
        if isinstance(raw, (str, Mapping, ComponentSpec)):
            return self._slots.install(path, KIND, _prepared(raw), inject=self._inject())
        slot = self._slots.adopt(path, KIND, raw, name=type(raw).__name__)
        self._adopted[path] = raw
        return slot

    # -- reading ---------------------------------------------------------
    def override_globs(self) -> list[str]:
        return sorted(p[len(OVERRIDE_PREFIX):] for p in self._slots.paths()
                      if p.startswith(OVERRIDE_PREFIX))

    def select(self, label: str) -> str:
        """The slot path serving ``label``. ``E061`` on an ambiguous tie."""
        glob = pick_glob(label, self.override_globs(), what="weight_update overrides", code="E061")
        return DEFAULT_PATH if glob is None else OVERRIDE_PREFIX + glob

    def customized(self) -> bool:
        """Whether anything here differs from the built-in default (decides what is persisted)."""
        if any(p.startswith(OVERRIDE_PREFIX) for p in self._slots.paths()):
            return True
        if DEFAULT_PATH in self._adopted:
            return type(self._adopted[DEFAULT_PATH]) is not KappaSoftmaxAverage
        slot = self._slots.get(DEFAULT_PATH)
        return not (slot.resolved.spec.type == "kappa_softmax" and not slot.resolved.spec.params
                    and slot.resolved.spec.import_path is None)

    def history(self) -> list[SwapRecord]:
        return [r for r in self._slots.history if r.slot.startswith(DEFAULT_PATH)]

    def strategy_for(self, label: str, aggregator: Aggregator) -> Any:
        """The instance serving group ``label``, built on first use. Never the prototype."""
        with self._lock:
            path = self.select(label)
            slot = self._slots.get(path)
            built = self._cache.get(label)
            if (built is not None and built.resolved is slot.resolved
                    and built.aggregator is aggregator):
                return built.instance
            instance = self._build_group(label, path, slot, aggregator)
            self._cache[label] = _Built(path, slot.resolved, aggregator, instance)
            return instance

    def _build_group(self, label: str, path: str, slot: Slot, aggregator: Aggregator) -> Any:
        if path in self._adopted:
            proto = self._adopted[path]
            if isinstance(proto, KappaSoftmaxAverage):
                instance = KappaSoftmaxAverage(aggregator)
            else:
                clone = getattr(proto, "clone", None)
                instance = clone() if callable(clone) else proto
        elif _plain_kappa(slot.resolved.spec):
            instance = KappaSoftmaxAverage(aggregator)
        else:
            instance = self._resolver.resolve(
                slot.resolved.spec, trusted=self._trusted, inject={"aggregator": aggregator}
            ).instance
        self._seed_state(label, path, instance)
        return instance

    def _seed_state(self, label: str, path: str, instance: Any) -> None:
        entry = self._restore.pop(label, None) or self._carry.pop(label, None)
        if entry is None:
            return
        load = getattr(instance, "load_state_dict", None)
        if (not callable(load) or entry.get("class") != type(instance).__name__
                or entry.get("slot") != path):
            if entry.get("state"):
                log.warning(
                    "W021: state saved for group %r (%s) does not fit %s at %s; dropped",
                    label, entry.get("class"), type(instance).__name__, path,
                )
            return
        try:
            load(copy.deepcopy(entry["state"]))
        except Exception as error:  # noqa: BLE001 - a resume must not end on one group's stale state
            log.warning(
                "W021: state saved for group %r at %s could not be loaded (%s: %s); dropped",
                label, path, type(error).__name__, error,
            )

    def _scratch(self, path: str) -> Any:
        """A throwaway instance of slot ``path`` for probing and validation."""
        if path in self._adopted:
            proto = self._adopted[path]
            clone = getattr(proto, "clone", None)
            if callable(clone):
                return clone()
            try:
                return copy.deepcopy(proto)
            except Exception:  # noqa: BLE001 - a user object that cannot be copied is probed as is
                return proto
        slot = self._slots.get(path)
        return self._resolver.resolve(
            slot.resolved.spec, trusted=self._trusted, inject=self._inject()
        ).instance

    # -- validation ------------------------------------------------------
    def validate(self, views: Sequence[GroupView], *, initial: bool) -> list[str]:
        """Check the plan against the groups that exist. Returns warnings; raises on errors.

        ``E060`` an override matches no group, or two groups share a label;
        ``E061`` ambiguous overrides; ``E064`` a strategy's ``validate`` or probe failed.
        """
        labels = [view.label for view in views]
        if len(set(labels)) != len(labels):
            duplicates = sorted({label for label in labels if labels.count(label) > 1})
            raise ImproperlyConfigured(
                f"E060: aggregation group labels collide: {duplicates}; the label {ROOT_LABEL!r} is "
                "reserved for the top group and cannot also be a node path with children"
            )
        notes: list[str] = []
        for glob in self.override_globs():
            if any(label == glob or fnmatch.fnmatchcase(label, glob) for label in labels):
                continue
            message = f"override {glob!r} matches no aggregation group; groups are {labels}"
            if initial and self.strict_globs:
                raise ImproperlyConfigured(f"E060: TRAINER.WEIGHT_UPDATE_OVERRIDES: {message}")
            notes.append(f"W022: {message}")
        probed: set[str] = set()
        for view in views:
            path = self.select(view.label)                     # E061
            self._check_group(path, view)
            if initial and path not in probed:
                probed.add(path)
                self._probe(path)
        for note in notes:
            log.warning(note)
        return notes

    def _check_group(self, path: str, view: GroupView, instance: Any | None = None) -> None:
        if instance is None:
            # Most strategies define no ``validate``; do not build one just to find out.
            declared = getattr(self._slots.get(path).component.factory, "validate", None)
            if not callable(declared) and path not in self._adopted:
                return
        strategy = instance if instance is not None else self._scratch(path)
        check = getattr(strategy, "validate", None)
        if not callable(check):
            return
        problems = [str(item) for item in (check(view) or ())]
        if problems:
            raise ImproperlyConfigured(
                f"E064: {path} cannot serve group {view.label!r}: " + "; ".join(problems)
            )

    def _probe(self, path: str, instance_factory: Any | None = None) -> None:
        if not self.guard.probe or self._aggregator is None:
            return
        slot = self._slots.get(path)
        if not self.guard.covers(slot.instance, self._aggregator):
            return
        verify_weight_update(
            instance_factory or (lambda: self._scratch(path)),
            floor=self._aggregator.weight_floor,
            cap=self._aggregator.weight_cap,
            tolerance=self.guard.tolerance,
            slot=path,
            mode=self.guard.mode,
        )

    # -- live changes ----------------------------------------------------
    def _selected_after(self, label: str, globs: Sequence[str]) -> str:
        glob = pick_glob(label, globs, what="weight_update overrides", code="E061")
        return DEFAULT_PATH if glob is None else OVERRIDE_PREFIX + glob

    def _mode(self, mode: str | None) -> str:
        chosen = (mode or self.migrate).lower()
        if chosen not in MIGRATE_MODES:
            raise ImproperlyConfigured(f"E064: migrate must be one of {list(MIGRATE_MODES)}, got {mode!r}")
        return chosen

    def set(
        self,
        glob: str | None,
        raw: Any,
        *,
        views: Sequence[GroupView],
        boundary: Boundary = Boundary.EPISODE,
        mode: str | None = None,
    ) -> SwapRecord | None:
        """Replace the default (``glob=None``), replace one override, or add an override. Now.

        Order is the atomicity argument: everything that can fail (parse, build, probe,
        per-group ``validate``, ambiguity, state migration, lock, boundary) happens first;
        the slot write is the single mutation; cache invalidation after it cannot fail.
        """
        chosen_mode = self._mode(mode)
        path = DEFAULT_PATH if glob is None else OVERRIDE_PREFIX + glob
        with self._lock:
            adding = path not in self._slots.paths()
            if glob is not None and (not glob or glob != glob.strip()):
                raise ImproperlyConfigured(f"E060: invalid override glob {glob!r}")
            if boundary.rank > Boundary.EPISODE.rank:
                raise ComponentLocked(KIND, path, "weight_update may only change at episode boundaries")
            if self._slots._slot_locked(path):
                raise ComponentLocked(KIND, path, "the slot is locked by COMPONENTS.LOCKED")
            adopted = not isinstance(raw, (str, Mapping, ComponentSpec))
            if adopted:
                candidate_instance, spec = raw, None
            else:
                spec = ComponentSpec.parse(KIND, _prepared(raw))
                candidate_instance = self._resolver.resolve(
                    spec, trusted=self._trusted, inject=self._inject()
                ).instance
            globs = sorted(set(self.override_globs()) | ({glob} if glob is not None else set()))
            affected = [view for view in views if self._selected_after(view.label, globs) == path]
            for view in affected:
                self._check_group(path, view, candidate_instance)
            if self.guard.probe and self._aggregator is not None and self.guard.covers(
                candidate_instance, self._aggregator
            ):
                verify_weight_update(
                    (lambda: self._clone_for_probe(candidate_instance, spec)),
                    floor=self._aggregator.weight_floor, cap=self._aggregator.weight_cap,
                    tolerance=self.guard.tolerance, slot=path, mode=self.guard.mode,
                )
            carried = self._collect_state(path, [v.label for v in affected], chosen_mode,
                                          candidate_instance)
            for label, entry in carried.items():
                # Prove each carried state loads before anything is written (it is applied lazily).
                trial = self._clone_for_probe(candidate_instance, spec)
                try:
                    trial.load_state_dict(copy.deepcopy(entry["state"]))
                except Exception as error:  # noqa: BLE001 - reported before anything is written
                    raise ImproperlyConfigured(
                        f"E064: {path}: {type(trial).__name__} cannot take the carried state of "
                        f"group {label!r} ({type(error).__name__}: {error}); nothing was changed"
                    ) from error
            # -- the single mutation ---------------------------------------
            if adding:
                if adopted:
                    self._slots.adopt(path, KIND, raw, name=type(raw).__name__)
                else:
                    self._slots.install(path, KIND, spec, inject=self._inject())
                record = None
            elif adopted:
                self._slots.adopt(path, KIND, raw, name=type(raw).__name__)
                record = None
            else:
                record = self._slots.swap(path, spec, boundary=boundary, inject=self._inject())
            # -- cannot fail beyond this line ------------------------------
            if adopted:
                self._adopted[path] = raw
            else:
                self._adopted.pop(path, None)
            for view in views:
                if self._selected_after(view.label, globs) == path:
                    self._cache.pop(view.label, None)
            self._carry.update(carried)
            return record

    def _clone_for_probe(self, instance: Any, spec: ComponentSpec | None) -> Any:
        if spec is not None:
            return self._resolver.resolve(spec, trusted=self._trusted, inject=self._inject()).instance
        clone = getattr(instance, "clone", None)
        if callable(clone):
            return clone()
        try:
            return copy.deepcopy(instance)
        except Exception:  # noqa: BLE001
            return instance

    def _collect_state(
        self, path: str, labels: Sequence[str], mode: str, candidate: Any
    ) -> dict[str, dict[str, Any]]:
        carried: dict[str, dict[str, Any]] = {}
        for label in sorted(labels):
            built = self._cache.get(label)
            if built is None:
                continue
            read = getattr(built.instance, "state_dict", None)
            state = dict(read()) if callable(read) else {}
            if not state:
                continue
            if mode == "fail":
                raise ImproperlyConfigured(
                    f"E064: group {label!r} holds {type(built.instance).__name__} state and "
                    "migrate is 'fail'"
                )
            if mode == "carry" and type(built.instance) is type(candidate):
                carried[label] = {
                    "slot": path, "class": type(candidate).__name__,
                    "state": copy.deepcopy(state),
                }
            else:
                log.info("weight_update state of group %r is discarded by the swap (mode=%s)", label, mode)
        return carried

    def remove(
        self, glob: str, *, views: Sequence[GroupView], boundary: Boundary = Boundary.EPISODE
    ) -> SwapRecord:
        """Delete one override; its groups fall back to the next most specific, or the default."""
        path = OVERRIDE_PREFIX + glob
        with self._lock:
            if path not in self._slots.paths():
                raise ComponentMissing("slot", path, self._slots.paths())
            if self._slots._slot_locked(path):
                raise ComponentLocked(KIND, path, "the slot is locked by COMPONENTS.LOCKED")
            if boundary.rank > Boundary.EPISODE.rank:
                raise ComponentLocked(KIND, path, "weight_update may only change at episode boundaries")
            before = self.override_globs()
            after = [g for g in before if g != glob]
            moved = [v for v in views if self._selected_after(v.label, before) == path]
            for view in moved:
                target = self._selected_after(view.label, after)          # E061 if it now ties
                self._check_group(target, view)
            record = self._slots.remove(path, boundary=boundary)
            self._adopted.pop(path, None)
            for view in moved:
                self._cache.pop(view.label, None)
            return record

    def request(self, glob: str | None, raw: Any, *, mode: str | None = None) -> None:
        """Queue a replacement for the next episode boundary. Validated eagerly."""
        self._mode(mode)
        path = DEFAULT_PATH if glob is None else OVERRIDE_PREFIX + glob
        if isinstance(raw, (str, Mapping, ComponentSpec)):
            self._resolver.validate(
                ComponentSpec.parse(KIND, _prepared(raw)), trusted=self._trusted,
                inject=self._inject(),
            )
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
        """Drain the queue as ``(glob, raw_or__REMOVE, mode)``, defaults first."""
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
        for path in self._slots.paths():
            if not (path == DEFAULT_PATH or path.startswith(OVERRIDE_PREFIX)):
                continue
            resolved = self._slots.get(path).resolved
            if path in self._adopted:
                comp = resolved.component
                slots[path] = {"adopted": f"{comp.module}.{comp.qualname}"}
            else:
                slots[path] = {"spec": resolved.spec.canonical(), "hash": resolved.spec_hash}
        groups: dict[str, Any] = {}
        for label, built in sorted(self._cache.items()):
            read = getattr(built.instance, "state_dict", None)
            state = dict(read()) if callable(read) else {}
            if state:
                groups[label] = {
                    "slot": built.slot_path,
                    "class": type(built.instance).__name__,
                    "state": state,
                }
        return {
            "version": 1,
            "slots": slots,
            "groups": groups,
            "history": [r.as_dict() for r in self.history()][-64:],
        }

    def load_state_dict(self, payload: Mapping[str, Any], *, strict: bool = False) -> list[str]:
        """Reconcile the live plan with a saved one. The saved plan wins (``W021``); under
        ``strict`` (``COMPONENTS.STRICT_RESTORE``) any difference is ``E066``."""
        notes: list[str] = []
        saved = dict(payload.get("slots") or {})
        with self._lock:
            for path, entry in sorted(saved.items()):
                if "spec" not in entry:
                    notes.append(
                        f"W021: {path} held an in-code instance ({entry.get('adopted')}) when saved; "
                        "the configured strategy is kept"
                    )
                    continue
                live = self._slots.get(path) if path in self._slots.paths() else None
                if live is not None and path not in self._adopted and live.resolved.spec_hash == entry.get("hash"):
                    continue
                message = f"{path}: the saved run used {entry['spec']}, this run is configured differently"
                if strict:
                    raise ImproperlyConfigured(f"E066: {message}; COMPONENTS.STRICT_RESTORE is on")
                self._slots.install(path, KIND, entry["spec"], inject=self._inject())
                self._adopted.pop(path, None)
                notes.append(f"W021: {message}; the saved plan wins")
            for path in [p for p in self._slots.paths()
                         if p.startswith(OVERRIDE_PREFIX) and p not in saved]:
                message = f"{path} exists now but not in the saved run (it had been removed)"
                if strict:
                    raise ImproperlyConfigured(f"E066: {message}; COMPONENTS.STRICT_RESTORE is on")
                self._slots.remove(path, boundary=Boundary.RUN_START)
                self._adopted.pop(path, None)
                notes.append(f"W021: {message}; the saved plan wins")
            self._cache.clear()
            self._restore = {label: dict(entry) for label, entry in dict(payload.get("groups") or {}).items()}
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
