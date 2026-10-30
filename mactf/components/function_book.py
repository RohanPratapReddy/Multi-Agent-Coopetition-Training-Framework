"""FunctionBook: per-node RL function resolution (document 17 section 8.4)."""

from __future__ import annotations

import fnmatch
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from ..exceptions import (
    ComponentLocked,
    ComponentMigrationError,
    ComponentSpecError,
    ImproperlyConfigured,
)
from .function_probes import ProbeContext, check_function
from .kinds import RL_FAMILIES, Boundary, Origin
from .registry import HUB
from .resolver import ComponentResolver, ComponentSpec, Resolved
from .migrations import is_stateful, migration_for
from .slots import Slot, SlotTable

log = logging.getLogger("mactf.components")

#: Families a single global object serves; a per-node override is meaningless for them.
GLOBAL_ONLY: frozenset[str] = frozenset({"rollup"})

VERIFY_MODES: tuple[str, ...] = ("custom", "all", "off")
UNCONSUMED_MODES: tuple[str, ...] = ("error", "warn")

_WILDCARDS = frozenset("*?[]!")
_RESERVED_KEYS = frozenset({"OVERRIDES", "UNCONSUMED", "STRICT_GLOBS", "VERIFY", "MIGRATE"})


def specificity(pattern: str) -> int:
    """How specific a glob is: its count of non-wildcard characters."""
    return sum(1 for char in pattern if char not in _WILDCARDS)


def _is_glob(pattern: str) -> bool:
    return any(char in _WILDCARDS for char in pattern)


def _default_path(family: str) -> str:
    return f"RL.{family}"


def _override_path(family: str, glob: str) -> str:
    return f"RL.{family}.{glob}"


@dataclass(frozen=True)
class BookSettings:
    """The non-spec keys of the ``RL`` block, plus the loss-compliance policy."""

    unconsumed: str = "error"
    strict_globs: bool = True
    verify: str = "custom"
    loss_compliance: str = "require"
    compliance_coefficient: float = 1.0
    guard_mode: str = "enforce"
    migrate: Mapping[str, str] = field(default_factory=dict)   # RL.MIGRATE, family -> mode (8.8.1)

    def __post_init__(self) -> None:
        if self.unconsumed not in UNCONSUMED_MODES:
            raise ImproperlyConfigured(f"RL.UNCONSUMED must be one of {list(UNCONSUMED_MODES)}, got {self.unconsumed!r}")
        if self.verify not in VERIFY_MODES:
            raise ImproperlyConfigured(f"RL.VERIFY must be one of {list(VERIFY_MODES)}, got {self.verify!r}")


@dataclass
class _Cached:
    resolved: Resolved
    raw: Any        # what the component built; migrations operate on it (8.8.2)
    instance: Any   # raw wrapped by the compliance or rollup guard, when the family has one


class FunctionBook:
    """Resolves ``(family, node path)`` to an instance, and owns the RL function slots."""

    def __init__(
        self,
        slots: SlotTable,
        resolver: ComponentResolver | None = None,
        *,
        settings: BookSettings | None = None,
        trusted: bool = True,
        probe_context: ProbeContext | None = None,
    ) -> None:
        self._slots = slots
        self._resolver = resolver or ComponentResolver()
        self._trusted = trusted
        self.settings = settings or BookSettings()
        self._probe_context = probe_context or ProbeContext()
        self._cache: dict[tuple[str, str, str], _Cached] = {}
        # queued swaps: slot path -> (family, parsed spec, migration mode); see 8.8.3
        self._pending: dict[str, tuple[str, ComponentSpec, str | None]] = {}
        #: queued swaps that failed when applied, for the episode metadata
        self.failed_swaps: list[dict[str, str]] = []
        self._lock = threading.RLock()

    # -- construction from the ``RL`` block ------------------------------------------
    @classmethod
    def from_settings(
        cls, slots: SlotTable, rl_block: Mapping[str, Any] | None, *,
        loss_compliance: str = "require", compliance_coefficient: float = 1.0,
        guard_mode: str = "enforce",
        resolver: ComponentResolver | None = None, trusted: bool = True,
    ) -> "FunctionBook":
        block = dict(rl_block or {})
        settings = BookSettings(
            unconsumed=str(block.get("UNCONSUMED", "error")).lower(),
            strict_globs=bool(block.get("STRICT_GLOBS", True)),
            verify=str(block.get("VERIFY", "custom")).lower(),
            loss_compliance=loss_compliance,
            compliance_coefficient=float(compliance_coefficient),
            guard_mode=str(guard_mode).lower(),
            migrate={str(k).lower(): str(v).lower() for k, v in dict(block.get("MIGRATE") or {}).items()},
        )
        return cls(slots, resolver, settings=settings, trusted=trusted)

    def configure(self, rl_block: Mapping[str, Any] | None) -> None:
        """Install every default and override named in the ``RL`` block. Run-start only."""
        block = {str(key).upper(): value for key, value in dict(rl_block or {}).items()}
        families = {name.lower(): name for name in RL_FAMILIES}
        unknown = sorted(
            key for key in block if key not in _RESERVED_KEYS and key.lower() not in families
        )
        if unknown:
            raise ImproperlyConfigured(
                f"unknown RL keys {unknown}; accepted: {sorted(_RESERVED_KEYS)} and the families "
                f"{[name.upper() for name in RL_FAMILIES]}"
            )
        for family in RL_FAMILIES:
            raw = block.get(family.upper())
            if raw is not None:
                self.install_default(family, raw)
        overrides = block.get("OVERRIDES") or {}
        if not isinstance(overrides, Mapping):
            raise ImproperlyConfigured("RL.OVERRIDES must map FAMILY -> {glob: spec}")
        for family_key, table in dict(overrides).items():
            family = str(family_key).lower()
            if family not in families:
                raise ImproperlyConfigured(
                    f"RL.OVERRIDES.{family_key}: unknown family; accepted: {sorted(families)}"
                )
            if not isinstance(table, Mapping):
                raise ImproperlyConfigured(f"RL.OVERRIDES.{family_key} must map glob -> spec")
            for glob, raw in dict(table).items():
                if raw is not None:
                    self.install_override(family, str(glob), raw)

    # -- installing -------------------------------------------------------------------
    def install_default(self, family: str, raw: Any) -> Slot:
        return self._install(family, _default_path(family), raw)

    def install_override(self, family: str, glob: str, raw: Any) -> Slot:
        if family in GLOBAL_ONLY:
            raise ImproperlyConfigured(
                f"E042: {family!r} is served by one global object; RL.OVERRIDES.{family.upper()} "
                f"cannot target {glob!r}. Use RL.{family.upper()}."
            )
        if not glob:
            raise ImproperlyConfigured("an override glob cannot be empty")
        return self._install(family, _override_path(family, glob), raw)

    def adopt_default(self, family: str, instance: Any, *, name: str = "instance") -> Slot:
        """An in-code instance for the default slot (highest precedence)."""
        return self._slots.adopt(_default_path(family), f"rl_function.{family}", instance, name=name)

    def _install(self, family: str, path: str, raw: Any) -> Slot:
        kind_id = f"rl_function.{family}"
        spec = ComponentSpec.parse(kind_id, raw)
        resolved = self._resolver.resolve(spec, trusted=self._trusted)
        self._verify(family, resolved)
        slot = self._slots.install(path, kind_id, spec)
        return slot

    def _verify(self, family: str, resolved: Resolved) -> None:
        mode = self.settings.verify
        if mode == "off":
            if resolved.component.origin is not Origin.BUILTIN and not getattr(self, "_w013_logged", False):
                self._w013_logged = True
                log.warning("W013: RL.VERIFY is 'off': custom RL functions run without a contract probe")
            return
        if mode == "custom" and resolved.component.origin is Origin.BUILTIN:
            return
        check_function(family, resolved.instance, name=resolved.component.name, ctx=self._probe_context)

    # -- lookup -----------------------------------------------------------------------
    def spec(self, family: str) -> dict[str, Any] | None:
        """The default slot's canonical spec, or ``None``; for consumers that build their own."""
        path = _default_path(family)
        if path not in self._slots.paths():
            return None
        return self._slots.get(path).resolved.spec.canonical()

    def has(self, family: str, path: str = "") -> bool:
        return self._select(family, path) is not None

    def get(self, family: str, path: str = "") -> Any | None:
        """The instance serving ``path``, or ``None`` when nothing is configured for it."""
        slot = self._select(family, path)
        if slot is None:
            return None
        return self._instance(family, slot, path)

    def slot_for(self, family: str, path: str = "") -> Slot | None:
        return self._select(family, path)

    def _candidates(self, family: str) -> list[tuple[str, Slot]]:
        prefix = f"RL.{family}."
        found = []
        for slot_path in self._slots.paths():
            if slot_path.startswith(prefix):
                found.append((slot_path[len(prefix):], self._slots.get(slot_path)))
        return found

    def _select(self, family: str, path: str) -> Slot | None:
        best: tuple[int, str, Slot] | None = None
        tied: list[str] = []
        for glob, slot in self._candidates(family):
            if glob == path:
                return slot
            if not fnmatch.fnmatchcase(path, glob):
                continue
            score = specificity(glob)
            if best is None or score > best[0]:
                best, tied = (score, glob, slot), []
            elif score == best[0]:
                tied.append(glob)
        if best is not None:
            if tied:
                raise ComponentSpecError(
                    f"rl_function.{family}",
                    f"E043: node {path!r} is matched by equally specific overrides "
                    f"{sorted([best[1], *tied])}",
                )
            return best[2]
        default = _default_path(family)
        return self._slots.get(default) if default in self._slots.paths() else None

    # -- building ------------------------------------------------------------------
    @staticmethod
    def _adopted(resolved: Resolved) -> bool:
        existing = HUB.registry(resolved.component.kind).find(resolved.component.name)
        return existing is None or existing.factory is not resolved.component.factory

    def _build_raw(self, family: str, resolved: Resolved) -> Any:
        """The raw instance one node uses for ``resolved`` (no guards applied)."""
        if self._adopted(resolved):
            if family == "synthesis":
                from ..coopetition.synthesis import adapt_synthesis

                return adapt_synthesis(resolved.instance, name=resolved.component.name)
            return resolved.instance
        params = dict(resolved.spec.params)
        if family == "synthesis":
            from ..coopetition.synthesis import build_synthesis

            return build_synthesis({"type": resolved.component.name, **params})
        return HUB.registry(resolved.component.kind).build(resolved.component.name, params)

    def _guard(self, family: str, raw: Any) -> Any:
        if family == "loss":
            from .loss_compliance import ensure_compliance

            return ensure_compliance(
                raw, mode=self.settings.loss_compliance,
                coefficient=self.settings.compliance_coefficient,
            )
        if family == "rollup":
            from .guards.rollup import guard_rollup

            return guard_rollup(raw, mode=self.settings.guard_mode, include_explicit=True)
        return raw

    def _instance(self, family: str, slot: Slot, path: str) -> Any:
        key = (family, slot.path, path)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None and cached.resolved is slot.resolved:
                return cached.instance
            raw = self._build_raw(family, slot.resolved)
            instance = self._guard(family, raw)
            self._cache[key] = _Cached(slot.resolved, raw, instance)
            return instance

    # -- swapping ---------------------------------------------------------------------
    def swap_default(
        self, family: str, raw: Any, *, boundary: Boundary, migrate: str | None = None,
    ) -> Any:
        return self._swap(family, _default_path(family), raw, boundary, migrate)

    def swap_override(
        self, family: str, glob: str, raw: Any, *, boundary: Boundary, migrate: str | None = None,
    ) -> Any:
        if family in GLOBAL_ONLY:
            raise ImproperlyConfigured(f"E042: {family!r} has no per-node overrides")
        return self._swap(family, _override_path(family, glob), raw, boundary, migrate)

    def _swap(self, family: str, path: str, raw: Any, boundary: Boundary, mode: str | None) -> Any:
        kind_id = f"rl_function.{family}"
        with self._lock:
            if path not in self._slots.paths():
                # Adding a slot is an install, allowed only where the family can be installed.
                if boundary.rank > Boundary.EPISODE.rank:
                    raise ComponentLocked(kind_id, path, "a new slot can only be installed at run_start or episode")
                return self._install(family, path, raw)
            spec = ComponentSpec.parse(kind_id, raw)
            candidate = self._resolver.resolve(spec, trusted=self._trusted)
            self._verify(family, candidate)            # 1. the probe (E047), nothing committed yet
            migrate = migration_for(mode, family=family, override=self.settings.migrate)
            staged = self._stage(family, path, candidate, migrate)  # 2. migrate every live node (E048)
            record = self._slots.swap(path, spec, boundary=boundary)  # 3. lock, mutability, commit
            slot = self._slots.get(path)
            for node, (old_raw, new_raw) in staged.items():  # 4. adopt the staged instances
                self._cache[(family, path, node)] = _Cached(
                    slot.resolved, new_raw, self._guard(family, new_raw),
                )
            return record

    def _stage(
        self, family: str, path: str, candidate: Resolved, migrate: Callable[[Any, Any], None],
    ) -> dict[str, tuple[Any, Any]]:
        """Build and migrate a new instance for every node that already has a live one.

        Nodes that never called the function have nothing to migrate and will build from the
        new slot on first use. Any exception propagates out of ``_swap`` before step 3, so a
        failed migration leaves every node on its old instance.
        """
        staged: dict[str, tuple[Any, Any]] = {}
        for (cached_family, cached_path, node), cached in list(self._cache.items()):
            if cached_family != family or cached_path != path:
                continue
            new_raw = self._build_raw(family, candidate)
            migrate(cached.raw, new_raw)
            staged[node] = (cached.raw, new_raw)
        return staged

    # -- queued swaps ---------------------------------------------------------------------
    def request(
        self, family: str, glob: str | None, raw: Any, *, migrate: str | None = None,
    ) -> str:
        """Queue a swap for the next episode boundary. Validates and probes *now*.

        ``glob=None`` addresses the family default. Returns the slot path that was queued.
        """
        kind_id = f"rl_function.{family}"
        path = _default_path(family) if glob is None else _override_path(family, glob)
        if glob is not None and family in GLOBAL_ONLY:
            raise ImproperlyConfigured(f"E042: {family!r} has no per-node overrides")
        with self._lock:
            if path in self._slots.paths() and self._slots._slot_locked(path):
                raise ComponentLocked(kind_id, path, "the slot is locked by COMPONENTS.LOCKED")
            spec = ComponentSpec.parse(kind_id, raw)
            candidate = self._resolver.resolve(spec, trusted=self._trusted)
            self._verify(family, candidate)
            migration_for(migrate, family=family, override=self.settings.migrate)  # validates the mode
            self._pending[path] = (family, spec, migrate)
        return path

    def drop_pending(self, path: str) -> bool:
        with self._lock:
            return self._pending.pop(path, None) is not None

    def pending(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                path: {"family": family, "spec": spec.canonical(), "migrate": migrate}
                for path, (family, spec, migrate) in self._pending.items()
            }

    def apply_pending(self, boundary: Boundary) -> list[Any]:
        """Apply every queued swap the boundary permits. A swap that fails stays reported, not queued."""
        applied: list[Any] = []
        with self._lock:
            for path in sorted(self._pending):
                family, spec, migrate = self._pending[path]
                try:
                    applied.append(self._swap(family, path, spec, boundary, migrate))
                except ComponentLocked:
                    continue              # a stricter boundary is needed; stays queued
                except Exception as error:  # noqa: BLE001 - one bad swap must not block the others
                    log.error("queued swap of %s failed and was dropped: %s", path, error)
                    self.failed_swaps.append({"slot": path, "error": f"{type(error).__name__}: {error}"})
                del self._pending[path]
        return applied

    # -- deleting ---------------------------------------------------------------------------
    def remove_default(self, family: str, *, boundary: Boundary) -> Any:
        """Delete the default: nodes without an override go back to the built-in behaviour."""
        return self._remove(family, _default_path(family), boundary)

    def remove_override(self, family: str, glob: str, *, boundary: Boundary) -> Any:
        """Delete one override: its nodes fall back to the next most specific match, or the default."""
        return self._remove(family, _override_path(family, glob), boundary)

    def _remove(self, family: str, path: str, boundary: Boundary) -> Any:
        with self._lock:
            record = self._slots.remove(path, boundary=boundary)
            for key in [key for key in self._cache if key[0] == family and key[1] == path]:
                dropped = self._cache.pop(key)
                if is_stateful(dropped.raw):
                    log.warning("removing %s discards the state of node %r", path, key[2])
            self._pending.pop(path, None)
            return record

    def forget(self, node_path: str) -> Callable[[], None]:
        """Drop cached instances of a node that left the tree or was rebuilt. Returns the inverse."""
        with self._lock:
            dropped = {key: self._cache.pop(key) for key in list(self._cache) if key[2] == node_path}

        def undo() -> None:
            with self._lock:
                self._cache.update(dropped)

        return undo

    # -- persistence ------------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """State of every live per-node instance that has any, keyed by slot and node."""
        entries: list[dict[str, Any]] = []
        with self._lock:
            for (family, path, node), cached in sorted(self._cache.items()):
                getter = getattr(cached.raw, "state_dict", None)
                if not callable(getter) or not is_stateful(cached.raw):
                    continue
                entries.append({
                    "family": family, "slot": path, "node": node,
                    "spec_hash": cached.resolved.spec_hash,
                    "component": cached.resolved.component.name,
                    "state": getter(),
                })
        return {"version": 1, "entries": entries}

    def load_state_dict(self, payload: Mapping[str, Any], *, strict: bool = False) -> list[str]:
        """Restore per-node state. Returns warnings; raises on mismatch when ``strict``.

        A saved entry is applied only when the slot still exists and still has the spec it
        had when the state was saved. State from a different spec is not loadable (a counter
        from ``n=4`` is not a counter for ``n=8``): that is ``W012``, or ``E049`` when strict.
        """
        warnings: list[str] = []
        for entry in list(dict(payload or {}).get("entries", [])):
            family, path, node = str(entry["family"]), str(entry["slot"]), str(entry["node"])
            problem: str | None = None
            if path not in self._slots.paths():
                problem = f"slot {path} is not configured any more"
            else:
                slot = self._slots.get(path)
                if slot.resolved.spec_hash != entry.get("spec_hash"):
                    problem = (
                        f"slot {path} now holds {slot.component.name!r} with a different spec "
                        f"than the saved {entry.get('component')!r}"
                    )
            if problem is not None:
                if strict:
                    raise ComponentMigrationError(f"E049: cannot restore state for {node!r}: {problem}")
                warnings.append(f"W012: state for {family}/{node} dropped: {problem}")
                continue
            self._instance(family, self._slots.get(path), node)         # builds and caches it
            cached = self._cache[(family, path, node)]
            loader = getattr(cached.raw, "load_state_dict", None)
            if not callable(loader):
                warnings.append(f"W012: {family}/{node}: {type(cached.raw).__name__} has no load_state_dict")
                continue
            loader(dict(entry.get("state") or {}))
        for text in warnings:
            log.warning(text)
        return warnings

    # -- validation against a tree and a policy -----------------------------------------
    def validate(
        self, node_paths: Iterable[str], *, policy: Any = None, hierarchy: bool = False,
        loss_block_authored: bool = False, in_code: Iterable[str] = (), initial: bool = True,
    ) -> list[str]:
        """Return warnings; raise ``ImproperlyConfigured`` for errors. Run at build and after mutation.

        ``in_code`` names the families the caller already supplied as constructor arguments
        (``"synthesis"`` for ``synthesis=``, ``"rollup"`` for ``hierarchical_scorer=``), see 8.5.4.
        """
        warnings: list[str] = []
        paths = sorted(set(node_paths))
        configured = {family for family in RL_FAMILIES if self._family_configured(family)}

        if "loss" in configured and loss_block_authored and _default_path("loss") in self._slots.paths():
            raise ImproperlyConfigured(
                "E040: both LOSS and RL.LOSS are set; they name the same slot. Keep one "
                "(RL.LOSS is the form that supports overrides)."
            )
        for family in sorted({f for f in set(in_code) if _default_path(f) in self._slots.paths()}):
            raise ImproperlyConfigured(
                f"E040: RL.{family.upper()} is set and the {family} was also supplied in code; "
                f"they name the same default. Keep one (a per-node RL.OVERRIDES.{family.upper()} "
                "entry may still sit on top of the in-code default)."
            )
        for family in sorted(configured & {"rollup", "synthesis"}):
            if not hierarchy:
                raise ImproperlyConfigured(
                    f"E045: RL.{family.upper()} is set but this run has no hierarchy; "
                    f"{family} only exists in a hierarchy (trainer.py:304-310)"
                )
        for family in sorted(configured):
            for glob, _slot in self._candidates(family):
                if not any(fnmatch.fnmatchcase(path, glob) for path in paths):
                    message = f"RL.{family}.{glob} matches no node in the tree {paths}"
                    if self.settings.strict_globs and initial:
                        raise ImproperlyConfigured(f"E044: {message}")
                    warnings.append(f"W011: {message}")
            for path in paths:
                self._select(family, path)  # raises E043 for an ambiguous pair
        consumed = self._consumed_by(policy, hierarchy)
        for family in sorted(configured - consumed):
            message = (
                f"RL.{family.upper()} is configured but nothing consumes it: policy "
                f"{getattr(policy, 'name', type(policy).__name__)!r} declares "
                f"consumes_functions={sorted(getattr(policy, 'consumes_functions', ()))}"
            )
            if self.settings.unconsumed == "error":
                raise ImproperlyConfigured(f"E041: {message}. Choose a policy that consumes it, "
                                           "or set RL.UNCONSUMED: warn")
            warnings.append(f"W010: {message}")
        for text in warnings:
            log.warning(text)
        return warnings

    def _family_configured(self, family: str) -> bool:
        return _default_path(family) in self._slots.paths() or bool(self._candidates(family))

    @staticmethod
    def _consumed_by(policy: Any, hierarchy: bool) -> frozenset[str]:
        declared = getattr(policy, "consumes_functions", None)
        families = set(declared) if declared is not None else {"loss"}
        if hierarchy:
            families |= {"rollup", "synthesis"}
        return frozenset(families)

    # -- reporting ----------------------------------------------------------------------
    def fingerprints(self) -> dict[str, dict[str, Any]]:
        return {
            path: fingerprint for path, fingerprint in self._slots.fingerprints().items()
            if path.startswith("RL.")
        }

    def describe(self) -> dict[str, Any]:
        return {
            "settings": {
                "unconsumed": self.settings.unconsumed, "strict_globs": self.settings.strict_globs,
                "verify": self.settings.verify,
            },
            "slots": {
                path: {"component": slot.component.name, "origin": slot.component.origin.value,
                       "spec": slot.resolved.spec.canonical()}
                for path in self._slots.paths() if path.startswith("RL.")
                for slot in [self._slots.get(path)]
            },
        }
