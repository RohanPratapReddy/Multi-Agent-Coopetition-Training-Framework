"""Slots and swaps (document 17 section 4.4).

A **slot** is a place in a run where a component is in use. A run has a fixed table of
slots, created when the engine builds, and every runtime change goes through one method:
:meth:`SlotTable.swap`. The boundary rule: a swap at boundary ``b`` is allowed iff
``rank(b) <= min(rank(old.mutability), rank(new.mutability))``.
"""

from __future__ import annotations

import fnmatch
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..exceptions import (
    ComponentContractError,
    ComponentError,
    ComponentLocked,
    ComponentMigrationError,
    ComponentMissing,
)
from .kinds import KIND_SPECS, Boundary
from .registry import HUB, Component
from .resolver import ComponentResolver, ComponentSpec, Resolved

log = logging.getLogger("mactf.components")

#: ``migrate(old_instance, new_instance) -> None`` carries state across a swap.
Migrate = Callable[[Any, Any], None]


def slot_lock_names(path: str) -> tuple[str, ...]:
    """The names a slot answers to in a ``COMPONENTS.LOCKED`` ``slot:`` pattern.

    The document's lock examples (4.4, 4.5, 15.x) use the *settings* spelling of a slot
    (``SCORING.CHAIN.penalty``, ``TRAINER.OPTIMIZER.default``) while the engine installs
    the scoring stages as ``scoring.<kind>``, the reward terms as
    ``scoring.reward.terms.<label>`` and the optimizer as ``OPTIMIZER[.<glob>]``. A pin written
    in either spelling must lock the slot, or the recommended production pin locks nothing.
    """
    names = [path]
    parts = path.split(".")
    if parts[0] == "scoring" and len(parts) == 2:
        names.append(f"SCORING.CHAIN.{parts[1]}")
    elif path.startswith("scoring.reward.terms."):
        names.append("SCORING.REWARD_TERMS." + path[len("scoring.reward.terms."):])
    elif path == "OPTIMIZER":
        names.append("TRAINER.OPTIMIZER.default")
    elif path.startswith("OPTIMIZER."):
        names.append("TRAINER.OPTIMIZER." + path[len("OPTIMIZER."):])
    return tuple(names)


def slot_locked_by(path: str, patterns: Any) -> bool:
    """True when any ``slot:`` pattern matches ``path`` under any of its names."""
    return any(
        fnmatch.fnmatchcase(f"slot:{name}", pattern)
        for pattern in patterns
        if pattern.startswith("slot:")
        for name in slot_lock_names(path)
    )


@dataclass
class Slot:
    path: str
    kind_id: str
    resolved: Resolved

    @property
    def component(self) -> Component:
        return self.resolved.component

    @property
    def instance(self) -> Any:
        return self.resolved.instance


@dataclass(frozen=True)
class SwapRecord:
    slot: str
    kind_id: str
    from_name: str
    to_name: str
    from_spec_hash: str
    to_spec_hash: str
    boundary: str
    episode: int
    at: float

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class SlotTable:
    """The slots of one run, plus the queue of swaps waiting for a boundary."""

    def __init__(self, resolver: ComponentResolver, *, trusted: bool = True) -> None:
        self._resolver = resolver
        self._trusted = trusted
        self._slots: dict[str, Slot] = {}
        self._pending: dict[str, tuple[ComponentSpec, Migrate | None]] = {}
        self._history: list[SwapRecord] = []
        self._lock = threading.RLock()
        self.episode = 0

    # -- construction ----------------------------------------------------
    def install(
        self, path: str, kind_id: str, raw: Any, *, inject: Mapping[str, Any] | None = None
    ) -> Slot:
        spec = ComponentSpec.parse(kind_id, raw)
        resolved = self._resolver.resolve(spec, trusted=self._trusted, inject=inject)
        slot = Slot(path=path, kind_id=kind_id, resolved=resolved)
        with self._lock:
            self._slots[path] = slot
        return slot

    def adopt(self, path: str, kind_id: str, instance: Any, *, name: str = "instance") -> Slot:
        """Record an in-code instance (highest precedence) without building it."""
        resolved = self._resolver.adopt(kind_id, instance, name=name)
        slot = Slot(path=path, kind_id=kind_id, resolved=resolved)
        with self._lock:
            self._slots[path] = slot
        return slot

    def adopt_registered(self, path: str, kind_id: str, instance: Any, *, name: str) -> Slot:
        """Record an instance of a *registered* component that was built outside the registry.

        The slot keeps the registered component (so its origin stays ``builtin`` and its
        fingerprint is the registry's), but no ``build`` runs: the default policy is created
        with a plain call so that a default run makes no registry call at all (19.4).
        """
        component = HUB.registry(kind_id).get(name)
        bad = KIND_SPECS[kind_id].instance_contract(instance)
        if bad:
            raise ComponentContractError(kind_id, name, bad)
        spec = ComponentSpec(kind_id=kind_id, type=component.name, params={})
        resolved = Resolved(
            spec=spec, component=component, instance=instance, spec_hash=spec.hash()
        )
        slot = Slot(path=path, kind_id=kind_id, resolved=resolved)
        with self._lock:
            self._slots[path] = slot
        return slot

    def get(self, path: str) -> Slot:
        try:
            return self._slots[path]
        except KeyError:
            raise ComponentMissing("slot", path, sorted(self._slots)) from None

    def paths(self) -> list[str]:
        return sorted(self._slots)

    def match(self, pattern: str) -> list[Slot]:
        return [self._slots[p] for p in sorted(self._slots) if fnmatch.fnmatchcase(p, pattern)]

    @property
    def history(self) -> tuple[SwapRecord, ...]:
        return tuple(self._history)

    # -- locking ---------------------------------------------------------
    @staticmethod
    def _slot_locked(path: str) -> bool:
        return slot_locked_by(path, HUB.effective_policy().locked)

    # -- swapping --------------------------------------------------------
    def swap(
        self,
        path: str,
        raw: Any,
        *,
        boundary: Boundary,
        migrate: Migrate | None = None,
        inject: Mapping[str, Any] | None = None,
    ) -> SwapRecord:
        """Replace a slot's component *now*, if the boundary permits it.

        Order matters: everything that can fail (lock, mutability, spec parse, build,
        contract, migration) happens before the slot is rewritten, so a failed swap
        leaves the old component in place and running (invariant I4's analogue here).
        """
        with self._lock:
            slot = self.get(path)
            if self._slot_locked(path):
                raise ComponentLocked(
                    slot.kind_id, path, "the slot is locked by COMPONENTS.LOCKED"
                )
            spec = ComponentSpec.parse(slot.kind_id, raw)
            new = self._resolver.resolve(spec, trusted=self._trusted, inject=inject)
            allowed = min(slot.component.mutability.rank, new.component.mutability.rank)
            if boundary.rank > allowed:
                needs = {0: "run_start", 1: "episode", 2: "step"}[allowed]
                raise ComponentLocked(
                    slot.kind_id,
                    path,
                    f"{slot.component.name!r}->{new.component.name!r} may only change at "
                    f"{needs} boundaries, not {boundary.value}",
                )
            if migrate is not None:
                try:
                    migrate(slot.instance, new.instance)
                except ComponentError:
                    raise
                except Exception as error:  # noqa: BLE001 - the old component keeps running
                    raise ComponentMigrationError(
                        f"state migration for slot {path!r} failed: "
                        f"{type(error).__name__}: {error}; the old component keeps running"
                    ) from error
            record = SwapRecord(
                slot=path,
                kind_id=slot.kind_id,
                from_name=slot.component.name,
                to_name=new.component.name,
                from_spec_hash=slot.resolved.spec_hash,
                to_spec_hash=new.spec_hash,
                boundary=boundary.value,
                episode=self.episode,
                at=time.time(),
            )
            slot.resolved = new
            self._history.append(record)
        log.info("slot %s: %s -> %s at %s", path, record.from_name, record.to_name, boundary.value)
        return record

    def remove(self, path: str, *, boundary: Boundary) -> SwapRecord:
        """Delete a slot. Allowed on the same boundary rule as ``swap``; recorded in history."""
        with self._lock:
            slot = self.get(path)
            if self._slot_locked(path):
                raise ComponentLocked(slot.kind_id, path, "the slot is locked by COMPONENTS.LOCKED")
            allowed = slot.component.mutability.rank
            if boundary.rank > allowed:
                needs = {0: "run_start", 1: "episode", 2: "step"}[allowed]
                raise ComponentLocked(
                    slot.kind_id, path,
                    f"{slot.component.name!r} may only be removed at {needs} boundaries, "
                    f"not {boundary.value}",
                )
            record = SwapRecord(
                slot=path, kind_id=slot.kind_id,
                from_name=slot.component.name, to_name="<removed>",
                from_spec_hash=slot.resolved.spec_hash, to_spec_hash="",
                boundary=boundary.value, episode=self.episode, at=time.time(),
            )
            del self._slots[path]
            self._pending.pop(path, None)
            self._history.append(record)
        log.info("slot %s: %s removed at %s", path, record.from_name, boundary.value)
        return record

    def request(self, path: str, raw: Any, *, migrate: Migrate | None = None) -> None:
        """Queue a swap for the next episode boundary. Validates eagerly: an invalid
        spec fails the request, not the next episode."""
        with self._lock:
            slot = self.get(path)
            if self._slot_locked(path):
                raise ComponentLocked(
                    slot.kind_id, path, "the slot is locked by COMPONENTS.LOCKED"
                )
            spec = ComponentSpec.parse(slot.kind_id, raw)
            self._resolver.validate(spec, trusted=self._trusted)
            self._pending[path] = (spec, migrate)

    def apply_pending(
        self, boundary: Boundary, *, inject: Mapping[str, Any] | None = None
    ) -> list[SwapRecord]:
        """Apply every queued swap that the boundary permits; keep the rest queued."""
        applied: list[SwapRecord] = []
        with self._lock:
            for path in sorted(self._pending):
                spec, migrate = self._pending[path]
                try:
                    applied.append(
                        self.swap(path, spec, boundary=boundary, migrate=migrate, inject=inject)
                    )
                except ComponentLocked:
                    continue  # a stricter boundary is needed; stays queued
                del self._pending[path]
        return applied

    def pending(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {path: spec.canonical() for path, (spec, _m) in self._pending.items()}

    def drop_pending(self, path: str) -> bool:
        with self._lock:
            return self._pending.pop(path, None) is not None

    # -- reporting -------------------------------------------------------
    def fingerprints(self) -> dict[str, dict[str, Any]]:
        return {path: slot.resolved.fingerprint() for path, slot in sorted(self._slots.items())}
