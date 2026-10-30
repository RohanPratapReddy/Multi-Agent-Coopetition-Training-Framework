"""StrategyGuard: what a custom peer strategy may not do (document 17 section 11.7)."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Mapping

from ...coopetition.strategy import BUILTIN_STRATEGY_TYPES, traits_of
from ...exceptions import ImproperlyConfigured, StrategyGuardViolation
from ...training.coordination import CoordinationContext, CoordinationError, CoordinationState

log = logging.getLogger("mactf.strategy")

GUARD_MODES = ("skip", "fallback", "enforce")
GUARD_SCOPES = ("custom", "all", "off")
_GUARD_KEYS = frozenset({"MODE", "SCOPE", "PROBE", "CHECK_FROZEN"})

CODES: dict[str, str] = {
    "Y001": "a report handed to aggregate_root is not a legal simplex for its group",
    "Y002": "a group whose members are all void was blended",
    "Y003": "a group was written twice, the root weights were recomputed differently, or a write bypassed the context",
    "Y004": "a child group was written after its ancestor group",
    "Y005": "the parameters of a frozen peer changed under FREEZE_MODE=strict",
    "Y006": "the tree structure changed outside a MutationPlan",
    "Y007": "coordinate() raised",
    "Y008": "a verb named a group or peer outside the scope of the call",
}
_HARD = frozenset({"Y005", "Y006"})


@dataclass(frozen=True)
class Violation:
    code: str
    peer: str | None
    detail: str

    @property
    def hard(self) -> bool:
        return self.code in _HARD

    def __str__(self) -> str:
        where = "" if self.peer is None else f" [{self.peer}]"
        return f"{self.code}{where}: {self.detail}"


@dataclass(frozen=True)
class GuardOutcome:
    violations: tuple[Violation, ...]
    action: str                       # "pass" | "skipped" | "fallback"


@dataclass(frozen=True)
class _Snapshot:
    topology: str
    frozen: Mapping[str, str]
    writes: int | None
    audited: int


def topology_digest(tree: Any) -> str:
    """A digest of paths, parents and roles; statuses are excluded (they are transient)."""
    rows = sorted(
        (node.path, str(node.parent), str(getattr(node, "role", "")))
        for node in tree.walk("root-first")
    )
    return hashlib.sha256(repr(rows).encode("utf-8")).hexdigest()[:16]


class StrategyGuard:
    """Snapshots, runs and audits one ``coordinate`` call."""

    def __init__(
        self,
        *,
        mode: str = "skip",
        scope: str = "custom",
        probe: bool = True,
        check_frozen: bool = True,
    ) -> None:
        if mode not in GUARD_MODES:
            raise ImproperlyConfigured(
                f"E070: TRAINER.STRATEGY_GUARD.MODE must be one of {list(GUARD_MODES)}, got {mode!r}"
            )
        if scope not in GUARD_SCOPES:
            raise ImproperlyConfigured(
                f"E070: TRAINER.STRATEGY_GUARD.SCOPE must be one of {list(GUARD_SCOPES)}, got {scope!r}"
            )
        self.mode = mode
        self.scope = scope
        self.probe = bool(probe)
        self.check_frozen = bool(check_frozen)

    # -- construction ----------------------------------------------------
    @classmethod
    def from_settings(cls, block: Mapping[str, Any] | None) -> "StrategyGuard":
        raw = {str(key).upper(): value for key, value in dict(block or {}).items()}
        unknown = sorted(set(raw) - _GUARD_KEYS)
        if unknown:
            raise ImproperlyConfigured(
                f"E070: TRAINER.STRATEGY_GUARD has unknown keys {unknown}; accepted: {sorted(_GUARD_KEYS)}"
            )
        return cls(
            mode=str(raw.get("MODE", "skip")).lower(),
            scope=str(raw.get("SCOPE", "custom")).lower(),
            probe=bool(raw.get("PROBE", True)),
            check_frozen=bool(raw.get("CHECK_FROZEN", True)),
        )

    def describe(self) -> dict[str, Any]:
        return {"mode": self.mode, "scope": self.scope, "probe": self.probe,
                "check_frozen": self.check_frozen}

    # -- scope -----------------------------------------------------------
    def covers(self, instance: Any) -> bool:
        if self.scope == "off":
            return False
        if self.scope == "all":
            return True
        return not any(type(instance) is built for built in BUILTIN_STRATEGY_TYPES)

    # -- the call --------------------------------------------------------
    def _snapshot(self, backend: Any, state: CoordinationState) -> _Snapshot:
        frozen: dict[str, str] = {}
        if self.check_frozen:
            freeze = backend.freeze
            for path in freeze.paths():
                if freeze.skips_write(path):
                    frozen[path] = backend.peer_digest(path)
        return _Snapshot(
            topology=topology_digest(backend.tree),
            frozen=frozen,
            writes=getattr(backend, "write_count", None),
            audited=sum(state.written.values()),
        )

    def _after(self, backend: Any, state: CoordinationState, before: _Snapshot) -> list[Violation]:
        found: list[Violation] = []
        if topology_digest(backend.tree) != before.topology:
            found.append(Violation("Y006", None, "paths, parents or roles of the tree changed"))
        for path, digest in sorted(before.frozen.items()):
            if backend.peer_digest(path) != digest:
                found.append(Violation("Y005", path, "its parameters changed while it was frozen"))
        if before.writes is not None:
            unaudited = (backend.write_count - before.writes) - (sum(state.written.values()) - before.audited)
            if unaudited != 0:
                found.append(Violation(
                    "Y003", None, f"{unaudited} parameter write(s) did not go through the context",
                ))
        return found

    def run(
        self,
        label: str,
        instance: Any,
        ctx: CoordinationContext,
        state: CoordinationState,
        *,
        backend: Any,
    ) -> GuardOutcome:
        """Run ``instance.coordinate(ctx)`` under the guard. Raises only ``E073``."""
        before = self._snapshot(backend, state)
        found: list[Violation] = []
        try:
            instance.coordinate(ctx)
        except CoordinationError as error:
            found.append(Violation(error.code, error.peer, error.detail))
        except Exception as error:  # noqa: BLE001 - a strategy is user code; Y007 reports it
            found.append(Violation("Y007", None, f"{type(error).__name__}: {error}"))
        found.extend(self._after(backend, state, before))
        if not found:
            return GuardOutcome((), "pass")
        if self.mode == "enforce" or any(item.hard for item in found):
            raise StrategyGuardViolation(label, found)
        listed = "; ".join(str(item) for item in found)
        fell_back = self.mode == "fallback" and self._fallback(instance, ctx)
        log.warning(
            "W030: peer_strategy %r broke the coordination contract (%s); %s", label, listed,
            "the standard mix ran for the group" if fell_back else "its outcome for the group stands",
        )
        state.record(
            "violation", ctx.group.label, ctx.group.members,
            codes=sorted({item.code for item in found}), action="fallback" if fell_back else "skipped",
        )
        return GuardOutcome(tuple(found), "fallback" if fell_back else "skipped")

    def _fallback(self, instance: Any, ctx: CoordinationContext) -> bool:
        """The standard mix for the call's group. True when something was written."""
        if not traits_of(instance).aggregates:
            return False
        free = ctx.unchecked()
        wrote = False
        for group in ctx.groups:
            if ctx.state.written.get(group.label):
                continue
            eligible = ctx.eligible_for(group.members)
            if eligible is not None and not any(eligible.values()):
                continue
            if group.is_root:
                free.aggregate_root(free.update_root(eligible))
                wrote = True
            elif not ctx.peers_void(group.members):
                free.aggregate_group(group, eligible=eligible)
                wrote = True
        return wrote
