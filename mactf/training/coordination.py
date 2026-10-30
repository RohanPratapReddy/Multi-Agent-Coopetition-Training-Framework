"""The coordination context: the only verbs a peer strategy has (document 17 section 11.6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from ..coopetition.strategy import GroupRef
from ..exceptions import ContractError

ROOT_LABEL = "root"


class CoordinationBackend(Protocol):
    """What ``MACTFTrainer`` provides to the context (and what the probe's synthetic trainer fakes)."""

    tree: Any
    freeze: Any
    signal_by_peer: Mapping[str, float]
    #: Parameter writes performed so far (``_write`` increments it). The guard compares its
    #: delta over one call with the number of writes the context audited (``Y003``).
    write_count: int

    def update_weights(self, eligible: Mapping[str, bool] | None = None) -> Any: ...
    def aggregate(self, report: Any = None, *, update_reference: bool = True) -> Any: ...
    def aggregate_group(self, parent: str | None, members: Sequence[str], *,
                        eligible: Mapping[str, bool] | None = None) -> Any: ...
    def peers_void(self, members: Sequence[str]) -> bool: ...
    def legal_report(self, report: Any, parent: str | None, members: Sequence[str]) -> list[str]: ...
    def record_signal_sharing(self, signals_map: Mapping[str, float], *, epoch: int) -> None: ...
    def peer_digest(self, path: str) -> str: ...
    def request_mutation(self, plan: Any, *, source: str = "request") -> Any: ...


class CoordinationError(ContractError):
    """A verb's pre-condition failed. ``code`` is a ``Y`` code (11.7); nothing was written."""

    def __init__(self, code: str, detail: str, *, peer: str | None = None) -> None:
        self.code = code
        self.detail = detail
        self.peer = peer
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class AuditEvent:
    seq: int
    kind: str
    group: str | None
    members: tuple[str, ...] = ()
    detail: Mapping[str, Any] = field(default_factory=dict)


class CoordinationState:
    """What happened during one epoch's sweep; shared by every context of that sweep."""

    def __init__(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.events: list[AuditEvent] = []
        self.written: dict[str, int] = {}
        self.root_report: Any = None
        self.root_eligible: dict[str, bool] | None = None
        self.produced: set[int] = set()

    def record(self, kind: str, group: str | None, members: Sequence[str] = (), **detail: Any) -> AuditEvent:
        event = AuditEvent(len(self.events), kind, group, tuple(members), dict(detail))
        self.events.append(event)
        return event


class CoordinationContext:
    """The verbs of one group call. See the table in 11.6."""

    def __init__(
        self,
        backend: CoordinationBackend,
        state: CoordinationState,
        groups: Sequence[GroupRef],
        *,
        owner: str,
        checks: bool,
    ) -> None:
        self._backend = backend
        self.state = state
        self.groups: tuple[GroupRef, ...] = tuple(groups)
        self.owner = owner
        self._checks = bool(checks)

    @property
    def checks(self) -> bool:
        """Whether the pre-conditions run. Read-only: a strategy cannot switch them off."""
        return self._checks

    # -- reading ---------------------------------------------------------
    @property
    def epoch(self) -> int:
        return self.state.epoch

    @property
    def group(self) -> GroupRef:
        return self.groups[0]

    @property
    def includes_root(self) -> bool:
        return any(group.is_root for group in self.groups)

    @property
    def tree(self) -> Any:
        return self._backend.tree

    @property
    def signals(self) -> dict[str, float]:
        return dict(self._backend.signal_by_peer or {})

    @property
    def freeze_mode(self) -> str:
        return self._backend.freeze.mode

    def peers_void(self, members: Sequence[str]) -> bool:
        return bool(self._backend.peers_void(list(members)))

    def eligible_for(self, members: Sequence[str]) -> dict[str, bool] | None:
        """Who may carry weight in a group of ``members``; ``None`` when everyone may."""
        return self._backend.freeze.eligible_for(list(members))

    @property
    def retired(self) -> frozenset[str]:
        """Peers that are out of play (document 17 section 12.7.5); rarely needed, because the
        ``TreeView`` a strategy receives is built from the active view."""
        return frozenset(self._backend.freeze.retired())

    def request_mutation(self, plan: Any, *, source: str = "strategy") -> Any:
        """Queue a topology change for the top of the next epoch; never applies it now (M2)."""
        request = getattr(self._backend, "request_mutation", None)
        if not callable(request):
            raise CoordinationError("Y008", "this backend cannot queue a mutation plan")
        return request(plan, source=source)

    def unchecked(self) -> "CoordinationContext":
        """The same context without the bookkeeping pre-conditions (scope, one write per group,
        order; the guard's fallback uses it). The compliance ones, Y001 and Y002, never switch
        off: a report that no verb produced is always checked, and a void group is never blended."""
        return CoordinationContext(self._backend, self.state, self.groups, owner=self.owner, checks=False)

    # -- checks ----------------------------------------------------------
    def _in_scope(self, group: GroupRef) -> None:
        if self.checks and group.label not in {item.label for item in self.groups}:
            raise CoordinationError(
                "Y008", f"group {group.label!r} is outside the scope of this call "
                        f"({[item.label for item in self.groups]})",
            )

    def _ancestor_labels(self, group: GroupRef) -> list[str]:
        labels: list[str] = []
        node = group.parent
        tree = self._backend.tree
        while node is not None:
            above = tree.parent(node)
            labels.append(ROOT_LABEL if above is None else above)
            node = above
        return labels

    def _precheck_write(self, group: GroupRef) -> None:
        if not self.checks:
            return
        self._in_scope(group)
        if self.state.written.get(group.label):
            raise CoordinationError("Y003", f"group {group.label!r} was already written this epoch")
        for label in self._ancestor_labels(group):
            if self.state.written.get(label):
                raise CoordinationError(
                    "Y004", f"group {group.label!r} would be written after its ancestor group {label!r}"
                )

    # -- mutating verbs --------------------------------------------------
    def update_root(self, eligible: Mapping[str, bool] | None = None) -> Any:
        if self.state.root_report is not None:
            if eligible is not None and self.checks and dict(eligible) != (self.state.root_eligible or {}):
                raise CoordinationError("Y003", "the root weights were already computed with a different eligibility")
            return self.state.root_report
        report = self._backend.update_weights(eligible=eligible)
        self.state.root_report = report
        self.state.root_eligible = None if eligible is None else dict(eligible)
        self.state.produced.add(id(report))
        self.state.record("update_root", ROOT_LABEL, entries=len(getattr(report, "weights", {}) or {}))
        return report

    def aggregate_root(self, report: Any = None, *, update_reference: bool = True) -> Any:
        root = next((g for g in self.groups if g.is_root), None)
        if root is None:
            raise CoordinationError("Y008", "aggregate_root was called from a call that does not serve the root group")
        self._precheck_write(root)
        if report is None:
            report = self.update_root()
        if id(report) not in self.state.produced:
            problems = self._backend.legal_report(report, None, root.members)
            if problems:
                raise CoordinationError("Y001", "the report is not legal: " + "; ".join(problems))
        result = self._backend.aggregate(report, update_reference=bool(update_reference))
        self.state.written[ROOT_LABEL] = self.state.written.get(ROOT_LABEL, 0) + 1
        self.state.record("aggregate_root", ROOT_LABEL, root.members, update_reference=bool(update_reference))
        return result

    def aggregate_group(self, group: GroupRef, *, eligible: Mapping[str, bool] | None = None) -> Any:
        if group.is_root:
            raise CoordinationError("Y008", "aggregate_group serves child groups; use aggregate_root for the root group")
        self._precheck_write(group)
        if self._backend.peers_void(list(group.members)):
            raise CoordinationError("Y002", f"every member of group {group.label!r} was voided; it is never blended")
        report = self._backend.aggregate_group(group.parent, list(group.members), eligible=eligible)
        self.state.produced.add(id(report))
        self.state.written[group.label] = self.state.written.get(group.label, 0) + 1
        self.state.record("aggregate_group", group.label, group.members)
        return report

    def skip(self, group: GroupRef, reason: str) -> None:
        self._in_scope(group)
        self.state.record("skip", group.label, group.members, reason=str(reason))

    def freeze_peer(self, path: str) -> bool:
        self._member(path)
        fresh = self._backend.freeze.add(path, f"strategy:{self.owner}")
        self.state.record("freeze", self.group.label, (path,))
        return fresh

    def thaw_peer(self, path: str) -> bool:
        self._member(path)
        gone = self._backend.freeze.discard(path, f"strategy:{self.owner}")
        self.state.record("thaw", self.group.label, (path,))
        return gone

    def _member(self, path: str) -> None:
        if self.checks and not any(path in group.members for group in self.groups):
            raise CoordinationError("Y008", f"{path!r} is not a member of the group(s) of this call", peer=path)

    def record_signal_sharing(self, signals: Mapping[str, float]) -> None:
        self._backend.record_signal_sharing(dict(signals), epoch=self.state.epoch)
        self.state.record("signals", self.group.label, tuple(signals), count=len(signals))
