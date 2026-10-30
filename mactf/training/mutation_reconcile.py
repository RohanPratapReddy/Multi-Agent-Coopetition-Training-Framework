"""Reconcile a trainer's topology with a checkpoint's (document 17 section 12.9).

Pure Python: imports no torch, so the CLI and the API can read a checkpoint's mutation state.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..coopetition.tree import PeerTree
from ..exceptions import MutationError
from .mutation import (
    BUILD_TOKEN, INTERNAL_SOURCES, MutationPlan, add_peer, modify_peer, parent_of, purge_peer,
    retire_peer, revive_peer, structure_digest, tree_state_digest,
)

log = logging.getLogger("mactf.mutation")

STATE_VERSION = 1
_STATE_KEYS = frozenset(
    {"version", "journal", "topology_epoch", "base_digest", "digest", "tombstones", "specs", "metadata", "pending"}
)
_ENTRY_KEYS = ("seq", "epoch", "topology_epoch", "source", "plan", "plan_digest", "before", "after")
_MISSING = object()


def _text(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key, "")
    if not isinstance(value, str):
        raise MutationError("E089", f"the checkpoint's mutation state has a non-text `{key}`: {value!r}")
    return value


def _count(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MutationError("E089", f"the checkpoint's mutation state has an invalid `{key}`: {value!r}")
    return value


def _mapping_of_mappings(raw: Mapping[str, Any], key: str) -> dict[str, dict[str, Any]]:
    value = raw.get(key) or {}
    if not isinstance(value, Mapping) or not all(isinstance(item, Mapping) for item in value.values()):
        raise MutationError("E089", f"the checkpoint's mutation state `{key}` must map a path to a mapping")
    return {str(path): copy.deepcopy(dict(item)) for path, item in value.items()}


@dataclass(frozen=True)
class MutationState:
    """What a checkpoint says about how its tree came to be (``identity["trainer"]["mutation"]``).

    ``{}`` (no state at all) is what a run that never mutated writes, which is why the checkpoint
    of such a run is byte-for-byte what it has always been (M11).
    """

    journal: tuple[dict[str, Any], ...] = ()
    topology_epoch: int = 0
    base_digest: str = ""        # the tree as ``build()`` made it: what a configuration reproduces
    digest: str = ""             # the tree this state describes: must equal the checkpoint's tree section
    tombstones: tuple[str, ...] = ()
    specs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    pending: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "specs", dict(self.specs or {}))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    # -- serialisation ---------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "journal": [copy.deepcopy(dict(entry)) for entry in self.journal],
            "topology_epoch": int(self.topology_epoch),
            "base_digest": self.base_digest,
            "digest": self.digest,
            "tombstones": list(self.tombstones),
            "specs": {path: copy.deepcopy(dict(spec)) for path, spec in self.specs.items()},
            "metadata": {path: copy.deepcopy(dict(meta)) for path, meta in self.metadata.items()},
            "pending": [copy.deepcopy(dict(row)) for row in self.pending],
        }

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "MutationState":
        """Validate the shape of a stored state. ``E089`` for anything that is not one."""
        if not isinstance(raw, Mapping):
            raise MutationError("E089", f"the checkpoint's mutation state is a {type(raw).__name__}, not a mapping")
        version = raw.get("version", STATE_VERSION)
        if version != STATE_VERSION:
            raise MutationError(
                "E089",
                f"the checkpoint's mutation state is version {version!r}; this build reads version {STATE_VERSION}",
            )
        unknown = sorted(set(raw) - _STATE_KEYS)
        if unknown:
            raise MutationError("E089", f"the checkpoint's mutation state has unknown keys {unknown}")
        journal = raw.get("journal") or []
        if isinstance(journal, (str, bytes)) or not isinstance(journal, Sequence):
            raise MutationError("E089", "the checkpoint's mutation journal must be a list")
        rows: list[dict[str, Any]] = []
        for index, entry in enumerate(journal):
            if not isinstance(entry, Mapping):
                raise MutationError("E089", f"journal entry {index} is not a mapping")
            missing = [key for key in _ENTRY_KEYS if key not in entry]
            if missing:
                raise MutationError("E089", f"journal entry {index} lacks {missing}")
            rows.append(copy.deepcopy(dict(entry)))
        pending = raw.get("pending") or []
        if isinstance(pending, (str, bytes)) or not isinstance(pending, Sequence) or not all(
            isinstance(row, Mapping) and "plan" in row for row in pending
        ):
            raise MutationError("E089", "the checkpoint's pending mutations must be a list of {source, plan}")
        return cls(
            journal=tuple(rows),
            topology_epoch=_count(raw, "topology_epoch"),
            base_digest=_text(raw, "base_digest"),
            digest=_text(raw, "digest"),
            tombstones=tuple(sorted(str(path) for path in (raw.get("tombstones") or ()))),
            specs=_mapping_of_mappings(raw, "specs"),
            metadata=_mapping_of_mappings(raw, "metadata"),
            pending=tuple(copy.deepcopy(dict(row)) for row in pending),
        )

    # -- coherence -------------------------------------------------------
    @property
    def user_driven(self) -> bool:
        """True when a journal entry came from a source that needs ``TRAINER.MUTATION.ENABLED``."""
        return any(entry["source"] not in INTERNAL_SOURCES for entry in self.journal)

    def verify_chain(self, tree_digest: str) -> list[str]:
        """Every way the journal fails to explain the checkpoint's tree. Empty means coherent.

        The chain is ``base -> entry 0 -> entry 1 -> ... -> digest``, each entry's ``before`` equal
        to the previous ``after``, ``seq`` contiguous, each plan hashing to its recorded digest, and
        the end equal to the tree section the checkpoint actually holds.
        """
        problems: list[str] = []
        previous = self.base_digest
        for index, entry in enumerate(self.journal):
            if entry["seq"] != index:
                problems.append(f"entry {index} has seq {entry['seq']!r}; the journal is not contiguous")
            if entry["before"] != previous:
                problems.append(
                    f"entry {index} ({entry['plan_digest']}) starts from structure {entry['before']}, "
                    f"but the structure before it was {previous}"
                )
            try:
                recomputed = MutationPlan.parse(entry["plan"]).digest()
            except MutationError as error:
                problems.append(f"entry {index}: the recorded plan does not parse: {error}")
            else:
                if recomputed != entry["plan_digest"]:
                    problems.append(
                        f"entry {index}: the recorded plan hashes to {recomputed}, not {entry['plan_digest']}"
                    )
            previous = entry["after"]
        if previous != self.digest:
            problems.append(f"the journal ends at structure {previous}, but the state describes {self.digest}")
        if self.digest != tree_digest:
            problems.append(
                f"the state describes structure {self.digest}, but the checkpoint's tree section is {tree_digest}"
            )
        return problems


@dataclass(frozen=True)
class TargetNode:
    path: str
    parent: str | None
    role: str
    retired: bool
    depth: int


@dataclass(frozen=True)
class Target:
    """The structure a checkpoint describes, in root-first order."""

    nodes: Mapping[str, TargetNode]
    specs: Mapping[str, Mapping[str, Any]]
    metadata: Mapping[str, Mapping[str, Any]]
    digest: str

    @classmethod
    def build(
        cls,
        tree_state: Mapping[str, Any],
        state: MutationState | None,
        build_meta: Mapping[str, Mapping[str, Any]],
    ) -> "Target":
        try:
            tree = PeerTree.from_dict(tree_state)
        except Exception as error:  # noqa: BLE001 - a tree that does not parse is a corrupt checkpoint
            raise MutationError("E089", f"the checkpoint's tree section does not parse: {error}") from error
        nodes = {
            node.path: TargetNode(node.path, node.parent, node.role, node.status == "pruned", int(node.depth))
            for node in tree.walk("root-first")
        }
        metadata: dict[str, dict[str, Any]] = {}
        for path in nodes:
            if state is not None and path in state.metadata:
                metadata[path] = copy.deepcopy(dict(state.metadata[path]))
            elif path in build_meta:
                metadata[path] = copy.deepcopy(dict(build_meta[path]))
        retired = [path for path, node in nodes.items() if node.retired]
        return cls(
            nodes=nodes,
            specs={} if state is None else {path: dict(spec) for path, spec in state.specs.items()},
            metadata=metadata,
            digest=structure_digest(tree, retired),
        )


@dataclass(frozen=True)
class ReconcilePlan:
    """The plan that turns the live structure into the stored one, and what it will do."""

    plan: MutationPlan | None
    revived: tuple[str, ...] = ()
    modified: tuple[str, ...] = ()
    rebuilt: tuple[str, ...] = ()
    purged: tuple[str, ...] = ()
    added: tuple[str, ...] = ()
    retired: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return self.plan is None

    def summary(self) -> str:
        parts = [
            f"{name} {len(items)}"
            for name, items in (
                ("added", self.added), ("purged", self.purged), ("retired", self.retired),
                ("revived", self.revived), ("rebuilt", self.rebuilt), ("modified", self.modified),
            )
            if items
        ]
        return ", ".join(parts) or "nothing"


def _ancestors(path: str) -> list[str]:
    result: list[str] = []
    parent = parent_of(path)
    while parent is not None:
        result.append(parent)
        parent = parent_of(parent)
    return result


def _rebuild_block(spec: Mapping[str, Any] | None, available: set[str]) -> dict[str, Any]:
    """The ``rebuild`` mapping for a stored spec: its architecture, or the build-time module."""
    block: dict[str, Any] = {}
    if spec is not None:
        if spec.get("factory") is not None:
            block["factory"] = spec["factory"]
        if spec.get("like") is not None and spec["like"] in available:
            block["like"] = spec["like"]
    return block or {"factory": BUILD_TOKEN}


def plan_reconciliation(
    tree: PeerTree,
    retired: Iterable[str],
    specs: Mapping[str, Mapping[str, Any]],
    target: Target,
) -> ReconcilePlan:
    """The operations, in an order every one of them is valid at its turn.

    1. ``revive`` every retired node that must be active in the end, or that a later step must
       touch (a retired node cannot be modified, and nothing can be added under it), with its
       retired ancestors, root first.
    2. ``modify`` the nodes whose architecture, role or metadata differ.
    3. ``purge`` the topmost live-only nodes, recursively and forced (a purge takes the subtree).
    4. ``add`` the stored-only nodes, parents first, each with its stored spec.
    5. ``retire`` what the stored tree says is retired and is not now, deepest first.
    """
    gone_now = set(retired)
    live = {node.path: node for node in tree.walk("root-first")}
    gone = {path for path in live if path not in target.nodes}
    kept = [path for path in live if path in target.nodes]
    kept_set = set(kept)
    created = [path for path in target.nodes if path not in live]

    # -- what changes on a node that exists on both sides ----------------
    changes: dict[str, dict[str, Any]] = {}
    for path in kept:
        node = live[path]
        change: dict[str, Any] = {}
        want_spec = target.specs.get(path)
        have_spec = specs.get(path)
        if (None if want_spec is None else dict(want_spec)) != (None if have_spec is None else dict(have_spec)):
            change["rebuild"] = _rebuild_block(want_spec, kept_set)
        if node.role != target.nodes[path].role:
            change["role"] = target.nodes[path].role
        have = dict(node.metadata)
        want = dict(target.metadata.get(path, {}))
        delta = {key: value for key, value in want.items() if have.get(key, _MISSING) != value}
        delta.update({key: None for key in have if key not in want})
        if delta:
            change["metadata"] = delta
        if change:
            changes[path] = change

    # -- which retired nodes must be active while the plan runs ----------
    must_be_active = set(changes)
    for path in created:
        parent = parent_of(path)
        if parent is not None and parent in kept_set:
            must_be_active.add(parent)
    for path in list(must_be_active):
        must_be_active.update(ancestor for ancestor in _ancestors(path) if ancestor in kept_set)
    revive = [
        path for path in kept
        if path in gone_now and (not target.nodes[path].retired or path in must_be_active)
    ]
    still_retired = {path for path in kept if path in gone_now and path not in revive}

    ops: list[Any] = []
    for path in revive:                                   # kept is root-first
        ops.append(revive_peer(path, reason="checkpoint reconciliation"))
    for path, change in changes.items():
        ops.append(modify_peer(path, **change))
    top_gone = [path for path in live if path in gone and parent_of(path) not in gone]
    for path in top_gone:
        ops.append(purge_peer(path, recursive=True, force=True, reason="checkpoint reconciliation"))

    available = set(kept_set)
    for path in created:                                  # target order is root-first
        spec = {key: value for key, value in dict(target.specs.get(path) or {}).items() if key != "path"}
        if spec.get("like") is not None and spec["like"] not in available:
            spec.pop("like")                              # the template is gone; the build-time module is the fallback
        spec["role"] = target.nodes[path].role
        meta = target.metadata.get(path)
        if meta:
            spec["metadata"] = dict(meta)
        ops.append(add_peer(path, **spec))
        available.add(path)

    to_retire = [
        path for path, node in target.nodes.items()
        if node.retired and path not in still_retired
    ]
    to_retire.sort(key=lambda item: target.nodes[item].depth, reverse=True)
    for path in to_retire:
        ops.append(retire_peer(path, reason="checkpoint reconciliation"))

    if not ops:
        return ReconcilePlan(plan=None)
    plan = MutationPlan.of(
        *ops, label=f"reconcile:{target.digest}", reason="the checkpoint's topology differs from the live one"
    )
    return ReconcilePlan(
        plan=plan,
        revived=tuple(revive),
        modified=tuple(changes),
        rebuilt=tuple(path for path, change in changes.items() if "rebuild" in change),
        purged=tuple(top_gone),
        added=tuple(created),
        retired=tuple(to_retire),
    )


class TopologyReconciler:
    """Make a built trainer's structure the one a checkpoint describes, before its parameters load."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

    # -- entry point -----------------------------------------------------
    def reconcile(
        self, identity: Mapping[str, Any], tree_state: Mapping[str, Any], *, strict: bool = True
    ) -> list[str]:
        """Returns the notes for the ``ResumeReport``. Raises ``MutationError`` and changes nothing
        when the checkpoint cannot be reproduced (``E080``, ``E088``, ``E089``)."""
        t = self.trainer
        notes: list[str] = []
        stored = identity.get("trainer") if isinstance(identity, Mapping) else None
        raw = stored.get("mutation") if isinstance(stored, Mapping) else None
        state = MutationState.parse(raw) if raw else None
        mutated = bool(t.mutation_journal or t.topology_epoch or t._node_specs or t._node_meta)
        if state is None and not mutated:
            # Neither side ever mutated: the restore's own comparison (a topology that differs is a
            # configuration that changed) is the whole story, exactly as before 12.
            return notes

        enforce = bool(strict and t.mutation_limits.strict_replay)
        target_digest = tree_state_digest(tree_state)
        if state is None and target_digest != t._base_digest:
            # A live run that mutated, and a checkpoint that is not a snapshot of *this* run's
            # initial tree and carries no journal: it belongs to some other configuration.
            # The restore's comparison names the difference.
            return notes

        problems: list[str] = []
        disabled = False
        if state is not None:
            problems.extend(state.verify_chain(target_digest))
            if state.base_digest != t._base_digest:
                problems.append(
                    f"the configuration builds structure {t._base_digest}, but the run started from "
                    f"{state.base_digest}: the hierarchy or the model sources changed since the checkpoint"
                )
            if state.user_driven and not t.mutation_limits.enabled:
                disabled = True
                problems.append(
                    "the journal holds user-requested mutations but TRAINER.MUTATION.ENABLED is false"
                )
        if problems:
            text = "; ".join(problems)
            if enforce:
                code = "E080" if disabled and len(problems) == 1 else "E089"
                raise MutationError(
                    code,
                    "the checkpoint's mutation journal does not describe a structure this configuration "
                    f"can resume (set TRAINER.MUTATION.STRICT_REPLAY: false to tolerate it): {text}",
                    findings=problems,
                )
            log.warning("W045: tolerated an inconsistent mutation journal: %s", text)
            notes.append(f"W045: tolerated an inconsistent mutation journal: {text}")

        target = Target.build(tree_state, state, t._build_meta)
        difference = plan_reconciliation(t.tree, t.freeze.retired(), t._node_specs, target)
        report = None
        if difference.plan is not None:
            saved_cycle = t._mutations_in_cycle
            try:
                report = t.engine.apply(difference.plan, source="replay")
            except MutationError as error:
                raise MutationError(
                    error.code,
                    f"cannot reproduce the checkpoint's topology ({difference.summary()}): {error}",
                    findings=error.findings,
                    rolled_back=error.rolled_back,
                ) from error
            finally:
                t._mutations_in_cycle = saved_cycle
        reached = structure_digest(t.tree, t.freeze.retired())
        if reached != target.digest:
            raise MutationError(
                "E089",
                f"after reconciliation the structure is {reached}, but the checkpoint's tree is "
                f"{target.digest}; this trainer cannot continue from this checkpoint",
                rolled_back=False,
            )
        self._adopt(state, target)
        if report is not None:
            self._rebase(report)
            notes.append(f"topology reconciled with the checkpoint ({difference.summary()})")
        elif state is not None and state.journal:
            notes.append(f"topology already matches the checkpoint ({len(state.journal)} journaled plan(s))")
        return notes

    # -- the bookkeeping that is the checkpoint's, not the plan's --------
    def _adopt(self, state: MutationState | None, target: Target) -> None:
        """Replace the records the replayed plan wrote with the ones the checkpoint holds."""
        t = self.trainer
        stored = state if state is not None else MutationState(base_digest=t._base_digest, digest=target.digest)
        t.mutation_journal[:] = [copy.deepcopy(dict(entry)) for entry in stored.journal]
        t.topology_epoch = int(stored.topology_epoch)
        t._tombstones.clear()
        t._tombstones.update(stored.tombstones)
        t._node_specs.clear()
        t._node_specs.update({path: dict(spec) for path, spec in stored.specs.items()})
        t._node_meta.clear()
        t._node_meta.update({path: dict(meta) for path, meta in stored.metadata.items()})
        t.engine.restore_pending(stored.pending)
        t._active_view = None

    def _rebase(self, report: Any) -> None:
        """The health baselines of 12.8, once, for the structure the restore just installed."""
        t = self.trainer
        t._loss_baseline = len(t.history) + (1 if t._phase == "observe" else 0)
        t._rate_baseline = len(t._epoch_seconds)
        grown = tuple(dict.fromkeys((*report.added, *report.rebuilt)))
        if grown and t._initial_rss is not None:
            t._initial_rss += t._resident_bytes(grown)
        record = getattr(t.logger, "mutation", None)
        if callable(record):
            record(report, reconciled=True)
