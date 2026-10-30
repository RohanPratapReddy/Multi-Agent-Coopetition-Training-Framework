"""The pure half of tree mutation (document 17 section 12): plans, limits, and the simulator.

No torch import: the CLI, the API and ``conf.py`` validate plans with it, and a test checks
rules of the tree without a model.
"""
from __future__ import annotations

import dataclasses
import fnmatch
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..coopetition.tree import (
    PeerTree, TreeTooLargeError, TreeValidationError, aggregation_groups, validate_path,
)
from ..exceptions import ComplianceError, MutationError

log = logging.getLogger("mactf.mutation")

#: ``init_from`` value meaning "the run's frozen reference policy". The reference is held by the
#: trainer (``trainer.reference``), it is not a tree node, and no operation can address it (M5).
#: A node *named* ``<peer>__reference`` is an ordinary node (distillation may look for one,
#: ``distill/step.py:1743``); mutation treats it like any other.
REFERENCE_TOKEN = "@reference"

#: ``factory`` value meaning "the module this path would have had at build time": the trainer's own
#: ``_module_for(path)`` (a ``models`` entry, the hierarchy level's model source, ``model_factory``
#: or the single model), with the run's finetuning applied. It is what a ``rebuild`` uses to go
#: back to the original architecture, and what the checkpoint reconciliation of 12.9 uses to
#: bring a node back whose architecture was never customised.
BUILD_TOKEN = "@build"

_ROLE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_PEER_KEYS = ("path", "factory", "like", "init_from", "role", "metadata", "optimizer")
_MODIFY_KEYS = ("role", "metadata", "optimizer", "init_from", "rebuild")
_REBUILD_KEYS = ("factory", "like", "init_from")


def _canonical(value: Any, where: str) -> Any:
    """A JSON round-trip, so a spec is plain data that hashes and journals identically."""
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise MutationError("E080", f"{where} must be JSON-serialisable and finite: {error}") from error


def digest_of(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _lower(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key).lower(): value for key, value in raw.items()}


@dataclass(frozen=True)
class Finding:
    """One thing the simulator or the engine found: ``E08x`` is hard, ``W04x`` is not."""

    code: str
    message: str
    path: str | None = None
    hard: bool = True

    def render(self) -> str:
        where = f" [{self.path}]" if self.path else ""
        return f"{self.code}{where}: {self.message}"


@dataclass(frozen=True)
class PeerSpec:
    path: str
    factory: Any = None
    like: str | None = None
    init_from: str | None = None
    role: str = "worker"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    optimizer: Mapping[str, Any] | None = None

    @classmethod
    def parse(cls, raw: "PeerSpec | Mapping[str, Any]", *, path: str | None = None) -> "PeerSpec":
        if isinstance(raw, PeerSpec):
            return raw if path is None or path == raw.path else dataclasses.replace(raw, path=path)
        if not isinstance(raw, Mapping):
            raise MutationError("E080", f"a peer spec is a mapping, got {type(raw).__name__}")
        data = _lower(raw)
        unknown = sorted(set(data) - set(_PEER_KEYS))
        if unknown:
            raise MutationError("E080", f"unknown peer spec keys {unknown}; accepted: {list(_PEER_KEYS)}")
        given = data.get("path", path)
        if path is not None and given != path:
            raise MutationError("E080", f"the spec names {given!r} but the operation names {path!r}")
        if not isinstance(given, str) or not given:
            raise MutationError("E080", "a peer spec needs a non-empty `path`")
        factory = data.get("factory")
        if factory is not None:
            if not isinstance(factory, (str, Mapping)) or not factory:
                raise MutationError(
                    "E080",
                    f"{given}: `factory` is a registered name or a {{type|import, params}} mapping; "
                    "register a callable with @register_peer_factory first",
                )
            if isinstance(factory, str) and factory.startswith("@") and factory != BUILD_TOKEN:
                raise MutationError("E080", f"{given}: unknown factory token {factory!r}; the only token is {BUILD_TOKEN!r}")
            factory = _canonical(factory, f"{given}: factory")
        for key in ("like", "init_from"):
            value = data.get(key)
            if value is not None and (not isinstance(value, str) or not value):
                raise MutationError("E080", f"{given}: `{key}` must be a node path")
            if isinstance(value, str) and value.startswith("@") and not (key == "init_from" and value == REFERENCE_TOKEN):
                raise MutationError(
                    "E080", f"{given}: `{key}` {value!r} is not a node path"
                    + (f"; the only token is {REFERENCE_TOKEN!r}, for `init_from`" if key == "init_from" else "")
                )
        if data.get("like") == REFERENCE_TOKEN:
            raise MutationError("E080", f"{given}: `like` names a node; use `init_from: '@reference'` for its weights")
        role = data.get("role", "worker")
        if not isinstance(role, str) or not _ROLE.match(role):
            raise MutationError("E080", f"{given}: role {role!r} must match {_ROLE.pattern}")
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise MutationError("E080", f"{given}: `metadata` must be a mapping")
        optimizer = data.get("optimizer")
        if optimizer is not None and (not isinstance(optimizer, Mapping) or not optimizer):
            raise MutationError("E080", f"{given}: `optimizer` must be a non-empty OPTIMIZER-style block")
        return cls(
            path=given,
            factory=factory,
            like=data.get("like"),
            init_from=data.get("init_from"),
            role=role,
            metadata=_canonical(dict(metadata), f"{given}: metadata"),
            optimizer=None if optimizer is None else _canonical(dict(optimizer), f"{given}: optimizer"),
        )

    def canonical(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"path": self.path}
        if self.factory is not None:
            payload["factory"] = self.factory
        if self.like is not None:
            payload["like"] = self.like
        if self.init_from is not None:
            payload["init_from"] = self.init_from
        if self.role != "worker":
            payload["role"] = self.role
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        if self.optimizer is not None:
            payload["optimizer"] = dict(self.optimizer)
        return payload

    def digest(self) -> str:
        return digest_of(self.canonical())


class OpKind(str, Enum):
    ADD = "add"
    MODIFY = "modify"
    RETIRE = "retire"
    REVIVE = "revive"
    PURGE = "purge"


@dataclass(frozen=True)
class Op:
    kind: OpKind
    path: str
    spec: PeerSpec | None = None
    changes: Mapping[str, Any] = field(default_factory=dict)
    recursive: bool = False
    force: bool = False
    reason: str = ""

    @classmethod
    def parse(cls, raw: "Op | Mapping[str, Any]") -> "Op":
        if isinstance(raw, Op):
            return raw
        if not isinstance(raw, Mapping):
            raise MutationError("E080", f"an operation is a mapping, got {type(raw).__name__}")
        data = _lower(raw)
        try:
            kind = OpKind(str(data.pop("op", "")).lower())
        except ValueError:
            raise MutationError(
                "E080", f"`op` must be one of {[kind.value for kind in OpKind]}, got {raw.get('op', raw.get('OP'))!r}"
            ) from None
        path = data.pop("path", None)
        if not isinstance(path, str) or not path:
            raise MutationError("E080", f"a {kind.value} operation needs a non-empty `path`")
        reason = str(data.pop("reason", "") or "")
        if kind is OpKind.ADD:
            stray = sorted(set(data) & {"recursive", "force", "set"})
            if stray:
                raise MutationError("E080", f"add {path!r}: {stray} do not apply to `add`")
            return cls(kind, path, spec=PeerSpec.parse(data, path=path), reason=reason)
        if kind is OpKind.MODIFY:
            changes = data.pop("set", None)
            if data:
                raise MutationError("E080", f"modify {path!r}: unknown keys {sorted(data)}; put the changes under `set`")
            return cls(kind, path, changes=cls._modify_changes(path, changes), reason=reason)
        recursive = bool(data.pop("recursive", False))
        force = bool(data.pop("force", False))
        if force and kind is not OpKind.PURGE:
            raise MutationError("E080", f"{kind.value} {path!r}: `force` applies to `purge` only")
        if data:
            raise MutationError("E080", f"{kind.value} {path!r}: unknown keys {sorted(data)}")
        return cls(kind, path, recursive=recursive, force=force, reason=reason)

    @staticmethod
    def _modify_changes(path: str, changes: Any) -> dict[str, Any]:
        if not isinstance(changes, Mapping) or not changes:
            raise MutationError("E080", f"modify {path!r} needs a non-empty `set` mapping")
        data = {str(key).lower(): value for key, value in changes.items()}
        unknown = sorted(set(data) - set(_MODIFY_KEYS))
        if unknown:
            raise MutationError("E080", f"modify {path!r}: cannot set {unknown}; accepted: {list(_MODIFY_KEYS)}")
        clean: dict[str, Any] = {}
        if "role" in data:
            if not isinstance(data["role"], str) or not _ROLE.match(data["role"]):
                raise MutationError("E080", f"modify {path!r}: role {data['role']!r} must match {_ROLE.pattern}")
            clean["role"] = data["role"]
        if "metadata" in data:
            if not isinstance(data["metadata"], Mapping):
                raise MutationError("E080", f"modify {path!r}: `metadata` must be a mapping")
            clean["metadata"] = _canonical(dict(data["metadata"]), f"modify {path}: metadata")
        if "optimizer" in data:
            block = data["optimizer"]
            if block is not None and (not isinstance(block, Mapping) or not block):
                raise MutationError("E080", f"modify {path!r}: `optimizer` is a block, or null to remove the override")
            clean["optimizer"] = None if block is None else _canonical(dict(block), f"modify {path}: optimizer")
        if "init_from" in data:
            value = data["init_from"]
            if not isinstance(value, str) or not value or (value.startswith("@") and value != REFERENCE_TOKEN):
                raise MutationError("E080", f"modify {path!r}: `init_from` must be a node path or '@reference'")
            clean["init_from"] = data["init_from"]
        if "rebuild" in data:
            rebuild = data["rebuild"]
            if not isinstance(rebuild, Mapping) or not rebuild:
                raise MutationError("E080", f"modify {path!r}: `rebuild` is a mapping of factory/like/init_from")
            odd = sorted(set(str(key).lower() for key in rebuild) - set(_REBUILD_KEYS))
            if odd:
                raise MutationError("E080", f"modify {path!r}: `rebuild` accepts {list(_REBUILD_KEYS)}, not {odd}")
            clean["rebuild"] = PeerSpec.parse({**dict(rebuild), "path": path}).canonical()
        if "init_from" in clean and "rebuild" in clean:
            raise MutationError("E080", f"modify {path!r}: `init_from` and `rebuild` both set weights; give one")
        return clean

    def canonical(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"op": self.kind.value, "path": self.path}
        if self.kind is OpKind.ADD:
            payload.update({k: v for k, v in self.spec.canonical().items() if k != "path"})
        elif self.kind is OpKind.MODIFY:
            payload["set"] = dict(self.changes)
        else:
            if self.recursive:
                payload["recursive"] = True
            if self.force:
                payload["force"] = True
        if self.reason:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True)
class MutationPlan:
    ops: tuple[Op, ...]
    label: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.ops:
            raise MutationError("E080", "a mutation plan needs at least one operation")

    @classmethod
    def of(cls, *ops: "Op | Mapping[str, Any]", label: str = "", reason: str = "") -> "MutationPlan":
        return cls(tuple(Op.parse(op) for op in ops), label=str(label), reason=str(reason))

    @classmethod
    def parse(cls, raw: "MutationPlan | Sequence[Any] | Mapping[str, Any]") -> "MutationPlan":
        """A plan, a list of operations, or ``{label, reason, ops: [...]}`` (keys in any case)."""
        if isinstance(raw, MutationPlan):
            return raw
        if isinstance(raw, Mapping):
            data = _lower(raw)
            unknown = sorted(set(data) - {"label", "reason", "ops"})
            if unknown:
                raise MutationError("E080", f"unknown plan keys {unknown}; accepted: label, reason, ops")
            items = data.get("ops")
            label, reason = str(data.get("label", "") or ""), str(data.get("reason", "") or "")
        else:
            items, label, reason = raw, "", ""
        if isinstance(items, (str, bytes)) or not isinstance(items, Iterable):
            raise MutationError("E080", "`ops` must be a list of operations")
        return cls.of(*list(items), label=label, reason=reason)

    def canonical(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ops": [op.canonical() for op in self.ops]}
        if self.label:
            payload["label"] = self.label
        if self.reason:
            payload["reason"] = self.reason
        return payload

    def digest(self) -> str:
        return digest_of(self.canonical())

    def paths(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(op.path for op in self.ops))

    def apply(self, trainer: Any, **kwargs: Any) -> Any:
        """Commit on ``trainer`` (``trainer.mutate``); returns a ``MutationReport``."""
        return trainer.mutate(self, **kwargs)

    def dry_run(self, trainer: Any, **kwargs: Any) -> Any:
        """Simulate and stage on ``trainer`` and discard: every check, no effect."""
        return trainer.mutate(self, dry_run=True, **kwargs)


def add_peer(path: str, **spec: Any) -> Op:
    return Op.parse({"op": "add", "path": path, **spec})


def modify_peer(path: str, **changes: Any) -> Op:
    return Op.parse({"op": "modify", "path": path, "set": changes})


def retire_peer(path: str, *, recursive: bool = False, reason: str = "") -> Op:
    return Op.parse({"op": "retire", "path": path, "recursive": recursive, "reason": reason})


def revive_peer(path: str, *, recursive: bool = False, reason: str = "") -> Op:
    return Op.parse({"op": "revive", "path": path, "recursive": recursive, "reason": reason})


def purge_peer(path: str, *, recursive: bool = False, force: bool = False, reason: str = "") -> Op:
    return Op.parse({"op": "purge", "path": path, "recursive": recursive, "force": force, "reason": reason})


_LIMIT_KEYS = (
    "ENABLED", "MIN_ACTIVE_ROOTS", "MAX_OPS_PER_PLAN", "MAX_PLANS_PER_EPOCH", "PROTECTED",
    "ALLOW_PURGE", "TOMBSTONE", "WATCHDOG_PRUNE", "JOURNAL_WARN", "STRICT_REPLAY", "SCHEDULE",
)
TOMBSTONE_MODES = ("run", "off")
PRUNE_MODES = ("purge", "retire", "off")


def _positive_int(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MutationError("E080", f"TRAINER.MUTATION.{key} must be an integer >= 1, got {value!r}")
    return value


def _flag(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise MutationError("E080", f"TRAINER.MUTATION.{key} must be true or false, got {value!r}")
    return value


@dataclass(frozen=True)
class MutationLimits:
    """The ``TRAINER.MUTATION`` block, validated."""

    enabled: bool = False
    min_active_roots: int = 1
    max_ops_per_plan: int = 64
    max_plans_per_epoch: int = 4
    protected: tuple[str, ...] = ()
    allow_purge: bool = True
    tombstone: str = "run"
    watchdog_prune: str = "purge"
    journal_warn: int = 1000
    strict_replay: bool = True
    schedule: tuple[tuple[int, MutationPlan], ...] = ()

    @classmethod
    def validate_block(cls, block: Mapping[str, Any] | None) -> dict[str, Any]:
        """Static checks (used by ``conf.py`` at load time, with no tree): returns the cleaned block."""
        data = {str(key).upper(): value for key, value in dict(block or {}).items()}
        unknown = sorted(set(data) - set(_LIMIT_KEYS))
        if unknown:
            raise MutationError("E080", f"unknown TRAINER.MUTATION keys {unknown}; accepted: {list(_LIMIT_KEYS)}")
        clean: dict[str, Any] = {}
        for key in ("ENABLED", "ALLOW_PURGE", "STRICT_REPLAY"):
            if key in data:
                clean[key] = _flag(data[key], key)
        for key in ("MIN_ACTIVE_ROOTS", "MAX_OPS_PER_PLAN", "MAX_PLANS_PER_EPOCH", "JOURNAL_WARN"):
            if key in data and data[key] is not None:
                clean[key] = _positive_int(data[key], key)
        if "PROTECTED" in data:
            globs = data["PROTECTED"] or []
            if isinstance(globs, str) or not all(isinstance(item, str) and item for item in globs):
                raise MutationError("E080", "TRAINER.MUTATION.PROTECTED must be a list of path globs")
            clean["PROTECTED"] = list(globs)
        if "TOMBSTONE" in data:
            if data["TOMBSTONE"] not in TOMBSTONE_MODES:
                raise MutationError("E080", f"TRAINER.MUTATION.TOMBSTONE must be one of {list(TOMBSTONE_MODES)}")
            clean["TOMBSTONE"] = data["TOMBSTONE"]
        if "WATCHDOG_PRUNE" in data:
            if data["WATCHDOG_PRUNE"] not in PRUNE_MODES:
                raise MutationError("E080", f"TRAINER.MUTATION.WATCHDOG_PRUNE must be one of {list(PRUNE_MODES)}")
            clean["WATCHDOG_PRUNE"] = data["WATCHDOG_PRUNE"]
        if "SCHEDULE" in data:
            entries = data["SCHEDULE"] or []
            if isinstance(entries, (str, Mapping)):
                raise MutationError("E080", "TRAINER.MUTATION.SCHEDULE must be a list of {AT_EPOCH, OPS} entries")
            rows = []
            for index, entry in enumerate(entries):
                if not isinstance(entry, Mapping):
                    raise MutationError("E080", f"TRAINER.MUTATION.SCHEDULE[{index}] must be a mapping")
                item = {str(key).upper(): value for key, value in entry.items()}
                odd = sorted(set(item) - {"AT_EPOCH", "LABEL", "REASON", "OPS"})
                if odd:
                    raise MutationError("E080", f"TRAINER.MUTATION.SCHEDULE[{index}]: unknown keys {odd}")
                at = item.get("AT_EPOCH")
                if isinstance(at, bool) or not isinstance(at, int) or at < 0:
                    raise MutationError("E080", f"TRAINER.MUTATION.SCHEDULE[{index}].AT_EPOCH must be an integer >= 0")
                plan = MutationPlan.parse(
                    {"label": item.get("LABEL") or f"schedule:{index}", "reason": item.get("REASON", ""), "ops": item.get("OPS")}
                )
                rows.append({"AT_EPOCH": at, "plan": plan.canonical()})
            clean["SCHEDULE"] = rows
        return clean

    @classmethod
    def from_settings(
        cls,
        block: Mapping[str, Any] | None,
        *,
        initial_roots: int,
        initial_nodes: int | None = None,
        growth_ratio: float | None = None,
    ) -> "MutationLimits":
        """The validated limits. ``initial_nodes`` and ``growth_ratio`` (``WATCHDOG.MAX_TREE_GROWTH_RATIO``)
        are optional; with both, a schedule that adds enough nodes to trip ``TreeGrowthProbe`` is ``W044``."""
        clean = cls.validate_block(block)
        roots = max(1, int(initial_roots))
        minimum = clean.get("MIN_ACTIVE_ROOTS", min(2, roots))
        if minimum > roots:
            raise MutationError(
                "E080", f"TRAINER.MUTATION.MIN_ACTIVE_ROOTS={minimum} exceeds the {roots} roots this run starts with"
            )
        schedule = tuple(
            (int(row["AT_EPOCH"]), MutationPlan.parse(row["plan"])) for row in clean.get("SCHEDULE", [])
        )
        if schedule and initial_nodes and growth_ratio:
            grown = sum(1 for _at, plan in schedule for op in plan.ops if op.kind is OpKind.ADD)
            if (int(initial_nodes) + grown) / int(initial_nodes) > float(growth_ratio):
                log.warning(
                    "W044: TRAINER.MUTATION.SCHEDULE adds %d node(s) to a tree of %d; TreeGrowthProbe "
                    "will report growth beyond WATCHDOG.MAX_TREE_GROWTH_RATIO=%s",
                    grown, int(initial_nodes), growth_ratio,
                )
        return cls(
            enabled=clean.get("ENABLED", False),
            min_active_roots=minimum,
            max_ops_per_plan=clean.get("MAX_OPS_PER_PLAN", 64),
            max_plans_per_epoch=clean.get("MAX_PLANS_PER_EPOCH", 4),
            protected=tuple(clean.get("PROTECTED", ())),
            allow_purge=clean.get("ALLOW_PURGE", True),
            tombstone=clean.get("TOMBSTONE", "run"),
            watchdog_prune=clean.get("WATCHDOG_PRUNE", "purge"),
            journal_warn=clean.get("JOURNAL_WARN", 1000),
            strict_replay=clean.get("STRICT_REPLAY", True),
            schedule=schedule,
        )

    def is_protected(self, path: str) -> bool:
        return any(path == glob or fnmatch.fnmatchcase(path, glob) for glob in self.protected)


ROOT = "root"          # == components.weight_plan.ROOT_LABEL (a test pins the equality)


def label_of(parent: str | None) -> str:
    return ROOT if parent is None else parent


def parent_of(path: str) -> str | None:
    head, _, _ = path.rpartition("/")
    return head or None


def structure_digest(tree: PeerTree, retired: Iterable[str]) -> str:
    """What a replay must reproduce: which nodes exist, their roles, and which are retired."""
    gone = set(retired)
    rows = [[node.path, node.role, node.path in gone] for node in tree.walk("root-first")]
    return digest_of(rows)


@dataclass(frozen=True)
class GroupChange:
    """One aggregation group whose members, or whose active members, differ after a plan."""

    label: str
    parent: str | None
    before: tuple[str, ...]
    after: tuple[str, ...]
    active_before: tuple[str, ...]
    active_after: tuple[str, ...]

    @property
    def appeared(self) -> bool:
        return not self.before and bool(self.after)

    @property
    def vanished(self) -> bool:
        return bool(self.before) and not self.after


@dataclass(frozen=True)
class Simulation:
    """The tree a plan would produce, and everything the engine needs to build it."""

    tree: PeerTree                       # the final structure; peers of existing nodes are shared
    retired: frozenset[str]
    tombstones: frozenset[str]
    added: tuple[str, ...]
    purged: tuple[str, ...]
    retired_now: tuple[str, ...]
    revived_now: tuple[str, ...]
    modified: tuple[str, ...]
    changes: tuple[GroupChange, ...]
    findings: tuple[Finding, ...]
    before_digest: str
    after_digest: str

    @property
    def hard(self) -> tuple[Finding, ...]:
        return tuple(item for item in self.findings if item.hard)

    @property
    def soft(self) -> tuple[Finding, ...]:
        return tuple(item for item in self.findings if not item.hard)

    @property
    def ok(self) -> bool:
        return not self.hard

    def raise_for_findings(self) -> None:
        """``MutationError`` carrying the first hard finding's code and every finding's text."""
        hard = self.hard
        if hard:
            raise MutationError(
                hard[0].code, hard[0].message, findings=[item.render() for item in self.findings]
            )

    def changed_labels(self) -> tuple[str, ...]:
        return tuple(change.label for change in self.changes)

    def active_paths(self) -> tuple[str, ...]:
        return tuple(node.path for node in self.tree.walk("root-first") if node.path not in self.retired)


@dataclass(frozen=True)
class MutationReport:
    """What a plan did (or, for a dry run, would do)."""

    plan_digest: str
    label: str
    reason: str
    source: str
    dry_run: bool
    epoch: int
    topology_epoch: int
    added: tuple[str, ...]
    retired: tuple[str, ...]
    revived: tuple[str, ...]
    purged: tuple[str, ...]
    modified: tuple[str, ...]
    rebuilt: tuple[str, ...]
    optimizers_rebuilt: tuple[str, ...]
    groups: tuple[str, ...]
    active_nodes: int
    retired_nodes: int
    before_digest: str
    after_digest: str
    warnings: tuple[str, ...]
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {key: (list(value) if isinstance(value, tuple) else value) for key, value in vars(self).items()}

    @property
    def changed(self) -> bool:
        return bool(self.added or self.retired or self.revived or self.purged or self.modified)


class _Work:
    """The mutable state of one simulation."""

    def __init__(self, tree: PeerTree, retired: set[str], tombstones: set[str]) -> None:
        self.tree = tree
        self.retired = retired
        self.tomb = tombstones
        self.added: list[str] = []
        self.purged: list[str] = []
        self.retired_now: list[str] = []
        self.revived_now: list[str] = []
        self.modified: list[str] = []
        self.findings: list[Finding] = []

    @property
    def failed(self) -> bool:
        return any(item.hard for item in self.findings)

    def fail(self, code: str, message: str, path: str | None = None) -> None:
        self.findings.append(Finding(code, message, path, hard=True))

    def warn(self, code: str, message: str, path: str | None = None) -> None:
        self.findings.append(Finding(code, message, path, hard=False))

    def drop(self, path: str) -> None:
        for bucket in (self.modified, self.retired_now, self.revived_now):
            if path in bucket:
                bucket.remove(path)


def _groups(tree: PeerTree, retired: set[str] | frozenset[str]) -> dict[str, tuple[str | None, tuple[str, ...], tuple[str, ...]]]:
    """``label -> (parent, members, active members)`` in ``aggregation_groups`` order."""
    result: dict[str, tuple[str | None, tuple[str, ...], tuple[str, ...]]] = {}
    for parent, members in aggregation_groups(tree):
        result[label_of(parent)] = (
            parent, tuple(members), tuple(name for name in members if name not in retired)
        )
    return result


class TopologySimulator:
    """Apply a plan to a clone and judge the result. Never raises for a bad plan: it reports."""

    def __init__(
        self,
        tree: PeerTree,
        *,
        retired: Iterable[str],
        tombstones: Iterable[str],
        limits: MutationLimits,
        feasible: Callable[[int], None],
    ) -> None:
        self.tree = tree
        self.retired = frozenset(retired)
        self.tombstones = frozenset(tombstones)
        self.limits = limits
        #: ``feasible(n)`` raises ``ComplianceError`` when ``WEIGHT_FLOOR``/``WEIGHT_CAP`` admit no
        #: weight vector for ``n`` active members: ``feasible_bounds`` with the run's bounds.
        self.feasible = feasible

    # -- entry point -----------------------------------------------------
    def run(self, plan: MutationPlan) -> Simulation:
        work = _Work(self.tree.clone(), set(self.retired), set(self.tombstones))
        before_groups = _groups(self.tree, self.retired)
        if len(plan.ops) > self.limits.max_ops_per_plan:
            work.fail(
                "E084",
                f"the plan has {len(plan.ops)} operations; TRAINER.MUTATION.MAX_OPS_PER_PLAN is "
                f"{self.limits.max_ops_per_plan}",
            )
        for op in plan.ops:
            if work.failed:
                break      # later operations are defined only on the tree the failed one would have made
            getattr(self, f"_{op.kind.value}")(op, work)
        changes: tuple[GroupChange, ...] = ()
        if not work.failed:
            changes = self._finalize(work, before_groups)
        return Simulation(
            tree=work.tree,
            retired=frozenset(work.retired),
            tombstones=frozenset(work.tomb),
            added=tuple(work.added),
            purged=tuple(work.purged),
            retired_now=tuple(work.retired_now),
            revived_now=tuple(work.revived_now),
            modified=tuple(work.modified),
            changes=changes,
            findings=tuple(work.findings),
            before_digest=structure_digest(self.tree, self.retired),
            after_digest=structure_digest(work.tree, work.retired),
        )

    # -- operations ------------------------------------------------------
    def _check_reference(self, work: _Work, path: str, field_name: str, ref: str | None) -> bool:
        if ref is None or ref == REFERENCE_TOKEN:
            return True
        if ref not in work.tree:
            work.fail("E083", f"`{field_name}` names {ref!r}, which is not a node of the tree", path)
            return False
        if ref == path:
            work.fail("E083", f"`{field_name}` names the node itself", path)
            return False
        return True

    def _add(self, op: Op, work: _Work) -> None:
        spec = op.spec
        assert spec is not None
        path = spec.path
        try:
            validate_path(path)
        except TreeValidationError as error:
            return work.fail("E081", str(error), path)
        if path in work.tree:
            return work.fail("E081", "the node already exists", path)
        if path in work.tomb:
            return work.fail(
                "E081",
                "the path was purged earlier in this run and cannot be reused "
                "(TRAINER.MUTATION.TOMBSTONE); its warning and escalation history must stay attached "
                "to one peer",
                path,
            )
        parent = parent_of(path)
        if parent is not None:
            if parent not in work.tree:
                return work.fail("E081", f"the parent {parent!r} does not exist", path)
            if parent in work.retired:
                return work.fail("E081", f"the parent {parent!r} is retired; revive it first", path)
        for name, ref in (("like", spec.like), ("init_from", spec.init_from)):
            if not self._check_reference(work, path, name, ref):
                return None
        try:
            work.tree.add_node(path, role=spec.role, metadata=dict(spec.metadata))
        except TreeTooLargeError as error:
            return work.fail("E084", str(error), path)
        except TreeValidationError as error:
            return work.fail("E081", str(error), path)
        work.added.append(path)
        return None

    def _modify(self, op: Op, work: _Work) -> None:
        path, changes = op.path, op.changes
        if path not in work.tree:
            return work.fail("E081", "no such node", path)
        if path in work.retired:
            return work.fail("E081", "the node is retired; revive it first", path)
        rebuild = changes.get("rebuild") or {}
        for name, ref in (
            ("init_from", changes.get("init_from")),
            ("rebuild.like", rebuild.get("like")),
            ("rebuild.init_from", rebuild.get("init_from")),
        ):
            if not self._check_reference(work, path, name, ref):
                return None
        if "rebuild" in changes and self.limits.is_protected(path):
            return work.fail("E082", "the node matches TRAINER.MUTATION.PROTECTED and cannot be rebuilt", path)
        node = work.tree[path]
        if "role" in changes:
            node.role = changes["role"]
        if "metadata" in changes:
            merged = dict(node.metadata)
            for key, value in changes["metadata"].items():
                if value is None:
                    merged.pop(key, None)
                else:
                    merged[key] = value
            node.metadata = merged
        if path not in work.modified:
            work.modified.append(path)
        return None

    def _targets(self, work: _Work, path: str, *, recursive: bool) -> list[str]:
        return [path, *work.tree.descendants(path)] if recursive else [path]

    def _retire(self, op: Op, work: _Work) -> None:
        path = op.path
        if path not in work.tree:
            return work.fail("E081", "no such node", path)
        targets = self._targets(work, path, recursive=op.recursive)
        for target in targets:
            if self.limits.is_protected(target):
                return work.fail("E082", "the node matches TRAINER.MUTATION.PROTECTED and cannot be retired", target)
        if not op.recursive:
            live = [child for child in work.tree[path].children if child not in work.retired]
            if live:
                return work.fail(
                    "E082",
                    f"{len(live)} child(ren) are still active ({live[:4]}); retire the branch with "
                    "`recursive: true`, or retire the children first",
                    path,
                )
        fresh = [target for target in targets if target not in work.retired]
        if not fresh:
            return work.warn("W040", "already retired; nothing to do", path)
        for target in fresh:
            work.retired.add(target)
            work.tree.mark(target, "pruned")
            if target in work.revived_now:
                work.revived_now.remove(target)      # revived earlier in this plan: back where it began
            else:
                work.retired_now.append(target)
        return None

    def _revive(self, op: Op, work: _Work) -> None:
        path = op.path
        if path not in work.tree:
            return work.fail("E081", "no such node", path)
        targets = self._targets(work, path, recursive=op.recursive)
        fresh = [target for target in targets if target in work.retired]
        if not fresh:
            return work.warn("W041", "not retired; nothing to do", path)
        for target in fresh:
            parent = work.tree[target].parent
            if parent is not None and parent in work.retired and parent not in fresh:
                return work.fail(
                    "E081", f"the parent {parent!r} is retired; revive it too (`recursive: true`) or first", target
                )
        for target in fresh:
            work.retired.discard(target)
            work.tree.mark(target, "active")
            if target in work.retired_now:
                work.retired_now.remove(target)      # retired earlier in this plan: back where it began
            else:
                work.revived_now.append(target)
        return None

    def _purge(self, op: Op, work: _Work) -> None:
        path = op.path
        if not self.limits.allow_purge:
            return work.fail("E082", "TRAINER.MUTATION.ALLOW_PURGE is false; retire the node instead", path)
        if path not in work.tree:
            return work.fail("E081", "no such node", path)
        targets = self._targets(work, path, recursive=True)
        if len(targets) > 1 and not op.recursive:
            return work.fail("E082", f"the node has {len(targets) - 1} descendant(s); pass `recursive: true`", path)
        for target in targets:
            if self.limits.is_protected(target):
                return work.fail("E082", "the node matches TRAINER.MUTATION.PROTECTED and cannot be purged", target)
        live = [target for target in targets if target not in work.retired]
        if live and not op.force:
            return work.fail(
                "E082",
                f"{len(live)} node(s) are active ({live[:4]}); retire first, or pass `force: true` "
                "to delete without the retire step",
                path,
            )
        work.tree.remove_node(path, recursive=True)
        for target in targets:
            work.retired.discard(target)
            work.drop(target)
            if target in work.added:
                work.added.remove(target)       # never existed outside this plan: no tombstone
                continue
            work.purged.append(target)
            if self.limits.tombstone == "run":
                work.tomb.add(target)
        return None

    # -- the checks that need the final tree -----------------------------
    def _finalize(
        self, work: _Work, before: dict[str, tuple[str | None, tuple[str, ...], tuple[str, ...]]]
    ) -> tuple[GroupChange, ...]:
        tree, retired = work.tree, work.retired
        roots = [path for path in tree.root_paths if path not in retired]
        if len(roots) < self.limits.min_active_roots:
            work.fail(
                "E082",
                f"{len(roots)} active root(s) would remain; TRAINER.MUTATION.MIN_ACTIVE_ROOTS is "
                f"{self.limits.min_active_roots}",
            )
            return ()
        for node in tree.walk("root-first"):
            if node.path in retired:
                continue
            if node.parent is not None and node.parent in retired:
                work.fail(
                    "E082",
                    f"an active node sits under the retired {node.parent!r}; retire it too, or revive the parent",
                    node.path,
                )
                return ()
            if node.children and all(child in retired for child in node.children):
                work.fail(
                    "E082",
                    "every child is retired, which leaves a branch that cannot be averaged; retire the "
                    "branch itself (`recursive: true`) instead",
                    node.path,
                )
                return ()
        after = _groups(tree, retired)
        changes: list[GroupChange] = []
        for label, (parent, members, active) in after.items():
            old = before.get(label)
            if old is not None and old[1] == members and old[2] == active:
                continue
            old_members, old_active = (old[1], old[2]) if old is not None else ((), ())
            changes.append(GroupChange(label, parent, old_members, members, old_active, active))
        for label, (parent, members, active) in before.items():
            if label not in after:
                changes.append(GroupChange(label, parent, members, (), active, ()))
        for change in changes:
            if change.vanished or (change.parent is not None and change.parent in retired):
                continue
            try:
                self.feasible(len(change.active_after))
            except ComplianceError as error:
                work.fail(
                    "E086",
                    f"group {change.label!r} would have {len(change.active_after)} active member(s): {error}",
                    change.label,
                )
                return ()
        return tuple(changes)


INTERNAL_SOURCES = ("watchdog", "replay")


def manifest_spec(spec: PeerSpec) -> dict[str, Any]:
    """The part of a ``PeerSpec`` that decides what a node *is*.

    ``init_from`` is dropped: it says where the first weights came from, and a reconciliation
    never needs it because the weights come from the checkpoint's ``models`` section.
    """
    payload = spec.canonical()
    payload.pop("init_from", None)
    return payload


def tree_state_digest(tree_state: Mapping[str, Any]) -> str:
    """``structure_digest`` of a stored tree section (``PeerTree.state_dict``): ``pruned`` is retired."""
    try:
        tree = PeerTree.from_dict(tree_state)
    except Exception as error:  # noqa: BLE001 - a tree that does not parse is a corrupt checkpoint
        raise MutationError("E089", f"the checkpoint's tree section does not parse: {error}") from error
    retired = [node.path for node in tree.walk("root-first") if node.status == "pruned"]
    return structure_digest(tree, retired)
