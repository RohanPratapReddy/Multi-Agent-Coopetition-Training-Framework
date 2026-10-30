"""The ``PathState`` registry: every container a node path or group label keys (document 17 section 12.6.2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from ..components.weight_plan import ROOT_LABEL


@dataclass(frozen=True)
class Container:
    """One piece of per-path (or per-group) trainer state."""

    name: str
    keyed_by: str                 # "path" or "group"
    get: Callable[[Any], dict]    # the live dict
    on_purge: str                 # "drop" or "keep"
    on_retire: str                # "keep" always today; a column so a future container must say


def _by_attr(attribute: str) -> Callable[[Any], dict]:
    return lambda trainer: getattr(trainer, attribute)


CONTAINERS: tuple[Container, ...] = (
    Container("peers", "path", _by_attr("peers"), "drop", "keep"),
    Container("optimizers", "path", _by_attr("optimizers"), "drop", "keep"),
    Container("signal_by_peer", "path", _by_attr("signal_by_peer"), "drop", "keep"),
    Container("_last_weights", "path", _by_attr("_last_weights"), "drop", "keep"),
    Container("_previous_scores", "path", _by_attr("_previous_scores"), "drop", "keep"),
    Container("_dead_counts", "path", _by_attr("_dead_counts"), "drop", "keep"),
    Container("_epoch_node_metrics", "path", _by_attr("_epoch_node_metrics"), "drop", "keep"),
    Container("controllers", "group", _by_attr("controllers"), "drop", "keep"),
    Container("_group_aggregators", "group", _by_attr("_group_aggregators"), "drop", "keep"),
)


class PathState:
    """Cleanup and audit over every container a node path or group label keys."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

    # -- cleanup ---------------------------------------------------------
    def forget(
        self,
        *,
        paths: Iterable[str],
        labels: Iterable[str],
        undo: Any,
        dropped_paths: Sequence[str],
    ) -> None:
        """Drop what a commit made stale, registering every inverse on ``undo``.

        ``dropped_paths`` are purged nodes: every per-path container loses them, and so does the
        function book and the freeze set. ``paths`` are nodes whose peer was replaced or
        re-seeded: the function book's cached instances for them are rebuilt lazily. ``labels``
        are the groups whose membership changed: the plans' per-group caches go.
        """
        t = self.trainer
        gone = list(dict.fromkeys(dropped_paths))
        for container in CONTAINERS:
            if container.keyed_by != "path" or container.on_purge != "drop":
                continue
            live = container.get(t)
            if container.name in ("peers", "optimizers"):
                continue                       # the commit already removed them, with their own undo
            undo.restore_dict(container.name, live)
            for path in gone:
                live.pop(path, None)
        for path in dict.fromkeys([*paths, *gone]):
            undo.push(f"functions:{path}", t.functions.forget(path))
        changed = list(dict.fromkeys(labels))
        undo.push("weight_plan.groups", t.weight_plan.forget_groups(changed))
        undo.push("strategy_plan.groups", t.strategy_plan.forget_groups(changed))

    # -- audit (M9) ------------------------------------------------------
    def audit(self) -> list[str]:
        """Every way the trainer can disagree with its tree. Empty means consistent."""
        t = self.trainer
        problems: list[str] = []
        tree = t.tree
        known = {node.path for node in tree.walk("root-first")}
        retired = t.freeze.retired()
        for container in CONTAINERS:
            live = container.get(t)
            if container.keyed_by == "path":
                stale = sorted(set(live) - known)
                if stale:
                    problems.append(f"{container.name} holds paths with no node: {stale[:4]}")
        if set(t.peers) != known:
            problems.append(
                f"peers and tree differ: missing={sorted(known - set(t.peers))[:4]} extra={sorted(set(t.peers) - known)[:4]}"
            )
        for node in tree.walk("root-first"):
            if node.peer is not t.peers.get(node.path):
                problems.append(f"{node.path}: the tree node and trainer.peers hold different peers")
            if node.path in retired:
                if node.status != "pruned":
                    problems.append(f"{node.path}: retired but status is {node.status!r}")
                continue
            if node.path not in t.optimizers and "optimizer:null" not in t.freeze.reasons(node.path):
                problems.append(f"{node.path}: active with no optimizer")
        labels = {
            ROOT_LABEL if parent is None else parent for parent, members in _group_pairs(tree)
            if any(member not in retired for member in members)
        }
        for key in t.controllers:
            if key != "root" and key not in known:
                problems.append(f"controllers holds group {key!r} with no node")
        for label in labels:
            key = "root" if label == ROOT_LABEL else label
            if key not in t.controllers and (label != ROOT_LABEL or t.controller is None):
                problems.append(f"group {label!r} has no controller")
        for label in t._group_aggregators:
            if label not in known:
                problems.append(f"_group_aggregators holds group {label!r} with no node")
        return problems

    def describe(self) -> list[dict[str, str]]:
        """What ``mactf trainer state-map`` prints: one row per container."""
        return [
            {"container": item.name, "keyed_by": item.keyed_by, "on_purge": item.on_purge, "on_retire": item.on_retire}
            for item in CONTAINERS
        ]


def _group_pairs(tree: Any) -> list[tuple[str | None, tuple[str, ...]]]:
    from ..coopetition.tree import aggregation_groups

    return aggregation_groups(tree)
