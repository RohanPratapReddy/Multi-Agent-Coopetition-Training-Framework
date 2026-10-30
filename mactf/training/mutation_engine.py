"""The mutation engine: simulate, stage, commit with an undo stack (document 17 section 12.5)."""
from __future__ import annotations

import copy
import dataclasses
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import torch

from ..components.resolver import ComponentSpec
from ..coopetition.aggregation import feasible_bounds
from ..coopetition.peer import Peer
from ..coopetition.strategy import group_layouts
from ..coopetition.tree import PeerTree, node_seed
from ..exceptions import ComplianceError, ContractError, ImproperlyConfigured, MutationError
from .freeze import RETIRED
from .mutation import (
    BUILD_TOKEN, INTERNAL_SOURCES, REFERENCE_TOKEN, MutationLimits, MutationPlan, MutationReport, OpKind,
    PeerSpec, Simulation, TopologySimulator, _groups, manifest_spec,
)

log = logging.getLogger("mactf.mutation")

PHASES_ALLOWED = ("idle", "observe")
# ``INTERNAL_SOURCES = ("watchdog", "replay")`` (the sources that are not the user asking: the
# watchdog's prune of 12.8 and the checkpoint reconciliation of 12.9) lives in ``mutation.py``
# with the other pure constants, because ``mutation_reconcile.py`` needs it without torch.


class Undo:
    """A stack of inverses. ``push`` before you mutate; ``rollback`` runs them newest first."""

    def __init__(self) -> None:
        self._steps: list[tuple[str, Callable[[], None]]] = []

    def push(self, label: str, inverse: Callable[[], None]) -> None:
        self._steps.append((label, inverse))

    def restore_dict(self, label: str, container: dict[Any, Any]) -> None:
        """Snapshot a dict (keeping its order) and register the restore."""
        saved = dict(container)

        def inverse() -> None:
            container.clear()
            container.update(saved)

        self.push(label, inverse)

    def restore_set(self, label: str, container: set[Any]) -> None:
        saved = set(container)

        def inverse() -> None:
            container.clear()
            container.update(saved)

        self.push(label, inverse)

    def restore_list(self, label: str, container: list[Any]) -> None:
        saved = list(container)

        def inverse() -> None:
            container[:] = saved

        self.push(label, inverse)

    def restore_attr(self, label: str, owner: Any, name: str) -> None:
        saved = getattr(owner, name)
        self.push(label, lambda: setattr(owner, name, saved))

    def rollback(self) -> list[str]:
        """Run every inverse. Returns the failures (empty when the rollback was complete)."""
        failures: list[str] = []
        while self._steps:
            label, inverse = self._steps.pop()
            try:
                inverse()
            except Exception as error:  # noqa: BLE001 - a failed inverse is reported, never raised
                failures.append(f"{label}: {type(error).__name__}: {error}")
        return failures


@dataclass
class Staged:
    """Everything a plan needs, built and checked, installed nowhere."""

    peers: dict[str, Peer] = field(default_factory=dict)          # new, and rebuilt, by path
    rebuilt: list[str] = field(default_factory=list)              # existing nodes whose peer is replaced
    reweighted: dict[str, torch.Tensor] = field(default_factory=dict)   # init_from on a live peer
    optimizer: Any = None                                         # StagedTopology
    notes: list[str] = field(default_factory=list)


class MutationEngine:
    """Owns the rules of changing a built trainer's peer tree."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer
        self._lock = threading.RLock()
        self._queue: list[tuple[MutationPlan, str]] = []
        #: Set when a rollback did not complete (``E089``, ``rolled_back=False``). The trainer's state
        #: is then unreliable: every later plan and the next ``run_epoch`` raise this error, so a
        #: run stops instead of continuing half-mutated (R9).
        self.poisoned: MutationError | None = None

    # -- gate ------------------------------------------------------------
    def _gate(self, source: str) -> None:
        t = self.trainer
        if self.poisoned is not None:
            raise self.poisoned
        if not t.built:
            raise MutationError("E088", "the trainer is not built; call build() or fit() first")
        internal = source in INTERNAL_SOURCES
        if not t.mutation_limits.enabled and not internal:
            raise MutationError(
                "E088",
                "tree mutation is off; set TRAINER.MUTATION.ENABLED: true (a run that never "
                "mutates keeps the topology it was built with, and the checkpoint it always wrote)",
            )
        if t._phase not in PHASES_ALLOWED:
            raise MutationError(
                "E088",
                f"a plan cannot run while the trainer is in phase {t._phase!r}; mutate between epochs, "
                "from an on_epoch_end callback, or queue it with request_mutation()",
            )
        blockers = t.mutation_blockers()
        if blockers:
            raise MutationError("E088", "this trainer cannot be mutated: " + "; ".join(blockers))
        if source != "replay" and t._mutations_in_cycle >= t.mutation_limits.max_plans_per_epoch:
            raise MutationError(
                "E088",
                f"{t._mutations_in_cycle} plans already ran this epoch; "
                f"TRAINER.MUTATION.MAX_PLANS_PER_EPOCH is {t.mutation_limits.max_plans_per_epoch}",
            )

    # -- simulate --------------------------------------------------------
    def _floor_cap(self) -> tuple[float, float]:
        coopetition = self.trainer._coopetition_settings()
        return float(coopetition.get("WEIGHT_FLOOR", 0.02)), float(coopetition.get("WEIGHT_CAP", 0.6))

    def _replay_limits(self) -> MutationLimits:
        """What a reconciliation may do (12.9): reproduce a structure that already existed.

        The structure came from plans that were valid under the limits of their day, and a
        configuration edited since must not make a checkpoint unloadable, so the *policy* limits
        are lifted (a root may be the last, a protected path may be purged, purge is allowed, a
        plan may be long, a purged path may come back). Every *structural* rule of the simulator
        (paths, parents, uniqueness, references, feasible weight bounds) still applies.
        """
        return dataclasses.replace(
            self.trainer.mutation_limits,
            min_active_roots=1,
            max_ops_per_plan=10_000,
            protected=(),
            allow_purge=True,
            tombstone="off",
        )

    def simulator(self, source: str = "api") -> TopologySimulator:
        t = self.trainer
        floor, cap = self._floor_cap()
        replay = source == "replay"
        return TopologySimulator(
            t.tree,
            retired=t.freeze.retired(),
            tombstones=() if replay else t._tombstones,
            limits=self._replay_limits() if replay else t.mutation_limits,
            feasible=lambda count: feasible_bounds(count, floor=floor, cap=cap),
        )

    def simulate(self, plan: MutationPlan, source: str = "api") -> Simulation:
        simulation = self.simulator(source).run(plan)
        simulation.raise_for_findings()
        return simulation

    # -- peers -----------------------------------------------------------
    def _factory_module(
        self, spec: PeerSpec, source: Peer | None, seed: int
    ) -> tuple[torch.nn.Module, bool]:
        t = self.trainer
        resolved = t.resolver.resolve(
            ComponentSpec.parse("peer_factory", spec.factory),
            trusted=t._components_trusted,
            inject={
                "spec": spec,
                "source": copy.deepcopy(source.module) if source is not None else None,
                "runtime": t._runtime,
                "settings": t.settings,
                "seed": seed,
            },
        )
        module = resolved.instance
        if not isinstance(module, torch.nn.Module):
            raise MutationError("E085", f"{spec.path}: the factory returned {type(module).__name__}, not an nn.Module")
        # A peer is blended into the others the moment it carries weight: a factory that hands
        # back non-finite parameters must not get that far (a shared module is the tie check's).
        tensors = [*module.named_parameters(), *module.named_buffers()]
        broken = [name for name, tensor in tensors
                  if tensor.is_floating_point() and not bool(torch.isfinite(tensor.detach()).all())]
        if broken:
            raise MutationError(
                "E085", f"{spec.path}: the factory returned non-finite parameters ({', '.join(broken[:3])})"
            )
        return module, bool(resolved.component.capabilities.get("finetuned", False))

    def _build_peer(
        self,
        spec: PeerSpec,
        *,
        tree: PeerTree,
        peers: Mapping[str, Peer],
        topology_epoch: int,
    ) -> Peer:
        """Steps 1-6 of 12.3, under a private RNG stream (M4)."""
        t = self.trainer
        runtime = t._runtime
        path = spec.path
        node = tree[path]
        seed = node_seed(int(runtime.seed), path, topology_epoch)
        sharing = str(getattr(t._hierarchy, "sharing", "independent"))
        if sharing == "shared_trunk" and node.depth > 1 and spec.like is not None:
            raise MutationError(
                "E085",
                f"{path}: `like` copies the module, trunk included, which would unshare it; in a "
                "shared_trunk run a new child uses a factory or the level's model source",
            )
        source = peers.get(spec.like) if spec.like is not None else None
        if spec.like is not None and source is None:
            raise MutationError("E083", f"{path}: `like` names {spec.like!r}, which has no peer", findings=[])
        devices = [runtime.device] if torch.device(runtime.device).type == "cuda" else []
        previous_tree = getattr(t, "_tree_in_build", None)
        t._tree_in_build = tree
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                finetuned = False
                if spec.factory == BUILD_TOKEN:
                    module = t._module_for(path)          # step 3 of 12.3, chosen explicitly
                elif spec.factory is not None:
                    module, finetuned = self._factory_module(spec, source, seed)
                elif source is not None:
                    module, finetuned = copy.deepcopy(source.module), True
                else:
                    module = t._module_for(path)
                if sharing == "shared_trunk" and node.depth > 1 and t._shared_trunk is not None:
                    from .trainer import _with_shared_trunk

                    module = _with_shared_trunk(t._shared_trunk, module)
                peer = t._assemble_peer(path, node.depth, module, runtime, finetuned=finetuned)
        except MutationError:
            raise
        except Exception as error:  # noqa: BLE001 - any construction failure is a refusal, not a crash
            raise MutationError("E085", f"{path}: building the peer failed: {type(error).__name__}: {error}") from error
        finally:
            t._tree_in_build = previous_tree
        if spec.init_from is not None:
            self._copy_weights(path, peer, spec.init_from, peers)
        return peer

    def _weights_of(self, path: str, ref: str, peers: Mapping[str, Peer]) -> torch.Tensor:
        t = self.trainer
        source = t.reference if ref == REFERENCE_TOKEN else peers.get(ref)
        if source is None:
            raise MutationError("E083", f"{path}: `init_from` {ref!r} has no peer to read weights from")
        return source.export_parameters().detach().clone()

    def _init_vector(self, path: str, target: Peer, ref: str, peers: Mapping[str, Peer]) -> torch.Tensor:
        """The weights ``init_from`` names, after the strict layout check (E083, E085)."""
        source = self.trainer.reference if ref == REFERENCE_TOKEN else peers.get(ref)
        vector = self._weights_of(path, ref, peers)
        if source is not None and source.adapter.parameter_shapes() != target.adapter.parameter_shapes():
            raise MutationError("E085", f"{path}: `init_from` {ref!r} has a different parameter layout")
        if vector.numel() != target.export_parameters().numel():
            raise MutationError(
                "E085",
                f"{path}: `init_from` {ref!r} has {vector.numel()} parameters, expected "
                f"{target.export_parameters().numel()}",
            )
        return vector

    def _copy_weights(self, path: str, target: Peer, ref: str, peers: Mapping[str, Peer]) -> None:
        target.import_parameters(self._init_vector(path, target, ref, peers))

    # -- stage -----------------------------------------------------------
    def _stage(self, plan: MutationPlan, simulation: Simulation) -> Staged:
        t = self.trainer
        epoch_next = t.topology_epoch + 1
        tree = simulation.tree
        staged = Staged()
        peers: dict[str, Peer] = {
            path: peer for path, peer in t.peers.items() if path not in simulation.purged
        }
        set_blocks: dict[str, Mapping[str, Any] | None] = {}
        survivors = set(simulation.added)
        for op in plan.ops:
            if op.kind is OpKind.ADD:
                spec = op.spec
                if spec.path not in survivors:
                    continue                      # added and then purged in the same plan
                peer = self._build_peer(spec, tree=tree, peers=peers, topology_epoch=epoch_next)
                staged.peers[spec.path] = peers[spec.path] = peer
                if spec.optimizer is not None:
                    set_blocks[spec.path] = dict(spec.optimizer)
            elif op.kind is OpKind.MODIFY:
                if op.path not in tree:
                    continue                      # purged later in the same plan
                changes = op.changes
                if "rebuild" in changes:
                    spec = PeerSpec.parse(changes["rebuild"], path=op.path)
                    peer = self._build_peer(spec, tree=tree, peers=peers, topology_epoch=epoch_next)
                    staged.peers[op.path] = peers[op.path] = peer
                    if op.path not in survivors and op.path not in staged.rebuilt:
                        staged.rebuilt.append(op.path)
                if "init_from" in changes:
                    target = peers[op.path]
                    vector = self._init_vector(op.path, target, changes["init_from"], peers)
                    if op.path in staged.peers:
                        staged.peers[op.path].import_parameters(vector)
                    else:
                        staged.reweighted[op.path] = vector
                if "optimizer" in changes:
                    set_blocks[op.path] = changes["optimizer"]
        self._check_layouts(simulation, peers, tuple(staged.peers))
        staged.notes.extend(self._check_plans(simulation))
        for change in simulation.changes:
            staged.notes.append(
                f"W042: group {change.label!r} changed membership ({len(change.active_before)} -> "
                f"{len(change.active_after)} active); its aggregator, controller state and per-group "
                "strategy state were rebuilt"
            )
        for path in staged.rebuilt:
            staged.notes.append(
                f"W043: {path} was rebuilt, so its optimizer was rebuilt and its moments are gone"
            )
        build = [path for path in staged.peers]
        active = list(simulation.active_paths())
        runtime = t._runtime
        staged.optimizer = t.optimizer_plan.stage_topology(
            set_blocks={path: block for path, block in set_blocks.items() if path in tree},
            build=build,
            drop=list(simulation.purged),
            active=active,
            live=t.optimizers,
            make=lambda path, entry: t._make_optimizer(path, entry, runtime, peers.get(path)),
        )
        staged.notes.extend(staged.optimizer.warnings)
        return staged

    def _check_layouts(
        self, simulation: Simulation, peers: Mapping[str, Peer], staged_paths: Iterable[str] = ()
    ) -> None:
        """M6 and M7 on the final tree, over active peers.

        A group is checked when its membership changed *or* when it holds a peer the plan built
        (a ``rebuild`` replaces a module without changing who is in the group).
        """
        t = self.trainer
        checks: dict[str, tuple[str, ...]] = {
            change.label: change.active_after for change in simulation.changes
        }
        groups = _groups(simulation.tree, simulation.retired)
        for path in staged_paths:
            node = simulation.tree.get(path)
            if node is None:
                continue
            label = "root" if node.parent is None else node.parent
            if label in groups:
                checks.setdefault(label, groups[label][2])
        for label, members in checks.items():
            if len(members) < 2:
                continue
            try:
                group_layouts(peers, members)
            except ContractError as error:
                raise MutationError("E085", f"group {label!r}: {error}") from error
        try:
            t._validate_tied_parameters(tree=simulation.tree, peers=peers, skip=simulation.retired)
        except ContractError as error:
            raise MutationError("E085", str(error)) from error

    def _check_plans(self, simulation: Simulation) -> list[str]:
        """M-G3's second half: every plan that keys on groups accepts the final groups (E087).

        Returns the plans' own notes (``W022`` for a glob that no longer matches a group).
        """
        from ..coopetition.strategy import tree_view
        from ..coopetition.weights import GroupView

        t = self.trainer
        floor, cap = self._floor_cap()
        groups = _groups(simulation.tree, simulation.retired)
        views = [
            GroupView(
                label=label, parent=parent, members=active,
                level=0 if parent is None else int(simulation.tree[parent].depth),
                floor=floor, cap=cap,
            )
            for label, (parent, _members, active) in groups.items()
            if active and (parent is None or parent not in simulation.retired)
        ]
        active_tree = simulation.tree.without(simulation.retired)
        notes: list[str] = []
        try:
            notes += t.weight_plan.validate(views, initial=False)
            notes += t.strategy_plan.validate(
                tree_view(active_tree, hierarchy=t._hierarchy is not None), initial=False
            )
            notes += t.kappa_plan.validate([view.label for view in views], initial=False)
        except ImproperlyConfigured as error:
            raise MutationError("E087", f"a plan refused the new topology: {error}") from error
        return [str(note) for note in notes]

    # -- commit ----------------------------------------------------------
    def _commit(self, plan: MutationPlan, simulation: Simulation, staged: Staged, source: str, undo: Undo) -> None:
        """Phase 3. ``undo`` is owned by ``apply`` so its ``except`` clause can always reach it."""
        t = self.trainer
        tree = simulation.tree
        purged = list(simulation.purged)

        # 1. The tree, in place: every holder of ``trainer.tree`` (freeze set, watchdog) sees it.
        previous_nodes = t.tree.adopt(tree)
        undo.push("tree", lambda: t.tree.restore(previous_nodes))

        # 2. Peers.
        undo.restore_dict("peers", t.peers)
        for path in purged:
            t.peers.pop(path, None)
        for path, peer in staged.peers.items():
            t.peers[path] = peer
            t.tree[path].peer = peer
        for path, vector in staged.reweighted.items():
            before = t.peers[path].export_parameters().detach().clone()
            t.peers[path].import_parameters(vector)
            undo.push(f"weights:{path}", lambda p=path, b=before: t.peers[p].import_parameters(b))

        # 3. Optimizers.
        undo.restore_dict("optimizers", t.optimizers)
        for path in purged:
            t.optimizers.pop(path, None)
        for path, facade in staged.optimizer.optimizers.items():
            peer = t.peers[path]
            old_facade = peer.optimizer
            undo.push(f"peer.optimizer:{path}", lambda p=peer, o=old_facade: setattr(p, "optimizer", o))
            t._install_optimizer(path, facade)
        undo.push("optimizer_plan", t.optimizer_plan.commit_topology(staged.optimizer))

        # 4. The freeze set: retired, revived, purged, and the optimizer:null reasons.
        saved_freeze = t.freeze.snapshot()
        undo.push("freeze", lambda: t.freeze.restore(saved_freeze))
        for path in purged:
            t.freeze.forget(path)
        for path in simulation.revived_now:
            t.freeze.discard(path, RETIRED)
        for path in simulation.retired_now:
            t.freeze.add(path, RETIRED)
        for path in staged.optimizer.frozen:
            t.freeze.add(path, "optimizer:null")
        for path in staged.optimizer.thawed:
            t.freeze.discard(path, "optimizer:null")
        t.freeze.sync()

        # 5. Controllers and aggregators of the groups that changed.
        self._rebind_groups(simulation, undo)

        # 6. Caches keyed by group label or node path, through the registry (M9).
        t.path_state.forget(
            paths=purged + list(staged.peers) + list(staged.reweighted),
            labels=simulation.changed_labels(),
            undo=undo,
            dropped_paths=purged,
        )

        # 7. Scalars and the journal.
        undo.restore_set("tombstones", t._tombstones)
        t._tombstones.clear()
        t._tombstones.update(simulation.tombstones)
        self._record_manifest(plan, simulation, undo)
        undo.restore_attr("topology_epoch", t, "topology_epoch")
        t.topology_epoch += 1
        undo.restore_list("journal", t.mutation_journal)
        t.mutation_journal.append(
            {
                "seq": len(t.mutation_journal),
                "epoch": int(t._epoch),
                "topology_epoch": int(t.topology_epoch),
                "source": source,
                "plan": plan.canonical(),
                "plan_digest": plan.digest(),
                "before": simulation.before_digest,
                "after": simulation.after_digest,
            }
        )
        undo.restore_attr("active_view", t, "_active_view")
        t._active_view = None
        undo.restore_attr("budget", t, "_mutations_in_cycle")
        t._mutations_in_cycle += 1

        # 8. The audit (M9). A problem here is a bug, and it is rolled back like any other fault.
        problems = t.path_state.audit()
        if problems:
            raise MutationError("E089", "the post-commit audit found: " + "; ".join(problems))

    def _record_manifest(self, plan: MutationPlan, simulation: Simulation, undo: Undo) -> None:
        """What a checkpoint needs to rebuild this structure on another trainer (12.9.2).

        ``_node_specs`` holds the effective spec of every node a plan created or rebuilt, so a
        reconciliation can build the same architecture; ``_node_meta`` holds the final metadata of
        every node a plan created or touched, because the tree section of a checkpoint carries no
        metadata. A node built from the configuration and never touched has no entry in either,
        and that absence is what "built from the configuration" means.
        """
        t = self.trainer
        undo.restore_dict("node_specs", t._node_specs)
        undo.restore_dict("node_meta", t._node_meta)
        for path in simulation.purged:
            t._node_specs.pop(path, None)
            t._node_meta.pop(path, None)
        survivors = set(simulation.added)
        tree = simulation.tree
        for op in plan.ops:
            if op.kind is OpKind.ADD and op.spec.path in survivors:
                t._node_specs[op.spec.path] = manifest_spec(op.spec)
            elif op.kind is OpKind.MODIFY and op.path in tree and "rebuild" in op.changes:
                # A rebuild replaces the architecture (``factory`` / ``like``) and nothing else: the
                # optimizer block and the role the node was created with stay in its spec.
                renewed = manifest_spec(PeerSpec.parse(op.changes["rebuild"], path=op.path))
                prior = t._node_specs.get(op.path, {"path": op.path})
                kept = {key: value for key, value in prior.items() if key not in ("factory", "like")}
                t._node_specs[op.path] = {**kept, **renewed}
        for path in dict.fromkeys((*simulation.added, *simulation.modified)):
            if path not in tree:
                continue
            metadata = dict(tree[path].metadata)
            if metadata or path in t._node_meta:
                t._node_meta[path] = metadata

    def _rebind_groups(self, simulation: Simulation, undo: Undo) -> None:
        t = self.trainer
        undo.restore_dict("controllers", t.controllers)
        undo.restore_dict("level_controllers", t._level_controllers)
        undo.restore_dict("group_aggregators", t._group_aggregators)
        base = dict(t._coopetition_settings())
        for change in simulation.changes:
            label, parent = change.label, change.parent
            t._group_aggregators.pop(label, None)
            key = "root" if parent is None else parent
            if change.vanished:
                t.controllers.pop(key, None)
                t._level_controllers.pop(parent, None)
                continue
            if parent is not None and parent in simulation.retired:
                continue                                  # the whole group is out of play: leave its controller
            count = max(1, len(change.active_after))
            controller = t._level_controllers.get(parent)
            if controller is None:
                controller = t.kappa_plan.build(label, base, num_peers=count)
                t._level_controllers[parent] = controller
                t.controllers[key] = controller
                continue
            saved = controller.snapshot()
            undo.push(f"kappa:{label}", lambda c=controller, s=saved: c.restore(s))
            controller.resize(count)

    # -- apply -----------------------------------------------------------
    def apply(
        self, plan: "MutationPlan | Mapping[str, Any] | list[Any]", *, dry_run: bool = False, source: str = "api"
    ) -> MutationReport:
        started = time.monotonic()
        plan = MutationPlan.parse(plan)
        with self._lock:
            self._gate(source)
            simulation = self.simulate(plan, source)
            staged = self._stage(plan, simulation)
            if dry_run:
                return self._report(plan, simulation, staged, source, dry_run=True, started=started)
            undo = Undo()
            try:
                self._commit(plan, simulation, staged, source, undo)
            except Exception as error:  # noqa: BLE001 - phase 3 must either finish or be undone
                problems = undo.rollback()
                if problems:
                    self.poisoned = MutationError(
                        "E089",
                        f"the commit failed ({error}) and the rollback did not complete: {problems}; "
                        "this trainer's state is unreliable and must be discarded",
                        rolled_back=False,
                    )
                    raise self.poisoned from error
                code = error.code if isinstance(error, MutationError) else "E089"
                raise MutationError(
                    code,
                    f"the commit failed and was rolled back: {error}",
                    findings=getattr(error, "findings", ()),
                    rolled_back=True,
                ) from error
            report = self._report(plan, simulation, staged, source, dry_run=False, started=started)
        self._finish(report)
        return report

    def _report(
        self, plan: MutationPlan, simulation: Simulation, staged: Staged, source: str, *, dry_run: bool, started: float
    ) -> MutationReport:
        t = self.trainer
        return MutationReport(
            plan_digest=plan.digest(), label=plan.label, reason=plan.reason, source=source,
            dry_run=dry_run, epoch=int(t._epoch),
            topology_epoch=int(t.topology_epoch),
            added=simulation.added, retired=simulation.retired_now, revived=simulation.revived_now,
            purged=simulation.purged, modified=simulation.modified, rebuilt=tuple(staged.rebuilt),
            optimizers_rebuilt=tuple(staged.optimizer.optimizers) if staged.optimizer is not None else (),
            groups=simulation.changed_labels(),
            active_nodes=len(simulation.active_paths()), retired_nodes=len(simulation.retired),
            before_digest=simulation.before_digest, after_digest=simulation.after_digest,
            warnings=tuple(item.render() for item in simulation.soft) + tuple(staged.notes),
            seconds=time.monotonic() - started,
        )

    def _finish(self, report: MutationReport) -> None:
        """After the commit: observers. An observer's failure never undoes a committed mutation."""
        from .. import signals

        t = self.trainer
        if report.source == "replay":
            # A reconciliation (12.9) restores a structure that already happened. It rebases the
            # health baselines itself, once, after the last plan; no callback or signal hears about
            # a topology change that the checkpoint says was already announced.
            return
        if len(t.mutation_journal) == t.mutation_limits.journal_warn:
            log.warning(
                "W045: the mutation journal reached %d entries; every checkpoint carries all of it "
                "(TRAINER.MUTATION.JOURNAL_WARN)",
                len(t.mutation_journal),
            )
        for note in report.warnings:
            log.warning("%s", note)
        for hook in (
            lambda: t._after_mutation(report),
            lambda: t._dispatch_many("on_tree_mutated", t, report),
            lambda: signals.tree_mutated.send(sender=t, report=report),
        ):
            try:
                hook()
            except Exception as error:  # noqa: BLE001 - observers are not part of the transaction
                log.error("an on_tree_mutated observer failed: %s", error)
                t.failures.append(f"on_tree_mutated observer: {error}")

    # -- deferred plans --------------------------------------------------
    def request(self, plan: "MutationPlan | Mapping[str, Any] | list[Any]", *, source: str = "request") -> MutationPlan:
        """Queue a plan for the top of the next epoch (the form a callback or a strategy uses)."""
        parsed = MutationPlan.parse(plan)
        with self._lock:
            self._queue.append((parsed, source))
        return parsed

    def pending(self) -> list[dict[str, Any]]:
        with self._lock:
            return [{"source": source, "plan": plan.canonical()} for plan, source in self._queue]

    def restore_pending(self, rows: "Iterable[Mapping[str, Any]]") -> int:
        """Replace the queue with what a checkpoint recorded (12.9). Returns how many plans were queued."""
        parsed = [(MutationPlan.parse(row["plan"]), str(row.get("source", "request"))) for row in rows]
        with self._lock:
            self._queue[:] = parsed
        return len(parsed)

    def discard_pending(self) -> int:
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count

    def apply_pending(self, epoch: int) -> list[MutationReport]:
        """Called at the top of ``run_epoch``: queued plans, then this epoch's schedule entries."""
        t = self.trainer
        due: list[tuple[MutationPlan, str]] = []
        with self._lock:
            due.extend(self._queue)
            self._queue.clear()
        done = {
            (int(entry["epoch"]), entry["plan_digest"])
            for entry in t.mutation_journal if entry.get("source") == "schedule"
        }
        for at_epoch, plan in t.mutation_limits.schedule:
            if at_epoch == int(epoch) and (int(epoch), plan.digest()) not in done:
                due.append((plan, "schedule"))
        reports: list[MutationReport] = []
        for plan, source in due:
            try:
                reports.append(self.apply(plan, source=source))
            except MutationError as error:
                if error.code == "E089" and not error.rolled_back:
                    raise                       # a rollback that failed does end the run (R9)
                # A queued or scheduled plan that is no longer valid must not end the run.
                t.failures.append(f"mutation {plan.label or plan.digest()} ({source}) refused: {error}")
                log.error("mutation %s (%s) refused: %s", plan.label or plan.digest(), source, error)
        return reports
