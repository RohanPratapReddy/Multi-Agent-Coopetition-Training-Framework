"""Hierarchical scoring and inference (document 10 sections 5 and 7).

One code path: :class:`HierarchicalScorer` scores a :class:`PeerTree` leaves-first
through per-node :class:`~mactf.scoring.pipeline.ScoringPipeline` instances. A flat
run is a depth-1 tree, so the same class scores today's swarm.

The property this module exists to guarantee is non-offsettable propagation:

    If any node in a branch violated a non-offsettable rule, every ancestor of
    that node is void for the episode and takes the floor.

Averaging a voided child with healthy siblings produces a number, and that number
must not launder a T0/T1/T2 breach, so :func:`branch_void` unions each node's void
set into every ancestor and the union is passed to
:func:`mactf.compliance.constrain` at every node.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

import torch

from ..compliance import effective_non_offsettable
from ..components import HUB
from ..components.guards.rollup import guard_rollup
from ..coopetition.tree import PeerNode, PeerTree, branch_void, node_seed
from ..exceptions import ContractError, ValidationError
from ..rl.functions.registry import FUNCTION_TYPES, build_function
from ..rl.functions.rollup import (
    ExplicitRollup,
    KappaWeightedRollup,
    MeanRollup,
    MinimumRollup,
    RollupStrategy,
    SumRollup,
    WorstCaseRollup,
)
from .base import ScoreContext, Scorer, ScorerOutput, ScoringKind
from .ledger import Ledger, RuleEvent
from .pipeline import ScoringPipeline
from .scorecard import EpisodeScorecard, PeerScorecard

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..coopetition.delegation import DelegationPolicy
    from ..coopetition.synthesis import SynthesisFunction
    from ..orchestration.episode import Task

__all__ = [
    "ExplicitRollup",
    "HierarchicalScorer",
    "InferenceNodeResult",
    "InferenceResult",
    "KappaWeightedRollup",
    "MeanRollup",
    "MinimumRollup",
    "ROLLUP_STRATEGIES",
    "ROLLUP_TYPES",
    "RollupStrategy",
    "SumRollup",
    "WorstCaseRollup",
    "branch_void",
    "build_rollup",
    "inference_pipeline",
    "reset_inference_pipeline",
    "run_inference",
]


#: The penalty rule a failed node is recorded under when inference is scored.
#: ``P14`` is sabotage/malicious compliance; a node that cannot produce an output
#: is closer to abandonment, so ``P23`` is the honest mapping: the node failed to
#: deliver a proposal the rest of the branch was relying on.
INFERENCE_FAILURE_RULE = "P23"

#: The process-wide inference scoring pipeline, created on first use.
#:
#: ``persist_warnings=True`` requests accumulate warnings here rather than in a
#: training pipeline's registries, which is the "separate inference namespace"
#: document 10 section 7.5 requires so the two counters cannot mix.
_INFERENCE_PIPELINE: ScoringPipeline | None = None


def inference_pipeline() -> ScoringPipeline:
    """The shared pipeline production inference warnings persist into."""
    global _INFERENCE_PIPELINE
    if _INFERENCE_PIPELINE is None:
        _INFERENCE_PIPELINE = ScoringPipeline()
    return _INFERENCE_PIPELINE


def reset_inference_pipeline() -> None:
    """Discard the shared inference pipeline, dropping its warnings."""
    global _INFERENCE_PIPELINE
    _INFERENCE_PIPELINE = None

#: Registry keys used by ``RL.ROLLUP`` and by spec resolution: the same live object as
#: ``rl.functions.registry.ROLLUP_TYPES`` (a ``LegacyTable`` over ``rl_function.rollup``), so a
#: rollup registered anywhere is available everywhere (document 17 section 8.6.1).
ROLLUP_STRATEGIES = FUNCTION_TYPES["rollup"]
ROLLUP_TYPES = ROLLUP_STRATEGIES


def build_rollup(spec: Any, *, guard_mode: str = "enforce") -> Any:
    """Resolve a rollup from a strategy, a name, or a spec mapping.

    A rollup that is not one of the six built-in classes is wrapped in a ``RollupGuard``
    (8.6.2). The built-ins are returned as they are, ``ExplicitRollup`` included: an
    ``ExplicitRollup(fn)`` constructed in code is the documented "I own the whole output"
    exception (``rl/functions/rollup.py``). The ``RL.ROLLUP`` path is stricter and guards it
    too (``FunctionBook``, 8.4).
    """
    if spec is None:
        return KappaWeightedRollup()
    if isinstance(spec, RollupStrategy) or hasattr(spec, "rollup"):
        return guard_rollup(spec, mode=guard_mode, include_explicit=False)
    payload: dict[str, Any] = {"type": spec} if isinstance(spec, str) else dict(spec)
    if "import" not in payload:
        payload.setdefault("type", "kappa_weighted")
        registry = HUB.registry("rl_function.rollup")
        if not registry.has(str(payload["type"])):
            raise ValidationError(
                f"unknown rollup {payload['type']!r}; accepted: " + ", ".join(registry.names())
            )
    built = build_function(payload, family="rollup")
    return guard_rollup(built, mode=guard_mode, include_explicit=False)


class HierarchicalScorer(Scorer):
    """Scores a whole tree in bottom-up order, one pipeline per node.

    ``pipeline_factory`` exists so a caller can vary configuration per level (a
    different ``REWARD_WEIGHTS`` for micro-agents, say). In the common case it is
    one shared ``ScoringPipeline``, which is stateless apart from the warning
    registries — and those are keyed by ``(node_path, peer)`` already.
    """

    kind = ScoringKind.HIERARCHY
    order = 0
    requires = frozenset({"tree", "ledgers"})

    def __init__(
        self,
        pipeline_factory: Callable[[str], ScoringPipeline] | ScoringPipeline | None = None,
        *,
        rollup: Any = None,
        attribution: str = "rollup",
        non_offsettable: Iterable[str] | None = None,
        rollup_source: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__()
        if pipeline_factory is None:
            pipeline_factory = ScoringPipeline()
        if isinstance(pipeline_factory, ScoringPipeline):
            self.pipeline_factory: Callable[[str], ScoringPipeline] = lambda path: pipeline_factory
            self._shared = pipeline_factory
        elif callable(pipeline_factory):
            self.pipeline_factory = pipeline_factory
            self._shared = None
        else:
            raise ValidationError(
                "HierarchicalScorer needs a ScoringPipeline or a callable returning one"
            )
        self._rollup: Any = build_rollup(rollup)
        self._rollup_source = rollup_source
        if attribution not in ("rollup", "own"):
            raise ValidationError(
                f"unknown attribution {attribution!r}; use 'rollup' or 'own'"
            )
        self.attribution = attribution
        self.non_offsettable: frozenset[str] = effective_non_offsettable(non_offsettable)

    @property
    def rollup(self) -> Any:
        """The strategy in force now: the source's choice when it has one, else the default."""
        if self._rollup_source is not None:
            chosen = self._rollup_source()
            if chosen is not None:
                return chosen
        return self._rollup

    @rollup.setter
    def rollup(self, value: Any) -> None:
        self._rollup = build_rollup(value)

    @property
    def points_table(self) -> Any:
        """The points table the tree is scored against (the root pipeline's)."""
        return self.pipeline_factory("").points_table

    # -- pipeline access -------------------------------------------------
    def pipeline_for(self, path: str) -> ScoringPipeline:
        pipeline = self.pipeline_factory(path)
        if not isinstance(pipeline, ScoringPipeline):
            raise ValidationError(
                f"pipeline_factory returned {type(pipeline).__name__} for "
                f"{path!r}, not a ScoringPipeline"
            )
        return pipeline

    @property
    def pipeline(self) -> ScoringPipeline:
        """The shared pipeline, when there is one; otherwise a depth-1 instance."""
        return self._shared or self.pipeline_for("")

    def registry_for(self, peer: str, *, node_path: str = "") -> Any:
        return self.pipeline_for(node_path).warning_scorer.registry_for_node(node_path, peer)

    # -- scoring ---------------------------------------------------------
    def score_tree(
        self,
        tree: PeerTree,
        ledgers: Mapping[str, Ledger],
        *,
        episode: int = 0,
        known_rules: set[str] | None = None,
        weights: Mapping[str, Mapping[str, float]] | None = None,
        scores: Mapping[str, float] | None = None,
        evidence: Mapping[str, Any] | None = None,
    ) -> EpisodeScorecard:
        """Score every node leaves-first and return a scorecard keyed by path.

        ``evidence`` maps node path to that node's episode evidence; nodes without
        an entry get no points breakdown (``card.points is None``).

        Step 1a of document 10 section 5.1 in ``attribution="rollup"`` mode
        attaches each descendant's events to the ancestor ledger tagged
        ``derived_from=<child path>``, so a parent's scorecard decomposes into
        "events this node caused" and "events inherited from these children".
        """
        if not isinstance(tree, PeerTree):
            raise ContractError(
                f"score_tree needs a PeerTree, got {type(tree).__name__}"
            )
        for path in ledgers:
            if path not in tree:
                raise ContractError(
                    f"ledgers names {path!r}, which is not a node in this tree"
                )

        card = EpisodeScorecard(episode=episode)
        outputs: dict[str, ScorerOutput] = {}
        cards: dict[str, PeerScorecard] = {}

        # Resolve each node's own void set first, then propagate: a parent must
        # see its branch's void before it is scored, not after.
        own_void = self._own_void(tree, ledgers, non_offsettable=self.non_offsettable)
        branch = branch_void(tree, own_void)

        for node in tree.walk("leaves-first"):
            pipeline = self.pipeline_for(node.path)
            ledger = self._ledger_for_node(node, tree, ledgers)
            child_outputs = tuple(
                self._child_output(outputs, cards, child, node_path=node.path)
                for child in node.children
            )
            peer_card = pipeline.score_peer(
                ledger,
                peer=ledgers[node.path].peer if node.path in ledgers else node.path,
                episode=episode,
                known_rules=known_rules,
                node_path=node.path,
                depth=node.depth,
                branch_void=branch.get(node.path, frozenset()),
                children=child_outputs,
                evidence=(evidence or {}).get(node.path),
            )
            card.add(peer_card)
            cards[node.path] = peer_card

            # The output this node reports upward: its own continuous score plus
            # its branch's void union, then the rollup of its children.
            own = ScorerOutput(
                kind=ScoringKind.HIERARCHY,
                amount=float(peer_card.final_score),
                breakdown={
                    "final_score": float(peer_card.final_score),
                    "reward": float(peer_card.continuous.reward.total),
                    "penalty": float(peer_card.continuous.penalty.total),
                    "warning": float(peer_card.continuous.warning),
                    "node_path": node.path,
                    "depth": node.depth,
                },
                rules=tuple(
                    dict.fromkeys(
                        event.rule_id
                        for event in ledger.events
                        if event.rule.is_reward and self.attribution == "rollup"
                    )
                ),
                metadata={"node_path": node.path, "peer": peer_card.peer},
                void_rules=tuple(sorted(branch.get(node.path, ()))),
            )
            outputs[node.path] = own

        card.metadata.update(
            {
                "attribution": self.attribution,
                "rollup": getattr(self.rollup, "name", type(self.rollup).__name__),
                "branch_void": {
                    path: sorted(rules) for path, rules in branch.items() if rules
                },
            }
        )
        return card

    def _own_void(
        self,
        tree: PeerTree,
        ledgers: Mapping[str, Ledger],
        *,
        non_offsettable: frozenset[str],
    ) -> dict[str, frozenset[str]]:
        """Each node's own non-offsettable rule ids, from its own ledger only.

        ``non_offsettable`` is the effective superset, so a custom set can add rules but
        can no longer narrow the catalog default (the old ``catalog_defaults`` branch let a
        non-empty set do exactly that).
        """
        result: dict[str, frozenset[str]] = {}
        for node in tree.walk("leaves-first"):
            ledger = ledgers.get(node.path)
            if ledger is None:
                result[node.path] = frozenset()
                continue
            result[node.path] = frozenset(
                event.rule_id for event in ledger.events if event.rule_id in non_offsettable
            )
        return result

    def _ledger_for_node(
        self,
        node: PeerNode,
        tree: PeerTree,
        ledgers: Mapping[str, Ledger],
    ) -> Ledger:
        """The node's own ledger, plus its descendants' events when attributing."""
        own = ledgers.get(node.path, Ledger(peer=node.path))
        if self.attribution == "own" or not node.children:
            return own
        merged = Ledger(peer=own.peer or node.path)
        merged.extend(own.events)
        for descendant in tree.descendants(node.path):
            child_ledger = ledgers.get(descendant)
            if child_ledger is None:
                continue
            for event in child_ledger.events:
                merged.record(
                    event.rule_id,
                    magnitude=event.magnitude,
                    severity=event.severity,
                    evidence=event.evidence,
                    source=event.source,
                    occurred_at=event.occurred_at,
                    tags=frozenset(event.tags) | {f"derived_from:{descendant}"},
                )
        return merged

    def _child_output(
        self,
        outputs: Mapping[str, ScorerOutput],
        cards: Mapping[str, PeerScorecard],
        child: str,
        *,
        node_path: str,
    ) -> ScorerOutput:
        if child in outputs:
            return outputs[child]
        card = cards.get(child)
        if card is None:
            raise ContractError(
                f"{node_path!r} has child {child!r} with no scorecard; score_tree walks "
                "leaves-first, so this indicates a corrupted tree"
            )
        return ScorerOutput(kind=ScoringKind.HIERARCHY, amount=float(card.final_score))

    # -- the stage interface ---------------------------------------------
    def score(self, ctx: ScoreContext) -> ScorerOutput:
        """Roll the context's child outputs up into this node's contribution."""
        weights = ctx.extras.get("rollup_weights")
        scores = ctx.extras.get("rollup_scores")
        return self.rollup.rollup(
            ctx.node_path or ctx.peer,
            ctx.children,
            weights=weights,
            scores=scores,
        )

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update(
            {
                "attribution": self.attribution,
                "rollup": getattr(self.rollup, "name", type(self.rollup).__name__),
                "non_offsettable": sorted(self.non_offsettable),
                "shared_pipeline": self._shared is not None,
            }
        )
        return payload


# ---------------------------------------------------------------------------
# Inference (document 10 section 7)
# ---------------------------------------------------------------------------
@dataclass
class InferenceNodeResult:
    """What one node did during an inference request."""

    path: str
    depth: int
    duration: float = 0.0
    output_shape: tuple[int, ...] | None = None
    selected: tuple[str, ...] = ()
    starved: tuple[str, ...] = ()
    batched: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "depth": self.depth,
            "output_shape": list(self.output_shape) if self.output_shape else None,
            "duration": self.duration,
            "selected": list(self.selected),
            "starved": list(self.starved),
            "batched": self.batched,
            "error": self.error,
        }


@dataclass
class InferenceResult:
    """The tree's answer plus everything needed to audit how it was reached."""

    outputs: dict[str, Any] = field(default_factory=dict)
    root_outputs: dict[str, Any] = field(default_factory=dict)
    per_node: list[InferenceNodeResult] = field(default_factory=list)
    batching: list[dict[str, Any]] = field(default_factory=list)
    cost: dict[str, Any] = field(default_factory=dict)
    scorecard: EpisodeScorecard | None = None
    mode: dict[str, bool] = field(default_factory=dict)
    duration: float = 0.0
    seed: int | None = None
    delegation: dict[str, Any] = field(default_factory=dict)

    @property
    def root_output(self) -> Any:
        """The single root's output, for the common single-root tree."""
        if len(self.root_outputs) == 1:
            return next(iter(self.root_outputs.values()))
        return self.root_outputs

    def as_dict(self) -> dict[str, Any]:
        def _serialise(value: Any) -> Any:
            if torch.is_tensor(value):
                return {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype).replace("torch.", ""),
                    "values": value.detach().cpu().tolist()
                    if value.numel() <= 1024
                    else None,
                }
            return value

        return {
            "outputs": {path: _serialise(value) for path, value in self.outputs.items()},
            "root_outputs": {
                path: _serialise(value) for path, value in self.root_outputs.items()
            },
            "per_node": [node.as_dict() for node in self.per_node],
            "batching": list(self.batching),
            "cost": dict(self.cost),
            "scorecard": self.scorecard.to_dict() if self.scorecard else None,
            "mode": dict(self.mode),
            "duration": self.duration,
            "seed": self.seed,
            "delegation": dict(self.delegation),
        }

    def describe(self) -> str:
        return (
            f"InferenceResult(nodes={len(self.per_node)}, roots={list(self.root_outputs)}, "
            f"invocations={self.cost.get('invocations', 0)}, "
            f"duration={self.duration:.3f}s)"
        )


def run_inference(
    tree: PeerTree,
    inputs: Any,
    *,
    delegation: "DelegationPolicy | None" = None,
    synthesis: "SynthesisFunction | None" = None,
    max_depth: int | None = None,
    budget: int | None = None,
    parallel: bool = False,
    max_workers: int = 4,
    score: bool = False,
    scorer: HierarchicalScorer | ScoringPipeline | None = None,
    task: "Task | None" = None,
    node_path: str | None = None,
    seed: int | None = None,
    persist_warnings: bool = False,
    isolate_failures: bool = True,
) -> InferenceResult:
    """Run the tree over ``inputs`` and synthesize an answer at each root.

    ``node_path=None`` starts at every root; a path starts at that subtree. The
    algorithm is depth-limited delegation, leaves-first execution, then
    leaves-first synthesis, exactly as document 10 section 7.2 specifies.

    ``parallel=True`` uses a bounded :class:`ThreadPoolExecutor`. It does not
    promise a speedup; it promises identical results, which is what criterion 29
    tests.

    ``isolate_failures=True`` (the default) implements document 10 section 9.2: a
    node that raises is recorded with its error and skipped, and its parent
    synthesizes from whatever proposals survived. ``False`` re-raises the first
    failure, which is what a caller debugging a bad module wants.
    """
    from ..coopetition.delegation import BroadcastDelegation
    from ..coopetition.synthesis import ConsensusSynthesis

    if not isinstance(tree, PeerTree):
        raise ContractError(f"run_inference needs a PeerTree, got {type(tree).__name__}")
    delegation = delegation or BroadcastDelegation()
    synthesis = synthesis or ConsensusSynthesis()
    if node_path is not None and node_path not in tree:
        raise ContractError(f"run_inference: no node at path {node_path!r}")
    if max_workers < 1:
        raise ValidationError(f"max_workers must be >= 1, got {max_workers!r}")

    depth_limit = int(max_depth if max_depth is not None else tree.max_depth)
    roots = (node_path,) if node_path is not None else tree.root_paths
    if not roots:
        raise ContractError("run_inference needs a tree with at least one root")
    started = time.monotonic()

    budget_state: dict[str, Any] = {
        "remaining": None if budget is None else int(budget),
        "used": 0,
        "total": None if budget is None else int(budget),
    }
    results: dict[str, InferenceNodeResult] = {}
    outputs: dict[str, Any] = {}
    delegation_log: dict[str, dict[str, Any]] = {}

    # -- 1. plan: delegate root-first, bounded by depth and budget ---------
    plan = _plan(tree, roots, delegation, depth_limit=depth_limit, budget_state=budget_state,
                 task=task, seeds=_seed_source(seed))
    for path, record in plan.items():
        results[path] = InferenceNodeResult(
            path=path,
            depth=tree[path].depth,
            selected=tuple(record["selected"]),
            starved=tuple(record["starved"]),
        )
        delegation_log[path] = {
            "selected": list(record["selected"]),
            "starved": list(record["starved"]),
        }

    # -- 2. prepare inputs against the root contract -----------------------
    prepared = _prepare_inputs(tree, roots, inputs)

    # -- 3. run the leaves and the internal synthesizers, leaves-first -----
    def run_node(node: PeerNode) -> None:
        record = results[node.path]
        node_started = time.monotonic()
        try:
            if node.is_leaf or not record.selected:
                value = _run_module(tree, node, prepared, seed=seed)
                outputs[node.path] = value
            else:
                proposals = {
                    child: outputs[child]
                    for child in record.selected
                    if child in outputs
                }
                if not proposals:
                    raise ContractError(
                        f"no proposals to synthesize for {node.path}: every selected "
                        f"child failed ({', '.join(record.selected) or 'none selected'})"
                    )
                outputs[node.path] = synthesis.synthesize(
                    node,
                    proposals,
                    scores=None,
                    weights=None,
                    task=task,
                )
            value = outputs[node.path]
            if torch.is_tensor(value):
                record.output_shape = tuple(int(size) for size in value.shape)
        except Exception as error:  # noqa: BLE001 - isolated per document 10 section 9.2
            record.error = f"{type(error).__name__}: {error}"
            outputs.pop(node.path, None)
            # Failure isolation (document 10 section 9.2): a broken node does not
            # abort the request when ``isolate_failures`` is set. The tree itself
            # is left untouched, because inference is observational — the same
            # reason warnings are not persisted (section 7.5). A parent whose every
            # selected child failed then fails its own empty-synthesis check,
            # which is how failure propagates up the branch.
            if not isolate_failures:
                raise
        finally:
            record.duration = time.monotonic() - node_started

    # Batching decision per level is recorded before any call, so a per-node
    # fallback is visible rather than a silent shape coercion (section 7.3).
    batching_log = _batching_log(tree, results, prepared)

    ordered = [node.path for node in tree.walk("leaves-first") if node.path in results]

    # Execution: a sequential leaves-first walk, or the same walk with each
    # level's independent nodes submitted together. Both produce identical
    # outputs because a node reads only its children's completed outputs.
    if parallel:
        with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
            for depth in sorted(tree.levels, reverse=True):
                batch = [
                    tree[path] for path in tree.level(depth) if path in results
                ]
                futures = [executor.submit(run_node, node) for node in batch]
                for future in futures:
                    future.result()
    else:
        for path in ordered:
            run_node(tree[path])

    root_outputs = {root: outputs.get(root) for root in roots}
    total = time.monotonic() - started

    result = InferenceResult(
        outputs=dict(outputs),
        root_outputs=root_outputs,
        per_node=[results[path] for path in ordered if path in results],
        batching=batching_log,
        cost={
            "invocations": budget_state["used"],
            "wall_time": total,
            "per_level": _per_level_cost(tree, results),
            "budget": budget,
        },
        mode={
            "warnings_persisted": bool(persist_warnings),
            "training": False,
            "kappa_updated": False,
            "checkpointed": False,
        },
        duration=total,
        seed=seed,
        delegation=delegation_log,
    )

    if score:
        result.scorecard = _score_inference(
            tree, outputs, scorer=scorer, task=task, roots=roots,
            persist_warnings=persist_warnings, per_node=result.per_node,
        )
    return result


# -- helpers ---------------------------------------------------------------
def _seed_source(seed: int | None) -> Callable[[str], int]:
    base = int(seed) if seed is not None else 0
    return lambda path: node_seed(base, path, 0)


def _plan(
    tree: PeerTree,
    roots: Sequence[str],
    delegation: "DelegationPolicy",
    *,
    depth_limit: int,
    budget_state: dict[str, int | None],
    task: "Task | None",
    seeds: Callable[[str], int],
) -> dict[str, dict[str, Any]]:
    """Root-first delegation, charging the budget before each planned call."""
    plan: dict[str, dict[str, Any]] = {}
    frontier = list(roots)
    while frontier:
        for path in frontier:
            node = tree[path]
            if node.depth > depth_limit:
                raise ContractError(
                    f"inference max_depth={depth_limit} reached at {path!r} "
                    f"(depth {node.depth})"
                )
            if node.children and node.depth < depth_limit:
                selected = delegation.select(
                    node,
                    task,
                    tree=tree,
                    scores=None,
                    weights=None,
                    epoch=0,
                )
            else:
                selected = ()
            starved = tuple(child for child in node.children if child not in selected)
            _charge(budget_state, node, len(selected) + 1)
            plan[path] = {"selected": tuple(selected), "starved": starved}
        next_frontier: list[str] = []
        for path in frontier:
            next_frontier.extend(plan[path]["selected"])
        frontier = [path for path in next_frontier if tree[path].depth <= depth_limit]
        if not frontier:
            break
    return plan


def _charge(budget_state: dict[str, int | None], node: PeerNode, cost: int) -> None:
    """Consume budget before a call, so a runaway tree fails fast (section 7.3)."""
    remaining = budget_state["remaining"]
    if remaining is None:
        budget_state["used"] = int(budget_state["used"] or 0) + cost
        return
    if cost > remaining:
        total = budget_state.get("total")
        used = int(budget_state["used"] or 0)
        raise ContractError(
            f"inference budget exhausted at {node.path!r}; {remaining} of "
            f"{total if total is not None else remaining + used} invocations remain "
            f"and {cost} would be used"
        )
    budget_state["remaining"] = remaining - cost
    budget_state["used"] = int(budget_state["used"] or 0) + cost


def _prepare_inputs(tree: PeerTree, roots: Sequence[str], inputs: Any) -> Mapping[str, Any]:
    """Validate ``inputs`` against the root's peer contract."""
    payload: Mapping[str, Any]
    if isinstance(inputs, Mapping):
        payload = dict(inputs)
    elif torch.is_tensor(inputs):
        payload = {"state": inputs}
    else:
        raise ContractError(
            f"inference inputs must be a tensor or a mapping, got {type(inputs).__name__}"
        )
    for root in roots:
        peer = tree[root].peer
        if peer is None:
            continue
        names = tuple(peer.adapter.contract.input_names)
        acceptable = set(names) | {"state", "input_ids", "inputs"}
        missing = [
            name
            for name in names
            if name not in payload and name not in {"attention_mask"}
        ]
        if missing and "state" in payload:
            # A single tensor is accepted for a one-argument module, which is the
            # overwhelmingly common case; anything else must be named.
            if len(names) == 1:
                continue
            raise ContractError(
                f"root {root!r} expects inputs {list(names)}; missing "
                + ", ".join(missing)
            )
        if missing:
            raise ContractError(
                f"root {root!r} expects inputs {sorted(acceptable)}; missing "
                + ", ".join(missing)
            )
    return payload


def _run_module(
    tree: PeerTree,
    node: PeerNode,
    inputs: Mapping[str, Any],
    *,
    seed: int | None,
) -> Any:
    """Call a leaf's module, falling back to the input tensor when it has no peer."""
    peer = node.peer
    if peer is None:
        value = inputs.get("state", inputs.get("inputs"))
        if value is None and len(inputs) == 1:
            value = next(iter(inputs.values()))
        if value is None:
            raise ContractError(
                f"node {node.path!r} has no peer and no 'state' input to pass through"
            )
        return value
    names = tuple(peer.adapter.contract.input_names)
    kwargs: dict[str, Any] = {}
    primary = None
    for name in names:
        if name in inputs:
            if primary is None:
                primary = inputs[name]
            elif name != "attention_mask":
                kwargs[name] = inputs[name]
        elif name == "attention_mask" and "attention_mask" in inputs:
            kwargs[name] = inputs["attention_mask"]
    if primary is None:
        primary = inputs.get("state")
    if primary is None:
        raise ContractError(
            f"node {node.path!r} expects inputs {list(names)}; none present in the request"
        )
    logits, _value = peer.adapter(primary, **kwargs)
    return logits


def _batching_decision(tree: PeerTree, paths: Sequence[str], inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Decide batched vs per-node for one level, with the reason (section 7.3)."""
    first = tree[paths[0]]
    return {
        "level": first.depth,
        "mode": "per_node",
        "reason": "modules are invoked individually so each node's contract is honoured",
        "first_path": paths[0],
        "paths": list(paths),
    }


def _batching_log(
    tree: PeerTree,
    results: Mapping[str, InferenceNodeResult],
    inputs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for depth, paths in sorted(tree.levels.items()):
        present = [path for path in paths if path in results]
        if len(present) <= 1:
            continue
        entries.append(
            {
                "level": depth,
                "mode": "per_node",
                "reason": "per-node calls preserve each node's input contract",
                "nodes": len(present),
            }
        )
    return entries


def _per_level_cost(tree: PeerTree, results: Mapping[str, InferenceNodeResult]) -> dict[str, float]:
    cost: dict[str, float] = {}
    for path, record in results.items():
        key = str(tree[path].depth)
        cost[key] = cost.get(key, 0.0) + record.duration
    return dict(sorted(cost.items(), key=lambda item: int(item[0])))


def _inference_scorer(
    scorer: HierarchicalScorer | ScoringPipeline | None,
    *,
    persist_warnings: bool,
) -> HierarchicalScorer:
    """Resolve the scorer for a request, honouring the persistence contract.

    ``persist_warnings=False`` (the default) always scores through a *new*
    pipeline, even when the caller supplied one: the registries are the only
    mutable state in scoring, so a fresh pipeline is what makes "this request
    cannot advance a training warning counter" true rather than aspirational
    (document 10 section 7.5).

    ``persist_warnings=True`` scores through the caller's pipeline when one was
    supplied, or through the process-wide :func:`inference_pipeline` otherwise, so
    production warnings accumulate in an inference namespace that training never
    reads.
    """
    if isinstance(scorer, HierarchicalScorer):
        source = scorer
    elif isinstance(scorer, ScoringPipeline):
        source = HierarchicalScorer(scorer)
    elif scorer is None:
        source = HierarchicalScorer(ScoringPipeline())
    else:
        raise ContractError(
            f"run_inference needs a HierarchicalScorer or ScoringPipeline, got "
            f"{type(scorer).__name__}"
        )
    if persist_warnings:
        return source if scorer is not None else HierarchicalScorer(inference_pipeline())
    return HierarchicalScorer(
        ScoringPipeline(
            score_config=source.pipeline.score_config,
            severity_weights=source.pipeline.severity_weights,
            warning_policy=source.pipeline.warning_policy,
        ),
        rollup=source.rollup,
        attribution=source.attribution,
        non_offsettable=source.non_offsettable,
    )


def _score_inference(
    tree: PeerTree,
    outputs: Mapping[str, Any],
    *,
    scorer: HierarchicalScorer | ScoringPipeline | None,
    task: "Task | None",
    roots: Sequence[str],
    persist_warnings: bool,
    per_node: Sequence[InferenceNodeResult] = (),
) -> EpisodeScorecard:
    """Emit rule events from the inferred outputs and score them.

    ``persist_warnings=False`` (the default) is implemented by scoring through a
    freshly built pipeline: the registries are new, so nothing a production
    request does can advance a training warning counter (document 10 section 7.5).

    A node that failed during the request is recorded as
    :data:`INFERENCE_FAILURE_RULE` evidence on its own ledger, so the failure
    reaches the scorecard through the penalty stage rather than being visible only
    in ``InferenceResult.per_node`` (document 10 section 9.2).
    """
    from .judge import TaskJudge

    judge = TaskJudge()
    failures = {record.path: record for record in per_node if not record.ok}
    ledgers: dict[str, Ledger] = {}
    targets = getattr(task, "targets", None) if task is not None else None
    for node in tree.walk("leaves-first"):
        path = node.path
        value = outputs.get(path)
        ledger = Ledger(peer=path)
        if targets is not None and torch.is_tensor(value):
            judge.judge(ledger, logits=value, targets=targets, peer=path)
        failed = failures.get(path)
        if failed is not None:
            ledger.record(
                INFERENCE_FAILURE_RULE,
                evidence=f"node failed during inference: {failed.error}",
                source="inference",
            )
        ledgers[path] = ledger

    hierarchical = _inference_scorer(scorer, persist_warnings=persist_warnings)
    card = hierarchical.score_tree(tree, ledgers, episode=0)
    card.metadata["inference"] = True
    card.metadata["persist_warnings"] = bool(persist_warnings)
    card.metadata["roots"] = list(roots)
    return card
