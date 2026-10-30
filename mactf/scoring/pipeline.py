"""The scoring pipeline: ordered stages, precedence, compliance, scorecard.

This module owns the ordering that makes scoring reproducible:

1. Validate and record rule events (``Ledger``).
2. Resolve precedence once, up front, so every stage reads the same decision.
3. Walk the stage chain by ascending ``order``: ``prepare``, ``score``,
   ``finalize``. Warnings run first because only they change what penalties see.
4. Compose ``ContinuousScore(reward, penalty, warning)``.
5. Suppress the reward when the controlling tier admits no reward rule.
6. Apply framework.md precedence and the non-offsettable floor.
7. Emit a ``PeerScorecard``.

Only this class builds a scorecard, so a custom stage cannot invent fields or
skip compliance.
"""

from __future__ import annotations

import copy
import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

from .. import signals
from ..compliance import (
    DEFAULT_MIN_FLOOR_LAMBDA,
    ComplianceResult,
    constrain,
    effective_non_offsettable,
    void_rule_ids,
)
from ..coopetition.pool import (
    CooperativeAllocation,
    CooperativePool,
    cooperative_total,
)
from ..exceptions import ComplianceGuardViolation, ImproperlyConfigured
from ..rules.catalog import ALL_RULES, get_rule
from ..rules.precedence import PrecedenceDecision, resolve
from ..rules.severity import Severity, weight_for
from .base import ScoreContext, ScorerOutput, ScoringKind
from .base import Scorer as _ScorerABC
from .continuous import (
    ContinuousScore,
    PenaltyBreakdown,
    RewardBreakdown,
)
from .continuous import score as continuous_score
from .ledger import Ledger
from .penalty import PenaltyScorer
from .reward_scorer import RewardScorer
from .scorecard import EpisodeScorecard, PeerScorecard, highest_severity, suppressed_from_rules
from .warning import EscalationOutcome, WarningPolicy, WarningRegistry
from .warning_scorer import DEFAULT_KNOWN_RULES, WarningScorer

__all__ = [
    "DEFAULT_KNOWN_RULES",
    "ScoringPipeline",
    "build_pool",
    "build_scorer",
    "build_stages",
    "migrate_stage_state",
    "severity_weight",
]

_log = logging.getLogger("mactf.scoring.pipeline")

#: Distinguishes "caller omitted ``pool``" from "caller passed ``None``".
#: Omitting it means "use the pipeline's configured pool"; passing ``None``
#: explicitly means "score this card without a split".
_USE_DEFAULT_POOL: Any = object()

#: The stage classes the framework ships. A performance shortcut only: the guard's own
#: ``_trusted`` applies the same rule, and ``COMPLIANCE.TRUST_BUILTIN = false`` makes
#: ``guard.enforce`` inspect them too (the structural checks stay on for custom stages).
_BUILTIN_STAGE_TYPES: tuple[type, ...] = (WarningScorer, RewardScorer, PenaltyScorer)

def _is_builtin_stage(stage: Any) -> bool:
    """Exact-type identity test. Never ``type(x) in (...)``: that calls ``==`` on the class,
    which a metaclass can override to impersonate a built-in and skip the guard."""
    return any(type(stage) is built for built in _BUILTIN_STAGE_TYPES)


#: The per-node stage kinds ``SCORING.STAGES`` may name, in chain order.
STAGE_KINDS: tuple[str, ...] = ("warning", "reward", "penalty")


class ScoringPipeline(_ScorerABC):
    """The default composite: warning, reward, penalty, then compliance."""

    kind = ScoringKind.PIPELINE
    order = 0

    REQUIRED_KINDS = (ScoringKind.WARNING, ScoringKind.REWARD, ScoringKind.PENALTY)

    def __init__(
        self,
        scorers: Sequence[Scorer] | None = None,
        *,
        score_config: Mapping[str, Any] | None = None,
        severity_weights: Mapping[str, Any] | None = None,
        warning_policy: WarningPolicy | None = None,
        registries: dict[str, WarningRegistry] | None = None,
        pool: CooperativePool | None = None,
        engine_config: Mapping[str, Any] | None = None,
        points_engine: Any = None,
        guard: Any = None,
    ) -> None:
        super().__init__(config=score_config or {})
        self.engine_config: dict[str, Any] = dict(engine_config or {})
        self._points_engine: Any = points_engine
        self.score_config: dict[str, Any] = dict(score_config or {})
        self.severity_weights: dict[str, Any] = dict(severity_weights or {})
        self.warning_policy: WarningPolicy = warning_policy or WarningPolicy.from_settings(
            dict(self.score_config.get("WARNINGS", {}))
        )
        self.pool = pool
        self._chain: list[Scorer] = []
        #: The context after every stage's ``prepare`` ran, set by ``_walk``.
        self._last_context: ScoreContext | None = None
        if scorers is None:
            for stage in build_stages(
                score_config=self.score_config,
                severity_weights=self.severity_weights,
                warning_policy=self.warning_policy,
                registries=registries,
            ):
                self._chain.append(stage)
        else:
            for stage in scorers:
                self._add_unchecked(stage)
        self._guard: Any = guard
        #: What the guard did for the most recent ``_walk`` (set even when clean).
        self._last_guard_report: Any = None
        self._validate_chain()

    # -- chain management ------------------------------------------------
    def _add_unchecked(self, scorer: Scorer) -> None:
        if not isinstance(scorer, _ScorerABC):
            raise ImproperlyConfigured(
                f"a scoring stage must be a Scorer subclass, got {type(scorer).__name__}"
            )
        self._chain.append(scorer)

    @property
    def chain(self) -> tuple[Scorer, ...]:
        return tuple(sorted(self._chain, key=lambda stage: stage.order))

    def add(self, scorer: Scorer) -> "ScoringPipeline":
        """Add a stage, validating the resulting chain."""
        self._add_unchecked(scorer)
        try:
            self._validate_chain()
        except ImproperlyConfigured:
            self._chain.remove(scorer)
            raise
        return self

    def remove(self, kind: ScoringKind) -> "ScoringPipeline":
        """Remove the stage of a given kind, validating the result."""
        matches = [stage for stage in self._chain if stage.kind == kind]
        if not matches:
            raise ImproperlyConfigured(f"no stage of kind {kind.value!r} to remove")
        if len(matches) > 1:
            raise ImproperlyConfigured(
                f"more than one stage of kind {kind.value!r}; remove by construction instead"
            )
        self._chain.remove(matches[0])
        try:
            self._validate_chain()
        except ImproperlyConfigured:
            self._chain.append(matches[0])
            raise
        return self

    def get(self, kind: ScoringKind) -> Scorer:
        for stage in self._chain:
            if stage.kind == kind:
                return stage
        raise ImproperlyConfigured(f"no stage of kind {kind.value!r} in the pipeline")

    def _validate_chain(self) -> None:
        if not self._chain:
            raise ImproperlyConfigured("a scoring pipeline needs at least one stage")
        by_kind: dict[ScoringKind, Scorer] = {}
        for stage in self._chain:
            if stage.order < 0:
                raise ImproperlyConfigured(
                    f"{type(stage).__name__} declares a negative order ({stage.order})"
                )
            if stage.kind == ScoringKind.HIERARCHY:
                # A hierarchy stage scores a whole tree; nesting it inside a
                # per-node chain would run the tree once per node. It belongs
                # outermost, driven by ``HierarchicalScorer.score_tree``.
                raise ImproperlyConfigured(
                    f"{type(stage).__name__} cannot be a stage of a per-node "
                    "ScoringPipeline; build a HierarchicalScorer around the pipeline"
                )
            if stage.kind == ScoringKind.PIPELINE:
                raise ImproperlyConfigured(
                    f"{type(stage).__name__} is a ScoringPipeline; a pipeline cannot be a "
                    "stage of another"
                )
            existing = by_kind.get(stage.kind)
            if existing is not None:
                raise ImproperlyConfigured(
                    f"two scorers with kind {stage.kind.value!r}: "
                    f"{type(existing).__name__} and {type(stage).__name__}"
                )
            by_kind[stage.kind] = stage
        orders: dict[int, Scorer] = {}
        for stage in self._chain:
            clash = orders.get(stage.order)
            if clash is not None:
                raise ImproperlyConfigured(
                    f"{type(stage).__name__} and {type(clash).__name__} both declare "
                    f"order={stage.order}; set an explicit order on one of them"
                )
            orders[stage.order] = stage
        missing = [kind.value for kind in self.REQUIRED_KINDS if kind not in by_kind]
        if missing:
            raise ImproperlyConfigured(
                "scoring pipeline is missing required stage(s): " + ", ".join(sorted(missing))
            )
        # The ordering hazard (doc 07 section 6): if warnings were scored after
        # penalties, a rule that matured this episode would be excluded from ``P``
        # as warning-only *and* counted as a penalty, or vice versa. A custom
        # stage that only shifts ``order`` is the easy way to break this, so the
        # invariant is checked rather than assumed.
        warning = by_kind.get(ScoringKind.WARNING)
        if warning is not None:
            early = [
                stage
                for kind in (ScoringKind.REWARD, ScoringKind.PENALTY)
                for stage in (by_kind.get(kind),)
                if stage is not None and stage.order <= warning.order
            ]
            if early:
                raise ImproperlyConfigured(
                    f"{type(warning).__name__} declares order={warning.order} but "
                    f"{type(early[0]).__name__} declares order={early[0].order}; the "
                    "warning stage must run before reward and penalty so an escalated "
                    "rule is neither excluded from P nor counted twice"
                )

    @property
    def guard(self) -> Any:
        """The compliance guard, built lazily so importing the pipeline never imports it."""
        if self._guard is None:
            from ..components.guards.compliance import ComplianceGuard, GuardPolicy

            self._guard = ComplianceGuard(
                GuardPolicy(
                    min_floor_lambda=float(
                        self.score_config.get("MIN_FLOOR_LAMBDA", DEFAULT_MIN_FLOOR_LAMBDA)
                    )
                )
            )
        return self._guard

    @guard.setter
    def guard(self, value: Any) -> None:
        self._guard = value

    def replace(
        self, scorer: Scorer, *, migrate: Callable[[Scorer, Scorer], None] | None = None
    ) -> Scorer:
        """Swap the stage of ``scorer.kind`` for ``scorer`` and return the old one.

        Atomic: ``migrate(old, new)`` runs first and may only write to ``new``; the chain is
        then validated, and a failure restores the old stage, so a rejected swap leaves the
        pipeline exactly as it was.
        """
        if not isinstance(scorer, _ScorerABC):
            raise ImproperlyConfigured(
                f"a scoring stage must be a Scorer subclass, got {type(scorer).__name__}"
            )
        old = self.get(scorer.kind)
        index = self._chain.index(old)
        if migrate is not None:
            migrate(old, scorer)
        self._chain[index] = scorer
        try:
            self._validate_chain()
        except ImproperlyConfigured:
            self._chain[index] = old
            raise
        return old

    def stage_inject(self, *, trusted: bool = True) -> dict[str, Any]:
        """The offer ``SlotTable.swap`` makes to a stage factory (see ``build_stages``)."""
        return {
            "config": dict(self.score_config),
            "policy": self.warning_policy,
            "trusted": trusted,
        }

    def sync_from_slots(self, slots: Any) -> list[ScoringKind]:
        """Make the chain match ``scoring.<kind>`` slots. Call at an episode boundary,
        after ``slots.apply_pending(Boundary.EPISODE, inject=self.stage_inject())``."""
        changed: list[ScoringKind] = []
        for kind in self.REQUIRED_KINDS:
            stage = slots.get(f"scoring.{kind.value}").instance
            if getattr(stage, "kind", None) is not kind:
                raise ImproperlyConfigured(
                    f"slot scoring.{kind.value} holds {type(stage).__name__}, whose kind is "
                    f"{getattr(getattr(stage, 'kind', None), 'value', None)!r}, not {kind.value!r}"
                )
            if stage is not self.get(kind):
                self.replace(stage, migrate=migrate_stage_state)
                changed.append(kind)
        return changed

    # -- registry compatibility -----------------------------------------
    @property
    def warning_scorer(self) -> WarningScorer:
        stage = self.get(ScoringKind.WARNING)
        assert isinstance(stage, WarningScorer)
        return stage

    @property
    def registries(self) -> dict[str, WarningRegistry]:
        """The warning stage's registries, keyed by ``node_path::peer``.

        Kept as a property so existing code that treats ``scorer.registries`` as
        a dict of peer -> registry keeps working: the warning stage only ever
        writes through :meth:`WarningScorer.registry_for`.
        """
        return self.warning_scorer.registries

    def registry_for(self, peer: str) -> WarningRegistry:
        return self.warning_scorer.registry_for(peer)

    def begin_episode(self, peer: str, *, episode: int) -> list[Any]:
        return self.warning_scorer.begin_episode(peer, episode=episode)

    # -- context ---------------------------------------------------------
    def _severity_table(self) -> dict[str, Any]:
        """The reward/penalty severity magnitudes, exactly as ``continuous.py`` resolves them.

        ``continuous.score`` does ``config.setdefault("SEVERITY_WEIGHTS", ...)``,
        so a ``SEVERITY_WEIGHTS`` entry inside ``SCORING`` wins over
        ``SEVERITY.WEIGHTS``. Mirroring that here is what keeps the two paths
        numerically identical.
        """
        if "SEVERITY_WEIGHTS" in self.score_config:
            return dict(self.score_config["SEVERITY_WEIGHTS"] or {})
        return dict(self.severity_weights)

    def _base_context(
        self,
        ledger: Ledger,
        *,
        peer: str,
        episode: int,
        known_rules: set[str] | None,
        node_path: str,
        depth: int,
        branch_void: frozenset[str],
        children: tuple[ScorerOutput, ...],
    ) -> ScoreContext:
        return ScoreContext(
            peer=peer,
            episode=episode,
            ledger=ledger,
            node_path=node_path,
            depth=depth,
            config=dict(self.score_config),
            severity_weights=self._severity_table(),
            warning_policy=self.warning_policy,
            known_rules=frozenset(known_rules or DEFAULT_KNOWN_RULES),
            branch_void=frozenset(branch_void),
            children=tuple(children),
        )

    # -- scoring ---------------------------------------------------------
    def score_peer(
        self,
        ledger: Ledger,
        *,
        peer: str | None = None,
        episode: int = 0,
        known_rules: set[str] | None = None,
        node_path: str = "",
        depth: int = 0,
        branch_void: frozenset[str] = frozenset(),
        children: tuple[ScorerOutput, ...] = (),
        pool: CooperativePool | None = _USE_DEFAULT_POOL,
        allocation: CooperativeAllocation | None = None,
        evidence: Any = None,
    ) -> PeerScorecard:
        peer = peer or ledger.peer or "peer"
        ledger.peer = peer
        active_pool = self.pool if pool is _USE_DEFAULT_POOL else pool

        ctx = self._base_context(
            ledger,
            peer=peer,
            episode=episode,
            known_rules=known_rules,
            node_path=node_path,
            depth=depth,
            branch_void=branch_void,
            children=children,
        )
        # Precedence is resolved once, before the walk, so a stage cannot see a
        # different decision than its neighbours (doc 07 section 7, step 3).
        decision = resolve(
            ledger.rules(),
            mode=str(self.score_config.get("PRECEDENCE_MODE", "strict")),
        )
        ctx = ctx.evolve(
            extras={
                **dict(ctx.extras),
                "allowed_rules": frozenset(rule.id for rule in decision.applied),
                "precedence": decision,
            }
        )

        outputs = self._walk(ctx)
        # ``_walk`` threads a new context through ``prepare``; the warning stage's
        # mutations and escalation set live only in *that* context. Resuming from
        # the caller's un-prepared one would drop them (and silently un-exclude
        # rules from the penalty total), so re-read the prepared context here.
        ctx = self._last_context

        continuous = self._compose(ctx, outputs, decision)

        non_offsettable = effective_non_offsettable(self.score_config.get("NON_OFFSETTABLE"))
        result: ComplianceResult = constrain(
            continuous,
            non_offsettable=non_offsettable,
            branch_void=branch_void,
        )

        suppressed = ()
        if continuous.decision is not None:
            suppressed = suppressed_from_rules(
                continuous.decision.suppressed,
                f"not in controlling tier T{continuous.decision.controlling_tier}",
            )
            already = {rule.id for rule in continuous.decision.suppressed}
            suppressed += suppressed_from_rules(
                tuple(
                    get_rule(rule_id)
                    for rule_id in continuous.suppressed_rewards
                    if rule_id not in already
                ),
                "reward suppressed by precedence",
            )

        void_rules = tuple(
            sorted(
                set(
                    void_rule_ids(continuous, non_offsettable=non_offsettable)
                )
                | set(branch_void)
            )
        )
        card = PeerScorecard(
            peer=peer,
            episode=episode,
            continuous=continuous,
            final_score=result.value,
            floor=result.floor,
            floor_reason=result.reason,
            warnings_issued=tuple(self._mutations(ctx, "issue")),
            warnings_escalated=tuple(self._mutations(ctx, "escalate")),
            suppressed=suppressed,
            violated_non_offsettable=void_rules,
            severity=highest_severity(ledger),
            metadata=self._card_metadata(result),
            branch_void=tuple(sorted(branch_void)),
        )

        self._attach_points(
            card,
            ledger,
            ctx,
            continuous,
            evidence=evidence,
            void_rules=void_rules,
        )

        if active_pool is not None:
            if allocation is None:
                allocation = active_pool.allocate({peer: cooperative_total(continuous.reward)})
            card.competitive, card.cooperative = active_pool.split(card, allocation)
            card.cooperative_denied = allocation.denied_for(peer)
            card.metadata["cooperative_pool"] = allocation.as_dict()
        signals.episode_peer_scored.send(sender=self, card=card, ledger=ledger)
        return card

    @property
    def points_engine(self) -> Any:
        """The doc 11 points engine, built lazily from ``SCORING.POINTS`` and ``ENGINE``."""
        if self._points_engine is None:
            from .guards import PointsEngine

            self._points_engine = PointsEngine.from_settings(
                {
                    "SCORING": dict(self.score_config),
                    "ENGINE": dict(self.engine_config),
                }
            )
        return self._points_engine

    @points_engine.setter
    def points_engine(self, engine: Any) -> None:
        self._points_engine = engine

    @property
    def points_table(self) -> Any:
        """The :class:`PointsTable` in force for this pipeline's points engine."""
        return self.points_engine.table

    def _attach_points(
        self,
        card: PeerScorecard,
        ledger: Ledger,
        ctx: ScoreContext,
        continuous: Any,
        *,
        evidence: Any,
        void_rules: tuple[str, ...],
    ) -> None:
        """Derive the 0-100 points breakdown beside ``final_score`` (doc 11).

        ``final_score`` is untouched: points are a second, auditable reading of the
        same ledger, and the two may legitimately disagree.
        """
        from .evidence import EpisodeEvidence

        if evidence is None and self._points_engine is None:
            return  # not computed: rewards cannot be judged without evidence

        if evidence is not None and not isinstance(evidence, EpisodeEvidence):
            evidence = (
                EpisodeEvidence.from_dict(evidence)
                if isinstance(evidence, Mapping)
                else EpisodeEvidence.from_subject(evidence)
            )
        mutations = tuple(ctx.extras.get("warning_mutations", ()))
        escalations = {
            item.rule_id: item.severity
            for item in mutations
            if item.action == "escalate" and item.severity is not None
        }
        active: dict[str, list[Severity]] = {}
        for item in mutations:
            if item.action == "issue" and item.severity is not None:
                active.setdefault(item.rule_id, []).append(item.severity)
        allowed = ctx.extras.get("allowed_rules")
        breakdown = self.points_engine.score(
            ledger,
            evidence=evidence,
            peer=card.peer,
            episode=card.episode,
            void_rules=void_rules,
            warning_only=set(ctx.warning_events) - set(ctx.escalated_rules),
            escalations=escalations,
            allowed_rules=allowed,
            suppressed_rewards=getattr(continuous, "suppressed_rewards", ()),
            active_warnings=active,
        )
        card.points = breakdown.as_dict()
        card.metadata["points"] = dict(card.points)
        card.metadata["points_table_version"] = breakdown.table_version
        if evidence is not None:
            card.metadata["evidence_digest"] = evidence.digest()

    def _walk(self, ctx: ScoreContext) -> dict[ScoringKind, ScorerOutput]:
        """Run every stage in ascending order, threading the context through.

        Stages see a read-only ledger. ``prepare`` may change only the fields its kind
        owns. After the walk the guard checks every non-built-in output (6.4).
        """
        guard = self.guard
        self._last_guard_report = None
        source = ctx.ledger.unwrap() if hasattr(ctx.ledger, "unwrap") else ctx.ledger
        snapshot = guard.snapshot(source)
        ctx = ctx.evolve(ledger=guard.view(source))
        chain = self.chain
        custom = any(not _is_builtin_stage(stage) for stage in chain)
        if custom:
            # A stage that edits ``ctx.config`` in place must not edit the pipeline's own
            # (the context holds a shallow copy of it): hand custom stages a private copy.
            ctx = ctx.evolve(
                config=copy.deepcopy(dict(ctx.config)),
                severity_weights=copy.deepcopy(dict(ctx.severity_weights)),
            )
        pristine = guard.pristine(ctx) if custom else None
        structural: list[Any] = []
        outputs: dict[ScoringKind, ScorerOutput] = {}
        for stage in chain:
            is_custom = not _is_builtin_stage(stage)
            try:
                prepared = stage.prepare(ctx)
                if not isinstance(prepared, ScoreContext):
                    raise ImproperlyConfigured(
                        f"{type(stage).__name__}.prepare() returned "
                        f"{type(prepared).__name__}, not a ScoreContext"
                    )
                if is_custom:
                    structural.extend(guard.check_prepare(stage, ctx, prepared, source))
                ctx = prepared
                output = stage.score(ctx)
                output = stage.finalize(ctx, output)
                if not isinstance(output, ScorerOutput):
                    raise ImproperlyConfigured(
                        f"{type(stage).__name__}.score() returned "
                        f"{type(output).__name__}, not a ScorerOutput"
                    )
            except Exception as error:
                if is_custom:
                    # A stage that edited the ledger and then raised must not leave the
                    # edit behind: report the tampering (and undo it) before the error.
                    tampered = guard.check_after_stage(stage, source, snapshot)
                    if tampered:
                        raise ComplianceGuardViolation(
                            [item.as_dict() for item in tampered]
                        ) from error
                raise
            if is_custom:
                structural.extend(guard.check_after_stage(stage, source, snapshot))
                structural.extend(guard.check_pristine(stage, ctx, pristine))
            outputs[stage.kind] = output
        if structural:
            raise ComplianceGuardViolation([item.as_dict() for item in structural])
        outputs, report = guard.enforce(ctx, outputs, chain, pristine)
        self._last_guard_report = report
        # Stash the prepared context so callers and the pipeline can read it.
        self._last_context = ctx
        return outputs

    def _card_metadata(self, result: ComplianceResult) -> dict[str, Any]:
        metadata: dict[str, Any] = {"compliance": result.as_dict()}
        report = self._last_guard_report
        if report is not None and report.violations:
            metadata["compliance_guard"] = report.as_dict()
        return metadata

    def _mutations(self, ctx: ScoreContext, action: str) -> list[Any]:
        mutations = ctx.extras.get("warning_mutations", ())
        return [item for item in mutations if getattr(item, "action", None) == action]

    def _compose(
        self,
        ctx: ScoreContext,
        outputs: Mapping[ScoringKind, ScorerOutput],
        decision: PrecedenceDecision,
    ) -> ContinuousScore:
        """Turn the three stage outputs into a ``ContinuousScore``.

        Reward suppression lives here, not in ``RewardScorer``, so a custom
        reward stage cannot bypass tier precedence.
        """
        reward_output = outputs[ScoringKind.REWARD]
        penalty_output = outputs[ScoringKind.PENALTY]
        warning_output = outputs[ScoringKind.WARNING]

        reward_breakdown = reward_output.breakdown
        reward = RewardBreakdown(
            r_task=float(reward_breakdown.get("r_task", 0.0)),
            r_epi=float(reward_breakdown.get("r_epi", 0.0)),
            r_collab=float(reward_breakdown.get("r_collab", 0.0)),
            r_align=float(reward_breakdown.get("r_align", 0.0)),
            weights=dict(reward_breakdown.get("weights", {})),
            rules=tuple(reward_output.rules),
            by_rule=dict(reward_breakdown.get("by_rule", {})),
        )

        penalty_breakdown = penalty_output.breakdown
        penalty = PenaltyBreakdown(
            amounts=dict(penalty_breakdown.get("amounts", {})),
            omega=float(penalty_breakdown.get("omega", 1.0)),
            omega_rules=tuple(penalty_breakdown.get("omega_rules", ())),
            rules=tuple(penalty_breakdown.get("rules", ())),
        )

        if decision.mode == "strict" and not any(rule.is_reward for rule in decision.applied):
            suppressed = reward.rules
            # Zero the reward but keep the weights, exactly as continuous.py:217-219 does.
            reward = RewardBreakdown(weights=reward.weights)
        else:
            allowed = {rule.id for rule in decision.applied}
            suppressed = tuple(
                event.rule_id for event in ctx.ledger.rewards() if event.rule_id not in allowed
            )

        return ContinuousScore(
            reward=reward,
            penalty=penalty,
            warning=float(warning_output.amount),
            warning_events=tuple(ctx.warning_events),
            decision=decision,
            suppressed_rewards=tuple(dict.fromkeys(suppressed)),
        )

    def score(self, ctx: ScoreContext) -> ScorerOutput:
        """Run the chain for an existing context and report it as one output.

        ``score_peer`` is the entry point that builds a scorecard; this method
        exists because the pipeline is itself a stage, so it can be composed
        into a :class:`~mactf.scoring.hierarchy.HierarchicalScorer` or inspected
        stage by stage.
        """
        decision = resolve(
            ctx.ledger.rules(),
            mode=str(ctx.config.get("PRECEDENCE_MODE", "strict")),
        )
        prepared = ctx.evolve(
            extras={
                **dict(ctx.extras),
                "allowed_rules": frozenset(rule.id for rule in decision.applied),
                "precedence": decision,
            }
        )
        outputs = self._walk(prepared)
        continuous = self._compose(self._last_context, outputs, decision)
        return ScorerOutput(
            kind=self.kind,
            amount=continuous.net,
            breakdown={
                "reward": continuous.reward.as_dict(),
                "penalty": continuous.penalty.as_dict(),
                "warning": continuous.warning,
                "net": continuous.net,
                "stages": {kind.value: out.as_dict() for kind, out in outputs.items()},
            },
            rules=tuple(
                dict.fromkeys(
                    rule_id
                    for output in outputs.values()
                    for rule_id in output.rules
                )
            ),
            metadata={"precedence": decision.describe()},
            void_rules=tuple(
                sorted(
                    set(outputs[ScoringKind.PENALTY].void_rules)
                )
            ),
        )

    def score_episode(
        self,
        ledgers: dict[str, Ledger],
        *,
        episode: int = 0,
        known_rules: set[str] | None = None,
        pool: CooperativePool | None = None,
        evidence: Mapping[str, Any] | None = None,
    ) -> EpisodeScorecard:
        card = EpisodeScorecard(episode=episode)
        active_pool = self.pool if pool is None else pool
        allocation = None
        if active_pool is not None and ledgers:
            # One allocation for the whole cohort: the budget is per-episode and
            # shared, so scoring peers one at a time would let the first peer
            # take budget the later ones need (framework.md 10.1.10).
            allocation = active_pool.allocate(
                {name: self._cooperative_request(ledger) for name, ledger in ledgers.items()}
            )
        for peer, ledger in ledgers.items():
            card.add(
                self.score_peer(
                    ledger,
                    peer=peer,
                    episode=episode,
                    known_rules=known_rules,
                    pool=active_pool,
                    allocation=allocation,
                    evidence=(evidence or {}).get(peer),
                )
            )
        signals.episode_scored.send(sender=self, scorecard=card)
        return card

    def _cooperative_request(self, ledger: Ledger) -> float:
        """The cooperative reward a ledger would earn, warnings aside.

        Warnings scale penalties only, so omitting them here cannot change the
        reward total; the precedence decision sees the same rule set, so the
        request matches what ``score_peer`` will compute.
        """
        preview = continuous_score(
            ledger,
            score_config=self.score_config,
            severity_weights=self.severity_weights,
        )
        return cooperative_total(preview.reward)

    # -- introspection ---------------------------------------------------
    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload["chain"] = [stage.describe() for stage in self.chain]
        return payload

    def state_dict(self) -> dict[str, Any]:
        return {
            "class": type(self).__name__,
            "config": dict(self.score_config),
            "severity_weights": dict(self.severity_weights),
            "stages": {
                stage.kind.value: stage.state_dict() for stage in self.chain
            },
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if payload.get("class") not in (None, type(self).__name__):
            raise ValueError(
                f"{type(self).__name__} cannot load state written by {payload.get('class')!r}"
            )
        self.score_config = dict(payload.get("config", self.score_config))
        self.severity_weights = dict(
            payload.get("severity_weights", self.severity_weights)
        )
        stages = payload.get("stages") or {}
        for stage in self.chain:
            state = stages.get(stage.kind.value)
            if state:
                stage.load_state_dict(state)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def build_stages(
    *,
    score_config: Mapping[str, Any] | None = None,
    severity_weights: Mapping[str, Any] | None = None,
    warning_policy: WarningPolicy | None = None,
    registries: dict[str, WarningRegistry] | None = None,
    specs: Mapping[str, Any] | None = None,
    slots: Any = None,
    trusted: bool = True,
) -> list[Scorer]:
    """The three stages, in the order document 07 mandates.

    ``specs`` is ``SCORING.STAGES``. When it is empty the three built-ins are built
    directly, exactly as before. When it is not, each stage is installed into
    ``slots`` (a ``SlotTable``) under ``scoring.<kind>``, so it can be swapped and
    fingerprinted later; ``trusted`` is false for stages named by an untrusted
    surface (the API), which refuses ``import:`` specs outside the allow-list.
    """
    config = dict(score_config or {})
    policy = warning_policy or WarningPolicy.from_settings(
        dict(config.get("WARNINGS", {}))
    )
    if not specs:
        return [
            WarningScorer(config=config, policy=policy, registries=registries),
            RewardScorer(config=config),
            PenaltyScorer(config=config),
        ]

    unknown = sorted(set(specs) - set(STAGE_KINDS))
    if unknown:
        raise ImproperlyConfigured(
            f"SCORING.STAGES has unknown kind(s) {unknown}; per-node stages are "
            f"{list(STAGE_KINDS)} (a hierarchy is selected by HIERARCHY.ROLLUP)"
        )
    if slots is None:
        from ..components.resolver import ComponentResolver
        from ..components.slots import SlotTable

        slots = SlotTable(ComponentResolver(), trusted=trusted)
    inject = {
        "config": config,
        "policy": policy,
        "registries": registries,
        "trusted": trusted,
    }
    stages: list[Scorer] = []
    for kind in STAGE_KINDS:
        slot = slots.install(f"scoring.{kind}", "scorer", specs.get(kind, kind), inject=inject)
        stage = slot.instance
        if stage.kind.value != kind:
            raise ImproperlyConfigured(
                f"SCORING.STAGES.{kind} resolves to {type(stage).__name__}, whose kind is "
                f"{stage.kind.value!r}, not {kind!r}"
            )
        stages.append(stage)
    return stages


def migrate_stage_state(old: Scorer, new: Scorer) -> None:
    """Carry state from ``old`` to ``new`` for a same-kind swap. Writes only to ``new``.

    Only the warning stage has state worth keeping: its registries hold every peer's
    warnings, clean streaks and expiry, which framework.md W1-W9 require to survive.
    The reward and penalty stages are stateless, so for them this is a kind check.
    """
    from ..exceptions import ComponentContractError

    if old.kind is not new.kind:
        raise ComponentContractError(
            "scorer", type(new).__name__,
            [f"cannot replace a {old.kind.value} stage with a {new.kind.value} stage"],
        )
    if old.kind is ScoringKind.WARNING:
        registries = getattr(old, "registries", None)
        if registries is None or not hasattr(new, "registries"):
            raise ComponentContractError(
                "scorer", type(new).__name__,
                ["a replacement WARNING stage exposes `registries` so warning state survives"],
            )
        new.registries = registries  # the live dict moves; ``new`` owns it from now on
        if hasattr(old, "policy") and hasattr(new, "policy"):
            new.policy = old.policy


def build_scorer(
    settings: Any,
    *,
    scorers: Sequence[Scorer] | None = None,
    slots: Any = None,
    trusted: bool = True,
) -> ScoringPipeline:
    """Construct a pipeline. With no ``scorers``, the three defaults are used.

    The cooperative pool is enabled whenever ``COOPETITION.COOPERATIVE_POOL`` is
    set and a budget is configured; when it is disabled the scorecard's
    ``competitive``/``cooperative`` fields stay ``None``, which reports "not
    computed" rather than a misleading zero.

    ``SCORING.STAGES`` (document 17 section 6.2) selects stages by spec; ``slots`` and
    ``trusted`` are handed to :func:`build_stages` for it. ``COMPLIANCE`` configures the
    guard every pipeline carries.
    """
    from ..components.guards.compliance import ComplianceGuard, GuardPolicy

    policy = WarningPolicy.from_settings(settings["WARNINGS"])
    score_config = dict(settings["SCORING"])
    guard_policy = GuardPolicy.from_settings(settings)
    if guard_policy.min_floor_lambda != DEFAULT_MIN_FLOOR_LAMBDA:
        # Recorded in the pipeline's config (and so its state dict) on purpose: the floor a
        # run was scored under is part of the run's identity. The default leaves it out,
        # which keeps default checkpoints byte-identical.
        score_config["MIN_FLOOR_LAMBDA"] = guard_policy.min_floor_lambda
    if guard_policy.mode == "fallback":
        _log.warning(
            "W003 COMPLIANCE.GUARD_MODE is 'fallback': a stage that breaks the compliance "
            "contract is replaced by the reference stage instead of failing the run"
        )
    severity_weights = dict(settings["SEVERITY"]["WEIGHTS"])
    pool = build_pool(settings)
    stage_specs = dict(score_config.get("STAGES") or {})
    stages = scorers
    if stages is None and stage_specs:
        stages = build_stages(
            score_config=score_config,
            severity_weights=severity_weights,
            warning_policy=policy,
            specs=stage_specs,
            slots=slots,
            trusted=trusted,
        )
    return ScoringPipeline(
        stages,
        score_config=score_config,
        severity_weights=severity_weights,
        warning_policy=policy,
        pool=pool,
        engine_config=dict(settings["ENGINE"]) if "ENGINE" in settings else {},
        guard=ComplianceGuard(guard_policy),
    )


def build_pool(settings: Any) -> CooperativePool | None:
    """Construct the configured cooperative pool, or ``None`` when disabled."""
    coop = settings["COOPETITION"] if "COOPETITION" in settings else {}
    if not bool(coop.get("COOPERATIVE_POOL", True)):
        return None
    from ..coopetition.pool import CooperativeBudget

    return CooperativePool(CooperativeBudget.from_settings(coop.get("COOPERATIVE_BUDGET")))


def severity_weight(severity: Severity, weights: dict[str, Any]) -> float:
    """Public helper so callers do not reach into the severity module."""
    return weight_for(severity, weights)


# ---------------------------------------------------------------------------
# Compatibility shim (doc 07 section 7.3)
# ---------------------------------------------------------------------------
# ``mactf.scoring.pipeline.Scorer`` used to be the concrete scorer. It now means
# the abstract base class, exported from ``mactf.scoring``. Code that still
# imports it from here gets the pipeline back with a DeprecationWarning, so a
# downgrade is impossible and the rename is visible rather than silent.
if TYPE_CHECKING:  # static checkers see the real class
    Scorer = ScoringPipeline  # type: ignore[misc,assignment]  # noqa: F811


def __getattr__(name: str) -> Any:
    if name == "Scorer":
        warnings.warn(
            "mactf.scoring.pipeline.Scorer now means the abstract base class; "
            "import ScoringPipeline, or from mactf.scoring import Scorer for the ABC",
            DeprecationWarning,
            stacklevel=2,
        )
        return ScoringPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
