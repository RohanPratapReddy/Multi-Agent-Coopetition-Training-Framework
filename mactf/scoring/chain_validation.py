"""Collect every problem in a stage chain without raising (document 17 section 6.7).

``ScoringPipeline._validate_chain`` raises on the first problem. A caller assembling a chain
from user input (the API, ``mactf components validate``) wants *all* the problems, each with
a code and a hint. :func:`validate_stage_chain` collects them with the same rules, in the
order ``_validate_chain`` meets them, so the first error reported is the one the pipeline
would raise (``test_chain_validation_agrees``); it then adds what the pipeline does not check
(each stage's own ``validate``, unknown ``requires`` keys, the lambda floor).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..compliance import DEFAULT_MIN_FLOOR_LAMBDA, effective_non_offsettable
from .base import Scorer, ScorerDiagnostic, ScoringKind

#: Context keys a stage may name in ``requires``: the ``ScoreContext`` fields plus the
#: hierarchical inputs. Anything else is a typo or an input nobody supplies.
PROVIDED_CONTEXT_KEYS: frozenset[str] = frozenset({
    "ledger", "config", "severity_weights", "warning_policy", "known_rules",
    "warning_total", "warning_events", "escalated_rules", "children", "branch_void",
    "extras", "allowed_rules", "tree", "ledgers", "peer", "episode", "node_path", "depth",
})

REQUIRED_KINDS = (ScoringKind.WARNING, ScoringKind.REWARD, ScoringKind.PENALTY)


def _error(code: str, message: str, hint: str = "") -> ScorerDiagnostic:
    return ScorerDiagnostic(level="error", code=code, message=message, hint=hint)


def validate_stage_chain(
    stages: Sequence[Scorer], settings: Mapping[str, Any] | None = None
) -> list[ScorerDiagnostic]:
    """Every problem in ``stages`` under ``settings``, in a stable order.

    Structure first (the order ``_validate_chain`` checks it), then per-stage diagnostics
    and the lambda floor.
    """
    settings = dict(settings or {})
    found: list[ScorerDiagnostic] = []
    by_kind: dict[ScoringKind, list[Scorer]] = {}
    real: list[Scorer] = []

    # 0. The pipeline refuses a non-Scorer while it adds stages, before it validates any.
    for stage in stages:
        if isinstance(stage, Scorer):
            real.append(stage)
        else:
            found.append(_error("chain.not_a_scorer", f"{type(stage).__name__} is not a Scorer"))

    # 1. Per stage, in chain order: the checks ``_validate_chain`` makes while it walks.
    for stage in real:
        if stage.order < 0:
            found.append(_error(
                "chain.negative_order", f"{type(stage).__name__} declares order={stage.order}"))
        if stage.kind is ScoringKind.HIERARCHY:
            found.append(_error(
                "chain.hierarchy_nested",
                f"{type(stage).__name__} scores a whole tree and cannot be a per-node stage",
                "build a HierarchicalScorer around the pipeline"))
        if stage.kind is ScoringKind.PIPELINE:
            found.append(_error(
                "chain.pipeline_nested", "a ScoringPipeline cannot be a stage of another"))
        earlier = by_kind.get(stage.kind)
        if earlier:
            found.append(_error(
                "chain.duplicate_kind",
                f"two stages of kind {stage.kind.value!r}: "
                f"{type(earlier[0]).__name__} and {type(stage).__name__}"))
        by_kind.setdefault(stage.kind, []).append(stage)

    # 2. Order collisions, then missing kinds, then the warning-first ordering.
    orders: dict[int, Scorer] = {}
    for stage in real:
        clash = orders.get(stage.order)
        if clash is not None:
            found.append(_error(
                "chain.order_collision",
                f"{type(stage).__name__} and {type(clash).__name__} both declare order={stage.order}",
                "set an explicit order on one of them"))
        orders[stage.order] = stage
    for kind in REQUIRED_KINDS:
        if kind not in by_kind:
            found.append(_error(
                "chain.missing_stage", f"no {kind.value} stage",
                "use 'zero_reward' or 'no_warnings' to turn a kind off on purpose"))
    warning = (by_kind.get(ScoringKind.WARNING) or [None])[0]
    if warning is not None:
        for kind in (ScoringKind.REWARD, ScoringKind.PENALTY):
            for stage in by_kind.get(kind, []):
                if stage.order <= warning.order:
                    found.append(_error(
                        "chain.warning_order",
                        f"{type(warning).__name__} (order {warning.order}) must run before "
                        f"{type(stage).__name__} (order {stage.order})",
                        "an escalated rule would be excluded from P and counted twice"))

    # 3. What the pipeline does not check on its own.
    for stage in real:
        unknown = sorted(set(stage.requires) - PROVIDED_CONTEXT_KEYS)
        if unknown:
            found.append(ScorerDiagnostic(
                level="warning", code="chain.unknown_requirement",
                message=f"{type(stage).__name__}.requires names {unknown}, which no pipeline supplies",
                hint=f"known keys: {sorted(PROVIDED_CONTEXT_KEYS)}"))
        found.extend(stage.validate(settings))

    scoring = dict(settings.get("SCORING", settings) or {})
    floor = float(
        (settings.get("COMPLIANCE") or {}).get("MIN_FLOOR_LAMBDA", DEFAULT_MIN_FLOOR_LAMBDA)
    )
    lambdas = dict(scoring.get("PENALTY_LAMBDAS") or {})
    default = float(scoring.get("DEFAULT_LAMBDA", 1.0))
    for rule_id in sorted(effective_non_offsettable(scoring.get("NON_OFFSETTABLE"))):
        if float(lambdas.get(rule_id, default)) < floor:
            found.append(_error(
                "chain.floor_lambda",
                f"the lambda of non-offsettable {rule_id} is below MIN_FLOOR_LAMBDA {floor:g}"))
    return found
