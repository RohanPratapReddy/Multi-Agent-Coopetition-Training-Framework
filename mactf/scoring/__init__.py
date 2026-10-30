"""Scoring: ledger, warning regime, continuous docx equations, and scorecards.

``Scorer`` is the abstract stage base class. ``ScoringPipeline`` is the concrete
composite that produces a ``PeerScorecard``; ``build_scorer`` returns one. The
legacy name ``mactf.scoring.pipeline.Scorer`` still resolves to the pipeline with
a ``DeprecationWarning`` (document 07 section 7.3).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# Names resolve on first access (PEP 562): importing a light module such as
# ``mactf.scoring.scorecard`` must not drag torch in through this package.
_EXPORTS: dict[str, str] = {
    "BINDINGS": "guards",
    "ClaimAudit": "evidence",
    "CohortNormalizer": "normalize",
    "ContinuousScore": "continuous",
    "DEFAULT_KNOWN_RULES": "pipeline",
    "DEFAULT_POINTS_TABLE": "points",
    "DEFAULT_REWARD_WEIGHTS": "reward_scorer",
    "DenialRecord": "points",
    "EpisodeEvidence": "evidence",
    "EpisodeScorecard": "scorecard",
    "EscalationOutcome": "warning",
    "GUARDS": "guards",
    "GuardVerdict": "guards",
    "HierarchicalScorer": "hierarchy",
    "InferenceNodeResult": "hierarchy",
    "InferenceResult": "hierarchy",
    "JudgeReport": "judge",
    "Ledger": "ledger",
    "NormalizationPolicy": "normalize",
    "NormalizationReport": "normalize",
    "PeerScorecard": "scorecard",
    "PenaltyBreakdown": "continuous",
    "PenaltyScorer": "penalty",
    "PointsBreakdown": "points",
    "PointsEngine": "guards",
    "PointsTable": "points",
    "ROLLUP_STRATEGIES": "hierarchy",
    "RewardBreakdown": "continuous",
    "RewardScorer": "reward_scorer",
    "RuleBinding": "guards",
    "RuleEvent": "ledger",
    "SCORER_TYPES": "registry",
    "ScoreContext": "base",
    "Scorer": "base",
    "ScorerDiagnostic": "base",
    "ScorerOutput": "base",
    "ScoringKind": "base",
    "ScoringPipeline": "pipeline",
    "SuppressedRule": "scorecard",
    "TaskJudge": "judge",
    "WarningMutation": "base",
    "WarningPolicy": "warning",
    "WarningRecord": "warning",
    "WarningRegistry": "warning",
    "WarningScorer": "warning_scorer",
    "build_bindings": "guards",
    "build_pool": "pipeline",
    "build_rollup": "hierarchy",
    "build_scorer": "pipeline",
    "build_scorer_from_spec": "registry",
    "build_stages": "pipeline",
    "evaluate_binding": "guards",
    "finalize_points": "points",
    "penalty_from_amounts": "continuous",
    "record_from_flag": "ledger",
    "register_scorer": "registry",
    "resolve_binding": "guards",
    "reward_from_buckets": "continuous",
    "reward_total_from_events": "reward_scorer",
    "run_inference": "hierarchy",
    "score": "continuous",
    "scorer_capabilities": "registry",
    "severity_fraction": "base",
    "severity_from_proportionality": "ledger",
    "severity_weight": "pipeline",
    "validate_bindings": "guards",
}

if TYPE_CHECKING:  # pragma: no cover - static analysers see the real imports
    from .base import ScoreContext, Scorer, ScorerDiagnostic, ScorerOutput, ScoringKind, WarningMutation, severity_fraction
    from .continuous import ContinuousScore, PenaltyBreakdown, RewardBreakdown, penalty_from_amounts, reward_from_buckets, score
    from .evidence import ClaimAudit, EpisodeEvidence
    from .guards import BINDINGS, GUARDS, GuardVerdict, PointsEngine, RuleBinding, build_bindings, evaluate_binding, resolve_binding, validate_bindings
    from .hierarchy import HierarchicalScorer, InferenceNodeResult, InferenceResult, ROLLUP_STRATEGIES, build_rollup, run_inference
    from .judge import JudgeReport, TaskJudge
    from .ledger import Ledger, RuleEvent, record_from_flag, severity_from_proportionality
    from .normalize import CohortNormalizer, NormalizationPolicy, NormalizationReport
    from .penalty import PenaltyScorer
    from .pipeline import DEFAULT_KNOWN_RULES, ScoringPipeline, build_pool, build_scorer, build_stages, severity_weight
    from .points import DEFAULT_POINTS_TABLE, DenialRecord, PointsBreakdown, PointsTable, finalize_points
    from .registry import SCORER_TYPES, build_scorer_from_spec, register_scorer, scorer_capabilities
    from .reward_scorer import DEFAULT_REWARD_WEIGHTS, RewardScorer, reward_total_from_events
    from .scorecard import EpisodeScorecard, PeerScorecard, SuppressedRule
    from .warning import EscalationOutcome, WarningPolicy, WarningRecord, WarningRegistry
    from .warning_scorer import WarningScorer

__all__ = ["BINDINGS", "ClaimAudit", "CohortNormalizer", "ContinuousScore", "DEFAULT_KNOWN_RULES", "DEFAULT_POINTS_TABLE", "DEFAULT_REWARD_WEIGHTS", "DenialRecord", "EpisodeEvidence", "EpisodeScorecard", "EscalationOutcome", "GUARDS", "GuardVerdict", "HierarchicalScorer", "InferenceNodeResult", "InferenceResult", "JudgeReport", "Ledger", "NormalizationPolicy", "NormalizationReport", "PeerScorecard", "PenaltyBreakdown", "PenaltyScorer", "PointsBreakdown", "PointsEngine", "PointsTable", "ROLLUP_STRATEGIES", "RewardBreakdown", "RewardScorer", "RuleBinding", "RuleEvent", "SCORER_TYPES", "ScoreContext", "Scorer", "ScorerDiagnostic", "ScorerOutput", "ScoringKind", "ScoringPipeline", "SuppressedRule", "TaskJudge", "WarningMutation", "WarningPolicy", "WarningRecord", "WarningRegistry", "WarningScorer", "build_bindings", "build_pool", "build_rollup", "build_scorer", "build_scorer_from_spec", "build_stages", "evaluate_binding", "finalize_points", "penalty_from_amounts", "record_from_flag", "register_scorer", "resolve_binding", "reward_from_buckets", "reward_total_from_events", "run_inference", "score", "scorer_capabilities", "severity_fraction", "severity_from_proportionality", "severity_weight", "validate_bindings"]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
