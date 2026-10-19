"""Scoring: ledger, warning regime, continuous docx equations, and scorecards."""

from __future__ import annotations

from .continuous import (
    ContinuousScore,
    PenaltyBreakdown,
    RewardBreakdown,
    penalty_from_amounts,
    reward_from_buckets,
    score,
)
from .ledger import Ledger, RuleEvent, record_from_flag, severity_from_proportionality
from .judge import JudgeReport, TaskJudge
from .pipeline import Scorer, build_scorer, severity_weight
from .scorecard import EpisodeScorecard, PeerScorecard, SuppressedRule
from .warning import EscalationOutcome, WarningPolicy, WarningRecord, WarningRegistry

__all__ = [
    "ContinuousScore",
    "EpisodeScorecard",
    "EscalationOutcome",
    "JudgeReport",
    "Ledger",
    "PeerScorecard",
    "PenaltyBreakdown",
    "RewardBreakdown",
    "RuleEvent",
    "Scorer",
    "SuppressedRule",
    "TaskJudge",
    "WarningPolicy",
    "WarningRecord",
    "WarningRegistry",
    "build_scorer",
    "penalty_from_amounts",
    "record_from_flag",
    "reward_from_buckets",
    "score",
    "severity_from_proportionality",
    "severity_weight",
    "TaskJudge",
]
