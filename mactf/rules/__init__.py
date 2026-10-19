"""Rule catalog, severity, and precedence primitives."""

from __future__ import annotations

from .catalog import (
    ALL_RULES,
    CATALOG,
    NEVER_WARNING_FIRST,
    NON_OFFSETTABLE,
    PENALTIES,
    REWARDS,
    Rule,
    TIERS,
    describe,
    get_rule,
    rewards_by_bucket,
    validate_catalog,
)
from .precedence import PrecedenceDecision, apply_no_netting, resolve
from .severity import Proportionality, Severity, check_monotone, escalate, weight_for

__all__ = [
    "ALL_RULES",
    "CATALOG",
    "NEVER_WARNING_FIRST",
    "NON_OFFSETTABLE",
    "PENALTIES",
    "REWARDS",
    "Rule",
    "TIERS",
    "PrecedenceDecision",
    "Proportionality",
    "Severity",
    "apply_no_netting",
    "check_monotone",
    "describe",
    "escalate",
    "get_rule",
    "resolve",
    "rewards_by_bucket",
    "validate_catalog",
    "weight_for",
]
