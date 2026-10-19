"""The continuous scorer, following the docx formulations.

Reward (docx Turn 3)::

    R = alpha * r_task + beta * r_epi + gamma * r_collab + delta * r_align

Penalty (docx Turn 3)::

    P = (sum_k lambda_k * p_k) * Omega

Warning (docx Turn 3) is supplied by :mod:`mactf.scoring.warning` as a
threshold-triggered amount. Net signal (docx Turn 3)::

    net = R - P - W

The docx's flat ``net`` is computed here, but the episode score is not allowed to
use it as the final number: :mod:`mactf.compliance` re-applies tier precedence
and the non-offsettable floor before anything reaches the loss or the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..rules.catalog import PENALTIES, REWARDS, Rule
from ..rules.precedence import PrecedenceDecision, resolve
from ..rules.severity import Severity, weight_for
from .ledger import Ledger


@dataclass
class RewardBreakdown:
    """Per-bucket reward components and their weighted total."""

    r_task: float = 0.0
    r_epi: float = 0.0
    r_collab: float = 0.0
    r_align: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)
    rules: tuple[str, ...] = ()

    @property
    def total(self) -> float:
        return sum(
            self.weights.get(bucket, 1.0) * getattr(self, bucket)
            for bucket in ("r_task", "r_epi", "r_collab", "r_align")
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "r_task": self.r_task,
            "r_epi": self.r_epi,
            "r_collab": self.r_collab,
            "r_align": self.r_align,
            "total": self.total,
            "rules": list(self.rules),
            "weights": dict(self.weights),
        }


@dataclass
class PenaltyBreakdown:
    """Per-rule penalty components, the Omega multiplier, and the total."""

    amounts: dict[str, float] = field(default_factory=dict)
    omega: float = 1.0
    omega_rules: tuple[str, ...] = ()
    rules: tuple[str, ...] = ()

    @property
    def raw_total(self) -> float:
        return sum(self.amounts.values())

    @property
    def total(self) -> float:
        return self.raw_total * self.omega

    def as_dict(self) -> dict[str, Any]:
        return {
            "amounts": dict(self.amounts),
            "raw_total": self.raw_total,
            "omega": self.omega,
            "omega_rules": list(self.omega_rules),
            "total": self.total,
            "rules": list(self.rules),
        }


@dataclass
class ContinuousScore:
    """The docx score for one peer episode, before framework constraints."""

    reward: RewardBreakdown
    penalty: PenaltyBreakdown
    warning: float
    warning_events: tuple[str, ...] = ()
    decision: PrecedenceDecision | None = None
    suppressed_rewards: tuple[str, ...] = ()

    @property
    def net(self) -> float:
        return self.reward.total - self.penalty.total - self.warning

    @property
    def has_non_offsettable(self) -> bool:
        return any(rule_id in {"P1", "P2", "P17", "P20", "P21", "P22"} for rule_id in self.penalty.amounts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward.as_dict(),
            "penalty": self.penalty.as_dict(),
            "warning": self.warning,
            "net": self.net,
            "suppressed_rewards": list(self.suppressed_rewards),
            "precedence": self.decision.describe() if self.decision else None,
        }


def _reward_breakdown(ledger: Ledger, config: dict[str, Any]) -> RewardBreakdown:
    weights = {key: float(value) for key, value in config.get("REWARD_WEIGHTS", {}).items()}
    breakdown = RewardBreakdown(weights=weights)
    fired: list[str] = []
    for event in ledger.rewards():
        bucket = event.rule.bucket
        if not hasattr(breakdown, bucket):
            continue
        # Severity scales a reward within its tier band, using the configured
        # magnitudes. A reward at S0 contributes nothing.
        magnitudes = config.get("SEVERITY_WEIGHTS") or {}
        scale = float(magnitudes.get(event.resolved_severity().label, 1.0))
        setattr(breakdown, bucket, getattr(breakdown, bucket) + event.magnitude * scale)
        fired.append(event.rule_id)
    breakdown.rules = tuple(dict.fromkeys(fired))
    return breakdown


def _penalty_breakdown(
    ledger: Ledger,
    config: dict[str, Any],
    *,
    allowed_rules: set[str] | None = None,
    warning_only_rules: set[str] | None = None,
) -> PenaltyBreakdown:
    lambdas = {key: float(value) for key, value in config.get("PENALTY_LAMBDAS", {}).items()}
    default_lambda = float(config.get("DEFAULT_LAMBDA", 1.0))
    amounts: dict[str, float] = {}
    for event in ledger.penalties():
        if allowed_rules is not None and event.rule_id not in allowed_rules:
            continue
        # framework.md W1: a warning is zero-score. A rule that only produced a
        # warning this episode must not also be subtracted as a penalty.
        if warning_only_rules and event.rule_id in warning_only_rules:
            continue
        lam = float(lambdas.get(event.rule_id, default_lambda))
        amounts[event.rule_id] = amounts.get(event.rule_id, 0.0) + lam * event.magnitude

    omega = 1.0
    omega_rules = set(config.get("OMEGA_RULES", []))
    overrides = config.get("OMEGA_RULE_OVERRIDES", {}) or {}
    triggered = [rule_id for rule_id in amounts if rule_id in omega_rules]
    if triggered:
        # docx Omega is an unbounded amplifier; we use a bounded product of the
        # configured multipliers so a single event cannot produce infinity.
        multiplier = 1.0
        base = float(config.get("SANDBOX_MULTIPLIER", 1000.0))
        for rule_id in triggered:
            factor = float(overrides.get(rule_id, base))
            multiplier *= max(1.0, factor)
        omega = multiplier

    return PenaltyBreakdown(
        amounts=amounts,
        omega=omega,
        omega_rules=tuple(sorted(triggered)),
        rules=tuple(sorted(amounts)),
    )


def score(
    ledger: Ledger,
    *,
    score_config: dict[str, Any],
    severity_weights: dict[str, Any],
    warning_total: float = 0.0,
    warning_events: tuple[str, ...] = (),
    mode: str | None = None,
    escalated_rules: set[str] | None = None,
) -> ContinuousScore:
    """Compute the docx continuous score for a ledger.

    ``mode`` selects precedence handling. Under ``strict`` only the controlling
    tier's rules contribute, which is the default. Under ``layered`` every tier
    contributes but the netting ban is still enforced downstream.

    ``warning_events`` lists rules that produced only a warning this episode; per
    framework.md W1 they score zero, so they are excluded from ``P`` entirely.
    ``escalated_rules`` are the ones whose warning matured into a penalty; they
    are scored normally.
    """
    config = dict(score_config)
    config.setdefault("SEVERITY_WEIGHTS", severity_weights)
    decision = resolve(ledger.rules(), mode=mode or str(config.get("PRECEDENCE_MODE", "strict")))

    escalated = set(escalated_rules or ())
    warning_only = {rule_id for rule_id in warning_events if rule_id not in escalated}

    allowed = {rule.id for rule in decision.applied}
    reward = _reward_breakdown(ledger, config)
    penalty = _penalty_breakdown(
        ledger,
        config,
        allowed_rules=allowed if decision.mode == "strict" else None,
        warning_only_rules=warning_only,
    )

    if decision.mode == "strict" and not any(rule.is_reward for rule in decision.applied):
        suppressed = reward.rules
        reward = RewardBreakdown(weights=reward.weights)
    else:
        suppressed = tuple(
            event.rule_id for event in ledger.rewards() if event.rule_id not in allowed
        )

    return ContinuousScore(
        reward=reward,
        penalty=penalty,
        warning=float(warning_total),
        warning_events=tuple(warning_events),
        decision=decision,
        suppressed_rewards=tuple(dict.fromkeys(suppressed)),
    )


def reward_from_buckets(
    *,
    r_task: float = 0.0,
    r_epi: float = 0.0,
    r_collab: float = 0.0,
    r_align: float = 0.0,
    weights: dict[str, float] | None = None,
) -> float:
    """Evaluate the docx reward equation directly, for tests and simple tasks."""
    weights = weights or {"r_task": 1.0, "r_epi": 1.5, "r_collab": 1.2, "r_align": 1.0}
    return (
        weights.get("r_task", 1.0) * r_task
        + weights.get("r_epi", 1.5) * r_epi
        + weights.get("r_collab", 1.2) * r_collab
        + weights.get("r_align", 1.0) * r_align
    )


def penalty_from_amounts(
    amounts: dict[str, float],
    *,
    lambdas: dict[str, float],
    default_lambda: float = 1.0,
    omega: float = 1.0,
) -> float:
    """Evaluate the docx penalty equation directly."""
    raw = sum(float(lambdas.get(rule_id, default_lambda)) * amount for rule_id, amount in amounts.items())
    return raw * max(1.0, float(omega))
