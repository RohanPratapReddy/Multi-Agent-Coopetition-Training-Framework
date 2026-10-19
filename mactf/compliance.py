"""Compliance: the bridge between the docx formulas and framework.md invariants.

The docx defines a single continuous number, ``net = R - P - W``. framework.md
section 3.1 rule 2 and section 6.1 forbid netting across tiers and forbid any
reward from reducing a non-offsettable penalty. This module performs that
reconciliation and reports it, so a run can prove which constraint fired.

Applied order:

1. Precedence resolves which rules control (``rules.precedence.resolve``).
2. Rewards are summed only from controlling-tier rewards.
3. Offsettable penalties are amplified by Omega and subtracted.
4. Non-offsettable penalties are subtracted again as an absolute floor.
5. The warning amount is subtracted last.
6. The result is clamped to ``[-floor_cap, reward_cap]`` configured per run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .rules.catalog import NON_OFFSETTABLE
from .rules.precedence import PrecedenceDecision, resolve

if TYPE_CHECKING:  # avoids a circular import at module load time
    from .scoring.continuous import ContinuousScore


@dataclass(frozen=True)
class ComplianceResult:
    """A fully constrained episode score with its full derivation."""

    value: float
    reward_component: float
    offsettable_penalty: float
    omega: float
    non_offsettable_penalty: float
    warning: float
    floor: float
    net_before_constraints: float
    decision: PrecedenceDecision
    violations: tuple[str, ...]
    reason: str | None

    @property
    def violated(self) -> bool:
        return bool(self.violations)

    def describe(self) -> str:
        parts = [
            f"value={self.value:.4f}",
            f"reward={self.reward_component:.4f}",
            f"offsettable={self.offsettable_penalty:.4f}",
            f"omega={self.omega:g}",
            f"floor={self.floor:.4f}",
            f"warning={self.warning:.4f}",
        ]
        if self.reason:
            parts.append(f"reason={self.reason}")
        return "compliance(" + ", ".join(parts) + ")"

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "reward_component": self.reward_component,
            "offsettable_penalty": self.offsettable_penalty,
            "omega": self.omega,
            "non_offsettable_penalty": self.non_offsettable_penalty,
            "warning": self.warning,
            "floor": self.floor,
            "net_before_constraints": self.net_before_constraints,
            "violations": list(self.violations),
            "reason": self.reason,
            "precedence": self.decision.describe(),
        }


def constrain(
    continuous: ContinuousScore,
    *,
    non_offsettable: set[str] | frozenset[str] = NON_OFFSETTABLE,
    warning: float | None = None,
    value_cap: float = 1e4,
) -> ComplianceResult:
    """Convert a docx continuous score into a compliant episode score."""
    penalty = continuous.penalty
    decision = continuous.decision or resolve(())

    offsettable = sum(
        amount for rule_id, amount in penalty.amounts.items() if rule_id not in non_offsettable
    )
    non_offset = sum(
        amount for rule_id, amount in penalty.amounts.items() if rule_id in non_offsettable
    )

    omega = penalty.omega if offsettable > 0.0 else 1.0
    reward = continuous.reward.total
    warnings = continuous.warning if warning is None else float(warning)

    net = reward - offsettable * omega - warnings
    floor = -abs(non_offset)
    value = net + floor
    value = max(-value_cap, min(value_cap, value))

    violations = tuple(sorted(rule_id for rule_id in penalty.amounts if rule_id in non_offsettable))
    reason = None
    if non_offset:
        reason = (
            f"non-offsettable penalty {', '.join(violations)} applied as an absolute floor of "
            f"{floor:.4f}; no reward can reduce it"
        )
    elif decision.suppressed:
        reason = (
            f"tier {decision.controlling_tier} controlled; suppressed "
            + ", ".join(rule.id for rule in decision.suppressed)
        )

    return ComplianceResult(
        value=value,
        reward_component=reward,
        offsettable_penalty=offsettable * omega,
        omega=omega,
        non_offsettable_penalty=non_offset,
        warning=warnings,
        floor=floor,
        net_before_constraints=continuous.net,
        decision=decision,
        violations=violations,
        reason=reason,
    )


def is_void(
    continuous: ContinuousScore,
    *,
    non_offsettable: set[str] | frozenset[str] = NON_OFFSETTABLE,
) -> bool:
    """True when task-success credit must be withheld for this episode."""
    return any(rule_id in non_offsettable for rule_id in continuous.penalty.amounts)


def void_rule_ids(
    continuous: ContinuousScore,
    *,
    non_offsettable: set[str] | frozenset[str] = NON_OFFSETTABLE,
) -> tuple[str, ...]:
    return tuple(
        sorted(rule_id for rule_id in continuous.penalty.amounts if rule_id in non_offsettable)
    )
