"""Precedence resolution (framework.md section 3).

Conflicts are resolved by tier, never averaged. The output of this module is a
:class:`PrecedenceDecision` that states which rule controls and which rules were
suppressed, so the decision is auditable rather than implicit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from ..exceptions import ComplianceError
from .catalog import Rule, TIERS, get_rule
from .severity import Severity

_DESCRIPTION = re.compile(
    r"^tier=(\S+) applied=\[(.*?)\] suppressed=\[(.*?)\] mode=(\S+) \((.*)\)$", re.DOTALL
)


@dataclass(frozen=True)
class PrecedenceDecision:
    """The outcome of applying the precedence ladder to a set of triggered rules."""

    controlling_tier: int | None
    applied: tuple[Rule, ...]
    suppressed: tuple[Rule, ...]
    mode: str
    rationale: str

    @property
    def controlling_tier_name(self) -> str | None:
        if self.controlling_tier is None:
            return None
        return TIERS[self.controlling_tier]

    @property
    def has_severe_penalty(self) -> bool:
        return any(rule.non_offsettable for rule in self.applied)

    def describe(self) -> str:
        applied = ", ".join(rule.id for rule in self.applied) or "none"
        suppressed = ", ".join(rule.id for rule in self.suppressed) or "none"
        tier = self.controlling_tier_name or "none"
        return (
            f"tier={tier} applied=[{applied}] suppressed=[{suppressed}] "
            f"mode={self.mode} ({self.rationale})"
        )

    @classmethod
    def from_description(cls, text: str) -> "PrecedenceDecision":
        """The exact inverse of :meth:`describe`, resolving rule ids through the catalog.

        ``describe()`` is the only form a decision takes on the wire, so a client
        rebuilds the object from it rather than carrying a second encoding.
        """
        match = _DESCRIPTION.match(text.strip())
        if match is None:
            raise ValueError(f"not a precedence description: {text!r}")
        tier_name, applied, suppressed, mode, rationale = match.groups()
        tier: int | None = None
        if tier_name != "none":
            by_name = {name: number for number, name in TIERS.items()}
            if tier_name not in by_name:
                raise ValueError(f"unknown tier {tier_name!r} in precedence description")
            tier = by_name[tier_name]

        def rules(listing: str) -> tuple[Rule, ...]:
            return tuple(get_rule(item.strip()) for item in listing.split(",") if item.strip()) if listing != "none" else ()

        return cls(
            controlling_tier=tier,
            applied=rules(applied),
            suppressed=rules(suppressed),
            mode=mode,
            rationale=rationale,
        )


def resolve(
    rules: Iterable[Rule],
    *,
    mode: str = "strict",
    override_tier: int | None = None,
) -> PrecedenceDecision:
    """Apply framework.md section 3.2 to a collection of triggered rules.

    ``strict`` (the default) implements step 3 literally: only the rules in the
    highest represented tier are scored. ``layered`` scores every tier but still
    rejects cross-tier netting downstream, which is useful when an operator wants
    full visibility into lower-tier events without letting them offset a breach.
    """
    triggered = list(dict.fromkeys(rules))
    if not triggered:
        return PrecedenceDecision(
            controlling_tier=None,
            applied=(),
            suppressed=(),
            mode=mode,
            rationale="no rules triggered",
        )

    if mode not in {"strict", "layered"}:
        raise ComplianceError(f"unknown precedence mode {mode!r}")

    controlling = override_tier if override_tier is not None else min(rule.tier for rule in triggered)

    if mode == "strict":
        applied = tuple(rule for rule in triggered if rule.tier == controlling)
        suppressed = tuple(rule for rule in triggered if rule.tier != controlling)
        rationale = f"only the controlling tier T{controlling} is scored"
    else:
        applied = tuple(triggered)
        suppressed = ()
        rationale = "all tiers scored; netting across tiers still forbidden"

    applied = tuple(sorted(applied, key=lambda rule: (rule.tier, rule.id)))
    suppressed = tuple(sorted(suppressed, key=lambda rule: (rule.tier, rule.id)))
    return PrecedenceDecision(
        controlling_tier=controlling,
        applied=applied,
        suppressed=suppressed,
        mode=mode,
        rationale=rationale,
    )


def escalate_ceiling(rule: Rule, *, max_tier: int) -> Severity:
    """The highest severity a rule may reach under warning escalation.

    framework.md 8.3 forbids escalating into P17, P20, P21, or P22; those rules
    only ever apply through their own triggers at their own severity.
    """
    if rule.non_offsettable and rule.default_severity >= Severity.SEVERE:
        return rule.default_severity
    return Severity(min(max(int(rule.default_severity), int(rule.default_severity)), max_tier))


def apply_no_netting(
    *,
    reward_total: float,
    penalty_total: float,
    applied: Iterable[Rule],
    non_offsettable: set[str],
    penalty_amounts: dict[str, float],
) -> tuple[float, float, str | None]:
    """Split a score into an offsettable net and a non-offsettable floor.

    Returns ``(net, floor, reason)`` where ``floor`` is <= 0 and must be added to
    the net without being reduced by any reward. This is the mechanical
    enforcement of framework.md 3.1 rule 2 and 6.1.
    """
    floors = {
        rule_id: amount
        for rule_id, amount in penalty_amounts.items()
        if rule_id in non_offsettable
    }
    if not floors:
        return reward_total - penalty_total, 0.0, None

    floor = -abs(sum(floors.values()))
    offending = sorted(floors)
    reason = (
        f"non-offsettable penalty(s) {', '.join(offending)} set a floor of {floor:.4f}; "
        "rewards earned in the same episode may not reduce it"
    )
    offsettable_penalty = sum(
        amount for rule_id, amount in penalty_amounts.items() if rule_id not in non_offsettable
    )
    net = reward_total - offsettable_penalty
    return net, floor, reason


def assert_monotone_in_tier(
    controlling_tier: int,
    *,
    reward_tier: int | None,
    penalty_tier: int | None,
) -> None:
    """Guard used by tests: a lower-tier reward may not dominate a higher-tier penalty."""
    if penalty_tier is None:
        return
    if reward_tier is not None and reward_tier > penalty_tier:
        # This is legal (the reward is simply suppressed), not an error.
        return
    if controlling_tier > penalty_tier:
        raise ComplianceError(
            "precedence resolution chose a tier lower than an active penalty tier"
        )
