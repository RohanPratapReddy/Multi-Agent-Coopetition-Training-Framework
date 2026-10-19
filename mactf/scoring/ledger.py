"""The per-episode rule ledger.

Peers and the meta-judge emit :class:`RuleEvent` records. The ledger validates
each event against the catalog, so an unknown rule ID, a reward carrying a
penalty severity, or a duplicate derived event fails before it can influence a
score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

from ..exceptions import RuleViolation
from ..rules.catalog import Rule, get_rule
from ..rules.severity import Proportionality, Severity


@dataclass(frozen=True)
class RuleEvent:
    """A single observation that a rule fired."""

    rule_id: str
    magnitude: float = 1.0
    severity: Severity | None = None
    evidence: str = ""
    source: str = "meta_judge"
    occurred_at: float | None = None
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise RuleViolation("a rule event requires a rule_id")
        if self.magnitude < 0:
            raise RuleViolation(f"{self.rule_id}: magnitude must be non-negative")
        if self.severity is not None and not isinstance(self.severity, Severity):
            object.__setattr__(self, "severity", Severity.parse(self.severity))

    @property
    def rule(self) -> Rule:
        return get_rule(self.rule_id)

    def resolved_severity(self) -> Severity:
        return self.severity if self.severity is not None else self.rule.default_severity

    def describe(self) -> str:
        return (
            f"{self.rule_id} x{self.magnitude:g} @ {self.resolved_severity().label} "
            f"from {self.source}"
            + (f" ({self.evidence})" if self.evidence else "")
        )


class Ledger:
    """An ordered, validated collection of rule events for one peer episode."""

    def __init__(self, *, peer: str = "") -> None:
        self.peer = peer
        self._events: list[RuleEvent] = []

    # -- mutation --------------------------------------------------------
    def record(
        self,
        rule_id: str,
        *,
        magnitude: float = 1.0,
        severity: Severity | str | None = None,
        evidence: str = "",
        source: str = "meta_judge",
        occurred_at: float | None = None,
        tags: Iterable[str] = (),
    ) -> RuleEvent:
        """Validate and append an event, returning it for inspection."""
        rule = get_rule(rule_id)
        resolved = None
        if severity is not None:
            resolved = Severity.parse(severity)
            if rule.is_reward and int(resolved) < int(Severity.NONE):
                raise RuleViolation(f"{rule_id}: rewards cannot carry a negative severity")
        event = RuleEvent(
            rule_id=rule.id,
            magnitude=float(magnitude),
            severity=resolved,
            evidence=evidence,
            source=source,
            occurred_at=occurred_at,
            tags=frozenset(tags),
        )
        self._events.append(event)
        return event

    def extend(self, events: Iterable[RuleEvent]) -> None:
        for event in events:
            # Re-validate through the public path so external events cannot bypass checks.
            self.record(
                event.rule_id,
                magnitude=event.magnitude,
                severity=event.severity,
                evidence=event.evidence,
                source=event.source,
                occurred_at=event.occurred_at,
                tags=event.tags,
            )

    def clear(self) -> None:
        self._events.clear()

    # -- queries ---------------------------------------------------------
    @property
    def events(self) -> tuple[RuleEvent, ...]:
        return tuple(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[RuleEvent]:
        return iter(self._events)

    def rewards(self) -> tuple[RuleEvent, ...]:
        return tuple(event for event in self._events if event.rule.is_reward)

    def penalties(self) -> tuple[RuleEvent, ...]:
        return tuple(event for event in self._events if event.rule.is_penalty)

    def for_rule(self, rule_id: str) -> tuple[RuleEvent, ...]:
        return tuple(event for event in self._events if event.rule_id == rule_id)

    def rules(self) -> tuple[Rule, ...]:
        return tuple(dict.fromkeys(event.rule for event in self._events))

    def triggered_ids(self) -> set[str]:
        return {event.rule_id for event in self._events}

    def has_non_offsettable(self) -> bool:
        return any(event.rule.non_offsettable for event in self._events)

    def bucket_total(self, bucket: str) -> float:
        """Sum magnitudes of reward events feeding a docx reward bucket."""
        return sum(
            event.magnitude
            for event in self._events
            if event.rule.is_reward and event.rule.bucket == bucket
        )

    def penalty_amounts(self, *, lambdas: dict[str, float], default_lambda: float) -> dict[str, float]:
        """Per-rule penalty magnitudes scaled by their docx lambda."""
        amounts: dict[str, float] = {}
        for event in self.penalties():
            lam = float(lambdas.get(event.rule_id, default_lambda))
            amounts[event.rule_id] = amounts.get(event.rule_id, 0.0) + lam * event.magnitude
        return amounts

    def severity_histogram(self) -> dict[str, int]:
        histogram = {tier.label: 0 for tier in Severity}
        for event in self._events:
            histogram[event.resolved_severity().label] += 1
        return histogram

    def describe(self) -> str:
        if not self._events:
            return f"ledger[{self.peer}] empty"
        lines = [f"ledger[{self.peer}] {len(self._events)} event(s):"]
        lines.extend(f"  - {event.describe()}" for event in self._events)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "peer": self.peer,
            "events": [
                {
                    "rule_id": event.rule_id,
                    "magnitude": event.magnitude,
                    "severity": event.resolved_severity().label,
                    "evidence": event.evidence,
                    "source": event.source,
                    "occurred_at": event.occurred_at,
                }
                for event in self._events
            ],
        }


def severity_from_proportionality(
    *,
    rule: Rule,
    proportionality: Proportionality,
    weights: dict[str, Any] | None = None,
) -> Severity:
    """Choose a severity for a rule from its proportionality factors."""
    return proportionality.suggest(ceiling=rule.default_severity)


def record_from_flag(
    ledger: Ledger,
    *,
    rule_id: str,
    proportionality: Proportionality | None = None,
    evidence: str = "",
    source: str = "peer_flag",
) -> RuleEvent:
    """Record a severity derived from proportionality factors rather than a guess."""
    rule = get_rule(rule_id)
    severity = rule.default_severity
    if proportionality is not None:
        severity = severity_from_proportionality(rule=rule, proportionality=proportionality)
    return ledger.record(rule_id, severity=severity, evidence=evidence, source=source)
