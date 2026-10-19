"""The warning regime (framework.md section 8).

A warning is a zero-score corrective record. It escalates to a penalty only
after ``K_ESC`` active warnings for the same rule ID within ``N_WIN`` episodes,
decays after ``N_DECAY`` consecutive clean episodes, and is scoped per rule ID so
warnings can never pool across rules (W6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .. import signals
from ..rules.catalog import NEVER_WARNING_FIRST, get_rule
from ..rules.severity import Severity, escalate, weight_for


@dataclass(frozen=True)
class WarningPolicy:
    """Warning parameters W1-W9 from framework.md section 8.2."""

    k_esc: int = 3
    n_win: int = 5
    n_decay: int = 3
    w_cap: int = 2
    p_warn: float = 5.0
    escalate: bool = True

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> "WarningPolicy":
        return cls(
            k_esc=int(settings.get("K_ESC", 3)),
            n_win=int(settings.get("N_WIN", 5)),
            n_decay=int(settings.get("N_DECAY", 3)),
            w_cap=int(settings.get("W_CAP", 2)),
            p_warn=float(settings.get("P_WARN", 5.0)),
            escalate=bool(settings.get("ESCALATE", True)),
        )

    def __post_init__(self) -> None:
        if self.k_esc < 2:
            raise ValueError("k_esc must be at least 2")
        if not 1 <= self.w_cap <= self.k_esc:
            raise ValueError("w_cap must be within [1, k_esc]")
        if self.n_win < 1 or self.n_decay < 1:
            raise ValueError("n_win and n_decay must be >= 1")


@dataclass
class WarningRecord:
    rule_id: str
    episode: int
    severity: Severity
    evidence: str = ""
    expiry_episode: int = 0
    active: bool = True

    def describe(self) -> str:
        state = "active" if self.active else "expired"
        return (
            f"warning[{self.rule_id}] ep={self.episode} sev={self.severity.label} "
            f"{state} expires={self.expiry_episode}"
        )


@dataclass
class EscalationOutcome:
    """Result of feeding one violation into the warning regime."""

    rule_id: str
    issued: list[WarningRecord] = field(default_factory=list)
    expired: list[WarningRecord] = field(default_factory=list)
    escalated: WarningRecord | None = None
    penalty: float = 0.0
    severity: Severity | None = None
    reason: str = ""

    @property
    def is_warning_only(self) -> bool:
        return self.escalated is None

    def describe(self) -> str:
        parts = []
        if self.issued:
            parts.append(f"issued {len(self.issued)} warning(s)")
        if self.expired:
            parts.append(f"expired {len(self.expired)}")
        if self.escalated is not None:
            parts.append(
                f"escalated {self.rule_id} to {self.severity.label if self.severity else '?'} "
                f"penalty={self.penalty:.3f}"
            )
        return f"{self.rule_id}: " + ("; ".join(parts) if parts else "no change")


class WarningRegistry:
    """Tracks active warnings per peer and per rule ID."""

    def __init__(self, policy: WarningPolicy | None = None) -> None:
        self.policy = policy or WarningPolicy()
        self._warnings: dict[str, dict[str, list[WarningRecord]]] = {}
        self._clean_streak: dict[str, dict[str, int]] = {}

    # -- introspection ---------------------------------------------------
    def active(self, peer: str, rule_id: str) -> list[WarningRecord]:
        records = self._warnings.get(peer, {}).get(rule_id, [])
        return [record for record in records if record.active]

    def all_active(self, peer: str) -> dict[str, list[WarningRecord]]:
        rules = self._warnings.get(peer, {})
        return {
            rule_id: [r for r in records if r.active]
            for rule_id, records in rules.items()
            if any(r.active for r in records)
        }

    def count(self, peer: str, rule_id: str) -> int:
        return len(self.active(peer, rule_id))

    # -- lifecycle -------------------------------------------------------
    def begin_episode(self, peer: str, *, episode: int) -> list[WarningRecord]:
        """Expire warnings whose decay window has elapsed.

        A warning expires when the peer has completed ``N_DECAY`` consecutive
        clean episodes *and* the record is outside the ``N_WIN`` window.
        """
        expired: list[WarningRecord] = []
        rules = self._warnings.get(peer, {})
        for rule_id, records in list(rules.items()):
            for record in records:
                if not record.active:
                    continue
                if episode < record.expiry_episode:
                    continue
                streak = self._clean_streak.get(peer, {}).get(rule_id, 0)
                if streak >= self.policy.n_decay:
                    record.active = False
                    expired.append(record)
                    signals.warning_expired.send(
                        sender=self, peer=peer, rule_id=rule_id, record=record
                    )
        return expired

    def issue(
        self,
        peer: str,
        rule_id: str,
        *,
        severity: Severity | str,
        episode: int,
        evidence: str = "",
        weights: dict[str, Any] | None = None,
    ) -> EscalationOutcome:
        """Record a warning-eligible violation and escalate if the cap is hit.

        framework.md W3: the *most recent* warning converts to a penalty at the
        next severity tier; earlier warnings stay zero-score. W8 forbids both a
        warning and a penalty for the same act, so the escalated record is
        marked inactive before the penalty is returned.
        """
        rule = get_rule(rule_id)
        resolved = Severity.parse(severity)
        outcome = EscalationOutcome(rule_id=rule_id)

        if rule_id in NEVER_WARNING_FIRST or not rule.warning_eligible:
            outcome.severity = resolved
            outcome.penalty = (
                weight_for(resolved, weights) if weights and resolved.label in weights else float(resolved) or 1.0
            )
            outcome.reason = f"{rule_id} is never warning-first (framework.md W9)"
            return outcome

        rule_bucket = self._warnings.setdefault(peer, {}).setdefault(rule_id, [])
        record = WarningRecord(
            rule_id=rule_id,
            episode=episode,
            severity=resolved,
            evidence=evidence,
            expiry_episode=episode + self.policy.n_win,
        )
        rule_bucket.append(record)
        outcome.issued.append(record)
        signals.warning_issued.send(
            sender=self, peer=peer, rule_id=rule_id, record=record, episode=episode
        )

        active = [r for r in rule_bucket if r.active]
        self._clean_streak.setdefault(peer, {})[rule_id] = 0

        if not self.policy.escalate:
            self._enforce_cap(rule_bucket, outcome)
            return outcome

        # Escalation is judged before the cap is applied, because the cap exists to
        # bound banked warnings (W6), not to suppress the escalation trigger (W3).
        if len(active) >= self.policy.k_esc:
            escalated = active[-1]
            escalated.active = False
            target = escalate(resolved, max_tier=3)
            outcome.escalated = escalated
            outcome.severity = target
            outcome.penalty = float(self.policy.p_warn)
            outcome.reason = (
                f"{len(active)} active {rule_id} warnings reached K_ESC="
                f"{self.policy.k_esc} within {self.policy.n_win} episodes"
            )
            signals.warning_escalated.send(
                sender=self, peer=peer, rule_id=rule_id, record=escalated,
                severity=target, penalty=outcome.penalty,
            )

        self._enforce_cap(rule_bucket, outcome)
        return outcome

    def _enforce_cap(self, rule_bucket: list[WarningRecord], outcome: EscalationOutcome) -> None:
        """Retire the oldest active warnings beyond ``W_CAP`` (framework.md W6)."""
        active = [r for r in rule_bucket if r.active]
        if len(active) > self.policy.w_cap:
            overflow = active[: len(active) - self.policy.w_cap]
            for retired in overflow:
                retired.active = False
                if retired not in outcome.issued:
                    outcome.expired.append(retired)

    def mark_clean(self, peer: str, rule_id: str) -> None:
        """Advance the consecutive-clean counter for a rule ID."""
        streaks = self._clean_streak.setdefault(peer, {})
        streaks[rule_id] = streaks.get(rule_id, 0) + 1

    def mark_clean_except(self, peer: str, violated: set[str], *, known_rules: set[str]) -> None:
        """Advance clean counters for every rule that did not fire this episode."""
        for rule_id in known_rules:
            if rule_id in violated:
                self._clean_streak.setdefault(peer, {})[rule_id] = 0
            else:
                self.mark_clean(peer, rule_id)

    def expunge(self, peer: str, rule_id: str) -> list[WarningRecord]:
        """Remove all warnings for a rule after an adjudicator reversal (W5)."""
        records = self._warnings.get(peer, {}).get(rule_id, [])
        removed = [record for record in records if record.active]
        for record in removed:
            record.active = False
        return removed

    def snapshot(self, peer: str) -> dict[str, Any]:
        return {
            "peer": peer,
            "active": {
                rule_id: [record.describe() for record in records]
                for rule_id, records in self.all_active(peer).items()
            },
            "clean_streak": dict(self._clean_streak.get(peer, {})),
        }

    def reset(self) -> None:
        self._warnings.clear()
        self._clean_streak.clear()
