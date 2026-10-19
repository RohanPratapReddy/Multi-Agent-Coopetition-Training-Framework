"""Scorecards: the per-peer, per-episode auditable scoring record."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..rules.catalog import Rule
from ..rules.severity import Severity
from .continuous import ContinuousScore
from .ledger import Ledger
from .warning import WarningRecord


@dataclass(frozen=True)
class SuppressedRule:
    rule_id: str
    reason: str


@dataclass
class PeerScorecard:
    """Everything needed to explain and audit one peer's episode score."""

    peer: str
    episode: int
    continuous: ContinuousScore
    final_score: float
    floor: float = 0.0
    floor_reason: str | None = None
    warnings_issued: tuple[WarningRecord, ...] = ()
    warnings_escalated: tuple[WarningRecord, ...] = ()
    suppressed: tuple[SuppressedRule, ...] = ()
    violated_non_offsettable: tuple[str, ...] = ()
    severity: Severity = Severity.NONE
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def net_before_constraints(self) -> float:
        return self.continuous.net

    @property
    def is_void(self) -> bool:
        """True when a T0/T1/T2 breach prevents any task-success credit."""
        return bool(self.violated_non_offsettable)

    @property
    def components(self) -> dict[str, float]:
        return {
            "reward": self.continuous.reward.total,
            "penalty": self.continuous.penalty.total,
            "warning": self.continuous.warning,
            "floor": self.floor,
            "final": self.final_score,
        }

    def explain(self) -> str:
        lines = [
            f"peer={self.peer} episode={self.episode} final={self.final_score:.4f}",
            f"  reward={self.continuous.reward.total:.4f} "
            f"({self.continuous.reward.as_dict()})",
            f"  penalty={self.continuous.penalty.total:.4f} "
            f"(raw={self.continuous.penalty.raw_total:.4f} "
            f"omega={self.continuous.penalty.omega:g})",
            f"  warning={self.continuous.warning:.4f}",
        ]
        if self.floor:
            lines.append(f"  floor={self.floor:.4f} ({self.floor_reason})")
        if self.suppressed:
            lines.append(
                "  suppressed=" + ", ".join(f"{item.rule_id}:{item.reason}" for item in self.suppressed)
            )
        if self.warnings_escalated:
            lines.append(
                "  escalated=" + ", ".join(record.rule_id for record in self.warnings_escalated)
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "peer": self.peer,
            "episode": self.episode,
            "final_score": self.final_score,
            "severity": self.severity.label,
            "void": self.is_void,
            "continuous": self.continuous.as_dict(),
            "floor": self.floor,
            "floor_reason": self.floor_reason,
            "suppressed": [item.__dict__ for item in self.suppressed],
            "non_offsettable": list(self.violated_non_offsettable),
            "warnings_issued": [record.describe() for record in self.warnings_issued],
            "warnings_escalated": [record.describe() for record in self.warnings_escalated],
            "metadata": dict(self.metadata),
        }


@dataclass
class EpisodeScorecard:
    """The scorecard for a whole episode across every peer."""

    episode: int
    peers: dict[str, PeerScorecard] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, card: PeerScorecard) -> None:
        self.peers[card.peer] = card

    def __getitem__(self, peer: str) -> PeerScorecard:
        return self.peers[peer]

    def __iter__(self):
        return iter(self.peers.values())

    @property
    def scores(self) -> dict[str, float]:
        return {peer: card.final_score for peer, card in self.peers.items()}

    @property
    def void_peers(self) -> list[str]:
        return [peer for peer, card in self.peers.items() if card.is_void]

    def ranking(self) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda item: (-item[1], item[0]))

    def summary(self) -> str:
        ordered = self.ranking()
        body = ", ".join(f"{peer}={score:.4f}" for peer, score in ordered) or "no peers"
        return f"episode {self.episode}: {body}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "peers": {peer: card.to_dict() for peer, card in self.peers.items()},
            "metadata": dict(self.metadata),
        }


def suppressed_from_rules(rules: tuple[Rule, ...], reason: str) -> tuple[SuppressedRule, ...]:
    return tuple(SuppressedRule(rule_id=rule.id, reason=reason) for rule in rules)


def highest_severity(ledger: Ledger) -> Severity:
    severities = [event.resolved_severity() for event in ledger.penalties()]
    return max(severities) if severities else Severity.NONE
