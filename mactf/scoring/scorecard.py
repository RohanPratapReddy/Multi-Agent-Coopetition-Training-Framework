"""Scorecards: the per-peer, per-episode auditable scoring record."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..rules.catalog import Rule
from ..rules.severity import Severity
from ..wire import need, opt, strings
from .continuous import ContinuousScore
from .ledger import Ledger
from .warning import WarningRecord


@dataclass(frozen=True)
class SuppressedRule:
    rule_id: str
    reason: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SuppressedRule":
        return cls(
            rule_id=str(need("SuppressedRule", payload, "rule_id")),
            reason=str(need("SuppressedRule", payload, "reason")),
        )


class PointsView(dict):
    """The points breakdown as stored (a plain ``dict``) that also reads as attributes.

    ``card.points["final_points"]`` and ``card.points.final_points`` are the same
    value; equality, JSON encoding and ``dict(...)`` are those of the underlying dict.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


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
    # Free-form per-peer record. ``metadata["distillation"]`` (document 14 section
    # 12) is written by ``MACTFTrainer`` after the cohort step, never by a scorer:
    # scoring runs before the step, so a scorer could only report the previous one.
    metadata: dict[str, Any] = field(default_factory=dict)
    # framework.md 9.2 split. ``None`` means "not computed", which is honest;
    # a zero would claim the component was measured and found empty.
    competitive: float | None = None
    cooperative: float | None = None
    cooperative_denied: float = 0.0
    #: Non-offsettable rule ids inherited from descendants (document 10 section 5).
    branch_void: tuple[str, ...] = ()
    #: The 0-100 rubric projection, when the points layer is enabled (doc 11).
    points: dict[str, Any] | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "points" and isinstance(value, dict) and not isinstance(value, PointsView):
            value = PointsView(value)
        object.__setattr__(self, name, value)

    @property
    def final_points(self) -> float | None:
        """``points.final_points`` (0-100), or ``None`` when the points layer did not run."""
        return None if self.points is None else float(self.points["final_points"])

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

    @property
    def has_component_split(self) -> bool:
        """True when the cooperative pool reported a split for this card."""
        return self.competitive is not None and self.cooperative is not None

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
        if self.has_component_split:
            lines.append(
                f"  split: competitive={self.competitive:.4f} "
                f"cooperative={self.cooperative:.4f}"
                + (
                    f" (denied={self.cooperative_denied:.4f})"
                    if self.cooperative_denied
                    else ""
                )
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
            "warnings_issued": [_describe(record) for record in self.warnings_issued],
            "warnings_escalated": [_describe(record) for record in self.warnings_escalated],
            "competitive": self.competitive,
            "cooperative": self.cooperative,
            "cooperative_denied": self.cooperative_denied,
            "branch_void": list(self.branch_void),
            "points": dict(self.points) if self.points is not None else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PeerScorecard":
        """Inverse of :meth:`to_dict`.

        ``warnings_issued`` and ``warnings_escalated`` serialize through
        ``WarningRecord.describe()``, which is text, so they come back as the
        description strings: the JSON is the authority on what the server observed
        and a ``WarningRecord`` cannot be rebuilt from a sentence. ``to_dict``
        re-emits those strings unchanged, so the serialized form round-trips.
        ``void`` is derived from ``non_offsettable`` and is ignored here.
        """
        owner = "PeerScorecard"
        return cls(
            peer=str(need(owner, payload, "peer")),
            episode=int(need(owner, payload, "episode")),
            continuous=ContinuousScore.from_dict(need(owner, payload, "continuous")),
            final_score=float(need(owner, payload, "final_score")),
            floor=float(opt(payload, "floor", 0.0)),
            floor_reason=opt(payload, "floor_reason", None),
            warnings_issued=strings(payload.get("warnings_issued")),  # type: ignore[arg-type]
            warnings_escalated=strings(payload.get("warnings_escalated")),  # type: ignore[arg-type]
            suppressed=tuple(
                SuppressedRule.from_dict(item) for item in (payload.get("suppressed") or ())
            ),
            violated_non_offsettable=strings(payload.get("non_offsettable")),
            severity=Severity.parse(opt(payload, "severity", "NONE")),
            metadata=dict(payload.get("metadata") or {}),
            competitive=_maybe_float(opt(payload, "competitive", None)),
            cooperative=_maybe_float(opt(payload, "cooperative", None)),
            cooperative_denied=float(opt(payload, "cooperative_denied", 0.0)),
            branch_void=strings(payload.get("branch_void")),
            points=dict(payload["points"]) if payload.get("points") is not None else None,
        )


def _describe(record: Any) -> str:
    """A warning's description, whether it is a record or the text one already produced."""
    return record if isinstance(record, str) else record.describe()


def _maybe_float(value: Any) -> float | None:
    return None if value is None else float(value)


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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EpisodeScorecard":
        peers = need("EpisodeScorecard", payload, "peers")
        return cls(
            episode=int(need("EpisodeScorecard", payload, "episode")),
            peers={name: PeerScorecard.from_dict(card) for name, card in peers.items()},
            metadata=dict(payload.get("metadata") or {}),
        )


def suppressed_from_rules(rules: tuple[Rule, ...], reason: str) -> tuple[SuppressedRule, ...]:
    return tuple(SuppressedRule(rule_id=rule.id, reason=reason) for rule in rules)


def highest_severity(ledger: Ledger) -> Severity:
    severities = [event.resolved_severity() for event in ledger.penalties()]
    return max(severities) if severities else Severity.NONE
