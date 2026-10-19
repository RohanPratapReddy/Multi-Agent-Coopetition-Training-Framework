"""The scoring pipeline: ledger -> warning regime -> continuous score -> scorecard.

This module owns the ordering that makes scoring reproducible:

1. Validate and record rule events (``Ledger``).
2. Feed warning-eligible events into the ``WarningRegistry``.
3. Evaluate the docx continuous equations.
4. Apply framework.md precedence and the non-offsettable floor.
5. Emit a ``PeerScorecard``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .. import signals
from ..compliance import ComplianceResult, constrain, void_rule_ids
from ..rules.catalog import ALL_RULES, NON_OFFSETTABLE, get_rule
from ..rules.severity import Severity, weight_for
from .continuous import ContinuousScore, score as continuous_score
from .ledger import Ledger
from .scorecard import EpisodeScorecard, PeerScorecard, highest_severity, suppressed_from_rules
from .warning import EscalationOutcome, WarningPolicy, WarningRegistry

DEFAULT_KNOWN_RULES = {rule.id for rule in ALL_RULES if rule.is_penalty}


@dataclass
class Scorer:
    """Scores episodes and owns the per-peer warning registries."""

    score_config: dict[str, Any]
    severity_weights: dict[str, Any]
    warning_policy: WarningPolicy
    registries: dict[str, WarningRegistry] | None = None

    def __post_init__(self) -> None:
        if self.registries is None:
            self.registries = {}

    # -- registry helpers ------------------------------------------------
    def registry_for(self, peer: str) -> WarningRegistry:
        assert self.registries is not None
        if peer not in self.registries:
            self.registries[peer] = WarningRegistry(self.warning_policy)
        return self.registries[peer]

    def begin_episode(self, peer: str, *, episode: int) -> list[Any]:
        return self.registry_for(peer).begin_episode(peer, episode=episode)

    # -- scoring ---------------------------------------------------------
    def score_peer(
        self,
        ledger: Ledger,
        *,
        peer: str | None = None,
        episode: int = 0,
        known_rules: set[str] | None = None,
    ) -> PeerScorecard:
        peer = peer or ledger.peer or "peer"
        ledger.peer = peer
        registry = self.registry_for(peer)
        registry.begin_episode(peer, episode=episode)

        warnings_issued = []
        warnings_escalated = []
        warning_penalty = 0.0
        warning_events: list[str] = []
        escalated_rules: set[str] = set()

        for event in ledger.penalties():
            if not event.rule.warning_eligible:
                continue
            outcome: EscalationOutcome = registry.issue(
                peer,
                event.rule_id,
                severity=event.resolved_severity(),
                episode=episode,
                evidence=event.evidence,
                weights=self.severity_weights,
            )
            warnings_issued.extend(outcome.issued)
            warning_events.append(outcome.rule_id)
            if outcome.escalated is not None:
                warnings_escalated.append(outcome.escalated)
                warning_penalty += outcome.penalty
                escalated_rules.add(outcome.rule_id)

        registry.mark_clean_except(
            peer,
            set(warning_events),
            known_rules=known_rules or DEFAULT_KNOWN_RULES,
        )

        continuous = continuous_score(
            ledger,
            score_config=self.score_config,
            severity_weights=self.severity_weights,
            warning_total=warning_penalty,
            warning_events=tuple(warning_events),
            escalated_rules=escalated_rules,
        )

        non_offsettable = set(self.score_config.get("NON_OFFSETTABLE", NON_OFFSETTABLE))
        result: ComplianceResult = constrain(continuous, non_offsettable=non_offsettable)

        suppressed = ()
        if continuous.decision is not None:
            suppressed = suppressed_from_rules(
                continuous.decision.suppressed,
                f"not in controlling tier T{continuous.decision.controlling_tier}",
            )
            already = {rule.id for rule in continuous.decision.suppressed}
            suppressed += suppressed_from_rules(
                tuple(
                    get_rule(rule_id)
                    for rule_id in continuous.suppressed_rewards
                    if rule_id not in already
                ),
                "reward suppressed by precedence",
            )

        card = PeerScorecard(
            peer=peer,
            episode=episode,
            continuous=continuous,
            final_score=result.value,
            floor=result.floor,
            floor_reason=result.reason,
            warnings_issued=tuple(warnings_issued),
            warnings_escalated=tuple(warnings_escalated),
            suppressed=suppressed,
            violated_non_offsettable=void_rule_ids(continuous, non_offsettable=non_offsettable),
            severity=highest_severity(ledger),
            metadata={"compliance": result.as_dict()},
        )
        signals.episode_peer_scored.send(sender=self, card=card, ledger=ledger)
        return card

    def score_episode(
        self,
        ledgers: dict[str, Ledger],
        *,
        episode: int = 0,
        known_rules: set[str] | None = None,
    ) -> EpisodeScorecard:
        card = EpisodeScorecard(episode=episode)
        for peer, ledger in ledgers.items():
            card.add(
                self.score_peer(ledger, peer=peer, episode=episode, known_rules=known_rules)
            )
        signals.episode_scored.send(sender=self, scorecard=card)
        return card


def build_scorer(settings: Any) -> Scorer:
    """Construct a :class:`Scorer` from a Settings object."""
    policy = WarningPolicy.from_settings(settings["WARNINGS"])
    return Scorer(
        score_config=dict(settings["SCORING"]),
        severity_weights=dict(settings["SEVERITY"]["WEIGHTS"]),
        warning_policy=policy,
    )


def severity_weight(severity: Severity, weights: dict[str, Any]) -> float:
    """Public helper so callers do not reach into the severity module."""
    return weight_for(severity, weights)
