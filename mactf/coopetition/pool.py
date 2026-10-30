"""The cooperative component of the score, bounded and reported separately.

framework.md 9.2 splits a peer's reward into two components:

* a **competitive** component, zero-sum across the cohort, and
* a **cooperative** component that is additive and not zero-sum: R10
  (synthesis), R11 (constructive critique), and R16 (cooperative assistance).

An unbounded additive component is farmable - R11 and R16 can be emitted many
times in one episode - so it is drawn from a per-episode pool with a per-peer
cap. The pool bounds the total without making it zero-sum.

The pool never changes ``final_score``. It reports an accounting split of the
reward side, which is what framework.md 9.2 requires and what keeps the
non-offsettable floor's arithmetic (and every existing score) untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from ..exceptions import ContractError, ImproperlyConfigured

if TYPE_CHECKING:  # pragma: no cover - import-time cycle avoidance
    from ..scoring.scorecard import PeerScorecard

#: framework.md 9.1/9.4: the awards that are cooperative rather than competitive.
COOPERATIVE_RULES: frozenset[str] = frozenset({"R10", "R11", "R16"})

#: Absorbs binary floating-point noise when deciding whether a request fits.
_TOLERANCE = 1e-12


@dataclass(frozen=True)
class CooperativeBudget:
    """The per-episode cooperative pool and its per-peer ceiling."""

    per_episode: float = 6.0  # total pool across all peers
    per_peer_cap: float = 3.0
    carry_over: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.per_episode) or self.per_episode < 0.0:
            raise ImproperlyConfigured(
                f"COOPERATIVE_BUDGET.PER_EPISODE must be a finite non-negative number, "
                f"got {self.per_episode!r}"
            )
        if not math.isfinite(self.per_peer_cap) or self.per_peer_cap < 0.0:
            raise ImproperlyConfigured(
                f"COOPERATIVE_BUDGET.PER_PEER_CAP must be a finite non-negative number, "
                f"got {self.per_peer_cap!r}"
            )

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any] | None) -> "CooperativeBudget":
        """Build a budget from a ``COOPERATIVE_BUDGET`` mapping, tolerating partial ones."""
        settings = settings or {}
        return cls(
            per_episode=float(settings.get("PER_EPISODE", 6.0)),
            per_peer_cap=float(settings.get("PER_PEER_CAP", 3.0)),
            carry_over=bool(settings.get("CARRY_OVER", False)),
        )


@dataclass(frozen=True)
class CooperativeAllocation:
    """The result of one allocation round.

    ``requested`` holds what each peer asked for (before gating or capping) and
    ``awards`` what it actually received, so the difference is auditable rather
    than silently rounded away.
    """

    awards: Mapping[str, float]
    granted: float
    denied: float  # requested but over budget
    requested: Mapping[str, float]
    reason: str | None = None

    def awarded_to(self, peer: str) -> float:
        return float(self.awards.get(peer, 0.0))

    def denied_for(self, peer: str) -> float:
        """How much of ``peer``'s request was withheld, including gating and capping."""
        requested = float(self.requested.get(peer, 0.0))
        return max(0.0, requested - self.awarded_to(peer))

    def as_dict(self) -> dict[str, Any]:
        return {
            "awards": {peer: float(value) for peer, value in self.awards.items()},
            "granted": self.granted,
            "denied": self.denied,
            "requested": {peer: float(value) for peer, value in self.requested.items()},
            "reason": self.reason,
        }


class CooperativePool:
    """Bounds and distributes the cooperative reward for one episode.

    ``allocate`` is order-invariant: the same requests in any dict order produce
    the same awards, because every summation runs over a sorted peer list and
    the reduction is proportional rather than first-come-first-served
    (framework.md 10.1.10).
    """

    def __init__(self, budget: CooperativeBudget | None = None) -> None:
        self.budget = budget or CooperativeBudget()
        self._carry = 0.0

    # -- budget bookkeeping ----------------------------------------------
    @property
    def carry(self) -> float:
        """Unspent budget carried into the current episode."""
        return self._carry

    @property
    def available(self) -> float:
        """The total this episode can grant."""
        return self.budget.per_episode + (self._carry if self.budget.carry_over else 0.0)

    def reset(self) -> None:
        self._carry = 0.0

    # -- allocation ------------------------------------------------------
    def allocate(
        self,
        requested: Mapping[str, float],
        *,
        eligible: Mapping[str, bool] | None = None,
    ) -> CooperativeAllocation:
        """Grant cooperative reward, in the fixed order the plan mandates.

        1. Zero every ineligible peer's request (participation gating).
        2. Cap each peer at ``per_peer_cap``.
        3. Grant in full when the total fits.
        4. Otherwise scale every grant by ``available / total``.
        5. Report the withheld amount as ``denied``.
        """
        eligible = eligible or {}
        original: dict[str, float] = {}
        effective: dict[str, float] = {}

        for peer in sorted(requested):
            amount = float(requested[peer])
            if not math.isfinite(amount):
                raise ContractError(
                    f"cooperative request for {peer!r} is not finite: {amount!r}"
                )
            if amount < 0.0:
                raise ContractError(
                    f"cooperative request for {peer!r} must be non-negative, got {amount!r}"
                )
            original[peer] = amount
            if not bool(eligible.get(peer, True)):
                effective[peer] = 0.0
            else:
                effective[peer] = min(amount, self.budget.per_peer_cap)

        # Sorted iteration makes the sums bit-identical regardless of input order.
        total = math.fsum(effective[peer] for peer in sorted(effective))
        available = self.available

        reason: str | None = None
        if total <= available + _TOLERANCE:
            awards = {peer: effective[peer] for peer in sorted(effective)}
        else:
            scale = available / total if total > 0.0 else 0.0
            awards = {peer: effective[peer] * scale for peer in sorted(effective)}
            reason = (
                f"scaled by {scale:.6f}: requested {total:.6f} exceeds pool {available:.6f}"
            )

        granted = math.fsum(awards[peer] for peer in sorted(awards))
        requested_total = math.fsum(original[peer] for peer in sorted(original))
        denied = max(0.0, requested_total - granted)

        if self.budget.carry_over:
            # Only the granted amount leaves the pool; the rest rolls forward.
            self._carry = max(0.0, available - granted)

        return CooperativeAllocation(
            awards=awards,
            granted=granted,
            denied=denied,
            requested=original,
            reason=reason,
        )

    # -- reporting -------------------------------------------------------
    def split(
        self,
        card: "PeerScorecard",
        allocation: CooperativeAllocation,
    ) -> tuple[float, float]:
        """Return ``(competitive, cooperative)`` for one scored peer.

        ``cooperative`` is the granted amount, so it reflects the budget.
        ``competitive`` is the non-cooperative remainder of the reward. The two
        sum to ``reward.total`` exactly whenever the request was granted in
        full; when the pool withheld something, ``reward.total - competitive``
        exceeds ``cooperative`` by ``allocation.denied_for(card.peer)``.
        """
        reward = card.continuous.reward
        cooperative = allocation.awarded_to(card.peer)
        requested = cooperative_total(reward)
        if not requested:
            # A hand-built breakdown without per-rule accounting: fall back to
            # what was asked of the pool, which is the same number by construction.
            requested = float(allocation.requested.get(card.peer, 0.0))
        competitive = float(reward.total) - requested
        return competitive, cooperative

    def describe(self) -> str:
        kind = "carrying" if self.budget.carry_over else "resetting"
        return (
            f"CooperativePool(per_episode={self.budget.per_episode:g}, "
            f"per_peer_cap={self.budget.per_peer_cap:g}, {kind})"
        )


def cooperative_total(reward: Any) -> float:
    """Sum the cooperative rules' weighted contributions from a reward breakdown.

    Falls back to zero when a reward breakdown predates per-rule accounting, so a
    synthetic breakdown never crashes the split.
    """
    rule_total = getattr(reward, "rule_total", None)
    if callable(rule_total):
        return math.fsum([float(rule_total(COOPERATIVE_RULES))])
    by_rule = getattr(reward, "by_rule", None) or {}
    return math.fsum(
        float(by_rule[rule_id])
        for rule_id in sorted(COOPERATIVE_RULES)
        if rule_id in by_rule
    )


def budget_from_settings(settings: Any) -> CooperativeBudget | None:
    """Build the configured budget, or ``None`` when the split is disabled."""
    coop = settings["COOPETITION"] if "COOPETITION" in settings else {}
    if not bool(coop.get("COOPERATIVE_POOL", True)):
        return None
    return CooperativeBudget.from_settings(coop.get("COOPERATIVE_BUDGET"))
