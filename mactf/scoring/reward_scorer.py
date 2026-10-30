"""The reward stage: ``R = alpha*r_task + beta*r_epi + gamma*r_collab + delta*r_align``.

This is :func:`mactf.scoring.continuous._reward_breakdown` with a public name and
a reportable shape. It reads reward events only, never penalty events and never
warning state, so a custom reward scorer cannot launder a breach by claiming it
did not see one.

Precedence suppression is *not* done here. ``ScoringPipeline`` applies
``rules.precedence.resolve`` and zeroes the reward when the controlling tier has
no reward rule, reproducing ``continuous.py:217-219``. Suppressing inside the
stage would let a custom reward scorer bypass tier precedence.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..exceptions import ImproperlyConfigured
from ..rules.severity import Severity
from .base import ScoreContext, Scorer, ScorerDiagnostic, ScorerOutput, ScoringKind, severity_fraction

#: The four docx reward buckets, in the order ``RewardBreakdown`` declares them.
BUCKETS: tuple[str, ...] = ("r_task", "r_epi", "r_collab", "r_align")

#: ``continuous.py``'s fallback when ``SCORING.REWARD_WEIGHTS`` is absent.
DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "r_task": 1.0,
    "r_epi": 1.5,
    "r_collab": 1.2,
    "r_align": 1.0,
}


class RewardScorer(Scorer):
    """Scores reward events into the four docx buckets."""

    kind = ScoringKind.REWARD
    order = 10
    requires = frozenset({"ledger", "config"})

    BUCKETS = BUCKETS

    # -- config ----------------------------------------------------------
    @property
    def weights(self) -> dict[str, float]:
        raw = self.config.get("REWARD_WEIGHTS") or DEFAULT_REWARD_WEIGHTS
        return {key: float(value) for key, value in raw.items()}

    def bucket_weights(self) -> dict[str, float]:
        """The weight applied to each bucket, defaulting an absent entry to 1.0."""
        weights = self.weights
        return {bucket: float(weights.get(bucket, 1.0)) for bucket in self.BUCKETS}

    # -- scoring ---------------------------------------------------------
    def score(self, ctx: ScoreContext) -> ScorerOutput:
        weights = self.bucket_weights()
        buckets = dict.fromkeys(self.BUCKETS, 0.0)
        by_rule: dict[str, float] = {}
        fired: list[str] = []
        ignored: list[str] = []
        severity_table = ctx.severity_weights or self.config.get("SEVERITY_WEIGHTS") or {}

        for event in ctx.ledger.rewards():
            bucket = event.rule.bucket
            if bucket not in buckets:
                # The rule is a reward but feeds no docx bucket. Record it rather
                # than dropping it silently, as continuous.py:126-127 does.
                ignored.append(event.rule_id)
                continue
            scale = severity_fraction(
                event.resolved_severity(), severity_table, default=1.0
            )
            contribution = event.magnitude * scale
            buckets[bucket] += contribution
            by_rule[event.rule_id] = by_rule.get(event.rule_id, 0.0) + weights[bucket] * contribution
            fired.append(event.rule_id)

        total = sum(weights[bucket] * buckets[bucket] for bucket in self.BUCKETS)
        rules = tuple(dict.fromkeys(fired))
        breakdown = {
            **buckets,
            "weights": dict(weights),
            "rules": list(rules),
            "by_rule": dict(by_rule),
            "ignored_rules": list(dict.fromkeys(ignored)),
            "total": total,
        }
        return ScorerOutput(
            kind=self.kind,
            amount=total,
            breakdown=breakdown,
            rules=rules,
            metadata={"ignored_rules": tuple(dict.fromkeys(ignored))},
        )

    # -- introspection ---------------------------------------------------
    def validate(self, settings: Mapping[str, Any]) -> list[ScorerDiagnostic]:
        diagnostics: list[ScorerDiagnostic] = []
        raw = settings.get("REWARD_WEIGHTS", self.config.get("REWARD_WEIGHTS")) or {}
        for bucket, value in raw.items():
            if bucket not in BUCKETS:
                diagnostics.append(
                    ScorerDiagnostic(
                        level="warning",
                        code="rewards.unknown_bucket",
                        message=f"REWARD_WEIGHTS has an entry for unknown bucket {bucket!r}",
                        hint=f"valid buckets are {', '.join(BUCKETS)}",
                    )
                )
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                diagnostics.append(
                    ScorerDiagnostic(
                        level="error",
                        code="rewards.non_numeric_weight",
                        message=f"REWARD_WEIGHTS[{bucket!r}] is not numeric: {value!r}",
                    )
                )
                continue
            if number < 0.0:
                # Reward is non-negative by definition; a negative weight would
                # let a reward event reduce the score.
                diagnostics.append(
                    ScorerDiagnostic(
                        level="error",
                        code="rewards.negative_weight",
                        message=f"REWARD_WEIGHTS[{bucket!r}] is negative: {number!r}",
                        hint="reward weights must be non-negative",
                    )
                )
        return diagnostics

    def assert_valid(self, settings: Mapping[str, Any] | None = None) -> None:
        """Raise when configuration is unusable, naming every problem."""
        problems = [
            diagnostic
            for diagnostic in self.validate(settings or self.config)
            if diagnostic.level == "error"
        ]
        if problems:
            raise ImproperlyConfigured(
                "; ".join(f"{item.code}: {item.message}" for item in problems)
            )


def reward_total_from_events(
    ctx: ScoreContext,
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """The reward total for a context, without constructing a scorer.

    Used by the cooperative pool's cohort pre-pass and by tests that need the
    reward number without a full stage walk.
    """
    scorer = RewardScorer(config={"REWARD_WEIGHTS": dict(weights or DEFAULT_REWARD_WEIGHTS)})
    return scorer.score(ctx).amount


def severity_scale(severity: Severity | str, table: Mapping[str, Any]) -> float:
    """Public helper mirroring the stage's severity handling."""
    return severity_fraction(Severity.parse(severity), table, default=1.0)
