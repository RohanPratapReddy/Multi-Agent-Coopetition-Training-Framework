"""Stages that switch a kind off on purpose (document 17 section 6.8)."""

from __future__ import annotations

from typing import Any

from ..components import HUB
from .base import ScoreContext, ScorerOutput
from .reward_scorer import RewardScorer
from .warning_scorer import WarningScorer


class ZeroRewardScorer(RewardScorer):
    """Every reward is zero: training on penalties alone, or an ablation of the reward side.

    Zero is a legal reward, so the guard accepts it. The breakdown keeps the weights so
    ``RewardBreakdown`` renders as it does when strict precedence suppresses the reward
    (``pipeline.py:532-537``).
    """

    def score(self, ctx: ScoreContext) -> ScorerOutput:
        weights = self.bucket_weights()
        breakdown: dict[str, Any] = {
            **dict.fromkeys(self.BUCKETS, 0.0),
            "weights": dict(weights),
            "rules": [],
            "by_rule": {},
            "ignored_rules": [],
            "total": 0.0,
        }
        return ScorerOutput(kind=self.kind, amount=0.0, breakdown=breakdown, rules=())


class NoWarningScorer(WarningScorer):
    """The warning-first regime off: every offence is scored at full penalty.

    This only ever *raises* a score's penalty side (a rule that would have been
    warning-only now scores), so it cannot weaken compliance. It subclasses
    ``WarningScorer`` so the registries, ``begin_episode`` and the checkpoint keep working;
    its registries stay empty.
    """

    def prepare(self, ctx: ScoreContext) -> ScoreContext:
        return ctx.evolve(
            warning_total=0.0,
            warning_events=(),
            escalated_rules=frozenset(),
            extras={**dict(ctx.extras), "warning_mutations": ()},
        )


HUB.registry("scorer").register_builtin("zero_reward", ZeroRewardScorer)
HUB.registry("scorer").register_builtin("no_warnings", NoWarningScorer)
