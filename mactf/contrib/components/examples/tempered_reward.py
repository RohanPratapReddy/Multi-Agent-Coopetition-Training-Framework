"""18.1: a reward stage that tempers credit.

Scale every reward by a factor in (0, 1], to study how much of a result is due to the reward
side. A *reward* stage is the one scorer kind where a different number is legitimate (6.1);
``ComplianceGuard`` still bounds it by ``COMPLIANCE.REWARD_CEILING`` (C008).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from mactf.components import register_scorer
from mactf.contrib.components.examples import registration_scope
from mactf.scoring.base import ScoreContext, ScorerOutput
from mactf.scoring.reward_scorer import RewardScorer


@register_scorer(
    "tempered_reward", version="1", description="RewardScorer scaled by a constant factor",
    scope=registration_scope(),
)
class TemperedReward(RewardScorer):
    def __init__(self, *, config: Mapping[str, Any] | None = None, factor: float = 0.5) -> None:
        super().__init__(config=config)
        if not 0.0 < factor <= 1.0:
            raise ValueError("factor must be in (0, 1]: a stage may temper a reward, not inflate it")
        self.factor = float(factor)

    def score(self, ctx: ScoreContext) -> ScorerOutput:
        base = super().score(ctx)
        breakdown = dict(base.breakdown)
        for bucket in self.BUCKETS:
            breakdown[bucket] = float(breakdown[bucket]) * self.factor
        breakdown["by_rule"] = {rule: float(value) * self.factor for rule, value in breakdown["by_rule"].items()}
        breakdown["total"] = float(breakdown["total"]) * self.factor
        return replace(base, amount=float(base.amount) * self.factor, breakdown=breakdown)


class InflatedReward(TemperedReward):
    """The bug the guard exists for: a copy of this stage with the range check removed."""

    def __init__(self, *, config: Mapping[str, Any] | None = None, factor: float = 1.0e6) -> None:
        super().__init__(config=config, factor=1.0)
        self.factor = float(factor)
