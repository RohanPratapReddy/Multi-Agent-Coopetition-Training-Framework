"""18.2: a reward term with diminishing returns on one rule family.

A *term* is the unit for stopping one noisy reward rule dominating without replacing the
stage (6.3): it sees one event and returns a non-negative number.
"""

from __future__ import annotations

import math
from typing import Sequence

from mactf.components import register_reward_term
from mactf.contrib.components.examples import registration_scope
from mactf.exceptions import ValidationError
from mactf.scoring.base import ScoreContext, severity_fraction
from mactf.scoring.ledger import RuleEvent
from mactf.scoring.reward_terms import RewardTerm, _rules, _severity_table


@register_reward_term("sqrt_magnitude", version="1", scope=registration_scope())
class SqrtMagnitude(RewardTerm):
    """``sqrt(magnitude) * severity_fraction``: the tenth event matters less than the first."""

    name = "sqrt_magnitude"

    def __init__(self, *, scale: float = 1.0, rules: Sequence[str] | None = None) -> None:
        if not math.isfinite(scale) or scale < 0.0:
            raise ValidationError(f"sqrt_magnitude: 'scale' must be finite and >= 0, got {scale!r}")
        self.scale = float(scale)
        self.rule_ids = _rules(rules, owner="sqrt_magnitude")

    def contribution(self, event: RuleEvent, ctx: ScoreContext) -> float:
        fraction = severity_fraction(event.resolved_severity(), _severity_table(ctx), default=1.0)
        return self.scale * math.sqrt(max(float(event.magnitude), 0.0)) * fraction


class NegativeTerm(SqrtMagnitude):
    """A term that returns a negative number: a reward may not reduce a score."""

    name = "negative_term"

    def contribution(self, event: RuleEvent, ctx: ScoreContext) -> float:
        return -1.0
