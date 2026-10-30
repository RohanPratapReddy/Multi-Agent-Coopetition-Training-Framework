"""18.3: a weight-update strategy, per group, with state (the module of 10.11).

An exponential moving average of each peer's score, centred on the median, clipped, then a
softmax at the controller's temperature.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping

import torch

from mactf.components import register_weight_update
from mactf.contrib.components.examples import registration_scope
from mactf.coopetition.report import AggregationReport
from mactf.coopetition.weights import GroupView, WeightStrategy, WeightTraits, make_report
from mactf.exceptions import ComplianceError


@register_weight_update("ema_softmax", version="1", scope=registration_scope())
class EmaSoftmax(WeightStrategy):
    """Softmax over EMA-smoothed, median-centred, clipped scores."""

    name = "ema_softmax"
    traits = WeightTraits()

    def __init__(self, *, alpha: float = 0.3, clip: float = 3.0) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not clip > 0.0:
            raise ValueError("clip must be > 0")
        self.alpha = float(alpha)
        self.clip = float(clip)
        self._ema: dict[str, float] = {}

    def validate(self, group: GroupView) -> list[str]:
        if len(group.members) < 2:
            return [f"needs at least two peers, group {group.label!r} has {len(group.members)}"]
        return []

    def apply(
        self,
        vectors: Mapping[str, torch.Tensor],
        scores: Mapping[str, float],
        *,
        kappa: float,
        level: int = 0,
        eligible: Mapping[str, bool] | None = None,
        confidence: Mapping[str, float] | None = None,
    ) -> AggregationReport:
        names = sorted(vectors)
        if not names:
            raise ComplianceError("cannot aggregate an empty peer set")
        flags = {name: bool(eligible.get(name, True)) if eligible else True for name in names}
        for name in names:
            if flags[name]:
                current = float(scores.get(name, 0.0))
                prior = self._ema.get(name, current)
                self._ema[name] = (1.0 - self.alpha) * prior + self.alpha * current
        active = [name for name in names if flags[name]]
        if not active:
            return make_report(
                {}, names, kappa=kappa, eligible=flags, fallback=True,
                adjustments=["no eligible peers; nothing to aggregate"],
            )
        values = [self._ema[name] for name in active]
        centre = statistics.median(values)
        deviations = [max(-self.clip, min(self.clip, value - centre)) for value in values]
        top = max(deviations)
        exps = [math.exp(float(kappa) * (value - top)) for value in deviations]
        total = sum(exps)
        weights = {name: exp / total for name, exp in zip(active, exps)}
        return make_report(
            weights, names, kappa=kappa, eligible=flags,
            adjustments=[f"ema_softmax alpha={self.alpha:g} clip={self.clip:g}"],
        )

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "class": type(self).__name__, "alpha": self.alpha, "clip": self.clip}

    def state_dict(self) -> dict[str, Any]:
        return {"ema": dict(sorted(self._ema.items()))}

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        self._ema = {str(k): float(v) for k, v in dict(payload.get("ema", {})).items()}


class SumsToThree(WeightStrategy):
    """The misbehaviour: weights that do not sum to one. ``WeightGuard`` answers with U005."""

    name = "sums_to_three"

    def apply(self, vectors, scores, *, kappa, level=0, eligible=None, confidence=None):
        names = sorted(vectors)
        return make_report({n: 3.0 / len(names) for n in names}, names, kappa=kappa, eligible=eligible)
