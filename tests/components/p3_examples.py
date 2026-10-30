"""The worked examples of document 17 sections 8.9.3, 10.11 and 11.12, as plain classes.

They are not registered here: a test registers them inside ``component_overlay`` so the
process registry is never touched.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, ClassVar, Mapping, Sequence

import torch

from mactf.coopetition.report import AggregationReport
from mactf.coopetition.strategy import PeerStrategy, StrategyTraits, TreeView
from mactf.coopetition.weights import GroupView, WeightStrategy, WeightTraits, make_report
from mactf.exceptions import ComplianceError, ValidationError
from mactf.rl.functions.rollup import _RollupBase
from mactf.scoring.base import ScorerOutput


# -- 10.11: a stateful, per-group weight strategy ------------------------------------
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
        return {"name": self.name, "class": type(self).__name__,
                "alpha": self.alpha, "clip": self.clip}

    def state_dict(self) -> dict[str, Any]:
        return {"ema": dict(sorted(self._ema.items()))}

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        self._ema = {str(k): float(v) for k, v in dict(payload.get("ema", {})).items()}


# -- 11.12: staggered mix --------------------------------------------------------------
class StaggeredMixStrategy(PeerStrategy):
    """Group ``g`` mixes in epoch ``e`` when ``(e + slot(g)) % period == 0``."""

    name = "staggered_mix"
    traits = StrategyTraits(aggregates=True, walk="tree")

    def __init__(self, *, period: int = 3) -> None:
        if isinstance(period, bool) or not isinstance(period, int) or period < 1:
            raise ValueError("period must be an integer >= 1")
        self.period = period
        self._slots: dict[str, int] = {}

    def validate(self, view: TreeView) -> list[str]:
        if view.hierarchy and len(view.groups) < 2:
            return ["a hierarchy with a single group has nothing to stagger"]
        return []

    def coordinate(self, ctx) -> None:
        for group in ctx.groups:
            slot = self._slots.setdefault(group.label, len(self._slots))
            if (ctx.epoch + slot) % self.period != 0:
                ctx.skip(group, f"not this group's turn (slot {slot % self.period} of {self.period})")
                continue
            eligible = ctx.eligible_for(group.members)
            if eligible is not None and not any(eligible.values()):
                ctx.skip(group, "every member is frozen")
            elif group.is_root:
                ctx.aggregate_root(ctx.update_root(eligible=eligible))
            elif ctx.peers_void(group.members):
                ctx.skip(group, "every member was voided this epoch")
            else:
                ctx.aggregate_group(group, eligible=eligible)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "period": self.period}

    def state_dict(self) -> dict[str, Any]:
        return {"class": type(self).__name__, "name": self.name, "slots": dict(self._slots)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._slots = {str(label): int(slot) for label, slot in dict(state.get("slots") or {}).items()}


# -- 8.9.3: a rollup that guards itself and passes the guard ------------------------------
class TrimmedMeanRollup(_RollupBase):
    """Mean of child amounts after dropping the highest and lowest ``trim``."""

    spec_type: ClassVar[str] = "trimmed_mean"
    name: ClassVar[str] = "trimmed_mean"

    def __init__(self, *, trim: int = 1, **base: Any) -> None:
        super().__init__(**base)
        if int(trim) < 0:
            raise ValidationError(f"trim must be >= 0, got {trim!r}")
        self.trim = int(trim)

    def rollup(
        self, parent: str, children: Sequence[ScorerOutput], *,
        weights: Mapping[str, float] | None = None, scores: Mapping[str, float] | None = None,
    ) -> ScorerOutput:
        resolved, _ = self._prepare(parent, children, weights)
        if not children:
            return self._output(parent, 0.0, children, weights=[], extras={"trim": self.trim})
        order = sorted(range(len(children)), key=lambda index: float(children[index].amount))
        keep = order[self.trim: len(order) - self.trim] if len(order) > 2 * self.trim else order
        amount = sum(float(children[index].amount) for index in keep) / len(keep)
        used = [resolved[index] if index in keep else 0.0 for index in range(len(children))]
        return self._output(
            parent, amount, children, weights=used,
            extras={"trim": self.trim, "kept": len(keep), "dropped": len(children) - len(keep)},
        )

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "trim": self.trim}


class MedianRollup(_RollupBase):
    """Median of the child amounts (a second trimming rollup for the property test)."""

    spec_type: ClassVar[str] = "median_rollup"
    name: ClassVar[str] = "median_rollup"

    def rollup(self, parent, children, *, weights=None, scores=None):
        self._prepare(parent, children, weights)
        amount = statistics.median(float(child.amount) for child in children) if children else 0.0
        return self._output(parent, amount, children, weights=[1.0] * len(children), extras={})
