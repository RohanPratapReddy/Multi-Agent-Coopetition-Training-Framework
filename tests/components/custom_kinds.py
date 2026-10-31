"""New and identity components, one set per kind, for the override suites (document 17 section 19).

Nothing here registers on import (invariant I5): a test registers a class inside
``override_scope``. The ``Same*`` classes behave exactly like the built-in they are named
after, so a run that uses them must be bitwise identical to one that uses the built-in.
"""

from __future__ import annotations

import copy
import math
from typing import Any, ClassVar, Mapping

import torch

from mactf.coopetition.aggregation import Aggregator
from mactf.coopetition.strategy import KappaSoftmaxStrategy
from mactf.coopetition.weights import (
    KappaSoftmaxAverage,
    WeightStrategy,
    make_report,
)
from mactf.rl.functions.advantage import GAEAdvantage
from mactf.rl.functions.loss import PolicyGradientLoss
from mactf.rl.loss import LossBreakdown
from mactf.scoring.base import ScorerOutput
from mactf.scoring.penalty import PenaltyScorer
from mactf.scoring.reward_scorer import RewardScorer
from mactf.scoring.reward_terms import RewardTerm
from mactf.scoring.warning_scorer import WarningScorer
from mactf.training.policy import DefaultTrainPolicy
from mactf.training.protocol import StepResult

from p3_examples import StaggeredMixStrategy, TrimmedMeanRollup  # noqa: F401  (re-exported)


# -- trainer_policy ----------------------------------------------------------------------
class SameDefaultPolicy(DefaultTrainPolicy):
    """The default step under the default's own name."""

    name: ClassVar[str] = "default"


class AccumulatingPolicy(DefaultTrainPolicy):
    """The default step (it already honours ``host.backward`` and accumulation)."""

    name: ClassVar[str] = "accumulating"


class VoidAwarePolicy(DefaultTrainPolicy):
    """The default step with advantage masks pinned to ``void``."""

    name: ClassVar[str] = "void_aware"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("advantage_masks", "void")
        super().__init__(**kwargs)


class _LiveBase:
    """A policy that returns a live loss and lets the trainer step (``self_stepping=False``)."""

    name: ClassVar[str] = "live"
    unsafe: ClassVar[bool] = False
    self_stepping: ClassVar[bool | None] = False
    requires_terms: ClassVar[frozenset[str]] = frozenset({"compliance_floor"})
    composes_distillation: ClassVar[bool] = False

    def __init__(self) -> None:
        self.host: Any = None
        self.epochs_seen = 0

    def attach(self, host: Any) -> None:
        self.host = host

    def forward(self, node: Any, batch: Mapping[str, Any]):
        _, peer = self.host.resolve_node(node)
        logits, values = peer.adapter(batch["states"])
        return peer, logits, values

    def state_dict(self) -> dict[str, Any]:
        return {"epochs_seen": self.epochs_seen}

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        self.epochs_seen = int(payload.get("epochs_seen", 0))


class AnchoredCurriculumPolicy(_LiveBase):
    """A finite live loss, scaled by how many epochs have passed, with the compliance term evidenced."""

    name: ClassVar[str] = "anchored_curriculum"

    def train_step(self, node: Any, batch: Mapping[str, Any]) -> StepResult:
        _, logits, _ = self.forward(node, batch)
        floor = float(batch.get("floor", 0.0))
        compliance = max(-floor, 0.0)
        scale = 1.0 / (1.0 + self.epochs_seen)
        loss = logits.pow(2).mean() * scale + compliance
        return StepResult(
            loss=loss,
            breakdown=LossBreakdown(
                compliance=compliance, total=float(loss.detach()),
                extras={"advantage_mean": -1.0 if batch.get("void") else 0.1},
            ),
            stepped=True,
        )

    def on_epoch_end(self, tree: Any, result: Any) -> None:
        self.epochs_seen += 1

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "epochs_seen": self.epochs_seen}


class DistillingPolicy(AnchoredCurriculumPolicy):
    """Declares that it composes distillation (it owns a step that delegates to ``host.distill``)."""

    name: ClassVar[str] = "distilling"
    composes_distillation: ClassVar[bool] = True


class QuietPolicy(AnchoredCurriculumPolicy):
    """Evidences nothing it promised: used to prove S004 for a policy that requires a term."""

    name: ClassVar[str] = "quiet"

    def train_step(self, node: Any, batch: Mapping[str, Any]) -> StepResult:
        _, logits, _ = self.forward(node, batch)
        loss = logits.pow(2).mean()
        return StepResult(loss=loss, breakdown=LossBreakdown(total=float(loss.detach())), stepped=True)


class SelfSteppingLiar(AnchoredCurriculumPolicy):
    """Claims it steps itself and returns a live loss (S007)."""

    name: ClassVar[str] = "self_stepping_liar"
    self_stepping: ClassVar[bool | None] = True


class UnsafePolicy(AnchoredCurriculumPolicy):
    name: ClassVar[str] = "unsafe_curriculum"
    unsafe: ClassVar[bool] = True


# -- weight_update -----------------------------------------------------------------------
def _share(names, flags, raw: Mapping[str, float], *, kappa: float, adjustments=None):
    active = [n for n in names if flags[n]]
    total = sum(raw[n] for n in active)
    weights = {n: raw[n] / total for n in active}
    return make_report(weights, names, kappa=kappa, eligible=flags, adjustments=list(adjustments or []))


def _flags(names, eligible):
    return {n: bool(eligible.get(n, True)) if eligible else True for n in names}


class SameKappa(KappaSoftmaxAverage):
    """``kappa_softmax`` itself, reachable through a spec that carries a parameter."""

    name = "kappa_softmax"

    def __init__(self, aggregator: Aggregator | None = None, *, tag: str = "") -> None:
        super().__init__(aggregator)
        self.tag = str(tag)


class EntropyFloorAverage(WeightStrategy):
    """Uniform over the eligible, nudged toward the better scorers; remembers how often it ran."""

    name = "entropy_floor"

    def __init__(self, *, nudge: float = 0.2) -> None:
        if not 0.0 <= float(nudge) < 1.0:
            raise ValueError("nudge must be in [0, 1)")
        self.nudge = float(nudge)
        self.calls = 0

    def apply(self, vectors, scores, *, kappa, level=0, eligible=None, confidence=None):
        names = sorted(vectors)
        flags = _flags(names, eligible)
        self.calls += 1
        active = [n for n in names if flags[n]]
        if not active:
            return make_report({}, names, kappa=kappa, eligible=flags, fallback=True,
                               adjustments=["no eligible peers"])
        best = max(active, key=lambda n: float(scores.get(n, 0.0)))
        raw = {n: (1.0 - self.nudge) + (self.nudge * len(active) if n == best else 0.0) for n in active}
        return _share(names, flags, raw, kappa=kappa, adjustments=[f"entropy_floor call {self.calls}"])

    def state_dict(self) -> dict[str, Any]:
        return {"calls": self.calls}

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        self.calls = int(payload.get("calls", 0))

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "class": type(self).__name__, "nudge": self.nudge}


class RankWeightedAverage(WeightStrategy):
    """Half uniform, half rank share, over the eligible."""

    name = "rank_weighted"

    def apply(self, vectors, scores, *, kappa, level=0, eligible=None, confidence=None):
        names = sorted(vectors)
        flags = _flags(names, eligible)
        active = [n for n in names if flags[n]]
        if not active:
            return make_report({}, names, kappa=kappa, eligible=flags, fallback=True,
                               adjustments=["no eligible peers"])
        ranked = sorted(active, key=lambda n: (float(scores.get(n, 0.0)), n))
        denominator = len(active) * (len(active) + 1) / 2.0
        raw = {n: 0.5 / len(active) + 0.5 * (ranked.index(n) + 1) / denominator for n in active}
        return _share(names, flags, raw, kappa=kappa)


class UniformCopy(WeightStrategy):
    """``uniform`` under another name."""

    name = "uniform_copy"

    def apply(self, vectors, scores, *, kappa, level=0, eligible=None, confidence=None):
        names = sorted(vectors)
        flags = _flags(names, eligible)
        active = [n for n in names if flags[n]]
        if not active:
            return make_report({}, names, kappa=kappa, eligible=flags, fallback=True,
                               adjustments=["no eligible peers"])
        return _share(names, flags, {n: 1.0 for n in active}, kappa=kappa)


# -- peer_strategy -----------------------------------------------------------------------
class SameKappaStrategy(KappaSoftmaxStrategy):
    name = "kappa_softmax"


DeterministicStagger = StaggeredMixStrategy


# -- scorer ------------------------------------------------------------------------------
class SamePenalty(PenaltyScorer):
    pass


class SameReward(RewardScorer):
    pass


class SameWarning(WarningScorer):
    pass


class TemperedRewardScorer(RewardScorer):
    """A reward stage that pays ``scale`` of the built-in amount: never more, so the ceiling holds."""

    def __init__(self, *, config=None, scale: float = 0.5) -> None:
        super().__init__(config=config)
        if not 0.0 <= float(scale) <= 1.0:
            raise ValueError("scale must be in [0, 1]")
        self.scale = float(scale)

    def score(self, ctx):
        output = super().score(ctx)
        from dataclasses import replace

        buckets = {k: (v * self.scale if isinstance(v, (int, float)) else v)
                   for k, v in dict(output.breakdown).items()}
        return replace(output, amount=float(output.amount) * self.scale, breakdown=buckets)


# -- reward_term -------------------------------------------------------------------------
class DiminishingTerm(RewardTerm):
    """Square root of the magnitude (never negative); applies to every reward rule."""

    name = "diminishing"

    def __init__(self, *, rules=None) -> None:
        if rules:
            self.rule_ids = frozenset(rules)

    def contribution(self, event, ctx) -> float:
        return math.sqrt(max(float(event.magnitude), 0.0))


# -- rl_function -------------------------------------------------------------------------
class SamePolicyGradient(PolicyGradientLoss):
    pass


class SameGAE(GAEAdvantage):
    pass


# -- aggregator --------------------------------------------------------------------------
class DeterministicAggregator(Aggregator):
    """The built-in aggregator with one extra, round-tripped constructor option."""

    def __init__(self, *, tag: str = "t", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tag = str(tag)

    def options(self) -> dict[str, Any]:
        return {**super().options(), "tag": self.tag}


# -- peer_factory ------------------------------------------------------------------------
def byte_clone(spec: Any, source: torch.nn.Module | None = None) -> torch.nn.Module:
    if source is None:
        raise ValueError("byte_clone needs `like`")
    return copy.deepcopy(source)


def seeded_reclone(spec: Any, source: torch.nn.Module | None = None, *, std: float = 0.0) -> torch.nn.Module:
    if source is None:
        raise ValueError("seeded_reclone needs `like`")
    module = copy.deepcopy(source)
    if std:
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.add_(torch.randn_like(parameter) * float(std))
    return module


# -- scheduler ---------------------------------------------------------------------------
def cosine_with_floor(optimizer, *, total_steps: int = 100, floor: float = 0.1):
    if total_steps < 1 or not 0.0 <= floor <= 1.0:
        raise ValueError("total_steps must be >= 1 and floor in [0, 1]")

    def factor(step: int) -> float:
        progress = min(step, total_steps) / total_steps
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class SameAdam(torch.optim.Adam):
    pass


__all__ = [name for name in dir() if name[0].isupper() or name in ("byte_clone", "seeded_reclone", "cosine_with_floor")]
