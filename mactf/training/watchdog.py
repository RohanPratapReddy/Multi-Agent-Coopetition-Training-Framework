"""The training watchdog: probes, health reports, and bounded actions.

The watchdog observes, snapshots, and stops or rewinds. It never edits a
hyperparameter, a learning rate, or a weight (document 09 section 3.6). Any
adjustment is emitted as a recommendation string for a caller to act on, because an
automatic learning-rate change triggered by a loss spike makes an experiment
unreproducible and hides the spike.

``evaluate`` is side-effect free and ``tick`` is ``evaluate`` plus actions. The API
needs a dry read (``GET /runs/{id}/watchdog`` must not take a snapshot), and both
paths share one probe evaluation so a report read over HTTP is the report the
trainer saw.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from ..exceptions import ImproperlyConfigured

__all__ = [
    "AggregationFallbackProbe",
    "CheckpointAgeProbe",
    "ChildStarvationProbe",
    "DeadBranchProbe",
    "DepthCostProbe",
    "EntropyProbe",
    "GradNormProbe",
    "HealthContext",
    "HealthProbe",
    "HealthReport",
    "InfLossProbe",
    "KappaProbe",
    "LossTrendProbe",
    "MACTFWatchdog",
    "MemoryProbe",
    "NaNLossProbe",
    "ProbeResult",
    "ScoreTrendProbe",
    "StaleProgressProbe",
    "ThroughputProbe",
    "TreeGrowthProbe",
    "WarningSpikeProbe",
    "WatchdogAction",
    "WatchdogPolicy",
    "WatchdogStatus",
    "default_probes",
]


class WatchdogStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"healthy": 0, "degraded": 1, "critical": 2}[self.value]

    @classmethod
    def worst(cls, statuses: Iterable["WatchdogStatus"]) -> "WatchdogStatus":
        chosen = cls.HEALTHY
        for status in statuses:
            if status.rank > chosen.rank:
                chosen = status
        return chosen


class WatchdogAction(str, Enum):
    NONE = "none"
    LOG = "log"
    SNAPSHOT = "snapshot"
    ROLLBACK = "rollback"
    PRUNE_BRANCH = "prune_branch"
    PAUSE = "pause"
    ABORT = "abort"

    @property
    def stops_training(self) -> bool:
        return self in (WatchdogAction.PAUSE, WatchdogAction.ABORT)


#: The actions that may be configured for a degradation or a critical status.
ACTION_NAMES = tuple(action.value for action in WatchdogAction)


@dataclass(frozen=True)
class WatchdogPolicy:
    """Every threshold, validated at construction.

    A watchdog that never fires because of a typo is worse than no watchdog,
    because it produces false confidence. So each bound is checked here rather than
    discovered at epoch 9.
    """

    # loss
    nan_policy: str = "abort"
    inf_policy: str = "abort"
    loss_increase_ratio: float = 3.0
    loss_patience: int = 2
    # gradients
    max_grad_norm: float = 100.0
    grad_norm_patience: int = 1
    # scores
    score_regression_tolerance: float = 0.10
    score_patience: int = 3
    # entropy and kappa
    entropy_collapse_ratio: float = 0.05
    entropy_patience: int = 2
    kappa_min: float = 1e-3
    kappa_max: float = 1e3
    # progress
    stale_epoch_seconds: float = 1800.0
    min_throughput_ratio: float = 0.25
    max_step_time_drift: float = 5.0
    # resources
    max_rss_growth_ratio: float = 2.0
    max_gpu_memory_ratio: float = 0.95
    # checkpoints and warnings
    max_checkpoint_age_seconds: float = 3600.0
    warning_spike_threshold: int = 10
    # hierarchy
    max_tree_growth_ratio: float = 2.0
    dead_branch_patience: int = 2
    child_starvation_patience: int = 3
    depth_cost_ratio: float = 4.0

    def __post_init__(self) -> None:
        for name in ("nan_policy", "inf_policy"):
            value = getattr(self, name)
            if value not in ("abort", "rollback", "ignore"):
                raise ImproperlyConfigured(
                    f"WATCHDOG.{name.upper()} must be 'abort', 'rollback', or 'ignore', "
                    f"got {value!r}"
                )
        if self.loss_increase_ratio < 1.0:
            raise ImproperlyConfigured(
                f"WATCHDOG.LOSS_INCREASE_RATIO must be >= 1.0, got {self.loss_increase_ratio!r}"
            )
        for name in (
            "loss_patience",
            "grad_norm_patience",
            "score_patience",
            "entropy_patience",
            "dead_branch_patience",
            "child_starvation_patience",
            "warning_spike_threshold",
        ):
            value = getattr(self, name)
            if int(value) < 1:
                raise ImproperlyConfigured(
                    f"WATCHDOG.{name.upper()} must be >= 1, got {value!r}"
                )
        if self.max_grad_norm <= 0:
            raise ImproperlyConfigured(
                f"WATCHDOG.MAX_GRAD_NORM must be > 0, got {self.max_grad_norm!r}"
            )
        if self.score_regression_tolerance < 0:
            raise ImproperlyConfigured(
                "WATCHDOG.SCORE_REGRESSION_TOLERANCE must be >= 0"
            )
        if not 0.0 <= self.entropy_collapse_ratio <= 1.0:
            raise ImproperlyConfigured(
                "WATCHDOG.ENTROPY_COLLAPSE_RATIO must be in [0, 1]"
            )
        if self.kappa_min <= 0 or self.kappa_max < self.kappa_min:
            raise ImproperlyConfigured(
                "WATCHDOG.KAPPA_MIN must be > 0 and KAPPA_MAX >= KAPPA_MIN"
            )
        if self.stale_epoch_seconds <= 0:
            raise ImproperlyConfigured("WATCHDOG.STALE_EPOCH_SECONDS must be > 0")
        if not 0.0 < self.min_throughput_ratio <= 1.0:
            raise ImproperlyConfigured(
                "WATCHDOG.MIN_THROUGHPUT_RATIO must be in (0, 1]"
            )
        if self.max_step_time_drift <= 0:
            raise ImproperlyConfigured("WATCHDOG.MAX_STEP_TIME_DRIFT must be > 0")
        if self.max_rss_growth_ratio < 1.0:
            raise ImproperlyConfigured("WATCHDOG.MAX_RSS_GROWTH_RATIO must be >= 1.0")
        if not 0.0 < self.max_gpu_memory_ratio <= 1.0:
            raise ImproperlyConfigured(
                "WATCHDOG.MAX_GPU_MEMORY_RATIO must be in (0, 1]"
            )
        if self.max_checkpoint_age_seconds <= 0:
            raise ImproperlyConfigured(
                "WATCHDOG.MAX_CHECKPOINT_AGE_SECONDS must be > 0"
            )
        if self.max_tree_growth_ratio < 1.0:
            raise ImproperlyConfigured("WATCHDOG.MAX_TREE_GROWTH_RATIO must be >= 1.0")
        if self.depth_cost_ratio <= 1.0:
            raise ImproperlyConfigured("WATCHDOG.DEPTH_COST_RATIO must be > 1.0")

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any] | Any) -> "WatchdogPolicy":
        block = _block(settings, "WATCHDOG")
        if not block:
            return cls()
        accepted = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        provided: dict[str, Any] = {}
        for key, value in block.items():
            lowered = str(key).lower()
            if lowered not in accepted:
                continue
            provided[lowered] = value
        return cls(**provided)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _block(settings: Any, name: str) -> dict[str, Any]:
    if settings is None:
        return {}
    if isinstance(settings, Mapping):
        block = settings.get(name)
        return dict(block) if isinstance(block, Mapping) else {}
    getter = getattr(settings, "get", None)
    if callable(getter):
        block = getter(name, None)
        if isinstance(block, Mapping):
            return dict(block)
    block = getattr(settings, name, None)
    return dict(block) if isinstance(block, Mapping) else {}


@dataclass(frozen=True)
class ProbeResult:
    """One probe's verdict."""

    name: str
    status: WatchdogStatus
    value: float | None = None
    threshold: float | None = None
    detail: str = ""
    node_path: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_problem(self) -> bool:
        return self.status is not WatchdogStatus.HEALTHY

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "value": self.value,
            "threshold": self.threshold,
            "detail": self.detail,
            "node_path": self.node_path,
            "metadata": dict(self.metadata),
        }

    def describe(self) -> str:
        text = f"{self.name}: {self.status.value}"
        if self.value is not None:
            text += f" value={self.value:.6g}"
        if self.threshold is not None:
            text += f" threshold={self.threshold:.6g}"
        if self.detail:
            text += f" ({self.detail})"
        return text


@dataclass
class HealthContext:
    """Everything the probes read. A plain object so a caller can build one.

    Every field is optional: a probe sees what exists and returns HEALTHY for what
    it cannot judge. That is deliberate — a probe that guesses is worse than one
    that abstains.
    """

    epoch: int = 0
    step: int = 0
    losses: Sequence[float] = ()
    grad_norms: Sequence[float] = ()
    scores: Mapping[str, float] = field(default_factory=dict)
    previous_scores: Mapping[str, float] = field(default_factory=dict)
    entropy: float | None = None
    max_entropy: float | None = None
    kappa: float | None = None
    aggregation_reports: Sequence[Any] = ()
    warnings_escalated: int = 0
    last_checkpoint_at: float | None = None
    checkpoint_count: int = 0
    epoch_seconds: Sequence[float] = ()
    steps_per_second: float | None = None
    previous_steps_per_second: float | None = None
    rss_bytes: int | None = None
    initial_rss_bytes: int | None = None
    gpu_memory_ratio: float | None = None
    tree_nodes: int | None = None
    initial_tree_nodes: int | None = None
    dead_branches: Mapping[str, int] = field(default_factory=dict)
    starved_children: Mapping[str, int] = field(default_factory=dict)
    level_durations: Mapping[int, float] = field(default_factory=dict)
    now: float | None = None
    monotonic: float | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)
    #: First index of ``losses`` measured on the current topology. ``LossTrendProbe`` judges only
    #: ``losses[loss_baseline:]``; the NaN and Inf probes still read all of ``losses`` (12.8.2).
    loss_baseline: int = 0
    #: Peers that are retired (still in the tree, out of play). ``tree_nodes`` counts only the others.
    retired_nodes: int = 0
    #: How many plans have committed, and the topology epoch they reached. Both 0 for a run that never mutated.
    mutations: int = 0
    topology_epoch: int = 0
    #: ``pruner(path, dead_epochs) -> nodes removed``. Set by a trainer; ``None`` for a context a
    #: caller built by hand, in which case ``prune_branch`` falls back to ``extras["tree"]``.
    pruner: Callable[[str, int], int] | None = None

    def copies(self, **changes: Any) -> "HealthContext":
        payload = dict(self.__dict__)
        payload.update(changes)
        return HealthContext(**payload)


class HealthProbe(Protocol):
    """A single health judgement. Must be side-effect free."""

    name: str

    def probe(self, context: HealthContext) -> ProbeResult: ...


class _BaseProbe:
    name = "probe"

    def __init__(self, policy: WatchdogPolicy) -> None:
        self.policy = policy

    def probe(self, context: HealthContext) -> ProbeResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def _healthy(self, value: float | None = None, **kwargs: Any) -> ProbeResult:
        return ProbeResult(name=self.name, status=WatchdogStatus.HEALTHY, value=value, **kwargs)


class NaNLossProbe(_BaseProbe):
    name = "nan_loss"

    def probe(self, context: HealthContext) -> ProbeResult:
        for index, value in enumerate(context.losses):
            if value is not None and math.isnan(float(value)):
                return ProbeResult(
                    name=self.name,
                    status=WatchdogStatus.CRITICAL,
                    value=float("nan"),
                    detail=f"NaN loss at epoch {index}",
                    metadata={"policy": self.policy.nan_policy},
                )
        return self._healthy()


class InfLossProbe(_BaseProbe):
    name = "inf_loss"

    def probe(self, context: HealthContext) -> ProbeResult:
        for index, value in enumerate(context.losses):
            if value is not None and math.isinf(float(value)):
                return ProbeResult(
                    name=self.name,
                    status=WatchdogStatus.CRITICAL,
                    value=float("inf"),
                    detail=f"infinite loss at epoch {index}",
                    metadata={"policy": self.policy.inf_policy},
                )
        return self._healthy()


class GradNormProbe(_BaseProbe):
    name = "grad_norm"

    def probe(self, context: HealthContext) -> ProbeResult:
        norms = [float(value) for value in context.grad_norms if value is not None]
        if not norms:
            return self._healthy()
        window = norms[-max(1, self.policy.grad_norm_patience):]
        above = [value for value in window if value > self.policy.max_grad_norm]
        if not above:
            return self._healthy(norms[-1], threshold=self.policy.max_grad_norm)
        worst = max(above)
        status = (
            WatchdogStatus.CRITICAL
            if len(above) >= len(window) and worst >= 10.0 * self.policy.max_grad_norm
            else WatchdogStatus.DEGRADED
        )
        return ProbeResult(
            name=self.name,
            status=status,
            value=worst,
            threshold=self.policy.max_grad_norm,
            detail=f"{len(above)} of the last {len(window)} grad norms exceeded the limit",
        )


class LossTrendProbe(_BaseProbe):
    name = "loss_trend"

    def probe(self, context: HealthContext) -> ProbeResult:
        losses = [float(value) for value in context.losses][max(0, int(context.loss_baseline)):]
        if len(losses) < 2:
            return self._healthy()
        history = losses[:-1]
        window = sorted(history[-5:])
        middle = len(window) // 2
        median = (
            window[middle] if len(window) % 2 else (window[middle - 1] + window[middle]) / 2.0
        )
        current = losses[-1]
        if median == 0:
            ratio = 1.0 if current == 0 else float("inf")
        else:
            ratio = abs(current) / abs(median)
        if ratio >= 10.0:
            return ProbeResult(
                name=self.name,
                status=WatchdogStatus.CRITICAL,
                value=current,
                threshold=median * self.policy.loss_increase_ratio,
                detail=f"loss is {ratio:.2f}x the trailing median {median:.6g}",
            )
        if ratio > self.policy.loss_increase_ratio:
            consecutive = 0
            for value in reversed(losses):
                if median and abs(value) / abs(median) > self.policy.loss_increase_ratio:
                    consecutive += 1
                else:
                    break
            if consecutive >= self.policy.loss_patience:
                return ProbeResult(
                    name=self.name,
                    status=WatchdogStatus.DEGRADED,
                    value=current,
                    threshold=median * self.policy.loss_increase_ratio,
                    detail=(
                        f"loss above {self.policy.loss_increase_ratio:g}x the trailing "
                        f"median for {consecutive} epochs"
                    ),
                )
        return self._healthy(current)


class ScoreTrendProbe(_BaseProbe):
    name = "score_trend"

    def probe(self, context: HealthContext) -> ProbeResult:
        current = context.scores
        previous = context.previous_scores
        if not current or not previous:
            return self._healthy()
        shared = sorted(set(current) & set(previous))
        if not shared:
            return self._healthy()
        mean_now = sum(float(current[name]) for name in shared) / len(shared)
        mean_before = sum(float(previous[name]) for name in shared) / len(shared)
        drop = mean_before - mean_now
        if mean_now < 0 and mean_before >= 0:
            return ProbeResult(
                name=self.name,
                status=WatchdogStatus.CRITICAL,
                value=mean_now,
                threshold=0.0,
                detail="mean score fell below the non-negative floor",
            )
        if drop > self.policy.score_regression_tolerance:
            return ProbeResult(
                name=self.name,
                status=WatchdogStatus.DEGRADED,
                value=mean_now,
                threshold=mean_before - self.policy.score_regression_tolerance,
                detail=f"mean score fell by {drop:.4f}",
            )
        return self._healthy(mean_now)


class EntropyProbe(_BaseProbe):
    name = "entropy"

    def probe(self, context: HealthContext) -> ProbeResult:
        entropy = context.entropy
        if entropy is None:
            return self._healthy()
        if entropy == 0.0 and (context.max_entropy or 0.0) > 0.0:
            return ProbeResult(
                name=self.name,
                status=WatchdogStatus.CRITICAL,
                value=entropy,
                threshold=0.0,
                detail="group entropy is exactly zero with more than one peer",
            )
        if context.max_entropy:
            ratio = entropy / context.max_entropy
            if ratio < self.policy.entropy_collapse_ratio:
                return ProbeResult(
                    name=self.name,
                    status=WatchdogStatus.DEGRADED,
                    value=ratio,
                    threshold=self.policy.entropy_collapse_ratio,
                    detail=f"entropy ratio {ratio:.4f} is below the collapse threshold",
                )
        return self._healthy(entropy)


class AggregationFallbackProbe(_BaseProbe):
    name = "aggregation_fallback"

    def probe(self, context: HealthContext) -> ProbeResult:
        reports = list(context.aggregation_reports)
        if not reports:
            return self._healthy()
        flags = [bool(getattr(report, "fallback", False)) for report in reports]
        if len(flags) >= 2 and all(flags[-2:]):
            return ProbeResult(
                name=self.name,
                status=WatchdogStatus.CRITICAL,
                value=1.0,
                detail="aggregation fell back twice consecutively",
            )
        if flags[-1]:
            return ProbeResult(
                name=self.name,
                status=WatchdogStatus.DEGRADED,
                value=1.0,
                detail="aggregation fell back to the reference policy",
            )
        return self._healthy(0.0)


class KappaProbe(_BaseProbe):
    name = "kappa"

    def probe(self, context: HealthContext) -> ProbeResult:
        kappa = context.kappa
        if kappa is None:
            return self._healthy()
        value = float(kappa)
        if not math.isfinite(value):
            return ProbeResult(
                name=self.name,
                status=WatchdogStatus.CRITICAL,
                value=value,
                detail="kappa is not finite",
            )
        if value < self.policy.kappa_min or value > self.policy.kappa_max:
            return ProbeResult(
                name=self.name,
                status=WatchdogStatus.DEGRADED,
                value=value,
                threshold=self.policy.kappa_max,
                detail=(
                    f"kappa outside [{self.policy.kappa_min:g}, {self.policy.kappa_max:g}]"
                ),
            )
        return self._healthy(value)


class StaleProgressProbe(_BaseProbe):
    name = "stale_progress"

    def probe(self, context: HealthContext) -> ProbeResult:
        if context.now is None or context.monotonic is None:
            return self._healthy()
        elapsed = float(context.monotonic) - float(context.extras.get("epoch_started", 0.0))
        if elapsed <= 0:
            return self._healthy()
        if elapsed >= 2.0 * self.policy.stale_epoch_seconds:
            return ProbeResult(
                name=self.name,
                status=WatchdogStatus.CRITICAL,
                value=elapsed,
                threshold=self.policy.stale_epoch_seconds,
                detail=f"no epoch completed in {elapsed:.1f}s",
            )
        if elapsed >= self.policy.stale_epoch_seconds:
            return ProbeResult(
                name=self.name,
                status=WatchdogStatus.DEGRADED,
                value=elapsed,
                threshold=self.policy.stale_epoch_seconds,
                detail=f"epoch has been running for {elapsed:.1f}s",
            )
        return self._healthy(elapsed)


class ThroughputProbe(_BaseProbe):
    name = "throughput"

    def probe(self, context: HealthContext) -> ProbeResult:
        current = context.steps_per_second
        previous = context.previous_steps_per_second
        if current is None or previous is None or previous <= 0:
            return self._healthy()
        ratio = float(current) / float(previous)
        if ratio < 0.10:
            return ProbeResult(
                name=self.name,
                status=WatchdogStatus.CRITICAL,
                value=ratio,
                threshold=self.policy.min_throughput_ratio,
                detail=f"throughput is {ratio:.2%} of the previous epoch",
            )
        if ratio < self.policy.min_throughput_ratio:
            return ProbeResult(
                name=self.name,
                status=WatchdogStatus.DEGRADED,
                value=ratio,
                threshold=self.policy.min_throughput_ratio,
                detail=f"throughput is {ratio:.2%} of the previous epoch",
            )
        return self._healthy(ratio)


class MemoryProbe(_BaseProbe):
    name = "memory"

    def probe(self, context: HealthContext) -> ProbeResult:
        metadata: dict[str, Any] = {}
        current = context.rss_bytes
        initial = context.initial_rss_bytes
        if current is not None and initial:
            metadata["source"] = "rss"
            ratio = float(current) / float(initial)
            if ratio > self.policy.max_rss_growth_ratio:
                return ProbeResult(
                    name=self.name,
                    status=WatchdogStatus.DEGRADED,
                    value=ratio,
                    threshold=self.policy.max_rss_growth_ratio,
                    detail=f"RSS grew to {ratio:.2f}x its first-epoch value",
                    metadata=metadata,
                )
        gpu = context.gpu_memory_ratio
        if gpu is not None:
            metadata["gpu"] = True
            if gpu > self.policy.max_gpu_memory_ratio:
                return ProbeResult(
                    name=self.name,
                    status=WatchdogStatus.DEGRADED,
                    value=float(gpu),
                    threshold=self.policy.max_gpu_memory_ratio,
                    detail=f"GPU memory at {gpu:.1%} of capacity",
                    metadata=metadata,
                )
        return self._healthy(None, metadata=metadata)


class CheckpointAgeProbe(_BaseProbe):
    name = "checkpoint_age"

    def probe(self, context: HealthContext) -> ProbeResult:
        if context.epoch <= 1 and context.checkpoint_count == 0:
            return self._healthy()
        if context.checkpoint_count == 0:
            return ProbeResult(
                name=self.name,
                status=WatchdogStatus.CRITICAL,
                value=None,
                detail=f"no checkpoint exists past epoch {context.epoch - 1}",
            )
        if context.last_checkpoint_at is None or context.now is None:
            return self._healthy()
        age = float(context.now) - float(context.last_checkpoint_at)
        if age > self.policy.max_checkpoint_age_seconds:
            return ProbeResult(
                name=self.name,
                status=WatchdogStatus.DEGRADED,
                value=age,
                threshold=self.policy.max_checkpoint_age_seconds,
                detail=f"last checkpoint is {age:.0f}s old",
            )
        return self._healthy(age)


class WarningSpikeProbe(_BaseProbe):
    name = "warning_spike"

    def probe(self, context: HealthContext) -> ProbeResult:
        count = int(context.warnings_escalated)
        if count > 2 * self.policy.warning_spike_threshold:
            return ProbeResult(
                name=self.name,
                status=WatchdogStatus.CRITICAL,
                value=float(count),
                threshold=float(self.policy.warning_spike_threshold),
                detail=f"{count} warnings escalated in one epoch",
            )
        if count > self.policy.warning_spike_threshold:
            return ProbeResult(
                name=self.name,
                status=WatchdogStatus.DEGRADED,
                value=float(count),
                threshold=float(self.policy.warning_spike_threshold),
                detail=f"{count} warnings escalated in one epoch",
            )
        return self._healthy(float(count))


class DeadBranchProbe(_BaseProbe):
    name = "dead_branch"

    def probe(self, context: HealthContext) -> ProbeResult:
        for path, count in sorted(context.dead_branches.items()):
            if int(count) >= self.policy.dead_branch_patience:
                return ProbeResult(
                    name=self.name,
                    status=WatchdogStatus.DEGRADED,
                    value=float(count),
                    threshold=float(self.policy.dead_branch_patience),
                    detail=f"{path} has reported no proposals for {count} epochs",
                    node_path=str(path),
                )
        return self._healthy()


class ChildStarvationProbe(_BaseProbe):
    name = "child_starvation"

    def probe(self, context: HealthContext) -> ProbeResult:
        for path, count in sorted(context.starved_children.items()):
            if int(count) >= self.policy.child_starvation_patience:
                return ProbeResult(
                    name=self.name,
                    status=WatchdogStatus.DEGRADED,
                    value=float(count),
                    threshold=float(self.policy.child_starvation_patience),
                    detail=f"{path} has never been selected for {count} epochs",
                    node_path=str(path),
                )
        return self._healthy()


class DepthCostProbe(_BaseProbe):
    name = "depth_cost"

    def probe(self, context: HealthContext) -> ProbeResult:
        durations = dict(context.level_durations)
        for depth in sorted(durations):
            if depth <= 1:
                continue
            parent = durations.get(depth - 1)
            if not parent or parent <= 0:
                continue
            ratio = durations[depth] / parent
            if ratio > self.policy.depth_cost_ratio:
                return ProbeResult(
                    name=self.name,
                    status=WatchdogStatus.DEGRADED,
                    value=ratio,
                    threshold=self.policy.depth_cost_ratio,
                    detail=(
                        f"level {depth} took {ratio:.2f}x its parent's wall time"
                    ),
                    node_path=f"depth:{depth}",
                )
        return self._healthy()


class TreeGrowthProbe(_BaseProbe):
    """The active tree is more than ``max_tree_growth_ratio`` times the tree the run was built with.

    The probe is advisory (``DEGRADED`` at most): a user who doubles a tree on purpose raises
    ``WATCHDOG.MAX_TREE_GROWTH_RATIO``, and ``W044`` says so when the schedule is validated.
    Its job is the callback or strategy that requests a peer every epoch by mistake;
    ``TRAINER.MUTATION.MAX_PLANS_PER_EPOCH`` bounds the rate, this bounds the total.
    """

    name = "tree_growth"

    def probe(self, context: HealthContext) -> ProbeResult:
        nodes = context.tree_nodes
        initial = context.initial_tree_nodes
        if not nodes or not initial or int(initial) <= 0:
            return self._healthy()
        ratio = float(nodes) / float(initial)
        limit = float(self.policy.max_tree_growth_ratio)
        metadata = {
            "nodes": int(nodes),
            "initial_nodes": int(initial),
            "retired": int(context.retired_nodes),
            "mutations": int(context.mutations),
            "topology_epoch": int(context.topology_epoch),
        }
        if ratio > limit:
            return ProbeResult(
                name=self.name,
                status=WatchdogStatus.DEGRADED,
                value=ratio,
                threshold=limit,
                detail=(
                    f"the tree has {int(nodes)} active nodes, {ratio:.2f}x the {int(initial)} it was "
                    f"built with, after {int(context.mutations)} mutation(s)"
                ),
                metadata=metadata,
            )
        return self._healthy(ratio, threshold=limit, metadata=metadata)


def default_probes(policy: WatchdogPolicy) -> list[HealthProbe]:
    """The documented probe set, in report order."""
    return [
        NaNLossProbe(policy),
        InfLossProbe(policy),
        GradNormProbe(policy),
        LossTrendProbe(policy),
        ScoreTrendProbe(policy),
        EntropyProbe(policy),
        AggregationFallbackProbe(policy),
        KappaProbe(policy),
        StaleProgressProbe(policy),
        ThroughputProbe(policy),
        MemoryProbe(policy),
        CheckpointAgeProbe(policy),
        WarningSpikeProbe(policy),
        DeadBranchProbe(policy),
        ChildStarvationProbe(policy),
        DepthCostProbe(policy),
        TreeGrowthProbe(policy),
    ]


@dataclass(frozen=True)
class HealthReport:
    """The watchdog's verdict for one epoch."""

    status: WatchdogStatus
    action: WatchdogAction
    epoch: int
    results: tuple[ProbeResult, ...] = ()
    per_node: Mapping[str, WatchdogStatus] = field(default_factory=dict)
    recommendations: tuple[str, ...] = ()
    snapshot: Any = None
    duration: float = 0.0
    monotonic: float = 0.0

    @property
    def problems(self) -> tuple[ProbeResult, ...]:
        return tuple(result for result in self.results if result.is_problem)

    @property
    def critical(self) -> bool:
        return self.status is WatchdogStatus.CRITICAL

    def worst(self, limit: int = 5) -> tuple[ProbeResult, ...]:
        ranked = sorted(self.problems, key=lambda item: (-item.status.rank, item.name))
        return tuple(ranked[:limit])

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "action": self.action.value,
            "epoch": self.epoch,
            "results": [result.as_dict() for result in self.results],
            "per_node": {path: status.value for path, status in self.per_node.items()},
            "recommendations": list(self.recommendations),
            "duration": self.duration,
            "monotonic": self.monotonic,
        }

    def describe(self) -> str:
        problems = self.worst()
        if not problems:
            return f"epoch {self.epoch}: {self.status.value} ({self.action.value})"
        return (
            f"epoch {self.epoch}: {self.status.value} ({self.action.value}) "
            + "; ".join(result.describe() for result in problems)
        )


class MACTFWatchdog:
    """Runs probes, takes snapshots, and applies the configured degradation action."""

    def __init__(
        self,
        *,
        checkpoint: Any,
        logger: Any,
        policy: WatchdogPolicy | None = None,
        probes: Sequence[HealthProbe] | None = None,
        snapshot_every_epochs: int = 1,
        snapshot_every_seconds: float | None = None,
        snapshot_on_degradation: bool = True,
        on_degradation: str = "snapshot",
        on_critical: str = "rollback",
        keep_history: int = 512,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if checkpoint is None:
            raise ImproperlyConfigured(
                "MACTFWatchdog requires a checkpoint writer; a watchdog that cannot "
                "snapshot is a monitor, and a monitor cannot recover a run"
            )
        if logger is None:
            raise ImproperlyConfigured("MACTFWatchdog requires a logger")
        self.checkpoint = checkpoint
        self.logger = logger
        self.policy = policy or WatchdogPolicy()
        if int(snapshot_every_epochs) < 0:
            raise ImproperlyConfigured("snapshot_every_epochs must be >= 0")
        if snapshot_every_seconds is not None and snapshot_every_seconds <= 0:
            raise ImproperlyConfigured("snapshot_every_seconds must be > 0 when set")
        if on_degradation not in ACTION_NAMES:
            raise ImproperlyConfigured(
                f"unknown WATCHDOG.ON_DEGRADATION {on_degradation!r}; accepted: "
                + ", ".join(ACTION_NAMES)
            )
        if on_critical not in ACTION_NAMES:
            raise ImproperlyConfigured(
                f"unknown WATCHDOG.ON_CRITICAL {on_critical!r}; accepted: "
                + ", ".join(ACTION_NAMES)
            )
        self.probes: list[HealthProbe] = list(probes or default_probes(self.policy))
        self.snapshot_every_epochs = int(snapshot_every_epochs)
        self.snapshot_every_seconds = snapshot_every_seconds
        self.snapshot_on_degradation = bool(snapshot_on_degradation)
        self.on_degradation = WatchdogAction(on_degradation)
        self.on_critical = WatchdogAction(on_critical)
        self.keep_history = max(1, int(keep_history))
        self.clock = clock
        self.history: list[HealthReport] = []
        self._last_snapshot_epoch = 0
        self._last_snapshot_time: float | None = None
        self._stop = False
        self._patience: dict[str, int] = {}

    # -- construction ----------------------------------------------------
    #: ``MACTFWatchdog.__init__`` arguments ``WATCHDOG`` may set beyond the policy.
    _OWN_KEYS: tuple[str, ...] = (
        "snapshot_every_epochs",
        "snapshot_every_seconds",
        "snapshot_on_degradation",
        "on_degradation",
        "on_critical",
        "keep_history",
    )

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any] | Any,
        *,
        checkpoint: Any,
        logger: Any,
        probes: Sequence[HealthProbe] | None = None,
    ) -> "MACTFWatchdog":
        """Build a watchdog from ``WATCHDOG``.

        ``checkpoint`` and ``logger`` stay required: a watchdog without a writer
        cannot snapshot, and ``__init__`` already refuses one. Unlike
        :meth:`WatchdogPolicy.from_settings`, a key that neither the policy nor the
        constructor declares is an error here: ``MAX_GRAD_NORMN: 100`` would leave
        the real threshold at its default and the probe would never fire.
        """
        block = {str(key).lower(): value for key, value in _block(settings, "WATCHDOG").items()}
        fields = set(WatchdogPolicy.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = sorted(set(block) - fields - set(cls._OWN_KEYS))
        if unknown:
            raise ImproperlyConfigured(
                f"WATCHDOG names unknown key(s) {unknown}; a typo in a threshold is a "
                "watchdog that never fires"
            )
        policy = WatchdogPolicy(**{key: value for key, value in block.items() if key in fields})
        own: dict[str, Any] = {key: block[key] for key in cls._OWN_KEYS if key in block}
        for action in ("on_degradation", "on_critical"):
            if action in own:
                own[action] = str(own[action]).lower()
        return cls(checkpoint=checkpoint, logger=logger, policy=policy, probes=probes, **own)

    # -- evaluation ------------------------------------------------------
    def evaluate(self, context: HealthContext) -> HealthReport:
        """Run every probe and decide the action, with no side effects."""
        started = float(self.clock())
        results: list[ProbeResult] = []
        for probe in self.probes:
            try:
                result = probe.probe(context)
                if not isinstance(result, ProbeResult):
                    raise TypeError(
                        f"returned {type(result).__name__}, not ProbeResult"
                    )
                results.append(result)
            except Exception as error:  # noqa: BLE001 - a broken probe must not stop training
                results.append(
                    ProbeResult(
                        name=getattr(probe, "name", type(probe).__name__),
                        status=WatchdogStatus.CRITICAL,
                        detail=f"probe raised {type(error).__name__}: {error}",
                        metadata={"probe_error": True},
                    )
                )
        status = WatchdogStatus.worst(result.status for result in results)
        per_node: dict[str, WatchdogStatus] = {}
        for result in results:
            if not result.node_path:
                continue
            per_node[result.node_path] = WatchdogStatus.worst(
                [per_node.get(result.node_path, WatchdogStatus.HEALTHY), result.status]
            )
        recommendations = self._recommendations(results, context)

        action = WatchdogAction.NONE
        if status is WatchdogStatus.CRITICAL:
            action = self.on_critical
        elif status is WatchdogStatus.DEGRADED:
            action = self.on_degradation if self.snapshot_on_degradation else WatchdogAction.LOG
        elif self._periodic_due(context):
            action = WatchdogAction.SNAPSHOT

        return HealthReport(
            status=status,
            action=action,
            epoch=int(context.epoch),
            results=tuple(results),
            per_node=per_node,
            recommendations=recommendations,
            duration=float(self.clock()) - started,
            monotonic=float(context.monotonic if context.monotonic is not None else self.clock()),
        )

    def _periodic_due(self, context: HealthContext) -> bool:
        if self.snapshot_every_epochs and (
            int(context.epoch) - self._last_snapshot_epoch >= self.snapshot_every_epochs
        ):
            return True
        if self.snapshot_every_seconds and self._last_snapshot_time is not None:
            now = float(context.monotonic if context.monotonic is not None else self.clock())
            if now - self._last_snapshot_time >= self.snapshot_every_seconds:
                return True
        return False

    @staticmethod
    def _recommendations(
        results: Sequence[ProbeResult], context: HealthContext
    ) -> tuple[str, ...]:
        """Advice, never an action. The watchdog reports; a caller decides."""
        advice: list[str] = []
        for result in results:
            if not result.is_problem:
                continue
            if result.name == "loss_trend":
                advice.append(
                    "loss is rising: inspect the data batch and the learning rate, "
                    "then consider lowering TRAINING.LEARNING_RATE manually"
                )
            elif result.name == "grad_norm":
                advice.append(
                    "gradient norms are large: consider a lower learning rate or a "
                    "tighter TRAINING.MAX_GRAD_NORM"
                )
            elif result.name == "entropy":
                advice.append(
                    "group entropy is collapsing: raise COOPETITION.KAPPA_MIN or widen "
                    "the coefficient of the losing peers"
                )
            elif result.name == "aggregation_fallback":
                advice.append(
                    "aggregation fell back to the reference: check WEIGHT_FLOOR and "
                    "DISAGREEMENT_MAX against the number of peers"
                )
            elif result.name == "kappa":
                advice.append("kappa left its configured range; inspect the controller")
            elif result.name == "stale_progress":
                advice.append(
                    "an epoch is running unusually long: inspect the slowest node's "
                    "module and the sandbox provider"
                )
            elif result.name == "dead_branch" and result.metadata.get("pruned"):
                advice.append(
                    f"branch {result.node_path!r} was removed from play ({result.metadata['pruned']}); "
                    "its weights are in the checkpoint written before the prune"
                )
            elif result.name == "dead_branch":
                advice.append(
                    f"branch {result.node_path!r} produced nothing: check that node's "
                    "module and its delegation policy"
                )
            elif result.name == "tree_growth":
                advice.append(
                    "the tree keeps growing: inspect the callbacks and strategies that call "
                    "request_mutation, or raise WATCHDOG.MAX_TREE_GROWTH_RATIO if the growth is intended"
                )
            elif result.name == "child_starvation":
                advice.append(
                    f"child {result.node_path!r} is never selected: review the "
                    "delegation policy's selection rule"
                )
            elif result.name == "checkpoint_age":
                advice.append(
                    "checkpoints are stale: lower WATCHDOG.SNAPSHOT_EVERY_EPOCHS"
                )
            elif result.name == "memory":
                advice.append(
                    "memory is growing: check for retained tensors between epochs"
                )
            elif result.name == "warning_spike":
                advice.append(
                    f"{context.warnings_escalated} warnings escalated this epoch: "
                    "review the scorecard's warnings_escalated list"
                )
        # Deduplicate while keeping the first occurrence's order.
        return tuple(dict.fromkeys(advice))

    # -- actions ---------------------------------------------------------
    def tick(self, context: HealthContext) -> HealthReport:
        """Evaluate, then act. A snapshot always precedes damage."""
        report = self.evaluate(context)
        snapshot = None
        if report.status is WatchdogStatus.CRITICAL:
            # Step 5 of document 09 section 3.4: snapshot *first*, so the rolled-back
            # state is also on disk. A rollback that destroys its own evidence is
            # useless.
            if report.action in (
                WatchdogAction.ROLLBACK,
                WatchdogAction.PRUNE_BRANCH,
                WatchdogAction.ABORT,
                WatchdogAction.PAUSE,
            ):
                snapshot = self._snapshot(context, reason=self._reason(report))
        elif report.action is WatchdogAction.SNAPSHOT:
            snapshot = self._snapshot(context, reason="periodic")

        report = HealthReport(
            status=report.status,
            action=report.action,
            epoch=report.epoch,
            results=report.results,
            per_node=report.per_node,
            recommendations=report.recommendations,
            snapshot=snapshot,
            duration=report.duration,
            monotonic=report.monotonic,
        )
        self.history.append(report)
        if len(self.history) > self.keep_history:
            self.history.pop(0)
        if report.status is not WatchdogStatus.HEALTHY:
            emit = getattr(self.logger, "health", None)
            if callable(emit):
                emit(report)
        if report.action is WatchdogAction.ROLLBACK:
            self.rollback(report)
        elif report.action is WatchdogAction.PRUNE_BRANCH:
            self.prune_branch(context)
        if report.action.stops_training:
            self._stop = True
        return report

    def _reason(self, report: HealthReport) -> str:
        names = [result.name for result in report.worst(limit=3)]
        return "critical: " + ", ".join(names)

    def _snapshot(self, context: HealthContext, *, reason: str, best: bool = False) -> Any:
        saver = getattr(self.checkpoint, "save", None)
        if not callable(saver):
            raise ImproperlyConfigured(
                "the configured checkpoint writer has no save() method"
            )
        record = saver(context, reason=reason, episode=int(context.epoch), best=best)
        self._last_snapshot_epoch = int(context.epoch)
        self._last_snapshot_time = float(
            context.monotonic if context.monotonic is not None else self.clock()
        )
        return record

    def snapshot(self, context: HealthContext, *, reason: str, best: bool = False) -> Any:
        """Take a snapshot on demand, outside a tick."""
        return self._snapshot(context, reason=reason, best=best)

    def rollback(self, report: HealthReport | None = None) -> Any:
        """Restore the previous checkpoint and emit the existing rollback signal."""
        from .. import signals

        restorer = getattr(self.checkpoint, "resume_from_last_checkpoint", None)
        result = None
        if callable(restorer):
            try:
                result = restorer()
            except Exception as error:  # noqa: BLE001 - a failed rollback is reported
                emit = getattr(self.logger, "error", None)
                if callable(emit):
                    emit(f"watchdog rollback failed: {error}")
                return None
        reason = self._reason(report) if report is not None else "watchdog rollback"
        signals.checkpoint_rolled_back.send(sender=self, reason=reason)
        return result

    def prune_branch(self, context: HealthContext) -> int:
        """Drop every dead subtree the report named; returns how many nodes left play.

        A trainer sets ``context.pruner``, which turns the removal into a mutation plan (12.8.5):
        the tree, the peers, the optimizers, the controllers and every per-path cache change
        together or not at all. A context a caller built by hand carries only a tree, and the
        original ``remove_node`` is the best that can be done for a structure nothing else
        depends on.
        """
        removed = 0
        pruner = context.pruner
        tree = context.extras.get("tree")
        if pruner is None and tree is None:
            return 0
        for path, count in sorted(context.dead_branches.items()):
            if int(count) < self.policy.dead_branch_patience:
                continue
            try:
                if pruner is not None:
                    removed += int(pruner(str(path), int(count)))
                else:
                    removed += int(tree.remove_node(str(path), recursive=True))
            except Exception as error:  # noqa: BLE001 - a failed prune is reported, never fatal
                if getattr(error, "code", None) == "E089" and getattr(error, "rolled_back", True) is False:
                    raise           # the engine could not undo a failed commit: the run must stop (R9)
                emit = getattr(self.logger, "error", None)
                if callable(emit):
                    emit(f"watchdog could not prune branch {path!r}: {error}")
        return removed

    def should_stop(self) -> bool:
        return self._stop

    def health_history(self, limit: int = 50) -> list[HealthReport]:
        return self.history[-int(limit):] if limit else list(self.history)

    # -- persistence -----------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        last = self.history[-1].as_dict() if self.history else None
        return {
            "policy": self.policy.as_dict(),
            "last_snapshot_epoch": self._last_snapshot_epoch,
            "last_snapshot_time": self._last_snapshot_time,
            "last_report": last,
            "patience": dict(self._patience),
            "stop": self._stop,
            "probes": [getattr(probe, "name", type(probe).__name__) for probe in self.probes],
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        state = dict(payload or {})
        self._last_snapshot_epoch = int(state.get("last_snapshot_epoch", 0))
        self._last_snapshot_time = state.get("last_snapshot_time")
        self._patience = {str(k): int(v) for k, v in (state.get("patience") or {}).items()}
        self._stop = bool(state.get("stop", False))
        raw_policy = state.get("policy")
        if isinstance(raw_policy, Mapping):
            accepted = set(WatchdogPolicy.__dataclass_fields__)  # type: ignore[attr-defined]
            self.policy = WatchdogPolicy(
                **{key: value for key, value in raw_policy.items() if key in accepted}
            )

    def describe(self) -> dict[str, Any]:
        return {
            "policy": self.policy.as_dict(),
            "probes": [getattr(probe, "name", type(probe).__name__) for probe in self.probes],
            "snapshot_every_epochs": self.snapshot_every_epochs,
            "snapshot_every_seconds": self.snapshot_every_seconds,
            "on_degradation": self.on_degradation.value,
            "on_critical": self.on_critical.value,
            "history": len(self.history),
            "status": self.history[-1].status.value if self.history else "unknown",
        }
