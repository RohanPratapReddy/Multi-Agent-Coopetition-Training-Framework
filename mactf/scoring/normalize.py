"""Cohort normalization: points -> calibrated score -> aggregation signal.

Document 11 section 7. Three stages, each with one job:

===========  ==============  =============  ============================
Stage        Value           Range          Purpose
===========  ==============  =============  ============================
2. Points    ``final_points`` ``[0, 100]``   absolute rubric score
3. Calibrate ``calibrated``   ``[0, 100]``   cohort-relative display score
4. Signal    ``signal``       ``[-1, 1]``    aggregation input
===========  ==============  =============  ============================

Only stage 4 reaches the optimizer. The transform is the single affine
``calibrated = target_mean + shrink * (points - mean)``, which preserves order for
any positive ``shrink`` and moves the tail toward a common band without erasing
the ranking.

This module imports ``math`` and ``statistics`` and nothing else. It runs inside
request handlers as well as training loops, so it must not pull torch in.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..exceptions import ImproperlyConfigured
from ..wire import need, opt, strings

__all__ = [
    "CohortNormalizer",
    "CohortStats",
    "MAX_POINTS",
    "MIN_POINTS",
    "NormalizationPolicy",
    "NormalizationReport",
    "NormalizedScore",
    "calibrate",
    "to_signal",
]


MAX_POINTS: float = 100.0
"""The ceiling of the scale. Reward weights sum to exactly this number."""

MIN_POINTS: float = 0.0
"""The floor of the scale. A void episode is pinned here."""


@dataclass(frozen=True)
class NormalizationPolicy:
    """How a cohort's points are compressed toward a common band."""

    enabled: bool = True
    target_mean: float = 62.0
    shrink: float = 0.35
    exclude_void: bool = True
    min_cohort: int = 2
    signal_scale: float = 50.0
    clip: tuple[float, float] = (MIN_POINTS, MAX_POINTS)

    def __post_init__(self) -> None:
        if not 0.0 <= self.shrink <= 1.0:
            raise ImproperlyConfigured("NORMALIZATION.SHRINK must be within [0, 1]")
        if not 0.0 < self.target_mean < 100.0:
            raise ImproperlyConfigured(
                "NORMALIZATION.TARGET_MEAN must be within (0, 100)"
            )
        if self.min_cohort < 1:
            raise ImproperlyConfigured("NORMALIZATION.MIN_COHORT must be >= 1")
        if self.signal_scale <= 0:
            raise ImproperlyConfigured("NORMALIZATION.SIGNAL_SCALE must be > 0")
        low, high = self.clip
        if not (0.0 <= low < high <= MAX_POINTS):
            raise ImproperlyConfigured(
                "NORMALIZATION.CLIP must be an increasing pair within [0, 100]"
            )

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any] | None) -> "NormalizationPolicy":
        """Read ``NORMALIZATION`` from a settings block or the nested block itself.

        Both shapes are accepted because the trainer is handed the ``COOPETITION``
        block while the API is handed the ``NORMALIZATION`` sub-block; requiring
        callers to unwrap would be a second convention to remember.
        """
        payload = dict(settings or {})
        if "NORMALIZATION" in payload and isinstance(payload["NORMALIZATION"], Mapping):
            payload = dict(payload["NORMALIZATION"])
        return cls(
            enabled=bool(payload.get("ENABLED", True)),
            target_mean=float(payload.get("TARGET_MEAN", 62.0)),
            shrink=float(payload.get("SHRINK", 0.35)),
            exclude_void=bool(payload.get("EXCLUDE_VOID", True)),
            min_cohort=int(payload.get("MIN_COHORT", 2)),
            signal_scale=float(payload.get("SIGNAL_SCALE", 50.0)),
            clip=(
                float((payload.get("CLIP") or (MIN_POINTS, MAX_POINTS))[0]),
                float((payload.get("CLIP") or (MIN_POINTS, MAX_POINTS))[1]),
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "target_mean": self.target_mean,
            "shrink": self.shrink,
            "exclude_void": self.exclude_void,
            "min_cohort": self.min_cohort,
            "signal_scale": self.signal_scale,
            "clip": [self.clip[0], self.clip[1]],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NormalizationPolicy":
        """Inverse of :meth:`as_dict` (lower-case keys; ``from_settings`` reads UPPER)."""
        clip = need("NormalizationPolicy", payload, "clip")
        return cls(
            enabled=bool(need("NormalizationPolicy", payload, "enabled")),
            target_mean=float(need("NormalizationPolicy", payload, "target_mean")),
            shrink=float(need("NormalizationPolicy", payload, "shrink")),
            exclude_void=bool(need("NormalizationPolicy", payload, "exclude_void")),
            min_cohort=int(need("NormalizationPolicy", payload, "min_cohort")),
            signal_scale=float(need("NormalizationPolicy", payload, "signal_scale")),
            clip=(float(clip[0]), float(clip[1])),
        )


def calibrate(points: float, *, mean: float, policy: NormalizationPolicy) -> float:
    """Compress one peer's points toward the cohort band.

    ``calibrated = target_mean + shrink * (points - mean)``, clamped to ``clip``.
    The transform is affine in ``points`` with a positive slope for any
    ``shrink > 0``, which is what preserves order (invariant N2).
    """
    low, high = policy.clip
    value = policy.target_mean + policy.shrink * (points - mean)
    return min(high, max(low, value))


def to_signal(calibrated: float, *, policy: NormalizationPolicy) -> float:
    """Map calibrated points onto the aggregator's bounded signal."""
    return (calibrated - policy.target_mean) / policy.signal_scale


@dataclass(frozen=True)
class CohortStats:
    """Distribution of one episode's points across the cohort."""

    n: int
    mean: float
    std: float
    minimum: float
    maximum: float
    p25: float
    p50: float
    p75: float
    iqr: float
    spread: float
    excluded_void: tuple[str, ...] = ()
    order: tuple[str, ...] = ()

    @classmethod
    def from_points(
        cls,
        points: Mapping[str, float],
        *,
        exclude: Sequence[str] = (),
    ) -> "CohortStats":
        """Statistics over the included peers, in the caller's input order.

        ``exclude`` removes peers *before* the mean is computed. That is the point
        of void exclusion: a fabricated answer must not lower the cohort mean and
        thereby raise everyone else's calibrated score.
        """
        excluded = tuple(str(name) for name in exclude)
        order = tuple(str(name) for name in points)
        kept = [float(points[name]) for name in order if name not in set(excluded)]
        if not kept:
            return cls(
                n=0,
                mean=0.0,
                std=0.0,
                minimum=0.0,
                maximum=0.0,
                p25=0.0,
                p50=0.0,
                p75=0.0,
                iqr=0.0,
                spread=0.0,
                excluded_void=excluded,
                order=order,
            )
        mean = sum(kept) / len(kept)
        std = statistics.pstdev(kept) if len(kept) > 1 else 0.0
        quartiles = statistics.quantiles(kept, n=4) if len(kept) > 1 else [kept[0]] * 3
        p25, p50, p75 = (float(quartiles[0]), float(quartiles[1]), float(quartiles[2]))
        return cls(
            n=len(kept),
            mean=mean,
            std=std,
            minimum=min(kept),
            maximum=max(kept),
            p25=p25,
            p50=p50,
            p75=p75,
            iqr=p75 - p25,
            spread=max(kept) - min(kept),
            excluded_void=excluded,
            order=order,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "p25": self.p25,
            "p50": self.p50,
            "p75": self.p75,
            "iqr": self.iqr,
            "spread": self.spread,
            "excluded_void": list(self.excluded_void),
            "order": list(self.order),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CohortStats":
        owner = "CohortStats"
        return cls(
            n=int(need(owner, payload, "n")),
            mean=float(need(owner, payload, "mean")),
            std=float(need(owner, payload, "std")),
            minimum=float(need(owner, payload, "minimum")),
            maximum=float(need(owner, payload, "maximum")),
            p25=float(need(owner, payload, "p25")),
            p50=float(need(owner, payload, "p50")),
            p75=float(need(owner, payload, "p75")),
            iqr=float(need(owner, payload, "iqr")),
            spread=float(need(owner, payload, "spread")),
            excluded_void=strings(payload.get("excluded_void")),
            order=strings(payload.get("order")),
        )


@dataclass(frozen=True)
class NormalizedScore:
    """One peer's calibrated result."""

    peer: str
    points: float
    calibrated: float
    percentile: float
    rank: int
    signal: float
    void: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "peer": self.peer,
            "points": self.points,
            "calibrated": self.calibrated,
            "percentile": self.percentile,
            "rank": self.rank,
            "signal": self.signal,
            "void": self.void,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NormalizedScore":
        owner = "NormalizedScore"
        return cls(
            peer=str(need(owner, payload, "peer")),
            points=float(need(owner, payload, "points")),
            calibrated=float(need(owner, payload, "calibrated")),
            percentile=float(need(owner, payload, "percentile")),
            rank=int(need(owner, payload, "rank")),
            signal=float(need(owner, payload, "signal")),
            void=bool(opt(payload, "void", False)),
        )


@dataclass(frozen=True)
class NormalizationReport:
    """The full derivation of one episode's calibration, for audit."""

    episode: int
    policy: NormalizationPolicy
    stats: CohortStats
    scores: tuple[NormalizedScore, ...]
    preserves_order: bool
    before_spread: float
    after_spread: float
    mean_shift: float
    adjustments: tuple[str, ...] = ()

    def by_peer(self) -> dict[str, NormalizedScore]:
        return {score.peer: score for score in self.scores}

    def signals(self) -> dict[str, float]:
        return {score.peer: score.signal for score in self.scores}

    def calibrated(self) -> dict[str, float]:
        return {score.peer: score.calibrated for score in self.scores}

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "policy": self.policy.as_dict(),
            "stats": self.stats.as_dict(),
            "scores": [score.as_dict() for score in self.scores],
            "preserves_order": self.preserves_order,
            "before_spread": self.before_spread,
            "after_spread": self.after_spread,
            "mean_shift": self.mean_shift,
            "adjustments": list(self.adjustments),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NormalizationReport":
        """Inverse of :meth:`as_dict`, rebuilding the nested policy, stats and scores."""
        owner = "NormalizationReport"
        return cls(
            episode=int(need(owner, payload, "episode")),
            policy=NormalizationPolicy.from_dict(need(owner, payload, "policy")),
            stats=CohortStats.from_dict(need(owner, payload, "stats")),
            scores=tuple(NormalizedScore.from_dict(item) for item in need(owner, payload, "scores")),
            preserves_order=bool(need(owner, payload, "preserves_order")),
            before_spread=float(need(owner, payload, "before_spread")),
            after_spread=float(need(owner, payload, "after_spread")),
            mean_shift=float(need(owner, payload, "mean_shift")),
            adjustments=strings(payload.get("adjustments")),
        )


class CohortNormalizer:
    """Stateless transformer. Holds only a policy.

    Stateless on purpose: kappa already carries the epoch-to-epoch memory, and a
    second controller with memory would fight it. Being stateless also means there
    is nothing to put in a checkpoint payload.
    """

    def __init__(self, policy: NormalizationPolicy | None = None) -> None:
        self.policy = policy or NormalizationPolicy()

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _percentiles(values: Mapping[str, float]) -> dict[str, tuple[float, int]]:
        """Midrank percentile and 1-based rank with ties sharing the better rank."""
        ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
        total = len(ordered)
        ranks: dict[str, int] = {}
        previous_value: float | None = None
        previous_rank = 0
        for index, (peer, value) in enumerate(ordered, start=1):
            if previous_value is not None and value == previous_value:
                ranks[peer] = previous_rank
            else:
                ranks[peer] = index
                previous_rank = index
                previous_value = value
        result: dict[str, tuple[float, int]] = {}
        names = list(values)
        for peer in names:
            value = values[peer]
            less = sum(1 for other in names if values[other] < value)
            equal = sum(1 for other in names if values[other] == value)
            percentile = (less + 0.5 * equal) / total if total else 0.0
            result[peer] = (percentile, ranks[peer])
        return result

    def _degenerate(
        self,
        *,
        episode: int,
        points: Mapping[str, float],
        void: Mapping[str, bool],
        excluded: tuple[str, ...],
        reason: str,
    ) -> NormalizationReport:
        """``n < min_cohort`` or ``spread == 0``: calibrated equals points.

        The one refinement document 11 section 7.5 spells out: a cohort with no
        spread and a zero mean returns ``0.0`` for everyone rather than
        ``target_mean``, because "everyone scored nothing" must not be a free pass.
        """
        policy = self.policy
        scores: list[NormalizedScore] = []
        percentiles = self._percentiles(dict(points))
        for peer, raw in points.items():
            is_void = bool(void.get(peer, False))
            if is_void:
                calibrated = MIN_POINTS
            elif policy.enabled and reason == "spread == 0" and raw > 0:
                calibrated = policy.target_mean
            else:
                calibrated = float(raw)
            percentile, rank = percentiles.get(peer, (0.0, 1))
            signal = (
                float(policy.clip[0] - policy.target_mean) / policy.signal_scale
                if is_void
                else to_signal(calibrated, policy=policy)
            )
            scores.append(
                NormalizedScore(
                    peer=peer,
                    points=float(raw),
                    calibrated=float(calibrated),
                    percentile=percentile,
                    rank=rank,
                    signal=signal,
                    void=is_void,
                )
            )
        ordered = sorted(scores, key=lambda item: item.peer)
        return NormalizationReport(
            episode=episode,
            policy=policy,
            stats=CohortStats.from_points(points, exclude=excluded),
            scores=tuple(ordered),
            preserves_order=True,
            before_spread=max(points.values()) - min(points.values()) if points else 0.0,
            after_spread=(
                max(score.calibrated for score in scores)
                - min(score.calibrated for score in scores)
                if scores
                else 0.0
            ),
            mean_shift=0.0,
            adjustments=(reason,),
        )

    # -- the transform ---------------------------------------------------
    def normalize(
        self,
        points: Mapping[str, float],
        *,
        episode: int = 0,
        void: Mapping[str, bool] | None = None,
    ) -> NormalizationReport:
        """Calibrate one cohort's points and derive its aggregation signal."""
        policy = self.policy
        raw_points = {str(peer): float(value) for peer, value in points.items()}
        voids = {str(peer): bool(value) for peer, value in (void or {}).items()}
        excluded = (
            tuple(sorted(peer for peer in raw_points if voids.get(peer, False)))
            if policy.exclude_void
            else ()
        )
        stats = CohortStats.from_points(raw_points, exclude=excluded)

        if not raw_points:
            return NormalizationReport(
                episode=episode,
                policy=policy,
                stats=stats,
                scores=(),
                preserves_order=True,
                before_spread=0.0,
                after_spread=0.0,
                mean_shift=0.0,
                adjustments=("empty cohort",),
            )
        if not policy.enabled:
            return self._degenerate(
                episode=episode,
                points=raw_points,
                void=voids,
                excluded=excluded,
                reason="normalization disabled",
            )
        if stats.n < max(1, policy.min_cohort):
            return self._degenerate(
                episode=episode,
                points=raw_points,
                void=voids,
                excluded=excluded,
                reason=f"cohort size {stats.n} is below MIN_COHORT={policy.min_cohort}",
            )
        if stats.spread == 0.0:
            return self._degenerate(
                episode=episode,
                points=raw_points,
                void=voids,
                excluded=excluded,
                reason="spread == 0",
            )

        percentiles = self._percentiles(raw_points)
        scores: list[NormalizedScore] = []
        for peer, raw in raw_points.items():
            is_void = bool(voids.get(peer, False))
            if is_void:
                calibrated = MIN_POINTS
                signal = (
                    float(policy.clip[0] - policy.target_mean) / policy.signal_scale
                )
            else:
                calibrated = calibrate(raw, mean=stats.mean, policy=policy)
                signal = to_signal(calibrated, policy=policy)
            percentile, rank = percentiles.get(peer, (0.0, 1))
            scores.append(
                NormalizedScore(
                    peer=peer,
                    points=raw,
                    calibrated=calibrated,
                    percentile=percentile,
                    rank=rank,
                    signal=signal,
                    void=is_void,
                )
            )
        ordered = sorted(scores, key=lambda item: item.peer)

        # N2, verified rather than asserted: order in the input implies order in
        # the calibrated values.
        pairs = [
            (a, b)
            for a in scores
            for b in scores
            if not a.void and not b.void and a.points < b.points
        ]
        preserves_order = all(a.calibrated <= b.calibrated for a, b in pairs)

        after_values = [score.calibrated for score in ordered if not score.void]
        calibrated_mean = (
            sum(after_values) / len(after_values) if after_values else 0.0
        )
        before_spread = stats.spread
        after_spread = (
            max(after_values) - min(after_values) if after_values else 0.0
        )
        adjustments = [
            f"cohort mean {stats.mean:.4f} calibrated to {policy.target_mean:g}",
            f"spread {before_spread:.4f} -> {after_spread:.4f} (shrink={policy.shrink:g})",
        ]
        if excluded:
            adjustments.append("void peers excluded from cohort statistics: "
                               + ", ".join(excluded))
        return NormalizationReport(
            episode=episode,
            policy=policy,
            stats=stats,
            scores=tuple(ordered),
            preserves_order=bool(preserves_order),
            before_spread=before_spread,
            after_spread=after_spread,
            mean_shift=calibrated_mean - stats.mean,
            adjustments=tuple(adjustments),
        )

    def describe(self) -> dict[str, Any]:
        return {"class": type(self).__name__, "policy": self.policy.as_dict()}
