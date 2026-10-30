"""test normalize invariants (split from test_normalize.py)."""

from __future__ import annotations

import random


import pytest


from mactf.scoring import CohortNormalizer, NormalizationPolicy


def _normalize(points, **policy):
    return CohortNormalizer(NormalizationPolicy(**policy)).normalize(points, episode=3)


def test_n2_normalization_preserves_order():
    rng = random.Random(7)
    for _ in range(50):
        points = {f"p{i}": rng.uniform(0, 100) for i in range(rng.randint(2, 8))}
        report = _normalize(points)
        assert report.preserves_order
        by_raw = sorted(report.scores, key=lambda s: s.points)
        calibrated = [s.calibrated for s in by_raw]
        assert calibrated == sorted(calibrated)


def test_normalization_compresses_spread_by_the_shrink_factor():
    report = _normalize({"a": 100.0, "b": 0.0})
    assert report.after_spread == pytest.approx(0.35 * report.before_spread)


def test_signal_is_bounded_by_the_clip_and_scale():
    report = _normalize({"a": 100.0, "b": 0.0, "c": 50.0})
    for score in report.scores:
        assert -1.0 <= score.signal <= 1.0


def test_void_peers_are_excluded_from_the_mean_and_pinned_to_the_floor():
    policy = NormalizationPolicy()
    report = CohortNormalizer(policy).normalize(
        {"a": 80.0, "b": 60.0, "v": 0.0}, episode=1, void={"v": True}
    )
    assert report.stats.excluded_void == ("v",)
    assert report.stats.mean == pytest.approx(70.0)
    void = next(s for s in report.scores if s.peer == "v")
    assert void.void and void.calibrated == 0.0
    assert void.signal == pytest.approx((0.0 - policy.target_mean) / policy.signal_scale)


def test_normalization_is_deterministic_and_input_order_independent():
    one = _normalize({"a": 90.0, "b": 50.0, "c": 20.0})
    two = _normalize({"c": 20.0, "a": 90.0, "b": 50.0})
    assert one.signals() == two.signals()
