"""Document 11 sections 12.4-12.5: cohort normalization and its invariants."""

from __future__ import annotations

import pytest


from mactf.exceptions import ImproperlyConfigured


from mactf.scoring import CohortNormalizer, NormalizationPolicy


def _normalize(points, **policy):
    return CohortNormalizer(NormalizationPolicy(**policy)).normalize(points, episode=3)


def test_calibration_matches_the_documented_formula():
    report = _normalize({"a": 90.0, "b": 50.0, "c": 70.0})
    mean = 70.0
    for score in report.scores:
        expected = 62.0 + 0.35 * (score.points - mean)
        assert score.calibrated == pytest.approx(expected)
        assert score.signal == pytest.approx((expected - 62.0) / 50.0)


def test_shrink_zero_collapses_to_the_target_mean_and_one_is_identity_shifted():
    flat = _normalize({"a": 90.0, "b": 10.0}, shrink=0.0)
    assert {round(s.calibrated, 9) for s in flat.scores} == {62.0}
    full = _normalize({"a": 90.0, "b": 10.0}, shrink=1.0, target_mean=50.0)
    assert [s.calibrated for s in full.scores] == pytest.approx([90.0, 10.0])


def test_a_cohort_smaller_than_min_cohort_is_passed_through():
    report = _normalize({"solo": 77.0})
    assert report.scores[0].calibrated == pytest.approx(77.0)
    assert report.adjustments


def test_a_zero_spread_cohort_scoring_nothing_stays_at_zero():
    report = _normalize({"a": 0.0, "b": 0.0})
    assert all(score.calibrated == 0.0 for score in report.scores)


def test_a_zero_spread_cohort_with_points_is_centred_on_the_target():
    report = _normalize({"a": 40.0, "b": 40.0})
    assert all(score.calibrated == pytest.approx(62.0) for score in report.scores)


def test_ties_share_a_rank_and_a_midrank_percentile():
    report = _normalize({"a": 10.0, "b": 10.0, "c": 90.0})
    ranks = {s.peer: s.rank for s in report.scores}
    assert ranks["c"] == 1 and ranks["a"] == ranks["b"] == 2
    percentiles = {s.peer: s.percentile for s in report.scores}
    assert percentiles["a"] == pytest.approx(percentiles["b"])


def test_disabled_normalization_is_a_pass_through_with_a_stated_reason():
    report = _normalize({"a": 90.0, "b": 10.0}, enabled=False)
    assert [s.calibrated for s in report.scores] == pytest.approx([90.0, 10.0])
    assert "normalization disabled" in report.adjustments


def test_empty_cohort_reports_itself():
    report = _normalize({})
    assert report.scores == () and report.adjustments == ("empty cohort",)


def test_signals_helper_maps_peers_to_signals():
    report = _normalize({"a": 90.0, "b": 50.0})
    assert set(report.signals()) == {"a", "b"}
    assert report.signals()["a"] > report.signals()["b"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"shrink": 1.5}, "SHRINK"),
        ({"target_mean": 0.0}, "TARGET_MEAN"),
        ({"min_cohort": 0}, "MIN_COHORT"),
        ({"signal_scale": 0.0}, "SIGNAL_SCALE"),
        ({"clip": (50.0, 10.0)}, "CLIP"),
    ],
)
def test_policy_validation(kwargs, message):
    with pytest.raises(ImproperlyConfigured, match=message):
        NormalizationPolicy(**kwargs)


def test_policy_reads_both_the_nested_and_the_flat_shapes():
    flat = NormalizationPolicy.from_settings({"SHRINK": 0.5})
    nested = NormalizationPolicy.from_settings({"NORMALIZATION": {"SHRINK": 0.5}})
    assert flat.shrink == nested.shrink == 0.5
