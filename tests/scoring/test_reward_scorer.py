"""``RewardScorer`` must equal ``continuous._reward_breakdown`` exactly."""

from __future__ import annotations

import pytest

from mactf.exceptions import ImproperlyConfigured
from mactf.scoring.base import ScoreContext
from mactf.scoring.continuous import _reward_breakdown
from mactf.scoring.ledger import Ledger
from mactf.scoring.reward_scorer import (
    BUCKETS,
    DEFAULT_REWARD_WEIGHTS,
    RewardScorer,
)


def _ctx(ledger: Ledger, config=None, weights=None) -> ScoreContext:
    return ScoreContext(
        peer=ledger.peer or "a",
        episode=1,
        ledger=ledger,
        config=dict(config or {}),
        severity_weights=dict(weights or {}),
    )


def test_buckets_match_the_docx_reward_equation():
    assert BUCKETS == ("r_task", "r_epi", "r_collab", "r_align")
    assert DEFAULT_REWARD_WEIGHTS == {
        "r_task": 1.0,
        "r_epi": 1.5,
        "r_collab": 1.2,
        "r_align": 1.0,
    }


def test_matches_the_continuous_breakdown_for_a_mixed_ledger(score_config, severity_weights):
    ledger = Ledger(peer="a")
    ledger.record("R1", magnitude=1.0, severity="S2")
    ledger.record("R2", magnitude=2.0, severity="S3")
    ledger.record("R4", magnitude=1.0, severity="S1")
    ledger.record("R12", magnitude=3.0, severity="S2")

    config = dict(score_config) | {"SEVERITY_WEIGHTS": severity_weights}
    expected = _reward_breakdown(ledger, config)
    actual = RewardScorer(config=config).score(_ctx(ledger, score_config, severity_weights))

    assert actual.amount == pytest.approx(expected.total, abs=1e-12)
    for bucket in BUCKETS:
        assert actual.breakdown[bucket] == pytest.approx(getattr(expected, bucket), abs=1e-12)
    assert actual.breakdown["by_rule"] == pytest.approx(expected.by_rule)


def test_severity_zero_contributes_nothing(severity_weights):
    ledger = Ledger(peer="a")
    ledger.record("R1", magnitude=1.0, severity="S0")
    output = RewardScorer().score(_ctx(ledger, weights=severity_weights))
    assert output.amount == pytest.approx(0.0)
    assert output.breakdown["r_task"] == pytest.approx(0.0)


def test_severity_scale_is_multiplicative(score_config, severity_weights):
    ledger = Ledger(peer="a")
    ledger.record("R1", magnitude=2.0, severity="S3")  # 4.0 scale
    output = RewardScorer(config=score_config).score(
        _ctx(ledger, score_config, severity_weights)
    )
    assert output.breakdown["r_task"] == pytest.approx(8.0)
    assert output.amount == pytest.approx(8.0)


def test_reads_rewards_only():
    ledger = Ledger(peer="a")
    ledger.record("R1", magnitude=1.0, severity="S2")
    ledger.record("P1", magnitude=100.0, severity="S3")
    output = RewardScorer().score(_ctx(ledger))
    assert "P1" not in output.rules
    assert output.amount > 0.0


def test_repeated_events_accumulate_within_a_rule():
    ledger = Ledger(peer="a")
    ledger.record("R1", magnitude=1.0, severity="S2")
    ledger.record("R1", magnitude=2.0, severity="S2")
    output = RewardScorer().score(_ctx(ledger))
    assert output.breakdown["r_task"] == pytest.approx(3.0)
    assert output.breakdown["by_rule"]["R1"] == pytest.approx(3.0)
    assert output.rules == ("R1",)


def test_unknown_bucket_is_recorded_not_silently_dropped():
    ledger = Ledger(peer="a")
    ledger.record("R1", magnitude=1.0, severity="S2")
    output = RewardScorer().score(_ctx(ledger))
    # R1 feeds r_task, so nothing is ignored here.
    assert output.breakdown["ignored_rules"] == []

    # A rule whose bucket is outside BUCKETS is reported. R1 is the only reward
    # with r_task; simulate a bogus bucket by patching the ledger's rule view.
    class Weird(RewardScorer):
        BUCKETS = ("r_epi",)

        def score(self, ctx):
            return super().score(ctx)

    output = Weird().score(_ctx(ledger))
    assert output.breakdown["ignored_rules"] == ["R1"]


def test_weight_defaults_to_one_for_an_absent_bucket():
    ledger = Ledger(peer="a")
    ledger.record("R1", magnitude=2.0, severity="S2")
    output = RewardScorer(config={"REWARD_WEIGHTS": {"r_epi": 3.0}}).score(_ctx(ledger))
    assert output.amount == pytest.approx(2.0)


def test_total_is_the_weighted_sum_not_the_raw_bucket_sum(score_config):
    ledger = Ledger(peer="a")
    ledger.record("R1", magnitude=1.0, severity="S2")
    ledger.record("R2", magnitude=1.0, severity="S2")
    output = RewardScorer(config=score_config).score(_ctx(ledger, score_config))
    assert output.amount == pytest.approx(1.0 * 1.0 + 1.5 * 1.0)


def test_validate_rejects_a_negative_weight():
    diagnostics = RewardScorer().validate({"REWARD_WEIGHTS": {"r_task": -1.0}})
    assert [item.code for item in diagnostics] == ["rewards.negative_weight"]
    assert diagnostics[0].level == "error"


def test_validate_warns_on_an_unknown_bucket():
    diagnostics = RewardScorer().validate({"REWARD_WEIGHTS": {"r_bogus": 1.0}})
    assert [item.code for item in diagnostics] == ["rewards.unknown_bucket"]
    assert diagnostics[0].level == "warning"


def test_validate_rejects_a_non_numeric_weight():
    diagnostics = RewardScorer().validate({"REWARD_WEIGHTS": {"r_task": "heavy"}})
    assert [item.code for item in diagnostics] == ["rewards.non_numeric_weight"]


def test_assert_valid_raises_naming_the_problem():
    with pytest.raises(ImproperlyConfigured, match="rewards.negative_weight"):
        RewardScorer().assert_valid({"REWARD_WEIGHTS": {"r_task": -2.0}})


def test_assert_valid_accepts_a_warning_level_diagnostic():
    RewardScorer().assert_valid({"REWARD_WEIGHTS": {"r_bogus": 1.0}})


def test_breakdown_shape_carries_every_documented_key(score_config):
    ledger = Ledger(peer="a")
    ledger.record("R1", magnitude=1.0, severity="S2")
    breakdown = RewardScorer(config=score_config).score(_ctx(ledger, score_config)).breakdown
    for key in (*BUCKETS, "weights", "rules", "by_rule", "ignored_rules", "total"):
        assert key in breakdown


def test_empty_ledger_scores_zero():
    output = RewardScorer().score(_ctx(Ledger(peer="a")))
    assert output.amount == pytest.approx(0.0)
    assert output.rules == ()
