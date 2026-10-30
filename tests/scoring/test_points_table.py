"""Document 11 section 12.1: the points table, its validation, and its settings."""

from __future__ import annotations

import copy

import pytest

from mactf import conf
from mactf.exceptions import ImproperlyConfigured
from mactf.rules.catalog import PENALTIES, REWARDS
from mactf.scoring import DEFAULT_POINTS_TABLE, PointsEngine, PointsTable
from mactf.scoring.points import DEFAULT_SEVERITY_FRACTIONS, SeverityPoints


def test_default_table_is_valid_and_rewards_sum_to_100():
    assert DEFAULT_POINTS_TABLE.validate() == []
    assert sum(DEFAULT_POINTS_TABLE.rewards.values()) == pytest.approx(100.0)
    assert set(DEFAULT_POINTS_TABLE.rewards) == {r.id for r in REWARDS}
    assert set(DEFAULT_POINTS_TABLE.penalties) == {p.id for p in PENALTIES}


def test_severity_fractions_are_monotone_and_anchor_s0_and_s4():
    severity = SeverityPoints(dict(DEFAULT_SEVERITY_FRACTIONS))
    values = [severity.fraction(f"S{i}") for i in range(5)]
    assert values == sorted(values)
    assert values[0] == 0.0 and values[4] == 1.0


def test_round_trip_through_a_dict():
    clone = PointsTable.from_dict(DEFAULT_POINTS_TABLE.as_dict())
    assert clone.as_dict() == DEFAULT_POINTS_TABLE.as_dict()


def test_a_missing_reward_is_reported():
    table = PointsTable(rewards={k: v for k, v in DEFAULT_POINTS_TABLE.rewards.items() if k != "R1"})
    assert any("missing R1" in problem for problem in table.validate())
    with pytest.raises(ImproperlyConfigured, match="missing R1"):
        table.assert_valid()


def test_rewards_that_do_not_sum_to_100_are_reported():
    rewards = dict(DEFAULT_POINTS_TABLE.rewards)
    rewards["R1"] += 5
    assert PointsTable(rewards=rewards).validate()


def test_an_unknown_rule_is_reported():
    penalties = {**DEFAULT_POINTS_TABLE.penalties, "P99": 1.0}
    assert any("P99" in problem for problem in PointsTable(penalties=penalties).validate())


def test_non_monotone_severity_is_reported():
    fractions = {**DEFAULT_SEVERITY_FRACTIONS, "S2": 0.9}
    assert PointsTable(severity=SeverityPoints(fractions)).validate()


def test_describe_summarises_the_table():
    text = DEFAULT_POINTS_TABLE.describe()
    assert "17 rewards" in text and "23 penalties" in text


def test_the_docx_reward_weights_are_not_mistaken_for_the_points_table():
    # SCORING.REWARD_WEIGHTS is the r_task/r_epi table; only SCORING.POINTS counts.
    table = PointsTable.from_settings({"SCORING": conf.DEFAULTS["SCORING"]})
    assert table.as_dict() == DEFAULT_POINTS_TABLE.as_dict()


def test_engine_reads_settings_and_overrides_the_cap():
    settings = copy.deepcopy(conf.DEFAULTS)
    settings["ENGINE"]["F_CAP"] = 1
    engine = PointsEngine.from_settings(settings)
    assert engine.f_cap == 1 and engine.bindings["R3"].max_instances == 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda s: s["ENGINE"].update(F_CAP=0), "F_CAP"),
        (lambda s: s["ENGINE"].update(P_MIN=0.0), "P_MIN"),
        (lambda s: s["ENGINE"].update(P12_NOISE_RATIO=1.5), "P12_NOISE_RATIO"),
        (lambda s: s["ENGINE"].update(R9_STEP_FACTOR=0.5), "R9_STEP_FACTOR"),
        (lambda s: s["COOPETITION"].update(SIGNAL_SOURCE="vibes"), "SIGNAL_SOURCE"),
        (lambda s: s["COOPETITION"]["NORMALIZATION"].update(SHRINK=2.0), "SHRINK"),
        (lambda s: s["SCORING"]["POINTS"]["REWARD_WEIGHTS"].pop("R1"), "missing R1"),
    ],
)
def test_validate_settings_rejects_bad_points_configuration(mutate, message):
    settings = copy.deepcopy(conf.DEFAULTS)
    mutate(settings)
    with pytest.raises(ImproperlyConfigured, match=message):
        conf._validate_points(settings)


def test_default_settings_pass_points_validation():
    conf._validate_points(copy.deepcopy(conf.DEFAULTS))
