"""``TRAINER.MUTATION``: the block validator and the limits it builds (document 17 section 12.10)."""

import logging

import pytest

from mactf.exceptions import ImproperlyConfigured, MutationError
from mactf.training.mutation import MutationLimits, MutationPlan


def test_unknown_key_is_refused():
    with pytest.raises(MutationError) as caught:
        MutationLimits.validate_block({"ENABLE": True})
    assert caught.value.code == "E080" and "ENABLED" in str(caught.value) and "MAX_OPS_PER_PLAN" in str(caught.value)


def test_min_active_roots_cannot_exceed_the_tree():
    with pytest.raises(MutationError, match="exceeds the 2 roots"):
        MutationLimits.from_settings({"MIN_ACTIVE_ROOTS": 3}, initial_roots=2)
    assert MutationLimits.from_settings({"MIN_ACTIVE_ROOTS": 2}, initial_roots=2).min_active_roots == 2


def test_min_active_roots_defaults_to_two_or_the_roots_when_fewer():
    assert MutationLimits.from_settings({}, initial_roots=5).min_active_roots == 2
    assert MutationLimits.from_settings({}, initial_roots=1).min_active_roots == 1


def test_schedule_entries_are_parsed_at_load():
    good = {"SCHEDULE": [{"AT_EPOCH": 2, "OPS": [{"op": "retire", "path": "p"}]}]}
    limits = MutationLimits.from_settings({**good, "ENABLED": True}, initial_roots=3)
    ((at, plan),) = limits.schedule
    assert at == 2 and isinstance(plan, MutationPlan) and plan.label == "schedule:0"
    for bad in (
        [{"AT_EPOCH": 2, "OPS": [{"op": "explode", "path": "p"}]}],       # a typo in an op
        [{"OPS": [{"op": "retire", "path": "p"}]}],                       # no epoch
        [{"AT_EPOCH": -1, "OPS": [{"op": "retire", "path": "p"}]}],
        [{"AT_EPOCH": True, "OPS": [{"op": "retire", "path": "p"}]}],
        [{"AT_EPOCH": 1, "OPS": [{"op": "retire", "path": "p"}], "WHEN": 3}],
        ["retire p"],
        "retire p",
    ):
        with pytest.raises(MutationError) as caught:
            MutationLimits.validate_block({"SCHEDULE": bad})
        assert caught.value.code == "E080"


def test_a_schedule_with_mutation_disabled_is_a_configuration_error():
    from mactf import conf

    settings = {"TRAINER": {"MUTATION": {"SCHEDULE": [{"AT_EPOCH": 1, "OPS": [{"op": "retire", "path": "p"}]}]}}}
    with pytest.raises(ImproperlyConfigured, match="never run"):
        conf._validate_mutation_block(settings)
    settings["TRAINER"]["MUTATION"]["ENABLED"] = True
    conf._validate_mutation_block(settings)


@pytest.mark.parametrize(
    "block",
    [
        {"ENABLED": "yes"}, {"ALLOW_PURGE": 1}, {"STRICT_REPLAY": None},
        {"MAX_OPS_PER_PLAN": 0}, {"MAX_PLANS_PER_EPOCH": True}, {"JOURNAL_WARN": -3}, {"MIN_ACTIVE_ROOTS": 1.5},
        {"TOMBSTONE": "forever"}, {"WATCHDOG_PRUNE": "delete"},
        {"PROTECTED": "peer-1"}, {"PROTECTED": ["ok", ""]}, {"PROTECTED": [3]},
    ],
)
def test_every_bad_value_is_e080(block):
    with pytest.raises(MutationError) as caught:
        MutationLimits.validate_block(block)
    assert caught.value.code == "E080"


def test_protected_globs_match_paths_and_globs():
    limits = MutationLimits(protected=("peer-1", "team-a/*"))
    assert limits.is_protected("peer-1") and limits.is_protected("team-a/x")
    assert not limits.is_protected("peer-2") and not limits.is_protected("team-b/x")


def test_the_defaults_are_inert():
    limits = MutationLimits()
    assert limits.enabled is False and limits.schedule == () and limits.protected == ()
    assert limits.tombstone == "run" and limits.watchdog_prune == "purge"


def test_w044_reports_a_schedule_that_will_trip_the_growth_probe(caplog):
    block = {"ENABLED": True, "SCHEDULE": [
        {"AT_EPOCH": 1, "OPS": [{"op": "add", "path": f"n{i}"} for i in range(6)]}
    ]}
    with caplog.at_level(logging.WARNING, logger="mactf.mutation"):
        MutationLimits.from_settings(block, initial_roots=3, initial_nodes=3, growth_ratio=2.0)
    assert any("W044" in record.message for record in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="mactf.mutation"):
        MutationLimits.from_settings(block, initial_roots=3, initial_nodes=3, growth_ratio=10.0)
    assert not caplog.records
