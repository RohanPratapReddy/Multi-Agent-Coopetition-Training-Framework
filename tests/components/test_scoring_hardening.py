"""H1 and H2 (document 17 section 6.6): the floor can be widened, never narrowed or zeroed."""

from __future__ import annotations

import copy

import pytest

from mactf import conf
from mactf.compliance import DEFAULT_MIN_FLOOR_LAMBDA, effective_non_offsettable
from mactf.exceptions import ImproperlyConfigured
from mactf.rules.catalog import NON_OFFSETTABLE
from mactf.scoring.hierarchy import HierarchicalScorer
from mactf.scoring.penalty import PenaltyScorer
from mactf.scoring.pipeline import ScoringPipeline, build_scorer

from scoring_helpers import default_pipeline, ledger_of, settings_copy

CONFIG = settings_copy()["SCORING"]


def validated(**blocks):
    settings = conf.Settings(copy.deepcopy(conf.DEFAULTS))
    for name, values in blocks.items():
        if name == "SCORING":
            settings["SCORING"].update(values)
        else:
            settings[name] = {**settings.get(name, {}), **values} if isinstance(values, dict) else values
    conf.validate_settings(settings)
    return settings


# ------------------------------------------------------------------- H1
def test_the_effective_set_is_the_catalog_set_plus_anything_configured():
    assert effective_non_offsettable() == NON_OFFSETTABLE
    assert effective_non_offsettable(None) == NON_OFFSETTABLE
    assert effective_non_offsettable([]) == NON_OFFSETTABLE  # an empty list narrows nothing
    assert effective_non_offsettable(["P3"]) == NON_OFFSETTABLE | {"P3"}
    assert effective_non_offsettable(["P1"]) == NON_OFFSETTABLE
    assert effective_non_offsettable(iter(["P4", "P4"])) == NON_OFFSETTABLE | {"P4"}


@pytest.mark.parametrize("narrowed", [[], ["P3"], ["P17"], ["P1", "P2"]])
def test_penalty_scorer_cannot_be_given_a_narrower_set(narrowed):
    stage = PenaltyScorer(config={**CONFIG, "NON_OFFSETTABLE": narrowed})
    assert stage.non_offsettable >= NON_OFFSETTABLE


def test_a_narrowed_set_still_voids_a_catalog_breach_through_the_pipeline():
    config = {**CONFIG, "NON_OFFSETTABLE": []}
    pipeline = ScoringPipeline(score_config=config, severity_weights=settings_copy()["SEVERITY"]["WEIGHTS"])
    card = pipeline.score_peer(ledger_of({"R1": 1.0, "P1": 1.0}), peer="a", episode=1)
    assert card.is_void and card.violated_non_offsettable == ("P1",)
    assert card.floor == -5.0
    # the pre-hardening code read the configured list alone, so [] meant "nothing is a floor"


def test_a_widened_set_adds_a_floor():
    config = {**CONFIG, "NON_OFFSETTABLE": [*NON_OFFSETTABLE, "P3"]}
    pipeline = ScoringPipeline(score_config=config, severity_weights=settings_copy()["SEVERITY"]["WEIGHTS"])
    card = pipeline.score_peer(ledger_of({"P3": 2.0}), peer="a", episode=1)
    assert card.violated_non_offsettable == ("P3",) and card.floor == -6.0  # lambda 3 x 2


def test_the_guard_reference_uses_the_widened_set_too():
    from mactf.components.guards.compliance import ComplianceGuard
    from mactf.scoring.base import ScoreContext

    guard = ComplianceGuard()
    config = {**CONFIG, "NON_OFFSETTABLE": ["P3"]}
    ctx = ScoreContext(peer="a", episode=1, ledger=ledger_of({"P3": 1.0}), config=config,
                       severity_weights=settings_copy()["SEVERITY"]["WEIGHTS"])
    reference = guard.reference_penalty(ctx, config)
    assert reference.void_rules == ("P3",)


def test_the_hierarchy_scorer_takes_the_same_superset():
    assert HierarchicalScorer(non_offsettable=None).non_offsettable == NON_OFFSETTABLE
    assert HierarchicalScorer(non_offsettable=[]).non_offsettable == NON_OFFSETTABLE
    assert HierarchicalScorer(non_offsettable=["P3"]).non_offsettable == NON_OFFSETTABLE | {"P3"}


def _scored_tree(non_offsettable, child_rule):
    from mactf.coopetition.tree import PeerTree

    tree = PeerTree()
    tree.add_node("peer-a")
    tree.add_node("peer-a/sub-1")
    pipeline = ScoringPipeline(
        score_config=dict(CONFIG), severity_weights=settings_copy()["SEVERITY"]["WEIGHTS"]
    )
    scorer = HierarchicalScorer(pipeline, attribution="own", non_offsettable=non_offsettable)
    ledgers = {"peer-a/sub-1": ledger_of({child_rule: 1.0}, peer="peer-a/sub-1")}
    return scorer.score_tree(tree, ledgers)


def test_a_custom_set_can_no_longer_narrow_the_catalog_in_a_tree():
    """Before H1 ``non_offsettable=['P3']`` made a child's P1 stop voiding its parent."""
    card = _scored_tree(["P3"], "P1")
    assert card.peers["peer-a"].is_void and "P1" in card.peers["peer-a"].branch_void
    assert _scored_tree(None, "P1").peers["peer-a"].is_void
    assert _scored_tree([], "P1").peers["peer-a"].is_void


def test_a_custom_set_widens_the_void_a_child_propagates():
    assert _scored_tree(["P3"], "P3").peers["peer-a"].is_void
    assert not _scored_tree(None, "P3").peers["peer-a"].is_void


# ------------------------------------------------------------------- H2
def test_the_default_floor_changes_nothing():
    assert DEFAULT_MIN_FLOOR_LAMBDA == CONFIG["DEFAULT_LAMBDA"] == 1.0
    assert all(CONFIG["PENALTY_LAMBDAS"][rule] >= 5.0 for rule in NON_OFFSETTABLE)


def test_a_zero_lambda_on_a_non_offsettable_rule_is_lifted_to_the_floor():
    config = {**CONFIG, "PENALTY_LAMBDAS": {**CONFIG["PENALTY_LAMBDAS"], "P1": 0.0, "P2": 0.5}}
    stage = PenaltyScorer(config=config)
    card = ScoringPipeline(
        [__import__("mactf.scoring.warning_scorer", fromlist=["x"]).WarningScorer(config=config),
         __import__("mactf.scoring.reward_scorer", fromlist=["x"]).RewardScorer(config=config), stage],
        score_config=config, severity_weights=settings_copy()["SEVERITY"]["WEIGHTS"],
    ).score_peer(ledger_of({"P1": 3.0}), peer="a", episode=1)
    assert card.continuous.penalty.amounts["P1"] == 3.0  # lambda 0 -> 1 (the default floor)
    assert PenaltyScorer(config={**config, "MIN_FLOOR_LAMBDA": 4.0}).min_floor_lambda == 4.0


def test_an_offsettable_lambda_is_not_floored():
    config = {**CONFIG, "PENALTY_LAMBDAS": {**CONFIG["PENALTY_LAMBDAS"], "P4": 0.0}}
    pipeline = ScoringPipeline(score_config=config, severity_weights=settings_copy()["SEVERITY"]["WEIGHTS"])
    card = pipeline.score_peer(ledger_of({"P4": 3.0, "P3": 1.0}), peer="a", episode=1)
    assert card.continuous.penalty.amounts.get("P4", 0.0) == 0.0


def test_validate_reports_a_narrowed_set_and_a_weak_floor_before_a_run():
    narrowed = PenaltyScorer(config=CONFIG).validate({"NON_OFFSETTABLE": ["P1"]})
    assert [d.code for d in narrowed] == ["penalties.non_offsettable_narrowed"]
    assert "P17" in narrowed[0].message and narrowed[0].level == "error"
    weak = PenaltyScorer(config=CONFIG).validate({"PENALTY_LAMBDAS": {"P17": 0.5}, "MIN_FLOOR_LAMBDA": 2.0})
    codes = {d.code for d in weak}
    assert "penalties.floor_lambda_below_minimum" in codes
    messages = " ".join(d.message for d in weak)
    assert "P17" in messages and "P1'" in messages  # P1 falls back to DEFAULT_LAMBDA (1.0) < 2.0
    assert PenaltyScorer(config=CONFIG).validate({}) == []
    assert PenaltyScorer(config=CONFIG).validate({"NON_OFFSETTABLE": [*NON_OFFSETTABLE, "P3"]}) == []


def test_a_negative_lambda_is_reported_once_not_twice():
    diagnostics = PenaltyScorer().validate({"PENALTY_LAMBDAS": {"P1": -1.0}})
    assert [d.code for d in diagnostics] == ["penalties.negative_lambda"]


# ------------------------------------------------------------- configuration load
def test_the_compliance_block_has_the_documented_defaults():
    assert conf.COMPLIANCE_DEFAULTS == {
        "MIN_FLOOR_LAMBDA": 1.0, "GUARD_MODE": "enforce", "REWARD_CEILING": 10000.0,
        "TOLERANCE": 1e-9, "TRUST_BUILTIN": True,
    }
    assert conf.DEFAULTS["COMPLIANCE"] == conf.COMPLIANCE_DEFAULTS
    assert conf.DEFAULTS["COMPLIANCE"] is not conf.COMPLIANCE_DEFAULTS
    assert "STAGES" not in conf.DEFAULTS["SCORING"]  # the default is the key being absent
    validated()  # and the defaults pass their own validator


@pytest.mark.parametrize(
    "block,fragment",
    [
        ({"GUARD_MODE": "off"}, "GUARD_MODE"),
        ({"GUARD_MODE": ""}, "GUARD_MODE"),
        ({"MIN_FLOOR_LAMBDA": 0}, "MIN_FLOOR_LAMBDA is out of range"),
        ({"MIN_FLOOR_LAMBDA": -1}, "MIN_FLOOR_LAMBDA is out of range"),
        ({"MIN_FLOOR_LAMBDA": float("nan")}, "MIN_FLOOR_LAMBDA is out of range"),
        ({"MIN_FLOOR_LAMBDA": "high"}, "MIN_FLOOR_LAMBDA must be a number"),
        ({"REWARD_CEILING": 0}, "REWARD_CEILING is out of range"),
        ({"REWARD_CEILING": float("inf")}, "REWARD_CEILING is out of range"),
        ({"TOLERANCE": -1e-9}, "TOLERANCE is out of range"),
        ({"TOLERANCE": 0.01}, "TOLERANCE is out of range"),
        ({"TRUST_BUILTIN": "yes"}, "TRUST_BUILTIN"),
        ({"MIN_FLOOR_LAMBDAS": 1}, "unknown key"),
    ],
)
def test_a_bad_compliance_block_is_refused(block, fragment):
    with pytest.raises(ImproperlyConfigured, match=fragment):
        validated(COMPLIANCE=block)


def test_compliance_must_be_a_mapping():
    settings = conf.Settings(copy.deepcopy(conf.DEFAULTS))
    settings["COMPLIANCE"] = 3
    with pytest.raises(ImproperlyConfigured, match="COMPLIANCE must be a mapping"):
        conf.validate_settings(settings)


def test_boundary_values_are_accepted():
    validated(COMPLIANCE={"TOLERANCE": 0.0, "MIN_FLOOR_LAMBDA": 5.0, "REWARD_CEILING": 1e-9, "GUARD_MODE": "FALLBACK"})
    validated(COMPLIANCE={"TOLERANCE": 1e-3})


@pytest.mark.parametrize("rule", ["P1", "P2", "P17", "P20", "P21", "P22"])
def test_a_floor_lambda_below_the_minimum_is_refused_at_load_naming_the_rule(rule):
    lambdas = {**CONFIG["PENALTY_LAMBDAS"], rule: 0.0}
    with pytest.raises(ImproperlyConfigured) as excinfo:
        validated(SCORING={"PENALTY_LAMBDAS": lambdas})
    assert rule in str(excinfo.value) and "MIN_FLOOR_LAMBDA" in str(excinfo.value)


def test_the_check_covers_the_reserved_rules_even_when_the_set_is_not_listed_in_lambdas():
    # a rule missing from PENALTY_LAMBDAS falls back to DEFAULT_LAMBDA, which is checked too
    lambdas = {k: v for k, v in CONFIG["PENALTY_LAMBDAS"].items() if k != "P17"}
    validated(SCORING={"PENALTY_LAMBDAS": lambdas})  # default 1.0 >= floor 1.0
    with pytest.raises(ImproperlyConfigured, match="P17"):
        validated(SCORING={"PENALTY_LAMBDAS": lambdas}, COMPLIANCE={"MIN_FLOOR_LAMBDA": 2.0})
    with pytest.raises(ImproperlyConfigured, match="P3"):
        validated(
            SCORING={"NON_OFFSETTABLE": [*NON_OFFSETTABLE, "P3"], "PENALTY_LAMBDAS": {**CONFIG["PENALTY_LAMBDAS"], "P3": 0.5}}
        )
    with pytest.raises(ImproperlyConfigured, match="not a number"):
        validated(SCORING={"PENALTY_LAMBDAS": {**CONFIG["PENALTY_LAMBDAS"], "P1": "big"}})


def test_a_higher_floor_is_accepted_when_the_lambdas_clear_it():
    validated(COMPLIANCE={"MIN_FLOOR_LAMBDA": 5.0})
    with pytest.raises(ImproperlyConfigured):
        validated(COMPLIANCE={"MIN_FLOOR_LAMBDA": 5.1})


def test_the_catalog_set_cannot_be_dropped_from_the_configuration():
    with pytest.raises(ImproperlyConfigured, match="NON_OFFSETTABLE must include"):
        validated(SCORING={"NON_OFFSETTABLE": ["P1"]})
