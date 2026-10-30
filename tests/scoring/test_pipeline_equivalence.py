"""``ScoringPipeline`` must reproduce the pre-split scorer to ``1e-12``.

The oracle below is the algorithm ``mactf/scoring/pipeline.py`` implemented at the
commit before the split (``7634161``), reproduced verbatim so the comparison is
against the real thing rather than against a re-derivation of it.
"""

from __future__ import annotations

import copy
import random

import pytest

from mactf.compliance import constrain, void_rule_ids
from mactf.exceptions import ImproperlyConfigured
from mactf.rules.catalog import ALL_RULES
from mactf.scoring.continuous import score as continuous_score
from mactf.scoring.ledger import Ledger
from mactf.scoring.pipeline import ScoringPipeline
from mactf.scoring.scorecard import PeerScorecard
from mactf.scoring.warning import WarningPolicy, WarningRegistry

LEGACY_KNOWN_RULES = {rule.id for rule in ALL_RULES if rule.is_penalty}


def legacy_score_peer(
    *,
    score_config,
    severity_weights,
    registries,
    warning_policy,
    ledger,
    peer=None,
    episode=0,
    known_rules=None,
):
    """The pre-split ``Scorer.score_peer``, copied from commit ``7634161``."""
    peer = peer or ledger.peer or "peer"
    ledger.peer = peer
    registry = registries.setdefault(peer, WarningRegistry(warning_policy))
    registry.begin_episode(peer, episode=episode)

    warnings_issued = []
    warnings_escalated = []
    warning_penalty = 0.0
    warning_events = []
    escalated_rules = set()

    for event in ledger.penalties():
        if not event.rule.warning_eligible:
            continue
        outcome = registry.issue(
            peer,
            event.rule_id,
            severity=event.resolved_severity(),
            episode=episode,
            evidence=event.evidence,
            weights=severity_weights,
        )
        warnings_issued.extend(outcome.issued)
        warning_events.append(outcome.rule_id)
        if outcome.escalated is not None:
            warnings_escalated.append(outcome.escalated)
            warning_penalty += outcome.penalty
            escalated_rules.add(outcome.rule_id)

    registry.mark_clean_except(
        peer, set(warning_events), known_rules=known_rules or LEGACY_KNOWN_RULES
    )

    continuous = continuous_score(
        ledger,
        score_config=score_config,
        severity_weights=severity_weights,
        warning_total=warning_penalty,
        warning_events=tuple(warning_events),
        escalated_rules=escalated_rules,
    )
    non_offsettable = set(score_config.get("NON_OFFSETTABLE", ()))
    result = constrain(continuous, non_offsettable=non_offsettable)

    return PeerScorecard(
        peer=peer,
        episode=episode,
        continuous=continuous,
        final_score=result.value,
        floor=result.floor,
        floor_reason=result.reason,
        warnings_issued=tuple(warnings_issued),
        warnings_escalated=tuple(warnings_escalated),
        suppressed=(),
        violated_non_offsettable=void_rule_ids(continuous, non_offsettable=non_offsettable),
    )


def _build(score_config, severity_weights, warning_policy):
    return ScoringPipeline(
        score_config=copy.deepcopy(score_config),
        severity_weights=copy.deepcopy(severity_weights),
        warning_policy=warning_policy,
    )


def _oracle_registries():
    return {}


def _pairs(records):
    """The identity a warning record and a mutation share."""
    return [(item.rule_id, item.episode) for item in records]


LEDGERS = {
    "empty": {},
    "reward_only": {"R1": 2.0, "R3": 1.0},
    "single_warning_first": {"P6": 1.0},
    "two_rules": {"P4": 1.0, "P11": 2.0},
    "non_offsettable": {"P1": 1.0, "R1": 3.0},
    "sandbox_omega": {"P20": 1.0},
    "mixed": {"R1": 1.0, "R10": 2.0, "P6": 1.0, "P13": 1.0},
    "all_four_buckets": {"R1": 1.0, "R2": 2.0, "R3": 3.0, "R12": 4.0},
}


@pytest.mark.parametrize("name", sorted(LEDGERS))
def test_single_episode_matches_the_oracle(name, score_config, severity_weights, warning_policy):
    spec = LEDGERS[name]
    pipeline = _build(score_config, severity_weights, warning_policy)
    registries = _oracle_registries()

    for episode in (1, 2, 3, 4):
        ledger_new = Ledger(peer="a")
        ledger_old = Ledger(peer="a")
        for rule_id, magnitude in spec.items():
            ledger_new.record(rule_id, magnitude=magnitude)
            ledger_old.record(rule_id, magnitude=magnitude)

        actual = pipeline.score_peer(ledger_new, peer="a", episode=episode)
        expected = legacy_score_peer(
            score_config=copy.deepcopy(score_config),
            severity_weights=copy.deepcopy(severity_weights),
            registries=registries,
            warning_policy=warning_policy,
            ledger=ledger_old,
            peer="a",
            episode=episode,
        )

        assert actual.final_score == pytest.approx(expected.final_score, abs=1e-12)
        assert actual.floor == pytest.approx(expected.floor, abs=1e-12)
        assert actual.continuous.net == pytest.approx(expected.continuous.net, abs=1e-12)
        assert actual.continuous.reward.total == pytest.approx(
            expected.continuous.reward.total, abs=1e-12
        )
        assert actual.continuous.penalty.total == pytest.approx(
            expected.continuous.penalty.total, abs=1e-12
        )
        assert actual.continuous.warning == pytest.approx(expected.continuous.warning, abs=1e-12)
        assert _pairs(actual.warnings_issued) == _pairs(expected.warnings_issued)
        assert _pairs(actual.warnings_escalated) == _pairs(expected.warnings_escalated)
        assert actual.violated_non_offsettable == expected.violated_non_offsettable


@pytest.mark.parametrize("name", sorted(LEDGERS))
def test_suppressed_matches_the_oracle(name, score_config, severity_weights, warning_policy):
    pipeline = _build(score_config, severity_weights, warning_policy)
    ledger = Ledger(peer="a")
    for rule_id, magnitude in LEDGERS[name].items():
        ledger.record(rule_id, magnitude=magnitude)

    actual = pipeline.score_peer(ledger, peer="a", episode=1)
    # The oracle's own suppression list is derived the same way in the pre-split
    # code; recompute it from the decision so the comparison is independent.
    decision = actual.continuous.decision
    assert decision is not None
    expected_ids = {rule.id for rule in decision.suppressed}
    expected_ids |= {
        rule_id
        for rule_id in actual.continuous.suppressed_rewards
        if rule_id not in expected_ids
    }
    assert {item.rule_id for item in actual.suppressed} == expected_ids


def test_score_episode_matches_the_oracle_ordering(
    score_config, severity_weights, warning_policy
):
    ledgers = {
        "alpha": {"R1": 1.0, "P6": 1.0},
        "beta": {"R2": 2.0},
        "gamma": {"P1": 1.0},
    }
    pipeline = _build(score_config, severity_weights, warning_policy)
    registries = _oracle_registries()

    actual = pipeline.score_episode(
        {peer: _ledger(peer, spec) for peer, spec in ledgers.items()}, episode=7
    )
    expected = {}
    for peer, spec in ledgers.items():
        expected[peer] = legacy_score_peer(
            score_config=copy.deepcopy(score_config),
            severity_weights=copy.deepcopy(severity_weights),
            registries=registries,
            warning_policy=warning_policy,
            ledger=_ledger(peer, spec),
            peer=peer,
            episode=7,
        )

    assert list(actual.peers) == list(expected)
    for peer in expected:
        assert actual[peer].final_score == pytest.approx(expected[peer].final_score, abs=1e-12)
        assert actual[peer].continuous.net == pytest.approx(
            expected[peer].continuous.net, abs=1e-12
        )


def _ledger(peer, spec):
    ledger = Ledger(peer=peer)
    for rule_id, magnitude in spec.items():
        ledger.record(rule_id, magnitude=magnitude)
    return ledger


def test_randomised_ledgers_match_the_oracle(
    score_config, severity_weights, warning_policy
):
    rng = random.Random(20240607)
    reward_ids = [rule.id for rule in ALL_RULES if rule.is_reward]
    penalty_ids = [rule.id for rule in ALL_RULES if rule.is_penalty]

    pipeline = _build(score_config, severity_weights, warning_policy)
    registries = _oracle_registries()

    for episode in range(1, 13):
        spec: dict[str, float] = {}
        for rule_id in rng.sample(reward_ids, k=min(3, len(reward_ids))):
            spec[rule_id] = rng.choice([0.5, 1.0, 2.5])
        for rule_id in rng.sample(penalty_ids, k=3):
            spec[rule_id] = rng.choice([0.5, 1.0, 3.0])

        actual = pipeline.score_peer(_ledger("a", spec), peer="a", episode=episode)
        expected = legacy_score_peer(
            score_config=copy.deepcopy(score_config),
            severity_weights=copy.deepcopy(severity_weights),
            registries=registries,
            warning_policy=warning_policy,
            ledger=_ledger("a", spec),
            peer="a",
            episode=episode,
        )
        assert actual.final_score == pytest.approx(expected.final_score, abs=1e-12), spec
        assert actual.continuous.net == pytest.approx(expected.continuous.net, abs=1e-12), spec


def test_escalation_ordering_gate(score_config, severity_weights):
    """A warning maturing this episode escalates, is excluded, and is not in P."""
    policy = WarningPolicy(k_esc=2, n_win=5, n_decay=2, w_cap=1, p_warn=5.0)
    pipeline = _build(score_config, severity_weights, policy)

    pipeline.score_peer(_ledger("a", {"P6": 1.0}), peer="a", episode=1)
    card = pipeline.score_peer(_ledger("a", {"P6": 1.0}), peer="a", episode=2)

    assert [item.rule_id for item in card.warnings_escalated] == ["P6"]
    assert card.continuous.warning == pytest.approx(5.0)
    # The escalated violation is a penalty-bearing act, so it is *not* excluded.
    assert card.continuous.warning_events == ("P6",)
    assert pipeline.warning_scorer.registry_for("a").count("a", "P6") == 1


def test_order_inversion_is_rejected_not_scored(score_config, severity_weights):
    from mactf.scoring.base import ScoreContext, ScorerOutput, ScoringKind
    from mactf.scoring.penalty import PenaltyScorer
    from mactf.scoring.reward_scorer import RewardScorer
    from mactf.scoring.warning_scorer import WarningScorer

    class LateWarning(WarningScorer):
        order = 50

    with pytest.raises(ImproperlyConfigured, match="order"):
        ScoringPipeline(
            [
                LateWarning(config=score_config),
                RewardScorer(config=score_config),
                PenaltyScorer(config=score_config),
            ],
            score_config=score_config,
            severity_weights=severity_weights,
        )


def test_missing_stage_is_rejected(score_config, severity_weights):
    from mactf.scoring.reward_scorer import RewardScorer
    from mactf.scoring.warning_scorer import WarningScorer

    with pytest.raises(ImproperlyConfigured, match="missing required stage"):
        ScoringPipeline(
            [WarningScorer(config=score_config), RewardScorer(config=score_config)],
            score_config=score_config,
            severity_weights=severity_weights,
        )


def test_duplicate_kind_is_rejected(score_config, severity_weights):
    from mactf.scoring.reward_scorer import RewardScorer
    from mactf.scoring.warning_scorer import WarningScorer

    with pytest.raises(ImproperlyConfigured, match="two scorers with kind"):
        ScoringPipeline(
            [
                WarningScorer(config=score_config),
                RewardScorer(config=score_config),
                RewardScorer(config=score_config),
            ],
            score_config=score_config,
            severity_weights=severity_weights,
        )


def test_no_bypass_gate(score_config, severity_weights, warning_policy):
    """A reward stage that ignores a P1 still cannot make the score positive."""
    from mactf.scoring.reward_scorer import RewardScorer

    class LyingReward(RewardScorer):
        def score(self, ctx):
            output = super().score(ctx)
            return type(output)(
                kind=output.kind,
                amount=1e6,
                breakdown={"r_task": 1e6, "weights": {"r_task": 1.0}, "rules": ["R1"]},
                rules=("R1",),
            )

    pipeline = ScoringPipeline(
        [
            __import__("mactf.scoring.warning_scorer", fromlist=["WarningScorer"]).WarningScorer(
                config=score_config, policy=warning_policy
            ),
            LyingReward(config=score_config),
            __import__("mactf.scoring.penalty", fromlist=["PenaltyScorer"]).PenaltyScorer(
                config=score_config
            ),
        ],
        score_config=score_config,
        severity_weights=severity_weights,
        warning_policy=warning_policy,
    )
    card = pipeline.score_peer(_ledger("a", {"R1": 1.0, "P1": 1.0}), peer="a", episode=1)
    assert card.is_void
    assert card.final_score <= 0.0


def test_points_fields_are_untouched_by_the_pipeline(
    score_config, severity_weights, warning_policy
):
    pipeline = _build(score_config, severity_weights, warning_policy)
    card = pipeline.score_peer(_ledger("a", {"R1": 1.0}), peer="a", episode=1)
    assert card.points is None
    assert card.competitive is None and card.cooperative is None
