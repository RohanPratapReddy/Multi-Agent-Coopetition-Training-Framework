"""The scorecard's competitive/cooperative split, per framework.md 9.2."""

from __future__ import annotations

import copy

import pytest

from mactf.coopetition.pool import (
    COOPERATIVE_RULES,
    CooperativeBudget,
    CooperativePool,
)
from mactf.scoring.ledger import Ledger
from mactf.scoring.pipeline import ScoringPipeline, build_pool, build_scorer
from mactf.scoring.warning import WarningPolicy


@pytest.fixture()
def pipeline(score_config, severity_weights, warning_policy):
    return ScoringPipeline(
        score_config=copy.deepcopy(score_config),
        severity_weights=copy.deepcopy(severity_weights),
        warning_policy=warning_policy,
        pool=CooperativePool(CooperativeBudget(per_episode=6.0, per_peer_cap=6.0)),
    )


@pytest.fixture()
def pool_free(score_config, severity_weights, warning_policy):
    return ScoringPipeline(
        score_config=copy.deepcopy(score_config),
        severity_weights=copy.deepcopy(severity_weights),
        warning_policy=warning_policy,
    )


def _cooperative_ledger(peer: str = "a") -> Ledger:
    """A ledger with R1 plus all three cooperative rules at their default severity."""
    ledger = Ledger(peer=peer)
    ledger.record("R1", magnitude=1.0, severity="S2")          # 1.0 * 1.0
    ledger.record("R10", magnitude=1.0, severity="S2")         # 1.2 * 1.0
    ledger.record("R11", magnitude=1.0, severity="S2")         # 1.2 * 1.0
    ledger.record("R16", magnitude=1.0, severity="S2")         # 1.2 * 1.0
    return ledger


def test_split_reports_five_points_of_cooperative_reward(pipeline):
    card = pipeline.score_peer(_cooperative_ledger(), peer="a", episode=1)
    # R10/R11/R16 all land in r_collab, weighted 1.2 each.
    assert card.cooperative == pytest.approx(3.6)
    assert card.cooperative_denied == pytest.approx(0.0)
    assert card.competitive == pytest.approx(1.0)  # R1 alone
    assert card.competitive + card.cooperative == pytest.approx(
        card.continuous.reward.total
    )
    assert card.has_component_split


def test_split_uses_configured_collab_weight(score_config, severity_weights, warning_policy):
    config = copy.deepcopy(score_config)
    config["REWARD_WEIGHTS"]["r_collab"] = 1.0
    pipeline = ScoringPipeline(
        score_config=config,
        severity_weights=copy.deepcopy(severity_weights),
        warning_policy=warning_policy,
        pool=CooperativePool(CooperativeBudget(per_episode=6.0, per_peer_cap=6.0)),
    )
    card = pipeline.score_peer(_cooperative_ledger(), peer="a", episode=1)
    assert card.cooperative == pytest.approx(3.0)
    assert card.competitive == pytest.approx(1.0)


def test_pool_does_not_change_final_score(pipeline, pool_free):
    with_pool = pipeline.score_peer(_cooperative_ledger(), peer="a", episode=1)
    without = pool_free.score_peer(_cooperative_ledger(), peer="a", episode=1)
    assert with_pool.final_score == without.final_score
    assert with_pool.continuous.net == without.continuous.net
    assert without.competitive is None
    assert without.cooperative is None
    assert without.cooperative_denied == 0.0
    assert not without.has_component_split


def test_to_dict_emits_null_when_the_split_is_disabled(pool_free):
    payload = pool_free.score_peer(_cooperative_ledger(), peer="a", episode=1).to_dict()
    assert "competitive" in payload and payload["competitive"] is None
    assert "cooperative" in payload and payload["cooperative"] is None
    assert payload["cooperative_denied"] == 0.0


def test_to_dict_keeps_every_existing_key(pipeline):
    payload = pipeline.score_peer(_cooperative_ledger(), peer="a", episode=1).to_dict()
    for key in (
        "peer",
        "episode",
        "final_score",
        "severity",
        "void",
        "continuous",
        "floor",
        "floor_reason",
        "suppressed",
        "non_offsettable",
        "warnings_issued",
        "warnings_escalated",
        "metadata",
    ):
        assert key in payload
    assert payload["cooperative"] == pytest.approx(3.6)


def test_episode_scoring_shares_one_budget_across_the_cohort(
    score_config, severity_weights, warning_policy
):
    pipeline = ScoringPipeline(
        score_config=copy.deepcopy(score_config),
        severity_weights=copy.deepcopy(severity_weights),
        warning_policy=warning_policy,
        pool=CooperativePool(CooperativeBudget(per_episode=3.6, per_peer_cap=3.6)),
    )
    ledgers = {peer: _cooperative_ledger(peer) for peer in ("a", "b")}
    card = pipeline.score_episode(ledgers, episode=1)
    # 7.2 requested against a 3.6 pool -> a 0.5 scale for both peers.
    assert card["a"].cooperative == pytest.approx(1.8)
    assert card["b"].cooperative == pytest.approx(1.8)
    assert card["a"].cooperative_denied == pytest.approx(1.8)
    assert card["a"].competitive == pytest.approx(1.0)
    assert card["a"].final_score == pytest.approx(
        ScoringPipeline(
            score_config=copy.deepcopy(score_config),
            severity_weights=copy.deepcopy(severity_weights),
            warning_policy=warning_policy,
        )
        .score_peer(_cooperative_ledger("a"), peer="a", episode=1)
        .final_score
    )


def test_per_peer_cap_limits_a_single_peer(pipeline):
    pipeline.pool = CooperativePool(CooperativeBudget(per_episode=100.0, per_peer_cap=1.0))
    card = pipeline.score_peer(_cooperative_ledger(), peer="a", episode=1)
    assert card.cooperative == pytest.approx(1.0)
    assert card.cooperative_denied == pytest.approx(2.6)
    # `competitive` is still the honest non-cooperative part of the reward.
    assert card.competitive == pytest.approx(1.0)


def test_explicit_none_pool_disables_the_split(pipeline):
    card = pipeline.score_peer(_cooperative_ledger(), peer="a", episode=1, pool=None)
    assert card.competitive is None
    assert card.cooperative is None
    assert not card.has_component_split

    # Omitting the argument keeps using the scorer's configured pool.
    defaulted = pipeline.score_peer(_cooperative_ledger(), peer="b", episode=1)
    assert defaulted.has_component_split


def test_explain_reports_the_split(pipeline):
    text = pipeline.score_peer(_cooperative_ledger(), peer="a", episode=1).explain()
    assert "split: competitive=" in text
    assert "cooperative=" in text


def test_metadata_records_the_allocation(pipeline):
    card = pipeline.score_peer(_cooperative_ledger(), peer="a", episode=1)
    allocation = card.metadata["cooperative_pool"]
    assert allocation["awards"] == {"a": pytest.approx(3.6)}
    assert allocation["granted"] == pytest.approx(3.6)


def test_build_pool_respects_the_enable_flag(settings):
    assert build_pool(settings) is not None
    disabled = settings.as_dict()
    disabled["COOPETITION"]["COOPERATIVE_POOL"] = False
    from mactf.conf import Settings

    assert build_pool(Settings(disabled)) is None


def test_build_scorer_wires_the_default_budget(settings):
    scorer = build_scorer(settings)
    assert scorer.pool is not None
    assert scorer.pool.budget.per_episode == pytest.approx(6.0)
    assert scorer.pool.budget.per_peer_cap == pytest.approx(3.0)


def test_warning_policy_is_unaffected_by_the_pool():
    # A guard against the pool being wired into the warning regime by accident.
    assert WarningPolicy(k_esc=3, n_win=5, n_decay=3, w_cap=2, p_warn=5.0).k_esc == 3
    assert COOPERATIVE_RULES.isdisjoint({"P1", "P6"})
