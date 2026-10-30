"""``WarningScorer``: ordering, idempotence, per-node scoping, serialization."""

from __future__ import annotations

import pytest

from mactf.exceptions import ImproperlyConfigured
from mactf.scoring.base import ScorerOutput, ScoreContext, ScoringKind
from mactf.scoring.ledger import Ledger
from mactf.scoring.warning import WarningPolicy
from mactf.scoring.warning_scorer import DEFAULT_KNOWN_RULES, WarningScorer


def _ctx(ledger, *, peer="a", episode=1, node_path="", severity_weights=None, known=()):
    return ScoreContext(
        peer=peer,
        episode=episode,
        ledger=ledger,
        node_path=node_path,
        severity_weights=severity_weights or {"S0": 0.0, "S1": 0.25, "S2": 1.0, "S3": 4.0, "S4": 25.0},
        known_rules=frozenset(known),
    )


def test_order_is_before_every_other_stage():
    from mactf.scoring.penalty import PenaltyScorer
    from mactf.scoring.reward_scorer import RewardScorer

    assert WarningScorer.order < RewardScorer.order < PenaltyScorer.order
    assert WarningScorer.order == 5


def test_known_rules_default_is_every_penalty_rule():
    from mactf.rules.catalog import ALL_RULES

    assert DEFAULT_KNOWN_RULES == frozenset(rule.id for rule in ALL_RULES if rule.is_penalty)


def test_non_warning_eligible_rules_never_issue():
    ledger = Ledger(peer="a")
    ledger.record("P1", magnitude=1.0, severity="S3")  # never warning-first
    scorer = WarningScorer()
    prepared = scorer.prepare(_ctx(ledger))
    assert prepared.warning_events == ()
    assert scorer.registry_for("a").all_active("a") == {}


def test_first_offence_issues_a_zero_score_warning():
    ledger = Ledger(peer="a")
    ledger.record("P6", magnitude=1.0, severity="S1")
    scorer = WarningScorer()
    prepared = scorer.prepare(_ctx(ledger))
    output = scorer.score(prepared)

    assert prepared.warning_events == ("P6",)
    assert prepared.warning_total == pytest.approx(0.0)
    assert prepared.escalated_rules == frozenset()
    assert [item.action for item in output.mutations] == ["issue"]
    assert output.amount == pytest.approx(0.0)
    assert output.breakdown["issued"] == 1
    assert output.breakdown["escalated"] == 0


def test_escalation_at_k_esc_sets_total_and_rule():
    policy = WarningPolicy(k_esc=3, n_win=5, n_decay=3, w_cap=2, p_warn=5.0)
    scorer = WarningScorer(policy=policy)
    prepared = None
    for episode in (1, 2, 3):
        ledger = Ledger(peer="a")
        ledger.record("P6", magnitude=1.0, severity="S1")
        prepared = scorer.prepare(_ctx(ledger, episode=episode))
    assert prepared is not None
    assert prepared.escalated_rules == frozenset({"P6"})
    assert prepared.warning_total == pytest.approx(5.0)
    # W8: the escalated act is not also a standalone warning.
    assert prepared.warning_only_rules == frozenset()


def test_escalated_rule_is_reported_as_a_mutation():
    policy = WarningPolicy(k_esc=2, n_win=5, n_decay=3, w_cap=1, p_warn=4.0)
    scorer = WarningScorer(policy=policy)
    for episode in (1, 2):
        ledger = Ledger(peer="a")
        ledger.record("P6", magnitude=1.0, severity="S1")
        prepared = scorer.prepare(_ctx(ledger, episode=episode))
    output = scorer.score(prepared)
    escalate = [item for item in output.mutations if item.action == "escalate"]
    assert len(escalate) == 1
    assert escalate[0].penalty == pytest.approx(4.0)
    assert output.breakdown["escalated"] == 1


def test_prepare_is_idempotent_per_episode():
    ledger = Ledger(peer="a")
    ledger.record("P6", magnitude=1.0, severity="S1")
    scorer = WarningScorer(policy=WarningPolicy(k_esc=2, w_cap=1))
    first = scorer.prepare(_ctx(ledger))
    second = scorer.prepare(_ctx(ledger))
    assert first is second
    assert scorer.registry_for("a").count("a", "P6") == 1


def test_prepare_is_idempotent_across_repeated_score_peer_calls():
    # A caller re-scoring one episode after an edit must not double-issue.
    scorer = WarningScorer(policy=WarningPolicy(k_esc=2, w_cap=1))
    ledger = Ledger(peer="a")
    ledger.record("P6", magnitude=1.0, severity="S1")
    scorer.prepare(_ctx(ledger, episode=1))
    scorer.prepare(_ctx(ledger, episode=1))
    assert scorer.registry_for("a").count("a", "P6") == 1


def test_reset_episode_cache_lets_the_regime_run_again():
    scorer = WarningScorer(policy=WarningPolicy(k_esc=5, w_cap=4))
    ledger = Ledger(peer="a")
    ledger.record("P6", magnitude=1.0, severity="S1")
    scorer.prepare(_ctx(ledger))
    scorer.reset_episode_cache()
    scorer.prepare(_ctx(ledger))
    assert scorer.registry_for("a").count("a", "P6") == 2


def test_score_does_not_advance_the_registry():
    ledger = Ledger(peer="a")
    ledger.record("P6", magnitude=1.0, severity="S1")
    scorer = WarningScorer(policy=WarningPolicy(k_esc=3, w_cap=2))
    prepared = scorer.prepare(_ctx(ledger))
    scorer.score(prepared)
    scorer.score(prepared)
    assert scorer.registry_for("a").count("a", "P6") == 1


def test_registries_are_isolated_per_peer():
    scorer = WarningScorer(policy=WarningPolicy(k_esc=3, w_cap=2))
    ledger_a = Ledger(peer="a")
    ledger_a.record("P6", magnitude=1.0, severity="S1")
    ledger_b = Ledger(peer="b")
    ledger_b.record("P6", magnitude=1.0, severity="S1")
    scorer.prepare(_ctx(ledger_a, peer="a"))
    scorer.prepare(_ctx(ledger_b, peer="b"))
    assert scorer.registry_for("a").count("a", "P6") == 1
    assert scorer.registry_for("b").count("b", "P6") == 1


def test_per_node_scoping_keeps_child_warnings_away_from_the_root():
    scorer = WarningScorer(policy=WarningPolicy(k_esc=2, w_cap=1))
    root = Ledger(peer="a")
    root.record("P6", magnitude=1.0, severity="S1")
    child = Ledger(peer="a")
    child.record("P6", magnitude=1.0, severity="S1")

    scorer.prepare(_ctx(child, node_path="root/0"))
    root_ctx = scorer.prepare(_ctx(root, node_path=""))
    assert root_ctx.warning_events == ("P6",)
    assert root_ctx.escalated_rules == frozenset()
    assert scorer.registry_for_node("root/0", "a").count("a", "P6") == 1


def test_flat_mode_key_is_the_peer_alone():
    scorer = WarningScorer()
    scorer.prepare(_ctx(Ledger(peer="a")))
    assert set(scorer.registries) == {"a"}


def test_clean_streak_advances_for_unviolated_rules():
    ledger = Ledger(peer="a")
    scorer = WarningScorer(policy=WarningPolicy(k_esc=3, n_decay=2, w_cap=2))
    prepared = scorer.prepare(_ctx(ledger, known={"P6", "P7"}))
    assert prepared.warning_events == ()
    registry = scorer.registry_for("a")
    assert registry.snapshot("a")["clean_streak"]["P6"] == 1


def test_expiry_after_decay_window():
    policy = WarningPolicy(k_esc=5, n_win=2, n_decay=2, w_cap=2)
    scorer = WarningScorer(policy=policy)
    ledger = Ledger(peer="a")
    ledger.record("P6", magnitude=1.0, severity="S1")
    scorer.prepare(_ctx(ledger, episode=1))

    # begin_episode runs before the clean counters advance, so a warning expires
    # at the start of the episode *after* N_DECAY clean episodes have completed.
    # That is the pre-split behaviour and is preserved deliberately.
    scorer.prepare(_ctx(Ledger(peer="a"), episode=2, known={"P6"}))
    scorer.prepare(_ctx(Ledger(peer="a"), episode=3, known={"P6"}))
    assert len(scorer.registry_for("a").active("a", "P6")) == 1

    scorer.prepare(_ctx(Ledger(peer="a"), episode=4, known={"P6"}))
    assert scorer.registry_for("a").active("a", "P6") == []


def test_state_dict_round_trips_every_registry():
    scorer = WarningScorer(policy=WarningPolicy(k_esc=3, w_cap=2))
    for node_path in ("", "root/0"):
        ledger = Ledger(peer="a")
        ledger.record("P6", magnitude=1.0, severity="S1")
        scorer.prepare(_ctx(ledger, node_path=node_path))

    payload = scorer.state_dict()
    assert set(payload["registries"]) == {"", "root/0"}
    assert payload["registries"][""]["a"]["warnings"]["a"]["P6"][0]["rule_id"] == "P6"

    restored = WarningScorer()
    restored.load_state_dict(payload)
    assert restored.policy.k_esc == 3
    assert restored.registry_for_node("", "a").count("a", "P6") == 1
    assert restored.registry_for_node("root/0", "a").count("a", "P6") == 1
    assert restored.state_dict()["registries"] == payload["registries"]


def test_state_dict_round_trip_preserves_the_escalation_countdown():
    scorer = WarningScorer(policy=WarningPolicy(k_esc=2, w_cap=1))
    for episode in (1, 2):
        ledger = Ledger(peer="a")
        ledger.record("P6", magnitude=1.0, severity="S1")
        scorer.prepare(_ctx(ledger, episode=episode))

    restored = WarningScorer()
    restored.load_state_dict(scorer.state_dict())
    ledger = Ledger(peer="a")
    ledger.record("P6", magnitude=1.0, severity="S1")
    prepared = restored.prepare(_ctx(ledger, episode=3))
    assert "P6" in prepared.escalated_rules


def test_load_state_dict_rejects_a_foreign_class():
    with pytest.raises(ValueError, match="cannot load state"):
        WarningScorer().load_state_dict({"class": "RewardScorer"})


def test_output_kind_and_rules():
    ledger = Ledger(peer="a")
    ledger.record("P6", magnitude=1.0, severity="S1")
    scorer = WarningScorer()
    output = scorer.score(scorer.prepare(_ctx(ledger)))
    assert isinstance(output, ScorerOutput)
    assert output.kind is ScoringKind.WARNING
    assert output.rules == ("P6",)
    assert output.metadata["warning_events"] == ("P6",)


def test_validate_accepts_a_healthy_policy():
    assert WarningScorer().validate({"WARNINGS": {}}) == []


def test_validate_reports_k_esc_too_small():
    diagnostics = WarningScorer().validate({"WARNINGS": {"K_ESC": 1}})
    codes = [item.code for item in diagnostics]
    assert "warnings.k_esc_too_small" in codes


def test_validate_reports_w_cap_out_of_range():
    diagnostics = WarningScorer().validate({"WARNINGS": {"K_ESC": 3, "W_CAP": 9}})
    assert [item.code for item in diagnostics] == ["warnings.w_cap_out_of_range"]


def test_validate_reports_a_zero_window():
    diagnostics = WarningScorer().validate({"WARNINGS": {"N_WIN": 0}})
    assert [item.code for item in diagnostics] == ["warnings.window_too_small"]


def test_validate_reports_a_negative_p_warn():
    diagnostics = WarningScorer().validate({"WARNINGS": {"P_WARN": -1.0}})
    assert [item.code for item in diagnostics] == ["warnings.negative_penalty"]


def test_validate_reports_a_non_integer_policy_value():
    diagnostics = WarningScorer().validate({"WARNINGS": {"K_ESC": "three"}})
    assert [item.code for item in diagnostics] == ["warnings.invalid_policy"]
    assert "K_ESC" in diagnostics[0].message


def test_validate_reports_every_problem_at_once():
    diagnostics = WarningScorer().validate(
        {"WARNINGS": {"K_ESC": 1, "W_CAP": 9, "N_WIN": 0, "P_WARN": -2.0}}
    )
    codes = sorted(item.code for item in diagnostics)
    assert codes == [
        "warnings.k_esc_too_small",
        "warnings.negative_penalty",
        "warnings.w_cap_out_of_range",
        "warnings.window_too_small",
    ]


def test_assert_valid_names_the_problem():
    with pytest.raises(ImproperlyConfigured, match="warnings.k_esc_too_small"):
        WarningScorer().assert_valid({"WARNINGS": {"K_ESC": 0}})
