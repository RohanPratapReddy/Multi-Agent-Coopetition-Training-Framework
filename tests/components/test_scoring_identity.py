"""Invariant I1 for scoring (document 17 section 6.5): a default pipeline is the old pipeline.

``test_default_pipeline_is_byte_identical`` runs the pipeline built from default settings and
the pre-split algorithm (reproduced in ``tests/scoring/test_pipeline_equivalence.py``) over
200 random episodes and compares every number with ``torch.equal``, not ``allclose``.
"""

from __future__ import annotations

import copy
import random

import pytest

torch = pytest.importorskip("torch")

from mactf import conf
from mactf.compliance import constrain, void_rule_ids
from mactf.components.guards.compliance import ComplianceGuard
from mactf.rules.catalog import ALL_RULES
from mactf.scoring.continuous import score as continuous_score
from mactf.scoring.ledger import Ledger
from mactf.scoring.pipeline import ScoringPipeline, build_scorer, build_stages
from mactf.scoring.warning import WarningPolicy, WarningRegistry

KNOWN = {rule.id for rule in ALL_RULES if rule.is_penalty}
SEVERITIES = (None, "S0", "S1", "S2", "S3", "S4")


def _oracle(*, config, weights, registries, policy, ledger, peer, episode):
    """The pre-split ``Scorer.score_peer`` (commit 7634161), as in test_pipeline_equivalence."""
    ledger.peer = peer
    registry = registries.setdefault(peer, WarningRegistry(policy))
    registry.begin_episode(peer, episode=episode)
    issued, escalated, warning_penalty = [], [], 0.0
    warning_events, escalated_rules = [], set()
    for event in ledger.penalties():
        if not event.rule.warning_eligible:
            continue
        outcome = registry.issue(
            peer, event.rule_id, severity=event.resolved_severity(), episode=episode,
            evidence=event.evidence, weights=weights,
        )
        issued.extend(outcome.issued)
        warning_events.append(outcome.rule_id)
        if outcome.escalated is not None:
            escalated.append(outcome.escalated)
            warning_penalty += outcome.penalty
            escalated_rules.add(outcome.rule_id)
    registry.mark_clean_except(peer, set(warning_events), known_rules=KNOWN)
    continuous = continuous_score(
        ledger, score_config=config, severity_weights=weights, warning_total=warning_penalty,
        warning_events=tuple(warning_events), escalated_rules=escalated_rules,
    )
    non_offsettable = set(config.get("NON_OFFSETTABLE", ()))
    result = constrain(continuous, non_offsettable=non_offsettable)
    return continuous, result, void_rule_ids(continuous, non_offsettable=non_offsettable), (
        [(r.rule_id, r.episode) for r in issued], [(r.rule_id, r.episode) for r in escalated]
    )


def _random_ledger(rng: random.Random, peer: str) -> dict:
    ids = [rule.id for rule in ALL_RULES]
    chosen = rng.sample(ids, rng.randint(0, 5))
    return {rule_id: (round(rng.uniform(0.1, 4.0), 3), rng.choice(SEVERITIES)) for rule_id in chosen}


def _record(spec: dict, peer: str) -> Ledger:
    ledger = Ledger(peer=peer)
    for rule_id, (magnitude, severity) in spec.items():
        rule_is_reward = [r for r in ALL_RULES if r.id == rule_id][0].is_reward
        ledger.record(
            rule_id, magnitude=magnitude,
            severity=None if (severity == "S0" and rule_is_reward) else severity,
        )
    return ledger


def _vector(continuous, result) -> "torch.Tensor":
    return torch.tensor(
        [
            result.value, result.floor, continuous.net, continuous.reward.total,
            continuous.penalty.total, continuous.warning, result.omega,
            result.offsettable_penalty, result.non_offsettable_penalty,
        ],
        dtype=torch.float64,
    )


def test_default_pipeline_is_byte_identical():
    settings = copy.deepcopy(conf.DEFAULTS)
    config = dict(settings["SCORING"])
    weights = dict(settings["SEVERITY"]["WEIGHTS"])
    policy = WarningPolicy.from_settings(settings["WARNINGS"])
    pipeline = ScoringPipeline(
        score_config=copy.deepcopy(config), severity_weights=dict(weights), warning_policy=policy
    )
    registries: dict = {}
    rng = random.Random(20260817)
    compared = 0
    for episode in range(1, 201):
        peer = f"p{episode % 3}"
        spec = _random_ledger(rng, peer)
        card = pipeline.score_peer(_record(spec, peer), peer=peer, episode=episode)
        continuous, result, void_ids, (issued, escalated) = _oracle(
            config=copy.deepcopy(config), weights=dict(weights), registries=registries,
            policy=policy, ledger=_record(spec, peer), peer=peer, episode=episode,
        )
        assert torch.equal(_vector(card.continuous, constrain(
            card.continuous, non_offsettable=set(config["NON_OFFSETTABLE"]))), _vector(continuous, result))
        assert torch.equal(
            torch.tensor([card.final_score, card.floor], dtype=torch.float64),
            torch.tensor([result.value, result.floor], dtype=torch.float64),
        )
        assert card.violated_non_offsettable == void_ids
        assert [(r.rule_id, r.episode) for r in card.warnings_issued] == issued
        assert [(r.rule_id, r.episode) for r in card.warnings_escalated] == escalated
        # nothing new reached the card: the guard is silent for built-ins
        assert set(card.metadata) == {"compliance"}
        assert card.metadata["compliance"] == result.as_dict()
        compared += 1
    assert compared == 200


def test_default_runs_never_inspect_anything(monkeypatch):
    """I1 in code: built-in stages do not go through a guard check that could alter them."""
    calls: list[str] = []
    for name in ("reference_penalty", "_inspect_penalty", "_inspect_reward", "_inspect_warning"):
        original = getattr(ComplianceGuard, name)

        def spy(self, *args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(ComplianceGuard, name, spy)
    pipeline = build_scorer(copy.deepcopy(conf.DEFAULTS))
    for episode in range(1, 8):
        pipeline.score_peer(_record({"P1": (1.0, None), "R1": (2.0, None), "P6": (1.0, "S1")}, "a"),
                            peer="a", episode=episode)
    assert calls == []


def test_a_default_pipeline_leaves_state_and_stages_as_before():
    settings = copy.deepcopy(conf.DEFAULTS)
    pipeline = build_scorer(settings)
    state = pipeline.state_dict()
    assert "STAGES" not in state["config"] and "MIN_FLOOR_LAMBDA" not in state["config"]
    assert [type(stage).__name__ for stage in pipeline.chain] == [
        "WarningScorer", "RewardScorer", "PenaltyScorer",
    ]
    assert pipeline.state_dict() == ScoringPipeline(
        score_config=dict(settings["SCORING"]), severity_weights=dict(settings["SEVERITY"]["WEIGHTS"]),
        warning_policy=WarningPolicy.from_settings(settings["WARNINGS"]),
    ).state_dict()


def test_empty_stages_build_the_same_objects_as_no_stages():
    config = dict(conf.DEFAULTS["SCORING"])
    plain = build_stages(score_config=config)
    empty = build_stages(score_config=config, specs={})
    assert [type(s) for s in plain] == [type(s) for s in empty]
    assert [s.config for s in plain] == [s.config for s in empty]
