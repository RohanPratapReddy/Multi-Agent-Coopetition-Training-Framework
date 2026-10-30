"""Adversarial verification of D2 for ``reward_term`` components (doc 17 section 6.3).

A reward term is the smallest scoring customisation: it returns a number per reward event. It is
attacked inside a real trainer with a breach by ``peer-1``: whatever it returns, mutates or
raises, the breach keeps its cost and the node stays void - or the episode is refused.
"""

from __future__ import annotations

import pytest

from mactf.components import register_reward_term
from mactf.exceptions import ComplianceError, ComplianceGuardViolation
from mactf.scoring.reward_terms import RewardTerm

from adv_support import assert_breach_cost_intact, breach_trainer, run_epochs
from p2_support import configure

REFUSED = (ComplianceError, ComplianceGuardViolation)


class _Term(RewardTerm):
    name = "adv"

    calls = 0

    def contribution(self, event, ctx):
        _Term.calls += 1
        return self.attack(event, ctx)


def _make(name, attack):
    return type(f"Term_{name}", (_Term,), {"name": name, "attack": staticmethod(attack)})


def _zero_the_ledger(event, ctx):
    ctx.ledger.unwrap()._events.clear()
    return 1.0

def _shrink_breach(event, ctx):
    for other in ctx.ledger.unwrap().events:
        object.__setattr__(other, "magnitude", 0.0)
    return 1.0

def _edit_config(event, ctx):
    ctx.config["NON_OFFSETTABLE"] = []
    ctx.config["PENALTY_LAMBDAS"] = {"P1": 0.0}
    ctx.severity_weights.clear()
    return 1.0

ATTACKS = {
    "nan": lambda e, c: float("nan"),
    "inf": lambda e, c: float("inf"),
    "negative": lambda e, c: -5.0,
    "huge": lambda e, c: 1e300,
    "raises": lambda e, c: (_ for _ in ()).throw(RuntimeError("plugin bug")),
    "not_a_number": lambda e, c: "lots",
    "none": lambda e, c: None,
    "zero_the_ledger": _zero_the_ledger,
    "shrink_breach": _shrink_breach,
    "edit_config": _edit_config,
}


@pytest.mark.parametrize("name", sorted(ATTACKS))
def test_a_malicious_reward_term_cannot_touch_the_breach(name, settings, tmp_path, component_overlay):
    register_reward_term(f"adv_{name}", _make(f"adv_{name}", ATTACKS[name]), scope="run")
    configure(settings, SCORING={"STAGES": {"reward": {
        "type": "composed_reward", "terms": [{"type": f"adv_{name}", "weight": 1.0}]}}})
    trainer = breach_trainer(settings, tmp_path)
    _Term.calls = 0
    try:
        run_epochs(trainer)
    except (*REFUSED, RuntimeError, TypeError, ValueError):
        # refused loudly (a code, or the term's own exception): nothing was scored cheaply
        assert _Term.calls, "the attack never ran"
        assert not [c for c in trainer.scorecards if "peer-1" in c.peers and not c.peers["peer-1"].is_void]
        return
    assert _Term.calls, "the attack never ran"
    assert_breach_cost_intact(trainer)


def test_a_huge_but_finite_term_is_held_by_the_ceiling(settings, tmp_path, component_overlay):
    register_reward_term("adv_big", _make("adv_big", lambda e, c: 1e12), scope="run")
    configure(settings, SCORING={"STAGES": {"reward": {
        "type": "composed_reward", "terms": [{"type": "adv_big", "weight": 1e6}]}}},
        COMPLIANCE={"GUARD_MODE": "fallback"})
    trainer = breach_trainer(settings, tmp_path)
    run_epochs(trainer)
    assert_breach_cost_intact(trainer)
    assert all(c.peers["peer-1"].continuous.reward.total <= 1e4 + 1e-6 for c in trainer.scorecards)
