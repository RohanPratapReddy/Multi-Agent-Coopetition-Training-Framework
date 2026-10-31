"""A scorer stage under a built-in's name cannot cheapen a breach (document 17 sections 6.4-6.6).

The attacks of ``test_adv_scoring`` are registered as ``scorer:penalty``, ``reward`` and
``warning`` with the override flag and reached through ``SCORING.STAGES``. The compliance guard
treats them as it treats a new name: refused with a ``C0xx`` code, or harmless and priced at no
less than the built-in cost, in both guard modes.
"""

from __future__ import annotations

import logging
import math

import pytest

import custom_kinds as ck
from adv_support import assert_breach_cost_intact, breach_trainer, run_epochs, victim_cards
from mactf.exceptions import ComplianceGuardViolation, LedgerReadOnlyError
from override_support import override_scope, put, w001
from p2_support import configure
from scoring_helpers import penalty_variant, reward_variant, warning_variant
from test_adv_scoring import (
    PENALTY_ATTACKS, PREPARE_ATTACKS, REFUSED_OTHERWISE, REWARD_ATTACKS, _Impostor, _codes, _zero,
)


@pytest.fixture(autouse=True)
def _scope(caplog):
    caplog.set_level(logging.WARNING, logger="mactf.components")
    with override_scope():
        yield


def _trainer(settings, tmp_path, stage, cls, mode=None):
    put("scorer", stage, cls)
    blocks = {"COMPONENTS": {"PLUGINS": "off"}, "SCORING": {"STAGES": {stage: stage}}}
    if mode:
        blocks["COMPLIANCE"] = {"GUARD_MODE": mode}
    configure(settings, **blocks)
    return breach_trainer(settings, tmp_path)


# -- the penalty stage ------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["enforce", "fallback"])
@pytest.mark.parametrize("name", sorted(PENALTY_ATTACKS))
def test_a_penalty_override_cannot_cheapen_the_breach(name, mode, settings, tmp_path):
    trainer = _trainer(settings, tmp_path, "penalty", penalty_variant(PENALTY_ATTACKS[name]), mode)
    try:
        run_epochs(trainer)
    except ComplianceGuardViolation as error:
        assert mode == "enforce" or _codes(error) & {"C006", "C007"}, (mode, _codes(error))
        return
    assert_breach_cost_intact(trainer)
    assert all(math.isfinite(card.final_score) for card in victim_cards(trainer))


def test_the_zero_penalty_override_really_is_the_stage_that_ran(settings, tmp_path, caplog):
    """Guards against a vacuous pass: the attack is refused, so the override must have been reached."""
    trainer = _trainer(settings, tmp_path, "penalty", penalty_variant(_zero), "enforce")
    with pytest.raises(ComplianceGuardViolation):
        run_epochs(trainer)
    assert w001(caplog) == ["scorer:penalty"]


def test_a_metaclass_cannot_make_an_override_pass_for_the_builtin(settings, tmp_path):
    evil = _Impostor("Evil", (penalty_variant(_zero),), {})
    trainer = _trainer(settings, tmp_path, "penalty", evil)
    with pytest.raises(ComplianceGuardViolation) as caught:
        run_epochs(trainer)
    assert "C002" in _codes(caught.value)


# -- prepare (context and ledger) -------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["enforce", "fallback"])
@pytest.mark.parametrize("name", sorted(PREPARE_ATTACKS))
def test_a_warning_override_that_attacks_the_context_is_refused(name, mode, settings, tmp_path):
    trainer = _trainer(settings, tmp_path, "warning", warning_variant(PREPARE_ATTACKS[name]), mode)
    try:
        run_epochs(trainer)
    except ComplianceGuardViolation as error:
        assert _codes(error) & {"C005", "C006", "C007"}, _codes(error)
        return
    pytest.fail(f"{name}: a context attack through the built-in's name was not refused")


@pytest.mark.parametrize("name", sorted(REFUSED_OTHERWISE))
def test_a_warning_override_that_fails_loudly_fails_the_run(name, settings, tmp_path):
    fn, error = REFUSED_OTHERWISE[name]
    trainer = _trainer(settings, tmp_path, "warning", warning_variant(fn))
    with pytest.raises(error):
        run_epochs(trainer)


# -- the reward stage ----------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["enforce", "fallback"])
@pytest.mark.parametrize("name", sorted(REWARD_ATTACKS))
def test_a_reward_override_cannot_buy_back_a_breach(name, mode, settings, tmp_path):
    trainer = _trainer(settings, tmp_path, "reward", reward_variant(REWARD_ATTACKS[name]), mode)
    try:
        run_epochs(trainer)
    except ComplianceGuardViolation:
        return
    assert_breach_cost_intact(trainer)


def test_the_largest_legal_reward_cannot_lift_a_void_node_above_its_floor(settings, tmp_path):
    from dataclasses import replace

    def max_reward(out, ctx):
        return replace(out, breakdown={**out.breakdown, "r_task": 9_999.0})

    trainer = _trainer(settings, tmp_path, "reward", reward_variant(max_reward))
    run_epochs(trainer)
    assert_breach_cost_intact(trainer)


def test_an_override_whose_score_raises_fails_the_episode_not_the_cost(settings, tmp_path):
    def boom(out, ctx):
        raise RuntimeError("plugin bug")

    trainer = _trainer(settings, tmp_path, "penalty", penalty_variant(boom))
    with pytest.raises(RuntimeError, match="plugin bug"):
        run_epochs(trainer)
    assert not victim_cards(trainer)


# -- the honest identity overrides ----------------------------------------------------------------------
@pytest.mark.parametrize("stage, cls", [("penalty", ck.SamePenalty), ("reward", ck.SameReward),
                                        ("warning", ck.SameWarning)])
def test_an_identity_override_prices_the_breach_exactly_as_the_builtin_does(stage, cls, settings, tmp_path):
    configure(settings, COMPONENTS={"PLUGINS": "off"})
    reference = breach_trainer(settings, tmp_path / "ref")
    run_epochs(reference)
    trainer = _trainer(settings, tmp_path / "over", stage, cls, "enforce")
    run_epochs(trainer)
    assert_breach_cost_intact(trainer)
    assert [c.final_score for c in victim_cards(trainer)] == [c.final_score for c in victim_cards(reference)]


def test_the_tempered_reward_scorer_never_beats_the_floor(settings, tmp_path):
    put("scorer", "tempered_reward", ck.TemperedRewardScorer)
    configure(settings, COMPONENTS={"PLUGINS": "off"},
              SCORING={"STAGES": {"reward": "tempered_reward"}}, COMPLIANCE={"GUARD_MODE": "enforce"})
    trainer = breach_trainer(settings, tmp_path)
    run_epochs(trainer)
    assert_breach_cost_intact(trainer)
