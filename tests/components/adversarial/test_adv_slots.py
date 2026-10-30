"""Adversarial verification of D2 for slot swaps in the middle of a run (doc 17 section 4.4).

A swap queued while a run is going is applied at the next episode boundary, and the scoring
pipeline then adopts it: the guard inspects the new stage from its first episode. (Regression:
the engine applied the swap to the slot but never to the pipeline, so the slot - and the
checkpoint manifest - named a stage the run was not using.)
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from mactf.components import register_scorer
from mactf.exceptions import ComplianceGuardViolation, ComponentLocked, ComponentMissing
from mactf.orchestration.engine import TrainerEngine
from mactf.scoring.base import ScoringKind
from mactf.components import HUB, HubPolicy

from adv_support import BREACH, HONEST_COST, breach_trainer, victim_cards
from p2_support import configure
from scoring_helpers import penalty_variant, reward_variant


def _zero(out, ctx):
    return replace(out, amount=0.0, void_rules=(), breakdown={**out.breakdown, "amounts": {}})


def _engine(settings, tmp_path, epochs=3):
    configure(settings, SCORING={"STAGES": {"penalty": "penalty", "reward": "reward"}})
    trainer = breach_trainer(settings, tmp_path, epochs=epochs)
    settings["DEVICE"] = "cpu"
    return trainer, TrainerEngine(settings, trainer=trainer)


def test_a_swap_to_a_benign_custom_stage_takes_effect_at_the_boundary(settings, tmp_path, component_overlay):
    class Mine(reward_variant()):
        pass

    register_scorer("adv_mine", Mine, scope="run")
    trainer, engine = _engine(settings, tmp_path)
    trainer.slots.request("scoring.reward", "adv_mine")
    assert engine.run()["status"] == "completed"
    assert type(trainer.scorer.get(ScoringKind.REWARD)).__name__ == "Mine"
    assert trainer.slots.get("scoring.reward").component.name == "adv_mine"


def test_a_swap_to_a_cheap_penalty_stage_is_refused_from_its_first_episode(settings, tmp_path, component_overlay):
    register_scorer("adv_zero_pen", penalty_variant(_zero), scope="run")
    trainer, engine = _engine(settings, tmp_path)
    trainer.slots.request("scoring.penalty", "adv_zero_pen")
    with pytest.raises(ComplianceGuardViolation) as caught:
        engine.run()
    assert caught.value.code in {"C002", "C004"}
    assert all(card.final_score <= HONEST_COST for card in victim_cards(trainer))


def test_a_swap_to_the_wrong_kind_leaves_the_running_stage_alone(settings, tmp_path, component_overlay):
    trainer, engine = _engine(settings, tmp_path)
    trainer.slots.request("scoring.reward", "penalty")           # a PENALTY stage in the reward slot
    before = trainer.scorer.get(ScoringKind.REWARD)
    with pytest.raises(Exception):
        engine.run()
    assert trainer.scorer.get(ScoringKind.REWARD) is before


def test_a_locked_slot_cannot_be_swapped(settings, tmp_path, component_overlay):
    register_scorer("adv_zero_pen", penalty_variant(_zero), scope="run")
    trainer, _ = _engine(settings, tmp_path)
    with HUB.scoped(policy=HubPolicy(locked=("slot:scoring.penalty",))):
        with pytest.raises(ComponentLocked):
            trainer.slots.request("scoring.penalty", "adv_zero_pen")
    with pytest.raises(ComponentMissing):
        trainer.slots.request("scoring.nonexistent", "penalty")
