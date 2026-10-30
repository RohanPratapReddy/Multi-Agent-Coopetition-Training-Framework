"""Selecting, replacing and turning off scoring stages (document 17 sections 6.2, 6.5, 6.8)."""

from __future__ import annotations

import pytest

from mactf import conf
from mactf.components import HUB, Boundary, ComponentKind, Origin, register
from mactf.components.resolver import ComponentResolver
from mactf.components.slots import SlotTable
from mactf.exceptions import (
    ComponentContractError,
    ComponentImportDenied,
    ComponentLocked,
    ComponentParamError,
    ComponentSpecError,
    ImproperlyConfigured,
    ValidationError,
)
from mactf.scoring.base import Scorer, ScoringKind
from mactf.scoring.null_stages import NoWarningScorer, ZeroRewardScorer
from mactf.scoring.penalty import PenaltyScorer
from mactf.scoring.pipeline import (
    STAGE_KINDS,
    ScoringPipeline,
    build_scorer,
    build_stages,
    migrate_stage_state,
)
from mactf.scoring.registry import SCORER_TYPES, build_scorer_from_spec
from mactf.scoring.reward_scorer import RewardScorer
from mactf.scoring.reward_terms import ComposedRewardScorer
from mactf.scoring.warning import WarningPolicy
from mactf.scoring.warning_scorer import WarningScorer

from scoring_helpers import (
    default_pipeline,
    ledger_of,
    penalty_variant,
    settings_copy,
    warning_variant,
)

CONFIG = settings_copy()["SCORING"]


# ------------------------------------------------------------------------- selection
def test_stage_kinds_are_the_three_per_node_kinds_in_chain_order():
    assert STAGE_KINDS == ("warning", "reward", "penalty")


def test_no_specs_build_the_builtins_directly():
    stages = build_stages(score_config=CONFIG)
    assert [type(stage) for stage in stages] == [WarningScorer, RewardScorer, PenaltyScorer]


def test_a_string_is_a_type_and_a_missing_kind_is_the_builtin():
    stages = build_stages(score_config=CONFIG, specs={"reward": "zero_reward"})
    assert [type(stage) for stage in stages] == [WarningScorer, ZeroRewardScorer, PenaltyScorer]


def test_a_mapping_spec_carries_params():
    stages = build_stages(
        score_config=CONFIG,
        specs={"reward": {"type": "composed_reward", "params": {"terms": [{"type": "capped", "params": {"limit": 1.5}}]}}},
    )
    reward = stages[1]
    assert isinstance(reward, ComposedRewardScorer) and reward.terms[0][0] == "capped"
    assert reward.config == CONFIG  # ``config`` was offered by the builder and taken


def test_the_builtin_stages_receive_what_they_declare():
    policy = WarningPolicy(k_esc=4)
    registries: dict = {}
    stages = build_stages(
        score_config=CONFIG, warning_policy=policy, registries=registries, specs={"penalty": "penalty"}
    )
    assert stages[0].policy is policy and stages[0].registries is registries
    assert stages[2].config == CONFIG


def test_stages_are_installed_into_slots_under_scoring_kind():
    slots = SlotTable(ComponentResolver())
    stages = build_stages(score_config=CONFIG, specs={"reward": "zero_reward"}, slots=slots)
    assert slots.paths() == ["scoring.penalty", "scoring.reward", "scoring.warning"]
    assert slots.get("scoring.reward").instance is stages[1]
    assert set(slots.fingerprints()) == set(slots.paths())


def test_unknown_stage_kinds_are_refused_naming_the_valid_ones():
    with pytest.raises(ImproperlyConfigured, match=r"unknown kind\(s\) \['hierarchy'\]"):
        build_stages(score_config=CONFIG, specs={"hierarchy": "hierarchy"})


def test_a_stage_of_the_wrong_kind_is_refused():
    with pytest.raises(ImproperlyConfigured, match="whose kind is 'penalty', not 'reward'"):
        build_stages(score_config=CONFIG, specs={"reward": "penalty"})


def test_a_spec_may_not_set_a_key_the_framework_injects():
    with pytest.raises(ComponentSpecError, match="E011"):
        build_stages(score_config=CONFIG, specs={"penalty": {"type": "penalty", "params": {"config": {}}}})


def test_an_unknown_parameter_is_named_with_the_accepted_ones():
    with pytest.raises(ComponentParamError, match="lamdbas"):
        build_stages(score_config=CONFIG, specs={"penalty": {"type": "penalty", "params": {"lamdbas": 1}}})


def test_an_import_spec_is_refused_on_an_untrusted_surface(component_overlay):
    # a trusted ``import:`` registers the class it names; the overlay keeps that out of the process
    with pytest.raises(ComponentImportDenied):
        build_stages(
            score_config=CONFIG, trusted=False,
            specs={"warning": {"import": "mactf.scoring.null_stages:NoWarningScorer"}},
        )
    stages = build_stages(
        score_config=CONFIG, trusted=True,
        specs={"warning": {"import": "mactf.scoring.null_stages:NoWarningScorer"}},
    )
    assert type(stages[0]) is NoWarningScorer


def test_scorer_from_spec_goes_through_the_registry(component_overlay):
    class Mine(RewardScorer):
        pass

    register(ComponentKind.SCORER, "mine", Mine, scope="run")
    assert type(build_scorer_from_spec({"type": "mine"})) is Mine
    with pytest.raises(ValidationError, match="not a"):
        PenaltyScorer.from_spec({"type": "mine"})
    # the messages the older tests match on survive
    with pytest.raises(ValidationError, match="requires a 'type' key"):
        Scorer.from_spec({})
    with pytest.raises(ValidationError, match="does not accept"):
        Scorer.from_spec({"type": "penalty", "nonexistent": 1})


def test_a_plugin_stage_is_selected_by_name_and_runs_under_the_guard(component_overlay):
    class Tempered(RewardScorer):
        def __init__(self, *, config=None, factor=0.5):
            super().__init__(config=config)
            self.factor = float(factor)

        def score(self, ctx):
            out = super().score(ctx)
            from dataclasses import replace

            return replace(out, amount=out.amount * self.factor,
                           breakdown={**out.breakdown, "r_task": out.breakdown["r_task"] * self.factor})

    register(ComponentKind.SCORER, "tempered", Tempered, scope="run")
    settings = settings_copy(SCORING={"STAGES": {"reward": {"type": "tempered", "params": {"factor": 0.25}}}})
    pipeline = build_scorer(settings)
    assert isinstance(pipeline.get(ScoringKind.REWARD), Tempered)
    card = pipeline.score_peer(ledger_of({"R1": 4.0}), peer="a", episode=1)
    plain = default_pipeline().score_peer(ledger_of({"R1": 4.0}), peer="a", episode=1)
    assert card.continuous.reward.r_task == plain.continuous.reward.r_task * 0.25
    assert pipeline.state_dict()["config"]["STAGES"]["reward"]["params"]["factor"] == 0.25


# ----------------------------------------------------------------------- registration
def test_a_warning_stage_must_subclass_warning_scorer(component_overlay):
    class NotAWarningScorer(Scorer):
        kind = ScoringKind.WARNING
        order = 5

        def score(self, ctx):  # pragma: no cover - never built
            raise AssertionError

    with pytest.raises(ComponentContractError, match="subclasses WarningScorer"):
        register(ComponentKind.SCORER, "bad_warning", NotAWarningScorer, scope="run")

    class Fine(WarningScorer):
        pass

    register(ComponentKind.SCORER, "fine_warning", Fine, scope="run")


def test_a_pipeline_or_unordered_stage_is_not_registrable(component_overlay):
    with pytest.raises(ComponentContractError, match="PIPELINE"):
        register(ComponentKind.SCORER, "nested", ScoringPipeline, scope="run")

    class Unordered(RewardScorer):
        order = -1

    with pytest.raises(ComponentContractError, match="order"):
        register(ComponentKind.SCORER, "unordered", Unordered, scope="run")


def test_the_new_stages_are_builtins_with_their_origin():
    for name in ("composed_reward", "zero_reward", "no_warnings"):
        assert name in SCORER_TYPES
        assert HUB.registry("scorer").get(name).origin is Origin.BUILTIN
    for name in ("severity_scaled", "capped"):
        assert HUB.registry("reward_term").get(name).origin is Origin.BUILTIN
    assert "zero_penalty" not in SCORER_TYPES  # there is deliberately no null penalty stage


def test_replacing_a_new_builtin_is_a_builtin_replacement(component_overlay):
    class Mine(ZeroRewardScorer):
        pass

    with pytest.raises(Exception, match="zero_reward"):
        register(ComponentKind.SCORER, "zero_reward", Mine, scope="run")


# ------------------------------------------------------------------------ null stages
LEDGER = {"R1": 2.0, "R3": 1.0, "P1": 1.0, "P3": 1.0, "P6": 1.0}


def test_zero_reward_pays_nothing_and_leaves_penalties_alone():
    settings = settings_copy(SCORING={"STAGES": {"reward": "zero_reward"}, "PRECEDENCE_MODE": "layered"})
    pipeline = build_scorer(settings)
    plain = default_pipeline(SCORING={"PRECEDENCE_MODE": "layered"})
    card = pipeline.score_peer(ledger_of(LEDGER), peer="a", episode=1)
    base = plain.score_peer(ledger_of(LEDGER), peer="a", episode=1)
    assert card.continuous.reward.total == 0.0 and base.continuous.reward.total > 0.0
    assert card.continuous.penalty.amounts == base.continuous.penalty.amounts
    assert card.violated_non_offsettable == base.violated_non_offsettable == ("P1",)
    assert card.final_score == base.final_score - base.continuous.reward.total
    assert "compliance_guard" not in card.metadata  # zero is a legal reward


def test_zero_reward_keeps_the_weights_so_the_breakdown_renders():
    out = ZeroRewardScorer(config=CONFIG).score(
        type("C", (), {})()  # the stage reads nothing from the context
    )
    assert out.amount == 0.0 and out.breakdown["weights"] == RewardScorer(config=CONFIG).bucket_weights()
    assert out.rules == () and out.breakdown["by_rule"] == {}


def test_no_warnings_scores_every_offence_at_full_penalty_and_keeps_registries_empty():
    settings = settings_copy(SCORING={"STAGES": {"warning": "no_warnings"}})
    pipeline = build_scorer(settings)
    plain = default_pipeline()
    spec = {"P6": 1.0}  # warning-first: the first offence is a warning and scores zero
    card = pipeline.score_peer(ledger_of(spec), peer="a", episode=1)
    base = plain.score_peer(ledger_of(spec), peer="a", episode=1)
    assert base.continuous.penalty.amounts == {} and card.continuous.penalty.amounts == {"P6": 1.0}
    assert card.final_score < base.final_score
    assert pipeline.registries == {} and card.warnings_issued == ()
    assert "compliance_guard" not in card.metadata
    assert isinstance(pipeline.get(ScoringKind.WARNING), NoWarningScorer)


def test_no_warnings_never_makes_a_score_kinder():
    for spec in ({"P6": 1.0}, {"P1": 1.0, "P6": 1.0}, {"P20": 1.0}, {"R1": 1.0}, {}):
        off = build_scorer(settings_copy(SCORING={"STAGES": {"warning": "no_warnings"}}))
        on = default_pipeline()
        assert off.score_peer(ledger_of(spec), peer="a", episode=1).final_score <= \
            on.score_peer(ledger_of(spec), peer="a", episode=1).final_score


def test_a_zero_penalty_stage_cannot_be_written():
    zero = penalty_variant(lambda out, ctx: type(out)(kind=out.kind, amount=0.0, breakdown={
        "amounts": {}, "raw_total": 0.0, "omega": 1.0, "omega_rules": [], "total": 0.0, "rules": []}))
    from scoring_helpers import pipeline_with
    from mactf.exceptions import ComplianceGuardViolation

    with pytest.raises(ComplianceGuardViolation) as excinfo:
        pipeline_with(penalty=zero).score_peer(ledger_of({"P1": 1.0}), peer="a", episode=1)
    assert excinfo.value.code == "C002"
    # ... while a ledger with no non-offsettable breach is not an obstacle to it
    pipeline_with(penalty=zero).score_peer(ledger_of({"P3": 1.0}), peer="a", episode=1)


# ------------------------------------------------------------------ replace and migrate
def _pipeline():
    return default_pipeline()


def test_replace_swaps_the_stage_of_that_kind_and_returns_the_old_one():
    pipeline = _pipeline()
    old = pipeline.get(ScoringKind.REWARD)
    new = ZeroRewardScorer(config=CONFIG)
    assert pipeline.replace(new) is old
    assert pipeline.get(ScoringKind.REWARD) is new
    assert [type(s) for s in pipeline.chain] == [WarningScorer, ZeroRewardScorer, PenaltyScorer]


def test_replace_is_atomic_when_the_chain_would_be_invalid():
    pipeline = _pipeline()
    before = list(pipeline.chain)

    class TooEarly(RewardScorer):
        order = 5  # collides with the warning stage

    with pytest.raises(ImproperlyConfigured, match="both declare order=5"):
        pipeline.replace(TooEarly(config=CONFIG))
    assert list(pipeline.chain) == before and pipeline.get(ScoringKind.REWARD) is before[1]
    with pytest.raises(ImproperlyConfigured, match="Scorer subclass"):
        pipeline.replace(object())  # type: ignore[arg-type]
    assert list(pipeline.chain) == before


def test_replace_is_atomic_when_migrate_fails():
    pipeline = _pipeline()
    before = list(pipeline.chain)

    def refuse(old, new):
        raise RuntimeError("cannot carry state")

    with pytest.raises(RuntimeError):
        pipeline.replace(ZeroRewardScorer(config=CONFIG), migrate=refuse)
    assert list(pipeline.chain) == before


def test_migrate_refuses_a_stage_of_another_kind():
    with pytest.raises(ComponentContractError, match="cannot replace a reward stage with a penalty"):
        migrate_stage_state(RewardScorer(), PenaltyScorer())
    pipeline = _pipeline()
    before = list(pipeline.chain)
    # ``replace`` swaps by the *new* stage's kind, so a penalty stage replaces the penalty
    # stage; handing it the reward stage's migrate hook is the mismatch that must fail
    with pytest.raises(ComponentContractError):
        pipeline.replace(PenaltyScorer(config=CONFIG), migrate=lambda old, new: migrate_stage_state(
            pipeline.get(ScoringKind.REWARD), new))
    assert list(pipeline.chain) == before


def test_swapping_the_warning_stage_keeps_every_peers_warning_state():
    """Invariant I6: no peer's warning count, clean streak or expiry is lost."""
    swapped = default_pipeline()
    steady = default_pipeline()
    spec = {"P6": 1.0}
    for episode in (1, 2):
        for pipeline in (swapped, steady):
            pipeline.score_peer(ledger_of(spec), peer="a", episode=episode)
    old = swapped.get(ScoringKind.WARNING)
    successor = warning_variant(lambda self, ctx: ctx)(
        config=swapped.score_config, policy=swapped.warning_policy
    )
    assert swapped.replace(successor, migrate=migrate_stage_state) is old
    assert successor.registries is old.registries and successor.policy is old.policy
    # the third offence escalates in the swapped pipeline exactly as in the one never swapped
    a = swapped.score_peer(ledger_of(spec), peer="a", episode=3)
    b = steady.score_peer(ledger_of(spec), peer="a", episode=3)
    assert [m.action for m in a.warnings_escalated] == [m.action for m in b.warnings_escalated] == ["escalate"]
    assert a.final_score == b.final_score and a.continuous.warning == b.continuous.warning > 0.0


def test_a_replacement_warning_stage_must_expose_registries():
    class Bare(WarningScorer):
        pass

    new = Bare(config=CONFIG)
    del new.registries
    with pytest.raises(ComponentContractError, match="registries"):
        migrate_stage_state(WarningScorer(config=CONFIG), new)


# --------------------------------------------------------------- the mid-run swap flow
def test_a_queued_swap_lands_at_the_next_episode_boundary_and_not_before():
    slots = SlotTable(ComponentResolver())
    settings = settings_copy(SCORING={"STAGES": {"reward": "reward"}})
    pipeline = build_scorer(settings, slots=slots)
    spec = {"R1": 3.0}
    first = pipeline.score_peer(ledger_of(spec), peer="a", episode=1)
    assert first.continuous.reward.total > 0
    # step 1: the request is validated eagerly ...
    slots.request("scoring.reward", "zero_reward", migrate=migrate_stage_state)
    with pytest.raises(ComponentSpecError):
        slots.request("scoring.reward", {"params": {}, "bogus": 1})
    # ... and changes nothing until the boundary
    assert pipeline.score_peer(ledger_of(spec), peer="a", episode=2).continuous.reward.total > 0
    assert slots.apply_pending(Boundary.STEP) == []  # a step boundary is too early for an episode component
    applied = slots.apply_pending(Boundary.EPISODE, inject=pipeline.stage_inject())
    assert [record.to_name for record in applied] == ["zero_reward"]
    assert pipeline.sync_from_slots(slots) == [ScoringKind.REWARD]
    assert pipeline.score_peer(ledger_of(spec), peer="a", episode=3).continuous.reward.total == 0.0
    assert pipeline.sync_from_slots(slots) == []  # idempotent


def test_a_swap_to_a_stage_of_the_wrong_kind_fails_and_the_run_continues():
    slots = SlotTable(ComponentResolver())
    pipeline = build_scorer(settings_copy(SCORING={"STAGES": {"reward": "reward"}}), slots=slots)
    slots.request("scoring.reward", "penalty", migrate=migrate_stage_state)
    with pytest.raises(Exception):
        slots.apply_pending(Boundary.EPISODE, inject=pipeline.stage_inject())
    assert isinstance(slots.get("scoring.reward").instance, RewardScorer)
    assert pipeline.sync_from_slots(slots) == []
    # without the migrate hook the slot would hold a penalty stage; sync refuses to adopt it
    slots.swap("scoring.reward", "penalty", boundary=Boundary.EPISODE, inject=pipeline.stage_inject())
    before = list(pipeline.chain)
    with pytest.raises(ImproperlyConfigured, match="whose kind is 'penalty', not 'reward'"):
        pipeline.sync_from_slots(slots)
    assert list(pipeline.chain) == before


def test_a_locked_stage_slot_cannot_be_swapped():
    slots = SlotTable(ComponentResolver())
    build_scorer(settings_copy(SCORING={"STAGES": {"reward": "reward"}}), slots=slots)
    from mactf.components import HubPolicy

    saved = HUB.policy
    HUB.configure(HubPolicy(locked=("slot:scoring.reward",)))
    try:
        with pytest.raises(ComponentLocked):
            slots.request("scoring.reward", "zero_reward")
    finally:
        HUB.configure(saved)


def test_stage_inject_offers_what_a_factory_may_take():
    pipeline = default_pipeline()
    offer = pipeline.stage_inject(trusted=False)
    assert set(offer) == {"config", "policy", "trusted"} and offer["trusted"] is False
    assert offer["policy"] is pipeline.warning_policy and offer["config"] == pipeline.score_config


# ------------------------------------------------------------------------------ build_scorer
def test_build_scorer_hands_the_guard_its_policy():
    pipeline = build_scorer(settings_copy(COMPLIANCE={"GUARD_MODE": "fallback", "REWARD_CEILING": 77.0}))
    assert pipeline.guard.policy.mode == "fallback" and pipeline.guard.policy.reward_ceiling == 77.0


def test_w003_is_logged_when_a_run_starts_in_fallback_and_not_otherwise(caplog):
    """6.4.2: ``W003`` is emitted when a run starts in ``fallback``."""
    with caplog.at_level("WARNING"):
        build_scorer(settings_copy())
        assert not [r for r in caplog.records if "W003" in r.getMessage()]
        build_scorer(settings_copy(COMPLIANCE={"GUARD_MODE": "fallback"}))
    assert len([r for r in caplog.records if "W003" in r.getMessage()]) == 1


def test_build_scorer_records_a_non_default_floor_only_when_it_is_not_the_default():
    assert "MIN_FLOOR_LAMBDA" not in build_scorer(settings_copy()).score_config
    assert build_scorer(settings_copy(COMPLIANCE={"MIN_FLOOR_LAMBDA": 2.0})).score_config["MIN_FLOOR_LAMBDA"] == 2.0


def test_explicit_scorers_win_over_stages():
    mine = [WarningScorer(config=CONFIG), ZeroRewardScorer(config=CONFIG), PenaltyScorer(config=CONFIG)]
    pipeline = build_scorer(settings_copy(SCORING={"STAGES": {"reward": "reward"}}), scorers=mine)
    assert type(pipeline.get(ScoringKind.REWARD)) is ZeroRewardScorer


def test_the_default_guard_is_built_lazily_from_the_score_config():
    pipeline = ScoringPipeline(score_config={**CONFIG, "MIN_FLOOR_LAMBDA": 3.0})
    assert pipeline._guard is None
    assert pipeline.guard.policy.min_floor_lambda == 3.0 and pipeline.guard is pipeline.guard
    marker = object()
    pipeline.guard = marker
    assert pipeline.guard is marker
