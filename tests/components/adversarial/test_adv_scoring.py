"""Independent adversarial verification of D2 for scoring stages (doc 17 sections 6.4-6.6).

Every attack is a *custom scorer stage* registered the documented way and run inside a real
``MACTFTrainer`` that is made to see a non-offsettable breach (``adv_support``). The claim under
test: whatever the stage does, the breach is either refused with a documented code
(``ComplianceGuardViolation`` / ``C0xx``) or the node is still void, still scored at no better
than the built-in cost, still carries zero weight - in both ``GUARD_MODE`` values.
"""

from __future__ import annotations

import math
import random
from abc import ABCMeta
from dataclasses import replace

import pytest

from mactf.components import register_scorer
from mactf.exceptions import ComplianceGuardViolation, LedgerReadOnlyError
from mactf.scoring.base import ScorerOutput
from mactf.scoring.penalty import PenaltyScorer
from mactf.scoring.reward_scorer import RewardScorer
from mactf.scoring.warning_scorer import WarningScorer

from p2_support import configure
from adv_support import (
    BREACH, HONEST_COST, assert_breach_cost_intact, breach_trainer, run_epochs, victim_cards,
)
from scoring_helpers import (
    ledger_of, penalty_variant, pipeline_with, reward_variant, stage_variant, warning_variant,
)


def _codes(error: ComplianceGuardViolation) -> set[str]:
    return {item["code"] for item in error.findings}


class _FloatLiar(float):
    """Its value is 0.0 (what arithmetic sees); ``float(x)`` - what a naive check reads - is 5.0."""

    def __float__(self) -> float:
        return 5.0


class _EqLiarDict(dict):
    def __eq__(self, other):  # noqa: D105 - claims to equal anything
        return True

    __hash__ = None


class _EqLiarSet(frozenset):
    def __eq__(self, other):  # noqa: D105
        return True

    __hash__ = frozenset.__hash__


def _amounts(out):
    return dict(out.breakdown["amounts"])


# -- penalty stages: output shape attacks -----------------------------------------------------
def _zero(out, ctx):
    return replace(out, amount=0.0, breakdown={**out.breakdown, "amounts": {}, "omega": 1.0}, void_rules=())

def _tenth(out, ctx):
    return replace(out, amount=out.amount / 10, breakdown={
        **out.breakdown, "amounts": {k: v / 10 for k, v in _amounts(out).items()}})

def _erase_void(out, ctx):
    return replace(out, void_rules=())

def _omega_half(out, ctx):
    return replace(out, breakdown={**out.breakdown, "omega": 0.5})

def _omega_nan(out, ctx):
    return replace(out, breakdown={**out.breakdown, "omega": float("nan")})

def _negative(out, ctx):
    return replace(out, amount=-9.0, breakdown={**out.breakdown, "amounts": {k: -9.0 for k in _amounts(out)}})

def _nan(out, ctx):
    return replace(out, amount=float("nan"), breakdown={**out.breakdown, "amounts": {k: float("nan") for k in _amounts(out)}})

def _inf_negative(out, ctx):
    return replace(out, breakdown={**out.breakdown, "amounts": {k: float("-inf") for k in _amounts(out)}})

def _bool_amount(out, ctx):
    return replace(out, breakdown={**out.breakdown, "amounts": {k: False for k in _amounts(out)}})

def _string_amount(out, ctx):
    return replace(out, breakdown={**out.breakdown, "amounts": {k: "5.0" for k in _amounts(out)}})

def _none_amounts(out, ctx):
    return replace(out, breakdown={**out.breakdown, "amounts": {k: None for k in _amounts(out)}})

def _not_a_mapping(out, ctx):
    return replace(out, breakdown=[("amounts", 1)])

def _ghost_rule(out, ctx):
    return replace(out, breakdown={**out.breakdown, "amounts": {**_amounts(out), "P99": 1.0}})

def _mutations(out, ctx):
    return replace(out, mutations=("forgive",))

def _float_liar(out, ctx):
    return replace(out, breakdown={**out.breakdown, "amounts": {k: _FloatLiar(0.0) for k in _amounts(out)}})

def _lazy_void(out, ctx):
    class Lazy:                      # an iterator that is empty the second time it is read
        used = False

        def __iter__(self):
            if Lazy.used:
                return iter(())
            Lazy.used = True
            return iter(out.void_rules)

    return replace(out, void_rules=Lazy())

def _mapping_flip(out, ctx):
    class Flip(dict):
        reads = 0

        def items(self):
            Flip.reads += 1
            return dict.items(self) if Flip.reads == 1 else iter(())

    return replace(out, breakdown={**out.breakdown, "amounts": Flip(_amounts(out))})

def _random_cost(out, ctx):
    rng = random.Random(ctx.episode)           # deterministic, but sometimes below the floor
    scale = rng.choice([0.0, 0.2, 0.9])
    return replace(out, breakdown={**out.breakdown, "amounts": {k: v * scale for k, v in _amounts(out).items()}})


PENALTY_ATTACKS = {
    "zero": _zero, "tenth": _tenth, "erase_void": _erase_void, "omega_half": _omega_half,
    "omega_nan": _omega_nan, "negative": _negative, "nan": _nan, "inf_negative": _inf_negative,
    "bool": _bool_amount, "string": _string_amount, "none": _none_amounts,
    "not_a_mapping": _not_a_mapping, "ghost_rule": _ghost_rule, "mutations": _mutations,
    "float_liar": _float_liar, "lazy_void": _lazy_void, "mapping_flip": _mapping_flip,
    "random_cost": _random_cost,
}


@pytest.mark.parametrize("mode", ["enforce", "fallback"])
@pytest.mark.parametrize("name", sorted(PENALTY_ATTACKS))
def test_a_malicious_penalty_stage_cannot_cheapen_the_breach(name, mode, settings, tmp_path, component_overlay):
    register_scorer(f"adv_{name}", penalty_variant(PENALTY_ATTACKS[name]), scope="run")
    configure(settings, SCORING={"STAGES": {"penalty": f"adv_{name}"}}, COMPLIANCE={"GUARD_MODE": mode})
    trainer = breach_trainer(settings, tmp_path)
    try:
        run_epochs(trainer)
    except ComplianceGuardViolation as error:
        assert mode == "enforce" or _codes(error) & {"C006", "C007"}, (mode, _codes(error))
        return
    # not refused: it must then be harmless (substituted by the reference, or read exactly once)
    assert_breach_cost_intact(trainer)
    assert all(math.isfinite(card.final_score) for card in victim_cards(trainer))


# -- penalty stage impersonating a built-in ----------------------------------------------------
class _Impostor(ABCMeta):
    """A metaclass whose classes compare equal to (and hash like) every built-in stage."""

    def __eq__(cls, other):
        return True

    def __hash__(cls):
        return hash(PenaltyScorer)


def test_a_metaclass_cannot_make_a_stage_pass_for_a_builtin(settings, tmp_path, component_overlay):
    """Regression: ``type(stage) in (PenaltyScorer, ...)`` called the class's ``__eq__``."""
    evil = _Impostor("Evil", (penalty_variant(_zero),), {})
    register_scorer("adv_impostor", evil, scope="run")
    configure(settings, SCORING={"STAGES": {"penalty": "adv_impostor"}})
    trainer = breach_trainer(settings, tmp_path)
    with pytest.raises(ComplianceGuardViolation) as caught:
        run_epochs(trainer)
    assert "C002" in _codes(caught.value)


def test_an_impostor_stage_is_inspected_in_a_bare_pipeline():
    pipeline = pipeline_with(penalty=_Impostor("Evil", (penalty_variant(_zero),), {}))
    with pytest.raises(ComplianceGuardViolation):
        pipeline.score_peer(ledger_of({BREACH: 1.0}), peer="a")


# -- context and ledger attacks (prepare) ------------------------------------------------------
class _SpoofedLedger:
    """Claims to be the guard's read-only view; holds none of the breach."""

    def __init__(self, view):
        from mactf.scoring.ledger import Ledger

        self._inner = Ledger(peer=view.peer)
        self._src = view.unwrap()
        self.peer = view.peer

    @property
    def __class__(self):
        from mactf.components.guards.compliance import ReadOnlyLedger

        return ReadOnlyLedger

    def unwrap(self):
        return self._src

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _swap_ledger(self, ctx):
    return ctx.evolve(ledger=_SpoofedLedger(ctx.ledger))

def _event_in_place(self, ctx):
    for event in ctx.ledger.unwrap().events:
        object.__setattr__(event, "magnitude", 0.0)
    return ctx

def _event_replaced(self, ctx):
    class Same(type(ctx.ledger.events[0])):
        def __eq__(self, other):
            return True

        __hash__ = object.__hash__

    ctx.ledger.unwrap()._events[:] = [Same("P2", 0.0)]
    return ctx

def _events_list_replaced(self, ctx):
    ctx.ledger.unwrap()._events = []
    return ctx

def _record_through_view(self, ctx):
    ctx.ledger.record("R1", magnitude=1.0)
    return ctx

def _record_through_source(self, ctx):
    ctx.ledger.unwrap().record("R1", magnitude=1.0)
    return ctx

def _clear_via_events(self, ctx):
    ctx.ledger._events.clear()
    return ctx

def _config_in_place(self, ctx):
    ctx.config["NON_OFFSETTABLE"] = []
    ctx.config["PENALTY_LAMBDAS"] = {BREACH: 0.0}
    ctx.config["MIN_FLOOR_LAMBDA"] = 0.0
    return ctx

def _config_liar(self, ctx):
    cfg = _EqLiarDict(ctx.config)
    cfg["NON_OFFSETTABLE"] = []
    return ctx.evolve(config=cfg)

def _config_nested_in_place(self, ctx):
    ctx.config.get("PENALTY_LAMBDAS", {}).update({BREACH: 0.0})
    ctx.config.setdefault("PENALTY_LAMBDAS", {})[BREACH] = 0.0
    return ctx

def _severity_in_place(self, ctx):
    ctx.severity_weights.clear()
    return ctx

def _branch_void_liar(self, ctx):
    return ctx.evolve(branch_void=_EqLiarSet())

def _known_rules_liar(self, ctx):
    return ctx.evolve(known_rules=_EqLiarSet())

def _hide_in_warning_only(self, ctx):
    return ctx.evolve(warning_events=(BREACH,), escalated_rules=frozenset())

def _allowed_rules(self, ctx):
    return ctx.evolve(extras={**ctx.extras, "allowed_rules": frozenset()})

def _precedence(self, ctx):
    return ctx.evolve(extras={**ctx.extras, "precedence": None})

def _peer_rename(self, ctx):
    return ctx.evolve(peer="somebody-else")

def _wrong_return(self, ctx):
    return None

PREPARE_ATTACKS = {
    "swap_ledger": _swap_ledger, "event_in_place": _event_in_place, "event_replaced": _event_replaced,
    "events_list_replaced": _events_list_replaced, "record_through_source": _record_through_source,
    "clear_via_events": _clear_via_events, "config_in_place": _config_in_place,
    "config_liar": _config_liar, "config_nested": _config_nested_in_place,
    "severity_in_place": _severity_in_place, "branch_void_liar": _branch_void_liar,
    "known_rules_liar": _known_rules_liar, "hide_in_warning_only": _hide_in_warning_only,
    "allowed_rules": _allowed_rules, "precedence": _precedence, "peer_rename": _peer_rename,
}
#: These end in an exception that is not the guard's own, but the run is still refused.
REFUSED_OTHERWISE = {"record_through_view": (_record_through_view, LedgerReadOnlyError),
                     "wrong_return": (_wrong_return, Exception)}


@pytest.mark.parametrize("mode", ["enforce", "fallback"])
@pytest.mark.parametrize("name", sorted(PREPARE_ATTACKS))
def test_a_malicious_prepare_is_refused_with_a_code(name, mode, settings, tmp_path, component_overlay):
    register_scorer(f"adv_{name}", warning_variant(PREPARE_ATTACKS[name]), scope="run")
    configure(settings, SCORING={"STAGES": {"warning": f"adv_{name}"}}, COMPLIANCE={"GUARD_MODE": mode})
    trainer = breach_trainer(settings, tmp_path)
    try:
        run_epochs(trainer)
    except ComplianceGuardViolation as error:
        assert _codes(error) & {"C005", "C006", "C007"}, _codes(error)
        return
    pytest.fail(f"{name}: a context attack was not refused")


@pytest.mark.parametrize("name", sorted(REFUSED_OTHERWISE))
def test_a_prepare_that_fails_loudly_fails_the_run(name, settings, tmp_path, component_overlay):
    fn, error = REFUSED_OTHERWISE[name]
    register_scorer(f"adv_{name}", warning_variant(fn), scope="run")
    configure(settings, SCORING={"STAGES": {"warning": f"adv_{name}"}})
    trainer = breach_trainer(settings, tmp_path)
    with pytest.raises(error):
        run_epochs(trainer)


@pytest.mark.parametrize("name", ["event_in_place", "event_replaced", "events_list_replaced",
                                  "record_through_source", "clear_via_events"])
def test_a_tampered_ledger_is_restored_exactly(name):
    ledger = ledger_of({BREACH: 1.0, "R1": 2.0})
    before = [(id(e), e.rule_id, e.magnitude) for e in ledger.events]
    container = ledger._events
    pipeline = pipeline_with(warning=warning_variant(PREPARE_ATTACKS[name]))
    with pytest.raises(ComplianceGuardViolation) as caught:
        pipeline.score_peer(ledger, peer="a")
    assert "C007" in _codes(caught.value)
    assert [(id(e), e.rule_id, e.magnitude) for e in ledger.events] == before
    assert ledger._events is container


def test_a_stage_that_tampers_and_then_raises_leaves_nothing_behind():
    def tamper_then_raise(self, ctx):
        object.__setattr__(ctx.ledger.unwrap().events[0], "magnitude", 0.0)
        raise RuntimeError("boom")

    ledger = ledger_of({BREACH: 1.0})
    with pytest.raises(ComplianceGuardViolation) as caught:
        pipeline_with(warning=warning_variant(tamper_then_raise)).score_peer(ledger, peer="a")
    assert "C007" in _codes(caught.value) and isinstance(caught.value.__cause__, RuntimeError)
    assert ledger.events[0].magnitude == 1.0


def test_a_bare_pipeline_run_charges_the_honest_cost_after_every_refusal():
    """A refused attack does not poison the next episode (no stateful residue in the pipeline)."""
    pipeline = pipeline_with(warning=warning_variant(_config_in_place))
    for _ in range(2):
        with pytest.raises(ComplianceGuardViolation):
            pipeline.score_peer(ledger_of({BREACH: 1.0}), peer="a")
    assert pipeline.score_config.get("NON_OFFSETTABLE") != []
    assert pipeline.score_config.get("PENALTY_LAMBDAS", {}).get(BREACH, 1.0) != 0.0


# -- reward and warning stages -----------------------------------------------------------------
def _huge_reward(out, ctx):
    return replace(out, amount=1e12, breakdown={**out.breakdown, "r_task": 1e12})

def _inf_reward(out, ctx):
    return replace(out, breakdown={**out.breakdown, "r_task": float("inf")})

def _negative_reward(out, ctx):
    return replace(out, breakdown={**out.breakdown, "r_task": -1e3})

def _weights_huge(out, ctx):
    return replace(out, breakdown={**out.breakdown, "weights": {"r_task": 1e12, "r_epi": 1.0, "r_collab": 1.0, "r_align": 1.0}})

def _reward_voids(out, ctx):
    return replace(out, void_rules=("P1",))

def _paid_for_nothing(out, ctx):
    return replace(out, breakdown={**out.breakdown, "by_rule": {"R99": 5.0}})

def _reward_float_liar(out, ctx):
    return replace(out, breakdown={**out.breakdown, "r_task": _FloatLiar(1e12)})

REWARD_ATTACKS = {"huge": _huge_reward, "inf": _inf_reward, "negative": _negative_reward,
                  "weights_huge": _weights_huge, "void_rules": _reward_voids,
                  "paid_for_nothing": _paid_for_nothing}


@pytest.mark.parametrize("mode", ["enforce", "fallback"])
@pytest.mark.parametrize("name", sorted(REWARD_ATTACKS))
def test_a_malicious_reward_stage_cannot_buy_back_a_breach(name, mode, settings, tmp_path, component_overlay):
    register_scorer(f"adv_r_{name}", reward_variant(REWARD_ATTACKS[name]), scope="run")
    configure(settings, SCORING={"STAGES": {"reward": f"adv_r_{name}"}}, COMPLIANCE={"GUARD_MODE": mode})
    trainer = breach_trainer(settings, tmp_path)
    try:
        run_epochs(trainer)
    except ComplianceGuardViolation:
        return
    assert_breach_cost_intact(trainer)


def test_a_reward_cannot_lift_a_void_node_above_its_floor_even_when_within_the_ceiling(settings, tmp_path, component_overlay):
    """The ceiling is generous (1e4): the floor must still hold for the largest legal reward."""
    def max_reward(out, ctx):
        return replace(out, breakdown={**out.breakdown, "r_task": 9_999.0})

    register_scorer("adv_max_reward", reward_variant(max_reward), scope="run")
    configure(settings, SCORING={"STAGES": {"reward": "adv_max_reward"}})
    trainer = breach_trainer(settings, tmp_path)
    run_epochs(trainer)
    assert_breach_cost_intact(trainer)


def test_a_reward_float_subclass_is_read_once():
    pipeline = pipeline_with(reward=reward_variant(_reward_float_liar), mode="fallback")
    card = pipeline.score_peer(ledger_of({BREACH: 1.0}), peer="a")
    assert card.is_void and card.final_score <= HONEST_COST


def test_a_warning_stage_cannot_turn_a_never_warning_first_rule_into_a_warning(settings, tmp_path, component_overlay):
    register_scorer("adv_warn", warning_variant(_hide_in_warning_only), scope="run")
    configure(settings, SCORING={"STAGES": {"warning": "adv_warn"}}, COMPLIANCE={"GUARD_MODE": "fallback"})
    trainer = breach_trainer(settings, tmp_path)
    with pytest.raises(ComplianceGuardViolation) as caught:          # a WARNING violation never falls back
        run_epochs(trainer)
    assert _codes(caught.value) == {"C005"}


def test_a_stage_whose_score_raises_fails_the_episode_not_the_cost(settings, tmp_path, component_overlay):
    def boom(out, ctx):
        raise RuntimeError("plugin bug")

    register_scorer("adv_boom", penalty_variant(boom), scope="run")
    configure(settings, SCORING={"STAGES": {"penalty": "adv_boom"}})
    trainer = breach_trainer(settings, tmp_path)
    with pytest.raises(RuntimeError, match="plugin bug"):
        run_epochs(trainer)
    assert not victim_cards(trainer)             # nothing was scored, so nothing was scored cheaply
