"""``ComplianceGuard`` against hostile and buggy stages (document 17 section 6.4).

Every case here is a stage a plugin author might write, by mistake or on purpose. The
guard must refuse each with the code 6.4.1 assigns it, in ``enforce`` mode, and in
``fallback`` mode must either substitute the reference or still refuse (C006, C007 and any
WARNING violation have nothing to substitute).
"""

from __future__ import annotations

import copy
import random
from dataclasses import replace

import pytest

from mactf.components.guards.compliance import (
    ComplianceGuard,
    GuardPolicy,
    ReadOnlyLedger,
)
from mactf.exceptions import (
    ComplianceGuardViolation,
    ImproperlyConfigured,
    LedgerReadOnlyError,
)
from mactf.scoring.base import ScoreContext, ScorerOutput, ScoringKind, WarningMutation
from mactf.scoring.ledger import Ledger
from mactf.scoring.pipeline import ScoringPipeline
from mactf.scoring.reward_scorer import RewardScorer
from mactf.scoring.penalty import PenaltyScorer

from scoring_helpers import (
    default_pipeline,
    ledger_of,
    penalty_variant,
    pipeline_with,
    reward_variant,
    settings_copy,
    warning_variant,
    with_output,
)

BREACH = {"R1": 2.0, "P1": 1.0, "P3": 1.0}


def refused(pipeline, spec=None, **kwargs):
    with pytest.raises(ComplianceGuardViolation) as excinfo:
        pipeline.score_peer(ledger_of(spec or BREACH), peer="a", episode=1, **kwargs)
    return excinfo.value


def codes(error) -> set[str]:
    return {item["code"] for item in error.findings}


def amounts_of(output):
    return dict(output.breakdown["amounts"])


# --------------------------------------------------------------------------- penalty stages
def _drop(output, ctx):
    return with_output(output, {"amounts": {}, "raw_total": 0.0, "total": 0.0}, amount=0.0, void_rules=())


def _halve_p1(output, ctx):
    amounts = amounts_of(output)
    amounts["P1"] *= 0.5
    return with_output(output, {"amounts": amounts})


def _forget_void(output, ctx):
    return replace(output, void_rules=())


def _no_omega(output, ctx):
    return with_output(output, {"omega": 1.0, "omega_rules": []})


def _bad_omega(value):
    return lambda output, ctx: with_output(output, {"omega": value})


def _poison(value, rule="P3"):
    def poison(output, ctx):
        amounts = amounts_of(output)
        amounts[rule] = value
        return with_output(output, {"amounts": amounts})

    return poison


def _phantom(output, ctx):
    return with_output(output, {"amounts": {**amounts_of(output), "P13": 40.0}})


def _mutations(output, ctx):
    return replace(output, mutations=(WarningMutation("a", "P1", 1, "clean"),))


PENALTY_CASES = [
    # name, function, ledger, first code that must appear, codes that must all appear
    ("zeroed", _drop, BREACH, {"C002", "C004"}),
    ("halved_floor", _halve_p1, BREACH, {"C002"}),
    ("forgot_to_void", _forget_void, BREACH, {"C004"}),
    ("omega_dropped", _no_omega, {"R1": 1.0, "P20": 1.0}, {"C003"}),
    ("omega_below_one", _bad_omega(0.5), BREACH, {"C003"}),
    ("omega_nan", _bad_omega(float("nan")), BREACH, {"C003"}),
    ("omega_text", _bad_omega("big"), BREACH, {"C003"}),
    ("negative_amount", _poison(-1.0), BREACH, {"C001"}),
    ("nan_amount", _poison(float("nan")), BREACH, {"C001"}),
    ("inf_amount", _poison(float("inf")), BREACH, {"C001"}),
    ("bool_amount", _poison(True), BREACH, {"C001"}),
    ("text_amount", _poison("lots"), BREACH, {"C001"}),
    ("nan_floor_amount", _poison(float("nan"), "P1"), BREACH, {"C001", "C002"}),
    ("phantom_rule", _phantom, BREACH, {"C009"}),
    ("mutations", _mutations, BREACH, {"C006"}),
    (
        "amounts_not_a_mapping",
        lambda output, ctx: with_output(output, {"amounts": [("P1", 5.0)]}),
        BREACH,
        {"C001"},
    ),
    (
        "breakdown_not_a_mapping",
        lambda output, ctx: replace(output, breakdown=[1, 2, 3]),
        BREACH,
        {"C001"},
    ),
    (
        "amount_nan",
        lambda output, ctx: replace(output, amount=float("nan")),
        BREACH,
        {"C001"},
    ),
    ("void_rules_garbage", lambda output, ctx: replace(output, void_rules=7), BREACH, {"C004"}),
    (
        "void_rules_other_rule",
        lambda output, ctx: replace(output, void_rules=("P3",)),
        BREACH,
        {"C004"},
    ),
]


@pytest.mark.parametrize("name,fn,spec,expected", PENALTY_CASES, ids=[c[0] for c in PENALTY_CASES])
def test_a_penalty_stage_that_weakens_a_breach_is_refused(name, fn, spec, expected):
    error = refused(pipeline_with(penalty=penalty_variant(fn)), spec)
    assert expected <= codes(error), error.findings
    assert error.code in expected | codes(error)
    assert error.code == error.findings[0]["code"]


@pytest.mark.parametrize("name,fn,spec,expected", PENALTY_CASES, ids=[c[0] for c in PENALTY_CASES])
def test_fallback_replaces_a_bad_penalty_with_the_reference_unless_it_is_out_of_role(
    name, fn, spec, expected
):
    reference = default_pipeline().score_peer(ledger_of(spec), peer="a", episode=1)
    pipeline = pipeline_with(penalty=penalty_variant(fn), mode="fallback")
    if expected & {"C006"}:
        refused(pipeline, spec)
        return
    card = pipeline.score_peer(ledger_of(spec), peer="a", episode=1)
    assert card.final_score == reference.final_score
    assert card.violated_non_offsettable == reference.violated_non_offsettable
    report = card.metadata["compliance_guard"]
    assert report["mode"] == "fallback" and report["substituted"] == ["penalty"]
    assert expected <= {item["code"] for item in report["violations"]}


def test_every_violation_is_attached_not_only_the_first():
    error = refused(pipeline_with(penalty=penalty_variant(_drop)))
    assert {item["code"] for item in error.findings} == {"C002", "C004"}
    assert all(item["stage"] == "CustomPenalty" and item["kind"] == "penalty" for item in error.findings)
    floor = next(item for item in error.findings if item["code"] == "C002")
    assert floor["rule_id"] == "P1" and floor["required"] == 5.0 and floor["observed"] == 0.0


def test_a_penalty_stage_may_charge_more_than_the_reference():
    """6.4.1: the guarantee is one-sided."""

    def double(output, ctx):
        amounts = {rule: value * 2.0 for rule, value in amounts_of(output).items()}
        return with_output(output, {"amounts": amounts, "total": output.amount * 2.0}, amount=output.amount * 2.0)

    card = pipeline_with(penalty=penalty_variant(double)).score_peer(
        ledger_of(BREACH), peer="a", episode=1
    )
    reference = default_pipeline().score_peer(ledger_of(BREACH), peer="a", episode=1)
    assert card.final_score < reference.final_score
    assert "compliance_guard" not in card.metadata


def test_a_penalty_stage_may_change_an_offsettable_rule():
    def cheaper_p3(output, ctx):
        amounts = amounts_of(output)
        amounts["P3"] = 0.5
        return with_output(output, {"amounts": amounts})

    card = pipeline_with(penalty=penalty_variant(cheaper_p3)).score_peer(
        ledger_of(BREACH), peer="a", episode=1
    )
    assert "compliance_guard" not in card.metadata
    assert card.continuous.penalty.amounts["P3"] == 0.5 and card.continuous.penalty.amounts["P1"] == 5.0


def test_a_penalty_stage_that_doubles_nothing_and_changes_nothing_passes_clean():
    pipeline = pipeline_with(penalty=penalty_variant(None))
    card = pipeline.score_peer(ledger_of(BREACH), peer="a", episode=1)
    reference = default_pipeline().score_peer(ledger_of(BREACH), peer="a", episode=1)
    assert card.final_score == reference.final_score
    assert "compliance_guard" not in card.metadata
    assert pipeline._last_guard_report.violations == ()


def test_the_floor_is_the_reference_even_when_the_stage_uses_a_cheaper_lambda():
    """A stage configured with a P1 lambda of 0.01 is held to the reference (5), not to itself.

    H2 lifts that lambda to the default floor of 1.0, so the stage charges 1.0: honest to its
    own arithmetic, below the project's reference, and therefore C002.
    """
    settings = settings_copy()
    cheap = {
        **settings["SCORING"],
        "PENALTY_LAMBDAS": {**settings["SCORING"]["PENALTY_LAMBDAS"], "P1": 0.01},
    }

    class Cheap(PenaltyScorer):
        def __init__(self, *, config=None):
            super().__init__(config=cheap)

    error = refused(pipeline_with(penalty=Cheap), {"P1": 1.0})
    floor = next(item for item in error.findings if item["code"] == "C002")
    assert floor["required"] == 5.0 and floor["observed"] == 1.0


# ---------------------------------------------------------------------------- reward stages
def _bucket(value, key="r_task"):
    return lambda output, ctx: with_output(output, {key: value})


REWARD_CASES = [
    ("above_ceiling", _bucket(1.0e9), {"C008"}),
    ("just_above_ceiling", _bucket(1.0e4 + 1.0), {"C008"}),
    ("nan_bucket", _bucket(float("nan")), {"C001"}),
    ("inf_bucket", _bucket(float("inf")), {"C001", "C008"} & {"C001"}),
    ("negative_bucket", _bucket(-1.0), {"C001"}),
    ("text_bucket", _bucket("a lot"), {"C001"}),
    ("none_bucket", _bucket(None), {"C001"}),
    ("negative_weight", lambda o, c: with_output(o, {"weights": {"r_task": -1.0}}), {"C001"}),
    ("nan_weight", lambda o, c: with_output(o, {"weights": {"r_task": float("nan")}}), {"C001"}),
    ("weights_not_mapping", lambda o, c: with_output(o, {"weights": [1, 2]}), {"C001"}),
    ("weight_inflates_total", lambda o, c: with_output(o, {"weights": {"r_task": 1.0e9}}), {"C008"}),
    ("amount_nan", lambda o, c: replace(o, amount=float("nan")), {"C001"}),
    ("amount_negative", lambda o, c: replace(o, amount=-3.0), {"C001"}),
    ("by_rule_phantom", lambda o, c: with_output(o, {"by_rule": {"R7": 1.0}}), {"C009"}),
    ("by_rule_negative", lambda o, c: with_output(o, {"by_rule": {"R1": -2.0}}), {"C001"}),
    ("void_rules", lambda o, c: replace(o, void_rules=("P1",)), {"C006"}),
    ("mutations", lambda o, c: replace(o, mutations=(WarningMutation("a", "R1", 1, "clean"),)), {"C006"}),
    ("breakdown_not_a_mapping", lambda o, c: replace(o, breakdown="oops"), {"C001"}),
]


@pytest.mark.parametrize("name,fn,expected", REWARD_CASES, ids=[c[0] for c in REWARD_CASES])
def test_a_reward_stage_that_breaks_its_role_is_refused(name, fn, expected):
    error = refused(pipeline_with(reward=reward_variant(fn)))
    assert expected <= codes(error), error.findings


@pytest.mark.parametrize("name,fn,expected", REWARD_CASES, ids=[c[0] for c in REWARD_CASES])
def test_fallback_replaces_a_bad_reward_with_the_reference(name, fn, expected):
    reference = default_pipeline().score_peer(ledger_of(BREACH), peer="a", episode=1)
    pipeline = pipeline_with(reward=reward_variant(fn), mode="fallback")
    if "C006" in expected:
        refused(pipeline)
        return
    card = pipeline.score_peer(ledger_of(BREACH), peer="a", episode=1)
    assert card.final_score == reference.final_score
    assert card.metadata["compliance_guard"]["substituted"] == ["reward"]


def test_a_reward_stage_cannot_lift_the_score_above_the_floor():
    """The central claim: a huge (but legal) reward never cancels a non-offsettable floor."""
    spec = {"R1": 1.0, "P1": 1.0}
    strict = pipeline_with(reward=reward_variant(_bucket(9000.0))).score_peer(
        ledger_of(spec), peer="a", episode=1
    )
    assert strict.is_void and strict.final_score == -5.0  # strict precedence suppresses the reward

    layered = settings_copy(SCORING={"PRECEDENCE_MODE": "layered"})
    card = pipeline_with(reward=reward_variant(_bucket(9000.0)), settings=layered).score_peer(
        ledger_of(spec), peer="a", episode=1
    )
    assert card.is_void
    # reward minus the reference floor: the breach costs exactly what it costs everywhere else
    assert card.final_score == pytest.approx(card.continuous.reward.total - 5.0)
    plain = default_pipeline(SCORING={"PRECEDENCE_MODE": "layered"}).score_peer(
        ledger_of(spec), peer="a", episode=1
    )
    assert card.continuous.reward.total - card.final_score == pytest.approx(
        plain.continuous.reward.total - plain.final_score
    )


def test_the_reward_ceiling_is_configurable():
    pipeline = pipeline_with(reward=reward_variant(_bucket(500.0)), reward_ceiling=100.0)
    assert "C008" in codes(refused(pipeline))
    assert pipeline_with(reward=reward_variant(_bucket(500.0)), reward_ceiling=1000.0).score_peer(
        ledger_of(BREACH), peer="a", episode=1
    )


def test_reward_for_a_rule_with_no_event_is_paid_only_when_the_ledger_has_it():
    ok = reward_variant(lambda o, c: with_output(o, {"by_rule": {"R1": 1.0}}))
    pipeline_with(reward=ok).score_peer(ledger_of(BREACH), peer="a", episode=1)
    assert "C009" in codes(refused(pipeline_with(reward=ok), {"P1": 1.0}))


# --------------------------------------------------------------------------- warning stages
def _hide_p1(self, ctx):
    return ctx.evolve(warning_events=tuple(ctx.warning_events) + ("P1",), escalated_rules=frozenset())


def test_a_warning_stage_cannot_demote_a_non_offsettable_breach_to_a_warning():
    error = refused(pipeline_with(warning=warning_variant(_hide_p1)))
    assert codes(error) == {"C005"} and error.findings[0]["rule_id"] == "P1"


def test_a_warning_stage_violation_is_not_repaired_in_fallback_mode():
    refused(pipeline_with(warning=warning_variant(_hide_p1), mode="fallback"))


def test_a_warning_stage_with_a_negative_amount_is_refused():
    stage = type(
        "NegWarning",
        (warning_variant(lambda self, ctx: ctx),),
        {"score": lambda self, ctx: replace(super(type(self), self).score(ctx), amount=-1.0)},
    )
    assert "C001" in codes(refused(pipeline_with(warning=stage)))


@pytest.mark.parametrize(
    "mutate,what",
    [
        (lambda s, c: c.evolve(peer="someone-else"), "peer"),
        (lambda s, c: c.evolve(episode=c.episode + 1), "episode"),
        (lambda s, c: c.evolve(node_path="x/y"), "node_path"),
        (lambda s, c: c.evolve(depth=3), "depth"),
        (lambda s, c: c.evolve(branch_void=frozenset()), "branch_void"),
        (lambda s, c: c.evolve(known_rules=frozenset({"P1"})), "known_rules"),
        (lambda s, c: c.evolve(severity_weights={"S0": 0, "S1": 0, "S2": 0, "S3": 0, "S4": 0}), "severity_weights"),
        (lambda s, c: c.evolve(config={**dict(c.config), "PENALTY_LAMBDAS": {}}), "config"),
        (lambda s, c: c.evolve(extras={**dict(c.extras), "allowed_rules": frozenset()}), "allowed_rules"),
        (lambda s, c: c.evolve(extras={**dict(c.extras), "precedence": None}), "precedence"),
    ],
)
def test_prepare_may_not_change_protected_context_fields(mutate, what):
    error = refused(pipeline_with(warning=warning_variant(mutate)), branch_void=frozenset({"P2"}))
    assert "C006" in codes(error)
    assert any(what in item["message"] for item in error.findings)


def test_a_reward_or_penalty_stage_may_not_change_warning_state_in_prepare():
    for factory in (reward_variant, penalty_variant):
        stage = factory(None, prepare=lambda self, ctx: ctx.evolve(warning_total=ctx.warning_total + 1.0))
        kind = "reward" if factory is reward_variant else "penalty"
        error = refused(pipeline_with(**{kind: stage}))
        assert codes(error) == {"C006"} and "warning_total" in error.findings[0]["message"]
        stage = factory(None, prepare=lambda self, ctx: ctx.evolve(
            extras={**dict(ctx.extras), "warning_mutations": (WarningMutation("a", "P1", 1, "clean"),)}))
        assert "C006" in codes(refused(pipeline_with(**{kind: stage})))


def test_a_stage_may_not_unvoid_a_node_whose_descendant_breached():
    """The branch-void a child propagated to this node survives every stage."""

    def unvoid(self, ctx):
        return ctx.evolve(branch_void=frozenset())

    for factory, kind in ((reward_variant, "reward"), (penalty_variant, "penalty")):
        error = refused(
            pipeline_with(**{kind: factory(None, prepare=unvoid)}),
            {"R1": 1.0}, branch_void=frozenset({"P1"}),
        )
        assert "C006" in codes(error)


def test_in_place_edits_of_the_context_are_caught_and_do_not_leak():
    pipeline = pipeline_with(
        penalty=penalty_variant(None, prepare=lambda self, ctx: (ctx.config["PENALTY_LAMBDAS"].update(P1=0.0), ctx)[1])
    )
    before = copy.deepcopy(pipeline.score_config)
    error = refused(pipeline)
    assert "C006" in codes(error) and "config" in error.findings[0]["message"]
    assert pipeline.score_config == before  # custom stages work on a private copy

    def poke_extras(self, ctx):
        ctx.extras["allowed_rules"] = frozenset()  # the dict is shared with later stages
        return ctx

    assert "C006" in codes(refused(pipeline_with(reward=reward_variant(None, prepare=poke_extras))))


def test_a_stage_that_edits_the_context_in_score_cannot_move_the_yardstick():
    """Zero the lambda in ``ctx.config`` during ``score``, then charge nothing: still C002."""

    def sabotage(output, ctx):
        ctx.config["PENALTY_LAMBDAS"]["P1"] = 0.0
        ctx.config["NON_OFFSETTABLE"] = []
        return _drop(output, ctx)

    pipeline = pipeline_with(penalty=penalty_variant(sabotage))
    before = copy.deepcopy(pipeline.score_config)
    assert codes(refused(pipeline)) == {"C006"}
    assert pipeline.score_config == before


# -------------------------------------------------------------------------------- the ledger
def test_read_only_ledger_refuses_every_write_and_shares_reads():
    source = ledger_of(BREACH)
    view = ComplianceGuard.view(source)
    assert isinstance(view, Ledger) and isinstance(view, ReadOnlyLedger)
    assert view.unwrap() is source and ComplianceGuard.view(view) is view
    assert view.events == source.events and len(view) == 3 and view.peer == "a"
    assert [e.rule_id for e in view.penalties()] == ["P1", "P3"]
    for call in (
        lambda: view.record("P1"),
        lambda: view.extend(source.events),
        lambda: view.clear(),
    ):
        with pytest.raises(LedgerReadOnlyError) as excinfo:
            call()
        assert excinfo.value.code == "C007"
    assert len(source) == 3
    source.record("R2")  # a live view
    assert len(view) == 4


@pytest.mark.parametrize("op", ["record", "extend", "clear"])
@pytest.mark.parametrize("factory,kind", [(reward_variant, "reward"), (penalty_variant, "penalty")])
def test_a_stage_that_writes_to_the_ledger_through_the_api_is_stopped(op, factory, kind):
    def tamper(output, ctx):
        {
            "record": lambda: ctx.ledger.record("P3", magnitude=9.0),
            "extend": lambda: ctx.ledger.extend(ctx.ledger.events),
            "clear": lambda: ctx.ledger.clear(),
        }[op]()
        return output

    ledger = ledger_of(BREACH)
    pipeline = pipeline_with(**{kind: factory(tamper)})
    with pytest.raises(LedgerReadOnlyError) as excinfo:
        pipeline.score_peer(ledger, peer="a", episode=1)
    assert excinfo.value.code == "C007"
    assert [e.rule_id for e in ledger.events] == ["R1", "P1", "P3"]


@pytest.mark.parametrize("factory,kind", [(reward_variant, "reward"), (penalty_variant, "penalty")])
def test_a_stage_that_reaches_for_the_event_list_is_caught_and_the_ledger_is_restored(factory, kind):
    def sneaky(output, ctx):
        ctx.ledger._events.clear()  # bypasses the public API
        return output

    ledger = ledger_of(BREACH)
    with pytest.raises(ComplianceGuardViolation) as excinfo:
        pipeline_with(**{kind: factory(sneaky)}).score_peer(ledger, peer="a", episode=1)
    assert {item["code"] for item in excinfo.value.findings} == {"C007"}
    assert [e.rule_id for e in ledger.events] == ["R1", "P1", "P3"]  # put back
    # ... and still refused in fallback mode: there is nothing to substitute for tampering
    ledger = ledger_of(BREACH)
    with pytest.raises(ComplianceGuardViolation):
        pipeline_with(**{kind: factory(sneaky)}, mode="fallback").score_peer(ledger, peer="a", episode=1)
    assert len(ledger) == 3


def test_a_stage_that_edits_the_ledger_and_then_raises_still_reports_the_edit():
    def sneaky_then_boom(output, ctx):
        ctx.ledger._events.append(ctx.ledger._events[0])
        raise RuntimeError("boom")

    ledger = ledger_of(BREACH)
    with pytest.raises(ComplianceGuardViolation) as excinfo:
        pipeline_with(reward=reward_variant(sneaky_then_boom)).score_peer(ledger, peer="a", episode=1)
    assert excinfo.value.code == "C007" and isinstance(excinfo.value.__cause__, RuntimeError)
    assert len(ledger) == 3


def test_prepare_may_not_replace_the_ledger():
    def swap(self, ctx):
        return ctx.evolve(ledger=Ledger(peer="a"))  # an empty, writable ledger

    for factory, kind in ((reward_variant, "reward"), (penalty_variant, "penalty")):
        assert "C007" in codes(refused(pipeline_with(**{kind: factory(None, prepare=swap)})))
    assert "C007" in codes(refused(pipeline_with(warning=warning_variant(lambda s, c: swap(s, c)))))


# --------------------------------------------------------------------- misc misbehaviour
def test_a_stage_that_raises_propagates_its_own_error_and_leaves_the_ledger_alone():
    def boom(output, ctx):
        raise RuntimeError("plugin bug")

    ledger = ledger_of(BREACH)
    for factory, kind in ((reward_variant, "reward"), (penalty_variant, "penalty")):
        with pytest.raises(RuntimeError, match="plugin bug"):
            pipeline_with(**{kind: factory(boom)}).score_peer(ledger, peer="a", episode=1)
        assert len(ledger) == 3


@pytest.mark.parametrize(
    "bad", [None, 3.5, "output", {"amount": 1.0}, (ScoringKind.REWARD,)], ids=repr
)
def test_a_stage_returning_the_wrong_type_is_refused(bad):
    stage = reward_variant(lambda o, c: bad)
    with pytest.raises(ImproperlyConfigured, match="not a ScorerOutput"):
        pipeline_with(reward=stage).score_peer(ledger_of(BREACH), peer="a", episode=1)


def test_prepare_returning_the_wrong_type_is_refused():
    stage = penalty_variant(None, prepare=lambda self, ctx: {"ctx": ctx})
    with pytest.raises(ImproperlyConfigured, match="not a ScoreContext"):
        pipeline_with(penalty=stage).score_peer(ledger_of(BREACH), peer="a", episode=1)


def test_a_non_deterministic_stage_is_refused_exactly_when_it_cheats():
    rng = random.Random(7)
    cheats: list[bool] = []

    def flaky(output, ctx):
        cheat = rng.random() < 0.5
        cheats.append(cheat)
        return _drop(output, ctx) if cheat else output

    pipeline = pipeline_with(penalty=penalty_variant(flaky))
    for episode in range(1, 41):
        ledger = ledger_of(BREACH)
        try:
            card = pipeline.score_peer(ledger, peer="a", episode=episode)
            assert cheats[-1] is False and card.is_void and card.continuous.penalty.amounts["P1"] == 5.0
        except ComplianceGuardViolation as error:
            assert cheats[-1] is True and "C002" in codes(error)
    assert any(cheats) and not all(cheats)


def test_two_bad_stages_are_both_reported():
    pipeline = pipeline_with(
        penalty=penalty_variant(_drop), reward=reward_variant(_bucket(1.0e9))
    )
    error = refused(pipeline)
    assert {"C002", "C004", "C008"} <= codes(error)
    assert {item["kind"] for item in error.findings} == {"penalty", "reward"}
    both = pipeline_with(
        penalty=penalty_variant(_drop), reward=reward_variant(_bucket(1.0e9)), mode="fallback"
    ).score_peer(ledger_of(BREACH), peer="a", episode=1)
    assert both.metadata["compliance_guard"]["substituted"] == ["penalty", "reward"]


def test_a_rule_scored_as_warning_only_cannot_be_charged_by_a_custom_stage():
    """P6 is warning-first: the first offence scores zero, so a stage charging for it is C009."""

    def charge(output, ctx):
        return with_output(output, {"amounts": {**amounts_of(output), "P6": 1.0}})

    pipeline = pipeline_with(penalty=penalty_variant(charge))
    assert "C009" in codes(refused(pipeline, {"P6": 1.0}))


def test_strict_precedence_admits_only_the_controlling_tier():
    """With a T1 breach present a lower-tier rule is suppressed; charging it is C009."""

    def charge(output, ctx):
        return with_output(output, {"amounts": {**amounts_of(output), "P11": 3.0}})

    assert "C009" in codes(refused(pipeline_with(penalty=penalty_variant(charge)), {"P1": 1.0, "P11": 1.0}))


# ------------------------------------------------------------------------- policy and trust
def test_trust_builtin_false_inspects_built_ins_and_finds_nothing():
    pipeline = default_pipeline(COMPLIANCE={"TRUST_BUILTIN": False})
    for episode, spec in enumerate([BREACH, {"P20": 1.0}, {"P6": 1.0}, {}, {"R1": 3.0}], start=1):
        card = pipeline.score_peer(ledger_of(spec), peer="a", episode=episode)
        assert "compliance_guard" not in card.metadata


def test_a_subclass_of_a_built_in_is_not_trusted():
    class Noop(PenaltyScorer):
        pass

    inspected = []
    guard = ComplianceGuard()
    original = guard._inspect_penalty
    guard._inspect_penalty = lambda *a, **k: (inspected.append(1), original(*a, **k))[1]
    pipeline = pipeline_with(penalty=Noop)
    pipeline.guard = guard
    pipeline.score_peer(ledger_of(BREACH), peer="a", episode=1)
    assert inspected == [1]


def test_guard_policy_validates_its_values():
    for kwargs in (
        {"mode": "off"}, {"min_floor_lambda": 0.0}, {"min_floor_lambda": float("nan")},
        {"reward_ceiling": 0.0}, {"reward_ceiling": float("inf")},
        {"tolerance": -1e-9}, {"tolerance": 1e-2}, {"tolerance": float("nan")},
    ):
        with pytest.raises(ImproperlyConfigured):
            GuardPolicy(**kwargs)
    policy = GuardPolicy.from_settings(
        {"COMPLIANCE": {"GUARD_MODE": "FALLBACK", "MIN_FLOOR_LAMBDA": 2, "REWARD_CEILING": 50,
                        "TOLERANCE": 0, "TRUST_BUILTIN": False}}
    )
    assert policy == GuardPolicy("fallback", 2.0, 50.0, 0.0, False)
    assert GuardPolicy.from_settings({}) == GuardPolicy() and GuardPolicy.from_settings(None) == GuardPolicy()


def test_a_floor_lambda_above_the_default_raises_the_reference():
    """COMPLIANCE.MIN_FLOOR_LAMBDA=50 lifts P1's floor from 5 to 50 for custom stages too."""
    settings = settings_copy(COMPLIANCE={"MIN_FLOOR_LAMBDA": 50.0})
    from mactf.scoring.pipeline import build_scorer

    pipeline = build_scorer(settings)
    assert pipeline.score_config["MIN_FLOOR_LAMBDA"] == 50.0
    card = pipeline.score_peer(ledger_of({"P1": 1.0}), peer="a", episode=1)
    assert card.continuous.penalty.amounts["P1"] == 50.0
    # a custom stage that charges the pre-hardening 5 is now below the reference
    stage = penalty_variant(lambda o, c: with_output(o, {"amounts": {"P1": 5.0}}))
    custom = pipeline_with(penalty=stage, settings=settings, min_floor_lambda=50.0)
    custom.score_config["MIN_FLOOR_LAMBDA"] = 50.0
    error = refused(custom, {"P1": 1.0})
    floor = next(item for item in error.findings if item["code"] == "C002")
    assert floor["required"] == 50.0 and floor["observed"] == 5.0


def test_guard_report_is_exposed_on_the_card_only_when_something_happened():
    pipeline = pipeline_with(reward=reward_variant(None))
    card = pipeline.score_peer(ledger_of(BREACH), peer="a", episode=1)
    assert "compliance_guard" not in card.metadata and pipeline._last_guard_report.violations == ()
    bad = pipeline_with(reward=reward_variant(_bucket(-1.0)), mode="fallback")
    card = bad.score_peer(ledger_of(BREACH), peer="a", episode=1)
    assert card.metadata["compliance_guard"]["violations"][0]["code"] == "C001"
    assert card.metadata["compliance_guard"]["violations"][0]["stage"] == "CustomReward"


def test_scoring_a_hierarchy_node_keeps_the_ledger_a_plain_ledger_for_the_caller():
    ledger = ledger_of(BREACH)
    pipeline = pipeline_with(reward=reward_variant(None))
    pipeline.score_peer(ledger, peer="a", episode=1)
    assert type(ledger) is Ledger and ledger.peer == "a" and len(ledger) == 3
