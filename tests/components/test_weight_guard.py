"""WeightGuard: what a custom weight strategy may hand to ``Aggregator.aggregate`` (document 17 section 10.4).

One test per violation code and per mode, an adversarial catalogue, and the Hypothesis property
``test_guard_property_weights_are_a_simplex``.
"""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings as hyp_settings, strategies as st

from mactf.coopetition.aggregation import Aggregator
from mactf.coopetition.report import AggregationReport
from mactf.coopetition.temperature import AdaptiveKappa
from mactf.coopetition.weight_guard import CODES, GuardOutcome, Violation, WeightGuard
from mactf.coopetition.weights import (
    BUILTIN_WEIGHT_TYPES,
    KappaSoftmaxAverage,
    NoAggregation,
    ReferenceOnlyAverage,
    UniformAverage,
    WeightStrategy,
    WeightTraits,
    make_report,
)
from mactf.exceptions import ComplianceError, ImproperlyConfigured, WeightGuardViolation

NAMES = ["a", "b", "c", "d"]
FLOOR, CAP = 0.02, 0.60
ALL = {name: True for name in NAMES}


def _report(weights, *, names=NAMES, eligible=None, **kwargs) -> AggregationReport:
    return AggregationReport(
        weights=dict(weights), raw_weights=dict(weights), kappa=1.0, entropy=0.0,
        order=tuple(sorted(names)), **kwargs,
    )


def _check(guard, report, *, eligible=None, strategy=None, names=NAMES):
    return guard.check(
        strategy or _Custom(), report, names=names, eligible=eligible or {n: True for n in names},
        floor=FLOOR, cap=CAP,
    )


class _Custom(WeightStrategy):
    name = "custom"

    def apply(self, vectors, scores, *, kappa, level=0, eligible=None, confidence=None):
        raise NotImplementedError


def _codes(violations):
    return sorted({item.code for item in violations})


GOOD = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}


# -- inspect: one case per code ----------------------------------------------------------
def test_a_legal_report_passes_untouched() -> None:
    outcome = _check(WeightGuard(), _report(GOOD))
    assert outcome.action == "pass" and outcome.violations == () and outcome.report.weights == GOOD


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), True, "0.5", None])
def test_u001_a_weight_that_is_not_a_finite_number(bad) -> None:
    weights = {**GOOD, "a": bad}
    found = WeightGuard().inspect(_report(weights), names=NAMES, eligible=ALL, floor=FLOOR, cap=CAP,
                                  traits=WeightTraits())
    assert _codes(found) == ["U001"] and found[0].peer == "a" and found[0].hard


def test_u002_a_negative_weight() -> None:
    weights = {"a": 0.7, "b": 0.5, "c": -0.1, "d": -0.1}
    found = WeightGuard().inspect(_report(weights), names=NAMES, eligible=ALL, floor=FLOOR, cap=CAP,
                                  traits=WeightTraits())
    assert _codes(found) == ["U002"] and {item.peer for item in found} == {"c", "d"}


def test_u002_tolerates_rounding_noise_below_the_tolerance() -> None:
    weights = {"a": 0.5, "b": 0.5 + 1e-12, "c": 0.0 - 1e-12, "d": 0.0}
    guard = WeightGuard()
    found = guard.inspect(_report(weights), names=NAMES, eligible={"a": True, "b": True, "c": False, "d": False},
                          floor=FLOOR, cap=CAP, traits=WeightTraits())
    assert _codes(found) == []


def test_u003_the_weights_must_name_exactly_the_group() -> None:
    missing = {k: v for k, v in GOOD.items() if k != "d"}
    extra = {**GOOD, "ghost": 0.0}
    for weights, text in ((missing, "missing ['d']"), (extra, "unexpected ['ghost']")):
        found = WeightGuard().inspect(_report(weights), names=NAMES, eligible=ALL, floor=FLOOR, cap=CAP,
                                      traits=WeightTraits())
        assert _codes(found) == ["U003"] and text in found[0].detail


def test_u004_a_void_peer_given_weight() -> None:
    weights = {"a": 0.3, "b": 0.3, "c": 0.2, "d": 0.2}
    eligible = {**ALL, "d": False}                       # d is void this epoch
    found = WeightGuard().inspect(_report(weights), names=NAMES, eligible=eligible, floor=FLOOR, cap=CAP,
                                  traits=WeightTraits())
    assert "U004" in _codes(found)
    assert any(item.code == "U004" and item.peer == "d" for item in found)


def test_u004_a_trimmed_peer_given_weight() -> None:
    weights = {"a": 0.3, "b": 0.3, "c": 0.2, "d": 0.2}
    found = WeightGuard().inspect(_report(weights, trimmed_peers=("d",)), names=NAMES, eligible=ALL,
                                  floor=FLOOR, cap=CAP, traits=WeightTraits())
    assert any(item.code == "U004" and item.peer == "d" for item in found)


@pytest.mark.parametrize("weights,text", [
    ({"a": 3.0, "b": 0.0, "c": 0.0, "d": 0.0}, "outside"),         # above the cap
    ({"a": 0.5, "b": 0.5, "c": 0.0, "d": 0.0}, "outside"),         # eligible and untrimmed at 0 < floor
    ({"a": 0.3, "b": 0.3, "c": 0.3, "d": 0.3}, "sum to"),          # sums to 1.2
    ({"a": 0.1, "b": 0.1, "c": 0.1, "d": 0.1}, "sum to"),
])
def test_u005_outside_the_bounds_or_not_a_simplex(weights, text) -> None:
    found = WeightGuard().inspect(_report(weights), names=NAMES, eligible=ALL, floor=FLOOR, cap=CAP,
                                  traits=WeightTraits())
    assert "U005" in _codes(found) and any(text in item.detail for item in found)


def test_an_eligible_peer_at_zero_is_legal_only_when_it_is_trimmed() -> None:
    weights = {"a": 0.5, "b": 0.3, "c": 0.2, "d": 0.0}
    guard = WeightGuard()
    untrimmed = guard.inspect(_report(weights), names=NAMES, eligible=ALL, floor=FLOOR, cap=CAP,
                              traits=WeightTraits())
    trimmed = guard.inspect(_report(weights, trimmed_peers=("d",)), names=NAMES, eligible=ALL,
                            floor=FLOOR, cap=CAP, traits=WeightTraits())
    assert _codes(untrimmed) == ["U005"] and trimmed == []


def test_u006_fallback_needs_the_declaration() -> None:
    report = _report(GOOD, fallback=True)
    guard = WeightGuard()
    assert _codes(guard.inspect(report, names=NAMES, eligible=ALL, floor=FLOOR, cap=CAP,
                                traits=WeightTraits())) == ["U006"]
    assert guard.inspect(report, names=NAMES, eligible=ALL, floor=FLOOR, cap=CAP,
                         traits=WeightTraits(may_fallback=True)) == []


def test_the_empty_group_has_nothing_to_check() -> None:
    weights = {name: 0.0 for name in NAMES}
    found = WeightGuard().inspect(_report(weights), names=NAMES, eligible={n: False for n in NAMES},
                                  floor=FLOOR, cap=CAP, traits=WeightTraits())
    assert found == []


def test_sum_tolerance_is_the_guards_tolerance() -> None:
    weights = {"a": 0.4 + 5e-7, "b": 0.3, "c": 0.2, "d": 0.1}
    assert _check(WeightGuard(tolerance=1e-6), _report(weights)).action == "pass"
    tight = WeightGuard(tolerance=1e-9)
    assert _check(tight, _report({"a": 0.4 + 1e-5, "b": 0.3, "c": 0.2, "d": 0.1})).action == "repaired"


# -- check: the three modes ----------------------------------------------------------------------
def test_repair_projects_a_soft_violation_onto_the_bounded_simplex() -> None:
    outcome = _check(WeightGuard(), _report({"a": 0.9, "b": 0.5, "c": 0.4, "d": 0.2}))      # sums to 2.0
    assert outcome.action == "repaired" and _codes(outcome.violations) == ["U005"]
    weights = outcome.report.weights
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(FLOOR - 1e-9 <= value <= CAP + 1e-9 for value in weights.values())
    assert any("weight_guard repaired" in note for note in outcome.report.adjustments)


def test_repair_zeroes_a_void_peer_and_renormalizes_the_rest() -> None:
    eligible = {**ALL, "d": False}
    outcome = _check(WeightGuard(), _report({"a": 0.3, "b": 0.3, "c": 0.2, "d": 0.2}), eligible=eligible)
    assert outcome.action == "repaired" and "U004" in _codes(outcome.violations)
    assert outcome.report.weights["d"] == 0.0
    assert sum(outcome.report.weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert "d" in outcome.report.gated_peers                       # the repaired report says why


def test_repair_of_all_zero_proposals_is_uniform_over_participants() -> None:
    outcome = _check(WeightGuard(), _report({n: 0.0 for n in NAMES}))
    assert outcome.action == "repaired"
    assert list(outcome.report.weights.values()) == pytest.approx([0.25] * 4)


@pytest.mark.parametrize("weights", [
    {**GOOD, "a": float("nan")}, {**GOOD, "a": -0.3, "b": 0.7}, {k: v for k, v in GOOD.items() if k != "d"},
])
def test_repair_cannot_fix_a_hard_violation_so_it_falls_back(weights) -> None:
    outcome = _check(WeightGuard(mode="repair"), _report(weights))
    assert outcome.action == "fallback" and outcome.report is None and outcome.violations[0].hard


def test_fallback_mode_never_repairs() -> None:
    outcome = _check(WeightGuard(mode="fallback"), _report({"a": 0.3, "b": 0.3, "c": 0.3, "d": 0.3}))
    assert outcome.action == "fallback" and outcome.report is None


def test_enforce_mode_raises_e063_with_every_violation() -> None:
    with pytest.raises(WeightGuardViolation) as caught:
        _check(WeightGuard(mode="enforce"), _report({"a": 3.0, "b": -1.0, "c": 0.0, "d": 0.0}))
    assert caught.value.code == "E063" and isinstance(caught.value, ComplianceError)
    assert {item.code for item in caught.value.violations} == {"U002"}     # stops at the first class
    assert "E063" in str(caught.value) and "custom" in str(caught.value)


def test_an_infeasible_repair_falls_back_instead_of_raising() -> None:
    names = ["a", "b", "c", "d", "e"]
    weights = {n: 0.5 for n in names}
    guard = WeightGuard()
    outcome = guard.check(_Custom(), _report(weights, names=names), names=names,
                          eligible={"a": True, "b": False, "c": False, "d": False, "e": False},
                          floor=FLOOR, cap=0.4)          # one participant, cap 0.4: no simplex exists
    assert outcome.action == "fallback"


# -- run: apply() raising --------------------------------------------------------------------------
@pytest.mark.parametrize("error", [KeyError("x"), RuntimeError("boom"), ValueError("v"), ComplianceError("c")])
def test_u007_apply_raising_becomes_a_fallback(error) -> None:
    def call(strategy):
        raise error

    outcome = WeightGuard().run(_Custom(), call, names=NAMES, eligible=ALL, floor=FLOOR, cap=CAP)
    assert outcome.action == "fallback" and outcome.violations[0].code == "U007"
    assert type(error).__name__ in outcome.violations[0].detail


def test_u007_under_enforce_raises_e063_chained() -> None:
    def call(strategy):
        raise KeyError("x")

    with pytest.raises(WeightGuardViolation) as caught:
        WeightGuard(mode="enforce").run(_Custom(), call, names=NAMES, eligible=ALL, floor=FLOOR, cap=CAP)
    assert isinstance(caught.value.__cause__, KeyError)


def test_run_checks_what_the_call_returns() -> None:
    outcome = WeightGuard().run(_Custom(), lambda s: _report(GOOD), names=NAMES, eligible=ALL,
                                floor=FLOOR, cap=CAP)
    assert outcome.action == "pass" and outcome.report.weights == GOOD


# -- scope ----------------------------------------------------------------------------------------
def _aggregator(cls=Aggregator):
    return cls(controller=AdaptiveKappa(num_peers=4))


def test_scope_custom_leaves_the_builtins_alone() -> None:
    guard = WeightGuard()
    for cls in BUILTIN_WEIGHT_TYPES:
        if cls is KappaSoftmaxAverage:
            instance = cls(_aggregator())
        elif cls.__name__ == "ExplicitWeightStrategy":
            instance = cls({"a": 1.0})
        else:
            instance = cls()
        assert not guard.covers(instance, _aggregator()), cls.__name__


def test_scope_custom_covers_everything_else_and_a_subclassed_aggregator() -> None:
    class Sub(Aggregator):
        pass

    guard = WeightGuard()
    assert guard.covers(_Custom(), _aggregator())
    assert guard.covers(KappaSoftmaxAverage(_aggregator(Sub)))                       # custom compute_weights
    assert guard.covers(KappaSoftmaxAverage(None), _aggregator(Sub))                 # the trainer's aggregator
    assert not guard.covers(KappaSoftmaxAverage(None), _aggregator())


def test_scope_all_and_off_and_the_no_aggregation_exemption() -> None:
    assert WeightGuard(scope="all").covers(UniformAverage())
    assert not WeightGuard(scope="off").covers(_Custom())
    assert not WeightGuard(scope="all").covers(NoAggregation())          # writes nothing: nothing to check
    assert WeightGuard(scope="all").covers(ReferenceOnlyAverage())


# -- construction ---------------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    {"mode": "skip"}, {"scope": "some"}, {"tolerance": 0.0}, {"tolerance": 0.5}, {"tolerance": float("nan")},
    {"tolerance": True},
])
def test_e060_bad_guard_settings(kwargs) -> None:
    with pytest.raises(ImproperlyConfigured, match="E060"):
        WeightGuard(**kwargs)


def test_from_settings_reads_every_key_and_rejects_strangers() -> None:
    guard = WeightGuard.from_settings({"mode": "Enforce", "SCOPE": "all", "TOLERANCE": 1e-4, "PROBE": False})
    assert guard.describe() == {"mode": "enforce", "scope": "all", "tolerance": 1e-4, "probe": False}
    assert WeightGuard.from_settings(None).describe()["mode"] == "repair"
    with pytest.raises(ImproperlyConfigured, match="E060.*unknown keys"):
        WeightGuard.from_settings({"MODES": "repair"})


def test_every_code_has_a_description() -> None:
    assert set(CODES) == {f"U00{i}" for i in range(1, 8)}
    assert str(Violation("U001", "a", "weight is nan")) == "U001 [a]: weight is nan"


# -- the property ----------------------------------------------------------------------------------
_weight = st.one_of(
    st.floats(min_value=-5, max_value=5),
    st.sampled_from([0.0, 1.0, 0.02, 0.6, -0.0, float("nan"), float("inf"), float("-inf")]),
)


@st.composite
def proposals(draw):
    size = draw(st.integers(min_value=1, max_value=7))
    names = [f"p{i}" for i in range(size)]
    weights = {name: draw(_weight) for name in names}
    eligible = {name: draw(st.booleans()) for name in names}
    trimmed = tuple(name for name in names if draw(st.booleans()) and draw(st.booleans()))
    return names, weights, eligible, trimmed


@hyp_settings(max_examples=300, deadline=None)
@given(case=proposals(), mode=st.sampled_from(["repair", "fallback"]))
def test_guard_property_weights_are_a_simplex(case, mode) -> None:
    """Whatever a stage proposes, what survives ``fallback``/``repair`` satisfies I3."""
    names, weights, eligible, trimmed = case
    floor, cap = 0.0, 1.0
    if sum(eligible.values()) and sum(1 for n in names if eligible[n] and n not in trimmed):
        floor, cap = 0.02, 0.6
        participants = sum(1 for n in names if eligible[n] and n not in trimmed)
        if participants * cap < 1.0:                                 # no simplex exists: widen the cap
            cap = 1.0
    report = _report(weights, names=names, trimmed_peers=trimmed)
    guard = WeightGuard(mode=mode)
    outcome = guard.check(_Custom(), report, names=names, eligible=eligible, floor=floor, cap=cap)
    if outcome.action == "fallback":
        assert outcome.report is None                                 # the caller runs kappa_softmax
        return
    result = outcome.report
    assert set(result.weights) == set(names)
    assert all(isinstance(v, float) and math.isfinite(v) and v >= -1e-6 for v in result.weights.values())
    participants = [n for n in names if eligible[n] and n not in trimmed]
    for name in names:
        if name not in participants:
            assert result.weights[name] <= 1e-6, "an ineligible or trimmed peer carries weight"
    if participants:
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-5)
        for name in participants:
            assert floor - 1e-5 <= result.weights[name] <= cap + 1e-5
    # and the guard accepts its own output
    again = guard.inspect(result, names=names, eligible=eligible, floor=floor, cap=cap, traits=WeightTraits())
    assert not any(item.hard for item in again)


@hyp_settings(max_examples=200, deadline=None)
@given(case=proposals())
def test_guard_property_enforce_rejects_exactly_what_inspect_finds(case) -> None:
    names, weights, eligible, trimmed = case
    report = _report(weights, names=names, trimmed_peers=trimmed)
    guard = WeightGuard(mode="enforce")
    found = guard.inspect(report, names=names, eligible=eligible, floor=0.0, cap=1.0, traits=WeightTraits())
    if found:
        with pytest.raises(WeightGuardViolation) as caught:
            guard.check(_Custom(), report, names=names, eligible=eligible, floor=0.0, cap=1.0)
        assert {item.code for item in caught.value.violations} == {item.code for item in found}
    else:
        assert guard.check(_Custom(), report, names=names, eligible=eligible, floor=0.0, cap=1.0).action == "pass"


def test_make_report_derives_gated_peers_and_zero_fills() -> None:
    report = make_report({"a": 1.0}, ["b", "a", "c"], kappa=2.0, eligible={"a": True, "b": False, "c": False})
    assert report.order == ("a", "b", "c") and report.weights == {"a": 1.0, "b": 0.0, "c": 0.0}
    assert report.gated_peers == ("b", "c") and report.entropy == pytest.approx(0.0)


def test_w017_is_logged_when_an_output_is_repaired(caplog) -> None:
    import logging
    with caplog.at_level(logging.WARNING):
        outcome = _check(WeightGuard(), _report({"a": 0.9, "b": 0.5, "c": 0.4, "d": 0.2}))
    assert outcome.action == "repaired"
    assert any(r.getMessage().startswith("W017") for r in caplog.records)
