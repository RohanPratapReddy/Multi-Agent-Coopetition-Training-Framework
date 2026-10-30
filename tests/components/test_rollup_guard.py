"""RollupGuard and the rollup unification (document 17 section 8.6; invariant R4; codes C010-C012)."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings as hyp_settings, strategies as st

from mactf.components import HUB, register_function
from mactf.components.guards.rollup import RollupGuard, UNION_BY_CONSTRUCTION, guard_rollup
from mactf.exceptions import ComplianceGuardViolation, ImproperlyConfigured, ValidationError
from mactf.rl.functions.rollup import (
    ExplicitRollup,
    KappaWeightedRollup,
    MeanRollup,
    MinimumRollup,
    SumRollup,
    WorstCaseRollup,
)
from mactf.scoring.base import ScorerOutput, ScoringKind
from mactf.scoring.hierarchy import (
    ROLLUP_STRATEGIES,
    HierarchicalScorer,
    build_rollup,
)

from p2_support import configure, hierarchy_trainer, tune
from p3_examples import MedianRollup, TrimmedMeanRollup


def _child(amount: float, void=(), rules=()) -> ScorerOutput:
    return ScorerOutput(kind=ScoringKind.PENALTY, amount=amount, void_rules=tuple(void), rules=tuple(rules))


def _out(**kwargs) -> ScorerOutput:
    return ScorerOutput(kind=ScoringKind.HIERARCHY, **kwargs)


class _Custom:
    """A rollup whose output is whatever the test says."""

    name = "custom"

    def __init__(self, produce):
        self.produce = produce
        self.calls = 0

    def rollup(self, parent, children, *, weights=None, scores=None):
        self.calls += 1
        return self.produce(parent, children)


# -- the worked example (8.9.3) ---------------------------------------------------------
def test_union_survives_trimming() -> None:
    guard = RollupGuard(TrimmedMeanRollup(trim=1))
    children = [_child(9.0, ["P1"]), _child(2.0), _child(3.0), _child(4.0), _child(1.0)]
    out = guard.rollup("root", children)
    assert out.amount == 3.0                   # 9.0 and 1.0 trimmed; mean of 2, 3, 4
    assert out.void_rules == ("P1",)           # the trimmed child's void is still carried up
    assert not guard.reports                   # a clean output leaves no report


class Forgetful:
    name = "forgetful"

    def rollup(self, parent, children, *, weights=None, scores=None):
        return ScorerOutput(kind=ScoringKind.HIERARCHY, amount=0.0)   # drops every void


def test_laundering_is_caught_in_enforce_mode_and_repaired_in_fallback() -> None:
    children = [_child(1.0, ["P2"])]
    with pytest.raises(ComplianceGuardViolation) as caught:
        RollupGuard(Forgetful(), mode="enforce").rollup("root", children)
    assert caught.value.code == "C010"
    guard = RollupGuard(Forgetful(), mode="fallback")
    repaired = guard.rollup("root", children)
    assert repaired.void_rules == ("P2",)
    assert guard.reports[-1].substituted == ("forgetful",)


def test_test_trimmed_mean_rollup_registers_and_runs_through_the_book(component_overlay) -> None:
    """``test_trimmed_mean_rollup``: the registered class is built by name and is guarded."""
    register_function("rollup", "trimmed_mean", TrimmedMeanRollup, version="1", scope="run")
    built = build_rollup({"type": "trimmed_mean", "trim": 1})
    assert isinstance(built, RollupGuard)               # not one of the six built-ins
    assert isinstance(built.inner, TrimmedMeanRollup)
    children = [_child(9.0, ["P1"]), _child(2.0), _child(3.0), _child(4.0), _child(1.0)]
    assert built.rollup("p", children).void_rules == ("P1",)


# -- C010 / C011 / C012 -----------------------------------------------------------------
@pytest.mark.parametrize("mode", ["enforce", "fallback"])
def test_c010_a_partial_void_set_is_refused(mode) -> None:
    custom = _Custom(lambda parent, children: _out(amount=0.0, void_rules=("P1",)))
    guard = RollupGuard(custom, mode=mode)
    children = [_child(1.0, ["P1", "P2"])]
    if mode == "enforce":
        with pytest.raises(ComplianceGuardViolation, match="C010"):
            guard.rollup("p", children)
    else:
        assert set(guard.rollup("p", children).void_rules) == {"P1", "P2"}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "x", None])
def test_c011_a_non_finite_or_non_numeric_amount(bad) -> None:
    custom = _Custom(lambda parent, children: _out(amount=bad))
    with pytest.raises(ComplianceGuardViolation, match="C011"):
        RollupGuard(custom).rollup("p", [_child(1.0)])
    fallback = RollupGuard(custom, mode="fallback").rollup("p", [_child(1.0), _child(3.0)])
    assert fallback.amount == pytest.approx(2.0)          # the reference kappa rollup: the mean


def test_c011_a_bool_is_not_an_amount() -> None:
    custom = _Custom(lambda parent, children: _out(amount=True))
    with pytest.raises(ComplianceGuardViolation, match="C011"):
        RollupGuard(custom).rollup("p", [_child(1.0)])


def test_c011_a_raising_rollup_and_a_wrong_return_type() -> None:
    def boom(parent, children):
        raise RuntimeError("plugin bug")

    with pytest.raises(ComplianceGuardViolation, match="C011") as caught:
        RollupGuard(_Custom(boom)).rollup("p", [_child(1.0)])
    assert isinstance(caught.value.__cause__, RuntimeError)
    with pytest.raises(ComplianceGuardViolation, match="C011"):
        RollupGuard(_Custom(lambda parent, children: 3.0)).rollup("p", [_child(1.0)])
    assert RollupGuard(_Custom(boom), mode="fallback").rollup("p", [_child(2.0)]).amount == 2.0


def test_c012_warning_mutations_are_refused() -> None:
    from mactf.scoring.base import WarningMutation

    mutation = WarningMutation.__new__(WarningMutation)
    custom = _Custom(lambda parent, children: ScorerOutput(
        kind=ScoringKind.HIERARCHY, amount=0.0, mutations=(mutation,)))
    with pytest.raises(ComplianceGuardViolation, match="C012"):
        RollupGuard(custom).rollup("p", [_child(1.0)])


def test_the_guard_reports_every_violation_at_once() -> None:
    custom = _Custom(lambda parent, children: _out(amount=float("nan")))
    with pytest.raises(ComplianceGuardViolation) as caught:
        RollupGuard(custom).rollup("p", [_child(1.0, ["P1"])])
    codes = {item["code"] for item in caught.value.findings}
    assert codes == {"C010", "C011"}


def test_a_bad_mode_and_a_non_rollup_are_refused() -> None:
    with pytest.raises(ImproperlyConfigured):
        RollupGuard(MeanRollup(), mode="skip")
    with pytest.raises(ImproperlyConfigured):
        RollupGuard(object())


# -- which classes are trusted -----------------------------------------------------------
def test_builtins_are_returned_unwrapped_and_subclasses_are_guarded() -> None:
    for cls in UNION_BY_CONSTRUCTION:
        assert guard_rollup(cls()) is not None and not isinstance(guard_rollup(cls()), RollupGuard)

    class Sub(MeanRollup):                       # a subclass may override rollup() and drop the union
        pass

    assert isinstance(guard_rollup(Sub()), RollupGuard)
    explicit = ExplicitRollup(lambda parent, children, **kw: _out(amount=0.0))
    assert guard_rollup(explicit, include_explicit=False) is explicit       # the in-code exception
    assert isinstance(guard_rollup(explicit, include_explicit=True), RollupGuard)  # the RL.ROLLUP path
    assert guard_rollup(None) is None
    wrapped = guard_rollup(Forgetful())
    assert guard_rollup(wrapped) is wrapped               # never wrapped twice


def test_an_explicit_rollup_on_the_rl_path_cannot_launder_a_void() -> None:
    laundering = ExplicitRollup(lambda parent, children, **kw: _out(amount=0.0))
    guarded = guard_rollup(laundering, mode="fallback", include_explicit=True)
    assert guarded.rollup("p", [_child(1.0, ["P9"])]).void_rules == ("P9",)


def test_describe_and_name_follow_the_wrapped_rollup() -> None:
    guard = RollupGuard(TrimmedMeanRollup(trim=2))
    assert guard.name == "trimmed_mean"
    described = guard.describe()
    assert described["guard"] == "rollup" and described["inner"]["trim"] == 2


# -- the unified table ----------------------------------------------------------------------
def test_one_table_serves_the_registry_and_the_scorer() -> None:
    from mactf.rl.functions.registry import FUNCTION_TYPES, ROLLUP_TYPES

    assert ROLLUP_STRATEGIES is FUNCTION_TYPES["rollup"] is ROLLUP_TYPES
    assert "kappa_weighted" in ROLLUP_STRATEGIES
    assert isinstance(build_rollup("kappa_weighted"), KappaWeightedRollup)


def test_a_rollup_registered_anywhere_is_visible_everywhere(component_overlay) -> None:
    register_function("rollup", "trimmed_mean", TrimmedMeanRollup, version="1", scope="run")
    built = build_rollup("trimmed_mean")
    assert isinstance(built, RollupGuard)
    assert "trimmed_mean" in HUB.registry("rl_function.rollup").names()
    with pytest.raises(ValidationError, match="unknown rollup"):
        build_rollup("median_unregistered")


# -- HierarchicalScorer.rollup_source -----------------------------------------------------------
def test_rollup_source_is_read_at_every_use() -> None:
    chosen = {"value": None}
    scorer = HierarchicalScorer(rollup_source=lambda: chosen["value"])
    assert isinstance(scorer.rollup, KappaWeightedRollup)            # no choice: the default
    chosen["value"] = MeanRollup()
    assert isinstance(scorer.rollup, MeanRollup)                      # a swap lands without a rebuild
    chosen["value"] = None
    assert isinstance(scorer.rollup, KappaWeightedRollup)


def test_without_a_source_the_attribute_behaves_as_before() -> None:
    scorer = HierarchicalScorer(rollup="mean")
    assert isinstance(scorer.rollup, MeanRollup)
    scorer.rollup = "sum"
    assert isinstance(scorer.rollup, SumRollup)


def test_the_book_serves_rl_rollup_through_the_scorer_and_guards_it(settings, tmp_path, component_overlay) -> None:
    register_function("rollup", "trimmed_mean", TrimmedMeanRollup, version="1", scope="run")
    tune(settings, EPOCHS=1)
    configure(settings, RL={"ROLLUP": {"type": "trimmed_mean", "params": {"trim": 0}}})
    trainer = hierarchy_trainer(settings, tmp_path).build()
    assert isinstance(trainer.scorer, HierarchicalScorer)
    assert isinstance(trainer.scorer.rollup, RollupGuard)               # RL.ROLLUP is live and guarded
    assert isinstance(trainer.scorer.rollup.inner, TrimmedMeanRollup)
    trainer.run_epoch(0)
    assert trainer.scorer.rollup.reports.maxlen == 64


def test_with_no_rl_rollup_the_scorer_keeps_its_default(settings, tmp_path) -> None:
    tune(settings, EPOCHS=1)
    trainer = hierarchy_trainer(settings, tmp_path).build()
    assert isinstance(trainer.scorer.rollup, KappaWeightedRollup)


# -- the property (19.7.1) -------------------------------------------------------------------------
_RULES = ["P1", "P2", "P17", "P20", "P21"]
_children = st.lists(
    st.tuples(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
        st.lists(st.sampled_from(_RULES), max_size=3, unique=True),
    ),
    min_size=1, max_size=9,
)


@hyp_settings(max_examples=200, deadline=None)
@given(rows=_children, trim=st.integers(min_value=0, max_value=4), which=st.sampled_from(["trimmed", "median"]))
def test_guard_property_rollup_union_survives_trimming(rows, trim, which) -> None:
    """Trimmed-mean and median rollups over random child sets with void children keep the union."""
    inner = TrimmedMeanRollup(trim=trim) if which == "trimmed" else MedianRollup()
    guard = RollupGuard(inner, mode="enforce")           # enforce: any loss of support would raise
    children = [_child(amount, void) for amount, void in rows]
    out = guard.rollup("root", children)
    union = {rule for child in children for rule in child.void_rules}
    assert set(out.void_rules) == union
    assert math.isfinite(out.amount)
    assert not guard.reports


@hyp_settings(max_examples=200, deadline=None)
@given(rows=_children)
def test_guard_property_fallback_only_adds_voids(rows) -> None:
    """Whatever a flawed rollup returns, the fallback output carries the full union."""
    guard = RollupGuard(Forgetful(), mode="fallback")
    children = [_child(amount, void) for amount, void in rows]
    out = guard.rollup("root", children)
    assert set(out.void_rules) == {rule for child in children for rule in child.void_rules}


# -- R5: branch_void beats whatever a rollup returns ----------------------------------------
def test_branch_void_beats_custom_rollup() -> None:
    from mactf.coopetition.tree import PeerTree
    from mactf.scoring.hierarchy import HierarchicalScorer
    from mactf.scoring.ledger import Ledger
    from mactf.scoring.pipeline import ScoringPipeline

    def rich_rollup(parent, children):
        # a lying custom rollup: huge positive amount, void set empty (fallback mode repairs the union only)
        return ScorerOutput(kind=ScoringKind.HIERARCHY, amount=1e3)

    tree = PeerTree()
    tree.add_node("peer-a")
    tree.add_node("peer-a/sub-1")
    bad = Ledger(peer="peer-a/sub-1")
    bad.record("P1", magnitude=1.0)
    config = {"REWARD_WEIGHTS": {"r_task": 1.0, "r_epi": 1.5, "r_collab": 1.2, "r_align": 1.0}}
    scorer = HierarchicalScorer(ScoringPipeline(score_config=config), rollup=_Custom(rich_rollup))
    card = scorer.score_tree(tree, {"peer-a/sub-1": bad})
    assert card.peers["peer-a"].branch_void == ("P1",)
    assert card.peers["peer-a"].is_void
    assert card.peers["peer-a"].final_score < 0.0
