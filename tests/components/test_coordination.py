"""CoordinationContext, StrategyGuard and the strategy probe (document 17 sections 11.6 to 11.8).

One adversarial test per code Y001-Y008, per mode, and the Hypothesis property
``test_guard_property_strategy_cannot_write``.
"""

from __future__ import annotations

import logging

import pytest
from hypothesis import given, settings as hyp_settings, strategies as st

from mactf.components.guards.strategy import (
    CODES,
    GuardOutcome,
    StrategyGuard,
    Violation,
    topology_digest,
)
from mactf.coopetition.strategy import (
    GroupRef,
    HierarchicalStrategy,
    IndependentStrategy,
    PeerStrategy,
    StrategyTraits,
    TreeView,
    legacy_dispatch,
)
from mactf.coopetition.strategy_probe import _Synthetic, verify_peer_strategy
from mactf.coopetition.weights import make_report
from mactf.exceptions import ImproperlyConfigured, StrategyGuardViolation
from mactf.testing import sweep, synthetic_backend, written
from p3_support import Recording
from mactf.training.coordination import (
    AuditEvent,
    CoordinationContext,
    CoordinationError,
    CoordinationState,
)

VIEW = TreeView(
    groups=(
        GroupRef("root", None, ("team-a", "team-b"), 0),
        GroupRef("team-a", "team-a", ("team-a/x", "team-a/y"), 1),
        GroupRef("team-b", "team-b", ("team-b/x", "team-b/y"), 1),
    ),
    paths=("team-a", "team-a/x", "team-a/y", "team-b", "team-b/x", "team-b/y"),
    hierarchy=True,
)
ROOT, TEAM_A, TEAM_B = VIEW.groups


class Scripted(PeerStrategy):
    """A strategy that does whatever the test says."""

    name = "scripted"
    traits = StrategyTraits(aggregates=True, walk="tree")

    def __init__(self, script):
        self.script = script

    def coordinate(self, ctx):
        self.script(ctx)


def run(script, *, void=(), mode="enforce", scope="all", group=ROOT, backend=None, state=None, strategy=None,
        pre=None):
    """Run ``script`` once under the guard for ``group``. Returns ``(outcome, backend, state)``."""
    backend = backend or Recording(VIEW, void)
    state = state or CoordinationState(1)
    if pre:
        pre(backend, state)
    strategy = strategy or Scripted(script)
    ctx = CoordinationContext(backend, state, (group,), owner="scripted", checks=True)
    guard = StrategyGuard(mode=mode, scope=scope, probe=False)
    outcome = guard.run("scripted", strategy, ctx, state, backend=backend)
    return outcome, backend, state


def codes(error: StrategyGuardViolation):
    return sorted({item.code for item in error.violations})


# -- the verbs that work ----------------------------------------------------------------------------------
def test_a_well_behaved_strategy_passes_and_is_audited() -> None:
    states = sweep(HierarchicalStrategy(), VIEW, epochs=1)
    events = [(e.kind, e.group) for e in states[0].events]
    # deepest first, the root group last; ``StrategyPlan.run`` would compute the root weights before the sweep
    assert events == [("aggregate_group", "team-b"), ("aggregate_group", "team-a"),
                      ("update_root", "root"), ("aggregate_root", "root")]
    assert written(states[0]) == ["team-b", "team-a", "root"]


def test_update_root_is_memoised_and_a_changed_eligibility_is_refused() -> None:
    def script(ctx):
        first = ctx.update_root()
        assert ctx.update_root() is first                                # once per epoch
        assert ctx.update_root(eligible=None) is first
        ctx.update_root(eligible={"team-a": False, "team-b": True})

    with pytest.raises(StrategyGuardViolation) as caught:
        run(script)
    assert codes(caught.value) == ["Y003"]


def test_aggregate_root_with_update_reference_false_is_audited() -> None:
    outcome, backend, state = run(lambda ctx: ctx.aggregate_root(update_reference=False))
    assert outcome.action == "pass" and backend.writes == [("root", False)]
    assert state.events[-1].detail == {"update_reference": False}


def test_record_signal_sharing_and_skip_are_audited_not_counted_as_writes() -> None:
    def script(ctx):
        ctx.record_signal_sharing(ctx.signals)
        ctx.skip(ROOT, "not now")

    outcome, backend, state = run(script)
    assert outcome.action == "pass" and backend.writes == [] and backend.shared
    assert [e.kind for e in state.events] == ["signals", "skip"]


def test_freeze_peer_adds_only_its_own_reason_and_thaw_removes_only_that() -> None:
    def pre(backend, state):
        backend.freeze.add("team-a/x", "optimizer:null")

    def script(ctx):
        assert ctx.freeze_peer("team-a/x") is False                      # already frozen by another reason
        assert ctx.freeze_peer("team-a/y") is True
        assert ctx.thaw_peer("team-a/x") is False                        # the optimizer reason keeps it frozen
        assert ctx.thaw_peer("team-a/y") is True

    outcome, backend, _ = run(script, group=TEAM_A, pre=pre)
    assert outcome.action == "pass"
    assert backend.freeze.reasons("team-a/x") == ("optimizer:null",) and backend.freeze.reasons("team-a/y") == ()


# -- Y001 .. Y008, one adversarial case each ------------------------------------------------------------------
def test_y001_a_report_the_context_did_not_produce_must_be_legal() -> None:
    bad = make_report({"team-a": 3.0, "team-b": 0.0}, ["team-a", "team-b"], kappa=1.0)       # sums to 3
    with pytest.raises(StrategyGuardViolation) as caught:
        run(lambda ctx: ctx.aggregate_root(bad))
    assert codes(caught.value) == ["Y001"] and "not legal" in str(caught.value)
    nan = make_report({"team-a": float("nan"), "team-b": 0.5}, ["team-a", "team-b"], kappa=1.0)
    with pytest.raises(StrategyGuardViolation, match="Y001"):
        run(lambda ctx: ctx.aggregate_root(nan))
    ghost = make_report({"team-a": 0.5, "team-b": 0.5, "ghost": 0.0}, ["team-a", "team-b", "ghost"], kappa=1.0)
    with pytest.raises(StrategyGuardViolation, match="Y001"):
        run(lambda ctx: ctx.aggregate_root(ghost))
    good = make_report({"team-a": 0.5, "team-b": 0.5}, ["team-a", "team-b"], kappa=1.0)
    outcome, backend, _ = run(lambda ctx: ctx.aggregate_root(good))
    assert outcome.action == "pass" and backend.writes == [("root", True)]


def test_y001_a_report_the_context_produced_is_not_re_checked() -> None:
    outcome, backend, _ = run(lambda ctx: ctx.aggregate_root(ctx.update_root()))
    assert outcome.action == "pass"


def test_y002_a_void_group_is_never_blended() -> None:
    with pytest.raises(StrategyGuardViolation) as caught:
        run(lambda ctx: ctx.aggregate_group(TEAM_A), void={"team-a/x", "team-a/y"}, group=TEAM_A)
    assert codes(caught.value) == ["Y002"]
    outcome, backend, _ = run(lambda ctx: ctx.aggregate_group(TEAM_A), void={"team-a/x"}, group=TEAM_A)
    assert outcome.action == "pass"                                     # only fully void groups are refused


def test_test_void_group_not_mixed_nothing_is_written() -> None:
    outcome, backend, state = run(lambda ctx: ctx.aggregate_group(TEAM_A), void={"team-a/x", "team-a/y"},
                                  group=TEAM_A, mode="skip")
    assert outcome.action == "skipped" and backend.writes == [] and state.written == {}


def test_group_written_once() -> None:
    """T5: a group is written at most once per epoch (Y003)."""
    def twice(ctx):
        ctx.aggregate_group(TEAM_A)
        ctx.aggregate_group(TEAM_A)

    outcome, backend, state = run(twice, group=TEAM_A, mode="skip")
    assert outcome.action == "skipped" and [v.code for v in outcome.violations] == ["Y003"]
    assert backend.writes == [("team-a", ("team-a/x", "team-a/y"))] and state.written == {"team-a": 1}


def test_parent_after_child() -> None:
    """T4: a group is never written after its ancestor group (Y004)."""
    def pre(backend, state):
        state.written["root"] = 1                                       # the root group has been written already

    with pytest.raises(StrategyGuardViolation) as caught:
        run(lambda ctx: ctx.aggregate_group(TEAM_A), group=TEAM_A, pre=pre)
    assert codes(caught.value) == ["Y004"] and "ancestor group 'root'" in str(caught.value)


def test_y004_a_deep_ancestor_counts_too() -> None:
    deep = TreeView(
        groups=(GroupRef("root", None, ("a",), 0), GroupRef("a", "a", ("a/b",), 1),
                GroupRef("a/b", "a/b", ("a/b/c", "a/b/d"), 2)),
        paths=("a", "a/b", "a/b/c", "a/b/d"), hierarchy=True,
    )

    def pre(backend, state):
        state.written["root"] = 1

    backend = Recording(deep)
    with pytest.raises(StrategyGuardViolation, match="Y004"):
        run(lambda ctx: ctx.aggregate_group(deep.groups[2]), group=deep.groups[2], backend=backend, pre=pre)


def test_y005_a_frozen_peers_parameters_must_not_change() -> None:
    backend = Recording(VIEW)
    backend.freeze.add("team-a/x", "optimizer:null")                    # strict: it must not be written

    def tamper(ctx):
        backend.digests["team-a/x"] = "overwritten"                     # the strategy reached the parameters directly

    for mode in ("skip", "fallback", "enforce"):                        # hard in every mode
        backend.digests.clear()
        with pytest.raises(StrategyGuardViolation) as caught:
            run(tamper, group=TEAM_A, backend=backend, mode=mode)
        assert codes(caught.value) == ["Y005"] and caught.value.violations[0].peer == "team-a/x"
        assert caught.value.code == "E073"


def test_y005_can_be_switched_off() -> None:
    backend = Recording(VIEW)
    backend.freeze.add("team-a/x", "optimizer:null")
    state = CoordinationState(1)
    ctx = CoordinationContext(backend, state, (TEAM_A,), owner="s", checks=True)
    guard = StrategyGuard(mode="skip", scope="all", check_frozen=False)
    outcome = guard.run("s", Scripted(lambda c: backend.digests.update({"team-a/x": "changed"})), ctx, state,
                        backend=backend)
    assert outcome.action == "pass"


def test_y006_the_tree_structure_must_not_change() -> None:
    backend = Recording(VIEW)

    def grow(ctx):
        backend.tree.add_node("team-a/sneaky")

    def prune(ctx):
        backend.tree.remove_node("team-a/y")

    def mutate_role(ctx):
        backend.tree["team-a/x"].role = "judge"

    for attack in (grow, prune, mutate_role):
        backend = Recording(VIEW)
        with pytest.raises(StrategyGuardViolation) as caught:
            run(attack, group=TEAM_A, backend=backend, mode="skip")
        assert codes(caught.value) == ["Y006"]


def test_y006_ignores_transient_statuses() -> None:
    backend = Recording(VIEW)
    before = topology_digest(backend.tree)
    backend.tree.mark("team-a/x", "frozen")
    backend.tree.mark("team-a/y", "failed")
    assert topology_digest(backend.tree) == before


def test_y007_a_raising_strategy_is_reported_by_mode() -> None:
    def boom(ctx):
        raise KeyError("third party")

    with pytest.raises(StrategyGuardViolation) as caught:
        run(boom, mode="enforce")
    assert codes(caught.value) == ["Y007"] and "KeyError" in str(caught.value)
    outcome, backend, state = run(boom, mode="skip")
    assert outcome.action == "skipped" and backend.writes == []
    assert state.events[-1].kind == "violation" and state.events[-1].detail["codes"] == ["Y007"]


def test_y007_fallback_runs_the_standard_mix_for_an_unwritten_group() -> None:
    def boom(ctx):
        raise RuntimeError("bug")

    outcome, backend, state = run(boom, mode="fallback", group=ROOT)
    assert outcome.action == "fallback" and backend.writes == [("root", True)]
    outcome, backend, state = run(boom, mode="fallback", group=TEAM_A)
    assert backend.writes == [("team-a", ("team-a/x", "team-a/y"))]
    # a strategy that does not aggregate gets no fallback mix
    class Quiet(Scripted):
        traits = StrategyTraits(aggregates=False)

    outcome, backend, _ = run(boom, mode="fallback", strategy=Quiet(boom))
    assert outcome.action == "skipped" and backend.writes == []


def test_fallback_never_mixes_a_void_or_all_frozen_group_or_a_written_one() -> None:
    def boom(ctx):
        raise RuntimeError("bug")

    outcome, backend, _ = run(boom, mode="fallback", group=TEAM_A, void={"team-a/x", "team-a/y"})
    assert backend.writes == [] and outcome.action == "skipped"
    backend = Recording(VIEW)
    backend.freeze.add("team-a/x", "optimizer:null")
    backend.freeze.add("team-a/y", "optimizer:null")
    outcome, backend, _ = run(boom, mode="fallback", group=TEAM_A, backend=backend)
    assert backend.writes == []

    def write_then_boom(ctx):
        ctx.aggregate_group(TEAM_A)
        raise RuntimeError("late")

    outcome, backend, _ = run(write_then_boom, mode="fallback", group=TEAM_A)
    assert len(backend.writes) == 1                                   # the half-finished group is left as it was


def test_y008_verbs_are_scoped_to_the_call() -> None:
    cases = [
        lambda ctx: ctx.aggregate_group(TEAM_B),                       # another group
        lambda ctx: ctx.aggregate_root(),                              # the root, from a team's call
        lambda ctx: ctx.skip(TEAM_B, "x"),
        lambda ctx: ctx.freeze_peer("team-b/x"),                       # a peer of another group
        lambda ctx: ctx.thaw_peer("team-b/x"),
        lambda ctx: ctx.freeze_peer("nobody"),
    ]
    for case in cases:
        with pytest.raises(StrategyGuardViolation) as caught:
            run(case, group=TEAM_A)
        assert codes(caught.value) == ["Y008"]
    with pytest.raises(StrategyGuardViolation, match="Y008"):
        run(lambda ctx: ctx.aggregate_group(ROOT), group=ROOT)         # a child verb on the root group


def test_y003_a_write_that_bypasses_the_context_is_found_by_the_counter() -> None:
    """T2: the context is the only verb set; a direct backend call moves ``write_count`` unaudited."""
    def bypass(ctx):
        ctx._backend.aggregate_group("team-a", ("team-a/x", "team-a/y"))

    outcome, _, state = run(bypass, group=TEAM_A, backend=Recording(VIEW), mode="skip")
    assert outcome.action == "skipped" and outcome.violations[0].code == "Y003"
    assert "did not go through the context" in outcome.violations[0].detail and state.written == {}
    with pytest.raises(StrategyGuardViolation, match="Y003"):
        run(bypass, group=TEAM_A, backend=Recording(VIEW), mode="enforce")


def test_strategy_cannot_write() -> None:
    """T2, end to end: every route a strategy has to a parameter is refused or reported."""
    attacks = {
        "twice": (lambda ctx: (ctx.aggregate_group(TEAM_A), ctx.aggregate_group(TEAM_A)), TEAM_A, "Y003"),
        "foreign group": (lambda ctx: ctx.aggregate_group(TEAM_B), TEAM_A, "Y008"),
        "bad report": (lambda ctx: ctx.aggregate_root(make_report({"team-a": 2.0}, ["team-a", "team-b"], kappa=1)),
                       ROOT, "Y001"),
    }
    for label, (script, group, expected) in attacks.items():
        outcome, backend, state = run(script, group=group, mode="skip")
        assert expected in {v.code for v in outcome.violations}, label
        assert len(backend.writes) <= 1, label                         # at most the one legal write happened
        assert sum(state.written.values()) == len(backend.writes), label


def test_the_context_checks_are_off_when_the_guard_does_not_cover() -> None:
    backend = Recording(VIEW)
    state = CoordinationState(1)
    ctx = CoordinationContext(backend, state, (ROOT,), owner="x", checks=False)
    ctx.aggregate_group(TEAM_B)                                         # out of scope, but unchecked
    ctx.aggregate_group(TEAM_B)                                         # twice
    assert len(backend.writes) == 2
    assert ctx.unchecked().checks is False
    assert CoordinationContext(backend, state, (ROOT,), owner="x", checks=True).unchecked().groups == (ROOT,)


def test_the_context_reads() -> None:
    backend = Recording(VIEW, {"team-a/x"})
    ctx = CoordinationContext(backend, CoordinationState(7), (TEAM_A,), owner="o", checks=True)
    assert ctx.epoch == 7 and ctx.group is TEAM_A and not ctx.includes_root
    assert ctx.peers_void(["team-a/x"]) and not ctx.peers_void(["team-a/x", "team-a/y"])
    assert ctx.freeze_mode == "strict" and ctx.tree is backend.tree
    assert ctx.eligible_for(["team-a/x"]) is None
    backend.freeze.add("team-a/y", "strategy:o")
    assert ctx.eligible_for(["team-a/x", "team-a/y"]) == {"team-a/x": True, "team-a/y": False}
    assert isinstance(ctx.signals, dict)
    assert CoordinationContext(backend, CoordinationState(1), (ROOT,), owner="o", checks=True).includes_root


# -- modes, scopes, settings -----------------------------------------------------------------------------------
def test_scope_custom_leaves_builtins_unguarded_and_all_covers_them() -> None:
    guard = StrategyGuard()
    assert not guard.covers(HierarchicalStrategy()) and not guard.covers(IndependentStrategy())

    class Sub(HierarchicalStrategy):
        pass

    assert guard.covers(Sub()) and guard.covers(Scripted(lambda ctx: None))
    assert StrategyGuard(scope="all").covers(HierarchicalStrategy())
    assert not StrategyGuard(scope="off").covers(Scripted(lambda ctx: None))


@pytest.mark.parametrize("kwargs", [{"mode": "warn"}, {"scope": "some"}])
def test_e070_bad_guard_settings(kwargs) -> None:
    with pytest.raises(ImproperlyConfigured, match="E070"):
        StrategyGuard(**kwargs)


def test_guard_from_settings() -> None:
    guard = StrategyGuard.from_settings({"mode": "Fallback", "SCOPE": "all", "PROBE": False, "CHECK_FROZEN": False})
    assert guard.describe() == {"mode": "fallback", "scope": "all", "probe": False, "check_frozen": False}
    assert StrategyGuard.from_settings(None).describe()["mode"] == "skip"
    with pytest.raises(ImproperlyConfigured, match="E070.*unknown keys"):
        StrategyGuard.from_settings({"MODES": "skip"})


def test_every_code_is_described_and_hard_ones_are_marked() -> None:
    assert set(CODES) == {f"Y00{i}" for i in range(1, 9)}
    assert Violation("Y005", "p", "d").hard and Violation("Y006", None, "d").hard
    assert not any(Violation(code, None, "d").hard for code in ("Y001", "Y002", "Y003", "Y004", "Y007", "Y008"))
    assert str(Violation("Y002", "g", "void")) == "Y002 [g]: void"


def test_the_guard_does_not_swallow_its_own_bugs() -> None:
    class Broken(StrategyGuard):
        def _snapshot(self, backend, state):
            raise RuntimeError("guard bug")

    ctx = CoordinationContext(Recording(VIEW), CoordinationState(1), (ROOT,), owner="x", checks=True)
    with pytest.raises(RuntimeError, match="guard bug"):
        Broken(scope="all").run("x", Scripted(lambda c: None), ctx, ctx.state, backend=ctx._backend)


def test_w030_is_logged_for_a_handled_violation(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="mactf.strategy"):
        run(lambda ctx: (_ for _ in ()).throw(RuntimeError("x")), mode="skip")
    assert any("W030" in record.message and "its outcome for the group stands" in record.message
               for record in caplog.records)


# -- the property (19.7.1) ------------------------------------------------------------------------------------------
_label = st.sampled_from(["root", "team-a", "team-b"])
_peer = st.sampled_from(list(VIEW.paths) + ["nobody"])
_action = st.one_of(
    st.tuples(st.just("agg_group"), _label),
    st.tuples(st.just("agg_group"), _label),
    st.just(("agg_root",)),
    st.tuples(st.just("update_root"), st.booleans()),
    st.tuples(st.just("skip"), _label),
    st.tuples(st.just("freeze"), _peer),
    st.tuples(st.just("thaw"), _peer),
    st.just(("bypass",)),
    st.just(("raise",)),
)


class Attacker(PeerStrategy):
    name = "attacker"
    traits = StrategyTraits(aggregates=True, walk="tree")

    def __init__(self, actions, swallow):
        self.actions = actions
        self.swallow = swallow

    def coordinate(self, ctx):
        groups = {g.label: g for g in VIEW.groups}
        for action in self.actions:
            try:
                kind = action[0]
                if kind == "agg_group":
                    ctx.aggregate_group(groups[action[1]])
                elif kind == "agg_root":
                    ctx.aggregate_root()
                elif kind == "update_root":
                    ctx.update_root(eligible=None if action[1] else {"team-a": False, "team-b": True})
                elif kind == "skip":
                    ctx.skip(groups[action[1]], "attack")
                elif kind == "freeze":
                    ctx.freeze_peer(action[1])
                elif kind == "thaw":
                    ctx.thaw_peer(action[1])
                elif kind == "bypass":
                    ctx._backend.write_count += 1                     # a write the audit never saw
                elif kind == "raise":
                    raise RuntimeError("attack")
            except CoordinationError:
                if not self.swallow:
                    raise


@hyp_settings(max_examples=300, deadline=None)
@given(actions=st.lists(_action, max_size=8), swallow=st.booleans(),
       void=st.sets(st.sampled_from(["team-a/x", "team-a/y", "team-b/x", "team-b/y"])),
       frozen=st.sets(st.sampled_from(list(VIEW.paths)), max_size=3),
       mode=st.sampled_from(["skip", "fallback", "enforce"]))
def test_guard_property_strategy_cannot_write(actions, swallow, void, frozen, mode) -> None:
    """A strategy that tries to write a frozen, void or foreign node: refused, and nothing else changes."""
    backend = Recording(VIEW, void)
    for path in frozen:
        backend.freeze.add(path, "optimizer:null")
    before_reasons = {path: backend.freeze.reasons(path) for path in VIEW.paths}
    before_tree = topology_digest(backend.tree)
    state = CoordinationState(1)
    guard = StrategyGuard(mode=mode, scope="all", probe=False)
    strategy = Attacker(actions, swallow)
    raised = False
    for group in reversed(VIEW.groups):                                  # the sweep: deepest first
        ctx = CoordinationContext(backend, state, (group,), owner="attacker", checks=True)
        try:
            guard.run("attacker", strategy, ctx, state, backend=backend)
        except StrategyGuardViolation:
            raised = True
            break
    # no group is written twice, and every audited write is a real write
    assert all(count <= 1 for count in state.written.values())
    audited = sum(state.written.values())
    unaudited = backend.write_count - audited
    assert len(backend.writes) == audited                                # the counter only moves for "bypass"
    # a fully void child group is never blended
    for label, members in (("team-a", ("team-a/x", "team-a/y")), ("team-b", ("team-b/x", "team-b/y"))):
        if set(members) <= set(void):
            assert state.written.get(label, 0) == 0 and all(w[0] != label for w in backend.writes)
    # children are written before parents, never after
    order = [event.group for event in state.events if event.kind in ("aggregate_group", "aggregate_root")]
    if "root" in order:
        assert order.index("root") == len(order) - 1 or all(label == "root" for label in order[order.index("root"):])
    # the tree is untouched and a strategy only ever moved its own freeze reasons
    assert topology_digest(backend.tree) == before_tree
    for path in VIEW.paths:
        now = set(backend.freeze.reasons(path))
        assert now - set(before_reasons[path]) <= {"strategy:attacker"}
        assert set(before_reasons[path]) - now <= {"strategy:attacker"}
    # the audit and the guard agree: an unaudited write is reported (or the run stopped hard)
    if unaudited > 0 and not raised:
        assert any(event.kind == "violation" for event in state.events)


@hyp_settings(max_examples=100, deadline=None)
@given(actions=st.lists(_action, min_size=1, max_size=6))
def test_guard_property_enforce_never_lets_a_violation_pass_silently(actions) -> None:
    backend = Recording(VIEW)
    state = CoordinationState(1)
    guard = StrategyGuard(mode="enforce", scope="all", probe=False)
    strategy = Attacker(actions, swallow=False)
    for group in reversed(VIEW.groups):
        ctx = CoordinationContext(backend, state, (group,), owner="attacker", checks=True)
        try:
            outcome = guard.run("attacker", strategy, ctx, state, backend=backend)
        except StrategyGuardViolation:
            return                                                       # it raised: the contract held
        assert outcome.violations == () and outcome.action == "pass"
        assert not any(event.kind == "violation" for event in state.events)


# -- the probe ---------------------------------------------------------------------------------------------------------
class Good(PeerStrategy):
    name = "good"
    traits = StrategyTraits(aggregates=True, walk="tree")

    def coordinate(self, ctx):
        for group in ctx.groups:
            if group.is_root:
                ctx.aggregate_root(ctx.update_root())
            elif not ctx.peers_void(group.members):
                ctx.aggregate_group(group)


def test_the_probe_accepts_a_good_strategy_and_the_builtins() -> None:
    verify_peer_strategy(lambda: Good(), slot="TRAINER.PEER_STRATEGY", view=VIEW)
    for builtin in (HierarchicalStrategy, IndependentStrategy):
        verify_peer_strategy(lambda b=builtin: b(), slot="x", view=VIEW)


@pytest.mark.parametrize("label,script,match", [
    ("blends a void group", lambda ctx: ctx.aggregate_group(ctx.group) if not ctx.group.is_root else None, "void.*Y002"),
    ("writes twice", lambda ctx: (ctx.aggregate_root(), ctx.aggregate_root()) if ctx.group.is_root else None, "Y003"),
    ("raises", lambda ctx: (_ for _ in ()).throw(ValueError("nope")), "ValueError: nope"),
    ("outside scope", lambda ctx: ctx.aggregate_group(VIEW.groups[1]) if ctx.group.label != "team-a" else None, "Y008"),
])
def test_the_probe_catches_misbehaviour_at_build(label, script, match) -> None:
    class Bad(PeerStrategy):
        name = "bad"
        traits = StrategyTraits(aggregates=True, walk="tree")

        def coordinate(self, ctx):
            script(ctx)

    with pytest.raises(ImproperlyConfigured, match=f"E074.*TRAINER.PEER_STRATEGY failed its probe.*{match}"):
        verify_peer_strategy(lambda: Bad(), slot="TRAINER.PEER_STRATEGY", view=VIEW)


def test_the_probe_rejects_a_strategy_that_defines_nothing_and_a_bad_describe() -> None:
    class Nothing(PeerStrategy):
        name = "nothing"

    with pytest.raises(ImproperlyConfigured, match="E074.*defines neither coordinate"):
        verify_peer_strategy(lambda: Nothing(), slot="s", view=VIEW)

    class BadDescribe(Good):
        def describe(self):
            return {"x": object()}

    with pytest.raises(ImproperlyConfigured, match="E074.*describe\\(\\) is not JSON"):
        verify_peer_strategy(lambda: BadDescribe(), slot="s", view=VIEW)


def test_the_probe_checks_that_a_freezing_strategy_releases_its_peers_in_detach() -> None:
    class Leaky(Good):
        name = "leaky"
        traits = StrategyTraits(aggregates=True, freezes=True, walk="tree")

        def coordinate(self, ctx):
            for group in ctx.groups:
                if not group.is_root:
                    ctx.freeze_peer(group.members[0])
            super().coordinate(ctx)

    with pytest.raises(ImproperlyConfigured, match="E074.*detach\\(\\) left"):
        verify_peer_strategy(lambda: Leaky(), slot="s", view=VIEW)

    class Tidy(Leaky):
        name = "tidy"

        def detach(self, host):
            host.freeze.clear_reason("strategy:tidy")

    verify_peer_strategy(lambda: Tidy(), slot="s", view=VIEW)


def test_the_probe_checks_state_round_trips_e077() -> None:
    class NotJson(Good):
        def state_dict(self):
            return {"x": object()}

    with pytest.raises(ImproperlyConfigured, match="E077.*not JSON-serialisable"):
        verify_peer_strategy(lambda: NotJson(), slot="s", view=VIEW)

    class Drifts(Good):
        def __init__(self):
            self.n = 0

        def state_dict(self):
            return {"n": self.n}

        def load_state_dict(self, payload):
            self.n = payload["n"] + 1

    with pytest.raises(ImproperlyConfigured, match="E077.*changed across"):
        verify_peer_strategy(lambda: Drifts(), slot="s", view=VIEW)

    class Rejects(Good):
        def state_dict(self):
            return {"n": 1}

        def load_state_dict(self, payload):
            raise ValueError("no")

    with pytest.raises(ImproperlyConfigured, match="E077.*rejected its own"):
        verify_peer_strategy(lambda: Rejects(), slot="s", view=VIEW)


def test_the_probe_runs_a_legacy_strategy_through_legacy_dispatch() -> None:
    class Legacy(PeerStrategy):
        name = "legacy"
        traits = StrategyTraits(aggregates=True)

        def on_epoch_end(self, trainer, epoch, report):
            pass

    verify_peer_strategy(lambda: Legacy(), slot="s", view=VIEW)
    flat_view = TreeView(groups=(GroupRef("root", None, ("a", "b"), 0),), paths=("a", "b"), hierarchy=False)
    verify_peer_strategy(lambda: Good(), slot="s", view=flat_view)           # no child group: no void case


def test_testing_helpers() -> None:
    backend = synthetic_backend(VIEW, void=["team-a/x"])
    assert backend.peers_void(["team-a/x"]) and not backend.peers_void(["team-a/y"])
    assert backend.legal_report(make_report({"a": 0.5, "b": 0.5}, ["a", "b"], kappa=1.0), None, ["a", "b"]) == []
    assert backend.legal_report(make_report({"a": 2.0, "b": 0.5}, ["a", "b"], kappa=1.0), None, ["a", "b"])
    states = sweep(Good(), VIEW, epochs=2)
    assert len(states) == 2 and written(states[0]) == ["team-b", "team-a", "root"]
    with pytest.raises(StrategyGuardViolation):
        sweep(Scripted(lambda ctx: ctx.aggregate_group(VIEW.groups[1])), VIEW, mode="enforce")
    assert AuditEvent(0, "k", "g").members == ()


def test_ineligible_gets_zero() -> None:
    """T3: eligibility can only shrink. An excluded peer gets exactly 0; a report that gives it weight is Y001 (U004)."""
    outcome, backend, state = run(lambda ctx: ctx.aggregate_group(TEAM_A, eligible={"team-a/x": False}),
                                  group=TEAM_A, mode="enforce")
    assert outcome.action == "pass"
    report = backend._report(["team-a/x", "team-a/y"], {"team-a/x": False})
    assert report.weights["team-a/x"] == 0.0 and report.weights["team-a/y"] == pytest.approx(1.0)

    class Strict(Recording):
        """A backend whose legality check knows that ``team-a`` is ineligible (as the trainer's does)."""

        def legal_report(self, report, parent, members):
            from mactf.coopetition.weight_guard import WeightGuard
            from mactf.coopetition.weights import WeightTraits
            found = WeightGuard().inspect(report, names=sorted(members),
                                          eligible={m: m != "team-a" for m in members},
                                          floor=0.0, cap=1.0, traits=WeightTraits())
            return [f"{item.code}: {item.detail}" for item in found]

    adds_back = make_report({"team-a": 0.5, "team-b": 0.5}, ["team-a", "team-b"], kappa=1.0)
    with pytest.raises(StrategyGuardViolation) as caught:
        run(lambda ctx: ctx.aggregate_root(adds_back), backend=Strict(VIEW))
    assert codes(caught.value) == ["Y001"] and "U004" in str(caught.value)
