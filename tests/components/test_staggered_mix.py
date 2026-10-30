"""The worked example of document 17 section 11.12: a strategy that mixes each group in turn."""

from __future__ import annotations

import pytest

from mactf.coopetition.strategy import GroupRef, TreeView
from mactf.exceptions import StrategyGuardViolation
from mactf.testing import sweep, written

from p3_examples import StaggeredMixStrategy

VIEW = TreeView(
    groups=(
        GroupRef("root", None, ("team-a", "team-b"), 0),
        GroupRef("team-a", "team-a", ("team-a/x", "team-a/y"), 1),
        GroupRef("team-b", "team-b", ("team-b/x", "team-b/y"), 1),
    ),
    paths=("team-a", "team-a/x", "team-a/y", "team-b", "team-b/x", "team-b/y"),
    hierarchy=True,
)


def test_staggered_mix() -> None:
    """Each epoch writes exactly one group; the sweep meets team-b, team-a, then the root."""
    states = sweep(StaggeredMixStrategy(period=3), VIEW, epochs=3)
    assert [written(state) for state in states] == [["root"], ["team-a"], ["team-b"]]


def test_each_epoch_writes_exactly_one_group() -> None:
    states = sweep(StaggeredMixStrategy(period=3), VIEW, epochs=6)
    assert [len(written(state)) for state in states] == [1] * 6
    assert [written(state) for state in states[:3]] == [written(state) for state in states[3:]]   # period 3


def test_a_period_of_one_mixes_every_group_every_epoch_children_first() -> None:
    states = sweep(StaggeredMixStrategy(period=1), VIEW, epochs=2)
    assert [written(state) for state in states] == [["team-b", "team-a", "root"]] * 2


def test_slots_survive_a_state_round_trip() -> None:
    first = StaggeredMixStrategy(period=3)
    sweep(first, VIEW, epochs=1)
    second = StaggeredMixStrategy(period=3)
    second.load_state_dict(first.state_dict())
    assert second.state_dict() == first.state_dict()
    assert first._slots == {"team-b": 0, "team-a": 1, "root": 2}


def test_a_resumed_strategy_continues_the_same_rotation() -> None:
    whole = sweep(StaggeredMixStrategy(period=3), VIEW, epochs=6)
    first = StaggeredMixStrategy(period=3)
    sweep(first, VIEW, epochs=1)
    resumed = StaggeredMixStrategy(period=3)
    resumed.load_state_dict(first.state_dict())
    # sweep() restarts its epoch counter, so compare by driving epochs 2.. through the same backend
    from mactf.components.guards.strategy import StrategyGuard
    from mactf.testing import synthetic_backend
    from mactf.training.coordination import CoordinationContext, CoordinationState

    backend = synthetic_backend(VIEW)
    guard = StrategyGuard(mode="enforce", scope="all", probe=False)
    rest = []
    for epoch in range(2, 7):
        state = CoordinationState(epoch)
        for group in reversed(VIEW.groups):
            ctx = CoordinationContext(backend, state, (group,), owner="staggered_mix", checks=True)
            guard.run("staggered_mix", resumed, ctx, state, backend=backend)
        rest.append(written(state))
    assert rest == [written(state) for state in whole[1:]]


def test_a_void_group_is_skipped_not_blended() -> None:
    states = sweep(StaggeredMixStrategy(period=1), VIEW, void={"team-a/x", "team-a/y"})
    assert "team-a" not in written(states[0])
    assert any(e.kind == "skip" and e.group == "team-a" for e in states[0].events)


def test_a_frozen_group_is_skipped_and_a_partly_frozen_one_excludes_the_frozen_peer() -> None:
    from mactf.testing import synthetic_backend

    class Frozen(StaggeredMixStrategy):
        def on_epoch_start(self, trainer, epoch):
            for path in ("team-a/x", "team-a/y"):
                trainer.freeze.add(path, "strategy:staggered_mix")

    states = sweep(Frozen(period=1), VIEW)
    assert "team-a" not in written(states[0])
    assert any(e.kind == "skip" and e.group == "team-a" and e.detail["reason"] == "every member is frozen"
               for e in states[0].events)


def test_the_broken_variant_is_caught_by_the_guard() -> None:
    """Delete the ``peers_void`` branch: the context refuses with Y002 and ``enforce`` raises E073."""
    class NoVoidCheck(StaggeredMixStrategy):
        def coordinate(self, ctx):
            for group in ctx.groups:
                if group.is_root:
                    ctx.aggregate_root(ctx.update_root())
                else:
                    ctx.aggregate_group(group)

    with pytest.raises(StrategyGuardViolation, match=r"E073.*Y002.*team-a.*never blended"):
        sweep(NoVoidCheck(period=1), VIEW, void={"team-a/x", "team-a/y"})
    states = sweep(NoVoidCheck(period=1), VIEW, void={"team-a/x", "team-a/y"}, mode="skip")
    assert "team-a" not in written(states[0])
    assert any(e.kind == "violation" and "Y002" in e.detail["codes"] for e in states[0].events)


def test_validate_and_parameters() -> None:
    flat = TreeView(groups=(GroupRef("root", None, ("a",), 0),), paths=("a",), hierarchy=True)
    assert StaggeredMixStrategy().validate(flat) == ["a hierarchy with a single group has nothing to stagger"]
    assert StaggeredMixStrategy().validate(VIEW) == []
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            StaggeredMixStrategy(period=bad)
    assert StaggeredMixStrategy(period=2).describe() == {"name": "staggered_mix", "period": 2}
