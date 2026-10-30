"""Test a peer strategy with no model, no GPU and no peers (document 17 section 11.12).

The same synthetic trainer the strategy probe uses (11.8): uniform legal reports, counted
writes, a tree shaped like the ``TreeView`` it is given.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..components.guards.strategy import StrategyGuard
from ..coopetition.strategy import TreeView
from ..coopetition.strategy_probe import _Synthetic
from ..training.coordination import CoordinationContext, CoordinationState


def synthetic_backend(view: TreeView, void: Iterable[str] = ()) -> _Synthetic:
    """A ``CoordinationBackend`` over ``view`` with no peers: uniform legal reports, counted writes."""
    return _Synthetic(view, frozenset(void))


def sweep(
    strategy: Any, view: TreeView, *, epochs: int = 1, void: Iterable[str] = (), mode: str = "enforce"
) -> list[CoordinationState]:
    """Run ``epochs`` sweeps of ``strategy`` under a guard that covers it; one state per epoch."""
    backend = synthetic_backend(view, void)
    guard = StrategyGuard(mode=mode, scope="all", probe=False)
    name = str(getattr(strategy, "name", None) or type(strategy).__name__)
    states: list[CoordinationState] = []
    for epoch in range(1, epochs + 1):
        strategy.on_epoch_start(backend, epoch)
        state = CoordinationState(epoch)
        for group in reversed(view.groups):                   # deepest first, as StrategyPlan.run
            ctx = CoordinationContext(backend, state, (group,), owner=name, checks=True)
            guard.run(name, strategy, ctx, state, backend=backend)
        states.append(state)
    return states


def written(state: CoordinationState) -> list[str]:
    """The labels of the groups an epoch wrote, in order."""
    return [event.group for event in state.events if event.kind in ("aggregate_root", "aggregate_group")]


__all__ = ["synthetic_backend", "sweep", "written"]
