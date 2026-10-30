"""The build-time probe of a peer strategy (document 17 section 11.8)."""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Sequence

from ..components.guards.strategy import StrategyGuard
from ..exceptions import ImproperlyConfigured, StrategyGuardViolation
from ..training.coordination import CoordinationContext, CoordinationError, CoordinationState
from ..training.freeze import FreezeSet
from .report import AggregationReport
from .strategy import (
    TreeView,
    coordinates_via_context,
    defines_nothing,
    legacy_dispatch,
    traits_of,
)
from .tree import PeerTree
from .weight_guard import WeightGuard
from .weights import WeightTraits, make_report


class _Synthetic:
    """A trainer with a tree and nothing else. Implements ``CoordinationBackend``."""

    def __init__(self, view: TreeView, void: frozenset[str]) -> None:
        size = len(view.paths) + 1
        self.tree = PeerTree(max_depth=max(8, size), max_nodes=max(64, size))
        for path in view.paths:                              # root-first: parents precede children
            self.tree.add_node(path)
        self.freeze = FreezeSet("strict")
        self.freeze.bind(self.tree)
        self.signal_by_peer: dict[str, float] = {
            path: 0.1 * (index + 1) for index, path in enumerate(view.paths)
        }
        self.write_count = 0
        self.void = set(void)
        self.shared: list[dict[str, float]] = []
        self._inspector = WeightGuard(mode="fallback", scope="all")

    def _report(self, names: Sequence[str], eligible: Mapping[str, bool] | None) -> AggregationReport:
        flags = {name: bool((eligible or {}).get(name, True)) for name in names}
        live = [name for name in names if flags[name]]
        share = 1.0 / len(live) if live else 0.0
        weights = {name: (share if flags[name] else 0.0) for name in names}
        return make_report(weights, names, kappa=1.0, eligible=flags, fallback=not live)

    def update_weights(self, eligible: Mapping[str, bool] | None = None) -> AggregationReport:
        return self._report(list(self.tree.root_paths), eligible)

    def aggregate(self, report: Any = None, *, update_reference: bool = True) -> Any:
        self.write_count += 1
        return report

    def aggregate_group(self, parent: str | None, members: Sequence[str], *,
                        eligible: Mapping[str, bool] | None = None) -> AggregationReport:
        self.write_count += 1
        return self._report(list(members), eligible)

    def peers_void(self, members: Sequence[str]) -> bool:
        return bool(members) and all(member in self.void for member in members)

    def legal_report(self, report: Any, parent: str | None, members: Sequence[str]) -> list[str]:
        found = self._inspector.inspect(
            report, names=sorted(members), eligible={name: True for name in members},
            floor=0.0, cap=1.0, traits=WeightTraits(may_fallback=True),
        )
        return [str(item) for item in found]

    def record_signal_sharing(self, signals_map: Mapping[str, float], *, epoch: int) -> None:
        self.shared.append(dict(signals_map))

    def peer_digest(self, path: str) -> str:
        return "synthetic"


def _cases(view: TreeView) -> list[tuple[str, frozenset[str]]]:
    cases: list[tuple[str, frozenset[str]]] = [("calm", frozenset())]
    voided = frozenset(path for group in view.groups if not group.is_root for path in group.members)
    if voided:
        cases.append(("void", voided))
    return cases


def _sweep(instance: Any, backend: _Synthetic, view: TreeView, epoch: int, guard: StrategyGuard,
           name: str) -> None:
    instance.on_epoch_start(backend, epoch)
    if not coordinates_via_context(instance):
        legacy_dispatch(backend, epoch, instance)
        return
    state = CoordinationState(epoch)
    for group in reversed(view.groups):                       # deepest first, as StrategyPlan.run
        ctx = CoordinationContext(backend, state, (group,), owner=name, checks=True)
        guard.run(name, instance, ctx, state, backend=backend)


def verify_peer_strategy(make: Callable[[], Any], *, slot: str, view: TreeView) -> None:
    """Probe ``make()``'s strategy over ``view``. ``E074`` for behaviour, ``E077`` for state.

    ``make`` returns a fresh instance each call, so the probe leaves nothing behind.
    """
    first = make()
    if defines_nothing(first):
        raise ImproperlyConfigured(
            f"E074: {slot}: {type(first).__name__} defines neither coordinate(ctx) nor on_epoch_end(); "
            "it would do nothing"
        )
    name = str(getattr(first, "name", None) or type(first).__name__)
    findings: list[str] = []
    try:
        json.dumps(first.describe(), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        findings.append(f"describe() is not JSON-serialisable ({error})")

    guard = StrategyGuard(mode="enforce", scope="all", probe=False)
    used: Any = None
    for case, void in _cases(view):
        instance = make()
        backend = _Synthetic(view, void)
        for epoch in (1, 2):
            try:
                _sweep(instance, backend, view, epoch, guard, name)
            except StrategyGuardViolation as error:
                findings.append(f"{case}, epoch {epoch}: " + "; ".join(str(item) for item in error.violations))
                break
            except CoordinationError as error:
                findings.append(f"{case}, epoch {epoch}: {error}")
                break
            except Exception as error:  # noqa: BLE001 - user code
                findings.append(f"{case}, epoch {epoch}: {type(error).__name__}: {error}")
                break
        if case == "calm":
            used = instance
            if traits_of(instance).freezes:
                try:
                    instance.detach(backend)
                except Exception as error:  # noqa: BLE001
                    findings.append(f"detach() raised {type(error).__name__}: {error}")
                left = [path for path in backend.freeze
                        if any(reason.startswith("strategy:") for reason in backend.freeze.reasons(path))]
                if left:
                    findings.append(f"traits.freezes is set but detach() left {sorted(left)} frozen")
    if findings:
        raise ImproperlyConfigured(f"E074: {slot} failed its probe: " + "; ".join(findings))
    _verify_state(make, used, slot=slot)


def _verify_state(make: Callable[[], Any], used: Any, *, slot: str) -> None:
    """``state_dict`` must be JSON and reload into a fresh instance to the same state."""
    read = getattr(used, "state_dict", None)
    if not callable(read):
        return
    try:
        text = json.dumps(read(), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ImproperlyConfigured(
            f"E077: {slot}: state_dict() is not JSON-serialisable ({error}); checkpoints store it as JSON"
        ) from error
    fresh = make()
    load = getattr(fresh, "load_state_dict", None)
    if not callable(load):
        raise ImproperlyConfigured(f"E077: {slot}: defines state_dict() but no load_state_dict()")
    try:
        load(json.loads(text))
        again = json.dumps(fresh.state_dict(), sort_keys=True, allow_nan=False)
    except Exception as error:  # noqa: BLE001
        raise ImproperlyConfigured(
            f"E077: {slot}: load_state_dict() rejected its own state_dict(): {error}"
        ) from error
    if again != text:
        raise ImproperlyConfigured(f"E077: {slot}: state_dict() changed across a load_state_dict() round trip")
