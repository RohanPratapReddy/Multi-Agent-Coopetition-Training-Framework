"""The build-time probe of a weight strategy (document 17 section 10.6)."""

from __future__ import annotations

import json
from typing import Any, Callable

import torch

from ..exceptions import ComplianceError, ImproperlyConfigured
from .report import AggregationReport
from .weight_guard import CODES, Violation, WeightGuard
from .weights import traits_of

_NAMES = ("p0", "p1", "p2", "p3")
_SPREAD = {"p0": 0.9, "p1": 0.1, "p2": -0.4, "p3": 0.3}
_TIES = {name: 5.0 for name in _NAMES}
_NEGATIVE = {name: -3.0 for name in _NAMES}

#: ``(label, scores, eligible)``. Chosen to cover: spread, exact ties, all-negative scores,
#: one void peer, two void peers.
_CASES: tuple[tuple[str, dict[str, float], dict[str, bool]], ...] = (
    ("spread", _SPREAD, {n: True for n in _NAMES}),
    ("ties", _TIES, {n: True for n in _NAMES}),
    ("negative", _NEGATIVE, {n: True for n in _NAMES}),
    ("one_void", _SPREAD, {"p0": True, "p1": False, "p2": True, "p3": True}),
    ("two_void", _SPREAD, {"p0": True, "p1": False, "p2": False, "p3": True}),
)


def verify_weight_update(
    make: Callable[[], Any],
    *,
    floor: float,
    cap: float,
    tolerance: float,
    slot: str,
    mode: str = "repair",
) -> None:
    """Probe ``make()``'s strategy; raise ``E064`` (output) or ``E067`` (state) with every finding.

    ``mode`` is the guard's ``MODE``. The framework owns the bounds (invariant A3), so a
    strategy that does not know ``WEIGHT_FLOOR`` and ``WEIGHT_CAP`` (a plain softmax, the
    document's own ``ema_softmax``) is not at fault when a participant lands outside them
    and the weights otherwise sum to 1: under ``repair`` that is exactly what the guard
    projects, so the probe does not report it. Under ``fallback`` and ``enforce`` the same
    output would be replaced or refused every epoch, so it is a finding there. A vector that
    does not sum to 1, a weight on a void peer and every hard violation are findings in
    every mode.

    ``make`` returns a *fresh* instance each call, so a stateful strategy starts clean in
    every case and the probe leaves nothing behind. A case whose bounds are infeasible for its
    participant count (``cap * n < 1`` or ``floor * n > 1``) is skipped: the aggregator
    itself refuses those, and the strategy is not at fault.
    """
    inspector = WeightGuard(mode="fallback", scope="all", tolerance=tolerance)
    findings: list[str] = []
    vectors = {name: torch.full((8,), float(index)) for index, name in enumerate(_NAMES)}
    for label, scores, eligible in _CASES:
        participants = [name for name in _NAMES if eligible[name]]
        if cap * len(participants) < 1.0 - 1e-9 or floor * len(participants) > 1.0 + 1e-9:
            continue
        strategy = make()
        traits = traits_of(strategy)
        if not traits.aggregates:
            return
        try:
            # Copies: ``_CASES`` is module state, and a strategy that edits what it is handed
            # (``eligible[void] = True``) must not rewrite the cases every later probe uses.
            report = strategy.apply(
                {name: vector.clone() for name, vector in vectors.items()}, dict(scores),
                kappa=1.0, level=0, eligible=dict(eligible), confidence=None,
            )
        except ComplianceError:
            continue                    # refusing a case it cannot serve is allowed
        except Exception as error:  # noqa: BLE001 - reported as a finding
            findings.append(f"{label}: {Violation('U007', None, f'{type(error).__name__}: {error}')}")
            continue
        if not isinstance(report, AggregationReport):
            findings.append(f"{label}: apply() returned {type(report).__name__}, not an AggregationReport")
            continue
        for item in inspector.inspect(
            report, names=_NAMES, eligible=eligible, floor=floor, cap=cap, traits=traits
        ):
            if mode == "repair" and item.code == "U005" and item.peer is not None:
                continue                      # a bound: the guard projects it (see the docstring)
            findings.append(f"{label}: {item} ({CODES[item.code]})")
    if findings:
        raise ImproperlyConfigured(f"E064: {slot} failed its probe: " + "; ".join(findings))
    _verify_state(make, slot=slot, vectors=vectors)


def _verify_state(make: Callable[[], Any], *, slot: str, vectors: dict[str, torch.Tensor]) -> None:
    """A strategy with ``state_dict`` must serialise to JSON and reload to the same state."""
    probe = make()
    read = getattr(probe, "state_dict", None)
    if not callable(read):
        return
    problems: list[str] = []
    try:
        probe.apply(
            {name: vector.clone() for name, vector in vectors.items()}, dict(_SPREAD),
            kappa=1.0, level=0,
            eligible={name: True for name in _NAMES}, confidence=None,
        )
    except Exception:  # noqa: BLE001 - output problems were reported above
        pass
    try:
        text = json.dumps(read(), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ImproperlyConfigured(
            f"E067: {slot}: state_dict() is not JSON-serialisable ({error}); checkpoints store it as JSON"
        ) from error
    fresh = make()
    load = getattr(fresh, "load_state_dict", None)
    if not callable(load):
        problems.append("defines state_dict() but no load_state_dict()")
    else:
        try:
            load(json.loads(text))
            again = json.dumps(fresh.state_dict(), sort_keys=True, allow_nan=False)
        except Exception as error:  # noqa: BLE001
            problems.append(f"load_state_dict() rejected its own state_dict(): {error}")
        else:
            if again != text:
                problems.append("state_dict() changed across a load_state_dict() round trip")
    if problems:
        raise ImproperlyConfigured(f"E067: {slot}: " + "; ".join(problems))
