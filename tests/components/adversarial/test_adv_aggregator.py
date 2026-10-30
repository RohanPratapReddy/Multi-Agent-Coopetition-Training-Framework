"""Adversarial verification of D2 for ``aggregator`` components (doc 17 section 10).

A custom ``Aggregator`` may change *how* weights are made and how parameters are blended, but
the trainer hands its ``aggregate`` only the peers that carry weight, and refuses a blend that is
not finite. The tests run a real trainer in which ``peer-1`` breaches (so it is void).
"""

from __future__ import annotations

import pytest
import torch

from mactf.components import register_aggregator
from mactf.coopetition.aggregation import AggregateResult, Aggregator
from mactf.exceptions import ComplianceError, ContractError

from adv_support import breach_trainer, run_epochs
from p2_support import configure

VOID = "peer-1"


class _IncludesEveryone(Aggregator):
    """Ignores the report's zero weights and averages every adapter it is given."""

    seen: list[tuple[str, ...]] = []

    def aggregate(self, adapters, report, *, reference=None, device="cpu"):
        type(self).seen.append(tuple(sorted(adapters)))
        vectors = [adapter.export_parameters().to(device=device) for adapter in adapters.values()]
        return AggregateResult(parameters=torch.stack(vectors).mean(0), report=report)


class _NaNBlend(Aggregator):
    def aggregate(self, adapters, report, *, reference=None, device="cpu"):
        result = super().aggregate(adapters, report, reference=reference, device=device)
        result.parameters = result.parameters * float("nan")
        return result


class _InfBlend(Aggregator):
    def aggregate(self, adapters, report, *, reference=None, device="cpu"):
        result = super().aggregate(adapters, report, reference=reference, device=device)
        result.parameters = result.parameters * float("inf")
        return result


class _ReportSwap(Aggregator):
    """Computes weights honestly, then hands ``aggregate`` a different report (void peer weighted)."""

    def aggregate(self, adapters, report, *, reference=None, device="cpu"):
        report.weights = {name: 1.0 / len(report.weights) for name in report.weights}
        return super().aggregate(adapters, report, reference=reference, device=device)


class _WrongShape(Aggregator):
    def aggregate(self, adapters, report, *, reference=None, device="cpu"):
        result = super().aggregate(adapters, report, reference=reference, device=device)
        result.parameters = result.parameters[:-1]
        return result


def _trainer(cls, name, settings, tmp_path):
    register_aggregator(name, cls, scope="run")
    configure(settings, TRAINER={"AGGREGATOR": name})
    return breach_trainer(settings, tmp_path)


def test_a_custom_aggregate_is_not_shown_the_void_peer(settings, tmp_path, component_overlay):
    _IncludesEveryone.seen = []
    trainer = _trainer(_IncludesEveryone, "adv_everyone", settings, tmp_path)
    run_epochs(trainer)
    assert _IncludesEveryone.seen, "the custom aggregate was never called"
    assert all(VOID not in names for names in _IncludesEveryone.seen), _IncludesEveryone.seen


@pytest.mark.parametrize("cls", [_NaNBlend, _InfBlend, _WrongShape])
def test_a_blend_that_is_not_finite_or_not_the_right_size_writes_nothing(cls, settings, tmp_path, component_overlay):
    trainer = _trainer(cls, f"adv_{cls.__name__.lower()}", settings, tmp_path)
    trainer.build()
    before = {path: peer.adapter.export_parameters().clone() for path, peer in trainer.peers.items()}
    with pytest.raises(Exception) as caught:
        run_epochs(trainer)
    assert isinstance(caught.value, (ComplianceError, ContractError, ValueError, RuntimeError))
    for path, peer in trainer.peers.items():
        assert torch.isfinite(peer.adapter.export_parameters()).all(), f"{path} was poisoned"


def test_an_aggregate_that_rewrites_the_report_to_weight_the_void_peer_is_refused(settings, tmp_path, component_overlay):
    trainer = _trainer(_ReportSwap, "adv_report_swap", settings, tmp_path)
    with pytest.raises(ComplianceError, match="no adapter registered for peer 'peer-1'"):
        run_epochs(trainer)
    for peer in trainer.peers.values():
        assert torch.isfinite(peer.adapter.export_parameters()).all()
