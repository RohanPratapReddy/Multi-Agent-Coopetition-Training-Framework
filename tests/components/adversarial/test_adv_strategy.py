"""Adversarial verification of D2 for peer strategies (doc 17 section 11).

A custom strategy decides who is mixed with whom. Whatever it does, a void peer's parameters
never enter a blend, a frozen peer is never written, a group is written once, the tree keeps its
shape - or the call is refused with a ``Y0xx`` code (``StrategyGuardViolation``) / the write
choke point refuses it (``WeightGuardViolation``). Probes are switched off so the *runtime*
guard, not the build-time probe, is what is attacked.
"""

from __future__ import annotations

import pytest
import torch

from mactf.components import register_peer_strategy
from mactf.coopetition.strategy import PeerStrategy, StrategyTraits
from mactf.coopetition.weights import make_report
from mactf.exceptions import StrategyGuardViolation, WeightGuardViolation

from adv_support import inject_breach
from p2_support import configure
from p3_support import flat, hier

VOID = "peer-1"


def _fabricated(ctx, kappa=1.0, who=None):
    names = sorted(who or ctx.group.members)
    return make_report({n: 1.0 / len(names) for n in names}, names, kappa=kappa)


class _S(PeerStrategy):
    traits = StrategyTraits(aggregates=True, walk="root")

    def coordinate(self, ctx):
        self.attack(ctx)


def _make(name, attack, **traits):
    attrs = {"name": name, "attack": staticmethod(attack)}
    if traits:
        attrs["traits"] = StrategyTraits(**traits)
    return type(f"Adv_{name}", (_S,), attrs)


def _unchecked_fabricated(ctx):
    ctx.unchecked().aggregate_root(_fabricated(ctx))

def _plain_fabricated(ctx):
    ctx.aggregate_root(_fabricated(ctx))

def _private_backend(ctx):
    ctx._backend.aggregate(_fabricated(ctx), update_reference=False)

def _flip_checks_off(ctx):
    ctx.checks = False
    ctx.aggregate_root(_fabricated(ctx))

def _nan_report(ctx):
    report = _fabricated(ctx)
    report.weights[VOID] = float("nan")
    ctx.aggregate_root(report)

def _write_twice(ctx):
    report = ctx.update_root()
    ctx.aggregate_root(report)
    ctx.aggregate_root(report)

def _raise(ctx):
    raise RuntimeError("plugin bug")

ATTACKS = {
    "unchecked_fabricated": _unchecked_fabricated, "plain_fabricated": _plain_fabricated,
    "private_backend": _private_backend, "flip_checks_off": _flip_checks_off,
    "nan_report": _nan_report, "write_twice": _write_twice, "raise": _raise,
}
REFUSED = (StrategyGuardViolation, WeightGuardViolation)


def _flat(settings, tmp_path, name, mode):
    configure(settings, TRAINER={"PEER_STRATEGY": name,
                                 "STRATEGY_GUARD": {"MODE": mode, "PROBE": False},
                                 "WEIGHT_GUARD": {"PROBE": False}})
    trainer = flat(settings, tmp_path, None, epochs=1, peers=3)
    trainer.build()
    inject_breach(trainer, [VOID])
    trainer.blends = []                           # (weights, peers shown) of every blend actually computed
    original = trainer.aggregator.aggregate

    def spy(adapters, report, **kwargs):
        trainer.blends.append((dict(report.weights), tuple(sorted(adapters)), bool(report.fallback)))
        return original(adapters, report, **kwargs)

    trainer.aggregator.aggregate = spy
    return trainer


def _void_never_blended(trainer):
    for weights, shown, fallback in trainer.blends:
        if not fallback:
            assert weights.get(VOID, 0.0) == 0.0, f"the void peer was weighted in a blend: {weights}"


def _params(trainer):
    return {p: peer.adapter.export_parameters().clone() for p, peer in trainer.peers.items()}


@pytest.mark.parametrize("name", sorted(ATTACKS))
def test_enforce_refuses_every_attack_and_nothing_is_written(name, settings, tmp_path, component_overlay):
    register_peer_strategy(name, _make(name, ATTACKS[name]), scope="run")
    trainer = _flat(settings, tmp_path, name, "enforce")
    with pytest.raises(REFUSED):
        trainer.run_epoch(0)
    _void_never_blended(trainer)


@pytest.mark.parametrize("mode", ["skip", "fallback"])
@pytest.mark.parametrize("name", ["unchecked_fabricated", "plain_fabricated", "private_backend",
                                  "flip_checks_off", "nan_report"])
def test_soft_modes_never_blend_the_void_peer(name, mode, settings, tmp_path, component_overlay):
    """In ``skip``/``fallback`` the run goes on; the void peer's parameters are in no blend."""
    register_peer_strategy(name, _make(name, ATTACKS[name]), scope="run")
    trainer = _flat(settings, tmp_path, name, mode)
    try:
        trainer.run_epoch(0)
    except WeightGuardViolation:
        return                                      # the write choke point refused the report: fine
    _void_never_blended(trainer)
    for report in trainer.reports:
        assert report.weights.get(VOID, 0.0) == 0.0, report.weights
    assert all(torch.isfinite(peer.adapter.export_parameters()).all() for peer in trainer.peers.values())


def test_the_write_choke_point_refuses_a_foreign_report_with_a_void_peer(settings, tmp_path, component_overlay):
    """Regression: ``ctx.unchecked()`` / ``ctx.checks = False`` / ``ctx._backend`` skipped Y001."""
    trainer = _flat(settings, tmp_path, "kappa_softmax", "enforce")
    trainer.run_epoch(0)                              # one honest epoch voids the victim
    names = sorted(trainer.peers)
    forged = make_report({n: 1.0 / len(names) for n in names}, names, kappa=1.0)
    before = _params(trainer)
    with pytest.raises(WeightGuardViolation) as caught:
        trainer.aggregate(forged)
    assert any(v.code == "U004" and v.peer == VOID for v in caught.value.violations)
    for path, tensor in before.items():
        assert torch.equal(tensor, trainer.peers[path].adapter.export_parameters())


def test_the_checks_flag_cannot_be_flipped_by_a_strategy():
    from mactf.training.coordination import CoordinationContext, CoordinationState, GroupRef

    ctx = CoordinationContext(object(), CoordinationState(0), (GroupRef("root", None, ("a",), 0),),
                              owner="x", checks=True)
    with pytest.raises(AttributeError):
        ctx.checks = False
    assert ctx.checks is True
