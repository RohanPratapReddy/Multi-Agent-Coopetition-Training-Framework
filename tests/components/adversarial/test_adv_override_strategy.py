"""A peer strategy under a built-in's name is judged like any other (document 17 section 11).

The attacks of ``test_adv_strategy`` are registered as ``peer_strategy:kappa_softmax``,
``independent`` and ``cooperative`` with the override flag and reached through the
configuration. The runtime guard (Y001-Y008, E073) and the write choke point still hold; the
build-time probe (E074) still refuses what it can see.
"""

from __future__ import annotations

import logging

import pytest
import torch

import custom_kinds as ck
from adv_support import inject_breach
from mactf.coopetition.strategy import PeerStrategy, StrategyTraits
from mactf.exceptions import ImproperlyConfigured, StrategyGuardViolation, WeightGuardViolation
from override_support import override_scope, put, w001
from p2_support import configure
from p3_support import flat
from test_adv_strategy import ATTACKS, REFUSED, VOID, _make, _params, _void_never_blended

TARGETS = ("kappa_softmax", "independent", "cooperative")


@pytest.fixture(autouse=True)
def _scope(caplog):
    caplog.set_level(logging.WARNING, logger="mactf.components")
    with override_scope():
        yield


def _flat(settings, tmp_path, target, mode, *, probe=False):
    configure(settings, COMPONENTS={"PLUGINS": "off"},
              TRAINER={"PEER_STRATEGY": target,
                       "STRATEGY_GUARD": {"MODE": mode, "PROBE": probe},
                       "WEIGHT_GUARD": {"PROBE": False}})
    trainer = flat(settings, tmp_path, None, epochs=1, peers=3)
    trainer.build()
    inject_breach(trainer, [VOID])
    trainer.blends = []
    original = trainer.aggregator.aggregate

    def spy(adapters, report, **kwargs):
        trainer.blends.append((dict(report.weights), tuple(sorted(adapters)), bool(report.fallback)))
        return original(adapters, report, **kwargs)

    trainer.aggregator.aggregate = spy
    return trainer


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("attack", sorted(ATTACKS))
def test_enforce_refuses_every_attack_under_a_builtin_name(attack, target, settings, tmp_path):
    put("peer_strategy", target, _make(target, ATTACKS[attack]))
    trainer = _flat(settings, tmp_path, target, "enforce")
    with pytest.raises(REFUSED):
        trainer.run_epoch(0)
    _void_never_blended(trainer)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("mode", ["skip", "fallback"])
@pytest.mark.parametrize("attack", ["unchecked_fabricated", "plain_fabricated", "private_backend",
                                    "flip_checks_off", "nan_report"])
def test_soft_modes_never_blend_the_void_peer_under_a_builtin_name(attack, mode, target, settings, tmp_path):
    put("peer_strategy", target, _make(target, ATTACKS[attack]))
    trainer = _flat(settings, tmp_path, target, mode)
    try:
        trainer.run_epoch(0)
    except WeightGuardViolation:
        return
    _void_never_blended(trainer)
    for report in trainer.reports:
        assert report.weights.get(VOID, 0.0) == 0.0, report.weights
    assert all(torch.isfinite(peer.adapter.export_parameters()).all() for peer in trainer.peers.values())


@pytest.mark.parametrize("target", TARGETS)
def test_the_override_is_announced_once_and_is_the_default_instance(target, settings, tmp_path, caplog):
    cls = _make(target, ATTACKS["raise"])
    put("peer_strategy", target, cls)
    trainer = _flat(settings, tmp_path, target, "skip")
    inner = getattr(trainer.strategy_plan.default_instance(), "inner",
                    trainer.strategy_plan.default_instance())
    assert isinstance(inner, cls), type(inner)
    assert w001(caplog) == [f"peer_strategy:{target}"]


# -- the build-time probe ------------------------------------------------------------------------------
def _bad(target, script):
    class Bad(PeerStrategy):
        name = target
        traits = StrategyTraits(aggregates=True, walk="tree")

        def coordinate(self, ctx):
            script(ctx)

    return Bad


PROBE_CASES = [
    pytest.param(lambda ctx: (ctx.aggregate_root(), ctx.aggregate_root()) if ctx.group.is_root else None,
                 "Y003", id="writes-twice"),
    pytest.param(lambda ctx: (_ for _ in ()).throw(ValueError("nope")), "ValueError: nope", id="raises"),
]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("script, match", PROBE_CASES)
def test_e074_the_probe_refuses_a_misbehaving_override_at_build(script, match, target, settings, tmp_path):
    put("peer_strategy", target, _bad(target, script))
    configure(settings, COMPONENTS={"PLUGINS": "off"},
              TRAINER={"PEER_STRATEGY": target, "STRATEGY_GUARD": {"MODE": "enforce", "PROBE": True}})
    with pytest.raises(ImproperlyConfigured, match=f"E074.*{match}"):
        flat(settings, tmp_path, None, epochs=1, peers=3).build()


def test_e074_a_strategy_that_defines_nothing_is_refused_under_a_builtin_name(settings, tmp_path):
    class Nothing(PeerStrategy):
        name = "kappa_softmax"

    put("peer_strategy", "kappa_softmax", Nothing)
    configure(settings, COMPONENTS={"PLUGINS": "off"},
              TRAINER={"PEER_STRATEGY": "kappa_softmax", "STRATEGY_GUARD": {"PROBE": True}})
    with pytest.raises(ImproperlyConfigured, match="E074|E072"):
        flat(settings, tmp_path, None, epochs=1, peers=3).build()


def test_the_probe_off_is_the_only_way_a_bad_override_reaches_the_run(settings, tmp_path):
    """The same bad override with ``PROBE: false`` builds; the *runtime* guard is then the defence."""
    put("peer_strategy", "kappa_softmax", _bad("kappa_softmax", PROBE_CASES[0].values[0]))
    trainer = _flat(settings, tmp_path, "kappa_softmax", "enforce", probe=False)
    assert trainer.strategy_plan.default_instance() is not None


# -- the honest overrides -------------------------------------------------------------------------------
def test_an_identity_strategy_override_leaves_the_void_peer_out(settings, tmp_path):
    put("peer_strategy", "kappa_softmax", ck.SameKappaStrategy)
    trainer = _flat(settings, tmp_path, "kappa_softmax", "enforce", probe=True)
    trainer.run_epoch(0)
    _void_never_blended(trainer)
    assert trainer.reports and all(r.weights.get(VOID, 0.0) == 0.0 for r in trainer.reports)


def test_a_new_name_strategy_is_held_to_the_same_standard(settings, tmp_path):
    from test_adv_strategy import _make as make_new

    put("peer_strategy", "zz_new", make_new("zz_new", ATTACKS["plain_fabricated"]))
    trainer = _flat(settings, tmp_path, "zz_new", "enforce")
    with pytest.raises(StrategyGuardViolation):
        trainer.run_epoch(0)
