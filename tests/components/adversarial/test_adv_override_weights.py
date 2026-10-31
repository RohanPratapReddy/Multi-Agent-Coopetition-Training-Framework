"""The void peer carries zero weight under a strategy that took a built-in's name (document 17 section 10).

Every attack of ``test_adv_weights`` is registered as ``weight_update:kappa_softmax`` (reached
through a spec with a parameter, F2), ``uniform`` and ``score_weighted`` with the override
flag. Repair, fallback and enforce behave exactly as for a new name; the build-time probe
still refuses what it can see; an honest identity override leaves the breach cost untouched.
"""

from __future__ import annotations

import logging
import math

import pytest
import torch

import custom_kinds as ck
from adv_support import assert_breach_cost_intact, breach_trainer, run_epochs, victim_cards
from mactf.exceptions import ImproperlyConfigured, WeightGuardViolation
from override_support import override_scope, put, w001
from p2_support import configure
from test_adv_weights import ATTACKS, VOID, _Strategy, _void_peer_weight_is_zero

TARGETS = {
    "kappa_softmax": {"type": "kappa_softmax", "params": {"tag": "t"}},
    "uniform": "uniform",
    "score_weighted": "score_weighted",
}


@pytest.fixture(autouse=True)
def _scope(caplog):
    caplog.set_level(logging.WARNING, logger="mactf.components")
    with override_scope():
        yield


def _strategy(attack: str, as_name: str):
    def __init__(self, *, tag: str = "") -> None:
        self.tag = tag

    return type(f"Adv_{attack}_{as_name}", (_Strategy,), {
        "name": as_name, "attack": staticmethod(ATTACKS[attack]), "__init__": __init__,
    })


def _trainer(attack, target, mode, settings, tmp_path, probe=False, scope="custom"):
    put("weight_update", target, _strategy(attack, target))
    configure(settings, COMPONENTS={"PLUGINS": "off"},
              TRAINER={"WEIGHT_UPDATE": TARGETS[target],
                       "WEIGHT_GUARD": {"MODE": mode, "SCOPE": scope, "PROBE": probe}})
    return breach_trainer(settings, tmp_path)


@pytest.mark.parametrize("target", sorted(TARGETS))
@pytest.mark.parametrize("mode", ["repair", "fallback"])
@pytest.mark.parametrize("attack", sorted(ATTACKS))
def test_a_void_peer_never_carries_weight_under_a_builtin_name(attack, mode, target, settings, tmp_path):
    trainer = _trainer(attack, target, mode, settings, tmp_path)
    run_epochs(trainer)
    _void_peer_weight_is_zero(trainer)
    assert all(card.is_void for card in victim_cards(trainer))


@pytest.mark.parametrize("target", sorted(TARGETS))
@pytest.mark.parametrize("attack", sorted(ATTACKS))
def test_enforce_mode_refuses_or_the_invariant_holds(attack, target, settings, tmp_path):
    trainer = _trainer(attack, target, "enforce", settings, tmp_path)
    try:
        run_epochs(trainer)
    except WeightGuardViolation:
        return
    _void_peer_weight_is_zero(trainer)


@pytest.mark.parametrize("target", sorted(TARGETS))
@pytest.mark.parametrize("attack", sorted(ATTACKS))
def test_the_build_time_probe_refuses_or_the_run_is_safe(attack, target, settings, tmp_path):
    try:
        trainer = _trainer(attack, target, "repair", settings, tmp_path, probe=True)
    except ImproperlyConfigured as error:
        assert "E064" in str(error) or "E067" in str(error)
        return
    run_epochs(trainer)
    _void_peer_weight_is_zero(trainer)


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_a_void_weight_is_repaired_and_the_guard_says_so(target, settings, tmp_path, caplog):
    trainer = _trainer("void_weight", target, "fallback", settings, tmp_path)
    with caplog.at_level(logging.WARNING):
        run_epochs(trainer)
    _void_peer_weight_is_zero(trainer)
    assert "W017" in caplog.text


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_a_strategy_cannot_edit_the_peers_through_its_inputs_under_a_builtin_name(target, settings, tmp_path):
    trainer = _trainer("mutate_vectors", target, "repair", settings, tmp_path)
    before = {p: peer.adapter.export_parameters().clone() for p, peer in trainer.peers.items()}
    trainer.update_weights()
    for path, tensor in before.items():
        assert torch.equal(tensor, trainer.peers[path].adapter.export_parameters()), path


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_the_guard_wraps_an_override_even_when_scope_is_custom_only(target, settings, tmp_path):
    """SCOPE=custom exempts the built-ins; an override is not one (F2 aside)."""
    trainer = _trainer("void_weight", target, "repair", settings, tmp_path, scope="custom")
    run_epochs(trainer)
    _void_peer_weight_is_zero(trainer)


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_every_override_is_announced_once(target, settings, tmp_path, caplog):
    _trainer("void_weight", target, "repair", settings, tmp_path)
    assert w001(caplog) == [f"weight_update:{target}"]


# -- the honest override --------------------------------------------------------------------------
def test_an_identity_override_leaves_the_breach_cost_and_the_weights_exactly_as_they_were(settings, tmp_path):
    configure(settings, COMPONENTS={"PLUGINS": "off"},
              TRAINER={"WEIGHT_UPDATE": {"type": "kappa_softmax"}})
    reference = breach_trainer(settings, tmp_path / "ref")
    run_epochs(reference)
    put("weight_update", "kappa_softmax", ck.SameKappa)
    configure(settings, TRAINER={"WEIGHT_UPDATE": TARGETS["kappa_softmax"]})
    trainer = breach_trainer(settings, tmp_path / "over")
    run_epochs(trainer)
    assert_breach_cost_intact(trainer)
    assert [r.weights for r in trainer.reports] == [r.weights for r in reference.reports]
    assert [c.final_score for c in victim_cards(trainer)] == [c.final_score for c in victim_cards(reference)]
    assert all(math.isfinite(w) for r in trainer.reports for w in r.weights.values())


@pytest.mark.parametrize("name, cls", [("entropy_floor", ck.EntropyFloorAverage), ("rank_weighted", ck.RankWeightedAverage)])
def test_the_new_weight_updates_also_leave_the_void_peer_at_zero(name, cls, settings, tmp_path):
    put("weight_update", name, cls)
    configure(settings, COMPONENTS={"PLUGINS": "off"},
              TRAINER={"WEIGHT_UPDATE": name, "WEIGHT_GUARD": {"MODE": "enforce"}})
    trainer = breach_trainer(settings, tmp_path)
    run_epochs(trainer)
    assert_breach_cost_intact(trainer)
    _void_peer_weight_is_zero(trainer)


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_the_attack_really_is_what_the_run_uses(target, settings, tmp_path):
    """Guards against a vacuous pass: the slot holds the attacker, wrapped in the weight guard."""
    from override_support import weight_instance

    trainer = _trainer("void_weight", target, "repair", settings, tmp_path)
    instance = weight_instance(trainer)
    inner = getattr(instance, "inner", getattr(instance, "_inner", instance))
    assert type(instance).__name__ != "KappaSoftmaxAverage"
    assert "Adv_void_weight" in type(inner).__name__, (type(instance), type(inner))

