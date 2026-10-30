"""Adversarial verification of D2 for weight_update strategies and aggregators (doc 17 section 10).

A custom strategy runs inside a real trainer that sees a breach by ``peer-1``. Whatever the
strategy returns, mutates or raises, the void peer must carry zero weight and the trainer must
log the documented fallback (``W017``) or raise ``WeightGuardViolation`` (``mode: enforce``).
"""

from __future__ import annotations

import math

import pytest
import torch

from mactf.components import register_aggregator, register_weight_update
from mactf.coopetition.aggregation import Aggregator
from mactf.coopetition.report import AggregationReport
from mactf.coopetition.weights import WeightStrategy, make_report
from mactf.exceptions import ImproperlyConfigured, WeightGuardViolation

from adv_support import breach_trainer, run_epochs, victim_cards
from p2_support import configure

VOID = "peer-1"


class _Greater(float):
    """A number whose comparisons all say 'no': it slips under every ``> tol`` test."""

    def __gt__(self, other):
        return False

    __lt__ = __ge__ = __le__ = __gt__

    def __float__(self):
        return 0.0


def _evenly(vectors, kappa, eligible=None, **kw):
    names = sorted(vectors)
    return make_report({n: 1.0 / len(names) for n in names}, names, kappa=kappa)


class _Strategy(WeightStrategy):
    name = "adv"

    def apply(self, vectors, scores, *, kappa, level=0, eligible=None, confidence=None):
        return self.attack(vectors, scores, kappa, eligible, confidence)


def _give_void_weight(vectors, scores, kappa, eligible, confidence):
    return _evenly(vectors, kappa)                               # includes the void peer

def _mutate_eligible(vectors, scores, kappa, eligible, confidence):
    for name in list(eligible):
        eligible[name] = True                                    # un-void by editing the guard's input
    return _evenly(vectors, kappa)

def _mutate_confidence_and_scores(vectors, scores, kappa, eligible, confidence):
    scores[VOID] = 1e9
    confidence[VOID] = 1e9
    return _evenly(vectors, kappa)

def _liar_numbers(vectors, scores, kappa, eligible, confidence):
    report = _evenly(vectors, kappa)
    report.weights[VOID] = _Greater(0.34)
    return report

def _nan(vectors, scores, kappa, eligible, confidence):
    report = _evenly(vectors, kappa)
    report.weights[VOID] = float("nan")
    return report

def _inf(vectors, scores, kappa, eligible, confidence):
    report = _evenly(vectors, kappa)
    report.weights = {k: float("inf") for k in report.weights}
    return report

def _negative(vectors, scores, kappa, eligible, confidence):
    names = sorted(vectors)
    return make_report({names[0]: -1.0, names[1]: 2.0, names[2]: 0.0}, names, kappa=kappa)

def _huge(vectors, scores, kappa, eligible, confidence):
    return make_report({n: 1e300 for n in vectors}, sorted(vectors), kappa=kappa)

def _string(vectors, scores, kappa, eligible, confidence):
    report = _evenly(vectors, kappa)
    report.weights = {k: "0.3333" for k in report.weights}
    return report

def _ghost_peer(vectors, scores, kappa, eligible, confidence):
    report = _evenly(vectors, kappa)
    report.weights["ghost"] = 0.0
    return report

def _not_a_report(vectors, scores, kappa, eligible, confidence):
    return {"peer-2": 1.0}

def _raises(vectors, scores, kappa, eligible, confidence):
    raise RuntimeError("plugin bug")

def _claims_fallback(vectors, scores, kappa, eligible, confidence):
    report = _evenly(vectors, kappa)
    report.fallback = True                                       # 'reset to the reference' with the void peer weighted
    return report

def _trim_lie(vectors, scores, kappa, eligible, confidence):
    report = _evenly(vectors, kappa)
    report.trimmed_peers = ()                                    # nothing trimmed, void peer weighted
    report.gated_peers = ()
    return report

def _property_report(vectors, scores, kappa, eligible, confidence):
    class Shifty(AggregationReport):
        reads = 0

        @property
        def weights(self):
            Shifty.reads += 1
            honest = {n: (0.0 if n == VOID else 0.5) for n in sorted(vectors)}
            return honest if Shifty.reads <= 6 else {n: 1.0 / 3 for n in honest}

        @weights.setter
        def weights(self, value):
            pass

    base = _evenly(vectors, kappa)
    return Shifty(weights={}, raw_weights=base.raw_weights, kappa=kappa, entropy=0.0, order=base.order)

def _mutate_vectors(vectors, scores, kappa, eligible, confidence):
    for tensor in vectors.values():
        tensor.zero_()
    return _evenly(vectors, kappa)

def _mapping_flip(vectors, scores, kappa, eligible, confidence):
    report = _evenly(vectors, kappa)

    class Flip(dict):
        reads = 0

        def __getitem__(self, key):
            Flip.reads += 1
            return 0.0 if (key == VOID and Flip.reads <= 8) else dict.__getitem__(self, key)

    report.weights = Flip(report.weights)
    return report


ATTACKS = {
    "void_weight": _give_void_weight, "mutate_eligible": _mutate_eligible,
    "mutate_confidence": _mutate_confidence_and_scores, "liar_numbers": _liar_numbers,
    "nan": _nan, "inf": _inf, "negative": _negative, "huge": _huge, "string": _string,
    "ghost_peer": _ghost_peer, "not_a_report": _not_a_report, "raises": _raises,
    "claims_fallback": _claims_fallback, "trim_lie": _trim_lie, "property_report": _property_report,
    "mutate_vectors": _mutate_vectors, "mapping_flip": _mapping_flip,
}


def _strategy(name):
    return type(f"Adv_{name}", (_Strategy,), {"name": name, "attack": staticmethod(ATTACKS[name])})


def _trainer(name, mode, settings, tmp_path, scope="custom", probe=False):
    register_weight_update(name, _strategy(name), scope="run")
    configure(settings, TRAINER={"WEIGHT_UPDATE": name,
                                 "WEIGHT_GUARD": {"MODE": mode, "SCOPE": scope, "PROBE": probe}})
    return breach_trainer(settings, tmp_path)


def _void_peer_weight_is_zero(trainer):
    assert trainer.reports
    for report in trainer.reports:
        assert report.weights[VOID] == 0.0, report.weights
        for value in report.weights.values():
            assert isinstance(value, float) and math.isfinite(value) and value >= 0.0
        assert abs(sum(report.weights.values()) - 1.0) < 1e-6


@pytest.mark.parametrize("mode", ["repair", "fallback"])
@pytest.mark.parametrize("name", sorted(ATTACKS))
def test_a_void_peer_never_carries_weight(name, mode, settings, tmp_path, component_overlay):
    trainer = _trainer(name, mode, settings, tmp_path)
    run_epochs(trainer)
    _void_peer_weight_is_zero(trainer)
    assert all(card.is_void for card in victim_cards(trainer))


@pytest.mark.parametrize("name", sorted(ATTACKS))
def test_enforce_mode_refuses_with_the_documented_error(name, settings, tmp_path, component_overlay):
    trainer = _trainer(name, "enforce", settings, tmp_path)
    try:
        run_epochs(trainer)
    except WeightGuardViolation as error:
        assert error.code if hasattr(error, "code") else True
        return
    # the attack was harmless in itself (it only mutated its own copy): the invariant must hold
    _void_peer_weight_is_zero(trainer)


@pytest.mark.parametrize("name", ["mutate_vectors"])
def test_a_strategy_cannot_edit_the_peers_through_its_inputs(name, settings, tmp_path, component_overlay):
    trainer = _trainer(name, "repair", settings, tmp_path)
    trainer.build()
    before = {p: peer.adapter.export_parameters().clone() for p, peer in trainer.peers.items()}
    report = trainer.update_weights()
    after = {p: peer.adapter.export_parameters() for p, peer in trainer.peers.items()}
    for path, tensor in before.items():
        assert torch.equal(tensor, after[path]), f"{path} was edited through the strategy's vectors"


def test_the_trainers_own_eligibility_dict_is_not_the_one_the_strategy_got(settings, tmp_path, component_overlay):
    seen = {}

    def spy(vectors, scores, kappa, eligible, confidence):
        seen["eligible"], seen["confidence"], seen["scores"] = eligible, confidence, scores
        eligible.clear()
        return _evenly(vectors, kappa)

    ATTACKS["spy"] = spy
    try:
        trainer = _trainer("spy", "repair", settings, tmp_path)
        run_epochs(trainer, 1)
    finally:
        ATTACKS.pop("spy")
    _void_peer_weight_is_zero(trainer)


@pytest.mark.parametrize("name", sorted(ATTACKS))
def test_the_build_time_probe_refuses_or_the_run_is_safe(name, settings, tmp_path, component_overlay):
    """With the probe on (the default) a visibly bad strategy never gets to run (E064)."""
    try:
        trainer = _trainer(name, "repair", settings, tmp_path, probe=True)
    except ImproperlyConfigured as error:
        assert "E064" in str(error)
        return
    run_epochs(trainer)
    _void_peer_weight_is_zero(trainer)


def test_a_probed_strategy_cannot_rewrite_the_probe_for_the_next_strategy():
    """Regression: the probe handed its module-level cases to ``apply`` - a strategy that set
    ``eligible[void] = True`` made every later strategy's void-weight bug invisible."""
    from mactf.coopetition.weight_probe import verify_weight_update

    kw = dict(floor=0.02, cap=0.6, tolerance=1e-6, slot="TRAINER.WEIGHT_UPDATE")

    class Edits(WeightStrategy):
        name = "edits"

        def apply(self, vectors, scores, *, kappa, level=0, eligible=None, confidence=None):
            for key in list(eligible):
                eligible[key] = True
            for key in list(scores):
                scores[key] = 0.0
            for vector in vectors.values():
                vector.zero_()
            return _evenly(vectors, kappa)

    class VoidBlind(WeightStrategy):
        name = "void_blind"

        def apply(self, vectors, scores, *, kappa, level=0, eligible=None, confidence=None):
            return _evenly(vectors, kappa)

    for _ in range(2):
        try:
            verify_weight_update(lambda: Edits(), **kw)
        except ImproperlyConfigured:
            pass
    with pytest.raises(ImproperlyConfigured, match="U004"):
        verify_weight_update(lambda: VoidBlind(), **kw)
