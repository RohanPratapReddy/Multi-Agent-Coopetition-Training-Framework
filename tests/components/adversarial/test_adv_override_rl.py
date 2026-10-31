"""RL functions under a built-in's name meet the same probe and the same compliance rule (document 17 section 8).

The broken functions of ``test_adv_rl_functions`` are registered as ``rl_function.advantage:gae``,
``loss:policy_gradient`` and ``rollup:mean`` with the override flag. The build-time probe
(F001-F012, E047) refuses them; a loss that claims the compliance term is still extended; the
rollup that drops a void set is refused. ``RL.VERIFY: off`` (W013) is the only road past the
probe, and the negative control below proves it is the probe, not luck, doing the refusing.
"""

from __future__ import annotations

import logging

import pytest
import torch

import custom_kinds as ck
from mactf.components.loss_compliance import carries_compliance
from mactf.exceptions import ComponentContractError, ImproperlyConfigured
from mactf.rl.functions.loss import PolicyGradientLoss, WeightedSumLoss
from override_support import override_scope, put, w001
from p2_support import configure, hierarchy_trainer, make_trainer, tune
from test_adv_rl_functions import _DropsVoid, _Ignores, _MutatesInputs, _NaNAdvantage, _Random

REFUSED = (ImproperlyConfigured, ComponentContractError)


@pytest.fixture(autouse=True)
def _scope(caplog):
    caplog.set_level(logging.WARNING)
    with override_scope():
        yield


def _build(settings, tmp_path, **rl):
    tune(settings)
    configure(settings, COMPONENTS={"PLUGINS": "off"}, RL=rl)
    return make_trainer(settings, tmp_path).build()


# -- advantage -----------------------------------------------------------------------------------
@pytest.mark.parametrize("target", ["gae", "td", "normalised"])
@pytest.mark.parametrize("cls", [_NaNAdvantage, _MutatesInputs, _Random], ids=lambda c: c.__name__)
def test_a_broken_advantage_under_a_builtin_name_is_refused_at_build(cls, target, settings, tmp_path):
    broken = type(cls.__name__, (cls,), {"spec_family": "advantage", "spec_type": target})
    put("rl_function.advantage", target, broken)
    with pytest.raises(REFUSED, match="F00[4-7]|probe|E047"):
        _build(settings, tmp_path, ADVANTAGE={"type": target})


@pytest.mark.parametrize("cls", [_NaNAdvantage, _MutatesInputs, _Random], ids=lambda c: c.__name__)
def test_verify_off_lets_the_same_bug_through_and_says_so(cls, settings, tmp_path, caplog):
    """The negative control: with the probe disabled the broken override is built (W013)."""
    broken = type(cls.__name__, (cls,), {"spec_family": "advantage", "spec_type": "gae"})
    put("rl_function.advantage", "gae", broken)
    with pytest.raises(REFUSED):
        _build(settings, tmp_path / "on", ADVANTAGE={"type": "gae"})                     # control: refused
    trainer = _build(settings, tmp_path / "off", ADVANTAGE={"type": "gae"}, VERIFY="off")
    assert trainer is not None
    assert "W013" in caplog.text


# -- loss --------------------------------------------------------------------------------------------
def test_a_loss_that_ignores_the_floor_under_a_builtin_name_is_extended(settings, tmp_path):
    put("rl_function.loss", "policy_gradient", _Ignores)
    trainer = _build(settings, tmp_path, LOSS={"type": "policy_gradient"})
    loss = trainer.loss_for("peer-1")
    assert isinstance(loss, WeightedSumLoss) and carries_compliance(loss)
    assert any(isinstance(term, _Ignores) for term in loss.terms)
    values = torch.randn(4, requires_grad=True)
    batch = {"values": values, "value_targets": values.detach().clone(), "compliance_floor": torch.tensor(-2.0)}
    total, breakdown = loss.compute(batch)
    assert breakdown.compliance > 0.0 and float(total.detach()) >= 2.0


def test_a_non_finite_loss_under_a_builtin_name_is_refused(settings, tmp_path):
    class NaNPolicyGradient(PolicyGradientLoss):
        def compute(self, batch):
            loss, breakdown = super().compute(batch)
            return loss * float("nan"), breakdown

    put("rl_function.loss", "policy_gradient", NaNPolicyGradient)
    with pytest.raises(REFUSED, match="E047|F00"):
        _build(settings, tmp_path, LOSS={"type": "policy_gradient"})


def test_a_non_finite_loss_gets_through_only_with_verify_off(settings, tmp_path, caplog):
    class NaNPolicyGradient(PolicyGradientLoss):
        def compute(self, batch):
            loss, breakdown = super().compute(batch)
            return loss * float("nan"), breakdown

    put("rl_function.loss", "policy_gradient", NaNPolicyGradient)
    assert _build(settings, tmp_path, LOSS={"type": "policy_gradient"}, VERIFY="off") is not None
    assert "W013" in caplog.text


# -- rollup -----------------------------------------------------------------------------------------------
def test_a_rollup_that_drops_a_void_set_under_a_builtin_name_is_refused(settings, tmp_path):
    broken = type("DropsVoidMean", (_DropsVoid,), {"name": "mean"})
    put("rl_function.rollup", "mean", broken)
    tune(settings)
    configure(settings, COMPONENTS={"PLUGINS": "off"}, RL={"ROLLUP": {"type": "mean"}})
    with pytest.raises(REFUSED, match="F010|void"):
        hierarchy_trainer(settings, tmp_path).build()


# -- the honest identity overrides pass every probe mode ----------------------------------------------------
@pytest.mark.parametrize("verify", ["custom", "all"])
def test_identity_overrides_pass_the_probe(verify, settings, tmp_path, caplog):
    put("rl_function.advantage", "gae", ck.SameGAE)
    put("rl_function.loss", "policy_gradient", ck.SamePolicyGradient)
    trainer = _build(settings, tmp_path, ADVANTAGE={"type": "gae"}, LOSS={"type": "policy_gradient"}, VERIFY=verify)
    assert trainer is not None
    assert sorted(w001(caplog)) == ["rl_function.advantage:gae", "rl_function.loss:policy_gradient"]


def test_a_new_name_is_probed_exactly_like_an_override(settings, tmp_path):
    broken = type("NewNaN", (_NaNAdvantage,), {"spec_family": "advantage", "spec_type": "zz_new_nan"})
    put("rl_function.advantage", "zz_new_nan", broken)
    with pytest.raises(REFUSED, match="F00[4-7]|probe|E047"):
        _build(settings, tmp_path, ADVANTAGE={"type": "zz_new_nan"})
